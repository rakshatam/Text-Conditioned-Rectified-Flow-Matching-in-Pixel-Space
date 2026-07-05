import os
import math
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

# ==============================================================================
# 1. ARCHITECTURE DEFINITIONS
# ==============================================================================
def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    half = dim // 2
    exp = -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / (half - 1)
    freqs = torch.exp(exp)
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return F.pad(emb, (0, 1)) if dim % 2 else emb

class ADMResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, temb_ch: int, dropout: float = 0.1, groups: int = 32):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch, eps=1e-6)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Sequential(nn.SiLU(), nn.Linear(temb_ch, 2 * out_ch))
        self.norm2 = nn.GroupNorm(groups, out_ch, eps=1e-6)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb_proj(temb).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.drop(F.silu(h)))
        return h + self.skip(x)

class SpatialSelfAttn(nn.Module):
    def __init__(self, channels: int, num_heads: int, head_dim: int = 64, groups: int = 32):
        super().__init__()
        inner = num_heads * head_dim
        self.norm = nn.GroupNorm(groups, channels, eps=1e-6)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner = inner
        self.qkv = nn.Linear(channels, 3 * inner, bias=False)
        self.proj_out = nn.Linear(inner, channels)
        nn.init.normal_(self.proj_out.weight, std=0.02)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        h = self.norm(x).view(B, C, N).transpose(1, 2)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        def split(t): return t.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(split(q), split(k), split(v))
        out = out.transpose(1, 2).reshape(B, N, self.inner)
        return x + self.proj_out(out).transpose(1, 2).view(B, C, H, W)

class SpatialCrossAttn(nn.Module):
    def __init__(self, channels: int, context_dim: int, num_heads: int, head_dim: int = 64, groups: int = 32):
        super().__init__()
        inner = num_heads * head_dim
        self.norm = nn.GroupNorm(groups, channels, eps=1e-6)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner = inner
        self.to_q = nn.Linear(channels, inner, bias=False)
        self.to_k = nn.Linear(context_dim, inner, bias=False)
        self.to_v = nn.Linear(context_dim, inner, bias=False)
        self.proj_out = nn.Linear(inner, channels)
        nn.init.normal_(self.proj_out.weight, std=0.02)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N, L = H * W, context.shape[1]
        h = self.norm(x).view(B, C, N).transpose(1, 2)
        q = self.to_q(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(context).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(context).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, N, self.inner)
        return x + self.proj_out(out).transpose(1, 2).view(B, C, H, W)

class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
    def forward(self, x): return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)
    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False))

class FlowMatchingUNet(nn.Module):
    def __init__(self, in_channels: int = 3, model_channels: int = 128, context_dim: int = 512,
                 channel_mult: tuple = (1, 2, 4), num_res_blocks: int = 2, head_dim: int = 64,
                 n_context_tokens: int = 16, dropout: float = 0.1, attention_resolutions: tuple = (32, 16),
                 input_size: int = 64):
        super().__init__()
        self.model_channels = model_channels
        self.context_dim = context_dim

        temb_ch = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, temb_ch), nn.SiLU(),
            nn.Linear(temb_ch, temb_ch),
        )

        self.ctx_expand = nn.Sequential(
            nn.Linear(context_dim, context_dim * 2), nn.GELU(),
            nn.Linear(context_dim * 2, context_dim * n_context_tokens),
        )
        self.ctx_pos = nn.Parameter(torch.zeros(1, n_context_tokens, context_dim))
        self.conv_in = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        ch = model_channels
        self._skip_ch = [ch]
        self.enc_slots = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            sp_size = input_size // (2 ** level)
            use_attn = sp_size in attention_resolutions
            n_heads = max(1, out_ch // head_dim)
            slots = nn.ModuleList()
            for _ in range(num_res_blocks):
                slot = nn.ModuleList([ADMResBlock(ch, out_ch, temb_ch, dropout)])
                ch = out_ch
                if use_attn:
                    slot.append(SpatialSelfAttn(ch, n_heads, head_dim))
                    slot.append(SpatialCrossAttn(ch, context_dim, n_heads, head_dim))
                slots.append(slot)
                self._skip_ch.append(ch)
            self.enc_slots.append(slots)
            self.downsamplers.append(Downsample(ch) if level < len(channel_mult) - 1 else None)

        mid_ch = ch
        mid_heads = max(1, mid_ch // head_dim)
        self.mid_res1 = ADMResBlock(mid_ch, mid_ch, temb_ch, dropout)
        self.mid_self = SpatialSelfAttn(mid_ch, mid_heads, head_dim)
        self.mid_cross = SpatialCrossAttn(mid_ch, context_dim, mid_heads, head_dim)
        self.mid_res2 = ADMResBlock(mid_ch, mid_ch, temb_ch, dropout)

        self.dec_slots = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = model_channels * mult
            sp_size = input_size // (2 ** level)
            use_attn = sp_size in attention_resolutions
            n_heads = max(1, out_ch // head_dim)
            n_dec = num_res_blocks + 1 if level == 0 else num_res_blocks
            slots = nn.ModuleList()
            for _ in range(n_dec):
                skip_ch = self._skip_ch.pop()
                slot = nn.ModuleList([ADMResBlock(ch + skip_ch, out_ch, temb_ch, dropout)])
                ch = out_ch
                if use_attn:
                    slot.append(SpatialSelfAttn(ch, n_heads, head_dim))
                    slot.append(SpatialCrossAttn(ch, context_dim, n_heads, head_dim))
                slots.append(slot)
            self.dec_slots.append(slots)
            self.upsamplers.append(Upsample(ch) if level > 0 else None)

        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, model_channels, eps=1e-6),
            nn.SiLU(),
            nn.Conv2d(model_channels, in_channels, 3, padding=1),
        )

    def _ctx(self, emb: torch.Tensor) -> torch.Tensor:
        B = emb.shape[0]
        ctx = self.ctx_expand(emb).view(B, -1, self.context_dim)
        return ctx + self.ctx_pos

    def _slot(self, h, slot, temb, ctx):
        h = slot[0](h, temb)
        if len(slot) > 1:
            h = slot[1](h)
            h = slot[2](h, ctx)
        return h

    def forward(self, x: torch.Tensor, t: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        temb = self.time_embed(timestep_embedding(t * 1000.0, self.model_channels))
        ctx = self._ctx(text_emb)
        h = self.conv_in(x)
        skips = [h]

        for level_slots, ds in zip(self.enc_slots, self.downsamplers):
            for slot in level_slots:
                h = self._slot(h, slot, temb, ctx)
                skips.append(h)
            if ds is not None:
                h = ds(h)

        h = self.mid_res1(h, temb)
        h = self.mid_self(h)
        h = self.mid_cross(h, ctx)
        h = self.mid_res2(h, temb)

        for level_slots, us in zip(self.dec_slots, self.upsamplers):
            for slot in level_slots:
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = self._slot(h, slot, temb, ctx)
            if us is not None:
                h = us(h)

        return self.conv_out(h)

# ==============================================================================
# 2. ODE SAMPLER
# ==============================================================================
@torch.no_grad()
def euler_sample(model, text_embs, steps: int = 30, cfg_scale: float = 3.0, generator=None):
    model.eval()
    device = next(model.parameters()).device
    B = text_embs.size(0)
    
    if generator is not None:
        zt = torch.randn(B, 3, 64, 64, device=device, generator=generator)
    else:
        zt = torch.randn(B, 3, 64, 64, device=device)
        
    uncond = torch.zeros_like(text_embs)
    EPS = 1e-3
    dt = 1.0 / steps
    
    for i in range(steps):
        t_val = i / steps * (1.0 - EPS) + EPS
        t = torch.full((B,), t_val, device=device)
        vc = model(zt, t, text_embs)
        vu = model(zt, t, uncond)
        zt = zt + (vu + cfg_scale * (vc - vu)) * dt
        
    return ((zt.clamp(-1, 1) + 1) / 2).clamp(0, 1)

# ==============================================================================
# 3. SETUP & MODEL LOADING
# ==============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "openai/clip-vit-base-patch32"

print("Loading FlowMatchingUNet (Production Mode)...")
model = FlowMatchingUNet().to(DEVICE)
# Checkpoint location: RFM_CKPT_PATH overrides this if set, otherwise falls
# back to the Kaggle path this project was originally trained against.
ckpt_path = os.environ.get(
    "RFM_CKPT_PATH",
    "your model path",
)

if os.path.exists(ckpt_path):
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    model.eval()
else:
    raise FileNotFoundError(f"Could not find model at {ckpt_path}")

print("Loading Unprojected CLIP Text Encoder...")
text_model = CLIPTextModel.from_pretrained(MODEL_NAME).to(DEVICE)
tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)

# Verified empirically (not assumed) against a stored training embedding,
# so this also works correctly if RFM_DATA_DIR points at a different dataset.
dataset_path = os.environ.get(
    "RFM_DATA_DIR",
    "/kaggle/input/models/qwertywell/anime-faces/pytorch/default/1",
)
if not os.path.exists(dataset_path):
    dataset_path = "/kaggle/working/anime_data_with_embeddings"
target_pt = sorted(glob.glob(os.path.join(dataset_path, "*.pt")))[0]
is_normalized = torch.allclose(
    torch.load(target_pt, map_location="cpu", weights_only=True).norm(), 
    torch.tensor(1.0, dtype=torch.float32), 
    atol=1e-2
)

# ==============================================================================
# 3.5 DISPLAY HELPER
# ==============================================================================
# imshow() stretching a raw 64x64 array can look blockier than the model's
# true output. Upscaling explicitly with PIL's LANCZOS filter first gives an
# honest read of actual quality, independent of matplotlib's resampling.
DISPLAY_SIZE = 320

def to_display_image(img_tensor: torch.Tensor, size: int = DISPLAY_SIZE) -> Image.Image:
    img_np = (img_tensor.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    pil_img = Image.fromarray(img_np, mode="RGB")
    return pil_img.resize((size, size), resample=Image.LANCZOS)

# ==============================================================================
# 4. ENHANCED PROMPTS
# ==============================================================================
enhanced_prompts = [
    "a girl with purple hair with yellow eyes.",  
    "a girl with pure white hair with blue eyes.",  
    "a girl with red eyes with long hair.",         
    "a girl with blue hair green eyes.",          
    "a girl with long twin tails hair.",               
    "a girl with green hair, wearing thick black glasses."  
]

# ==============================================================================
# 5. GENERATION
# ==============================================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# CFG and step count tuned empirically: strong enough guidance to make
# attributes render clearly without pushing the ODE integration into an
# unstable regime.
CFG_SCALE = 2.5   
STEPS = 50        
STARTING_SEED = 2026       

print("\nGenerating enhanced structural gallery...")
for i, prompt in enumerate(enhanced_prompts):
    # Encode text
    inputs = tokenizer([prompt], padding=True, max_length=77, truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = text_model(**inputs)
        # Matches how the training dataset's .pt embeddings were generated;
        # using the projected/normalized CLIP embedding instead would put
        # new prompts in a different vector space than the model trained on.
        text_emb = outputs.pooler_output 
        if is_normalized:
            text_emb = torch.nn.functional.normalize(text_emb, p=2, dim=-1)

    # Dynamic Seed
    current_seed = STARTING_SEED + (i * 900)
    g = torch.Generator(device=DEVICE).manual_seed(current_seed)
    
    # ODE Sampling
    img_tensor = euler_sample(
        model, 
        text_emb, 
        steps=STEPS, 
        cfg_scale=CFG_SCALE, 
        generator=g
    )
    
    # Plotting -- ONLY this line changed from your original (imshow(img_np) ->
    # imshow(to_display_image(...))), plus interpolation='nearest' since the
    # array is already properly upscaled and shouldn't be resampled again.
    axes[i].imshow(to_display_image(img_tensor[0]), interpolation="nearest")
    axes[i].set_title(f"\"{prompt}\"", fontsize=9)
    axes[i].axis("off")

plt.tight_layout()
plt.savefig("enhanced_gallery.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved enhanced_gallery.png")

# ==============================================================================
# 6. EMBEDDING INTERPOLATION TEST
# ==============================================================================
# Interpolates between two NOVEL prompts (not dataset embeddings), since
# interpolating within the training set only shows behavior between points
# already seen. This tests generalization to the space between two prompts
# the model never trained on.
INTERP_PROMPT_A = enhanced_prompts[0]   # "a girl with purple hair with yellow eyes."
INTERP_PROMPT_B = enhanced_prompts[3]   # "a girl with blue hair green eyes."
N_INTERP_STEPS  = 7
INTERP_SEED     = 4242

def encode_prompt(prompt: str) -> torch.Tensor:
    inputs = tokenizer([prompt], padding=True, max_length=77, truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = text_model(**inputs)
        emb = out.pooler_output
        if is_normalized:
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
    return emb

print(f"\nInterpolating: \"{INTERP_PROMPT_A}\"  ->  \"{INTERP_PROMPT_B}\"")

emb_a = encode_prompt(INTERP_PROMPT_A)
emb_b = encode_prompt(INTERP_PROMPT_B)

fig2, axes2 = plt.subplots(1, N_INTERP_STEPS, figsize=(3 * N_INTERP_STEPS, 3.4))

for i, alpha in enumerate(torch.linspace(0, 1, N_INTERP_STEPS)):
    alpha = alpha.item()
    emb_interp = (1 - alpha) * emb_a + alpha * emb_b
    if is_normalized:
        # Linear interpolation between unit vectors leaves the unit sphere
        # except at the endpoints -- renormalize to match training scale.
        emb_interp = torch.nn.functional.normalize(emb_interp, p=2, dim=-1)

    # Re-seeded fresh each step (not one generator reused) so every alpha
    # starts from identical noise -- only the embedding changes across the row.
    g_step = torch.Generator(device=DEVICE).manual_seed(INTERP_SEED)
    img_tensor = euler_sample(model, emb_interp, steps=STEPS, cfg_scale=CFG_SCALE, generator=g_step)

    axes2[i].imshow(to_display_image(img_tensor[0]), interpolation="nearest")
    axes2[i].set_title(f"α={alpha:.2f}", fontsize=9)
    axes2[i].axis("off")

plt.tight_layout()
plt.savefig("interpolation_test.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved interpolation_test.png")
print("Expect a smooth purple->blue and yellow->green transition with no "
      "jump or unrelated identity appearing mid-sequence.")
