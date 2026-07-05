"""
master_pipeline.py — Pixel-Space Rectified Flow Matching, 64x64 Anime Faces

Architecture: ADM-style UNet (Dhariwal & Nichol, 2021) with Imagen-style
text cross-attention. Training: Rectified Flow Matching (Liu et al., 2022)
with logit-normal timestep sampling (as used in SD3 / Flux). Trained from
scratch in pixel space, no VAE, conditioned on 512-d text embeddings.

Notable design decisions (see README for full rationale):
  * Attention only at 32x32 / 16x16, never at 64x64 (avoids seam artifacts
    from position-free full-resolution self-attention).
  * No horizontal flip augmentation (hairstyles are directional).
  * Attention proj_out uses small non-zero init, not zero (conv_out's
    zero-init alone is enough for stable output; double-zeroing stalls
    cross-attention gradients).
  * OmniTracker classifies gradients via isinstance(), not name matching.
  * Three optimizer groups (backbone / self-attn / cross-attn) since
    cross-attention bootstraps more slowly than self-attention.
  * conditioning_probe isolates the effect of the text embedding by holding
    starting noise fixed across two different embeddings.
"""


import os, glob, copy, math, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ══════════════════════════════════════════════════════════════════
#  DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════
class OmniTracker:
    def __init__(self, model):
        self.model = model
        self._cat_map = None   # built on first update(), cached thereafter
        # Persists across epochs so each epoch's grad_norm can be compared
        # to this run's own recent baseline, not just a fixed cutoff.
        self.grad_norm_history = []
        self.reset()

    def _build_cat_map(self):
        """Map each parameter to its module type via isinstance(), not name
        matching (name substrings collide between self- and cross-attn)."""
        cat = {}
        for _, mod in self.model.named_modules():
            if isinstance(mod, SpatialCrossAttn):
                for p in mod.parameters(): cat[id(p)] = "cross"
            elif isinstance(mod, SpatialSelfAttn):
                for p in mod.parameters(): cat[id(p)] = "self"
            elif isinstance(mod, ADMResBlock):
                for p in mod.parameters(): cat[id(p)] = "res"
        return cat

    def reset(self):
        self.s = {k: [] for k in [
            "loss", "t_mean", "v_pred_std", "v_pred_mae",
            "cosine_sim", "x1_mse",
            "grad_resblock", "grad_self_attn", "grad_cross_attn",
            "grad_norm", "dead_ratio",
        ]}

    def update(self, loss_val, data, noise, zt, v_pred, t_exp, grad_norm):
        if self._cat_map is None:
            self._cat_map = self._build_cat_map()

        self.s["loss"].append(loss_val)
        self.s["t_mean"].append(t_exp.float().mean().item())
        vp = v_pred.float()
        self.s["v_pred_std"].append(vp.std().item())
        self.s["v_pred_mae"].append((vp - (data - noise).float()).abs().mean().item())

        x1 = zt.float() + (1 - t_exp.float()) * vp
        B = data.size(0)
        self.s["cosine_sim"].append(
            F.cosine_similarity(x1.reshape(B,-1), data.reshape(B,-1).float(), dim=1).mean().item()
        )
        self.s["x1_mse"].append(F.mse_loss(x1, data.float()).item())

        gv = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        self.s["grad_norm"].append(min(gv, 99999.0) if not (math.isnan(gv) or math.isinf(gv)) else 99999.0)

        g_res, g_sa, g_ca = [], [], []
        total, dead = 0, 0
        for n, p in self.model.named_parameters():
            if not p.requires_grad: continue
            total += p.numel()
            dead  += (p.abs() < 1e-7).sum().item()
            if p.grad is None: continue
            gm = p.grad.float().abs().mean().item()
            if math.isnan(gm) or math.isinf(gm): gm = 99999.0
            cat = self._cat_map.get(id(p))
            if   cat == "cross": g_ca.append(gm)
            elif cat == "self":  g_sa.append(gm)
            elif cat == "res":   g_res.append(gm)

        def _a(lst): return sum(lst)/len(lst) if lst else 0.0
        self.s["grad_resblock"].append(_a(g_res))
        self.s["grad_self_attn"].append(_a(g_sa))
        self.s["grad_cross_attn"].append(_a(g_ca))
        if total: self.s["dead_ratio"].append(dead/total)

    def report(self):
        out = {k: round(sum(v)/len(v), 6) if v else 0.0 for k, v in self.s.items()}
        res = out.get("grad_resblock", 1e-12)
        ca  = out.get("grad_cross_attn", 0.0)
        out["attn_ca_ratio"] = round(ca / max(res, 1e-12), 6)
        out["HEALTH_GRAD_FLOW"]    = "PASS" if out.get("grad_resblock", 0) > 1e-7 else "FAIL"
        out["HEALTH_EXPLOSION"]    = "PASS" if out.get("grad_norm", 0)     < 1000  else "FAIL"
        out["HEALTH_V_SCALE"]      = "PASS" if out.get("v_pred_std", 0)    > 0.01  else "FAIL"
        out["HEALTH_DATA_PRED"]    = "PASS" if out.get("cosine_sim", 0)    > 0.1   else "FAIL"
        out["HEALTH_DEAD_NEURONS"] = "PASS" if out.get("dead_ratio", 0)    < 0.1   else "FAIL"
        r = out["attn_ca_ratio"]
        out["HEALTH_CROSS_ATTN"]   = ("FAIL(starved)" if r < 0.05 else
                                       "WARN(weak)"   if r < 0.15 else "PASS")

        # Relative check vs this run's own trailing median gradient norm.
        # HEALTH_EXPLOSION (>1000) only catches catastrophic blowups; a
        # jump to hundreds of times the normal band is worth flagging even
        # if clipping already bounded the actual parameter update.
        gn = out.get("grad_norm", 0.0)
        hist = self.grad_norm_history[-10:]
        if len(hist) >= 3:
            med = sorted(hist)[len(hist) // 2]
            if med > 1e-9 and gn > med * 8:
                out["HEALTH_GRAD_NORM_SPIKE"] = (
                    f"WARN ({gn:.2f} vs recent median {med:.3f}, "
                    f"{gn/med:.0f}x -- likely a single bad batch; "
                    f"clipping bounded the actual update, but watch next epoch)"
                )
            else:
                out["HEALTH_GRAD_NORM_SPIKE"] = "PASS"
        else:
            out["HEALTH_GRAD_NORM_SPIKE"] = "PASS (building history)"
        self.grad_norm_history.append(gn)

        return out


# ══════════════════════════════════════════════════════════════════
#  TIME EMBEDDING  (called internally with t*1000)
# ══════════════════════════════════════════════════════════════════
def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    half  = dim // 2
    exp   = -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / (half - 1)
    freqs = torch.exp(exp)
    args  = t.float()[:, None] * freqs[None]
    emb   = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return F.pad(emb, (0, 1)) if dim % 2 else emb


# ══════════════════════════════════════════════════════════════════
#  ADM RESBLOCK — scale-shift GroupNorm (AdaGN), as in ADM / GLIDE / Imagen.
#  Scale+shift gives the timestep more modulation control than additive.
# ══════════════════════════════════════════════════════════════════
class ADMResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, temb_ch: int,
                 dropout: float = 0.1, groups: int = 32):
        super().__init__()
        self.norm1    = nn.GroupNorm(groups, in_ch,  eps=1e-6)
        self.conv1    = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Sequential(nn.SiLU(), nn.Linear(temb_ch, 2 * out_ch))
        self.norm2    = nn.GroupNorm(groups, out_ch, eps=1e-6)
        self.drop     = nn.Dropout(dropout)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip     = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb_proj(temb).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.drop(F.silu(h)))
        return h + self.skip(x)


# ══════════════════════════════════════════════════════════════════
#  SPATIAL SELF-ATTENTION
#  proj_out uses small normal init (std=0.02), not zero — conv_out's
#  zero-init alone already guarantees zero output at step 0; zeroing
#  proj_out too just adds a redundant gradient bottleneck (see README).
# ══════════════════════════════════════════════════════════════════
class SpatialSelfAttn(nn.Module):
    def __init__(self, channels: int, num_heads: int,
                 head_dim: int = 64, groups: int = 32):
        super().__init__()
        inner          = num_heads * head_dim
        self.norm      = nn.GroupNorm(groups, channels, eps=1e-6)
        self.num_heads = num_heads
        self.head_dim  = head_dim
        self.inner     = inner
        self.qkv       = nn.Linear(channels, 3 * inner, bias=False)
        self.proj_out  = nn.Linear(inner, channels)
        nn.init.normal_(self.proj_out.weight, std=0.02)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        h = self.norm(x).view(B, C, N).transpose(1, 2)         # [B, N, C]
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        def split(t): return t.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(split(q), split(k), split(v))
        out = out.transpose(1, 2).reshape(B, N, self.inner)
        return x + self.proj_out(out).transpose(1, 2).view(B, C, H, W)


# ══════════════════════════════════════════════════════════════════
#  SPATIAL CROSS-ATTENTION  (text conditioning, Imagen style)
#  Same proj_out rationale as SpatialSelfAttn above.
# ══════════════════════════════════════════════════════════════════
class SpatialCrossAttn(nn.Module):
    def __init__(self, channels: int, context_dim: int,
                 num_heads: int, head_dim: int = 64, groups: int = 32):
        super().__init__()
        inner          = num_heads * head_dim
        self.norm      = nn.GroupNorm(groups, channels, eps=1e-6)
        self.num_heads = num_heads
        self.head_dim  = head_dim
        self.inner     = inner
        self.to_q      = nn.Linear(channels,    inner, bias=False)
        self.to_k      = nn.Linear(context_dim, inner, bias=False)
        self.to_v      = nn.Linear(context_dim, inner, bias=False)
        self.proj_out  = nn.Linear(inner, channels)
        nn.init.normal_(self.proj_out.weight, std=0.02)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N, L = H * W, context.shape[1]
        h = self.norm(x).view(B, C, N).transpose(1, 2)         # [B, N, C]
        q = self.to_q(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(context).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(context).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)           # [B, H, N, D]
        out = out.transpose(1, 2).reshape(B, N, self.inner)
        return x + self.proj_out(out).transpose(1, 2).view(B, C, H, W)


# ══════════════════════════════════════════════════════════════════
#  DOWNSAMPLE / UPSAMPLE
# ══════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════
#  FLOW MATCHING UNET
#  attention_resolutions=(32,16): no attention at 64×64. Spatial
#  coherence at full res comes from conv zero-padding (GLIDE/Imagen/ADM).
# ══════════════════════════════════════════════════════════════════
class FlowMatchingUNet(nn.Module):
    def __init__(
        self,
        in_channels: int           = 3,
        model_channels: int        = 128,
        context_dim: int           = 512,
        channel_mult: tuple        = (1, 2, 4),
        num_res_blocks: int        = 2,
        head_dim: int              = 64,
        n_context_tokens: int      = 16,
        dropout: float             = 0.1,
        attention_resolutions: tuple = (32, 16),
        input_size: int            = 64,
    ):
        super().__init__()
        self.model_channels = model_channels
        self.context_dim    = context_dim

        temb_ch = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, temb_ch), nn.SiLU(),
            nn.Linear(temb_ch, temb_ch),
        )

        # Single 512-d text embedding → n_context_tokens tokens for cross-attn
        self.ctx_expand = nn.Sequential(
            nn.Linear(context_dim, context_dim * 2), nn.GELU(),
            nn.Linear(context_dim * 2, context_dim * n_context_tokens),
        )
        self.ctx_pos = nn.Parameter(torch.zeros(1, n_context_tokens, context_dim))

        self.conv_in = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # ── Encoder ────────────────────────────────────────────────
        ch = model_channels
        self._skip_ch = [ch]
        self.enc_slots    = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        for level, mult in enumerate(channel_mult):
            out_ch   = model_channels * mult
            sp_size  = input_size // (2 ** level)
            use_attn = sp_size in attention_resolutions
            n_heads  = max(1, out_ch // head_dim)
            slots    = nn.ModuleList()
            for _ in range(num_res_blocks):
                slot = nn.ModuleList([ADMResBlock(ch, out_ch, temb_ch, dropout)])
                ch   = out_ch
                if use_attn:
                    slot.append(SpatialSelfAttn(ch, n_heads, head_dim))
                    slot.append(SpatialCrossAttn(ch, context_dim, n_heads, head_dim))
                slots.append(slot)
                self._skip_ch.append(ch)
            self.enc_slots.append(slots)
            self.downsamplers.append(
                Downsample(ch) if level < len(channel_mult) - 1 else None
            )

        # ── Middle ─────────────────────────────────────────────────
        mid_ch    = ch
        mid_heads = max(1, mid_ch // head_dim)
        self.mid_res1  = ADMResBlock(mid_ch, mid_ch, temb_ch, dropout)
        self.mid_self  = SpatialSelfAttn(mid_ch, mid_heads, head_dim)
        self.mid_cross = SpatialCrossAttn(mid_ch, context_dim, mid_heads, head_dim)
        self.mid_res2  = ADMResBlock(mid_ch, mid_ch, temb_ch, dropout)

        # ── Decoder ────────────────────────────────────────────────
        self.dec_slots  = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch   = model_channels * mult
            sp_size  = input_size // (2 ** level)
            use_attn = sp_size in attention_resolutions
            n_heads  = max(1, out_ch // head_dim)
            n_dec    = num_res_blocks + 1 if level == 0 else num_res_blocks
            slots    = nn.ModuleList()
            for _ in range(n_dec):
                skip_ch = self._skip_ch.pop()
                slot    = nn.ModuleList([ADMResBlock(ch + skip_ch, out_ch, temb_ch, dropout)])
                ch      = out_ch
                if use_attn:
                    slot.append(SpatialSelfAttn(ch, n_heads, head_dim))
                    slot.append(SpatialCrossAttn(ch, context_dim, n_heads, head_dim))
                slots.append(slot)
            self.dec_slots.append(slots)
            self.upsamplers.append(Upsample(ch) if level > 0 else None)

        # ── Output ─────────────────────────────────────────────────
        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, model_channels, eps=1e-6),
            nn.SiLU(),
            nn.Conv2d(model_channels, in_channels, 3, padding=1),
        )
        # Zero-init: v_pred=0 at step 0. proj_out layers stay non-zero (see
        # SpatialSelfAttn) — this is the only zero-init the model needs.
        nn.init.zeros_(self.conv_out[-1].weight)
        nn.init.zeros_(self.conv_out[-1].bias)

    def _ctx(self, emb: torch.Tensor) -> torch.Tensor:
        B   = emb.shape[0]
        ctx = self.ctx_expand(emb).view(B, -1, self.context_dim)
        return ctx + self.ctx_pos

    def _slot(self, h, slot, temb, ctx):
        h = slot[0](h, temb)
        if len(slot) > 1:
            h = slot[1](h)
            h = slot[2](h, ctx)
        return h

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                text_emb: torch.Tensor) -> torch.Tensor:
        temb = self.time_embed(timestep_embedding(t * 1000.0, self.model_channels))
        ctx  = self._ctx(text_emb)

        h     = self.conv_in(x)
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
                h    = torch.cat([h, skip], dim=1)
                h    = self._slot(h, slot, temb, ctx)
            if us is not None:
                h = us(h)

        return self.conv_out(h)


# ══════════════════════════════════════════════════════════════════
#  DYNAMIC EMA
# ══════════════════════════════════════════════════════════════════
class DynamicEMA:
    def __init__(self, model: nn.Module, max_decay: float = 0.9999):
        self.max_decay = max_decay
        self.shadow    = copy.deepcopy(model).float()
        self.shadow.eval()
        for p in self.shadow.parameters(): p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module, step: int):
        decay = min(self.max_decay, 0.9 + 0.0999 * step / 1000.0)
        for ep, mp in zip(self.shadow.parameters(), model.parameters()):
            ep.data.mul_(decay).add_(mp.data.float(), alpha=1.0 - decay)


# ══════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════
class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.paths     = sorted(
            glob.glob(os.path.join(img_dir, "*.png")) +
            glob.glob(os.path.join(img_dir, "*.jpg"))
        )
        self.transform = transform

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform: img = self.transform(img)
        ep  = os.path.splitext(self.paths[idx])[0] + ".pt"
        emb = torch.load(ep, map_location="cpu", weights_only=True).float() \
              if os.path.exists(ep) else torch.zeros(512)
        return img, emb

    def check_embeddings(self):
        missing, zero_norm, ok = 0, 0, 0
        norms = []
        for p in self.paths:
            ep = os.path.splitext(p)[0] + ".pt"
            if not os.path.exists(ep):
                missing += 1
                continue
            try:
                emb = torch.load(ep, map_location="cpu", weights_only=True).float()
                n   = emb.norm().item()
                norms.append(n)
                if n < 1e-6: zero_norm += 1
                else:        ok += 1
            except Exception:
                missing += 1
        total = len(self.paths)
        return {
            "total": total, "missing": missing,
            "zero_norm": zero_norm, "ok": ok,
            "missing_pct": 100.0 * missing / total if total else 100.0,
            "mean_norm":   sum(norms) / len(norms) if norms else 0.0,
        }


# ══════════════════════════════════════════════════════════════════
#  EULER ODE SAMPLER
# ══════════════════════════════════════════════════════════════════
@torch.no_grad()
def euler_sample(model, text_embs, steps: int = 30, cfg_scale: float = 3.0,
                  generator=None):
    """generator: fixes starting noise so validation grids are comparable
    across epochs — without it, each call draws different noise from the
    advancing global RNG, confounding "model improved" with "noise differed."
    """
    model.eval()
    device = next(model.parameters()).device
    B      = text_embs.size(0)
    if generator is not None:
        zt = torch.randn(B, 3, 64, 64, device=device, generator=generator)
    else:
        zt = torch.randn(B, 3, 64, 64, device=device)
    uncond = torch.zeros_like(text_embs)
    EPS    = 1e-3
    dt     = 1.0 / steps
    for i in range(steps):
        t_val = i / steps * (1.0 - EPS) + EPS
        t     = torch.full((B,), t_val, device=device)
        vc    = model(zt, t, text_embs)
        vu    = model(zt, t, uncond)
        zt    = zt + (vu + cfg_scale * (vc - vu)) * dt
    return ((zt.clamp(-1, 1) + 1) / 2).clamp(0, 1)


@torch.no_grad()
def conditioning_probe(model, emb_a, emb_b, steps: int = 30,
                        cfg_scale: float = 3.0, seed: int = 999):
    """Same starting noise, two different text embeddings — isolates whether
    conditioning has any causal effect, since comparing validation-grid
    samples to each other conflates embedding and noise differences."""
    model.eval()
    device = next(model.parameters()).device
    g = torch.Generator(device=device).manual_seed(seed)
    zt0    = torch.randn(1, 3, 64, 64, device=device, generator=g)
    uncond = torch.zeros_like(emb_a)
    EPS, dt = 1e-3, 1.0 / steps

    def run(zt, emb):
        for i in range(steps):
            t_val = i / steps * (1.0 - EPS) + EPS
            t = torch.full((1,), t_val, device=device)
            vc = model(zt, t, emb)
            vu = model(zt, t, uncond)
            zt = zt + (vu + cfg_scale * (vc - vu)) * dt
        return zt

    img_a = run(zt0.clone(), emb_a)
    img_b = run(zt0.clone(), emb_b)
    diff  = (img_a - img_b).abs().mean().item()
    imgs  = torch.cat([
        ((img_a.clamp(-1, 1) + 1) / 2).clamp(0, 1),
        ((img_b.clamp(-1, 1) + 1) / 2).clamp(0, 1),
    ], dim=0)
    return imgs, diff


# ══════════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════════
def run_training():
    print("=" * 65)
    print("  ADM UNet + RFM | pixel space | 64×64 | text cross-attn")
    print("=" * 65)
    set_seed(42)

    accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=1)
    device      = accelerator.device

    # No RandomHorizontalFlip: anime hair is directional.
    # Flipping creates two conflicting orientation distributions that
    # produce a vertical seam where both patterns collide at x=32.
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # RFM_DATA_DIR overrides the default; must contain <name>.png/.jpg
    # paired with <name>.pt embedding tensors of shape [512].
    dataset_path = os.environ.get("RFM_DATA_DIR")
    if not dataset_path:
        dataset_path = "your dataset path"
    if not os.path.exists(dataset_path):
        dataset_path = "your dataset path"

    dataset    = ImageDataset(dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True,
                            num_workers=2, pin_memory=True, drop_last=True)

    # ── Pre-flight embedding check ────────────────────────────────
    if accelerator.is_main_process:
        cov = dataset.check_embeddings()
        print(f"\n  Embedding coverage: {cov['ok']}/{cov['total']} real | "
              f"{cov['missing']} missing ({cov['missing_pct']:.1f}%) | "
              f"mean norm={cov['mean_norm']:.2f}")
        if cov["missing_pct"] > 50:
            raise RuntimeError(
                f"{cov['missing_pct']:.1f}% of .pt embedding files are missing. "
                f"Expected <image_name>.pt beside each image in {dataset_path}. "
                f"Fix embedding locations before training."
            )
        if cov["missing_pct"] > 5:
            print(f"  [WARNING] {cov['missing_pct']:.1f}% missing — "
                  f"watch HEALTH_CROSS_ATTN in epoch 1 logs.\n")
        else:
            print(f"  [OK] Embeddings healthy.\n")

    model = FlowMatchingUNet(
        in_channels          = 3,
        model_channels       = 128,
        context_dim          = 512,
        channel_mult         = (1, 2, 4),
        num_res_blocks       = 2,
        head_dim             = 64,
        n_context_tokens     = 16,
        dropout              = 0.1,
        attention_resolutions = (32, 16),
        input_size           = 64,
    ).to(device)

    # ── Parameter groups: backbone / self-attn / cross-attn ──────
    # Classified via isinstance(), not name matching (proj_out exists in
    # both attention types, so substrings collide). Cross-attention gets a
    # higher LR since its K/V start uncorrelated with Q (text vs. image
    # features), bootstrapping slower than self-attention's single-source
    # Q/K/V. See grad_self_attn / grad_cross_attn in the diagnostic output.
    cat_of_param = {}
    for _, mod in accelerator.unwrap_model(model).named_modules():
        if isinstance(mod, SpatialCrossAttn):
            for p in mod.parameters(): cat_of_param[id(p)] = "cross"
        elif isinstance(mod, SpatialSelfAttn):
            for p in mod.parameters(): cat_of_param[id(p)] = "self"

    SHARED_COND_KEYS = ("ctx_expand", "ctx_pos", "time_embed")
    base_p, self_p, cross_p = [], [], []
    for n, p in accelerator.unwrap_model(model).named_parameters():
        cat = cat_of_param.get(id(p))
        if cat == "cross":
            cross_p.append(p)
        elif cat == "self":
            self_p.append(p)
        elif any(k in n for k in SHARED_COND_KEYS):
            self_p.append(p)   # text-expansion/time-embed grouped with self
        else:
            base_p.append(p)

    optimizer = torch.optim.AdamW([
        {"params": base_p,  "lr": 1e-4, "weight_decay": 1e-2},
        {"params": self_p,  "lr": 5e-4, "weight_decay": 1e-3},
        # Reversible hyperparameter choice, not an architecture change.
        {"params": cross_p, "lr": 8e-4, "weight_decay": 1e-3},
    ])

    epochs          = 100
    steps_per_epoch = math.ceil(len(dataloader) / accelerator.gradient_accumulation_steps)
    total_steps     = epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: min(1.0, s / 500) * 0.5 * (1.0 + math.cos(math.pi * s / total_steps)),
    )

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    tracker = OmniTracker(accelerator.unwrap_model(model))
    ema     = DynamicEMA(accelerator.unwrap_model(model))

    fixed_embs = torch.stack(
        [dataset[i][1] for i in range(min(4, len(dataset)))]
    ).to(device)

    # Fixed noise every epoch so changes in the saved grid reflect the
    # model, not which noise happened to be drawn.
    val_generator = torch.Generator(device=device).manual_seed(999)

    # Most-different pair among the 4 fixed embeddings — the starkest test
    # of whether cross-attention distinguishes them at all.
    _e = fixed_embs / fixed_embs.norm(dim=1, keepdim=True)
    _sim = _e @ _e.T
    _sim.fill_diagonal_(2.0)  # exclude self-matches from argmin
    _i, _j = divmod(_sim.argmin().item(), _sim.shape[0])
    probe_emb_a = fixed_embs[_i:_i+1]
    probe_emb_b = fixed_embs[_j:_j+1]
    print(f"  Conditioning probe will compare dataset indices {_i} and {_j} "
          f"(cosine sim={_sim[_i,_j].item():.3f}, most different pair)\n")

    # ── Optional resume ───────────────────────────────────────────
    RESUME   = False
    CKPT     = "rfm_unet_checkpoint.pt"
    EMA_CKPT = "rfm_unet_ema_checkpoint.pt"
    if RESUME and os.path.exists(CKPT):
        raw = accelerator.unwrap_model(model)
        raw.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
        if os.path.exists(EMA_CKPT):
            ema.shadow.load_state_dict(
                torch.load(EMA_CKPT, map_location="cpu", weights_only=True)
            )
        print(f"[RESUME] Loaded {CKPT}")

    print(f"[TRAINING] {epochs} epochs | "
          f"eff batch={32 * accelerator.num_processes} | "
          f"steps/epoch={steps_per_epoch}\n")

    global_step = 0

    for epoch in range(epochs):
        model.train()
        tracker.reset()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}",
                    disable=not accelerator.is_main_process)

        for batch_idx, (img, text_emb) in enumerate(pbar):
            with accelerator.accumulate(model):
                data     = img.to(device).float()
                text_emb = text_emb.to(device).float()
                B        = data.size(0)

                # Catches dead/missing embeddings immediately instead of
                # discovering it epochs later in the diagnostics.
                if batch_idx == 0 and accelerator.is_main_process:
                    pre_norm = text_emb.norm(dim=1).mean().item()
                    if pre_norm < 1e-4:
                        print(f"\n  [WARNING epoch {epoch+1}] text_emb norm≈0 "
                              f"before CFG dropout. Embeddings may be missing or "
                              f"collapsed — check dataset .pt files.\n")

                # CFG: 15% of samples use null (zero) conditioning
                mask = torch.rand(B, device=device) < 0.15
                text_emb = text_emb.clone()
                text_emb[mask] = 0.0

                noise = torch.randn_like(data)

                # Logit-normal t: u~N(0,1), t=sigmoid(u)
                # → 60% of batches in [0.3,0.7] (hard zone) vs 40% for uniform
                t     = torch.sigmoid(torch.randn(B, device=device))
                t_exp = t.view(-1, 1, 1, 1)

                zt       = (1.0 - t_exp) * noise + t_exp * data
                target_v = data - noise

                with accelerator.autocast():
                    v_pred = model(zt, t, text_emb)
                    loss   = F.mse_loss(v_pred.float(), target_v.float())

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    gn = accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    if accelerator.is_main_process:
                        tracker.update(
                            loss.item(), data, noise, zt, v_pred, t_exp, gn
                        )
                        ema.update(accelerator.unwrap_model(model), global_step)
                        global_step += 1

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            if accelerator.is_main_process:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr":   f"{scheduler.get_last_lr()[0]:.2e}",
                })

        if accelerator.is_main_process:
            rep = tracker.report()
            print(f"\nEpoch {epoch+1:3d} | loss={rep['loss']:.5f} | "
                  f"cosine={rep['cosine_sim']:.4f} | "
                  f"norm={rep['grad_norm']:.3f} | "
                  f"GradFlow={rep['HEALTH_GRAD_FLOW']} | "
                  f"CrossAttn={rep['HEALTH_CROSS_ATTN']} "
                  f"(ratio={rep['attn_ca_ratio']:.4f})")
            if rep["HEALTH_GRAD_NORM_SPIKE"] != "PASS" and \
               not rep["HEALTH_GRAD_NORM_SPIKE"].startswith("PASS"):
                print(f"  *** {rep['HEALTH_GRAD_NORM_SPIKE']} ***")

            torch.save(
                accelerator.unwrap_model(model).state_dict(),
                "rfm_unet_checkpoint.pt"
            )
            torch.save(ema.shadow.state_dict(), "rfm_unet_ema_checkpoint.pt")

            imgs = euler_sample(ema.shadow, fixed_embs, steps=30, cfg_scale=3.0,
                                generator=val_generator)
            save_image(imgs, f"rfm_val_epoch_{epoch+1:03d}.png", nrow=2)

            # Saved every epoch: same noise, two contrasting embeddings.
            # If this stays near-identical while loss keeps dropping, the
            # bottleneck is conditioning specifically, not the backbone.
            probe_imgs, probe_diff = conditioning_probe(
                ema.shadow, probe_emb_a, probe_emb_b, steps=30, cfg_scale=3.0
            )
            save_image(probe_imgs, f"rfm_condprobe_epoch_{epoch+1:03d}.png", nrow=2)
            rep["conditioning_probe_diff"] = round(probe_diff, 5)
            print(f"  Conditioning probe (same noise, contrasting embeddings): "
                  f"pixel diff={probe_diff:.4f} "
                  f"({'text has visible effect' if probe_diff > 0.03 else 'still weak/no visible effect'})\n")

            with open(f"rfm_diag_epoch_{epoch+1:03d}.json", "w") as f:
                json.dump({"EPOCH": epoch + 1, "STATS": rep}, f, indent=4)


if __name__ == "__main__":
    run_training()
