"""The common visual encoder / decoder stem.

The organizers fix the **visual boundary**: a common encoder maps an RGB frame to a spatial
latent grid of `R/8 x R/8 x C_grid`, a common decoder maps a predicted grid back to RGB.  The
**architecture is fixed** (this module) and the **weights are yours**, trained on the
distributed data; the kit's `stem/` holds reference weights you may start from or retrain.
Everything between the two is yours.

Four properties are part of the contract:

**Convolutional feature extraction only.**  No positional encoding, no temporal attention or
convolution, no Transformer / SSM / RNN, no global pooling, and no state or keypoint head --
your design begins where the stem ends.  `assert_scope()` checks the module list against that
rule.

**Exactly 8x spatial reduction, fully convolutional.**  The grid is `R/8` per side (16 at
128 px) with `C_grid` = 48 channels.

**The decoder receives only `z`.**  No raw RGB and no `ctx`.  So everything needed to redraw
the scene -- the appearance axes, the room, the light direction, the cast shadow, the sensor
grain -- has to survive inside `R/8 x R/8 x C_grid` and be re-carried every block.

**The latent is bounded to [-1, 1] by a `tanh`.**  So a `float16` latent cannot overflow, and the
runner's starting latent `tanh(N(0, 1))` lives on the same range.

`cost()` returns the parameter count and the per-call FLOPs at a given resolution.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

REDUCTION = 8                   # R -> R/8, fixed


@dataclass(frozen=True)
class StemConfig:
    """The stem's fixed numbers."""

    c_grid: int = 48            # latent channels: 4x compression against the pixel count;
                                # 32 grids in fp16 are 0.79 MB of `ctx`, against the 4 MB cap
    base: int = 32              # channel width at R/2; R/4 is 2x, R/8 is 3x
    enc_blocks: int = 1
    dec_blocks: int = 2
    groups: int = 8             # GroupNorm groups

    @property
    def channels(self):
        return self.base, 2 * self.base, 3 * self.base


def _norm(c, groups):
    # GroupNorm and not BatchNorm: the stem is a **stateless per-frame map**, and a
    # BatchNorm carries running statistics, which is state outside `ctx` and makes a
    # single-frame call depend on what else was in the batch.
    return nn.GroupNorm(min(groups, c), c)


class ResBlock(nn.Module):
    def __init__(self, c, groups):
        super().__init__()
        self.n1, self.n2 = _norm(c, groups), _norm(c, groups)
        self.c1 = nn.Conv2d(c, c, 3, 1, 1)
        self.c2 = nn.Conv2d(c, c, 3, 1, 1)

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return x + h


class VisualEncoder(nn.Module):
    """`visual_encode(frame) -> visual_grid`.  RGB (B,3,R,R) -> (B,C_grid,R/8,R/8)."""

    def __init__(self, cfg: StemConfig = StemConfig()):
        super().__init__()
        c1, c2, c3 = cfg.channels
        self.cfg = cfg
        # 4x4 stride-2 convs for the three halvings.  The encoder runs every control step,
        # so full-resolution channels are the most expensive place to buy capacity; the
        # capacity is bought at R/8 instead, where a pixel costs 1/64th as much.
        self.d1 = nn.Conv2d(3, c1, 4, 2, 1)
        self.d2 = nn.Conv2d(c1, c2, 4, 2, 1)
        self.d3 = nn.Conv2d(c2, c3, 4, 2, 1)
        self.blocks = nn.ModuleList([ResBlock(c3, cfg.groups)
                                     for _ in range(cfg.enc_blocks)])
        self.out_n = _norm(c3, cfg.groups)
        self.out = nn.Conv2d(c3, cfg.c_grid, 1)

    def forward(self, x):
        h = F.silu(self.d1(x))
        h = F.silu(self.d2(h))
        h = self.d3(h)
        for b in self.blocks:
            h = b(h)
        return torch.tanh(self.out(F.silu(self.out_n(h))))


class VisualDecoder(nn.Module):
    """`visual_decode(z_final) -> frames`.  (B,C_grid,R/8,R/8) -> (B,3,R,R)."""

    def __init__(self, cfg: StemConfig = StemConfig()):
        super().__init__()
        c1, c2, c3 = cfg.channels
        self.cfg = cfg
        self.inp = nn.Conv2d(cfg.c_grid, c3, 1)
        self.blocks = nn.ModuleList([ResBlock(c3, cfg.groups)
                                     for _ in range(cfg.dec_blocks)])
        # PixelShuffle upsampling rather than transposed convolution: same cost, no
        # checkerboard, and the cheap place to do the last halving is *before* it
        # rather than after, which a transposed conv to full resolution gets wrong.
        self.u1n, self.u1 = _norm(c3, cfg.groups), nn.Conv2d(c3, c2 * 4, 3, 1, 1)
        self.u2n, self.u2 = _norm(c2, cfg.groups), nn.Conv2d(c2, c1 * 4, 3, 1, 1)
        self.u3n, self.u3 = _norm(c1, cfg.groups), nn.Conv2d(c1, 3 * 4, 3, 1, 1)

    def forward(self, z):
        h = self.inp(z)
        for b in self.blocks:
            h = b(h)
        h = F.pixel_shuffle(self.u1(F.silu(self.u1n(h))), 2)
        h = F.pixel_shuffle(self.u2(F.silu(self.u2n(h))), 2)
        h = F.pixel_shuffle(self.u3(F.silu(self.u3n(h))), 2)
        return torch.sigmoid(h)


class Stem(nn.Module):
    """Both halves: what you train."""

    def __init__(self, cfg: StemConfig = StemConfig()):
        super().__init__()
        self.cfg = cfg
        self.encoder = VisualEncoder(cfg)
        self.decoder = VisualDecoder(cfg)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    # -- the contract ------------------------------------------------------------

    def assert_scope(self):
        """The stem's exclusion list, checked against the actual module tree."""
        banned = (nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d, nn.MultiheadAttention,
                  nn.LSTM, nn.GRU, nn.RNN, nn.Transformer, nn.TransformerEncoder,
                  nn.Linear, nn.Embedding, nn.BatchNorm2d, nn.BatchNorm1d,
                  nn.Conv1d, nn.Conv3d)
        for name, m in self.named_modules():
            if isinstance(m, banned):
                raise AssertionError(
                    f"{name} ({type(m).__name__}) is outside the stem's fixed scope: the stem "
                    "is convolutional feature extraction only -- no positional "
                    "encoding, temporal mixing, global pooling, or state head")
        for name, p in self.named_parameters():
            if p.dim() == 4 and p.shape[-1] * p.shape[-2] > 16:
                raise AssertionError(f"{name}: kernel larger than 4x4")
        return True

    def latent_shape(self, res: int):
        if res % REDUCTION:
            raise ValueError(f"resolution {res} is not a multiple of {REDUCTION}")
        g = res // REDUCTION
        return (self.cfg.c_grid, g, g)

    def cost(self, res: int = 128) -> dict:
        """Parameter count and per-call FLOPs.

        FLOPs = 2 x MACs, counted analytically from the convolution shapes, so the number
        is reproducible without a GPU.  Norms and activations are ignored; they are under
        1 % here, and the validator's counter (which includes them) is the one reported.
        """
        g = res // REDUCTION
        c1, c2, c3 = self.cfg.channels
        C = self.cfg.c_grid
        R = res

        def conv(hw, cin, cout, k):
            return hw * hw * cin * cout * k * k

        enc = (conv(R // 2, 3, c1, 4) + conv(R // 4, c1, c2, 4)
               + conv(R // 8, c2, c3, 4)
               + self.cfg.enc_blocks * 2 * conv(g, c3, c3, 3)
               + conv(g, c3, C, 1))
        dec = (conv(g, C, c3, 1)
               + self.cfg.dec_blocks * 2 * conv(g, c3, c3, 3)
               + conv(g, c3, c2 * 4, 3) + conv(R // 4, c2, c1 * 4, 3)
               + conv(R // 2, c1, 12, 3))
        n_enc = sum(p.numel() for p in self.encoder.parameters())
        n_dec = sum(p.numel() for p in self.decoder.parameters())
        return {
            "res": res, "c_grid": C, "grid": g,
            "params_encoder": n_enc, "params_decoder": n_dec,
            "params_total": n_enc + n_dec,
            "mflops_encode": 2 * enc / 1e6,
            "mflops_decode_per_frame": 2 * dec / 1e6,
            "latent_scalars_per_frame": C * g * g,
            "pixel_scalars_per_frame": 3 * R * R,
            "compression": 3 * R * R / (C * g * g),
        }


# --------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------
#
# The loss the reference weights were trained with; `stem/reference_stem.pt` records the
# rest of the recipe.


def stem_loss(pred, true, mask, lam: float = 4.0, delta: float = 0.05,
              w_edge: float = 0.25):
    """Mask-weighted multi-scale Huber plus a gradient term.

    `pred`, `true`: frames `(B, 3, R, R)`; `mask`: `(B, R, R)`, a per-pixel weight that
    up-weights object pixels by `1 + lam` (zeros give an unweighted loss).  There is no
    temporal term: the stem is a per-frame map.
    """
    w = 1.0 + lam * mask.unsqueeze(1)
    tot = 0.0
    a, b, ww = pred, true, w
    for i in range(3):
        r = (a - b).abs()
        h = torch.where(r <= delta, 0.5 * r * r / delta, r - 0.5 * delta)
        tot = tot + (ww * h).sum() / (ww.sum() * h.shape[1])
        if i < 2:
            a, b, ww = (F.avg_pool2d(a, 2), F.avg_pool2d(b, 2), F.avg_pool2d(ww, 2))
    tot = tot / 3.0
    if w_edge > 0:
        def grad(x):
            gx = x[..., :, 1:] - x[..., :, :-1]
            gy = x[..., 1:, :] - x[..., :-1, :]
            return gx, gy
        pgx, pgy = grad(pred)
        tgx, tgy = grad(true)
        ex = (w[..., :, 1:] * (pgx - tgx).abs()).sum() / (w[..., :, 1:].sum() * 3)
        ey = (w[..., 1:, :] * (pgy - tgy).abs()).sum() / (w[..., 1:, :].sum() * 3)
        tot = tot + w_edge * 0.5 * (ex + ey)
    return tot
