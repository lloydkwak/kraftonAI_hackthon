"""The fixed learner: the one model, and the one training recipe, that every submitted 2-2 dataset
trains.

2-2 scores a dataset, so the training procedure is fixed: architecture, optimiser, schedule, batch
size, epoch count, seeding and the action normalisation are constants, and the only input is the
data. This module is that recipe. The organisers train every submitted dataset with it from
scratch, once per training seed (0, 1 and 2), and it does not change during the contest.

**The model, `StackPolicy`** (`CANON_ARCH = 'stack'`): the four frames of the observation window
stacked as twelve channels into a truncated ResNet-18 (stem and layers 1-3, GroupNorm, no
pretrained weights); a spatial-softmax branch off layer 2 (to 128 D) and a global-average-pooled
branch off layer 3 (to 64 D), concatenated to 192 D; the newest decision's surge, sway, yaw rate
and `d_target` through a 64 D embedding; an MLP head on the 256 D concatenation (linear layers
256 -> 256 -> 128 -> 3) with `tanh`, mapped onto the action bounds. Deterministic, about 3.0 M
parameters. No recurrence, no numeric history, no actuator state: motion is read from the image
stack, so each decision depends only on the last 4.5 s of input. `RECIPE` names the recipe; the
leaderboard records it.

**Training** (`train`): AdamW (`LR`, `WEIGHT_DECAY`), `EPOCHS` epochs of batches of `BATCH`, a
linear warm-up over the first `WARMUP_FRAC` of steps then a cosine decay, Huber loss (`HUBER_DELTA`)
on all three action channels, each normalised to [-1, 1] over its bounds so no channel's units set
the objective. Numeric inputs are scaled by the published constants `NUMERIC_SCALE`, not by
statistics of your data.

**GroupNorm, not BatchNorm**, so the model computes the same function when it drives one window
at a time in closed loop as it did in training.

**No augmentation.** Nothing is added to your frames during training: any variation the policy
should be robust to has to be in the dataset.

**No validation split and no early stopping.** Nothing is held out: all records are training data.

`Policy` (`arch='gru'`) is an alternative recurrent model over the same window; it is not the
contest's learner. The `NUMERIC_MODES`, `FRAME_MODES`, `numeric_frame_mask`, `history_dropout` and
`LearnedPolicy(ablate=...)` options are diagnostics; the contest recipe uses none of them
(`'full'`, `'all'`, no dropout, no ablation).
"""
import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np                                                          # noqa: E402
import torch                                                                # noqa: E402
import torch.nn as nn                                                       # noqa: E402
import torch.nn.functional as F                                             # noqa: E402

from .closedloop import STRIDE, WINDOW                                      # noqa: E402

# --------------------------------------------------------------------------- the contract

EPOCHS = 30
BATCH = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
WARMUP_FRAC = 0.05
HUBER_DELTA = 0.5

#: Action bounds (signed stern thrust, azimuth within +/- pi/2, signed bow thrust), defined with the format.
from .dataset import ACTION_HI, ACTION_LO                                   # noqa: E402,F401

#: Numeric-input normalisation (published constants, not measured from the submission):
#: surge / sway / yaw rate in m/s, m/s, rad/s, then achieved [T_stern, delta, T_bow], then
#: `d_target` in metres.
NUMERIC_SCALE = np.array([2.0, 0.6, 0.6, 1.0, np.pi / 2, 1.0, 18.0], np.float32)
NUMERIC_DIM = len(NUMERIC_SCALE) + 1          # + history mask

#: Which numeric inputs the policy is given (a diagnostic option; the recipe is 'full'). `StackPolicy`
#: reads only surge, sway, yaw rate and `d_target` whatever the mode.
NUMERIC_MODES = {                             # d_target is a command and is always given
    'full':        np.array([1, 1, 1, 1, 1, 1, 1], np.float32),
    'motion_only': np.array([1, 1, 1, 0, 0, 0, 1], np.float32),
    'none':        np.array([0, 0, 0, 0, 0, 0, 1], np.float32),
}


def numeric_mask(mode):
    return NUMERIC_MODES[mode]


def numeric_frame_mask(mode):
    """(WINDOW, 7) per-position numeric mask (a diagnostic option). 'actuator_last': the achieved
    actuator state is given for the newest position only."""
    m = np.ones((WINDOW, 7), np.float32)
    if mode == 'actuator_last':
        m[:-1, 3:6] = 0.0
    elif mode not in (None, 'full'):
        raise ValueError(mode)
    return m


#: Which frames of the observation window the policy is given (a diagnostic option; the recipe is
#: 'all'). 'last' keeps only the newest frame (and its numeric row).
FRAME_MODES = {'all': np.ones(WINDOW, np.float32),
               'last': np.array([0] * (WINDOW - 1) + [1], np.float32),
               # the minimum that can read the guide's *motion*: the newest frame and the one
               # before it in the window, 1.5 s apart (stride 3 at 0.5 s)
               'last2': np.array([0] * (WINDOW - 2) + [1, 1], np.float32)}


def frame_mask(mode):
    return FRAME_MODES[mode]


# ------------------------------------------------------------------------------ the model

def _gn(c):
    return nn.GroupNorm(min(32, c // 4), c)


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.n1 = _gn(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.n2 = _gn(cout)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), _gn(cout))

    def forward(self, x):
        y = F.relu(self.n1(self.c1(x)))
        y = self.n2(self.c2(y))
        return F.relu(y + (x if self.short is None else self.short(x)))


class SpatialSoftmax(nn.Module):
    """Per-channel expected (row, col) of a softmax over the feature map -- keypoints.

    Under `bow_up` the vessel is pinned to the frame centre, so *where* in the image the guide
    and the landmarks are is the state; a global pool alone would throw that away. Coordinates
    in [-1, 1]."""

    def __init__(self, temperature=1.0):
        super().__init__()
        self.t = nn.Parameter(torch.tensor(float(temperature)))

    def forward(self, x):
        b, c, h, w = x.shape
        p = F.softmax(x.reshape(b, c, h * w) / self.t, dim=-1)
        rr = torch.linspace(-1, 1, h, device=x.device)
        cc = torch.linspace(-1, 1, w, device=x.device)
        grid_r = rr[:, None].expand(h, w).reshape(1, 1, h * w)
        grid_c = cc[None, :].expand(h, w).reshape(1, 1, h * w)
        return torch.cat([(p * grid_r).sum(-1), (p * grid_c).sum(-1)], dim=1)     # (b, 2c)


class Encoder(nn.Module):
    """Truncated ResNet-18 (stem, layer1-3) with the two branches, to 192 D per frame (or per stack of
    frames: `in_ch` = 3 x frames for the stacked learner)."""

    def __init__(self, width=64, in_ch=3):
        super().__init__()
        w = width
        self.stem = nn.Sequential(nn.Conv2d(in_ch, w, 7, 2, 3, bias=False), _gn(w), nn.ReLU(inplace=True),
                                  nn.MaxPool2d(3, 2, 1))                # 128 -> 32
        self.layer1 = nn.Sequential(Block(w, w, 1), Block(w, w, 1))          # 32
        self.layer2 = nn.Sequential(Block(w, 2 * w, 2), Block(2 * w, 2 * w, 1))     # 16
        self.layer3 = nn.Sequential(Block(2 * w, 4 * w, 2), Block(4 * w, 4 * w, 1))  # 8
        self.keypoints = SpatialSoftmax()
        self.kp_proj = nn.Linear(2 * 2 * w, 128)
        self.app_proj = nn.Linear(4 * w, 64)

    def forward(self, x):
        x = self.layer1(self.stem(x))
        l2 = self.layer2(x)
        l3 = self.layer3(l2)
        kp = self.kp_proj(self.keypoints(l2))
        app = self.app_proj(l3.mean(dim=(2, 3)))
        return torch.cat([kp, app], dim=1)                                    # (b, 192)


class Policy(nn.Module):
    def __init__(self, hidden=256, numeric_dim=NUMERIC_DIM):
        super().__init__()
        self.enc = Encoder()
        self.num = nn.Sequential(nn.Linear(numeric_dim, 64), nn.ReLU(inplace=True))
        self.gru = nn.GRU(192 + 64, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(inplace=True), nn.Linear(128, 3))
        self.register_buffer('lo', torch.as_tensor(ACTION_LO))
        self.register_buffer('hi', torch.as_tensor(ACTION_HI))

    def forward(self, rgb, numeric, mask):
        """rgb (b, T, 3, H, W) float in [-1, 1]; numeric (b, T, 6) scaled; mask (b, T).
        Returns the action in *normalised* units, (b, 3) in [-1, 1]."""
        b, T = rgb.shape[:2]
        f = self.enc(rgb.reshape(b * T, *rgb.shape[2:])).reshape(b, T, -1)
        n = self.num(torch.cat([numeric, mask[..., None]], dim=-1))
        z = torch.cat([f, n], dim=-1) * mask[..., None]           # padded frames carry nothing
        out, _ = self.gru(z)                                       # h0 = 0 every call
        return torch.tanh(self.head(out[:, -1]))

    def denorm(self, a):
        return self.lo + (a + 1.0) * 0.5 * (self.hi - self.lo)


class StackPolicy(nn.Module):
    """The contest's model: the window's frames **stacked as channels** into one encoder -- twelve
    input channels for four frames -- the newest frame's ego motion (surge, sway, yaw rate) and
    `d_target` embedded and concatenated to the image features, and an MLP head. No recurrence, no
    numeric history and no actuator state; motion is read from the image stack. Same call signature
    as `Policy`; returns the action in normalised units, (b, 3) in [-1, 1] (`denorm` maps it back)."""
    NUMERIC_IDX = (0, 1, 2, 6)                  # u, v, r, d_target of the newest frame
    numeric_in = len(NUMERIC_SCALE)             # the closed-loop wrapper hands over the whole row

    def __init__(self, hidden=256, numeric_dim=NUMERIC_DIM, frames=WINDOW):
        super().__init__()
        self.n_frames = frames                  # not `frames`: that name is the wrapper's frame *mode* ('all' / 'last')
        self.enc = Encoder(in_ch=3 * frames)
        self.num = nn.Sequential(nn.Linear(len(self.NUMERIC_IDX), 64), nn.ReLU(inplace=True))
        self.head = nn.Sequential(nn.Linear(192 + 64, hidden), nn.ReLU(inplace=True),
                                  nn.Linear(hidden, 128), nn.ReLU(inplace=True), nn.Linear(128, 3))
        self.register_buffer('lo', torch.as_tensor(ACTION_LO))
        self.register_buffer('hi', torch.as_tensor(ACTION_HI))

    def forward(self, rgb, numeric, mask):
        b, T = rgb.shape[:2]
        x = (rgb * mask[:, :, None, None, None]).reshape(b, T * rgb.shape[2], *rgb.shape[3:])
        f = self.enc(x)
        n = self.num(numeric[:, -1][:, list(self.NUMERIC_IDX)])
        return torch.tanh(self.head(torch.cat([f, n], dim=-1)))

    def denorm(self, a):
        return self.lo + (a + 1.0) * 0.5 * (self.hi - self.lo)


ARCHS = {'gru': Policy, 'stack': StackPolicy}
CANON_ARCH = 'stack'                          # the contest's learner; 'gru' is the alternative `Policy`
RECIPE = f'{CANON_ARCH}/1'                    # the learner stamp a leaderboard records


def make_model(arch=CANON_ARCH, **kw):
    return ARCHS[arch](**kw)


def norm_action(a):
    a = np.asarray(a, np.float32)
    return np.clip(2.0 * (a - ACTION_LO) / (ACTION_HI - ACTION_LO) - 1.0, -1.0, 1.0)


def n_params(model=None):
    model = model or make_model()
    return sum(p.numel() for p in model.parameters())


# ------------------------------------------------------------------------------- the data

class Dataset:
    """A dataset (the arrays `dataset.read` returns) held on the training device as one frame per
    record plus window indices.

    Frames are stored once (a decision has one frame) and a window is four indices into them, so
    18,000 records at 128 px are 885 MB rather than 3.5 GB. Windows never cross an episode
    start: window positions before the episode's step 0 are masked, as in closed loop."""

    def __init__(self, rgb, ego_motion, actuator_state, action, episode, step, device='cuda',
                 numeric='full', frames='all', d_target=None, history_dropout=0.0, numeric_frame_mask=None):
        """`history_dropout` (a diagnostic option): at training, each of the older window positions is
        dropped -- frame and numeric row zeroed, history mask 0 -- independently with this probability.
        `numeric_frame_mask` (a diagnostic option): a (WINDOW, 7) mask over the numeric inputs per
        window position, e.g. `actuator_last`. Both are off in the contest recipe."""
        n = len(action)
        rgb = np.ascontiguousarray(rgb)
        self.rgb = torch.as_tensor(rgb).to(device)                          # (n, H, W, 3) u8
        if d_target is None:
            d_target = np.full(n, 12.0, np.float32)     # no d_target given: the default 12 m
        num = np.concatenate([ego_motion, actuator_state,
                              np.asarray(d_target, np.float32)[:, None]], axis=1)
        self.num = torch.as_tensor(num / NUMERIC_SCALE * numeric_mask(numeric)).to(device)
        self.numeric = numeric
        self.y = torch.as_tensor(norm_action(action)).to(device)
        idx = np.empty((n, WINDOW), np.int64)
        mask = np.empty((n, WINDOW), np.float32)
        ep = np.asarray(episode)
        st = np.asarray(step)
        for j in range(WINDOW):
            back = STRIDE * (WINDOW - 1 - j)
            ok = st >= back
            idx[:, j] = np.where(ok, np.arange(n) - back, np.arange(n))
            mask[:, j] = ok
        assert np.all(ep[idx[:, 0]] == ep) and np.all(ep[idx[:, -1]] == ep)
        self.idx = torch.as_tensor(idx).to(device)
        self.mask = torch.as_tensor(mask * frame_mask(frames)).to(device)
        self.frames = frames
        self.device = device
        self.history_dropout = float(history_dropout)
        self.nfm = None if numeric_frame_mask is None else torch.as_tensor(
            np.asarray(numeric_frame_mask, np.float32)).to(device)             # (T, 7)

    def __len__(self):
        return self.y.shape[0]

    def batch(self, i):
        idx, mask = self.idx[i], self.mask[i]                                # (b, T)
        if self.history_dropout > 0:
            keep = (torch.rand(mask.shape, device=mask.device) >= self.history_dropout).float()
            keep[:, -1] = 1.0                                                 # the newest frame stays
            mask = mask * keep
        rgb = self.rgb[idx].permute(0, 1, 4, 2, 3).float().mul_(1 / 127.5).sub_(1.0)
        rgb = rgb * mask[:, :, None, None, None]
        num = self.num[idx] * mask[..., None]
        if self.nfm is not None:
            num = num * self.nfm[None]
        return rgb, num, mask, self.y[i]


# --------------------------------------------------------------------------- determinism

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def strict_determinism(on=True):
    torch.backends.cudnn.deterministic = bool(on)
    torch.backends.cudnn.benchmark = not on
    torch.use_deterministic_algorithms(bool(on), warn_only=True)


# --------------------------------------------------------------------------------- train

def train(data, seed=0, epochs=EPOCHS, batch=BATCH, lr=LR, log=None, arch=CANON_ARCH):
    dev = data.device
    seed_everything(seed)
    model = make_model(arch).to(dev)
    model.arch = arch
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    n = len(data)
    per_epoch = max(n // batch, 1)
    total = epochs * per_epoch
    warm = max(int(WARMUP_FRAC * total), 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: (t + 1) / warm if t < warm
        else 0.5 * (1 + np.cos(np.pi * (t - warm) / max(total - warm, 1))))
    gen = torch.Generator(device=dev).manual_seed(seed + 1)
    model.train()
    t = 0
    hist = []
    for ep in range(epochs):
        perm = torch.randperm(n, generator=gen, device=dev)
        run = 0.0
        for k in range(per_epoch):
            i = perm[k * batch:(k + 1) * batch]
            rgb, num, mask, y = data.batch(i)
            loss = F.huber_loss(model(rgb, num, mask), y, delta=HUBER_DELTA)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run += float(loss)
            t += 1
        hist.append(run / per_epoch)
        if log:
            log(f"    epoch {ep + 1}/{epochs} loss {hist[-1]:.4f}")
    model.eval()
    return model, hist


def save(model, path, meta=None):
    meta = dict(meta or {})
    meta['arch'] = getattr(model, 'arch', 'gru')
    meta['numeric_dim'] = model.num[0].in_features if meta['arch'] == 'gru' else NUMERIC_DIM
    torch.save(dict(state=model.state_dict(), meta=meta), path)


def load(path, device='cpu'):
    ck = torch.load(path, map_location=device)
    meta = ck.get('meta', {})
    arch = meta.get('arch', 'gru')
    m = make_model(arch, numeric_dim=meta.get('numeric_dim', NUMERIC_DIM)).to(device)
    m.arch = arch
    m.load_state_dict(ck['state'])
    m.eval()
    m.numeric = ck.get('meta', {}).get('numeric', 'full')
    m.frames = ck.get('meta', {}).get('frames', 'all')
    m.numeric_frame = ck.get('meta', {}).get('numeric_frame', 'full')
    return m, ck.get('meta', {})


# ------------------------------------------------------------------------------- closed loop

class LearnedPolicy:
    """The trained model as the harness's observation-only policy.

    `ablate` names a diagnostic ablation (None in scoring): 'blackout' zeroes the frames, 'shuffle'
    permutes them in time, 'no_image' is blackout with the numeric inputs kept, 'no_numeric' keeps
    only the frames, 'single_frame' masks all but the newest frame."""
    privileged = False

    def __init__(self, model, device='cpu', ablate=None):
        self.m = model.to(device).eval()
        self.dev = device
        self.ablate = ablate
        # numeric inputs this model was built for: the GRU learner's first layer takes them plus the
        # history mask; the stacked learner takes the full published row and selects inside
        self.nd = getattr(model, 'numeric_in', None) or (model.num[0].in_features - 1)
        self.nmask = torch.as_tensor(numeric_mask(getattr(model, 'numeric', 'full'))[:self.nd],
                                     device=device)
        self.fmask = torch.as_tensor(frame_mask(getattr(model, 'frames', 'all')), device=device)
        self.nfm = torch.as_tensor(numeric_frame_mask(getattr(model, 'numeric_frame', 'full'))[:, :self.nd],
                                   device=device)

    @torch.no_grad()
    def act(self, obs):
        rgb = torch.as_tensor(obs['rgb'], device=self.dev)[None].permute(0, 1, 4, 2, 3).float()
        rgb = rgb.mul_(1 / 127.5).sub_(1.0)
        d = np.asarray(obs.get('d_target', np.full(WINDOW, 12.0)), np.float32)[:, None]
        raw = np.concatenate([obs['ego_motion'], obs['actuator_state'], d], 1) / NUMERIC_SCALE
        num = torch.as_tensor(raw[:, :self.nd], device=self.dev)[None] * self.nmask * self.nfm[None]
        mask = torch.as_tensor(obs['history_mask'], device=self.dev)[None] * self.fmask
        a = self.ablate
        if a in ('blackout', 'no_image'):
            rgb = torch.zeros_like(rgb)
        elif a == 'shuffle':
            rgb = rgb[:, torch.tensor([3, 0, 2, 1])]
        elif a == 'no_numeric':
            num = torch.zeros_like(num)
        elif a == 'single_frame':
            mask = mask * torch.tensor([0, 0, 0, 1.0], device=self.dev)
        rgb = rgb * mask[:, :, None, None, None]
        num = num * mask[..., None]
        out = self.m.denorm(self.m(rgb, num, mask))[0].cpu().numpy()
        return out


def load_policy(path, ablate=None, device='cpu'):
    m, _ = load(path, device)
    return LearnedPolicy(m, device, ablate)
