"""Training skeleton -- the data path is written, the model and the loss are yours.

    python template/train.py --dataset dataset --memmap work/frames.npy --steps 20000 --out work/ckpt.pt

What is written for you: decoding the dataset once into a uint8 memmap (`wam.package`),
sampling windows of frames with their actions (labelled episodes) or without them
(video-only episodes), and running frames through the stem's encoder.  What is not:
the model (`template/model.py`), the loss, the optimiser schedule, what to do with the
video-only majority of the data.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if KIT not in sys.path:
    sys.path.insert(0, KIT)

from wam import contract as CT                                        # noqa: E402
from wam import package as PK                                         # noqa: E402

import model as M                                                     # noqa: E402  (template/model.py)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Data:
    """The dataset as one uint8 memmap `(N_frames, R, R, 3)` plus per-episode offsets and
    actions.  `actions` is NaN on video-only episodes (`labelled` says which)."""

    def __init__(self, dataset_dir: str, memmap_path: str):
        if not os.path.exists(memmap_path):
            os.makedirs(os.path.dirname(os.path.abspath(memmap_path)), exist_ok=True)
            log("decoding the dataset once into", memmap_path)
            PK.to_memmap(dataset_dir, memmap_path, log=log)
        self.frames = np.load(memmap_path, mmap_mode="r")
        idx = np.load(memmap_path.replace(".npy", "") + "_index.npz")
        self.offsets, self.actions, self.labelled = idx["offsets"], idx["actions"], idx["labelled"]
        self.n_episodes = len(self.offsets) - 1
        log(f"{self.n_episodes} episodes, {len(self.frames):,} frames, "
            f"{int(self.labelled.sum())} with actions")

    def window(self, rng: np.random.Generator, T: int, labelled_only: bool = False):
        """One window of `T` consecutive frames from one episode: frames `(T, 3, R, R)`
        float32 in [0, 1] and actions `(T - 1,)` (NaN if the episode is video-only).
        `a[t]` applies between frame `t` and frame `t + 1`."""
        pool = np.flatnonzero(self.labelled) if labelled_only else np.arange(self.n_episodes)
        e = int(rng.choice(pool))
        lo, hi = int(self.offsets[e]), int(self.offsets[e + 1])
        s = int(rng.integers(lo, hi - T + 1))
        fr = torch.from_numpy(np.array(self.frames[s:s + T])).permute(0, 3, 1, 2).float() / 255.0   # a copy: the memmap is read-only
        # actions are stored per episode without the (absent) action after the last frame
        a_lo = lo - e                       # action index of frame `lo` (T-1 actions per episode)
        acts = self.actions[a_lo + (s - lo): a_lo + (s - lo) + T - 1]
        return fr, torch.from_numpy(np.asarray(acts, dtype=np.float32))

    def batch(self, rng, B: int, T: int, labelled_only: bool = False):
        fr, ac = zip(*(self.window(rng, T, labelled_only) for _ in range(B)))
        return torch.stack(fr), torch.stack(ac)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.path.join(KIT, "dataset"))
    ap.add_argument("--memmap", default=os.path.join(KIT, "work", "frames.npy"))
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--window", type=int, default=M.T_CTX + 8, help="frames per training window")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=os.path.join(KIT, "work", "ckpt.pt"))
    a = ap.parse_args(argv)

    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    data = Data(a.dataset, a.memmap)

    stem = M.load_stem().to(dev)          # frozen here; retrain it if you want (wam.stem.stem_loss is its recipe's loss)
    for p in stem.parameters():
        p.requires_grad_(False)
    built = M.build()
    graphs = {k: m.to(dev).train() for k, m in built["graphs"].items()}
    params = [p for m in graphs.values() for p in m.parameters()]
    if not params:
        raise SystemExit("template/model.py has no parameters yet -- fill the TODO blocks first")
    opt = torch.optim.AdamW(params, lr=a.lr)

    for step in range(1, a.steps + 1):
        frames, actions = data.batch(rng, a.batch, a.window)            # (B, T, 3, R, R), (B, T-1)
        frames, actions = frames.to(dev), actions.to(dev)
        with torch.no_grad():
            B, T = frames.shape[:2]
            grids = stem.encoder(frames.reshape(B * T, *frames.shape[2:])).reshape(B, T, CT.C_GRID, CT.GRID, CT.GRID)
        has_actions = ~torch.isnan(actions[:, 0])                        # video-only windows have NaN actions

        # TODO: your loss.  You have `grids` (B, T, C, g, g) -- the stem's latents of T
        # consecutive frames -- and `actions` (B, T-1) where `has_actions` is True.  A
        # rollout through your PredictStep / Update in latent space, compared with the
        # later grids, is the obvious shape; decoding through `stem.decoder` gives a pixel
        # term if you want one.  What you do with the windows that have no actions is the
        # question this dataset asks.
        raise NotImplementedError("TODO: compute `loss` from grids / actions / has_actions")

        opt.zero_grad(set_to_none=True)
        loss.backward()                                                  # noqa: F821  (defined by your TODO)
        opt.step()
        if step % 100 == 0:
            log(f"step {step}/{a.steps} loss {loss.item():.5f}")          # noqa: F821
        if step % 2000 == 0 or step == a.steps:
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            torch.save({k: m.state_dict() for k, m in graphs.items()} | {"stem": stem.state_dict()}, a.out)


if __name__ == "__main__":
    main()
