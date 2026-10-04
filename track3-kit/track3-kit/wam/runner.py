"""The ONNX runner: the one program that executes a submission.

    grids = visual_encode(context frames);  ctx = encode(grids, context actions)
    per block: z = initial_latent(seed); N x predict_step; frames = visual_decode(z);
         ctx = update(ctx, z, block actions)                  -- open loop, 3-1
    per step: a = act(ctx); env.step(a); grid = visual_encode(frame);
         ctx = observe(ctx, grid, a)                          -- closed loop, 3-2

and nothing else: no ground-truth observation reaches `update`, raw RGB reaches
`visual_encode` only, and the only randomness is the runner's own `initial_latent` from the
published seed bundle.  The official scorer and the validator drive submissions through this
class, so there is one execution path.

Execution provider: CPU by default; the official scorer uses the pinned CUDA runtime
(`wam.runtime`).  Every call is timed and counted.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import numpy as np

from . import contract as CT
from . import runtime as RT


@dataclass
class CallStats:
    n: int = 0
    seconds: float = 0.0
    times: list = field(default_factory=list)

    def add(self, dt):
        self.n += 1
        self.seconds += dt
        self.times.append(dt)


class Submission:
    """A submission directory: `manifest.json` + `graphs/*.onnx`, opened with ONNX
    Runtime.  Graph calls go through `run`, which checks names and dtypes and times
    every call."""

    def __init__(self, path: str, provider: str = "CPUExecutionProvider", threads: int = 4):
        """`provider`: an ONNX Runtime provider name or one of `runtime.ALIASES`
        (`"pinned"` / `"cuda"` / `"cpu"`).  The provider asked for must be the one that
        ends up active -- `runtime.make_session` raises on a silent fallback."""
        self.path = path
        with open(os.path.join(path, "manifest.json")) as f:
            self.manifest = CT.Manifest.from_json(json.load(f))
        bad = self.manifest.check()
        if bad:
            raise ValueError("manifest: " + "; ".join(bad))
        self.provider = RT.resolve(provider)[0]
        self.sessions = {}
        for name in self.manifest.graphs:
            gp = os.path.join(path, "graphs", f"{name}.onnx")
            if not os.path.exists(gp):
                raise FileNotFoundError(f"graph {name!r} missing: {gp}")
            self.sessions[name] = RT.make_session(gp, provider, threads)
        self.stats = {name: CallStats() for name in self.manifest.graphs}
        self.spec = CT.io_spec(self.manifest)

    # -- one call -----------------------------------------------------------------

    def run(self, name: str, **inputs):
        sess = self.sessions[name]
        feed = {}
        for i in sess.get_inputs():
            if i.name not in inputs:
                raise KeyError(f"{name}: input {i.name!r} not supplied")
            x = np.ascontiguousarray(inputs[i.name])
            want = {"tensor(float)": np.float32, "tensor(float16)": np.float16,
                    "tensor(double)": np.float64, "tensor(int64)": np.int64}.get(i.type)
            if want is not None and x.dtype != want:
                x = x.astype(want)
            feed[i.name] = x
        t0 = time.perf_counter()
        out = sess.run(None, feed)
        self.stats[name].add(time.perf_counter() - t0)
        names = [o.name for o in sess.get_outputs()]
        return dict(zip(names, out))

    # -- the three procedures -------------------------------------------------------

    def frames_to_input(self, frames_uint8):
        """uint8 `(B, R, R, 3)` -> float32 `(B, 3, R, R)` in [0, 1]."""
        x = np.asarray(frames_uint8)
        return (x.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)

    def visual_encode(self, frames_uint8):
        return self.run("visual_encode", frame=self.frames_to_input(frames_uint8))["grid"]

    def encode_context(self, context_frames, context_actions):
        """The diagnostic segment -> `ctx`.  `context_frames` uint8 `(B, T, R, R, 3)`, `context_actions` `(B, T-1)`."""
        B, T = context_frames.shape[:2]
        grids = np.stack([self.visual_encode(context_frames[:, t]) for t in range(T)], axis=1)
        if T != CT.DIAG_FRAMES or context_actions.shape[1] != CT.DIAG_ACTIONS:
            raise ValueError(f"encode takes the whole diagnostic segment, "
                             f"{CT.DIAG_FRAMES} frames and {CT.DIAG_ACTIONS} actions; got "
                             f"{T} and {context_actions.shape[1]}")
        return self.run("encode", grids=grids, actions=context_actions.astype(np.float32))["ctx"]

    def predict(self, ctx, future_actions, seeds):
        """The open-loop rollout.  `future_actions` `(B, L)`, `seeds` one per block (`ceil(L/K)`).
        Returns predicted frames uint8 `(B, L, R, R, 3)` and the final ctx."""
        m = self.manifest
        B, L = future_actions.shape
        K, N = m.chunk_size, m.prediction_steps
        n_blocks = -(-L // K)
        if len(seeds) < n_blocks:
            raise ValueError(f"{n_blocks} blocks need {n_blocks} seeds, got {len(seeds)}")
        out = []
        sched = CT.refinement_schedule(N)
        for b in range(n_blocks):
            acts = future_actions[:, b * K:(b + 1) * K].astype(np.float32)
            if acts.shape[1] < K:                      # last, short block: pad the actions
                acts = np.concatenate([acts, np.repeat(acts[:, -1:], K - acts.shape[1], 1)], 1)
            z = CT.initial_latent(int(seeds[b]), B, K)
            for level, step in sched:
                z = self.run("predict_step", z=z, level=np.full(B, level, np.float32),
                             step=np.full(B, step, np.float32), actions=acts, ctx=ctx)["z_next"]
            frames = self.run("visual_decode", z_final=z)["frames"]         # (B, K, 3, R, R)
            out.append(CT.quantize(frames.transpose(0, 1, 3, 4, 2)))
            ctx = self.run("update", ctx=ctx, z_final=z, actions=acts)["ctx_next"]
        pred = np.concatenate(out, axis=1)[:, :L]
        return pred, ctx

    def act(self, ctx):
        return self.run("act", ctx=ctx)["action"]

    def observe(self, ctx, frame_uint8, action_prev):
        grid = self.visual_encode(frame_uint8)
        return self.run("observe", ctx=ctx, grid=grid,
                        action_prev=np.asarray(action_prev, np.float32))["ctx_next"]

    # -- bookkeeping ----------------------------------------------------------------

    def timing(self) -> dict:
        """Per-graph call count, mean and p95 wall-clock in ms."""
        rep = {}
        for k, s in self.stats.items():
            if s.n:
                t = np.asarray(s.times) * 1e3
                rep[k] = dict(calls=s.n, mean_ms=float(t.mean()), p95_ms=float(np.percentile(t, 95)))
        return rep

    def reset_stats(self):
        for s in self.stats.values():
            s.n, s.seconds, s.times = 0, 0.0, []


def open_stem(path: str, provider: str = "CPUExecutionProvider", threads: int = 4) -> Submission:
    """Open a directory holding only the two stem graphs (`visual_encode`,
    `visual_decode`) -- the starter kit's reference stem, which the scorer uses for
    the copy-last-frame floor (the last context frame through the stem).  No manifest is needed; a synthetic
    manifest with `K = 1` is used for the shapes."""
    sub = Submission.__new__(Submission)
    sub.path = path
    sub.manifest = CT.Manifest(context_length=1, chunk_size=1, prediction_steps=1, ctx_shape=(1,))
    sub.provider = RT.resolve(provider)[0]
    sub.sessions = {}
    for name in CT.STEM_GRAPHS:
        gp = os.path.join(path, "graphs", f"{name}.onnx")
        if not os.path.exists(gp):
            raise FileNotFoundError(f"stem graph {name!r} missing: {gp}")
        sub.sessions[name] = RT.make_session(gp, provider, threads)
    sub.stats = {name: CallStats() for name in CT.STEM_GRAPHS}
    sub.spec = CT.io_spec(sub.manifest)
    return sub


# --------------------------------------------------------------------------------
# The two evaluation procedures, end to end
# --------------------------------------------------------------------------------


def rollout_prediction(sub: Submission, context_frames, context_actions, future_actions,
                       seed_base: int = 0, n_seeds: int = 1):
    """3-1 for a batch of episodes: returns predicted frames `(S, B, L, R, R, 3)` uint8,
    one rollout per generation seed (scored as the mean over a published seed bundle, never
    best-of-N)."""
    B, L = future_actions.shape
    n_blocks = -(-L // sub.manifest.chunk_size)
    seeds = CT.seed_bundle(seed_base, n_blocks, n_seeds)
    ctx0 = sub.encode_context(context_frames, context_actions)
    preds = []
    for s in range(n_seeds):
        pred, _ = sub.predict(ctx0, future_actions, seeds[s])
        preds.append(pred)
    return np.stack(preds)


def run_control(sub: Submission, env, render_fn, steps: int = 512, diag_actions=None):
    """3-2 for a batch of episodes -- organizer-side: it needs the hidden environment and
    renderer, which do not ship.  `env` is an environment already reset at the episode's
    start; `render_fn(states) -> uint8 (B, R, R, 3)` is the hidden renderer.  Applies the
    diagnostic segment with `act` not called and nothing scored, then `steps` of
    act / step / encode / observe.  The segment is the published waveform, or per episode
    the 31 actions `diag_actions` `(B, 31)` (an upright start's balance-controller segment).
    Returns the per-episode return and the state and action histories."""
    from . import diagnostic as D
    B = env.B
    if diag_actions is None:
        diag_actions = np.repeat(D.waveform()[None, :], B, axis=0)
    diag_actions = np.asarray(diag_actions, dtype=np.float32).reshape(B, -1)
    frames = [render_fn(env.s)]
    for k in range(diag_actions.shape[1]):
        env.step(diag_actions[:, k].astype(float))
        frames.append(render_fn(env.s))
    ctx = sub.encode_context(np.stack(frames, axis=1), diag_actions)
    J = np.zeros(B)
    states = np.empty((B, steps + 1, 7))
    actions = np.empty((B, steps))
    for t in range(steps):
        states[:, t] = env.s
        a = np.clip(sub.act(ctx), -1.0, 1.0)
        actions[:, t] = a
        _, r, _ = env.step(a)
        J += r
        ctx = sub.observe(ctx, render_fn(env.s), a)
    states[:, steps] = env.s
    return J, states, actions
