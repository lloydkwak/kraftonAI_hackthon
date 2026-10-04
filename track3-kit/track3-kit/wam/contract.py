"""The submission contract, as concrete tensors: graph names, shapes, dtypes, limits.

The runner, the validator, the scorer and the exporter all import from here and nowhere else,
so the contract has one definition.

Conventions:

* Every tensor carries a leading batch axis `B` (dynamic), so a scorer can run many
  evaluation episodes at once.  A graph must give the same answer per row whatever
  `B` is -- the validator checks it.
* Frames are float32 `(B, 3, R, R)` in `[0, 1]`, channels first, `R` = 128.
  The runner converts uint8 renders with `x / 255`, and quantises decoded frames back
  with `quantize` (below), so the scorer sees uint8 frames.
* Latent grids are `(B, C, g, g)` with `C` = 48 and `g` = `R / 8` = 16; the
  future latent is `(B, K, C, g, g)`.
* `ctx` is **one** tensor whose shape and dtype the manifest declares; its first axis
  is `B`.  Its byte size per row is capped at 4 MB.
* `encode` always receives the full diagnostic segment, `(B, 32, C, g, g)` grids and
  `(B, 31)` actions; the manifest's `T_ctx <= 32` declares how much of it the
  model keeps and sizes `ctx`.
* Actions are float32 in `[-1, 1]`: `(B, 31)` for `encode`, `(B, K)` for
  `predict_step` / `update`, `(B,)` for `observe`'s previous action and `act`'s output.
* `level` and `step` are float32 `(B,)`: `level_i = i / N`, `step_i = 1 / N` --
  published so a model may depend on them; a deterministic model ignores both.
* A graph's inputs must be a **subset** of the spec's (an ignored input may be left
  out of the graph -- exporters prune them -- and the runner feeds only what a graph
  declares); its outputs must match the spec exactly.
* `initial_latent(seed)`: `tanh(N(0, 1))`, shape `(B, K, C, g, g)`, drawn from a
  published seed bundle.  The stem's latent is `tanh`-bounded, so the
  starting latent lives on the same range.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

RES = 128                       # frame resolution, px
C_GRID = 48                     # latent channels
REDUCTION = 8
GRID = RES // REDUCTION         # 16
T_CTX_MAX = 32                  # the diagnostic segment is exactly 32 frames
K_MAX = 8                       # frames per block
N_MAX = 32                      # predict_step calls per block
N_PER_K = 4                     # N <= 4K: at most 4 predict_step calls per frame
CTX_BYTES_MAX = 4 * 1024 * 1024  # ctx, per row
PARAMS_MAX = 500_000_000        # 0.5B: the parameter limit, and where the size score reaches zero
PARAMS_SIZE_FULL = 1_000_000    # the size score is full at or below this many parameters
LATENT_DTYPES = ("float32", "float16")

# No FLOP caps: compute is bounded by the wall-clock limits (control step p95 <= 40 ms, <= 100 ms per generated
# frame, `encode` <= 1 s) and the 6 GB GPU memory limit.  The validator counts FLOPs and the leaderboard
# publishes them, for reference.
ENCODE_SECONDS_MAX = 1.0        # per encode call under the pinned runtime, B = 1
ARCHIVE_BYTES_MAX = 4 * 1024 ** 3   # the uploaded submission archive, 4 GiB
TRIVIAL_MARGIN = 1.15           # qualify at >= this x the same set's copy-last-frame S_pred
SCORE_POINTS = 100.0            # scores are in points: s_pred = 100 T/L, S_ctrl = 100 x steps standing / 512;
                                # the final score 0.5 S_pred + 0.4 S_ctrl + 10 b_size, out of 100

DIAG_FRAMES = 32                # the diagnostic segment: 32 frames, 31 actions between them
DIAG_ACTIONS = 31

PREDICTION_GRAPHS = ("visual_encode", "visual_decode", "encode", "predict_step", "update")
COMBINED_GRAPHS = PREDICTION_GRAPHS + ("observe", "act")
GRAPHS = COMBINED_GRAPHS        # every submission carries all seven
STEM_GRAPHS = ("visual_encode", "visual_decode")      # fixed architecture, your weights


@dataclass(frozen=True)
class Manifest:
    """Written as `manifest.json` beside `graphs/`.  It carries no participant name or ID: a submission is
    identified by the submission ID it is uploaded with."""

    context_length: int                             # T_ctx
    chunk_size: int                                 # K
    prediction_steps: int                           # N
    ctx_shape: tuple                                # per row, without B
    ctx_dtype: str = "float32"
    latent_dtype: str = "float32"
    input_resolution: tuple = (RES, RES)
    visual_grid: tuple = (GRID, GRID, C_GRID)
    future_latent_grid: tuple = (GRID, GRID, C_GRID)
    team: str = ""                                  # organiser tooling's label only; never written unless set
    external_weights: tuple = ()                    # external data / pretrained weights you used, free text
    notes: str = ""

    @property
    def graphs(self):
        return GRAPHS

    def ctx_bytes(self) -> int:
        n = int(np.prod(self.ctx_shape))
        return n * (2 if self.ctx_dtype == "float16" else 4)

    def check(self) -> list:
        """Static checks a manifest must pass before any graph is opened.  Returns a
        list of violations (empty = fine)."""
        bad = []
        if not (1 <= self.context_length <= T_CTX_MAX):
            bad.append(f"context_length {self.context_length} outside [1, {T_CTX_MAX}]")
        if not (1 <= self.chunk_size <= K_MAX):
            bad.append(f"chunk_size K={self.chunk_size} outside [1, {K_MAX}]")
        if not (1 <= self.prediction_steps <= N_MAX):
            bad.append(f"prediction_steps N={self.prediction_steps} outside [1, {N_MAX}]")
        if self.prediction_steps > N_PER_K * self.chunk_size:
            bad.append(f"N={self.prediction_steps} > 4K={N_PER_K * self.chunk_size}: at most 4 predict_step calls per frame")
        if tuple(self.input_resolution) != (RES, RES):
            bad.append(f"input_resolution must be {(RES, RES)}")
        if tuple(self.visual_grid) != (GRID, GRID, C_GRID):
            bad.append(f"visual_grid must be {(GRID, GRID, C_GRID)}")
        if tuple(self.future_latent_grid) != (GRID, GRID, C_GRID):
            bad.append(f"future_latent_grid must be {(GRID, GRID, C_GRID)}")
        if self.latent_dtype not in LATENT_DTYPES or self.ctx_dtype not in LATENT_DTYPES:
            bad.append("dtypes must be float32 or float16")
        if self.ctx_bytes() > CTX_BYTES_MAX:
            bad.append(f"ctx is {self.ctx_bytes()} bytes per row, cap {CTX_BYTES_MAX} (4 MB)")
        return bad

    def to_json(self) -> dict:
        d = dict(context_length=self.context_length,
                 chunk_size=self.chunk_size, prediction_steps=self.prediction_steps,
                 ctx=dict(shape=list(self.ctx_shape), dtype=self.ctx_dtype),
                 latent_dtype=self.latent_dtype, input_resolution=list(self.input_resolution),
                 visual_grid=list(self.visual_grid), future_latent_grid=list(self.future_latent_grid),
                 external_weights=list(self.external_weights), notes=self.notes)
        if self.team:                               # organiser tooling only; participants' manifests have none
            d["team"] = self.team
        return d

    @staticmethod
    def from_json(d: dict) -> "Manifest":
        return Manifest(context_length=int(d["context_length"]),
                        chunk_size=int(d["chunk_size"]), prediction_steps=int(d["prediction_steps"]),
                        ctx_shape=tuple(d["ctx"]["shape"]), ctx_dtype=d["ctx"].get("dtype", "float32"),
                        latent_dtype=d.get("latent_dtype", "float32"),
                        input_resolution=tuple(d.get("input_resolution", (RES, RES))),
                        visual_grid=tuple(d.get("visual_grid", (GRID, GRID, C_GRID))),
                        future_latent_grid=tuple(d.get("future_latent_grid", (GRID, GRID, C_GRID))),
                        team=d.get("team", ""), external_weights=tuple(d.get("external_weights", ())),
                        notes=d.get("notes", ""))


def io_spec(m: Manifest) -> dict:
    """Expected input / output names and per-row shapes of every graph, given the
    manifest.  `None` in a shape is the batch axis."""
    C, g, K, T = C_GRID, GRID, m.chunk_size, m.context_length
    ctx = (None,) + tuple(m.ctx_shape)
    return {
        "visual_encode": dict(inputs={"frame": (None, 3, RES, RES)},
                              outputs={"grid": (None, C, g, g)}),
        "visual_decode": dict(inputs={"z_final": (None, K, C, g, g)},
                              outputs={"frames": (None, K, 3, RES, RES)}),
        # `encode` always receives the whole diagnostic segment -- 32 grids and 31
        # actions -- and keeps what its declared `T_ctx` wants.
        "encode": dict(inputs={"grids": (None, DIAG_FRAMES, C, g, g),
                               "actions": (None, DIAG_ACTIONS)},
                       outputs={"ctx": ctx}),
        "predict_step": dict(inputs={"z": (None, K, C, g, g), "level": (None,), "step": (None,),
                                     "actions": (None, K), "ctx": ctx},
                             outputs={"z_next": (None, K, C, g, g)}),
        "update": dict(inputs={"ctx": ctx, "z_final": (None, K, C, g, g), "actions": (None, K)},
                       outputs={"ctx_next": ctx}),
        "observe": dict(inputs={"ctx": ctx, "grid": (None, C, g, g), "action_prev": (None,)},
                        outputs={"ctx_next": ctx}),
        "act": dict(inputs={"ctx": ctx}, outputs={"action": (None,)}),
    }


def quantize(frames) -> np.ndarray:
    """float in [0, 1] -> uint8, the way the renderer does it.

    A submission ships uint8 RGB and that is what the scorer sees: the runner quantizes
    every decoded block exactly like this before anything is compared.
    """
    a = np.clip(np.asarray(frames, dtype=np.float64), 0.0, 1.0)
    return (a * 255.0 + 0.5).astype(np.uint8)


def refinement_schedule(n: int):
    """`(level_i, step_i)` for `i` in `0..N-1`.  Published; a model may ignore it."""
    return [(i / n, 1.0 / n) for i in range(n)]


def initial_latent(seed: int, batch: int, k: int, dtype=np.float32) -> np.ndarray:
    """The runner's starting latent for one block.  `tanh(N(0,1))` on the
    stem's own `[-1, 1]` range, from a seed the runner owns."""
    rng = np.random.default_rng(seed)
    return np.tanh(rng.standard_normal((batch, k, C_GRID, GRID, GRID))).astype(dtype)


def seed_bundle(base: int, n_blocks: int, n_seeds: int) -> np.ndarray:
    """The published seeds, one per (generation seed, block).  Deterministic in
    `base` so a local scorer and the official one draw the same latents."""
    return (np.asarray(base, dtype=np.int64) * 1_000_003
            + np.arange(n_seeds)[:, None] * 7919 + np.arange(n_blocks)[None, :]) % (2 ** 31 - 1)
