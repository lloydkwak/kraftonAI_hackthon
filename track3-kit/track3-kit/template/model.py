"""Track 3 submission template -- fill the TODO blocks, keep the signatures.

What is given (do not change): the common visual stem -- its architecture is fixed by the
contract and its reference weights are in `../stem/`; you may retrain them -- the contract
(`wam/contract.py`), the ONNX runner (`wam/runner.py`), the validator (`wam/validator.py`)
and the exporter (`wam/export.py`).  Everything between the stem's latent grid and the
graphs below is yours: the model, the training, the loss.

The five participant graphs and their tensors, per row (a leading batch axis `B` is added
to every tensor; a graph must give the same answer per row whatever `B` is):

    encode(grids (32, C, g, g), actions (31,))              -> ctx      (your CTX_SHAPE)
    predict_step(z (K, C, g, g), level (), step (), actions (K,), ctx) -> z_next (K, C, g, g)
    update(ctx, z_final (K, C, g, g), actions (K,))          -> ctx_next
    observe(ctx, grid (C, g, g), action_prev ())             -> ctx_next
    act(ctx)                                                 -> action () in [-1, 1]

with `C` = 48, `g` = 16 (`wam.contract`).  `ctx` is the one tensor that carries state
between calls; nothing else may.  `update` receives your own prediction, never a
ground-truth frame.  A graph may leave out an input it ignores; it may not add one.

Run `python template/export.py --out <dir>` once the TODOs are filled:
it exports the graph set, writes `manifest.json` and runs the validator on the result.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if KIT not in sys.path:
    sys.path.insert(0, KIT)

from wam import contract as CT                                        # noqa: E402
from wam import stem as ST                                            # noqa: E402

# --------------------------------------------------------------------------------
# Declare your submission (the manifest)
# --------------------------------------------------------------------------------

T_CTX = 8                      # how many of the 32 diagnostic frames your model keeps (<= 32)
K = 1                          # frames per predict_step / visual_decode call (<= 8)
N = 1                          # predict_step calls per block of K frames (<= 4K, <= 32)
# TODO: the layout of your `ctx`, as one flat shape per row.  The default is a window of
# T_CTX latent grids plus the T_CTX - 1 actions between them; <= 4 MB per row.
CTX_SHAPE = (T_CTX * CT.C_GRID * CT.GRID * CT.GRID + (T_CTX - 1),)


def manifest(notes: str = "", external_weights=()) -> CT.Manifest:
    """List external data and pretrained weights in `external_weights`."""
    return CT.Manifest(context_length=T_CTX, chunk_size=K, prediction_steps=N,
                       ctx_shape=tuple(CTX_SHAPE), notes=notes,
                       external_weights=tuple(external_weights))


# --------------------------------------------------------------------------------
# The stem (given)
# --------------------------------------------------------------------------------

def load_stem(path: str = os.path.join(KIT, "stem", "reference_stem.pt")) -> ST.Stem:
    """The common stem with the reference weights.  Retrain it if you want -- the
    architecture is fixed (`wam.stem.Stem.assert_scope`), the weights are yours."""
    ck = torch.load(path, map_location="cpu")
    stem = ST.Stem(ST.StemConfig(**ck["cfg"]))
    stem.load_state_dict(ck["state"])
    return stem.eval()


# --------------------------------------------------------------------------------
# Your graphs.  Each is an nn.Module whose forward takes the contract's inputs in the
# contract's order and returns the contract's output.  Shapes below are per row, with B
# in front.
# --------------------------------------------------------------------------------

class Encode(nn.Module):
    """grids (B, 32, C, g, g), actions (B, 31) -> ctx (B, *CTX_SHAPE).

    Runs once per episode over the diagnostic segment: the only place the episode's
    link lengths and the machine's response are observable.  Keep what you need."""

    def forward(self, grids: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO: build ctx from the diagnostic segment")


class PredictStep(nn.Module):
    """z (B, K, C, g, g), level (B,), step (B,), actions (B, K), ctx -> z_next (B, K, C, g, g).

    One step of your latent dynamics.  `z` is the block's current latent (the runner
    starts each block from tanh(N(0, 1)) with a published seed); `level` = i / N and
    `step` = 1 / N for call i of N.  A deterministic model may ignore z, level and step."""

    def forward(self, z, level, step, actions, ctx) -> torch.Tensor:
        raise NotImplementedError("TODO: predict the next K latent grids")


class Update(nn.Module):
    """ctx, z_final (B, K, C, g, g), actions (B, K) -> ctx_next.

    Called after each block with your own final latent -- there is no ground truth
    between blocks.  Typically: shift the window, append z_final and the actions."""

    def forward(self, ctx, z_final, actions) -> torch.Tensor:
        raise NotImplementedError("TODO: carry ctx forward with your own prediction")


class Observe(nn.Module):
    """ctx, grid (B, C, g, g), action_prev (B,) -> ctx_next.

    Closed-loop control: the true frame's latent grid arrives every step, with the action
    that was applied before it."""

    def forward(self, ctx, grid, action_prev) -> torch.Tensor:
        raise NotImplementedError("TODO: fold the observed grid into ctx")


class Act(nn.Module):
    """ctx -> action (B,) in [-1, 1].

    Reads ctx and chooses the cart force.  It may not write ctx."""

    def forward(self, ctx) -> torch.Tensor:
        raise NotImplementedError("TODO: choose the action")


# --------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------

def build(checkpoint: str | None = None, stem_path: str | None = None) -> dict:
    """Everything the exporter needs: the stem and one module per graph.  Load your
    trained weights here."""
    stem = load_stem(stem_path) if stem_path else load_stem()
    graphs = {"encode": Encode(), "predict_step": PredictStep(), "update": Update(),
              "observe": Observe(), "act": Act()}
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu")
        # TODO: load your state dicts, e.g. graphs["predict_step"].load_state_dict(state["predict_step"])
        # and, if you retrained it, stem.load_state_dict(state["stem"])
        raise NotImplementedError("TODO: load the checkpoint into the modules above")
    return dict(stem=stem, graphs={k: m.eval() for k, m in graphs.items()})
