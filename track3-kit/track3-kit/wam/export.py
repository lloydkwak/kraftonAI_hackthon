"""Export torch modules to a submission directory: the seven ONNX graphs and `manifest.json`.

`export_graph_set` (below) is the participant's path -- one `nn.Module` per participant graph,
taking the contract's inputs and returning its outputs (`contract.io_spec`) -- and is what
`template/export.py` calls.  `export_submission` is the organizers' version for their own
reference models; given only a stem it writes the two stem graphs, which is how the kit's
`stem/graphs/` was produced.

**One way to lay out the shared `ctx`** (every participant graph reads and writes the same state),
used by the wrappers here and by the template's default `CTX_SHAPE`: one flat float32 vector holding
the last `T` latent grids, then the last `T - 1` actions --
`ctx = [grids(T*C*g*g), actions(T-1)]`, 49 155 floats at `T` = 4 (0.19 MB).
`predict_step` reads the grids, `update` shifts them and appends its own prediction and
the action (no ground truth), `observe` does the same with the observed grid, and
`act` reads all of it.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn

from . import contract as CT
from . import stem as ST

OPSET = 17


# --------------------------------------------------------------------------------
# The shared ctx and the wrappers
# --------------------------------------------------------------------------------


class FlatCtx:
    """Pack / unpack the reference `ctx`."""

    def __init__(self, t_ctx: int, c: int = CT.C_GRID, g: int = CT.GRID):
        self.t, self.c, self.g = t_ctx, c, g
        self.n_grid = t_ctx * c * g * g

    @property
    def shape(self):
        return (self.n_grid + self.t - 1,)

    def pack(self, grids, actions):
        return torch.cat([grids.reshape(grids.shape[0], -1), actions], dim=1)

    def unpack(self, ctx):
        B = ctx.shape[0]
        grids = ctx[:, :self.n_grid].reshape(B, self.t, self.c, self.g, self.g)
        return grids, ctx[:, self.n_grid:]


class EncodeG(nn.Module):
    def __init__(self, fc: FlatCtx):
        super().__init__()
        self.fc = fc

    def forward(self, grids, actions):
        t = self.fc.t
        return self.fc.pack(grids[:, -t:], actions[:, -(t - 1):])


class PredictStepG(nn.Module):
    """K = 1 deterministic latent dynamics (an organizer reference model), ignoring `z`,
    `level` and `step`, as a deterministic model may."""

    def __init__(self, net, fc: FlatCtx):
        super().__init__()
        self.net, self.fc = net, fc

    def forward(self, z, level, step, actions, ctx):
        grids, _ = self.fc.unpack(ctx)
        B = ctx.shape[0]
        c = grids.reshape(B, self.fc.t * self.fc.c, self.fc.g, self.fc.g)
        z1 = self.net.predict_step(None, 0.0, 1.0, actions[:, 0], c)     # (B, C, g, g)
        return z1[:, None]                                                # (B, 1, C, g, g)


class UpdateG(nn.Module):
    def __init__(self, fc: FlatCtx):
        super().__init__()
        self.fc = fc

    def forward(self, ctx, z_final, actions):
        grids, acts = self.fc.unpack(ctx)
        grids = torch.cat([grids[:, 1:], z_final[:, -1:]], dim=1)
        acts = torch.cat([acts[:, 1:], actions[:, -1:]], dim=1)
        return self.fc.pack(grids, acts)


class ObserveG(nn.Module):
    def __init__(self, fc: FlatCtx):
        super().__init__()
        self.fc = fc

    def forward(self, ctx, grid, action_prev):
        grids, acts = self.fc.unpack(ctx)
        grids = torch.cat([grids[:, 1:], grid[:, None]], dim=1)
        acts = torch.cat([acts[:, 1:], action_prev[:, None]], dim=1)
        return self.fc.pack(grids, acts)


class ActG(nn.Module):
    def __init__(self, pol, fc: FlatCtx):
        super().__init__()
        self.pol, self.fc = pol, fc

    def forward(self, ctx):
        grids, acts = self.fc.unpack(ctx)
        return self.pol.act((grids, acts))


class DecodeG(nn.Module):
    def __init__(self, dec):
        super().__init__()
        self.dec = dec

    def forward(self, z_final):
        B, K = z_final.shape[:2]
        y = self.dec(z_final.reshape(B * K, *z_final.shape[2:]))
        return y.reshape(B, K, *y.shape[1:])


# --------------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------------


def _export(mod, args, names_in, names_out, path):
    mod = mod.eval()
    dyn = {n: {0: "B"} for n in list(names_in) + list(names_out)}
    torch.onnx.export(mod, tuple(args), path, input_names=list(names_in),
                      output_names=list(names_out), dynamic_axes=dyn,
                      opset_version=OPSET, dynamo=False, do_constant_folding=True)


def export_submission(out_dir: str, stem: ST.Stem, dyn=None,
                      pol=None, t_ctx: int = 4, team: str = "organizers-reference",
                      notes: str = "") -> CT.Manifest:
    """Write `graphs/*.onnx` and `manifest.json`.  `dyn` gives the prediction graphs and
    `pol` the control graphs `observe` / `act`; a submission needs both.  Without `dyn` only
    the two stem graphs are written -- the starter kit's architecture-plus-reference-weights."""
    os.makedirs(os.path.join(out_dir, "graphs"), exist_ok=True)
    gd = os.path.join(out_dir, "graphs")
    C, g, R = CT.C_GRID, CT.GRID, CT.RES
    stem = stem.eval().cpu()
    dev = "cpu"
    B = 2
    _export(stem.encoder, (torch.zeros(B, 3, R, R),), ["frame"], ["grid"],
            os.path.join(gd, "visual_encode.onnx"))
    K = 1
    _export(DecodeG(stem.decoder), (torch.zeros(B, K, C, g, g),), ["z_final"], ["frames"],
            os.path.join(gd, "visual_decode.onnx"))
    if dyn is None:
        return None
    fc = FlatCtx(t_ctx)
    ctx0 = torch.zeros(B, *fc.shape)
    _export(EncodeG(fc), (torch.zeros(B, CT.DIAG_FRAMES, C, g, g), torch.zeros(B, CT.DIAG_ACTIONS)),
            ["grids", "actions"], ["ctx"], os.path.join(gd, "encode.onnx"))
    _export(PredictStepG(dyn.cpu(), fc),
            (torch.zeros(B, K, C, g, g), torch.zeros(B), torch.zeros(B), torch.zeros(B, K), ctx0),
            ["z", "level", "step", "actions", "ctx"], ["z_next"],
            os.path.join(gd, "predict_step.onnx"))
    _export(UpdateG(fc), (ctx0, torch.zeros(B, K, C, g, g), torch.zeros(B, K)),
            ["ctx", "z_final", "actions"], ["ctx_next"], os.path.join(gd, "update.onnx"))
    if pol is None:
        raise ValueError("observe / act are part of every submission: pass `pol`")
    _export(ObserveG(fc), (ctx0, torch.zeros(B, C, g, g), torch.zeros(B)),
            ["ctx", "grid", "action_prev"], ["ctx_next"], os.path.join(gd, "observe.onnx"))
    _export(ActG(pol.cpu(), fc), (ctx0,), ["ctx"], ["action"], os.path.join(gd, "act.onnx"))
    m = CT.Manifest(context_length=t_ctx, chunk_size=K, prediction_steps=1,
                    ctx_shape=fc.shape, team=team, notes=notes)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(m.to_json(), f, indent=1)
    return m


# --------------------------------------------------------------------------------
# The participant's path: any nn.Module per graph, the contract's tensors
# --------------------------------------------------------------------------------


def _example(shape, batch: int):
    return torch.zeros(*[batch if s is None else s for s in shape], dtype=torch.float32)


def export_graph_set(out_dir: str, stem: ST.Stem, graphs: dict, manifest: CT.Manifest,
                     batch: int = 2) -> CT.Manifest:
    """Write `graphs/*.onnx` and `manifest.json` from one `nn.Module` per participant graph
    (`encode`, `predict_step`, `update`, `observe`, `act`),
    each taking the contract's inputs in the contract's order (`contract.io_spec`).  The
    stem's two graphs are exported from `stem`.  This is what the kit's template calls;
    the organizer-side `export_submission` above is the same thing for the organizers' reference models."""
    bad = manifest.check()
    if bad:
        raise ValueError("manifest: " + "; ".join(bad))
    want = [g for g in manifest.graphs if g not in CT.STEM_GRAPHS]
    missing = [g for g in want if g not in graphs]
    extra = [g for g in graphs if g not in want]
    if missing or extra:
        raise ValueError(f"a submission needs graphs {want}: missing {missing}, unexpected {extra}")
    gd = os.path.join(out_dir, "graphs")
    os.makedirs(gd, exist_ok=True)
    spec = CT.io_spec(manifest)
    C, g, R, K = CT.C_GRID, CT.GRID, CT.RES, manifest.chunk_size
    stem = stem.eval().cpu()
    _export(stem.encoder, (torch.zeros(batch, 3, R, R),), ["frame"], ["grid"],
            os.path.join(gd, "visual_encode.onnx"))
    _export(DecodeG(stem.decoder), (torch.zeros(batch, K, C, g, g),), ["z_final"], ["frames"],
            os.path.join(gd, "visual_decode.onnx"))
    for name in want:
        io = spec[name]
        args = tuple(_example(shape, batch) for shape in io["inputs"].values())
        _export(graphs[name].eval().cpu(), args, list(io["inputs"]), list(io["outputs"]),
                os.path.join(gd, f"{name}.onnx"))
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest.to_json(), f, indent=1)
    return manifest
