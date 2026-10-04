#!/usr/bin/env python3
"""Check your submission will be accepted before you send it.

    usvsim validate-submission SUBMISSION
    python -m usvsim.validate_submission SUBMISSION     # the same, as a module

SUBMISSION is a `t2-1-submission/1` directory (`docs/PACKAGING.md`) or, locally, a bare `.py`
(the layout check then only notes that a contest submission is a directory).

Runs the same gates the intake does, in order, and stops at the first failure. Passing
this does not mean your score will be good; it means your submission will be scored at
all. A submission that fails validation is refused and not scored.
"""
import argparse
import os
import sys
import time


import numpy as np

from . import data, metric, plant
from . import submission as submission_api
from .submission import load

CHECKS = []


def check(name):
    def deco(f):
        CHECKS.append((name, f))
        return f
    return deco


@check("loads and exposes make_sim(config, seed)")
def _loads(ctx):
    ctx['mod'] = load(ctx['path'])
    return None


@check("make_sim returns an object with reset() and step()")
def _shape(ctx):
    sim = ctx['mod'].make_sim(None, 0)
    for m in ('reset', 'step'):
        if not callable(getattr(sim, m, None)):
            return f"the object make_sim returned has no callable .{m}()"
    return None


@check("reset() accepts a 6-element initial state and step() returns a 6-vector")
def _contract(ctx):
    sim = ctx['mod'].make_sim(None, 0)
    sim.reset(np.zeros(6))
    s = sim.step([0.5, 0.1, 0.0], metric.DT)
    s = np.asarray(s if s is not None else getattr(sim, 's', None), float)
    if s is None or s.shape != (6,):
        return f"step() gave {None if s is None else s.shape}, expected a 6-vector"
    return None


@check("runs the released logs without raising")
def _runs(ctx):
    t0 = time.perf_counter()
    ctx['traj'] = data.replay(ctx['mod'], ctx['lset'])
    ctx['seconds'] = time.perf_counter() - t0
    if ctx['traj'].shape != ctx['lset'].states.shape:
        return (f"produced {ctx['traj'].shape}, expected {ctx['lset'].states.shape}. "
                f"Log one state every {metric.LOG_EVERY} steps of {metric.DT} s")
    return None


@check("output is finite")
def _finite(ctx):
    bad = int((~np.isfinite(ctx['traj'])).any(axis=(1, 2)).sum())
    if bad:
        return (f"{bad} of {len(ctx['lset'])} replays contain NaN or inf. This is not a "
                f"rejection -- non-finite samples score as the metric's cap -- but it "
                f"usually means your model diverges and you are leaving points behind")
    return None


@check("not worse than the shipped simulator")
def _better(ctx):
    e = float(metric.error(ctx['traj'], ctx['lset'].states).mean())
    base = float(metric.error(data.replay(plant, ctx['lset']),
                              ctx['lset'].states).mean())
    ctx['E'], ctx['E_base'] = e, base
    if not e < base:
        return (f"E = {e:.4f} against the shipped simulator's {base:.4f}. Every group "
                f"would score 0; check your units and sign conventions")
    return None


def add_args(p):
    p.add_argument('submission')
    p.add_argument('--data', default=None)


def run(a):

    try:
        entry, man = submission_api.resolve(a.submission)
    except ValueError as e:
        print(f"  FAIL  layout\n        {e}", file=sys.stderr)
        return 2
    if man is not None:
        print(f"  ok    layout ({man['format']}, entry {man['entry']})")
    else:
        print("  ok    layout (bare module; a contest submission is a directory, see PACKAGING.md)")

    lset = data.load('released', a.data)
    keep = [i for g in sorted(set(lset.groups)) for i in lset.rows(g)[:4]]
    ctx = {'path': entry,
           'lset': data.LogSet([lset.names[i] for i in keep],
                               [lset.groups[i] for i in keep],
                               lset.commands[keep], lset.states[keep])}

    warned = False
    for name, fn in CHECKS:
        try:
            problem = fn(ctx)
        except Exception as e:
            print(f"  FAIL  {name}\n        {type(e).__name__}: {e}")
            return 1
        soft = problem is not None and name == "output is finite"
        if problem is not None and not soft:
            print(f"  FAIL  {name}\n        {problem}")
            return 1
        if soft:
            warned = True
            print(f"  WARN  {name}\n        {problem}")
        else:
            print(f"  ok    {name}")

    print(f"\n  {len(ctx['lset'])} replays in {ctx['seconds']:.2f} s "
          f"({1000 * ctx['seconds'] / len(ctx['lset']):.0f} ms each); "
          f"E {ctx['E']:.4f} against E_base {ctx['E_base']:.4f}")
    print("  accepted" + (" (with warnings)" if warned else ""))
    return 0



def main(argv=None):
    p = argparse.ArgumentParser(description="2-1 submission validator")
    add_args(p)
    return run(p.parse_args(argv))


if __name__ == '__main__':
    sys.exit(main())
