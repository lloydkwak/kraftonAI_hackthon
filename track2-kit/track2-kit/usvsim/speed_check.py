#!/usr/bin/env python3
"""How long your simulator takes to be scored.

    usvsim speed-check SUBMISSION [--replays N]
    python -m usvsim.speed_check SUBMISSION       # the same, as a module

SUBMISSION is a `t2-1-submission/1` directory (`docs/PACKAGING.md`) or, locally, a bare `.py`.

Scoring replays your simulator over the hidden set on **one pinned core**. This measures
the same work on the released logs and extrapolates. If you are far outside the budget,
the usual causes are per-step Python object allocation and calling into a framework once
per 0.05 s tick; both are fixable without changing your model.
"""
import argparse
import os
import sys
import time


from . import data, metric, plant
from .submission import load

HIDDEN_REPLAYS = 360             # 60 per group x 6 scoring groups
BUDGET_SECONDS = 600.0           # per submission, one core


def add_args(p):
    p.add_argument('submission')
    p.add_argument('--replays', type=int, default=20)
    p.add_argument('--data', default=None)


def run(a):

    if hasattr(os, 'sched_setaffinity'):
        try:
            os.sched_setaffinity(0, {sorted(os.sched_getaffinity(0))[0]})
            pinned = True
        except OSError:
            pinned = False
    else:
        pinned = False

    lset = data.load('released', a.data)
    n = min(a.replays, len(lset))
    small = data.LogSet(lset.names[:n], lset.groups[:n],
                        lset.commands[:n], lset.states[:n])
    mod = load(a.submission)

    data.replay(mod, data.LogSet(small.names[:1], small.groups[:1],
                                 small.commands[:1], small.states[:1]))   # warm up
    t0 = time.perf_counter()
    data.replay(mod, small)
    dt = time.perf_counter() - t0

    steps = n * small.commands.shape[1]
    projected = dt / n * HIDDEN_REPLAYS * (metric.HIDDEN_SECS / metric.RELEASED_SECS)   # 360 replays of 120 s; the released logs are 60 s
    print(f"  core pinned: {'yes' if pinned else 'no (results will be noisier)'}")
    print(f"  {n} replays, {steps} steps, {dt:.2f} s "
          f"({1e6 * dt / steps:.1f} us per step)")
    print(f"  projected over the hidden set ({HIDDEN_REPLAYS} replays of {metric.HIDDEN_SECS:.0f} s): "
          f"{projected:.1f} s of a {BUDGET_SECONDS:.0f} s budget "
          f"({100 * projected / BUDGET_SECONDS:.0f}%)")
    if projected > BUDGET_SECONDS:
        print("  OVER BUDGET")
        return 1
    print(f"  headroom: {BUDGET_SECONDS / max(projected, 1e-9):.1f}x")
    return 0



def main(argv=None):
    p = argparse.ArgumentParser(description="2-1 speed checker")
    add_args(p)
    return run(p.parse_args(argv))


if __name__ == '__main__':
    sys.exit(main())
