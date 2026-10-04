#!/usr/bin/env python3
"""Score your simulator the way the contest will.

    usvsim replay-eval SUBMISSION
    usvsim replay-eval SUBMISSION --quick          # at most 5 replays per group (the kit ships 2)
    usvsim replay-eval examples/submission
    python -m usvsim.replay_eval SUBMISSION        # the same, as a module

SUBMISSION is a `t2-1-submission/1` directory (`docs/PACKAGING.md`) or, locally, a bare `.py`.
Its entry module must define `make_sim(config, seed)` returning an object with
`reset(initial_state)` and `step(command, dt)`. See `usvsim/plant.py` for the contract
and `examples/` for worked submissions.

This runs against the **released** logs, which you also fit on, so the number it prints is
optimistic -- it is the training score, not an estimate of your hidden-set score. Its job
is to tell you that your submission loads, runs, and is not worse than the shipped
simulator. The gap between this number and your leaderboard number is how much
you overfitted.

Exit codes: 0 scored, 2 could not load the submission.
"""
import argparse
import os
import sys


import numpy as np

from . import data, metric, plant
from .submission import load


def add_args(p):
    p.add_argument('submission')
    p.add_argument('--quick', action='store_true', help="5 replays per group")
    p.add_argument('--data', default=None, help="path to the data directory")


def run(a):

    if not os.path.exists(a.submission):
        print(f"no such submission: {a.submission}", file=sys.stderr)
        return 2
    try:
        mod = load(a.submission)
    except Exception as e:
        print(f"could not load {a.submission}: {e}", file=sys.stderr)
        return 2

    lset = data.load('released', a.data)
    if a.quick:
        keep = [i for g in sorted(set(lset.groups)) for i in lset.rows(g)[:5]]
        lset = data.LogSet([lset.names[i] for i in keep], [lset.groups[i] for i in keep],
                           lset.commands[keep], lset.states[keep])

    e = metric.error(data.replay(mod, lset), lset.states)
    base = metric.error(data.replay(plant, lset), lset.states)

    print(f"released logs: {len(lset)} replays")
    print(f"  {'group':12s} {'E_base':>8s} {'E':>8s} {'skill':>7s}")
    skills = {}
    for g in lset.scoring_groups():
        rows = lset.rows(g)
        eg, bg = float(e[rows].mean()), float(base[rows].mean())
        skills[g] = metric.skill(eg, bg)
        print(f"  {g:12s} {bg:8.4f} {eg:8.4f} {skills[g]:7.3f}")
    worst = int(np.argmax(e))
    print(f"  worst replay: {lset.names[worst]} at E = {e[worst]:.4f}")
    print(f"\n  S_sim (on the logs you fit) = {metric.s_sim(skills):.2f}")
    return 0



def main(argv=None):
    p = argparse.ArgumentParser(description="2-1 self-scorer")
    add_args(p)
    return run(p.parse_args(argv))


if __name__ == '__main__':
    sys.exit(main())
