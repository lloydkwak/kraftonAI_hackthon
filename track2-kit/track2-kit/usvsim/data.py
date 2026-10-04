"""Loading the shipped logs, and replaying a simulator over them.

A log set is a set of replays. Each replay is a command sequence and the vessel's
*measured* state as it followed it. The measurement is noisy -- position to about 2 cm,
heading to 0.3 degrees -- drawn once and shipped, so the same submission always scores the
same number on them.

  released   what you fit on: 12 replays of 60 s, two per manoeuvre group, six groups
             (`straight`, `accel_decel`, `turn`, `scurve`, `brake`, `lowspeed`), all scored.
             Each replay was recorded in a weak, constant water current of its own (speed
             0.03-0.10 m/s, direction random, not disclosed): positions are ground-relative,
             velocities water-relative, so the drift shows as the difference between the two.
  (hidden)   not shipped; the problem page on the site describes what the scorer replays.
"""
import json
import os

import numpy as np

from . import metric

# The kit's own data directory, unless USVSIM_DATA points elsewhere. The override exists
# because a submission is allowed to read the released logs at construction time --
# `examples/residual_model.py` fits itself from them -- so the scorer has to be able to
# say which copy of the data a submission is being run against.
DATA = (os.environ.get('USVSIM_DATA')
        or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'data'))


class LogSet:
    def __init__(self, names, groups, commands, states):
        self.names = tuple(names)
        self.groups = tuple(groups)
        self.commands = commands            # (N, T, 3) at 20 Hz
        self.states = states                # (N, T // 2, 6) at 10 Hz, measured

    def __len__(self):
        return len(self.names)

    def rows(self, group):
        return [i for i, g in enumerate(self.groups) if g == group]

    def scoring_groups(self):
        return sorted(set(self.groups))             # every group is scored


def load(name, data_dir=None):
    z = np.load(os.path.join(data_dir or DATA, f'{name}.npz'), allow_pickle=False)
    names = [str(n) for n in z['names']]
    return LogSet(names, [n.rsplit('_', 1)[0] for n in names], z['commands'], z['states'])


def anchors(data_dir=None):
    with open(os.path.join(data_dir or DATA, 'anchors.json')) as f:
        return json.load(f)


def replay_one(sim, commands):
    """One sequence. This is exactly what the scorer does -- match it."""
    sim.reset(np.zeros(6))
    out = []
    for k, cmd in enumerate(commands):
        sim.step(cmd, metric.DT)
        if (k + 1) % metric.LOG_EVERY == 0:
            out.append(np.asarray(sim.s, float).copy())
    return np.array(out)


def replay(mod, lset, config=None, seed=0):
    sim = mod.make_sim(config, seed)
    return np.array([replay_one(sim, c) for c in lset.commands])
