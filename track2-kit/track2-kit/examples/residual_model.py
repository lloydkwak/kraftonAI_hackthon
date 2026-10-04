"""Worked example: learn the gap instead of deriving it.

Submissions are not required to be physics. This one keeps the shipped plant untouched and
adds a *learned correction* to its three accelerations, as a function of the current
velocities and the current command. Nothing about the vessel is assumed; the correction is
a degree-3 polynomial in the six inputs, fitted by ridge regression to what the
released logs actually did.

How the training target is built, since this is where the work is. The logs give you
measured velocities at 10 Hz. Differentiate them and you have the accelerations the real
vessel produced; subtract what the shipped plant would have produced from the same state
and command, and the remainder is exactly the gap. The catch is that the velocity channels
carry 0.02 m/s of noise and differencing at 0.1 s multiplies it by 10, so the target is
mostly noise and the ridge penalty is doing real work: too little of it and the fitted
correction sends a rollout to infinity on some replays (the scorer counts every non-finite
sample at the cap); even with the value below, many 120 s hidden replays overflow.
Pre-smoothing the velocities was tried and consistently made it *worse*, which is
worth knowing before you spend an afternoon on filter design -- the noise is zero-mean so
least squares already averages it away, while the smoothing also flattens the fast
transients, which is exactly where the two models differ most. The released records also
carry each replay's own constant water current in the positions (not in the velocities);
this example fits on velocities, so it never sees it.

It is deterministic: closed-form least squares on shipped data, no sampling, no `seed`.

**It will not score well, and that is the honest result rather than a flaw in the
example.** It scores about 35 on the 60 s released logs it is fitted on and under 10 on the
120 s hidden replays, against 100 for getting the structure right. A correction fitted one
step at a time is scored over a 120 s rollout of a vessel whose yaw is unstable, so whatever
error the fit leaves behind compounds. An exactly-right functional form carries no error to compound; an
approximately-right one does, however flexible it is. If you want the top of the scale,
the lesson here is that flexibility is not the axis to push on: model the mechanism.
"""
import os
import sys
from itertools import combinations_with_replacement

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from usvsim import data, metric
from usvsim.plant import Plant, thruster_map

RIDGE = 1e-4        # chosen by replay error on the released logs, not by taste
SMOOTH = 1          # moving average on the velocities; 1 = none, and none measured best
DEGREE = 3


def _features(vel, cmd):
    """(N, 3) velocities and (N, 3) commands -> every monomial up to `DEGREE`.

    Raw polynomials in the six inputs, deliberately: putting `sin(delta)` or the thruster
    map in here would be smuggling in physics, and the point of this example is to see how
    far a model that assumes nothing about the vessel actually gets.
    """
    X = np.concatenate([np.asarray(vel, float), np.asarray(cmd, float)], axis=1)
    cols = [np.ones(len(X))]
    for d in range(1, DEGREE + 1):
        for combo in combinations_with_replacement(range(X.shape[1]), d):
            cols.append(np.prod(X[:, combo], axis=1))
    return np.stack(cols, axis=1)


def _smooth(x, w):
    if w <= 1:
        return x
    k = np.ones(w) / w
    pad = w // 2
    padded = np.pad(x, ((0, 0), (pad, pad), (0, 0)), mode='edge')
    return np.apply_along_axis(lambda c: np.convolve(c, k, mode='valid'), 1, padded)


def _fit():
    lset = data.load('released')
    dt = metric.DT * metric.LOG_EVERY                     # 0.1 s between logged states
    states, cmds = lset.states, lset.commands

    vel = _smooth(states[:, :, 3:6], SMOOTH)
    acc = (vel[:, 2:, :] - vel[:, :-2, :]) / (2 * dt)     # centred difference
    mid = vel[:, 1:-1, :]
    cmd = cmds[:, 1::metric.LOG_EVERY, :][:, 1:-1, :]     # command at each logged tick

    base = Plant()
    predicted = np.empty_like(acc)
    u, v, r = mid[..., 0], mid[..., 1], mid[..., 2]
    fx, fy, n = thruster_map(cmd[..., 0],
                             np.clip(cmd[..., 1], -np.pi / 2, np.pi / 2), cmd[..., 2])
    predicted[..., 0] = (fx - base.X_u * u) / base.mass
    predicted[..., 1] = (fy - base.Y_v * v) / base.mass
    predicted[..., 2] = (n - base.N_r * r) / base.izz

    X = _features(mid.reshape(-1, 3), cmd.reshape(-1, 3))
    Y = (acc - predicted).reshape(-1, 3)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    X, Y = X[ok], Y[ok]
    A = X.T @ X + RIDGE * len(X) * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ Y)                    # (F, 3)


_W = None


def weights():
    global _W
    if _W is None:
        _W = _fit()
    return _W


class Sim:
    def __init__(self):
        self.base = Plant()
        self.w = weights()

    def reset(self, initial_state=None):
        self.s = (np.zeros(6) if initial_state is None
                  else np.array(initial_state, float))
        return self.s.copy()

    def step(self, command, dt=0.05):
        # A cubic correction inside an open-loop rollout overflows on some replays (more
        # of them the longer the replay): the correction feeds the state that produced it. That is a real
        # property of this approach rather than a bug to hide, and the metric already says
        # what a non-finite sample is worth (the cap). The errstate only stops numpy
        # printing about it.
        with np.errstate(over='ignore', invalid='ignore'):
            d = self.base.derivs(self.s, command)
            corr = _features(self.s[None, 3:6],
                             np.asarray(command, float)[None, :]) @ self.w
            d[3:6] += corr[0]
            self.s = self.s + dt * d
        return self.s.copy()


def make_sim(config=None, seed=0):
    return Sim()
