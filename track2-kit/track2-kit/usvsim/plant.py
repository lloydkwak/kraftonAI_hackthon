"""The public nominal USV simulator.

This is the whole of what is shipped. It is a *simplified* model of the vessel: planar
rigid-body motion, linear damping on each axis, thrust applied without dynamics. The
reference environment your submission is scored against is this vessel with several
things added, and finding out what they are is subproblem 2-1.

Five parameters are exposed, in `params.json`. Fitting them is a legitimate and expected
first submission -- `examples/submission/` is that fit, packaged as a submission. It will not get you a high score, and
that is by design: the difference between this model and the reference is *structural*,
not a matter of coefficients.

You may edit these parameters, subclass `Plant`, or throw the file away and write your own
simulator. The only fixed contract is:

    make_sim(config, seed) -> object with .reset(initial_state) and .step(command, dt)

`command` is `[T_stern, delta, T_bow]`:

    T_stern   stern azimuth thruster magnitude, -1 to 1
    delta     stern azimuth thruster direction, radians, clipped to +/- pi/2
    T_bow     bow tunnel thruster magnitude, -1 to 1, lateral only

`state` is `[x, y, heading, surge, sway, yaw_rate]`, with body x through the bow, body y
to **port**, and positive yaw rate counter-clockwise seen from above.

The scorer builds your simulator once through `make_sim`, with one fixed `seed`, and runs
it once.
"""
import json
import os

import numpy as np

ARM = 1.2            # m, thruster moment arm from the vessel origin
K_STERN = 400.0      # N at full stern thrust demand
K_BOW = 150.0        # N at full bow thrust demand
DELTA_MAX = np.pi / 2

PARAMS_PATH = os.path.join(os.path.dirname(__file__), 'params.json')


def default_params():
    with open(PARAMS_PATH) as f:
        return json.load(f)


def thruster_map(T_stern, delta, T_bow):
    """Commands to a planar wrench (surge force, sway force, yaw moment).

    Fixed geometry: both the public simulator and the reference use exactly this map. The
    gap you are looking for is not here.
    """
    fy_stern = K_STERN * T_stern * np.sin(delta)
    fy_bow = K_BOW * T_bow
    return (K_STERN * T_stern * np.cos(delta),
            fy_stern + fy_bow,
            ARM * (fy_bow - fy_stern))


class Plant:
    """Planar 3-DOF rigid body with linear damping, integrated with forward Euler."""

    def __init__(self, params=None):
        p = dict(default_params())
        p.update(params or {})
        self.mass = float(p['mass'])
        self.izz = float(p['izz'])
        self.X_u = float(p['X_u'])
        self.Y_v = float(p['Y_v'])
        self.N_r = float(p['N_r'])

    def reset(self, initial_state=None):
        self.s = (np.zeros(6) if initial_state is None
                  else np.array(initial_state, float))
        return self.s.copy()

    def derivs(self, s, command):
        T_stern, delta, T_bow = command
        fx, fy, n = thruster_map(T_stern,
                                 float(np.clip(delta, -DELTA_MAX, DELTA_MAX)),
                                 T_bow)
        _, _, psi, u, v, r = s
        return np.array([u * np.cos(psi) - v * np.sin(psi),
                         u * np.sin(psi) + v * np.cos(psi),
                         r,
                         (fx - self.X_u * u) / self.mass,
                         (fy - self.Y_v * v) / self.mass,
                         (n - self.N_r * r) / self.izz])

    def step(self, command, dt=0.05):
        self.s = self.s + dt * self.derivs(self.s, command)
        return self.s.copy()


def make_sim(config=None, seed=0):
    """`config` is a dict of parameter overrides, or None. `seed` is ignored: the plant has no
    randomness, and the argument exists only so the interface is stable."""
    return Plant(config)
