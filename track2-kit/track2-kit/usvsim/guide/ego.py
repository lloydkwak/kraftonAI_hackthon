"""The public half of the plant: the thruster geometry, and the shipped nominal plant as an object
the closed-loop harness can drive.

The geometry (`thruster_map`, the same map as `usvsim.plant.thruster_map`) is public and shared by
the public and the reference plant. The plant object is 2-1's shipped `usvsim.plant.Plant`
behind an adapter that adds the two things the harness reads: `.s` (it has it) and `.act`, the
achieved actuator state -- which for the public plant is the command itself, azimuth clipped,
because the public plant has no actuator dynamics. That is the public plant, not the reference
one; the difference between the two is what 2-1 is about. `SimPlant` puts any 2-1 simulator
behind the same interface.
"""
import numpy as np

ARM, K_STERN, K_BOW = 1.2, 400.0, 150.0
DELTA_MAX = np.pi / 2


def thruster_map(T_stern, delta, T_bow):
    """Stern azimuth (magnitude + direction) + bow tunnel (magnitude) -> wrench (surge force, sway
    force, yaw moment)."""
    fy_s = K_STERN * T_stern * np.sin(delta)
    fy_b = K_BOW * T_bow
    return (K_STERN * T_stern * np.cos(delta), fy_s + fy_b, ARM * (fy_b - fy_s))


def unmap(fx, fy, n):
    fy_b, fy_s = (fy + n / ARM) / 2.0, (fy - n / ARM) / 2.0
    return np.hypot(fx, fy_s) / K_STERN, np.arctan2(fy_s, fx), fy_b / K_BOW


def _plant_module():
    from .. import plant                                    # the kit's public plant
    return plant


class PublicPlant:
    """`usvsim.plant.Plant` with the harness's interface: `reset(state)`, `step(cmd, dt)`,
    `.s`, `.act`."""

    def __init__(self, params=None):
        self._p = _plant_module().Plant(params)
        self.act = np.zeros(3)
        self.s = self._p.reset()

    def reset(self, state=None):
        self.s = self._p.reset(state)
        self.act = np.zeros(3)
        return self.s.copy()

    def step(self, cmd, dt=0.05):
        c = np.asarray(cmd, float)
        self.act = np.array([c[0], float(np.clip(c[1], -DELTA_MAX, DELTA_MAX)), c[2]])
        self.s = self._p.step(c, dt)
        return self.s.copy()


def make_public_plant():
    return PublicPlant()


class SimPlant:
    """Any 2-1 simulator -- `make_sim(config, seed)` from a submission, or an object with
    `reset(state)` and `step(cmd, dt)` -- behind the harness's interface: `.s` is the state the last
    step returned (or the object's own `.s`), `.act` the command it was given with the azimuth
    clipped, as for the shipped plant, since a 2-1 simulator reports no actuator state of its own."""

    def __init__(self, sim):
        self._sim = sim
        self.act = np.zeros(3)
        self.s = np.zeros(6)

    def reset(self, state=None):
        out = self._sim.reset(np.zeros(6) if state is None else np.asarray(state, float))
        self.s = np.asarray(out if out is not None else getattr(self._sim, 's', np.zeros(6)), float).copy()
        self.act = np.zeros(3)
        return self.s.copy()

    def step(self, cmd, dt=0.05):
        c = np.asarray(cmd, float)
        self.act = np.array([c[0], float(np.clip(c[1], -DELTA_MAX, DELTA_MAX)), c[2]])
        out = self._sim.step(c, dt)
        self.s = np.asarray(out if out is not None else getattr(self._sim, 's', self.s), float).copy()
        return self.s.copy()
