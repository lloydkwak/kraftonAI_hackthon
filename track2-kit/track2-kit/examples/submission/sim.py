"""Worked example: fit the dials, change nothing structural.

This is the submission a careful participant writes first. It keeps the shipped plant's *form* --
linear damping on each axis, rigid-body mass and inertia, thrust applied directly -- and
fits the five numbers that form exposes to the released logs. It is about as far as parameter
fitting goes: about 8 on the released logs and in the single digits on the hidden replays,
because the difference to the reference is structural.

The constants below were fitted by the organisers (a coordinate scan, then Nelder-Mead) on
released logs only -- an earlier, larger release of the same library (30 replays, of which this
kit ships 12); nothing hidden. The nominal values are in brackets so the size of the fit is
visible.
"""
import numpy as np

ARM, K_STERN, K_BOW = 1.2, 400.0, 150.0
DELTA_MAX = np.pi / 2

X_U = 180.039        # [200.0]  linear surge damping
Y_V = 1271.53        # [300.0]  linear sway damping
N_R = 430.392        # [900.0]  linear yaw damping
MASS = 303.933       # [250.0]
IZZ = 1471.85        # [270.8]


class Sim:
    def reset(self, initial_state=None):
        self.s = np.zeros(6) if initial_state is None else np.array(initial_state, float)
        self.delta = 0.0
        return self.s.copy()

    def step(self, command, dt=0.05):
        T_stern, delta_cmd, T_bow = command
        self.delta = float(np.clip(delta_cmd, -DELTA_MAX, DELTA_MAX))

        fy_s = K_STERN * T_stern * np.sin(self.delta)
        fy_b = K_BOW * T_bow
        fx = K_STERN * T_stern * np.cos(self.delta)
        fy = fy_s + fy_b
        n = ARM * (fy_b - fy_s)

        _, _, psi, u, v, r = self.s
        d = np.array([u * np.cos(psi) - v * np.sin(psi),
                      u * np.sin(psi) + v * np.cos(psi),
                      r,
                      (fx - X_U * u) / MASS,
                      (fy - Y_V * v) / MASS,
                      (n - N_R * r) / IZZ])
        self.s = self.s + dt * d
        return self.s.copy()


def make_sim(config=None, seed=0):
    """`seed` is accepted for API compatibility and ignored: this simulator has no randomness."""
    return Sim()
