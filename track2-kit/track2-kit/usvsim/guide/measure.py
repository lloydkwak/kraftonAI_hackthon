"""The measurement model: the noise on the released logs' states, and on `ego_motion` in 2-2.

A released log carries measurement noise, as a real vessel's log would. The hidden set does
not: it is the clean trajectory, so full marks is that trajectory reproduced (E = 0).

`SIGMA` holds the standard deviations per channel; positions are also rounded to
`POSITION_QUANT`. Timing jitter is not modelled.
"""
import numpy as np

SIGMA = dict(position=0.02,                 # m, per axis
             heading=np.deg2rad(0.3),       # rad
             surge=0.02,                    # m/s
             sway=0.02,                     # m/s
             yawrate=np.deg2rad(0.5))       # rad/s
POSITION_QUANT = 0.01                       # m


def measure(traj, rng):
    """(N, T, 6) clean trajectory -> one independent noisy measurement of it.

    Heading is wrapped after perturbation, because the metric compares wrapped angles.
    """
    out = np.asarray(traj, float).copy()
    out[..., 0:2] += rng.normal(0.0, SIGMA['position'], out[..., 0:2].shape)
    out[..., 0:2] = np.round(out[..., 0:2] / POSITION_QUANT) * POSITION_QUANT
    out[..., 2] += rng.normal(0.0, SIGMA['heading'], out[..., 2].shape)
    out[..., 2] = np.arctan2(np.sin(out[..., 2]), np.cos(out[..., 2]))
    out[..., 3] += rng.normal(0.0, SIGMA['surge'], out[..., 3].shape)
    out[..., 4] += rng.normal(0.0, SIGMA['sway'], out[..., 4].shape)
    out[..., 5] += rng.normal(0.0, SIGMA['yawrate'], out[..., 5].shape)
    return out
