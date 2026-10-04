"""The 2-1 replay error, exactly as the contest computes it.

    E   = mean over the four channels of (mean over time of the capped, normalised error)
    E_g = mean of E over the replays in maneuver group g
    skill_g = clip(1 - E_g / E_base_g, 0, 1)
    S_sim   = 100 x the mean of skill_g over the six groups

Channels, and what one unit of each means:

    position   1.0 m
    heading    0.2 rad
    surge      0.3 m/s
    yaw rate   0.15 rad/s

Every per-sample contribution is capped at 3. Two consequences worth understanding before
you optimise against this:

  * Sway is not a channel. It is still scored, indirectly and unavoidably -- position is
    the integral of body velocity rotated into the world frame, so a sway error shows up
    as a position error.
  * The cap is why open-loop replay is scorable at all. There is no corrective controller
    in the loop, so a modelling error does not stay where it started: the trajectory it
    produces feeds back into the next step. Without a cap, `E` would mostly measure *when*
    your model parted company with the reference rather than how good the model is. Past
    3 m of position error, or 34 degrees of heading, the metric stops counting how much
    further. A sample that is not finite counts as the cap.
"""
import numpy as np

NORM = dict(position=1.0, heading=0.2, surge=0.3, yawrate=0.15)
CLIP = 3.0
DT = 0.05                 # command interval, seconds
LOG_EVERY = 2             # state logged every second tick -> 10 Hz
RELEASED_SECS = 60.0      # a released replay
HIDDEN_SECS = 120.0       # a hidden-set replay


def _bounded(v, norm, clip):
    z = v / norm
    return np.clip(np.where(np.isfinite(z), z, clip), 0, clip)


def error(traj, reference):
    """Per-replay error. Both are (N, T, 6) stacks; returns (N,)."""
    traj = np.asarray(traj, float)
    reference = np.asarray(reference, float)
    with np.errstate(invalid='ignore', over='ignore'):
        dpsi = traj[:, :, 2] - reference[:, :, 2]
        ch = dict(
            position=np.linalg.norm(traj[:, :, :2] - reference[:, :, :2], axis=2),
            heading=np.abs(np.arctan2(np.sin(dpsi), np.cos(dpsi))),
            surge=np.abs(traj[:, :, 3] - reference[:, :, 3]),
            yawrate=np.abs(traj[:, :, 5] - reference[:, :, 5]))
        terms = [_bounded(v, NORM[k], CLIP).mean(axis=1) for k, v in ch.items()]
    return sum(terms) / len(terms)


def skill(e_group, e_base_group):
    """100 % is the true trajectory reproduced (E = 0), 0 is no better than the shipped simulator."""
    return float(np.clip(1.0 - e_group / e_base_group, 0, 1))


def s_sim(skills):
    return 100.0 * float(np.mean(list(skills.values())))
