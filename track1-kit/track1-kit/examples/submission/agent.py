"""The worked example submission's agent: the whole interface, with the perception field
of every condition read once so you can see where your own policy would start.

The interface, which is all the scorer calls:

    Agent(config_path)     once per process; `config_path` is the absolute path of the
                           file `submission.json` names, inside this directory
    reset(meta)            once per episode; clear everything
    act(obs) -> [v, w]     every tick: surge setpoint in m/s, yaw-rate setpoint in rad/s

`obs` is a dict of numpy arrays -- six common fields plus `obs["perception"]`, whose type
depends on the condition `meta["observation_mode"]` announced at reset. The exact shapes,
dtypes and conventions are in `docs/OBSERVATION.md` of the starter kit; the values used
below are the ones a first agent needs.

The policy is tutorial-grade on purpose: pure pursuit toward the current waypoint, slower
when something is close ahead. It never turns away from an obstacle.
"""

import json
import math

import numpy as np

V_MAX, W_MAX = 2.0, 0.6            # the action ranges' upper ends; anything outside is clipped


class Agent:
    def __init__(self, config_path=None):
        cfg = {}
        if config_path:
            with open(config_path) as f:
                cfg = json.load(f)
        self.cruise = V_MAX * float(cfg.get("cruise_fraction", 0.75))
        self.gain = float(cfg.get("steer_gain", 1.4))
        self.caution_m = float(cfg.get("caution_range_m", 12.0))
        self.mode = None

    def reset(self, meta):
        # meta: boundary polygon, the full waypoint chain, arrival radii, hull size
        # (a constant 3.0 x 2.0 m), observation_mode. No obstacle geometry -- that is
        # what the perception field is for.
        self.mode = meta["observation_mode"]

    def act(self, obs):
        dist, bearing = (float(v) for v in obs["wp_polar"])      # bearing > 0 is to port
        w = max(-W_MAX, min(W_MAX, self.gain * bearing))

        ahead = self._clear_ahead_m(obs["perception"])
        v = self.cruise
        if ahead < self.caution_m:
            v = max(0.3, self.cruise * ahead / self.caution_m)
        v = min(v, max(0.4, 0.5 * dist))                          # ease into the waypoint
        return np.array([v, w], dtype=np.float32)

    # ---- reading each condition's perception field --------------------------------

    def _clear_ahead_m(self, perception) -> float:
        """How far the water directly ahead looks clear, in metres, from whichever
        sensor this episode has. Returns a large number when nothing is seen."""
        if self.mode == "1-3":
            # 180 ranges over 360 degrees, index 0 at the bow increasing to port, 2 degrees
            # apart, 50 m maximum. No return is 50.0, never 0.
            r = perception
            fan = np.concatenate([r[:8], r[-8:]])                # +/- 16 degrees about the bow
            return float(fan.min())
        if self.mode == "1-1":
            # A fixed-length list with a validity mask; positions are in the body frame,
            # +x through the bow, +y to port; extent is (length, width).
            valid = perception["valid"]
            pos, ext = perception["pos"][valid], perception["extent"][valid]
            ahead = (pos[:, 0] > 0) & (np.abs(pos[:, 1]) < 0.5 * ext.max(axis=1) + 2.0)
            return float(pos[ahead, 0].min()) if ahead.any() else 100.0
        if self.mode in ("1-2", "1-4"):
            # A 200 x 200 x 3 uint8 picture, 0.5 m/px, bow toward row 0, the hull centred
            # on the corner shared by the four central pixels. It is the studio's picture
            # (docs/OBSERVATION.md 5.2): flat base colours inside a class, plus shoreline,
            # shadow, outline and anti-aliased edge tones around them -- so classify by
            # *closeness* to a few known tones, not by exact match. Water and the tones
            # water can wear (the shallow band near a bank, a body's shadow) are free; the
            # hull and its own outline ring are ours; everything else ahead is an obstacle.
            img = perception
            column = img[:100, 98:102, :].astype(np.int16)       # the strip ahead of the bow
            def near(tone, tol=40):
                return np.abs(column - np.array(tone, dtype=np.int16)).max(axis=-1) <= tol
            free = (near((0x16, 0x30, 0x3B)) | near((0x1E, 0x3B, 0x42)) | near((0x0E, 0x1E, 0x25))
                    | near((0xF2, 0x6E, 0x7A)) | near((0x40, 0x18, 0x1C)))
            rows = np.nonzero((~free).any(axis=1))[0]
            if len(rows) == 0:
                return 100.0
            nearest_row = rows.max()                              # the largest row is nearest
            return float((100 - nearest_row) * 0.5)
        return 100.0
