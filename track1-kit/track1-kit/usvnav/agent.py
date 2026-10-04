"""The agent interface, and the one agent that ships.

`TutorialAgent` demonstrates the three methods and usually fails to complete a course: it
steers at the current waypoint and ignores everything else. No competent agent is
distributed.

The interface, in full:

    class MyAgent:
        def __init__(self, config_path=None): ...   # a path inside your submission
        def reset(self, meta): ...                  # once per episode
        def act(self, obs) -> [v_cmd, w_cmd]: ...   # every tick

No state may carry from one episode to the next; `reset` is where it is cleared.
"""

from __future__ import annotations

import numpy as np

from .plant import V_CRUISE, W_MAX


class TutorialAgent:
    """Minimal interface demonstration. Steers at the current waypoint, ignores everything,
    and so usually ends in a collision."""

    def __init__(self, config_path=None):
        self.config_path = config_path

    def reset(self, meta):
        self.meta = meta

    def act(self, obs):
        _, bearing = obs["wp_polar"]
        return np.array([V_CRUISE * 0.6, float(np.clip(bearing, -W_MAX, W_MAX))])
