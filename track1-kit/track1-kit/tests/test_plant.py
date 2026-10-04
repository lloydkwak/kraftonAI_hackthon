"""The plant and what the observation reports about it.

The `(v, w)` command is an autopilot's setpoint and the vessel is Track 2's public nominal model.
These checks pin what that model is meant to do in Track 1 -- reach and hold the commanded surge, slip in a turn
but not absurdly, shed speed in a hard turn at full speed, coast on a lower setpoint, reverse only when told -- and
two properties of the drift: the 1-4 drift is an added over-ground velocity that moves the hull and never turns
it, and the reported `u` is over ground. The plant integrates through-water velocities and adds the drift only to
the position.
"""

from __future__ import annotations

import math

import numpy as np

from usvnav import plant
from usvnav.plant import (DT, V_MAX, V_MIN, W_MAX, NoDisturbance, OUDisturbance, Vessel, sanitize)


def _drive(ticks, action, drift=(0.0, 0.0), boat=None):
    v = boat or Vessel(0.0, 0.0, 0.0)
    for _ in range(ticks):
        v.step(np.array(action, dtype=float), drift_world=drift)
    return v


def test_the_parameters_are_track_2s_public_nominal_model():
    """Both tracks sail one boat: Track 2's public nominal model and its thruster geometry."""
    assert (plant.MASS, plant.IZZ, plant.X_U, plant.Y_V, plant.N_R) == (250.0, 270.8, 200.0, 300.0, 900.0)
    assert (plant.ARM, plant.K_STERN, plant.K_BOW) == (1.2, 400.0, 150.0)
    assert plant.K_STERN / plant.X_U == V_MAX          # full stern thrust holds exactly the top speed


def test_straight_ahead_the_commanded_surge_is_reached_and_held_without_slip():
    v = _drive(300, (1.5, 0.0))
    assert abs(v.u - 1.5) < 1e-6 and abs(v.v) < 1e-12 and abs(v.r) < 1e-12
    assert abs(v.psi) < 1e-12 and abs(v.y) < 1e-9


def test_a_hard_turn_at_full_speed_slips_and_sheds_speed_within_bounds():
    v = _drive(200, (V_MAX, W_MAX))
    slip = math.degrees(math.atan2(abs(v.v), v.u))
    assert 1.4 < v.u < 1.8, v.u                        # the stern thruster is shared between surge and turning
    assert 0.4 < v.r < W_MAX, v.r                      # yaw authority falls with speed
    assert 5.0 < slip < 15.0, slip                     # the stern swings out, about ten degrees


def test_a_lower_setpoint_coasts_instead_of_braking_and_only_a_negative_command_reverses():
    v = _drive(100, (V_MAX, 0.0))
    x0 = v.x
    _drive(100, (0.0, 0.0), boat=v)
    coast = v.x - x0
    assert 2.0 < coast < 3.0, coast                    # u0 * mass / X_u = 2.5 m with no thrust
    assert v.u >= 0.0
    _drive(200, (V_MIN, 0.0), boat=v)
    assert abs(v.u - V_MIN) < 1e-3, v.u               # reverse is capped at 40 %, which still reaches V_MIN


def test_turning_on_the_spot_uses_the_bow_thruster():
    v = _drive(100, (0.0, W_MAX))
    assert abs(v.r - W_MAX) < 0.05, v.r
    assert abs(v.u) < 0.05


def test_a_head_drift_slows_the_boat_by_the_drift_and_no_more():
    """The drift is an added over-ground velocity. The surge the plant integrates is through the water."""
    v = _drive(600, (1.5, 0.0), drift=(-0.3, 0.0))
    assert abs(v.u - 1.2) < 1e-6, v.u
    assert abs(v.v) < 1e-9
    assert abs(v.u_tw - 1.5) < 1e-6


def test_a_cross_drift_is_reported_as_sway_exactly_and_never_as_yaw():
    v = _drive(600, (1.5, 0.0), drift=(0.0, 0.3))
    assert abs(v.v - 0.3) < 1e-9 and abs(v.u - 1.5) < 1e-6
    assert abs(v.r) < 1e-12 and abs(v.psi) < 1e-12     # a force at the centre of mass has no moment


def test_the_run_is_deterministic():
    a = _drive(400, (1.8, 0.4)); b = _drive(400, (1.8, 0.4))
    assert (a.x, a.y, a.psi, a.u, a.v, a.r) == (b.x, b.y, b.psi, b.u, b.v, b.r)


def test_the_disturbance_stays_inside_its_cap_and_is_seeded():
    d = OUDisturbance(seed=3)
    peak = max(float(np.linalg.norm(d.step())) for _ in range(20000))
    assert peak <= OUDisturbance.CAP + 1e-12
    a = [tuple(OUDisturbance(seed=5).step()) for _ in range(1)]
    b = [tuple(OUDisturbance(seed=5).step()) for _ in range(1)]
    assert a == b
    assert not NoDisturbance().step().any()


def test_sanitize_clips_finite_values_and_rejects_the_rest():
    assert list(sanitize([9.0, -9.0])) == [V_MAX, -W_MAX]
    assert list(sanitize([-9.0, 9.0])) == [V_MIN, W_MAX]
    for bad in ([float("nan"), 0.0], [1.0], [1.0, 2.0, 3.0], [float("inf"), 0.0]):
        try:
            sanitize(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} was accepted")
