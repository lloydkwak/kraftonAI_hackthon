"""A hand-built straight reach for the lane rules' tests.

Built rather than generated so every number a rule compares against is exact: an 80 m wide
channel from x = 0 to x = 400 whose end caps are the box edges (as generated courses draw them),
the route on the centreline y = 70, and on request a pier gate at x = 200 whose 5 m aperture is
centred on y = 90 -- the line the downstream lane runs on. Piers stay inside the pier size
range, 8-20 m by 4-10 m, so a clean course reports nothing at all.
"""

from __future__ import annotations

import numpy as np

from usvnav import coursefile as cf
from usvnav.geometry import Rect
from usvnav.world import PIER, Course, Lane, StaticBody

Y_LANE = 90.0               # the line the gate is built around
APERTURE = (87.5, 92.5)     # the gate's opening in y: 5 m, centred on Y_LANE
GATE_X = 200.0
GATE_DEPTH = 10.0           # the piers' extent along the reach


def boundary():
    return np.array([[0.0, 30.0], [400.0, 30.0], [400.0, 110.0], [0.0, 110.0]])


def gate(dx=0.0):
    """Piers from y = 77.5 up to the bank at 110, leaving APERTURE open; the route at y = 70
    passes below. `dx` shifts the whole gate along x (its piers may then run past the world
    box's edge -- nothing clips them; the boundary is what it is)."""
    x = GATE_X + dx
    return [StaticBody(PIER, Rect(x, 82.5, GATE_DEPTH, 10.0, 0.0)),     # y 77.5 .. 87.5
            StaticBody(PIER, Rect(x, 96.5, GATE_DEPTH, 8.0, 0.0)),      # y 92.5 .. 100.5
            StaticBody(PIER, Rect(x, 105.25, GATE_DEPTH, 9.5, 0.0))]    # y 100.5 .. 110


def wall():
    """A wall across the whole channel except APERTURE, so the ego's only way is the lane's passage."""
    lower = [StaticBody(PIER, Rect(GATE_X, y, GATE_DEPTH, 10.0, 0.0)) for y in (35.0, 45.0, 55.0, 65.0, 75.0)]
    lower.append(StaticBody(PIER, Rect(GATE_X, 83.75, GATE_DEPTH, 7.5, 0.0)))   # y 80 .. 87.5
    return lower + gate()[1:]


def along(y=Y_LANE, *, speed=1.0, headway=60.0, offset=0.0, vessels=((12.0, 3.0),), x0=0.0, x1=400.0):
    """An open lane down the reach at height `y`, running +x at a constant speed."""
    return Lane(closed=False, points=np.array([[x0, y], [x1, y]]), speeds=np.array([speed, speed]),
                vessels=list(vessels), headway_s=headway, offset_s=offset)


def course(bodies=(), lanes=(), traffic=()):
    b = boundary()
    return Course(b, list(bodies), np.array([[160.0, 70.0], [240.0, 70.0], [325.3, 70.0]]),
                  np.array([5.0, 5.0, 3.0]), (74.7, 70.0, 0.0), list(traffic), list(lanes),
                  terrain_extent=cf.default_terrain_extent(b))


def rules(problems):
    return {p["rule"] for p in problems}


def berth_boundary(side=+1, x=300.0, width=5.0, depth=8.0):
    """lanefix's reach with a berth slip cut into one bank: `side = +1` notches the y = 110 bank
    (the slip runs from y = 110 up to y = 110 + depth), `-1` the y = 30 bank. Vertices stay in
    the fixture's (counter-clockwise) order."""
    b = boundary()
    hx = width / 2.0
    if side > 0:      # the y = 110 edge runs from (400, 110) to (0, 110): decreasing x
        notch = np.array([[x + hx, 110.0], [x + hx, 110.0 + depth], [x - hx, 110.0 + depth], [x - hx, 110.0]])
        return np.vstack([b[:3], notch, b[3:]])
    notch = np.array([[x - hx, 30.0], [x - hx, 30.0 - depth], [x + hx, 30.0 - depth], [x + hx, 30.0]])
    return np.vstack([b[:1], notch, b[1:]])


def berth_lane(x=300.0, side=+1, depth=8.0, *, speed=1.0, headway=60.0, offset=0.0, vessels=((12.0, 3.0),)):
    """An open lane from the left cap at y = 90 into the berth at `x`: level to 13 m before the
    berth, a 45° leg up to a point 7 m outside the mouth on the slip's axis, then the axis to the
    back wall's centre, where it ends. Every turn is 45° (L11-clean) and the last 15 m are the
    axis, so the spawn rectangle -- half a vessel long -- lies inside the slip."""
    y_mouth = 110.0 if side > 0 else 30.0
    pts = np.array([[0.0, 90.0], [x - 13.0, 90.0], [x, y_mouth - side * 7.0], [x, y_mouth], [x, y_mouth + side * depth]])
    return Lane(closed=False, points=pts, speeds=np.full(5, speed), vessels=list(vessels), headway_s=headway, offset_s=offset)
