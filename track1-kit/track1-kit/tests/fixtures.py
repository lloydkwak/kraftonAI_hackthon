"""Courses for the shipped tests, from files rather than from the generator.

The generator is not in the kit, so a shipped test cannot call it. What a participant's copy
does hold is every course file in the kit: the practice set, the schema document's example
course, and the validator's smoke-run course. Those are the fixtures.
"""

from __future__ import annotations

import atexit
import os
import pathlib
import shutil
import stat
import tempfile

from usvnav import coursefile
from usvnav.contest import write_set

ROOT = pathlib.Path(__file__).resolve().parent.parent

COURSE_FILES = (sorted((ROOT / "sets" / "practice" / "courses").glob("*.json"))
                + [ROOT / "docs" / "examples" / "course.json", ROOT / "tools" / "demo-course.json"])


def scratch(prefix: str = "usvnav-") -> pathlib.Path:
    """A fresh temporary directory, removed when the interpreter exits.

    `tempfile.mkdtemp` alone leaves every directory behind, and some tests write large
    files (the size-cap test writes a weights file just over the size cap), so each suite
    run would leave them in the temp directory. Removal is at exit rather than per test so
    a failing test's directory can still be inspected while the run is going.
    """
    d = pathlib.Path(tempfile.mkdtemp(prefix=prefix))
    atexit.register(remove_tree, d)
    return d


def remove_tree(path) -> None:
    """`rmtree` that first makes everything writable: a board's intake copy is read-only,
    and `rmtree(ignore_errors=True)` would leave it behind in the temp directory."""
    path = pathlib.Path(path)
    if not path.exists():
        return
    for dirpath, dirnames, filenames in os.walk(path):
        try:
            os.chmod(dirpath, stat.S_IRWXU)
        except OSError:
            pass
        for f in filenames:
            try:
                os.chmod(pathlib.Path(dirpath) / f, stat.S_IRUSR | stat.S_IWUSR)
            except OSError:
                pass
    shutil.rmtree(path, ignore_errors=True)


def courses():
    """Every fixture course, loaded fresh (a scored course is the file)."""
    return [coursefile.load(p) for p in COURSE_FILES if p.is_file()]


def course(i: int = 0):
    return courses()[i]


def course_with_traffic():
    """The first fixture course with moving vessels -- on a lane or on a legacy closed loop.
    `Course.moving(t)` is the one path every consumer takes to either."""
    for c in courses():
        if c.traffic or c.lanes:
            return c
    raise AssertionError("no fixture course has moving vessels")


def easy_course(n_waypoints: int = 4, *, lanes=(), bodies=()):
    """A hand-built straight reach (`lanefix`: 80 m wide, 400 m long, route on the centreline) with
    `n_waypoints` waypoints spaced down the reach -- a course the shipped tutorial agent completes.

    The shipped tests that need a *completed* run (the runner's records, the studio's end-of-run
    checks) build on this rather than on the practice courses: the practice courses are not easy
    -- rocks on the chord, lanes, out-and-back -- and the tutorial agent, which never turns away
    from an obstacle, fails them, as RULES.md says it usually does. A fixture that a test needs to
    be easy must be built to be easy.
    """
    import lanefix
    import numpy as np
    from usvnav.rules import RADIUS_GOAL, RADIUS_INTERMEDIATE
    from usvnav.world import Course
    n = max(2, int(n_waypoints))
    xs = np.linspace(130.0, 350.0, n)
    b = lanefix.boundary()
    # The start sits 8 m off the centreline (heading along the reach), so a pure-pursuit pilot is
    # correcting -- unsaturated, and by a little less every tick -- all through the first leg. The
    # runner's tests need that: an agent adding 1e-3 of noise must differ from tick 0 (a clipped
    # yaw command would swallow the noise).
    return Course(b, list(bodies), np.array([[x, 70.0] for x in xs]),
                  np.array([RADIUS_INTERMEDIATE] * (n - 1) + [RADIUS_GOAL]), (60.0, 62.0, 0.0), [], list(lanes),
                  terrain_extent=coursefile.default_terrain_extent(b))


def write_easy_set(out_dir, *, n: int = 1, conditions=("1-1",), prefix: str = "fx",
                   kind: str = "public", seed0: int = 1000) -> dict:
    """A small set of easy courses, ids `<prefix>-NN`: the first has four waypoints, the next three
    (tests tell them apart by `len(meta["waypoints"])`)."""
    eps = [(f"{prefix}-{i + 1:02d}", easy_course(4 if i == 0 else 3), seed0 + i, conditions) for i in range(n)]
    return write_set(out_dir, eps, kind=kind)


def write_fixture_set(out_dir, *, n: int = 2, conditions=("1-1",), prefix: str = "fx",
                      kind: str = "public", seed0: int = 1000) -> dict:
    """A small set from the first `n` fixture courses, ids `<prefix>-NN`."""
    eps = [(f"{prefix}-{i + 1:02d}", c, seed0 + i, conditions)
           for i, c in enumerate(courses()[:n])]
    return write_set(out_dir, eps, kind=kind)
