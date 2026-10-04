"""Dependency-free test runner: `python tests/run.py [name-substring ...]`.

pytest is not in the kit and these checks ship with it, so the runner is 30 lines rather
than a dependency.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import time
import traceback

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))          # `pilot.py`, the shared fixture


def _load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv):
    wanted = argv[1:]
    files = sorted(HERE.glob("test_*.py")) + sorted(HERE.glob("internal/test_*.py"))
    passed = failed = 0
    for path in files:
        mod = _load(path)
        for name in sorted(n for n in dir(mod) if n.startswith("test_")):
            if wanted and not any(w in f"{path.stem}.{name}" for w in wanted):
                continue
            t0 = time.perf_counter()
            try:
                getattr(mod, name)()
            except Exception:
                failed += 1
                print(f"FAIL  {path.stem}.{name}")
                traceback.print_exc()
            else:
                passed += 1
                print(f"ok    {path.stem}.{name}  ({1000*(time.perf_counter()-t0):.0f} ms)")
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
