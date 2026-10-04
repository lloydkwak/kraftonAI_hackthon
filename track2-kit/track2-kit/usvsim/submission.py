"""Loading a submission: the layout check and the loader.

The same code the contest scorer runs, shipped so that "it passed locally" means the same
thing as "it will be accepted".
"""
import importlib.util
import os
import sys


SUBMISSION_FORMAT = 't2-1-submission/1'


def resolve(path):
    """A submission is a `t2-1-submission/1` directory (PACKAGING.md) or, for local use, a bare
    `.py`. Returns `(entry_module_path, manifest_or_None)`; raises ValueError on a bad layout."""
    import json
    if os.path.isdir(path):
        mpath = os.path.join(path, 'manifest.json')
        if not os.path.exists(mpath):
            raise ValueError(f"{path}: manifest.json missing (PACKAGING.md)")
        man = json.load(open(mpath))
        if man.get('format') != SUBMISSION_FORMAT:
            raise ValueError(f"{path}: format is {man.get('format')!r}, expected {SUBMISSION_FORMAT!r}")
        if not isinstance(man.get('entry'), str) or not man['entry'].strip():
            raise ValueError(f"{path}: manifest.json needs a non-empty 'entry'")
        entry = os.path.normpath(os.path.join(path, man['entry']))
        if not entry.startswith(os.path.abspath(path) + os.sep) and not entry.startswith(path.rstrip('/') + os.sep):
            raise ValueError(f"{path}: entry {man['entry']!r} points outside the submission")
        if not os.path.isfile(entry) or not entry.endswith('.py'):
            raise ValueError(f"{path}: entry {man['entry']!r} is not a Python file in the submission")
        return entry, man
    if os.path.isfile(path) and path.endswith('.py'):
        return path, None
    raise ValueError(f"{path}: neither a submission directory nor a .py file")


def load(path, name=None):
    path, _ = resolve(path)
    d = os.path.dirname(os.path.abspath(path))
    if d not in sys.path:                      # helper modules beside the entry import by name
        sys.path.insert(0, d)
    name = name or f"_submission_{abs(hash(os.path.abspath(path)))}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"not an importable Python module: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, 'make_sim'):
        raise ValueError(f"{path} defines no make_sim(config, seed)")
    return mod
