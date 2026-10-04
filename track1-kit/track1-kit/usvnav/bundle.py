"""What ships in the participant kit, as an allow-list (organiser-side).

**An allow-list.** `SHIPPED` names every path the kit contains; `INTERNAL` names every path
it must not. `audit()` walks the tree and fails if a file is in neither, so adding a module
is a decision about which side it belongs on rather than a default.

**An import check.** `audit()` also parses every shipped module and reports any import that
reaches the internal side, so the kit cannot depend on a module it does not contain.

**A build.** `build(dest)` copies the allow-list and stamps it (`VERSION.txt`).
"""

from __future__ import annotations

import ast
import datetime
import pathlib
import shutil
import subprocess

from . import __version__

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: Written into every build by `build()`: the kit's identity, to quote when writing to the
#: organisers.
STAMP = "VERSION.txt"

#: Everything the kit contains. Directories are taken whole, minus `INTERNAL`.
#: `submit.sh` is the participant's submission script (validate, zip, upload).
SHIPPED = (
    "pyproject.toml",
    "README.md",
    "submit.sh",
    "usvnav/",
    "tests/",
    "tools/editor.html",
    "tools/studio.html",
    "tools/demo-course.json",
    "examples/",
    "docs/",
    "sets/",
)

#: Everything that does not ship, each with a short reason.
INTERNAL = {
    "usvnav/internal/": "organiser-only agents and tools",
    "usvnav/generate.py": "the course generator (organiser tool)",
    "usvnav/scenario.py": "part of the course generator",
    "usvnav/channel.py": "part of the course generator",
    "usvnav/lanegen.py": "part of the course generator",
    "usvnav/accel.py": "organiser-side acceleration of the scorer's simulator; the kit runs the Python it "
                       "ships, and the scorer checks the kernels draw the same bytes before using them",
    "usvnav/_native": "the built native kernels behind usvnav/accel.py",
    "accel/": "the source of the native kernels",
    "tools/accel_sweep.py": "the byte-identity check for the native kernels",
    "INTERNAL.md": "organiser notes",
    "PATCHNOTES.md": "published on the site as the patch notes page",
    "tests/internal/": "tests of the organiser-only parts",
    "tools/x1.py": "organiser tool",
    "tools/x1_summary.py": "organiser tool",
    "tools/x2.py": "organiser tool",
    "tools/x2_types.py": "organiser tool",
    "tools/x3.py": "organiser tool",
    "tools/x4.py": "organiser tool",
    "tools/x5.py": "organiser tool",
    "tools/signoff.py": "organiser tool",
    "tools/figures.py": "organiser tool",
    "results/": "organiser measurements",
    "dist/": "the built kit itself",
    ".venv/": "",
    ".git/": "",
}

#: Build products, which are neither shipped nor withheld -- they are regenerated.
#: Deliberately short: data files have to be on one side of the line on purpose, so only
#: generated output is exempt.
_IGNORED = (".pyc",)

#: Directories a tool regenerates (`pip install -e .` writes `usvnav.egg-info/`). The general
#: rule is `_git_ignored()`: whatever `.gitignore` names is a build product and is skipped;
#: this tuple is the fallback for a tree without git.
_GENERATED = ("usvnav.egg-info/",)


def _git_ignored(root: pathlib.Path) -> frozenset[str]:
    """The files git ignores under `root` (relative, posix), or an empty set without git.
    Those are build products; the audit reads `.gitignore` instead of keeping its own list.
    """
    try:
        out = subprocess.run(["git", "-C", str(root), "ls-files", "--others", "--ignored",
                              "--exclude-standard", "-z"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return frozenset()
    if out.returncode != 0:
        return frozenset()
    return frozenset(p for p in out.stdout.split("\0") if p)


def _is_internal(rel: str) -> str | None:
    for prefix, why in INTERNAL.items():
        if rel == prefix or rel.startswith(prefix):
            return why
    return None


def _is_shipped(rel: str) -> bool:
    if _is_internal(rel):
        return False
    return any(rel == s or (s.endswith("/") and rel.startswith(s)) for s in SHIPPED)


def shipped_files(root: pathlib.Path = ROOT):
    """Every file the kit contains, in sorted order. Git-ignored files never ship."""
    ignored = _git_ignored(root)
    out = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if _is_shipped(rel) and not rel.endswith(".pyc") and rel not in ignored:
            out.append(rel)
    return out


def _imports(path: pathlib.Path):
    """Every module name a file imports, including inside functions (hence `ast` rather
    than a regex).
    """
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return []
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:                      # relative: resolve against the package
                pkg = path.relative_to(ROOT).parent.as_posix().replace("/", ".")
                parts = pkg.split(".")
                base = ".".join(parts[:len(parts) - node.level + 1])
                names.append(f"{base}.{node.module}" if node.module else base)
            else:
                names.append(node.module or "")
    return names


def _module_path(name: str) -> str | None:
    """Where a dotted module name would live in this tree, if it lives here at all."""
    parts = name.split(".")
    if parts[0] not in ("usvnav", "tools", "tests"):
        return None
    stem = "/".join(parts)
    for candidate in (f"{stem}.py", f"{stem}/__init__.py"):
        if (ROOT / candidate).exists():
            return candidate
    return None


def audit(root: pathlib.Path = ROOT):
    """Return `(unclassified, leaks)`; both empty is the only acceptable result.

    A leak is any shipped module importing something in this tree that does not ship.
    """
    unclassified, leaks = [], []
    ignored = _git_ignored(root)
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if (rel.startswith(".") or rel.endswith(_IGNORED)
                or rel.startswith(_GENERATED) or rel in ignored):
            continue
        if _is_internal(rel):
            continue
        if not _is_shipped(rel):
            unclassified.append(rel)
            continue
        if path.suffix == ".py":
            for name in _imports(path):
                target = _module_path(name)
                if target is not None and not _is_shipped(target):
                    leaks.append((rel, name))
    return unclassified, leaks


def _git(*args: str) -> str:
    """One git query against the tree, or "" when there is no git or no repository."""
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def stamp_text(now: datetime.datetime | None = None) -> str:
    """The contents of `VERSION.txt`: version, commit, build time, and what to do with them."""
    commit = _git("rev-parse", "--short=12", "HEAD") or "unknown"
    dirty = " +uncommitted" if _git("status", "--porcelain", "--untracked-files=no") else ""
    when = (now or datetime.datetime.now(datetime.timezone.utc).astimezone()).isoformat(timespec="seconds")
    return (f"usvnav {__version__}\n"
            f"commit {commit}{dirty}\n"
            f"built {when}\n"
            "\n"
            "This is the build of the Track 1 kit you have. Quote the three lines above when you\n"
            "write to the organisers; the kit's zip checksum is published beside the download.\n")


def build(dest: pathlib.Path) -> pathlib.Path:
    """Copy the allow-list into `dest`, which is emptied first, and stamp it (`VERSION.txt`)."""
    dest = pathlib.Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    for rel in shipped_files():
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, target)
    (dest / STAMP).write_text(stamp_text(), encoding="utf-8")
    return dest


def cmd_bundle(args):
    unclassified, leaks = audit()
    for rel in unclassified:
        print(f"  [unfiled ] {rel}")
    for rel, name in leaks:
        print(f"  [leak    ] {rel} imports {name}")
    if unclassified or leaks:
        print("\nEvery file has to be on one side of the line and no shipped file may import")
        print("the internal side. Add it to SHIPPED or to INTERNAL in usvnav/bundle.py.")
        return 1
    files = shipped_files()
    print(f"{len(files)} files ship, {len(INTERNAL)} paths held back:")
    for prefix, why in sorted(INTERNAL.items()):
        if why:
            print(f"  {prefix:<22} {why}")
    if args.dest:
        out = build(args.dest)
        n = sum(1 for _ in out.rglob("*") if _.is_file())
        print(f"\nwrote {out}  ({n} files)")
    return 0
