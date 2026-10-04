"""The 2-2 dataset format, and its materialiser (`read`: the directory -> what the learner trains on).

A submission is a directory of action-labelled decision records, one file per episode:

    dataset/
      manifest.json          what is in here; read for validation, never as learner input
      episodes/
        ep_00000.npz         one episode, lossless (zlib), arrays below
        ...

Per-episode arrays, all indexed by decision `t` (one record per 0.5 s):

    rgb             (T, 128, 128, 3)  uint8    the top-view observation at decision t (0.5 m/px, bow up)
    ego_motion      (T, 3)            float32  measured surge, sway, yaw rate at t
    actuator_state  (T, 3)            float32  achieved [T_stern, delta, T_bow] at t
    d_target        (T,)              float32  the distance to keep in force at t, metres
    action          (T, 3)            float32  the command applied over [t, t+1)
    step            (T,)              int32    0..T-1
    timestamp       (T,)              float32  step * 0.5
    reset           (T,)              bool     step == 0

**The alignment rule is the whole contract**: `record[t]`'s observation determines
`record[t]`'s action, which is applied over the following interval. The materialiser trusts
nothing about that beyond what it can check (contiguous steps, consistent timestamps, one reset
per episode) -- whether the action really was the one applied is yours to get right. Nothing
checks it: a dataset whose actions are shifted by a step is accepted and scored like any other.

**Why one file per episode.** No observation window may cross an episode start, and the learner
builds its windows from `(episode, step)`. A per-episode file makes the boundary a fact of the
layout rather than a column that could be wrong. `npz` because it is lossless and needs nothing
beyond numpy. The manifest is JSON.
"""
import hashlib
import json
import os
import zipfile

import numpy as np
from numpy.lib import format as _npf

from . import path as guide, raster
from .closedloop import PER
from .samples import DT

FORMAT = 't2-2-dataset/1'
DECISION_S = PER * DT
ARRAYS = {
    'rgb': (np.uint8, 4), 'ego_motion': (np.float32, 2), 'actuator_state': (np.float32, 2),
    'd_target': (np.float32, 1), 'action': (np.float32, 2), 'step': (np.int32, 1),
    'timestamp': (np.float32, 1), 'reset': (np.bool_, 1),
}
#: What follows the leading `T` in each array's shape (the table above / SCHEMA.md). The validator
#: checks the whole shape against this, not the rank alone: a `(T, 1)` action has the right rank and
#: broadcasts silently through every bounds check.
SHAPES = {
    'rgb': (raster.SIZE_PX, raster.SIZE_PX, 3), 'ego_motion': (3,), 'actuator_state': (3,),
    'd_target': (), 'action': (3,), 'step': (), 'timestamp': (), 'reset': (),
}
assert all(len(SHAPES[k]) + 1 == nd for k, (_, nd) in ARRAYS.items())

#: Action bounds: signed stern thrust, stern azimuth within +/- pi/2, signed bow thrust. Defined here, beside
#: the format, so that the validator needs numpy only; the fixed learner imports them from this module.
ACTION_LO = np.array([-1.0, -np.pi / 2, -1.0], np.float32)
ACTION_HI = np.array([1.0, np.pi / 2, 1.0], np.float32)


def member_path(root, rel):
    """The path of a manifest member (an episode entry's `file`) under the dataset directory `root`.
    The manifest names files inside the submission and nothing else, so this refuses -- ValueError,
    with the reason -- an absolute path, any `..` component, an empty or `.` path, and a path whose
    real location (through symlinks) is not under `root`. Intake's copy is what gets read; a member
    that points outside it would read from the intake machine instead."""
    if not isinstance(rel, str) or not rel.strip():
        raise ValueError(f"member path {rel!r} is empty; expected a path like 'episodes/ep_00000.npz'")
    if os.path.isabs(rel) or rel.startswith(('/', '\\')):
        raise ValueError(f"member path {rel!r} is absolute; members are named relative to the dataset directory")
    parts = [q for q in rel.replace('\\', '/').split('/') if q not in ('', '.')]
    if not parts:
        raise ValueError(f"member path {rel!r} names the dataset directory itself, not a file in it")
    if '..' in parts:
        raise ValueError(f"member path {rel!r} leaves the dataset directory ('..')")
    full = os.path.join(root, rel)
    real_root = os.path.realpath(root)
    if not os.path.realpath(full).startswith(real_root.rstrip(os.sep) + os.sep):
        raise ValueError(f"member path {rel!r} resolves outside the dataset directory (a symlink?)")
    return full


def episode_from_records(rec):
    """A `closedloop.run_episode(..., record=True)['records']` dict -> one episode's arrays."""
    T = len(rec['action'])
    step = np.arange(T, dtype=np.int32)
    return dict(rgb=np.asarray(rec['rgb'], np.uint8),
                ego_motion=np.asarray(rec['ego_motion'], np.float32),
                actuator_state=np.asarray(rec['actuator_state'], np.float32),
                d_target=np.asarray(rec.get('d_target', np.full(T, guide.D_TARGET)), np.float32),
                action=np.asarray(rec['action'], np.float32),
                step=step, timestamp=(step * DECISION_S).astype(np.float32),
                reset=(step == 0))


def write(path, episodes, meta=None, method=None):
    """Write `episodes` (list of array dicts as `episode_from_records` makes) under `path`. `method`, if
    given, is written to `method.md` beside the manifest for your own notes; it is not part of the format."""
    os.makedirs(os.path.join(path, 'episodes'), exist_ok=True)
    entries = []
    for i, ep in enumerate(episodes):
        name = f"ep_{i:05d}.npz"
        np.savez_compressed(os.path.join(path, 'episodes', name), **ep)
        entries.append(dict(file=f"episodes/{name}", n=int(len(ep['action']))))
    manifest = dict(format=FORMAT, n_episodes=len(episodes),
                    n_records=int(sum(e['n'] for e in entries)),
                    decision_s=DECISION_S, raster=dict(size_px=raster.SIZE_PX, m_per_px=raster.M_PER_PX),
                    renderer=raster.renderer_id(), episodes=entries)
    manifest.update(meta or {})
    json.dump(manifest, open(os.path.join(path, 'manifest.json'), 'w'), indent=1)
    if method is not None:
        with open(os.path.join(path, 'method.md'), 'w') as f:
            f.write(method)
    return manifest


def read_manifest(path):
    return json.load(open(os.path.join(path, 'manifest.json')))


#: Cap on what one episode file *declares* as its uncompressed size, summed over the archive's
#: members and read from the zip directory before anything is decompressed. The whole budget at
#: 128 px is 885 MB of frames, so no honest episode file comes near 1 GiB; an archive that
#: claims more is a decompression bomb and is refused unopened.
MAX_EPISODE_BYTES = 1 << 30


def inspect_episode(path, entry):
    """What an episode file declares, without decompressing any array: `{name: (shape, dtype)}`
    from the `.npy` headers, and the declared uncompressed byte total from the zip directory.
    Intake reads this first and refuses a file whose declaration is already out of bounds."""
    out, total = {}, 0
    with zipfile.ZipFile(member_path(path, entry['file'])) as z:
        for info in z.infolist():
            total += info.file_size
            name = info.filename[:-4] if info.filename.endswith('.npy') else info.filename
            with z.open(info) as fh:
                version = _npf.read_magic(fh)
                header = _npf.read_array_header_1_0 if version == (1, 0) else _npf.read_array_header_2_0
                shape, _, dtype = header(fh)
            out[name] = (tuple(shape), dtype)
    return out, total


def read_episode(path, entry):
    """The format's arrays and nothing else: pickled objects are refused, members outside
    `ARRAYS` are never decompressed."""
    z = np.load(member_path(path, entry['file']), allow_pickle=False)
    return {k: z[k] for k in z.files if k in ARRAYS}


def read(path):
    """The materialiser: the directory -> the dict the learner consumes, plus the manifest.

    Episodes are numbered in manifest order; `step` is taken from the file. The learner's
    `Dataset` derives its windows from these two columns and nothing else."""
    man = read_manifest(path)
    parts = []
    for i, e in enumerate(man['episodes']):
        ep = read_episode(path, e)
        ep = dict(ep, episode=np.full(len(ep['action']), i, np.int32))
        parts.append(ep)
    keys = ('rgb', 'ego_motion', 'actuator_state', 'd_target', 'action', 'step', 'episode')
    data = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    return data, man


def frame_hash(frame):
    return hashlib.blake2b(np.ascontiguousarray(frame).tobytes(), digest_size=8).hexdigest()


def thumb_hash(frame):
    """A near-duplicate key: the frame averaged to 16 x 16 and quantised to 16 levels."""
    f = np.asarray(frame, np.float32).reshape(16, 8, 16, 8, 3).mean(axis=(1, 3))
    return hashlib.blake2b((f // 16).astype(np.uint8).tobytes(), digest_size=8).hexdigest()

