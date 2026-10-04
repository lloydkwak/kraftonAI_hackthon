"""The 2-2 dataset validator: the checks intake runs on a dataset directory, shipped so you can
run them first (`usvsim validate-dataset DIR`). It needs numpy only.

Errors fail intake. Warnings are for you. Coverage checks catch *extreme skew only* -- a dataset
that cannot possibly work -- and leave "will it work" to the score. So a straight-line-only
dataset gets a warning and a score, not a rejection.

    archive     (before decompression) the members' declared sizes under the per-file cap, no
                object dtype, no single file declaring more records than the whole budget
    format      manifest, member paths inside the directory, arrays, dtypes, shapes, contiguous
                steps, one reset per episode, timestamps at the decision period, finite values;
                unknown arrays are ignored
    ranges      actions inside the action bounds, d_target inside [D_MIN, D_MAX], raster 128 px
    budget      n_records <= 18,000; duplicates count against it
    duplicates  exact and near-duplicate frames, reported; near-duplicates > 30% warn
    coverage    yaw-rate, actuator-state and action spread; a d_target that never varies
    leakage     (intake only) any frame byte-identical to a hidden-set frame is an error

The leakage check needs the hashes of the hidden-set frames, which the organisers pass in at
intake (`hidden_hashes`). A participant running this has none and gets every other check.
"""
import os

import numpy as np

from . import dataset as F, path as guide
from .dataset import ACTION_HI, ACTION_LO

BUDGET = 18000                      # decision records
NEAR_DUP_WARN = 0.30
MIN_TURNING_FRAC = 0.05             # records with |yaw rate| > 0.05 rad/s
MIN_AZIMUTH_STD = 0.05              # rad, achieved stern azimuth across the set
MIN_ACTION_STD = np.array([0.02, 0.02, 0.02])


class Report:
    def __init__(self):
        self.errors, self.warnings, self.stats = [], [], {}

    def error(self, m):
        self.errors.append(m)

    def warn(self, m):
        self.warnings.append(m)

    @property
    def ok(self):
        return not self.errors

    def text(self):
        out = [f"{'PASS' if self.ok else 'FAIL'}: {len(self.errors)} error(s), {len(self.warnings)} warning(s)"]
        out += [f"  ERROR    {m}" for m in self.errors]
        out += [f"  WARNING  {m}" for m in self.warnings]
        out += [f"  {k:22s} {v}" for k, v in self.stats.items()]
        return "\n".join(out)


def validate(path, hidden_hashes=None, budget=BUDGET):
    r = Report()
    mpath = os.path.join(path, 'manifest.json')
    if not os.path.exists(mpath):
        r.error("manifest.json missing")
        return r
    try:
        man = F.read_manifest(path)
    except Exception as e:                                   # noqa: BLE001
        r.error(f"manifest.json unreadable: {e}")
        return r
    if man.get('format') != F.FORMAT:
        r.error(f"format is {man.get('format')!r}, expected {F.FORMAT!r}")
    eps = man.get('episodes') or []
    if not eps:
        r.error("manifest lists no episodes")
        return r

    n_total, dups, near, n_turn = 0, 0, {}, 0
    seen = set()
    az, acts, dts, resets = [], [], [], 0
    leaked = 0
    extra_arrays, n_extra = set(), 0
    for i, e in enumerate(eps):
        try:
            f = F.member_path(path, e.get('file', ''))
        except ValueError as ex:
            r.error(f"episode {i}: {ex}")
            continue
        if not os.path.exists(f):
            r.error(f"episode {i}: file {e.get('file')!r} missing")
            continue
        # -- what the file declares, before any array is decompressed ---------------------------
        try:
            declared, nbytes = F.inspect_episode(path, e)
        except Exception as ex:                              # noqa: BLE001
            r.error(f"episode {i}: not a readable npz archive ({ex})")
            continue
        extra = set(declared) - set(F.ARRAYS)
        if extra:
            extra_arrays |= extra
            n_extra += 1
        if nbytes > F.MAX_EPISODE_BYTES:
            r.error(f"episode {i}: declares {nbytes / 2 ** 20:.0f} MiB uncompressed, over the "
                    f"{F.MAX_EPISODE_BYTES >> 20} MiB per-file cap; not loaded")
            continue
        if any(dt.kind == 'O' for _, dt in declared.values()):
            r.error(f"episode {i}: object (pickled) arrays are not allowed; not loaded")
            continue
        n_decl = max((shp[0] for k, (shp, _) in declared.items() if k in F.ARRAYS and shp), default=0)
        if n_decl > budget:
            r.error(f"episode {i}: declares {n_decl} records, over the budget of {budget} by itself; "
                    f"not loaded")
            continue
        try:
            ep = F.read_episode(path, e)
        except Exception as ex:                              # noqa: BLE001
            r.error(f"episode {i}: unreadable ({ex})")
            continue
        bad = False
        for k, (dt, nd) in F.ARRAYS.items():
            if k not in ep:
                r.error(f"episode {i}: array {k!r} missing")
                bad = True
                continue
            if ep[k].dtype != dt:
                r.error(f"episode {i}: {k} is {ep[k].dtype}, expected {np.dtype(dt)}")
                bad = True
            if ep[k].ndim != nd or ep[k].shape[1:] != F.SHAPES[k]:
                want = "(T, " + ", ".join(map(str, F.SHAPES[k])) + ")" if F.SHAPES[k] else "(T,)"
                r.error(f"episode {i}: {k} has shape {tuple(ep[k].shape)}, expected {want}")
                bad = True
        if bad:
            continue
        T = len(ep['action'])
        if T < 1:
            r.error(f"episode {i}: empty")
            continue
        if any(len(ep[k]) != T for k in F.ARRAYS):
            r.error(f"episode {i}: arrays disagree on length")
            continue
        if not np.array_equal(ep['step'], np.arange(T)):
            r.error(f"episode {i}: step is not contiguous 0..{T - 1}")
        if not np.allclose(ep['timestamp'], ep['step'] * F.DECISION_S, atol=1e-3):
            r.error(f"episode {i}: timestamp is not step x {F.DECISION_S}")
        if not (ep['reset'][0] and not ep['reset'][1:].any()):
            r.error(f"episode {i}: reset must be true at step 0 only")
        for k in ('ego_motion', 'actuator_state', 'd_target', 'action'):
            if not np.all(np.isfinite(ep[k])):
                r.error(f"episode {i}: {k} has non-finite values")
        if np.any(ep['action'] < ACTION_LO - 1e-6) or np.any(ep['action'] > ACTION_HI + 1e-6):
            r.error(f"episode {i}: action outside bounds [{ACTION_LO.tolist()}, {ACTION_HI.tolist()}]")
        if np.any(ep['d_target'] < guide.D_MIN - 1e-6) or np.any(ep['d_target'] > guide.D_MAX + 1e-6):
            r.error(f"episode {i}: d_target outside [{guide.D_MIN}, {guide.D_MAX}] m")
        if e.get('n') != T:
            r.warn(f"episode {i}: manifest says n={e.get('n')}, file has {T}")
        n_total += T
        resets += int(ep['reset'].sum())
        n_turn += int((np.abs(ep['ego_motion'][:, 2]) > 0.05).sum())
        az.append(ep['actuator_state'][:, 1])
        acts.append(ep['action'])
        dts.append(ep['d_target'])
        for fr in ep['rgb']:
            h = F.frame_hash(fr)
            if h in seen:
                dups += 1
            seen.add(h)
            th = F.thumb_hash(fr)
            near[th] = near.get(th, 0) + 1
            if hidden_hashes and h in hidden_hashes:
                leaked += 1

    if n_extra:
        r.warn(f"{n_extra} episode file(s) carry arrays outside the format ({sorted(extra_arrays)}); "
               f"ignored, never loaded")
    if n_total == 0:
        return r
    r.stats.update(n_episodes=len(eps), n_records=n_total, exact_duplicate_frames=dups,
                   near_duplicate_frames=int(sum(v - 1 for v in near.values())),
                   turning_fraction=round(n_turn / n_total, 3))
    if n_total > budget:
        r.error(f"{n_total} records exceed the budget of {budget}")
    if man.get('n_records') != n_total:
        r.warn(f"manifest n_records {man.get('n_records')} != {n_total} counted")
    if leaked:
        r.error(f"{leaked} frame(s) identical to hidden-set frames")
    near_frac = r.stats['near_duplicate_frames'] / n_total
    if near_frac > NEAR_DUP_WARN:
        r.warn(f"{near_frac:.0%} of frames are near-duplicates (16 x 16 thumbnails); they count "
               f"against the budget and teach little")
    if n_turn / n_total < MIN_TURNING_FRAC:
        r.warn(f"only {n_turn / n_total:.1%} of records have |yaw rate| > 0.05 rad/s: almost no "
               f"turning")
    az = np.concatenate(az)
    if az.std() < MIN_AZIMUTH_STD:
        r.warn(f"achieved stern azimuth std {az.std():.3f} rad: the actuator state barely varies")
    a = np.concatenate(acts)
    low = a.std(0) < MIN_ACTION_STD
    if low.any():
        r.warn(f"action channel(s) {np.nonzero(low)[0].tolist()} nearly constant (std {a.std(0).round(3).tolist()})")
    d = np.concatenate(dts)
    r.stats['d_target_range'] = (round(float(d.min()), 2), round(float(d.max()), 2))
    if d.max() - d.min() < 1.0:
        r.warn(f"d_target spans only {d.max() - d.min():.2f} m; the hidden episodes vary it over "
               f"[{guide.D_MIN}, {guide.D_MAX}] and may change it mid-episode")
    if n_total < 0.5 * budget:
        r.warn(f"{n_total} records is under half the budget")
    return r
