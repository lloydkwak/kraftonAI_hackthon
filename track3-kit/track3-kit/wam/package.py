"""The dataset: how it is packaged, and how to read it.

Per episode, two files:

    ep_000123.mp4   lossless H.264 in RGB (`libx264rgb -qp 0 -pix_fmt gbrp`), 25 Hz.  The
                    pixel format matters: `-qp 0` into the default `yuv420p` would be
                    lossless only of the converted signal and drop three quarters of the
                    chroma.  Every episode was verified byte-identical after a decode round
                    trip at packaging time.
    ep_000123.npz   `actions` float32 `(T-1,)` if the episode is labelled, else absent;
                    `length` `T`; `episode_id`; `labelled`.  No reward, no termination,
                    no state.

plus `manifest.json` (the episode list with lengths and the label flag, and `dev_subset`,
the 10 % development subset), `sha256sums.txt`, and `to_memmap`, the one-time decode that
lets a training loop read a memmap rather than a codec.

`load_episode` asserts `T` frames and `T-1` actions on every read.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import subprocess

import numpy as np

from . import contract as CT

PRESET = "medium"       # x264 preset used for packaging; every preset is lossless at -qp 0


def _ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


BITEXACT = ["-fflags", "+bitexact", "-flags:v", "+bitexact", "-map_metadata", "-1"]


def encode_video(frames: np.ndarray, path: str, fps: int = 25, preset: str = PRESET, bitexact: bool = False):
    """uint8 `(T, R, R, 3)` -> lossless H.264 RGB file.  Returns the byte size.  `bitexact` strips the
    muxer's metadata so the same frames give the same bytes."""
    T, H, W, _ = frames.shape
    p = subprocess.run(
        [_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264rgb", "-qp", "0", "-preset", preset, "-pix_fmt", "gbrp"]
        + (BITEXACT if bitexact else []) + [path],
        input=np.ascontiguousarray(frames).tobytes(), capture_output=True)
    if p.returncode != 0:
        raise RuntimeError("ffmpeg encode failed: " + p.stderr.decode()[:400])
    return os.path.getsize(path)


def decode_video(path: str, res: int = CT.RES) -> np.ndarray:
    """Lossless file -> uint8 `(T, R, R, 3)`."""
    d = subprocess.run([_ffmpeg(), "-hide_banner", "-loglevel", "error", "-i", path,
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
    if d.returncode != 0:
        raise RuntimeError("ffmpeg decode failed: " + d.stderr.decode()[:400])
    buf = np.frombuffer(d.stdout, dtype=np.uint8)
    return buf.reshape(-1, res, res, 3)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def savez_deterministic(path: str, **arrays):
    """`np.savez` with fixed zip timestamps, so the same arrays give the same bytes (`np.savez` stamps the
    wall clock into every member)."""
    import io
    import zipfile
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, arr in arrays.items():
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.asanyarray(arr), allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            zf.writestr(info, buf.getvalue())


def package_episode(out_dir: str, episode_id: int, frames: np.ndarray, actions, labelled: bool,
                    verify: bool = True, deterministic: bool = False) -> dict:
    """Write one episode's pair of files; returns its manifest row.  `deterministic` writes both
    files byte-reproducibly."""
    stem = os.path.join(out_dir, f"ep_{episode_id:06d}")
    size = encode_video(frames, stem + ".mp4", bitexact=deterministic)
    if verify:
        back = decode_video(stem + ".mp4", frames.shape[1])
        if back.shape != frames.shape or not np.array_equal(back, frames):
            raise RuntimeError(f"episode {episode_id}: decode is not byte-identical")
    side = dict(length=np.int64(len(frames)), episode_id=np.int64(episode_id),
                labelled=np.bool_(labelled))
    if labelled:
        a = np.asarray(actions, dtype=np.float32)
        assert len(a) == len(frames) - 1, "T frames need T-1 actions"
        side["actions"] = a
    (savez_deterministic if deterministic else np.savez)(stem + ".npz", **side)
    return dict(episode_id=episode_id, length=int(len(frames)), labelled=bool(labelled),
                video=os.path.basename(stem + ".mp4"), sidecar=os.path.basename(stem + ".npz"),
                video_bytes=int(size))


def package_pilot(pilot_dir: str, out_dir: str, dev_fraction: float = 0.10, seed: int = 0,
                  label_frac: float | None = None, label_seed: int = 20260911, log=None) -> dict:
    """Organizer-side (it reads the generator's shards, which do not ship): package every
    episode of `pilot_dir`'s shards.  Family and physics are read from the shard manifest
    rows and **not** written out.

    `label_frac` re-draws the labelled split at packaging time (stratified by family)
    instead of taking the shards' own `labelled` flags; `None` keeps the shards' flags."""
    from . import dataset as DS
    os.makedirs(out_dir, exist_ok=True)
    shards = sorted(glob.glob(os.path.join(pilot_dir, "shard_*.npz")))
    relabel = None
    if label_frac is not None:
        metas = []
        for path in shards:                                  # meta only; frames stay on disk
            metas += json.loads(str(np.load(path, allow_pickle=False)["meta"]))
        metas.sort(key=lambda m: m["index"])
        lab = DS.assign_labels([m["family"] for m in metas], np.random.default_rng(label_seed), label_frac)
        relabel = {m["index"]: bool(l) for m, l in zip(metas, lab)}
    rows = []
    for path in shards:
        eps = DS.load_shard(path, participant_view=False, with_masks=False)
        for e in eps:
            labelled = relabel[e.meta["index"]] if relabel is not None else e.meta["labelled"]
            rows.append(package_episode(out_dir, e.meta["index"], e.frames, e.actions, labelled))
        if log:
            log(f"  packaged {len(rows)} episodes ({os.path.basename(path)})")
    rng = np.random.default_rng(seed)
    ids = np.array([r["episode_id"] for r in rows])
    dev = sorted(int(i) for i in rng.choice(ids, size=max(1, int(round(dev_fraction * len(ids)))),
                                            replace=False))
    manifest = dict(format="hangang-track3-dataset/1", resolution=CT.RES, fps=25,
                    codec="libx264rgb -qp 0 -pix_fmt gbrp (lossless RGB)", n_episodes=len(rows),
                    n_frames=int(sum(r["length"] for r in rows)),
                    n_labelled=int(sum(r["labelled"] for r in rows)),
                    label_frac=(float(label_frac) if label_frac is not None else None),
                    label_rule="stratified by behaviour family; re-drawn at packaging" if label_frac is not None
                    else "the generation-time split",
                    bytes_video=int(sum(r["video_bytes"] for r in rows)),
                    action_convention="a[t] applies between o[t] and o[t+1]",
                    fields="video + (actions on labelled episodes); no reward, no termination, no state",
                    dev_subset=dev, episodes=rows)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    with open(os.path.join(out_dir, "sha256sums.txt"), "w") as f:
        for name in sorted(os.listdir(out_dir)):
            if name.endswith((".mp4", ".npz", ".json")):
                f.write(f"{sha256(os.path.join(out_dir, name))}  {name}\n")
    return manifest


def write_sums(out_dir: str):
    with open(os.path.join(out_dir, "sha256sums.txt"), "w") as f:
        for name in sorted(os.listdir(out_dir)):
            if name.endswith((".mp4", ".npz", ".json")):
                f.write(f"{sha256(os.path.join(out_dir, name))}  {name}\n")


def package_append(out_dir: str, rows: list, dev_fraction: float = 0.10, seed: int = 0, note: str = "") -> dict:
    """Add already-written episodes (`rows`, from `package_episode`) to a packaged dataset without
    touching a byte of the existing episode files -- participants download only the new ones.  The manifest's
    totals are recomputed, the dev subset keeps its old members and gains `dev_fraction` of the new ones
    (seeded), and `sha256sums.txt` is rewritten over every file."""
    path = os.path.join(out_dir, "manifest.json")
    with open(path) as f:
        man = json.load(f)
    old = {r["episode_id"] for r in man["episodes"]}
    new = [r for r in rows if r["episode_id"] not in old]
    if len(new) != len(rows):
        raise ValueError(f"{len(rows) - len(new)} appended episodes already in the manifest")
    ids = np.array(sorted(r["episode_id"] for r in new))
    rng = np.random.default_rng(seed)
    dev_new = sorted(int(i) for i in rng.choice(ids, size=max(1, int(round(dev_fraction * len(ids)))), replace=False))
    eps = man["episodes"] + sorted(new, key=lambda r: r["episode_id"])
    man.update(n_episodes=len(eps), n_frames=int(sum(r["length"] for r in eps)),
               n_labelled=int(sum(r["labelled"] for r in eps)), bytes_video=int(sum(r["video_bytes"] for r in eps)),
               dev_subset=sorted(man["dev_subset"] + dev_new), episodes=eps)
    man.setdefault("releases", []).append(dict(note=note, episodes=[int(ids[0]), int(ids[-1])], n=len(new),
                                               n_labelled=int(sum(r["labelled"] for r in new))))
    with open(path, "w") as f:
        json.dump(man, f, indent=1)
    write_sums(out_dir)
    return man


def load_episode(pkg_dir: str, episode_id: int):
    """Decode one packaged episode.  Returns `(frames, actions or None)`; asserts `T` frames, `T-1` actions."""
    stem = os.path.join(pkg_dir, f"ep_{episode_id:06d}")
    frames = decode_video(stem + ".mp4")
    side = np.load(stem + ".npz")
    assert int(side["length"]) == len(frames), "length field disagrees with the video"
    for k in ("rewards", "done", "states"):
        assert k not in side.files, f"{k!r} must not ship"
    acts = side["actions"] if "actions" in side.files else None
    if acts is not None:
        assert len(acts) == len(frames) - 1, "T frames, T-1 actions"
    return frames, acts


def to_memmap(pkg_dir: str, out_path: str, episode_ids=None, log=None):
    """Decode once into a uint8 memmap `(N_frames, R, R, 3)` plus an index `npz` with
    per-episode offsets and actions -- what a training loop reads.  `episode_ids` decodes a subset."""
    with open(os.path.join(pkg_dir, "manifest.json")) as f:
        man = json.load(f)
    rows = man["episodes"]
    if episode_ids is not None:
        want = set(int(i) for i in episode_ids)
        rows = [r for r in rows if r["episode_id"] in want]
    n = sum(r["length"] for r in rows)
    mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint8, shape=(n, CT.RES, CT.RES, 3))
    off = [0]
    acts, lab, ids = [], [], []
    for r in rows:
        frames, a = load_episode(pkg_dir, r["episode_id"])
        mm[off[-1]:off[-1] + len(frames)] = frames
        off.append(off[-1] + len(frames))
        acts.append(a if a is not None else np.zeros(len(frames) - 1, np.float32) * np.nan)
        lab.append(a is not None)
        ids.append(r["episode_id"])
        if log and len(ids) % 50 == 0:
            log(f"  memmap: {len(ids)}/{len(rows)} episodes")
    if log:
        log(f"  memmap: flushing {mm.nbytes / 1e9:.1f} GB to disk (this can take minutes)")
    mm.flush()
    np.savez(out_path.replace(".npy", "") + "_index.npz", offsets=np.asarray(off),
             actions=np.concatenate(acts), labelled=np.asarray(lab), episode_ids=np.asarray(ids))
    return out_path
