"""A PNG encoder for the studio's frames: zlib and struct, nothing else (the kit's one dependency is
numpy). Written here rather than borrowed from Track 1's `usvnav/png.py` so that no `usvnav` module
is loaded beyond the renderer's own (the ones `raster.renderer_id()` hashes)."""
import struct
import zlib

import numpy as np


def _chunk(tag, data):
    body = tag + data
    return struct.pack('>I', len(data)) + body + struct.pack('>I', zlib.crc32(body) & 0xffffffff)


def encode_png(rgb, scale=1):
    """`rgb` (H, W, 3) uint8 -> PNG bytes; `scale` repeats every pixel (nearest neighbour), so a
    128 px frame can be shown pixel for pixel at 3x without the browser smoothing it."""
    a = np.asarray(rgb, np.uint8)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    if scale > 1:
        a = np.repeat(np.repeat(a, scale, axis=0), scale, axis=1)
    h, w = a.shape[:2]
    rows = np.concatenate([np.zeros((h, 1), np.uint8), a.reshape(h, w * 3)], axis=1)   # filter byte 0
    return (b'\x89PNG\r\n\x1a\n'
            + _chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
            + _chunk(b'IDAT', zlib.compress(rows.tobytes(), 6))
            + _chunk(b'IEND', b''))
