"""A minimal PNG encoder using only the standard library, plus lossy frame archiving.

Why not Pillow for PNG: the protocol only needs "store a uint8 array as PNG", and a C-extension
dependency is not worth it for that. Problems in this step are also hard to trace inside a
library; with our own encoder they are right in front of us.
"""

from __future__ import annotations

import io
import struct
import zlib

import numpy as np

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def encode_png(image: np.ndarray) -> bytes:
    """Encode an HxWx3 (RGB) or HxW (grayscale) uint8 array as PNG bytes."""
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"Expected HxWx3 or HxWx4, got {arr.shape}")
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]

    height, width = arr.shape[:2]
    # Each row starts with filter byte 0 (None).
    raw = b"".join(b"\x00" + arr[row].tobytes() for row in range(height))
    return b"".join(
        [
            PNG_SIGNATURE,
            _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
            _chunk(b"IDAT", zlib.compress(raw, 6)),
            _chunk(b"IEND", b""),
        ]
    )


def _chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


#: Lossy quality of archived frames. Measured on 146 real LIBERO frames (256x256): WebP q90 is
#: 9.1% of the PNG size with a mean pixel error of 1.11/255, and encodes in 5.1 ms per frame,
#: faster than our PNG encoder (7.1 ms). JPEG at similar quality is 15.9% with error 1.18, worse
#: on both counts, so JPEG is not used.
FRAME_WEBP_QUALITY = 90


def encode_frame(image: np.ndarray) -> tuple[bytes, str]:
    """Encode one frame in the archive format; returns (bytes, extension).

    An episode archives on the order of two thousand frames, and lossless PNG would waste disk:
    these frames are for people replaying and analyzing runs, not what the model receives (that
    path still uses `encode_png`, byte for byte).

    Pillow is imported **lazily**, so machines without a simulator can still run every pure-logic
    test; in simulator environments LIBERO/robosuite have pulled Pillow in already. Without it,
    frames fall back to PNG, which works the same and is only larger.
    """
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        return encode_png(arr), "png"
    buffer = io.BytesIO()
    Image.fromarray(arr).save(buffer, "WEBP", quality=FRAME_WEBP_QUALITY, method=4)
    return buffer.getvalue(), "webp"


def decode_png_size(data: bytes) -> tuple[int, int]:
    """Read (width, height) from PNG bytes. Tests use it to check encoded images."""
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("Not a PNG")
    width, height = struct.unpack(">II", data[16:24])
    return int(width), int(height)
