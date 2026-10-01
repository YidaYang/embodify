"""Versioned backend wire values; no pickle or arbitrary object construction."""
from __future__ import annotations

import base64
import io
import json
import zlib
from dataclasses import asdict

import numpy as np

from .base import ArmInfo, ArmState, BackendInfo, CameraInfo, Snapshot
from ..types import TaskSpec

VERSION = 1
MAX_LINE = 64 * 1024 * 1024
MAX_PIXELS = 4096 * 4096


def dumps(value):
    def scalar(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    data = (json.dumps(value, ensure_ascii=False, allow_nan=False, default=scalar,
                       separators=(",", ":")) + "\n").encode("utf-8")
    if len(data) > MAX_LINE:
        raise ValueError("backend message too large")
    return data


def loads(data):
    def invalid(value):
        raise ValueError("non-finite JSON value")
    value = json.loads(data, parse_constant=invalid)
    if not isinstance(value, dict) or value.get("v") != VERSION:
        raise ValueError("incompatible backend protocol")
    return value


def codecs():
    try:
        from PIL import features
        if features.check("webp"):
            return ["webp", "rgb-zlib"]
    except ImportError:
        pass
    return ["rgb-zlib"]


def encode_image(image, codec):
    arr = np.asarray(image)
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("expected RGB uint8 image")
    h, w = arr.shape[:2]
    if h <= 0 or w <= 0 or h * w > MAX_PIXELS:
        raise ValueError("invalid image dimensions")
    if codec == "webp":
        from PIL import Image
        buf = io.BytesIO()
        # Agent-visible pixels remain exact; journal compression is a separate decision.
        Image.fromarray(arr).save(buf, "WEBP", lossless=True, method=2)
        blob = buf.getvalue()
    elif codec == "rgb-zlib":
        blob = zlib.compress(arr.tobytes(), 3)
    else:
        raise ValueError("unsupported image codec")
    return {"codec": codec, "shape": [h, w, 3], "data": base64.b64encode(blob).decode("ascii")}


def decode_image(value):
    shape = value["shape"]
    if (not isinstance(shape, list) or len(shape) != 3 or shape[2] != 3
            or any(type(x) is not int or x <= 0 for x in shape)
            or shape[0] * shape[1] > MAX_PIXELS):
        raise ValueError("invalid image dimensions")
    blob = base64.b64decode(value["data"], validate=True)
    size = shape[0] * shape[1] * 3
    if value["codec"] == "rgb-zlib":
        decoder = zlib.decompressobj()
        raw = decoder.decompress(blob, size + 1)
        if len(raw) != size or not decoder.eof or decoder.unused_data:
            raise ValueError("invalid compressed image")
        return np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy()
    if value["codec"] == "webp":
        from PIL import Image
        with Image.open(io.BytesIO(blob)) as image:
            if image.format != "WEBP" or list(image.size) != [shape[1], shape[0]]:
                raise ValueError("image header mismatch")
            return np.array(image.convert("RGB"), dtype=np.uint8)
    raise ValueError("unsupported image codec")


def vector(value, size):
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (size,) or not np.isfinite(arr).all():
        raise ValueError("invalid pose vector")
    return arr


def pack_snapshot(snapshot, codec, images=True):
    return {
        "step": int(snapshot.step),
        "arms": {name: {"pos": arm.pos.tolist(), "quat": arm.quat.tolist(),
                        "opening": None if arm.gripper_opening is None else float(arm.gripper_opening), "command": arm.gripper_command}
                 for name, arm in snapshot.arms.items()},
        "images": {name: encode_image(img, codec) for name, img in snapshot.images.items()} if images else {},
    }


def unpack_snapshot(value):
    if type(value["step"]) is not int or value["step"] < 0:
        raise ValueError("invalid step")
    arms = {}
    for name, arm in value["arms"].items():
        opening = None if arm["opening"] is None else float(arm["opening"])
        if (opening is not None and not np.isfinite(opening)) or arm["command"] not in ("open", "close"):
            raise ValueError("invalid gripper state")
        arms[name] = ArmState(vector(arm["pos"], 3), vector(arm["quat"], 4), opening, arm["command"])
    if not arms:
        raise ValueError("empty arms")
    return Snapshot(value["step"], arms, {n: decode_image(v) for n, v in value["images"].items()})


def pack_task(task):
    # Omit an unset per-task budget so peers without that field still read ordinary tasks.
    data = asdict(task)
    if data["episode_budget"] is None:
        del data["episode_budget"]
    return data


def unpack_task(value):
    task = TaskSpec(**value)
    if task.episode_budget is not None and (type(task.episode_budget) is not int or task.episode_budget <= 0):
        raise ValueError("invalid task episode budget")
    return task


def unpack_info(value):
    data = dict(value)
    if type(data.get("supports_coordinated_control", False)) is not bool:
        raise ValueError("invalid coordinated capability")
    data["arms"] = tuple(ArmInfo(**dict(a, rotation_axes=tuple(a["rotation_axes"]))) for a in data["arms"])
    data["cameras"] = tuple(CameraInfo(**c) for c in data["cameras"])
    for field in ("forbidden_terms", "move_stop_reasons", "gripper_stop_reasons"):
        data[field] = tuple(tuple(v) if isinstance(v, list) else v for v in data[field])
    return BackendInfo(**data)
