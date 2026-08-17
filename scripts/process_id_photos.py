#!/usr/bin/env python3
"""Deterministic, local-only MODNet batch processing for JPEG ID photos."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

try:
    import cv2
    import numpy as np
    import onnxruntime as ort
    from PIL import Image, ImageOps
except ImportError as error:  # pragma: no cover - exercised by environment preflight
    raise SystemExit(
        "Missing local dependency. Use a Python environment containing "
        "numpy, Pillow, opencv-python, and onnxruntime. "
        f"Import error: {error}"
    ) from error


IMAGE_SUFFIXES = {".jpg", ".jpeg"}
UNSUPPORTED_IMAGE_SUFFIXES = {".bmp", ".gif", ".png", ".tif", ".tiff", ".webp"}
DEFAULT_TARGET_SIZE = (240, 320)
DEFAULT_BACKGROUND_RGB = (66, 142, 217)
DEFAULT_MIN_BYTES = 15_360
DEFAULT_MAX_BYTES = 70_000
DEFAULT_MODNET_SHORT_EDGE = 512
DEFAULT_FACE_RATIOS = (63 / 240, 53 / 320, 120 / 240, 120 / 320)
ALPHA_BACKGROUND_CUTOFF = 0.01
ALPHA_FOREGROUND_CUTOFF = 0.995
JPEG_BACKGROUND_TOLERANCE = 12.0


def parse_size(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", value)
    if not match:
        raise argparse.ArgumentTypeError("Use WIDTHxHEIGHT, for example 240x320")
    width, height = (int(part) for part in match.groups())
    if width < 32 or height < 32:
        raise argparse.ArgumentTypeError("Width and height must both be at least 32")
    return width, height


def parse_rgb(value: str) -> tuple[int, int, int]:
    try:
        channels = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use R,G,B, for example 66,142,217") from error
    if len(channels) != 3 or any(channel < 0 or channel > 255 for channel in channels):
        raise argparse.ArgumentTypeError("RGB must contain three integers from 0 through 255")
    return channels  # type: ignore[return-value]


def parse_box(value: str) -> tuple[int, int, int, int]:
    try:
        box = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use four comma-separated integers") from error
    if len(box) != 4:
        raise argparse.ArgumentTypeError("Use four comma-separated integers")
    return box  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Process a flat directory of JPEG ID photos entirely locally with "
            "MODNet continuous alpha matting. The output directory must not exist."
        )
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--target-size", type=parse_size, default=DEFAULT_TARGET_SIZE)
    parser.add_argument(
        "--background-rgb", type=parse_rgb, default=DEFAULT_BACKGROUND_RGB
    )
    parser.add_argument("--min-bytes", type=int, default=DEFAULT_MIN_BYTES)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument(
        "--background-tolerance",
        type=float,
        default=JPEG_BACKGROUND_TOLERANCE,
        help="Decoded safe-background RGB Euclidean-distance gate.",
    )
    parser.add_argument(
        "--modnet-short-edge",
        type=int,
        default=DEFAULT_MODNET_SHORT_EDGE,
        help=(
            "MODNet input short edge; must be a multiple of 32. Default 512 is "
            "validated. 768/1024 have caused false holes in clothing."
        ),
    )
    parser.add_argument(
        "--composition-face-box",
        type=parse_box,
        help=(
            "Target face box x,y,w,h in output pixels. The default scales the "
            "validated 63,53,120,120 composition from a 240x320 target."
        ),
    )
    parser.add_argument(
        "--overrides-json",
        type=Path,
        help=(
            "Optional JSON object keyed by filename. Each value may contain "
            "source_face_box, crop_box, and target_face_box arrays."
        ),
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=0,
        help="Require this many source JPEGs; 0 disables the count check.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N deterministic filenames; use 1 for a sample.",
    )
    parser.add_argument(
        "--padding-policy",
        choices=("jpeg-comment", "fail"),
        default="jpeg-comment",
        help=(
            "If the best JPEG is below min-bytes, add a standards-compliant empty "
            "JPEG comment payload or fail. Padding does not change decoded pixels."
        ),
    )
    return parser


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def validate_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    source_dir = args.source_dir.resolve(strict=True)
    model_path = args.model.resolve(strict=True)
    output_dir = args.output_dir.resolve(strict=False)
    report_path = args.report.resolve(strict=False)
    if not source_dir.is_dir():
        raise NotADirectoryError(source_dir)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists; choose a new directory: {output_dir}"
        )
    if report_path.exists():
        raise FileExistsError(f"Report already exists; choose a new path: {report_path}")
    if paths_overlap(source_dir, output_dir):
        raise ValueError("Source and output directories must be separate and non-nested")
    if source_dir == report_path or source_dir in report_path.parents:
        raise ValueError("Report must not be written inside the source directory")
    if paths_overlap(output_dir, report_path):
        raise ValueError("Report and output paths must be separate and non-nested")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    return source_dir, output_dir, report_path, model_path


def discover_sources(source_dir: Path) -> tuple[list[Path], list[str]]:
    sources: list[Path] = []
    unsupported: list[str] = []
    for path in source_dir.iterdir():
        if not path.is_file():
            continue
        if path.is_symlink():
            unsupported.append(path.name)
            continue
        suffix = path.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            sources.append(path)
        elif suffix in UNSUPPORTED_IMAGE_SUFFIXES:
            unsupported.append(path.name)
    sources.sort(key=lambda path: (path.name.casefold(), path.name))
    unsupported.sort(key=lambda name: (name.casefold(), name))
    return sources, unsupported


def validate_box(
    values: Any,
    *,
    kind: str,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    if not isinstance(values, list) or len(values) != 4:
        raise ValueError(f"{kind} must be an array of four integers")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError(f"{kind} must contain integers")
    a, b, c, d = (int(value) for value in values)
    if kind == "crop_box":
        if not (0 <= a < c <= width and 0 <= b < d <= height):
            raise ValueError(f"crop_box is outside the source bounds: {values}")
    else:
        if not (0 <= a < width and 0 <= b < height and c > 0 and d > 0):
            raise ValueError(f"{kind} is invalid: {values}")
        if a + c > width or b + d > height:
            raise ValueError(f"{kind} is outside the image bounds: {values}")
    return a, b, c, d


def load_overrides(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    resolved = path.resolve(strict=True)
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("overrides-json must contain a top-level JSON object")
    result: dict[str, dict[str, Any]] = {}
    for name, entry in raw.items():
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("Override keys must be plain filenames without directories")
        if not isinstance(entry, dict):
            raise ValueError(f"Override for {name!r} must be an object")
        unknown = set(entry) - {"source_face_box", "crop_box", "target_face_box"}
        if unknown:
            raise ValueError(f"Unknown override fields for {name!r}: {sorted(unknown)}")
        result[name] = entry
    return result


def load_rgb_bytes(payload: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(payload)) as image:
        if image.format not in {"JPEG", "MPO"}:
            raise ValueError(f"Input content is {image.format}, not JPEG/MPO")
        image.seek(0)  # JPEG files from some cameras decode as MPO; use the first frame.
        converted = ImageOps.exif_transpose(image).convert("RGB")
        return np.asarray(converted)


def load_rgb(path: Path) -> np.ndarray:
    return load_rgb_bytes(path.read_bytes())


def default_target_face_box(target_size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = target_size
    x, y, face_width, face_height = DEFAULT_FACE_RATIOS
    return (
        int(round(width * x)),
        int(round(height * y)),
        int(round(width * face_width)),
        int(round(height * face_height)),
    )


def detect_face(rgb: np.ndarray) -> tuple[tuple[int, int, int, int], int]:
    source_height, source_width = rgb.shape[:2]
    detect_scale = min(1.0, 1600.0 / max(source_height, source_width))
    if detect_scale < 1.0:
        resized = cv2.resize(
            rgb,
            (
                max(32, int(round(source_width * detect_scale))),
                max(32, int(round(source_height * detect_scale))),
            ),
            interpolation=cv2.INTER_AREA,
        )
    else:
        resized = rgb
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    min_face = max(30, int(round(min(height, width) * 0.045)))
    max_face = max(min_face + 1, int(round(min(height, width) * 0.55)))
    candidates: list[tuple[int, int, int, int]] = []
    for cascade_name, neighbors in (
        ("haarcascade_frontalface_default.xml", 5),
        ("haarcascade_frontalface_alt2.xml", 4),
    ):
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_name)
        found = cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=neighbors,
            minSize=(min_face, min_face),
            maxSize=(max_face, max_face),
        )
        candidates.extend(tuple(int(value) for value in face) for face in found)
        if candidates:
            break
    if not candidates:
        raise RuntimeError(
            "No frontal face detected. Supply source_face_box or crop_box in overrides-json."
        )

    def score(face: tuple[int, int, int, int]) -> float:
        x, y, face_width, face_height = face
        center_x = (x + face_width / 2) / width
        center_y = (y + face_height / 2) / height
        scale = face_width / min(width, height)
        squareness = abs(face_width - face_height) / max(face_width, face_height)
        return (
            ((center_x - 0.50) / 0.28) ** 2
            + ((center_y - 0.36) / 0.30) ** 2
            + ((scale - 0.18) / 0.18) ** 2
            + squareness
        )

    detected = min(candidates, key=score)
    inverse = 1.0 / detect_scale
    restored = tuple(int(round(value * inverse)) for value in detected)
    return restored, len(candidates)  # type: ignore[return-value]


def crop_from_face(
    source: np.ndarray,
    source_face: tuple[int, int, int, int],
    target_face: tuple[int, int, int, int],
    target_size: tuple[int, int],
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    source_height, source_width = source.shape[:2]
    sx, sy, sw, sh = source_face
    tx, ty, tw, th = target_face
    source_face_aspect = sw / sh
    target_face_aspect = tw / th
    if abs(source_face_aspect / target_face_aspect - 1.0) > 0.02:
        raise ValueError(
            "Source and target face-box aspect ratios differ by more than 2%; "
            "provide a corrected source_face_box override."
        )
    scale = 0.5 * (sw / tw + sh / th)
    crop_width = target_size[0] * scale
    crop_height = target_size[1] * scale
    if crop_width > source_width + 0.5 or crop_height > source_height + 0.5:
        raise ValueError(
            "Requested face composition exceeds the source frame. Use a smaller "
            "target face box or an explicit crop_box override."
        )
    left = sx - tx * scale
    top = sy - ty * scale
    if (
        left < -0.5
        or top < -0.5
        or left + crop_width > source_width + 0.5
        or top + crop_height > source_height + 0.5
    ):
        raise ValueError(
            "Requested composition does not fit inside the source frame without "
            "shifting the configured face box. Supply an approved crop_box or "
            "adjust the target composition."
        )
    box = (
        int(round(left)),
        int(round(top)),
        int(round(left + crop_width)),
        int(round(top + crop_height)),
    )
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= source_width and 0 <= y0 < y1 <= source_height):
        raise ValueError(f"Calculated crop is outside the source frame: {box}")
    crop = source[y0:y1, x0:x1].copy()
    if crop.size == 0:
        raise ValueError("Calculated crop is empty")
    return crop, box


def mapped_target_face_box(
    source_face: tuple[int, int, int, int] | None,
    crop_box: tuple[int, int, int, int],
    target_size: tuple[int, int],
    fallback: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    if source_face is None:
        return fallback
    sx, sy, sw, sh = source_face
    x0, y0, x1, y1 = crop_box
    scale_x = target_size[0] / (x1 - x0)
    scale_y = target_size[1] / (y1 - y0)
    return (
        int(round((sx - x0) * scale_x)),
        int(round((sy - y0) * scale_y)),
        int(round(sw * scale_x)),
        int(round(sh * scale_y)),
    )


def choose_crop(
    source: np.ndarray,
    override: dict[str, Any],
    target_size: tuple[int, int],
    default_target_face: tuple[int, int, int, int],
) -> tuple[
    np.ndarray,
    tuple[int, int, int, int],
    tuple[int, int, int, int] | None,
    tuple[int, int, int, int],
    str,
    int,
    bool,
]:
    source_height, source_width = source.shape[:2]
    target_face = default_target_face
    if "target_face_box" in override:
        target_face = validate_box(
            override["target_face_box"],
            kind="target_face_box",
            width=target_size[0],
            height=target_size[1],
        )
    if "crop_box" in override:
        box = validate_box(
            override["crop_box"],
            kind="crop_box",
            width=source_width,
            height=source_height,
        )
        x0, y0, x1, y1 = box
        crop_width, crop_height = x1 - x0, y1 - y0
        target_ratio = target_size[0] / target_size[1]
        crop_ratio = crop_width / crop_height
        if abs(crop_ratio - target_ratio) > 0.005:
            raise ValueError(
                f"crop_box aspect ratio {crop_ratio:.5f} does not match target "
                f"ratio {target_ratio:.5f}"
            )
        crop_source_face: tuple[int, int, int, int] | None = None
        composition_verified = False
        if "source_face_box" in override:
            crop_source_face = validate_box(
                override["source_face_box"],
                kind="source_face_box",
                width=source_width,
                height=source_height,
            )
            fx, fy, fw, fh = crop_source_face
            if not (x0 <= fx and y0 <= fy and fx + fw <= x1 and fy + fh <= y1):
                raise ValueError("source_face_box must be fully inside crop_box")
            composition_verified = True
        return (
            source[y0:y1, x0:x1].copy(),
            box,
            crop_source_face,
            target_face,
            "crop-override",
            0,
            composition_verified,
        )
    if "source_face_box" in override:
        source_face = validate_box(
            override["source_face_box"],
            kind="source_face_box",
            width=source_width,
            height=source_height,
        )
        candidate_count = 0
        crop_source = "face-override"
    else:
        source_face, candidate_count = detect_face(source)
        source_face = validate_box(
            list(source_face),
            kind="source_face_box",
            width=source_width,
            height=source_height,
        )
        crop_source = "haar-face-detection"
    crop, crop_box = crop_from_face(source, source_face, target_face, target_size)
    return (
        crop,
        crop_box,
        source_face,
        target_face,
        crop_source,
        candidate_count,
        True,
    )


def create_session(model_path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = min(8, os.cpu_count() or 1)
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def modnet_input(rgb: np.ndarray, short_edge: int) -> np.ndarray:
    height, width = rgb.shape[:2]
    ratio = short_edge / min(height, width)
    if height <= width:
        new_height = short_edge
        new_width = max(32, int(round(width * ratio / 32.0)) * 32)
    else:
        new_width = short_edge
        new_height = max(32, int(round(height * ratio / 32.0)) * 32)
    interpolation = cv2.INTER_AREA if ratio <= 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(rgb, (new_width, new_height), interpolation=interpolation)
    tensor = resized.astype(np.float32) / 255.0
    tensor = (tensor - 0.5) / 0.5
    return np.transpose(tensor, (2, 0, 1))[None]


def predict_alpha(
    session: ort.InferenceSession,
    rgb: np.ndarray,
    short_edge: int,
) -> tuple[np.ndarray, list[int], float]:
    tensor = modnet_input(rgb, short_edge)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    started = time.perf_counter()
    raw = session.run([output_name], {input_name: tensor})[0]
    seconds = time.perf_counter() - started
    alpha = np.squeeze(raw).astype(np.float32)
    if alpha.ndim != 2 or not np.isfinite(alpha).all():
        raise RuntimeError(f"MODNet returned an invalid alpha tensor: {raw.shape}")
    alpha = cv2.resize(
        alpha,
        (rgb.shape[1], rgb.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    alpha = np.clip(alpha, 0.0, 1.0)
    # Preserve the continuous matte. Only numerical endpoints are snapped.
    alpha[alpha <= ALPHA_BACKGROUND_CUTOFF] = 0.0
    alpha[alpha >= ALPHA_FOREGROUND_CUTOFF] = 1.0
    return alpha, list(tensor.shape), seconds


def estimate_old_background(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    known = alpha <= ALPHA_BACKGROUND_CUTOFF
    minimum_known = max(100, int(round(alpha.size * 0.001)))
    if int(np.count_nonzero(known)) < minimum_known:
        raise RuntimeError(
            "MODNet found too little sure background for deterministic replacement; "
            "inspect the crop and matte instead of forcing delivery."
        )
    known_float = known.astype(np.float32)
    sigma = max(18.0, min(rgb.shape[:2]) * 0.032)
    weights = cv2.GaussianBlur(known_float, (0, 0), sigmaX=sigma)
    weighted = cv2.GaussianBlur(
        rgb.astype(np.float32) * known_float[..., None],
        (0, 0),
        sigmaX=sigma,
    )
    screen = weighted / np.maximum(weights[..., None], 1e-5)
    fallback = np.median(rgb[known], axis=0)
    screen[weights < 1e-4] = fallback
    return screen


def replace_background(
    rgb: np.ndarray,
    alpha: np.ndarray,
    background_rgb: tuple[int, int, int],
    target_size: tuple[int, int],
) -> tuple[Image.Image, np.ndarray, dict[str, Any]]:
    background = np.asarray(background_rgb, dtype=np.float32)
    old_background = estimate_old_background(rgb, alpha)
    a = alpha[..., None]
    composite = rgb.astype(np.float32) + (1.0 - a) * (
        background[None, None, :] - old_background
    )
    composite[alpha <= ALPHA_BACKGROUND_CUTOFF] = background
    composite = np.clip(np.rint(composite), 0, 255).astype(np.uint8)
    resized_rgb = np.asarray(
        Image.fromarray(composite, "RGB").resize(
            target_size, Image.Resampling.LANCZOS
        )
    ).copy()
    alpha_interpolation = (
        cv2.INTER_AREA
        if target_size[0] <= alpha.shape[1] and target_size[1] <= alpha.shape[0]
        else cv2.INTER_LINEAR
    )
    target_alpha = cv2.resize(alpha, target_size, interpolation=alpha_interpolation)
    target_alpha = np.clip(target_alpha, 0.0, 1.0)
    sure_background = target_alpha <= ALPHA_BACKGROUND_CUTOFF
    resized_rgb[sure_background] = np.asarray(background_rgb, dtype=np.uint8)
    exact_fraction = float(
        np.mean(
            np.all(
                resized_rgb[sure_background]
                == np.asarray(background_rgb, dtype=np.uint8),
                axis=1,
            )
        )
    )
    return (
        Image.fromarray(resized_rgb, "RGB"),
        target_alpha,
        {
            "sure_background_pixels": int(np.count_nonzero(sure_background)),
            "preencode_exact_background_fraction": round(exact_fraction, 6),
        },
    )


def encode_once(image: Image.Image, quality: int, subsampling: int) -> bytes:
    buffer = io.BytesIO()
    image.save(
        buffer,
        format="JPEG",
        quality=quality,
        subsampling=subsampling,
        optimize=True,
        progressive=False,
    )
    return buffer.getvalue()


def pad_jpeg_comment(payload: bytes, minimum: int, maximum: int) -> tuple[bytes, int]:
    if not payload.startswith(b"\xff\xd8"):
        raise ValueError("Cannot pad a payload without a JPEG SOI marker")
    required = max(0, minimum - len(payload))
    if required == 0:
        return payload, 0
    prefix = bytearray(payload[:2])
    remaining = required
    while remaining > 0:
        marker_total = min(remaining, 65_537)
        next_remaining = remaining - marker_total
        if 0 < next_remaining < 4:
            marker_total -= 4 - next_remaining
        if marker_total < 4:
            marker_total = 4
        data_length = marker_total - 4
        prefix.extend(b"\xff\xfe")
        prefix.extend((data_length + 2).to_bytes(2, "big"))
        prefix.extend(b"\x00" * data_length)
        remaining -= marker_total
    padded = bytes(prefix) + payload[2:]
    if not minimum <= len(padded) <= maximum:
        raise RuntimeError(
            f"JPEG comment padding produced {len(padded)} bytes outside "
            f"{minimum}-{maximum}"
        )
    return padded, len(padded) - len(payload)


def encode_jpeg(
    image: Image.Image,
    minimum: int,
    maximum: int,
    padding_policy: str,
) -> tuple[bytes, dict[str, Any]]:
    candidates: list[tuple[int, int, bytes]] = []
    for subsampling in (0, 1, 2):
        for quality in range(100, 0, -1):
            payload = encode_once(image, quality, subsampling)
            if len(payload) <= maximum:
                candidates.append((quality, subsampling, payload))
                break
    if not candidates:
        raise RuntimeError(
            f"Could not encode the target dimensions at or below {maximum} bytes"
        )
    quality, subsampling, payload = max(
        candidates,
        key=lambda item: (item[0], -item[1]),
    )
    padding_bytes = 0
    if len(payload) < minimum:
        if padding_policy == "fail":
            raise RuntimeError(
                f"Best JPEG is {len(payload)} bytes, below minimum {minimum}"
            )
        payload, padding_bytes = pad_jpeg_comment(payload, minimum, maximum)
    return payload, {
        "quality": quality,
        "subsampling": subsampling,
        "optimize": True,
        "padding_policy": padding_policy,
        "padding_bytes": padding_bytes,
    }


def alpha_metrics(alpha: np.ndarray) -> dict[str, Any]:
    foreground = alpha >= 0.5
    ys, xs = np.where(foreground)
    bbox = (
        [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        if len(xs)
        else None
    )
    return {
        "alpha_mean": round(float(alpha.mean()), 6),
        "transparent_fraction": round(
            float(np.mean(alpha <= ALPHA_BACKGROUND_CUTOFF)), 6
        ),
        "soft_edge_fraction": round(
            float(np.mean((alpha > 0.05) & (alpha < 0.95))), 6
        ),
        "opaque_fraction": round(
            float(np.mean(alpha >= ALPHA_FOREGROUND_CUTOFF)), 6
        ),
        "foreground_fraction": round(float(foreground.mean()), 6),
        "subject_bbox": bbox,
        "head_top_ratio": round(float(ys.min() / alpha.shape[0]), 6) if len(ys) else None,
    }


def validate_payload(
    payload: bytes,
    target_size: tuple[int, int],
    minimum: int,
    maximum: int,
    background_rgb: tuple[int, int, int],
    target_alpha: np.ndarray,
    background_tolerance: float,
) -> dict[str, Any]:
    if not minimum <= len(payload) <= maximum:
        raise ValueError(f"JPEG size {len(payload)} is outside {minimum}-{maximum}")
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        if image.format != "JPEG":
            raise ValueError(f"Unexpected output format: {image.format}")
        if image.size != target_size:
            raise ValueError(f"Unexpected output dimensions: {image.size}")
        if image.mode != "RGB":
            raise ValueError(f"Unexpected output mode: {image.mode}")
        decoded = np.asarray(image)
    safe = (target_alpha <= ALPHA_BACKGROUND_CUTOFF).astype(np.uint8)
    safe = cv2.erode(safe, np.ones((5, 5), np.uint8), iterations=1).astype(bool)
    if int(np.count_nonzero(safe)) < 64:
        raise ValueError("Too few safe background pixels for decoded JPEG QA")
    delta = decoded.astype(np.float32) - np.asarray(background_rgb, dtype=np.float32)
    distances = np.linalg.norm(delta, axis=2)[safe]
    p99 = float(np.percentile(distances, 99))
    far_fraction = float(np.mean(distances > background_tolerance))
    if p99 > background_tolerance or far_fraction > 0.01:
        raise ValueError(
            "Decoded JPEG background is outside tolerance: "
            f"p99={p99:.3f}, far_fraction={far_fraction:.6f}"
        )
    return {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "format": "JPEG",
        "mode": "RGB",
        "width": target_size[0],
        "height": target_size[1],
        "decoded_background_distance_p95": round(
            float(np.percentile(distances, 95)), 3
        ),
        "decoded_background_distance_p99": round(p99, 3),
        "decoded_background_far_fraction": round(far_fraction, 6),
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(
        payload, ensure_ascii=False, indent=2, allow_nan=False
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".codex-tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        json.loads(temporary.read_text(encoding="utf-8"))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def report_base(
    args: argparse.Namespace,
    model_path: Path,
    model_hash: str,
    selected_count: int,
    discovered_count: int,
    warnings: list[str],
) -> dict[str, Any]:
    target_face = args.composition_face_box or default_target_face_box(args.target_size)
    return {
        "schema_version": 1,
        "status": "running",
        "exit_reason": None,
        "output_committed": False,
        "privacy": {
            "inference": "local-only",
            "generative_model_used": False,
            "external_image_service_used": False,
            "absolute_paths_recorded": False,
        },
        "config": {
            "target_size": list(args.target_size),
            "background_rgb": list(args.background_rgb),
            "min_bytes": args.min_bytes,
            "max_bytes": args.max_bytes,
            "background_tolerance": args.background_tolerance,
            "modnet_short_edge": args.modnet_short_edge,
            "continuous_alpha": True,
            "alpha_background_cutoff": ALPHA_BACKGROUND_CUTOFF,
            "alpha_foreground_cutoff": ALPHA_FOREGROUND_CUTOFF,
            "composition_face_box": list(target_face),
            "padding_policy": args.padding_policy,
            "expected_count": args.expected_count,
            "limit": args.limit,
        },
        "model": {
            "family": "MODNet",
            "format": "ONNX",
            "filename": model_path.name,
            "sha256": model_hash,
            "provider": "CPUExecutionProvider",
        },
        "runtime": {
            "python": sys.version.split()[0],
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "pillow": Image.__version__,
            "onnxruntime": ort.__version__,
        },
        "counts": {
            "discovered": discovered_count,
            "selected": selected_count,
            "processed": 0,
            "failed": 0,
        },
        "warnings": warnings,
        "files": [],
        "failures": [],
    }


def run(args: argparse.Namespace) -> int:
    if args.min_bytes <= 0 or args.max_bytes < args.min_bytes:
        raise ValueError("Require 0 < min-bytes <= max-bytes")
    if args.expected_count < 0 or args.limit < 0:
        raise ValueError("expected-count and limit must not be negative")
    if args.modnet_short_edge < 32 or args.modnet_short_edge % 32:
        raise ValueError("modnet-short-edge must be at least 32 and a multiple of 32")
    if not np.isfinite(args.background_tolerance) or not (
        0 < args.background_tolerance <= math.sqrt(3 * 255**2)
    ):
        raise ValueError("background-tolerance must be finite and within (0, 441.673]")
    source_dir, output_dir, report_path, model_path = validate_paths(args)
    sources, unsupported = discover_sources(source_dir)
    if unsupported:
        raise RuntimeError(
            "Unsupported image files are present. Convert them to JPEG first so "
            "output filenames can remain unchanged."
        )
    if not sources:
        raise RuntimeError("No .jpg or .jpeg files found in the source directory")
    if args.expected_count and len(sources) != args.expected_count:
        raise RuntimeError(
            f"Expected {args.expected_count} source JPEGs, found {len(sources)}"
        )
    selected = sources[: args.limit] if args.limit else sources
    overrides = load_overrides(args.overrides_json)
    missing_override_names = sorted(set(overrides) - {path.name for path in sources})
    if missing_override_names:
        raise ValueError(
            "overrides-json contains filenames absent from source-dir: "
            + ", ".join(repr(name) for name in missing_override_names)
        )
    target_face = args.composition_face_box or default_target_face_box(args.target_size)
    target_face = validate_box(
        list(target_face),
        kind="target_face_box",
        width=args.target_size[0],
        height=args.target_size[1],
    )
    warnings: list[str] = []
    if args.modnet_short_edge > DEFAULT_MODNET_SHORT_EDGE:
        warning = (
            f"MODNet short edge {args.modnet_short_edge} is experimental; 768/1024 "
            "have produced false holes in clothing. Inspect a sample before batch use."
        )
        warnings.append(warning)
        print(f"WARNING: {warning}", file=sys.stderr, flush=True)
    model_hash = sha256_file(model_path)
    report = report_base(
        args,
        model_path,
        model_hash,
        len(selected),
        len(sources),
        warnings,
    )
    source_hashes = {path.name: sha256_file(path) for path in selected}
    try:
        descriptor = os.open(
            report_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as error:
        raise FileExistsError(
            f"Report was claimed by another process: {report_path}"
        ) from error
    else:
        os.close(descriptor)
    report["status"] = "reserved"
    atomic_json(report_path, report)
    load_started = time.perf_counter()
    try:
        session = create_session(model_path)
    except Exception as error:
        report["status"] = "failed"
        report["exit_reason"] = "model_session_failed"
        report["counts"]["failed"] = 1
        report["failures"] = [
            {"name": None, "error": f"{type(error).__name__}: {error}"}
        ]
        atomic_json(report_path, report)
        print(f"FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    report["model"]["load_seconds"] = round(time.perf_counter() - load_started, 4)
    report["model"]["session_providers"] = session.get_providers()

    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.staging-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        for index, source_path in enumerate(selected, 1):
            print(f"[{index}/{len(selected)}] processing", flush=True)
            try:
                source_payload = source_path.read_bytes()
                payload_hash = hashlib.sha256(source_payload).hexdigest()
                if payload_hash != source_hashes[source_path.name]:
                    raise RuntimeError("Source changed after the initial directory snapshot")
                source_rgb = load_rgb_bytes(source_payload)
                (
                    crop,
                    crop_box,
                    source_face,
                    used_target_face,
                    crop_source,
                    candidates,
                    composition_verified,
                ) = choose_crop(
                    source_rgb,
                    overrides.get(source_path.name, {}),
                    args.target_size,
                    target_face,
                )
                actual_target_face = mapped_target_face_box(
                    source_face,
                    crop_box,
                    args.target_size,
                    used_target_face,
                )
                alpha, input_shape, inference_seconds = predict_alpha(
                    session,
                    crop,
                    args.modnet_short_edge,
                )
                image, target_alpha, background_metrics = replace_background(
                    crop,
                    alpha,
                    args.background_rgb,
                    args.target_size,
                )
                payload, encoding = encode_jpeg(
                    image,
                    args.min_bytes,
                    args.max_bytes,
                    args.padding_policy,
                )
                validation = validate_payload(
                    payload,
                    args.target_size,
                    args.min_bytes,
                    args.max_bytes,
                    args.background_rgb,
                    target_alpha,
                    args.background_tolerance,
                )
                (staging / source_path.name).write_bytes(payload)
                entry = {
                    "name": source_path.name,
                    "source_sha256": source_hashes[source_path.name],
                    "source_width": int(source_rgb.shape[1]),
                    "source_height": int(source_rgb.shape[0]),
                    "crop_source": crop_source,
                    "crop_box": list(crop_box),
                    "source_face_box": list(source_face) if source_face else None,
                    "requested_target_face_box": list(used_target_face),
                    "target_face_box": list(actual_target_face),
                    "composition_verified": composition_verified,
                    "face_candidate_count": candidates,
                    "model_input_shape": input_shape,
                    "inference_seconds": round(inference_seconds, 4),
                    "encoding": encoding,
                    **alpha_metrics(target_alpha),
                    **background_metrics,
                    **validation,
                }
                report["files"].append(entry)
                report["counts"]["processed"] += 1
            except Exception as error:
                failure = {
                    "name": source_path.name,
                    "error": f"{type(error).__name__}: {error}",
                }
                report["failures"].append(failure)
                report["counts"]["failed"] = 1
                report["status"] = "failed"
                report["exit_reason"] = "image_processing_failed"
                atomic_json(report_path, report)
                print(f"FAILED: {failure['error']}", file=sys.stderr, flush=True)
                return 1

        changed_sources = [
            path.name
            for path in selected
            if sha256_file(path) != source_hashes[path.name]
        ]
        if changed_sources:
            report["status"] = "failed"
            report["exit_reason"] = "source_changed_during_run"
            report["failures"] = [
                {
                    "name": name,
                    "error": "Source hash changed during processing; output not committed",
                }
                for name in changed_sources
            ]
            report["counts"]["failed"] = len(changed_sources)
            atomic_json(report_path, report)
            return 1

        current_sources, current_unsupported = discover_sources(source_dir)
        if not args.limit and (
            current_unsupported
            or [path.name for path in current_sources] != [path.name for path in sources]
        ):
            report["status"] = "failed"
            report["exit_reason"] = "source_file_set_changed_during_run"
            report["failures"] = [
                {
                    "name": None,
                    "error": "Source filename set changed during processing; output not committed",
                }
            ]
            report["counts"]["failed"] = 1
            atomic_json(report_path, report)
            return 1

        staging_mismatches = [
            entry["name"]
            for entry in report["files"]
            if sha256_file(staging / entry["name"]) != entry["sha256"]
        ]
        if staging_mismatches:
            report["status"] = "failed"
            report["exit_reason"] = "staging_output_hash_mismatch"
            report["failures"] = [
                {
                    "name": name,
                    "error": "Staged output hash differs from the validated payload",
                }
                for name in staging_mismatches
            ]
            report["counts"]["failed"] = len(staging_mismatches)
            atomic_json(report_path, report)
            return 1

        report["status"] = "ready_to_commit"
        report["exit_reason"] = None
        atomic_json(report_path, report)
        try:
            if output_dir.exists():
                raise FileExistsError(
                    "Output directory appeared during processing; refusing to replace it"
                )
            os.rename(staging, output_dir)
        except Exception as error:
            report["status"] = "failed"
            report["exit_reason"] = "output_commit_failed"
            report["counts"]["failed"] = 1
            report["failures"] = [
                {"name": None, "error": f"{type(error).__name__}: {error}"}
            ]
            atomic_json(report_path, report)
            return 1

    report["status"] = "pass"
    report["output_committed"] = True
    report["exit_reason"] = "all_selected_files_processed_and_validated"
    atomic_json(report_path, report)
    print(
        f"PASS processed={report['counts']['processed']} report={report_path}",
        flush=True,
    )
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as error:
        print(f"FATAL: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
