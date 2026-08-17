#!/usr/bin/env python3
"""Independent specification QA and visual triage for processed ID photos."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw
except ImportError as error:  # pragma: no cover - environment preflight
    raise SystemExit(
        "Missing local dependency. Use a Python environment containing "
        f"numpy, Pillow, opencv-python, and onnxruntime. Import error: {error}"
    ) from error

from process_id_photos import (
    ALPHA_BACKGROUND_CUTOFF,
    IMAGE_SUFFIXES,
    create_session,
    load_rgb_bytes,
    paths_overlap,
    predict_alpha,
    replace_background,
    sha256_file,
)


@dataclass
class VisualEntry:
    metrics: dict[str, Any]
    source: np.ndarray
    output: np.ndarray
    alpha: np.ndarray
    possible_loss: np.ndarray
    interior_holes: np.ndarray
    edge_residual: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Re-run MODNet locally, verify output specifications, and create "
            "contact sheets for edge, missing-hair, hole, and background review."
        )
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-report", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--qa-dir", type=Path, required=True)
    parser.add_argument(
        "--background-tolerance",
        type=float,
        default=None,
        help=(
            "Decoded RGB Euclidean-distance gate. Defaults to the processing "
            "report and may only be made stricter."
        ),
    )
    parser.add_argument(
        "--source-background-separation",
        type=float,
        default=25.0,
        help="Minimum source-color distance from its corner backdrop estimate.",
    )
    parser.add_argument(
        "--possible-loss-fraction",
        type=float,
        default=0.00015,
        help="Image-area fraction that triggers possible missing-hair review.",
    )
    parser.add_argument(
        "--interior-hole-fraction",
        type=float,
        default=0.00015,
        help="Image-area fraction that triggers enclosed matte-hole review.",
    )
    parser.add_argument(
        "--edge-residual-fraction",
        type=float,
        default=0.02,
        help="Outer-edge fraction that triggers background-residual review.",
    )
    parser.add_argument(
        "--head-top-min-ratio",
        type=float,
        default=0.015,
        help="Minimum subject top margin as a fraction of output height.",
    )
    parser.add_argument(
        "--head-top-max-ratio",
        type=float,
        default=0.20,
        help="Maximum subject top margin as a fraction of output height.",
    )
    parser.add_argument(
        "--allow-visual-flags",
        action="store_true",
        help=(
            "Exit 0 despite visual triage flags. Use only after explicit human "
            "acceptance; the report still records every flag."
        ),
    )
    return parser


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


def image_map(directory: Path) -> dict[str, Path]:
    paths = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    symlinks = [path.name for path in paths if path.is_symlink()]
    if symlinks:
        raise ValueError(f"Image symlinks are not accepted in {directory}")
    result = {path.name: path for path in paths}
    if len(result) != len(paths):
        raise RuntimeError(f"Duplicate case-sensitive filenames in {directory}")
    return result


def strict_crop(source: np.ndarray, box: Any) -> np.ndarray:
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError("batch report crop_box must contain four integers")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in box):
        raise ValueError("batch report crop_box must contain integers")
    x0, y0, x1, y1 = (int(value) for value in box)
    height, width = source.shape[:2]
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"batch report crop_box is outside source bounds: {box}")
    return source[y0:y1, x0:x1].copy()


def resize_rgb(rgb: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    return np.asarray(
        Image.fromarray(rgb, "RGB").resize(target_size, Image.Resampling.LANCZOS)
    )


def target_alpha(alpha: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    interpolation = (
        cv2.INTER_AREA
        if target_size[0] <= alpha.shape[1] and target_size[1] <= alpha.shape[0]
        else cv2.INTER_LINEAR
    )
    return np.clip(
        cv2.resize(alpha, target_size, interpolation=interpolation), 0.0, 1.0
    )


def largest_component_area(mask: np.ndarray) -> int:
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return 0
    return int(stats[1:, cv2.CC_STAT_AREA].max(initial=0))


def enclosed_holes(foreground: np.ndarray, minimum_y: int) -> np.ndarray:
    inverse = (~foreground).astype(np.uint8)
    count, labels, _, _ = cv2.connectedComponentsWithStats(inverse, connectivity=8)
    if count <= 1:
        return np.zeros_like(foreground)
    border_labels = np.unique(
        np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
    )
    holes = (labels != 0) & ~np.isin(labels, border_labels)
    holes[: max(0, minimum_y)] = False
    return holes


def hair_roi(
    shape: tuple[int, int], target_face_box: list[int]
) -> np.ndarray:
    height, width = shape
    x, y, face_width, face_height = (int(value) for value in target_face_box)
    left = max(0, int(round(x - 0.75 * face_width)))
    right = min(width, int(round(x + 1.75 * face_width)))
    top = max(0, int(round(y - 0.85 * face_height)))
    bottom = min(height, int(round(y + 2.25 * face_height)))
    roi = np.zeros(shape, dtype=bool)
    roi[top:bottom, left:right] = True
    return roi


def estimate_source_backdrop(source: np.ndarray) -> tuple[np.ndarray, float]:
    height, width = source.shape[:2]
    patch_height = max(4, int(round(height * 0.12)))
    patch_width = max(4, int(round(width * 0.12)))
    corners = np.concatenate(
        (
            source[:patch_height, :patch_width].reshape(-1, 3),
            source[:patch_height, -patch_width:].reshape(-1, 3),
        ),
        axis=0,
    ).astype(np.float32)
    backdrop = np.median(corners, axis=0)
    corner_distances = np.linalg.norm(corners - backdrop, axis=1)
    adaptive_noise = float(np.percentile(corner_distances, 99))
    return backdrop, adaptive_noise


def robust_z(values: list[float], minimum_scale: float) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    scale = max(minimum_scale, 1.4826 * mad)
    return ((array - median) / scale).tolist()


def decoded_spec(
    path: Path,
    target_size: tuple[int, int],
    minimum: int,
    maximum: int,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    violations: list[str] = []
    payload = path.read_bytes()
    byte_count = len(payload)
    if not minimum <= byte_count <= maximum:
        violations.append("file_size_out_of_range")
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            image_format = image.format
            image_mode = image.mode
            image_size = image.size
            output = np.asarray(image.convert("RGB"))
    except Exception as error:
        raise ValueError(f"Output cannot be decoded: {error}") from error
    if image_format != "JPEG":
        violations.append("format_not_jpeg")
    if image_mode != "RGB":
        violations.append("mode_not_rgb")
    if image_size != target_size:
        violations.append("dimensions_mismatch")
    return output, violations, {
        "bytes": byte_count,
        "format": image_format,
        "mode": image_mode,
        "width": image_size[0],
        "height": image_size[1],
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def measure_one(
    source_path: Path,
    output_path: Path,
    batch_entry: dict[str, Any],
    session: Any,
    config: dict[str, Any],
    args: argparse.Namespace,
) -> VisualEntry:
    target_size = tuple(int(value) for value in config["target_size"])
    background_rgb = tuple(int(value) for value in config["background_rgb"])
    minimum = int(config["min_bytes"])
    maximum = int(config["max_bytes"])
    short_edge = int(config["modnet_short_edge"])
    source_payload = source_path.read_bytes()
    source_hash = hashlib.sha256(source_payload).hexdigest()
    source = load_rgb_bytes(source_payload)
    crop = strict_crop(source, batch_entry.get("crop_box"))
    source_aligned = resize_rgb(crop, target_size)
    alpha, input_shape, inference_seconds = predict_alpha(session, crop, short_edge)
    matte = target_alpha(alpha, target_size)
    expected_image, _, _ = replace_background(
        crop,
        alpha,
        background_rgb,
        target_size,
    )
    expected_output = np.asarray(expected_image)
    output, violations, spec = decoded_spec(
        output_path, target_size, minimum, maximum
    )
    if output.shape[:2] != (target_size[1], target_size[0]):
        # Keep reporting possible even when dimensions are wrong.
        output = resize_rgb(output, target_size)

    if source_hash != batch_entry.get("source_sha256"):
        violations.append("source_hash_changed")
    if spec["sha256"] != batch_entry.get("sha256"):
        violations.append("output_hash_differs_from_batch_report")
    for field in ("bytes", "format", "mode", "width", "height"):
        if spec[field] != batch_entry.get(field):
            violations.append(f"{field}_differs_from_batch_report")
    if input_shape != batch_entry.get("model_input_shape"):
        violations.append("model_input_shape_differs_from_batch_report")

    background = np.asarray(background_rgb, dtype=np.float32)
    output_dist = np.linalg.norm(output.astype(np.float32) - background, axis=2)
    safe_background = (matte <= ALPHA_BACKGROUND_CUTOFF).astype(np.uint8)
    safe_background = cv2.erode(
        safe_background, np.ones((5, 5), np.uint8), iterations=1
    ).astype(bool)
    if int(np.count_nonzero(safe_background)) < 64:
        violations.append("insufficient_safe_background_for_qa")
        background_distances = np.asarray([float("inf")], dtype=np.float32)
    else:
        background_distances = output_dist[safe_background]
    background_p99 = float(np.percentile(background_distances, 99))
    background_far_fraction = float(
        np.mean(background_distances > args.background_tolerance)
    )
    if background_p99 > args.background_tolerance or background_far_fraction > 0.01:
        violations.append("decoded_background_outside_tolerance")

    foreground = matte >= 0.5
    ys, xs = np.where(foreground)
    target_face_box = batch_entry.get("target_face_box")
    if not isinstance(target_face_box, list) or len(target_face_box) != 4:
        raise ValueError("batch report target_face_box is missing or invalid")
    roi = hair_roi(foreground.shape, target_face_box)
    backdrop, corner_noise = estimate_source_backdrop(source_aligned)
    source_distance = np.linalg.norm(
        source_aligned.astype(np.float32) - backdrop, axis=2
    )
    separation = max(args.source_background_separation, corner_noise + 8.0)
    source_nonbackground = source_distance > separation
    edge_radius = max(2, int(round(min(foreground.shape) * 0.04)))
    near_subject = cv2.dilate(
        foreground.astype(np.uint8),
        np.ones((edge_radius * 2 + 1, edge_radius * 2 + 1), np.uint8),
        iterations=1,
    ).astype(bool)
    possible_loss = (
        (matte <= 0.05)
        & source_nonbackground
        & near_subject
        & roi
    )

    face_bottom = int(target_face_box[1] + target_face_box[3] * 0.80)
    holes = enclosed_holes(foreground, face_bottom)
    outer_edge = cv2.dilate(
        foreground.astype(np.uint8), np.ones((9, 9), np.uint8), iterations=1
    ).astype(bool) & ~foreground
    # Measure residual only where the continuous matte says "sure background".
    # The 0<alpha<0.5 fringe is a legitimate soft transition, not contamination.
    edge_background_probe = outer_edge & (matte <= ALPHA_BACKGROUND_CUTOFF)
    edge_probe_pixels = int(np.count_nonzero(edge_background_probe))
    edge_denominator = max(1, edge_probe_pixels)
    sure_edge_residual = edge_background_probe & (
        output_dist > args.background_tolerance
    )
    sure_edge_residual_fraction = (
        int(np.count_nonzero(sure_edge_residual)) / edge_denominator
    )
    soft_edge_probe = outer_edge & (matte > ALPHA_BACKGROUND_CUTOFF) & (matte < 0.5)
    soft_probe_pixels = int(np.count_nonzero(soft_edge_probe))
    decoded_expected_distance = np.linalg.norm(
        output.astype(np.float32) - expected_output.astype(np.float32), axis=2
    )
    soft_edge_residual = soft_edge_probe & (
        decoded_expected_distance > max(8.0, args.background_tolerance)
    )
    soft_edge_residual_fraction = int(np.count_nonzero(soft_edge_residual)) / max(
        1, soft_probe_pixels
    )
    edge_residual = sure_edge_residual | soft_edge_residual
    edge_residual_fraction = max(
        sure_edge_residual_fraction, soft_edge_residual_fraction
    )

    soft_edge_fraction = float(np.mean((matte > 0.05) & (matte < 0.95)))
    possible_pixels = int(np.count_nonzero(possible_loss))
    hole_pixels = int(np.count_nonzero(holes))
    image_area = matte.size
    visual_flags: list[str] = []
    if batch_entry.get("composition_verified") is not True:
        visual_flags.append("composition_unverified_crop_override")
    if possible_pixels >= max(4, int(math.ceil(image_area * args.possible_loss_fraction))):
        visual_flags.append("possible_missing_hair_or_edge_detail")
    if hole_pixels >= max(4, int(math.ceil(image_area * args.interior_hole_fraction))):
        visual_flags.append("possible_interior_matte_hole")
    if edge_residual_fraction > args.edge_residual_fraction:
        visual_flags.append("near_edge_background_residual")
    if edge_probe_pixels < max(8, int(round(image_area * 0.0001))):
        visual_flags.append("insufficient_near_edge_background_probe")
    if soft_edge_fraction < 0.00005:
        visual_flags.append("matte_has_almost_no_soft_edge")
    if not len(xs):
        visual_flags.append("no_foreground_detected")
    else:
        measured_head_top_ratio = float(ys.min() / matte.shape[0])
        if not (
            args.head_top_min_ratio
            <= measured_head_top_ratio
            <= args.head_top_max_ratio
        ):
            visual_flags.append("composition_top_margin_outlier")

    batch_alpha_mean = batch_entry.get("alpha_mean")
    alpha_mean_delta = (
        abs(float(matte.mean()) - float(batch_alpha_mean))
        if isinstance(batch_alpha_mean, (int, float))
        else None
    )
    if alpha_mean_delta is None or alpha_mean_delta > 0.002:
        violations.append("matte_forward_check_differs_from_batch")

    bbox = (
        [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        if len(xs)
        else None
    )
    metrics = {
        "name": output_path.name,
        **spec,
        "spec_violations": sorted(set(violations)),
        "model_input_shape": input_shape,
        "inference_seconds": round(float(inference_seconds), 4),
        "alpha_mean": round(float(matte.mean()), 6),
        "alpha_mean_delta_from_batch": (
            round(float(alpha_mean_delta), 6) if alpha_mean_delta is not None else None
        ),
        "soft_edge_fraction": round(soft_edge_fraction, 6),
        "foreground_fraction": round(float(foreground.mean()), 6),
        "subject_bbox": bbox,
        "head_top_ratio": (
            round(float(ys.min() / matte.shape[0]), 6) if len(ys) else None
        ),
        "decoded_background_distance_p99": (
            round(background_p99, 3) if math.isfinite(background_p99) else None
        ),
        "decoded_background_far_fraction": round(background_far_fraction, 6),
        "source_backdrop_rgb_estimate": np.round(backdrop, 3).tolist(),
        "source_backdrop_corner_noise_p99": round(corner_noise, 3),
        "source_background_separation_used": round(separation, 3),
        "possible_loss_pixels": possible_pixels,
        "largest_possible_loss_component": largest_component_area(possible_loss),
        "interior_hole_pixels": hole_pixels,
        "largest_interior_hole_component": largest_component_area(holes),
        "edge_residual_pixels": int(np.count_nonzero(edge_residual)),
        "edge_residual_fraction": round(edge_residual_fraction, 6),
        "sure_edge_probe_pixels": edge_probe_pixels,
        "sure_edge_residual_fraction": round(sure_edge_residual_fraction, 6),
        "soft_edge_probe_pixels": soft_probe_pixels,
        "soft_edge_residual_fraction": round(soft_edge_residual_fraction, 6),
        "soft_edge_expected_distance_p95": (
            round(
                float(np.percentile(decoded_expected_distance[soft_edge_probe], 95)),
                3,
            )
            if soft_probe_pixels
            else None
        ),
        "visual_flags": visual_flags,
    }
    return VisualEntry(
        metrics=metrics,
        source=source_aligned,
        output=output,
        alpha=matte,
        possible_loss=possible_loss,
        interior_holes=holes,
        edge_residual=edge_residual,
    )


def add_cohort_flags(entries: list[VisualEntry]) -> None:
    if len(entries) < 5:
        for entry in entries:
            entry.metrics["cohort_z_foreground_fraction"] = None
            entry.metrics["cohort_z_head_top_ratio"] = None
        return
    foreground_z = robust_z(
        [float(entry.metrics["foreground_fraction"]) for entry in entries], 0.005
    )
    head_values = [
        float(entry.metrics["head_top_ratio"] or 0.0) for entry in entries
    ]
    head_z = robust_z(head_values, 0.005)
    for index, entry in enumerate(entries):
        entry.metrics["cohort_z_foreground_fraction"] = round(foreground_z[index], 3)
        entry.metrics["cohort_z_head_top_ratio"] = round(head_z[index], 3)
        if max(abs(foreground_z[index]), abs(head_z[index])) >= 4.0:
            entry.metrics["visual_flags"].append("composition_cohort_outlier")


def color_overlay(entry: VisualEntry) -> np.ndarray:
    overlay = entry.output.astype(np.float32).copy()

    def paint(mask: np.ndarray, color: tuple[int, int, int], strength: float) -> None:
        if np.any(mask):
            overlay[mask] = (
                overlay[mask] * (1.0 - strength)
                + np.asarray(color, dtype=np.float32) * strength
            )

    paint(entry.possible_loss, (255, 145, 0), 0.72)
    paint(entry.interior_holes, (255, 25, 25), 0.82)
    paint(entry.edge_residual, (255, 0, 255), 0.76)
    return np.clip(np.rint(overlay), 0, 255).astype(np.uint8)


def fit_panel(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    copy = image.copy()
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    panel = Image.new("RGB", size, (245, 245, 245))
    panel.paste(copy, ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2))
    return panel


def make_contact_sheets(entries: list[VisualEntry], qa_dir: Path) -> list[Path]:
    ordered = sorted(
        entries,
        key=lambda item: (
            -len(item.metrics["spec_violations"]),
            -len(item.metrics["visual_flags"]),
            -item.metrics["possible_loss_pixels"],
            item.metrics["name"],
        ),
    )
    columns, rows = 1, 2
    tile_width, tile_height = 1060, 400
    per_page = columns * rows
    panel_size = (240, 320)
    outputs: list[Path] = []
    for page, start in enumerate(range(0, len(ordered), per_page), 1):
        sheet = Image.new(
            "RGB", (columns * tile_width, rows * tile_height), (230, 230, 230)
        )
        draw = ImageDraw.Draw(sheet)
        for offset, entry in enumerate(ordered[start : start + per_page]):
            row, column = divmod(offset, columns)
            x0, y0 = column * tile_width, row * tile_height
            draw.rectangle(
                (x0 + 3, y0 + 3, x0 + tile_width - 4, y0 + tile_height - 4),
                fill=(250, 250, 250),
                outline=(175, 175, 175),
                width=2,
            )
            index = start + offset + 1
            entry.metrics["contact_index"] = index
            entry.metrics["contact_page"] = page
            entry.metrics["contact_tile"] = offset + 1
            flags = entry.metrics["visual_flags"] + entry.metrics["spec_violations"]
            heading = f"#{index:03d} flags={','.join(flags) if flags else 'none'}"
            draw.text((x0 + 12, y0 + 10), heading[:105], fill=(25, 25, 25))
            alpha_rgb = np.repeat(
                np.rint(entry.alpha[..., None] * 255).astype(np.uint8), 3, axis=2
            )
            panels = (
                ("SOURCE", entry.source),
                ("ALPHA", alpha_rgb),
                ("OUTPUT", entry.output),
                ("OVERLAY", color_overlay(entry)),
            )
            for panel_index, (label, rgb) in enumerate(panels):
                px = x0 + 12 + panel_index * 260
                sheet.paste(fit_panel(Image.fromarray(rgb, "RGB"), panel_size), (px, y0 + 45))
                draw.text((px + 90, y0 + 372), label, fill=(35, 35, 35))
        path = qa_dir / f"contact_sheet_{page:02d}.png"
        sheet.save(path, format="PNG", optimize=True)
        outputs.append(path)
    return outputs


def validate_top_level(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path, Path, dict[str, Any]]:
    source_dir = args.source_dir.resolve(strict=True)
    output_dir = args.output_dir.resolve(strict=True)
    batch_report = args.batch_report.resolve(strict=True)
    model_path = args.model.resolve(strict=True)
    qa_dir = args.qa_dir.resolve(strict=False)
    if not source_dir.is_dir() or not output_dir.is_dir():
        raise NotADirectoryError("source-dir and output-dir must both be directories")
    if not batch_report.is_file() or not model_path.is_file():
        raise FileNotFoundError("batch-report and model must both be files")
    if qa_dir.exists():
        raise FileExistsError(f"qa-dir already exists; choose a new directory: {qa_dir}")
    if paths_overlap(source_dir, output_dir):
        raise ValueError("source-dir and output-dir must be separate and non-nested")
    if paths_overlap(qa_dir, source_dir) or paths_overlap(qa_dir, output_dir):
        raise ValueError("qa-dir must be separate from source-dir and output-dir")
    maximum_rgb_distance = math.sqrt(3 * 255**2)
    if args.background_tolerance is not None and (
        not math.isfinite(args.background_tolerance)
        or not 0 < args.background_tolerance <= maximum_rgb_distance
    ):
        raise ValueError("background-tolerance must be finite and within (0, 441.673]")
    if not math.isfinite(args.source_background_separation) or not (
        0 < args.source_background_separation <= maximum_rgb_distance
    ):
        raise ValueError(
            "source-background-separation must be finite and within (0, 441.673]"
        )
    fractions = (
        args.possible_loss_fraction,
        args.interior_hole_fraction,
        args.edge_residual_fraction,
        args.head_top_min_ratio,
        args.head_top_max_ratio,
    )
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in fractions):
        raise ValueError("Fraction thresholds must be finite and within [0, 1]")
    if args.head_top_min_ratio >= args.head_top_max_ratio:
        raise ValueError("Require head-top-min-ratio < head-top-max-ratio")
    raw = json.loads(batch_report.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1 or raw.get("status") != "pass":
        raise ValueError("batch-report must be a schema_version 1 passing report")
    qa_dir.parent.mkdir(parents=True, exist_ok=True)
    return source_dir, output_dir, batch_report, model_path, qa_dir, raw


def run(args: argparse.Namespace) -> int:
    source_dir, output_dir, batch_report_path, model_path, qa_dir, batch = (
        validate_top_level(args)
    )
    config = batch.get("config")
    if not isinstance(config, dict):
        raise ValueError("batch-report config is missing")
    required_config = {
        "target_size",
        "background_rgb",
        "min_bytes",
        "max_bytes",
        "background_tolerance",
        "modnet_short_edge",
    }
    if not required_config.issubset(config):
        raise ValueError("batch-report config lacks required fields")
    processing_tolerance = float(config["background_tolerance"])
    if not math.isfinite(processing_tolerance) or processing_tolerance <= 0:
        raise ValueError("batch-report background_tolerance is invalid")
    if args.background_tolerance is None:
        args.background_tolerance = processing_tolerance
    elif args.background_tolerance > processing_tolerance:
        raise ValueError(
            "QA background-tolerance may not be wider than the processing gate"
        )
    if sha256_file(model_path) != batch.get("model", {}).get("sha256"):
        raise ValueError("QA model hash differs from the processing model hash")
    sources = image_map(source_dir)
    outputs = image_map(output_dir)
    entries_raw = batch.get("files")
    if not isinstance(entries_raw, list):
        raise ValueError("batch-report files must be an array")
    batch_map = {
        entry.get("name"): entry
        for entry in entries_raw
        if isinstance(entry, dict) and isinstance(entry.get("name"), str)
    }
    if len(batch_map) != len(entries_raw):
        raise ValueError("batch-report contains duplicate or invalid file entries")
    if not batch_map:
        raise ValueError("batch-report contains no file entries")
    expected_names = set(batch_map)
    filename_violations: list[str] = []
    batch_counts = batch.get("counts")
    if not isinstance(batch_counts, dict):
        filename_violations.append("batch_report_counts_missing")
    else:
        if int(batch_counts.get("selected", -1)) != len(entries_raw):
            filename_violations.append("batch_selected_count_inconsistent")
        if int(batch_counts.get("processed", -1)) != len(entries_raw):
            filename_violations.append("batch_processed_count_inconsistent")
        if int(batch_counts.get("failed", -1)) != 0:
            filename_violations.append("batch_failed_count_nonzero")
        if int(batch_counts.get("discovered", -1)) != len(sources):
            filename_violations.append("batch_discovered_count_inconsistent")
    if batch.get("output_committed") is not True:
        filename_violations.append("batch_output_not_committed")
    if batch.get("failures") not in ([], None):
        filename_violations.append("batch_failures_not_empty")
    source_order = sorted(sources, key=lambda value: (value.casefold(), value))
    report_limit = int(config.get("limit", 0))
    if report_limit < 0:
        filename_violations.append("batch_limit_is_negative")
        report_limit = 0
    derived_names = source_order[:report_limit] if report_limit else source_order
    if expected_names != set(derived_names):
        filename_violations.append("batch_file_set_differs_from_deterministic_selection")
    expected_count = int(config.get("expected_count", 0))
    if expected_count and len(sources) != expected_count:
        filename_violations.append("source_count_differs_from_batch_expected_count")
    if set(outputs) != expected_names:
        filename_violations.append("output_filename_set_differs_from_batch_report")
    if not expected_names.issubset(sources):
        filename_violations.append("source_files_from_batch_report_are_missing")
    if int(config.get("limit", 0)) == 0 and set(sources) != expected_names:
        filename_violations.append("full_batch_source_filename_set_differs_from_report")
    unexpected_output_files = sorted(
        path.name
        for path in output_dir.iterdir()
        if path.is_file() and path.suffix.lower() not in IMAGE_SUFFIXES
    )
    if unexpected_output_files:
        filename_violations.append("output_directory_contains_non_jpeg_files")

    session = create_session(model_path)
    qa_temporary = tempfile.TemporaryDirectory(
        prefix=f".{qa_dir.name}.staging-", dir=qa_dir.parent
    )
    qa_staging = Path(qa_temporary.name)
    entries: list[VisualEntry] = []
    failures: list[dict[str, str]] = []
    for index, name in enumerate(sorted(expected_names, key=lambda value: (value.casefold(), value)), 1):
        print(f"[{index}/{len(expected_names)}] QA", flush=True)
        if name not in sources or name not in outputs:
            failures.append({"name": name, "error": "Missing source or output file"})
            continue
        try:
            entries.append(
                measure_one(
                    sources[name],
                    outputs[name],
                    batch_map[name],
                    session,
                    config,
                    args,
                )
            )
        except Exception as error:
            failures.append(
                {"name": name, "error": f"{type(error).__name__}: {error}"}
            )
    add_cohort_flags(entries)
    contact_sheets = make_contact_sheets(entries, qa_staging) if entries else []
    files = [
        entry.metrics
        for entry in sorted(entries, key=lambda item: (item.metrics["name"].casefold(), item.metrics["name"]))
    ]
    specification_failed_files = sum(bool(item["spec_violations"]) for item in files)
    specification_violation_total = sum(
        len(item["spec_violations"]) for item in files
    ) + len(filename_violations)
    visual_flagged_files = sum(bool(item["visual_flags"]) for item in files)
    visual_flag_total = sum(len(item["visual_flags"]) for item in files)
    if filename_violations or failures or specification_failed_files:
        status = "failed"
        exit_code = 1
        exit_reason = "specification_or_execution_failure"
    elif visual_flagged_files and not args.allow_visual_flags:
        status = "review_required"
        exit_code = 2
        exit_reason = "automatic_visual_flags_require_human_review"
    elif visual_flagged_files:
        status = "accepted_with_visual_flags"
        exit_code = 0
        exit_reason = "visual_flags_explicitly_allowed"
    else:
        status = "pass"
        exit_code = 0
        exit_reason = "automated_checks_passed_manual_contact_sheet_review_still_required"
    report = {
        "schema_version": 1,
        "status": status,
        "exit_code": exit_code,
        "exit_reason": exit_reason,
        "manual_review_required": True,
        "visual_flags_allowed": bool(args.allow_visual_flags),
        "privacy": {
            "inference": "local-only",
            "generative_model_used": False,
            "external_image_service_used": False,
            "absolute_paths_recorded": False,
        },
        "config": {
            "target_size": config["target_size"],
            "background_rgb": config["background_rgb"],
            "min_bytes": config["min_bytes"],
            "max_bytes": config["max_bytes"],
            "modnet_short_edge": config["modnet_short_edge"],
            "background_tolerance": args.background_tolerance,
            "source_background_separation": args.source_background_separation,
            "possible_loss_fraction": args.possible_loss_fraction,
            "interior_hole_fraction": args.interior_hole_fraction,
            "edge_residual_fraction": args.edge_residual_fraction,
            "head_top_min_ratio": args.head_top_min_ratio,
            "head_top_max_ratio": args.head_top_max_ratio,
        },
        "counts": {
            "batch_report_files": len(expected_names),
            "measured": len(files),
            "execution_failures": len(failures),
            "specification_failed_files": specification_failed_files,
            "specification_violations_total": specification_violation_total,
            "visual_flagged_files": visual_flagged_files,
            "visual_flags_total": visual_flag_total,
        },
        "filename_violations": filename_violations,
        "failures": failures,
        "contact_sheets": [path.name for path in contact_sheets],
        "overlay_legend": {
            "orange": "Source detail near the subject received near-zero alpha; possible missing hair or edge detail.",
            "red": "Enclosed low-alpha matte region below the face; possible clothing or subject hole.",
            "magenta": "Sure background is off-color or a soft edge differs from the alpha-aware expected composite.",
        },
        "limitations": [
            "Automatic visual flags are triage signals, not final acceptance decisions.",
            "Source backdrop texture can imitate missing edge detail.",
            "Legitimate gaps between hair, neck, and clothing can resemble matte holes.",
            "Cohort outlier checks cannot detect a systematic error shared by the full batch.",
            "Inspect every contact sheet at enlarged scale before delivery.",
        ],
        "files": files,
    }
    atomic_json(qa_staging / "qa_report.json", report)
    if qa_dir.exists():
        raise FileExistsError(
            "qa-dir appeared during QA; refusing to replace an existing directory"
        )
    os.rename(qa_staging, qa_dir)
    qa_temporary.cleanup()
    print(
        f"{status.upper()} measured={len(files)} "
        f"visual_flagged_files={visual_flagged_files} "
        f"report={qa_dir / 'qa_report.json'}",
        flush=True,
    )
    return exit_code


def main() -> int:
    args = build_parser().parse_args()
    try:
        return run(args)
    except Exception as error:
        print(f"FATAL: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
