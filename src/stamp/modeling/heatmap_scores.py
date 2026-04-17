"""Utility for loading and aligning semantic heatmap scores with STAMP features.

Supports three score sources (checked in order):
1. Direction-benchmark heatmap H5 files ({heatmap_dir}/{slide_name}.h5)
   with /x, /y, /pos, /neg, /scores/{label}
2. Inline masks in STAMP feature H5 files (/tile/{mask_name})
3. External .npy sidecar files ({heatmap_dir}/{slide_name}.npy)

Coordinate handling:
    STAMP stores tile coordinates in **micrometers**.  WSIVL heatmap H5 files
    may store coordinates in **level-0 pixels** (when WSIVL was pointed at its
    own embedding H5 files rather than STAMP feature files).  This module
    auto-detects the unit mismatch by comparing tile strides and converts
    pixel coordinates to micrometers using the MPP from STAMP metadata.

Diagnostics:
    When ``diagnostics_dir`` is passed, a per-slide CSV is written with
    every STAMP tile coordinate, the matched heatmap coordinate (if any),
    and the assigned score.  A summary CSV is also written listing matched /
    unmatched counts per slide.
"""

import csv
import logging
from collections import defaultdict, deque
from pathlib import Path

import h5py
import numpy as np

_logger = logging.getLogger("stamp")


def load_heatmap_scores(
    *,
    slide_name: str,
    stamp_coords_um: np.ndarray,
    tile_size_um: float,
    tile_size_px: int | None = None,
    heatmap_dir: Path | None = None,
    score_key: str = "pos",
    feature_h5_path: Path | None = None,
    normalize: bool = True,
    diagnostics_dir: Path | None = None,
) -> np.ndarray:
    """Load heatmap scores aligned to STAMP feature tile order.

    Args:
        slide_name: Slide identifier (stem of the H5 file, without extension).
        stamp_coords_um: STAMP tile coordinates in micrometers, shape (N, 2).
        tile_size_um: Tile size in micrometers (from STAMP H5 metadata).
        tile_size_px: Tile size in pixels (from STAMP H5 metadata).  Required
            for automatic pixel-to-micrometer conversion.
        heatmap_dir: Directory containing heatmap H5 or .npy files.
        score_key: Which score to read:

            - ``"pos"``, ``"neg"`` — root-level datasets in the heatmap H5
            - A class label found under ``/scores/{label}``
            - ``"max_class"`` — per-tile maximum across **all** classes in
              ``/scores/`` (plus ``/pos`` and ``/neg`` if present).  This
              highlights diagnostically relevant tiles regardless of which
              class they belong to.

            For inline masks this is the dataset name prefix under ``/tile/``.
        feature_h5_path: Path to the STAMP feature H5 file (for inline mask
            fallback).
        normalize: If ``True``, normalise raw scores to [0, 1] via min-max.

    Returns:
        (N,) float32 array of scores aligned to ``stamp_coords_um`` order.
    """
    n_tiles = stamp_coords_um.shape[0]

    # --- 1. Direction-benchmark heatmap H5 ---
    if heatmap_dir is not None:
        if score_key == "max_class":
            result = _load_max_class_scores(heatmap_dir, slide_name)
        else:
            result = _load_direction_heatmap_h5(heatmap_dir, slide_name, score_key)
        if result is not None:
            scores, hm_x, hm_y = result
            aligned = _align_scores(
                stamp_coords_um, hm_x, hm_y, scores,
                tile_size_um=tile_size_um,
                tile_size_px=tile_size_px,
                slide_name=slide_name,
                diagnostics_dir=diagnostics_dir,
            )
            return _normalize(aligned) if normalize else aligned

    # --- 2. Inline mask in STAMP H5 ---
    if feature_h5_path is not None:
        inline = _load_inline_scores(feature_h5_path, score_key)
        if inline is not None:
            if len(inline) != n_tiles:
                _logger.warning(
                    "Inline mask length %d != tile count %d for %s, skipping",
                    len(inline), n_tiles, slide_name,
                )
            else:
                return _normalize(inline) if normalize else inline

    # --- 3. External .npy ---
    if heatmap_dir is not None:
        npy = _load_npy_scores(heatmap_dir, slide_name)
        if npy is not None:
            if len(npy) != n_tiles:
                _logger.warning(
                    ".npy score length %d != tile count %d for %s, skipping",
                    len(npy), n_tiles, slide_name,
                )
            else:
                return _normalize(npy) if normalize else npy

    raise FileNotFoundError(
        f"No heatmap scores found for slide '{slide_name}'. "
        f"Searched: heatmap_dir={heatmap_dir}, feature_h5={feature_h5_path}"
    )


# ---------------------------------------------------------------------------
# Source-specific loaders
# ---------------------------------------------------------------------------

def _load_direction_heatmap_h5(
    heatmap_dir: Path,
    slide_name: str,
    score_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load scores + coordinates from a direction-benchmark heatmap H5.

    Returns ``(scores, x, y)`` or ``None`` if the file doesn't exist.
    """
    h5_path = heatmap_dir / f"{slide_name}.h5"
    if not h5_path.exists():
        return None

    with h5py.File(str(h5_path), "r") as f:
        if "x" not in f or "y" not in f:
            _logger.warning("Heatmap H5 %s missing /x or /y datasets", h5_path)
            return None

        x = f["x"][:].astype(np.float64)
        y = f["y"][:].astype(np.float64)

        # Try /scores/{score_key} first, then root /{score_key}
        scores = None
        if "scores" in f and score_key in f["scores"]:
            scores = f["scores"][score_key][:].astype(np.float32)
        elif score_key in f:
            scores = f[score_key][:].astype(np.float32)

        if scores is None:
            available = []
            if "scores" in f:
                available.extend(f"scores/{k}" for k in f["scores"])
            available.extend(k for k in f if k not in ("x", "y", "scores"))
            raise KeyError(
                f"score_key '{score_key}' not found in {h5_path}. "
                f"Available: {available}"
            )

    return scores, x, y


def _load_max_class_scores(
    heatmap_dir: Path,
    slide_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load all class scores and return per-tile maximum.

    For each tile, takes ``max(score_class1, score_class2, ...)``.
    This highlights diagnostically relevant tiles regardless of which class
    they belong to — stroma and artifacts (low in all classes) get suppressed.
    """
    h5_path = heatmap_dir / f"{slide_name}.h5"
    if not h5_path.exists():
        return None

    with h5py.File(str(h5_path), "r") as f:
        if "x" not in f or "y" not in f:
            _logger.warning("Heatmap H5 %s missing /x or /y datasets", h5_path)
            return None

        x = f["x"][:].astype(np.float64)
        y = f["y"][:].astype(np.float64)

        # Collect all available class scores
        all_scores: list[np.ndarray] = []

        # From /scores/ group
        if "scores" in f:
            for label in f["scores"]:
                all_scores.append(f["scores"][label][:].astype(np.float32))

        # From root /pos, /neg (if not already covered by /scores/)
        if not all_scores:
            for key in ("pos", "neg"):
                if key in f:
                    all_scores.append(f[key][:].astype(np.float32))

        if not all_scores:
            available = list(f.keys())
            raise KeyError(
                f"No class scores found in {h5_path}. Available keys: {available}"
            )

        # Per-tile maximum across all classes
        stacked = np.stack(all_scores, axis=0)  # (n_classes, N)
        max_scores = stacked.max(axis=0)  # (N,)

    return max_scores, x, y


def _load_inline_scores(
    feature_h5_path: Path,
    mask_name: str,
) -> np.ndarray | None:
    """Load a semantic mask from inside the STAMP feature H5 file."""
    with h5py.File(str(feature_h5_path), "r") as f:
        tile_grp = f.get("tile")
        if tile_grp is None:
            return None
        # Exact match first
        if mask_name in tile_grp:
            return tile_grp[mask_name][:].astype(np.float32)
        # Prefix match (e.g. "auto_semantic_mask" matches "auto_semantic_mask_slide_a")
        for key in tile_grp:
            if key.startswith(mask_name):
                return tile_grp[key][:].astype(np.float32)
    return None


def _load_npy_scores(
    heatmap_dir: Path,
    slide_name: str,
) -> np.ndarray | None:
    """Load scores from a standalone .npy sidecar file."""
    npy_path = heatmap_dir / f"{slide_name}.npy"
    if npy_path.exists():
        return np.load(str(npy_path)).astype(np.float32)
    return None


# ---------------------------------------------------------------------------
# Coordinate alignment
# ---------------------------------------------------------------------------

def _detect_and_convert_coords(
    stamp_coords_um: np.ndarray,
    hm_x: np.ndarray,
    hm_y: np.ndarray,
    tile_size_um: float,
    tile_size_px: int | None,
) -> np.ndarray:
    """Detect if heatmap coordinates are in pixels and convert to micrometers.

    Detection logic:
        Compute the median tile stride (spacing between adjacent tiles) in both
        coordinate systems.  STAMP's stride should be close to ``tile_size_um``.
        If the heatmap stride is close to ``tile_size_px`` instead, the heatmap
        coordinates are in level-0 pixels and need conversion via MPP.

    Returns:
        (M, 2) float64 array of heatmap coordinates in **micrometers**.
    """
    hm_coords = np.stack([hm_x, hm_y], axis=1)  # (M, 2)

    # Compute heatmap tile stride from coordinate spacing
    hm_stride = _estimate_stride(hm_x, hm_y)
    stamp_stride = _estimate_stride(
        stamp_coords_um[:, 0], stamp_coords_um[:, 1],
    )

    if hm_stride is None or stamp_stride is None:
        # Single tile or unable to determine — assume same unit
        return hm_coords

    # Check if heatmap stride matches STAMP stride (both in um)
    if abs(hm_stride - stamp_stride) / max(stamp_stride, 1e-8) < 0.1:
        _logger.debug("Heatmap coords appear to be in micrometers (stride=%.1f)", hm_stride)
        return hm_coords

    # Check if heatmap stride matches tile_size_px (coords in pixels)
    if tile_size_px is not None and tile_size_px > 0:
        if abs(hm_stride - tile_size_px) / max(tile_size_px, 1e-8) < 0.1:
            mpp = tile_size_um / tile_size_px
            _logger.info(
                "Heatmap coords detected as level-0 pixels (stride=%.0f px, "
                "tile_size_px=%d). Converting to micrometers with MPP=%.4f",
                hm_stride, tile_size_px, mpp,
            )
            return hm_coords * mpp

    # Fallback: try to infer conversion factor from stride ratio
    ratio = stamp_stride / hm_stride
    # If ratio is close to a plausible MPP (0.1 - 10.0), use it
    if 0.05 < ratio < 20.0 and abs(ratio - 1.0) > 0.1:
        _logger.info(
            "Heatmap coords appear to use different units (stride=%.1f vs "
            "STAMP=%.1f). Converting with inferred factor=%.4f",
            hm_stride, stamp_stride, ratio,
        )
        return hm_coords * ratio

    return hm_coords


def _estimate_stride(x: np.ndarray, y: np.ndarray) -> float | None:
    """Estimate the tile stride from coordinate arrays.

    Returns the median spacing between adjacent unique coordinate values,
    or ``None`` if there are fewer than 2 unique values in both axes.
    """
    diffs = []
    for vals in (x, y):
        unique = np.sort(np.unique(vals))
        if len(unique) >= 2:
            diffs.append(np.median(np.diff(unique)))
    if not diffs:
        return None
    return float(np.min(diffs))


_alignment_summary: list[dict] = []


def _align_scores(
    stamp_coords_um: np.ndarray,
    hm_x: np.ndarray,
    hm_y: np.ndarray,
    scores: np.ndarray,
    *,
    tile_size_um: float,
    tile_size_px: int | None,
    slide_name: str = "",
    diagnostics_dir: Path | None = None,
) -> np.ndarray:
    """Align heatmap scores to STAMP tile order by coordinate matching.

    Automatically detects if heatmap coordinates are in pixels and converts
    them to micrometers before matching.
    """
    # Step 1: detect unit and convert heatmap coords to micrometers
    hm_coords_um = _detect_and_convert_coords(
        stamp_coords_um, hm_x, hm_y,
        tile_size_um=tile_size_um,
        tile_size_px=tile_size_px,
    )

    n_stamp = stamp_coords_um.shape[0]
    n_hm = hm_coords_um.shape[0]

    # Step 2: fast path — same count + coordinates match after conversion
    if n_stamp == n_hm:
        if np.allclose(stamp_coords_um, hm_coords_um, atol=1.0, rtol=0):
            if diagnostics_dir:
                _write_diagnostics(
                    diagnostics_dir, slide_name, stamp_coords_um,
                    hm_coords_um, scores, matched_mask=np.ones(n_stamp, dtype=bool),
                    aligned_scores=scores.copy(),
                )
                _alignment_summary.append({
                    "slide": slide_name, "stamp_tiles": n_stamp,
                    "heatmap_tiles": n_hm, "matched": n_stamp, "unmatched": 0,
                })
            return scores.copy()

    # Step 3: coordinate-based lookup with tolerance
    decimals = 0  # round to nearest integer micrometer
    ref = np.round(stamp_coords_um.astype(np.float64), decimals)
    oth = np.round(hm_coords_um.astype(np.float64), decimals)

    # Build mapping: coordinate -> queue of indices (from heatmap)
    buckets: dict[tuple, deque] = defaultdict(deque)
    for j, key in enumerate(map(tuple, oth)):
        buckets[key].append(j)

    aligned_scores = np.full(n_stamp, np.nan, dtype=np.float32)
    matched_mask = np.zeros(n_stamp, dtype=bool)
    matched_hm_indices = np.full(n_stamp, -1, dtype=np.int64)
    matched = 0
    for i, key in enumerate(map(tuple, ref)):
        if buckets[key]:
            j = buckets[key].popleft()
            aligned_scores[i] = scores[j]
            matched_mask[i] = True
            matched_hm_indices[i] = j
            matched += 1

    # Partial match: unmatched STAMP tiles get score=0 (suppressed)
    aligned_scores = np.nan_to_num(aligned_scores, nan=0.0)

    if matched == 0:
        _logger.warning(
            "[%s] Heatmap alignment: 0/%d STAMP tiles matched. "
            "All tiles get score=0. "
            "STAMP stride ≈ %s, heatmap stride ≈ %s, "
            "tile_size_um=%.1f, tile_size_px=%s.",
            slide_name, n_stamp,
            _estimate_stride(stamp_coords_um[:, 0], stamp_coords_um[:, 1]),
            _estimate_stride(hm_x, hm_y),
            tile_size_um, tile_size_px,
        )
    elif matched < n_stamp:
        _logger.warning(
            "[%s] Heatmap alignment: %d/%d STAMP tiles matched. "
            "%d unmatched tiles get score=0.",
            slide_name, matched, n_stamp, n_stamp - matched,
        )
    else:
        _logger.debug("[%s] Heatmap alignment: all %d tiles matched.", slide_name, n_stamp)

    # Diagnostics export
    if diagnostics_dir:
        _write_diagnostics(
            diagnostics_dir, slide_name, stamp_coords_um,
            hm_coords_um, scores, matched_mask=matched_mask,
            aligned_scores=aligned_scores, matched_hm_indices=matched_hm_indices,
        )
        _alignment_summary.append({
            "slide": slide_name, "stamp_tiles": n_stamp,
            "heatmap_tiles": n_hm, "matched": matched,
            "unmatched": n_stamp - matched,
        })

    return aligned_scores


def _write_diagnostics(
    diagnostics_dir: Path,
    slide_name: str,
    stamp_coords: np.ndarray,
    hm_coords: np.ndarray,
    raw_hm_scores: np.ndarray,
    matched_mask: np.ndarray,
    aligned_scores: np.ndarray,
    matched_hm_indices: np.ndarray | None = None,
) -> None:
    """Write a per-tile CSV showing match status for a single slide."""
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    safe_name = slide_name.replace("/", "_").replace("\\", "_")
    csv_path = diagnostics_dir / f"{safe_name}_alignment.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "tile_idx", "stamp_x_um", "stamp_y_um",
            "matched", "hm_x_um", "hm_y_um", "score",
        ])
        for i in range(len(stamp_coords)):
            if matched_mask[i] and matched_hm_indices is not None and matched_hm_indices[i] >= 0:
                j = matched_hm_indices[i]
                writer.writerow([
                    i, f"{stamp_coords[i, 0]:.1f}", f"{stamp_coords[i, 1]:.1f}",
                    "yes", f"{hm_coords[j, 0]:.1f}", f"{hm_coords[j, 1]:.1f}",
                    f"{aligned_scores[i]:.6f}",
                ])
            else:
                writer.writerow([
                    i, f"{stamp_coords[i, 0]:.1f}", f"{stamp_coords[i, 1]:.1f}",
                    "no", "", "", f"{aligned_scores[i]:.6f}",
                ])


def write_alignment_summary(diagnostics_dir: Path) -> None:
    """Write a summary CSV with per-slide matched/unmatched counts.

    Call this once after all slides have been processed (e.g. at end of
    training setup or after encode_slides).
    """
    if not _alignment_summary:
        return
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    csv_path = diagnostics_dir / "alignment_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "slide", "stamp_tiles", "heatmap_tiles", "matched", "unmatched",
        ])
        writer.writeheader()
        writer.writerows(_alignment_summary)
    _logger.info("Alignment summary written to %s (%d slides)", csv_path, len(_alignment_summary))
    _alignment_summary.clear()


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize(scores: np.ndarray) -> np.ndarray:
    """Min-max normalise scores to [0, 1]."""
    s_min = scores.min()
    s_max = scores.max()
    if s_max - s_min < 1e-8:
        return np.ones_like(scores)
    return ((scores - s_min) / (s_max - s_min)).astype(np.float32)
