"""Utility for loading and aligning semantic heatmap scores with STAMP features.

Supports three score sources (checked in order):
1. Direction-benchmark heatmap H5 files ({heatmap_dir}/{slide_name}.h5)
   with /x, /y, /pos, /neg, /scores/{label}
2. Inline masks in STAMP feature H5 files (/tile/{mask_name})
3. External .npy sidecar files ({heatmap_dir}/{slide_name}.npy)
"""

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
    heatmap_dir: Path | None = None,
    score_key: str = "pos",
    feature_h5_path: Path | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Load heatmap scores aligned to STAMP feature tile order.

    Args:
        slide_name: Slide identifier (stem of the H5 file, without extension).
        stamp_coords_um: STAMP tile coordinates in micrometers, shape (N, 2).
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
            aligned = _align_scores(stamp_coords_um, hm_x, hm_y, scores)
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

def _align_scores(
    stamp_coords_um: np.ndarray,
    hm_x: np.ndarray,
    hm_y: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    """Align heatmap scores to STAMP tile order by coordinate matching.

    If coordinates already match (same source, same order), this is a fast
    identity mapping.  Otherwise it builds a lookup from heatmap coordinates
    and permutes the scores to match STAMP's tile order.

    Uses the same approach as EAGLE's ``_align_vir2_to_ctp_by_coords``
    (eagle.py:267-300).
    """
    n_stamp = stamp_coords_um.shape[0]
    n_hm = len(hm_x)

    # Build heatmap coord pairs
    hm_coords = np.stack([hm_x, hm_y], axis=1)  # (M, 2)

    # Quick check: if shapes match and coords are close, assume same order
    if n_stamp == n_hm:
        if np.allclose(stamp_coords_um, hm_coords, atol=1.0, rtol=0):
            return scores.copy()

    # Round to avoid floating-point mismatches
    decimals = 1
    ref = np.round(stamp_coords_um.astype(np.float64), decimals)
    oth = np.round(hm_coords.astype(np.float64), decimals)

    # Build mapping: coordinate -> queue of indices (from heatmap)
    buckets: dict[tuple, deque] = defaultdict(deque)
    for j, key in enumerate(map(tuple, oth)):
        buckets[key].append(j)

    aligned_scores = np.full(n_stamp, np.nan, dtype=np.float32)
    matched = 0
    for i, key in enumerate(map(tuple, ref)):
        if buckets[key]:
            aligned_scores[i] = scores[buckets[key].popleft()]
            matched += 1

    if matched == 0:
        raise ValueError(
            f"No coordinate matches found between STAMP ({n_stamp} tiles) and "
            f"heatmap ({n_hm} tiles). Coordinates may use different units "
            f"(STAMP: um, heatmap: px). Check that WSIVL h5_dir pointed to "
            f"STAMP feature files."
        )

    if matched < n_stamp:
        _logger.warning(
            "Heatmap alignment: %d/%d STAMP tiles matched. "
            "Unmatched tiles get score=0.",
            matched, n_stamp,
        )
        aligned_scores = np.nan_to_num(aligned_scores, nan=0.0)

    return aligned_scores


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
