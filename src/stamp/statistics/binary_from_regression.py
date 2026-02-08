"""Evaluate a regression model as a binary classifier.

Minimal helpers that wrap *existing* inference output (a continuous score
per sample) so that it can be assessed against binary ground-truth labels.

Typical usage::

    from stamp.statistics.binary_from_regression import (
        binarize_labels,
        extract_score,
        postprocess_score,
        aggregate_patient_scores,
        get_thresholds,
        evaluate_thresholds,
    )

    # 1. Prepare DataFrame with columns: patient_id, y_raw, y_score
    # 2. Convert labels
    y_true = binarize_labels(df["y_raw"], positive_class="pos")
    # 3. Aggregate to patient-level (optional)
    df = aggregate_patient_scores(df)
    # 4. Compute thresholds & evaluate
    thresholds = get_thresholds(y_true, df["y_score"].values)
    summary_df, preds_df = evaluate_thresholds(df, thresholds=thresholds)
"""

from __future__ import annotations

import logging
import warnings
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

__author__ = "STAMP contributors"
__license__ = "MIT"

_logger = logging.getLogger("stamp")

# ── Common positive/negative conventions ────────────────────────────────
_POSITIVE_CONVENTIONS: dict[str, set[str]] = {
    "pos": {"pos", "positive"},
    "yes": {"yes"},
    "true": {"true"},
    "1": {"1"},
}
_NEGATIVE_CONVENTIONS: dict[str, set[str]] = {
    "neg": {"neg", "negative"},
    "no": {"no"},
    "false": {"false"},
    "0": {"0"},
}


def _infer_positive_label(labels: set[str]) -> str:
    """Try to automatically identify the positive label from common conventions."""
    for _key, pos_set in _POSITIVE_CONVENTIONS.items():
        for lbl in labels:
            if lbl in pos_set:
                return lbl
    raise ValueError(
        f"Cannot infer which of {labels} is the positive class. "
        "Please provide positive_class explicitly."
    )


# ── 1. Label binarization ──────────────────────────────────────────────


def binarize_labels(
    y_raw: Any,
    *,
    positive_class: str | None = None,
    negative_class: str | None = None,
    normalize_strings: bool = True,
) -> npt.NDArray[np.int_]:
    """Convert ground-truth labels to ``{0, 1}`` integer array.

    Parameters
    ----------
    y_raw
        Array-like of labels.  Accepted types: bool, int/float in {0,1},
        or strings (e.g. ``"pos"``/``"neg"``).
    positive_class
        Which value maps to **1**.  Required when the automatic convention
        matching cannot determine it.
    negative_class
        Which value maps to **0** (optional, for extra safety).
    normalize_strings
        If *True*, string labels are lower-cased and stripped before
        comparison.

    Returns
    -------
    numpy.ndarray
        Integer array with values in ``{0, 1}``.
    """
    arr = np.asarray(y_raw)

    # ── booleans ────────────────────────────────────────────────────────
    if arr.dtype == np.bool_:
        return arr.astype(int)

    # ── numeric already in {0, 1} ──────────────────────────────────────
    if np.issubdtype(arr.dtype, np.number):
        unique = set(np.unique(arr))
        if unique <= {0, 1, 0.0, 1.0}:
            return arr.astype(int)
        raise ValueError(
            f"Numeric labels contain values outside {{0, 1}}: {unique}. "
            "Provide string labels or preprocess to 0/1."
        )

    # ── strings / categorical ──────────────────────────────────────────
    str_arr = np.array([str(v) for v in arr])
    if normalize_strings:
        str_arr = np.array([s.strip().lower() for s in str_arr])
        if positive_class is not None:
            positive_class = positive_class.strip().lower()
        if negative_class is not None:
            negative_class = negative_class.strip().lower()

    unique_labels = set(str_arr)
    if len(unique_labels) != 2:
        raise ValueError(
            f"Expected exactly 2 unique labels for binary evaluation, "
            f"got {len(unique_labels)}: {unique_labels}"
        )

    if positive_class is not None:
        if positive_class not in unique_labels:
            raise ValueError(
                f"positive_class='{positive_class}' not found in labels {unique_labels}"
            )
        return (str_arr == positive_class).astype(int)

    if negative_class is not None:
        if negative_class not in unique_labels:
            raise ValueError(
                f"negative_class='{negative_class}' not found in labels {unique_labels}"
            )
        return (str_arr != negative_class).astype(int)

    # Try auto-inference from common conventions
    pos_label = _infer_positive_label(unique_labels)
    return (str_arr == pos_label).astype(int)


# ── 2. Score extraction ────────────────────────────────────────────────


def extract_score(
    model_output: Any,
    *,
    task_name: str | None = None,
    task_index: int | None = None,
) -> npt.NDArray[np.float64]:
    """Extract a 1-D score array from various model output formats.

    Parameters
    ----------
    model_output
        Scalar, 1-D array, dict, tuple/list, or 2-D array.
    task_name
        Key when *model_output* is a ``dict``.
    task_index
        Index when *model_output* is a tuple/list or 2-D array.
    """
    # scalar → 1-D
    if np.isscalar(model_output):
        return np.array([float(model_output)])

    # dict
    if isinstance(model_output, dict):
        if task_name is not None:
            return np.asarray(model_output[task_name], dtype=float).ravel()
        if len(model_output) == 1:
            return np.asarray(next(iter(model_output.values())), dtype=float).ravel()
        raise ValueError(
            f"model_output is a dict with {len(model_output)} keys "
            f"({list(model_output.keys())}). Provide task_name."
        )

    # tuple / list
    if isinstance(model_output, (tuple, list)):
        if task_index is not None:
            return np.asarray(model_output[task_index], dtype=float).ravel()
        if len(model_output) == 1:
            return np.asarray(model_output[0], dtype=float).ravel()
        raise ValueError(
            f"model_output is a sequence of length {len(model_output)}. "
            "Provide task_index."
        )

    # ndarray
    arr = np.asarray(model_output, dtype=float)
    if arr.ndim <= 1:
        return arr.ravel()
    if arr.ndim == 2:
        if task_index is not None:
            return arr[:, task_index]
        if arr.shape[1] == 1:
            return arr[:, 0]
        raise ValueError(
            f"model_output has shape {arr.shape}. Provide task_index."
        )
    raise ValueError(f"Unsupported model_output shape: {arr.shape}")


# ── 3. Post-processing ────────────────────────────────────────────────


def postprocess_score(
    scores: Any,
    *,
    mode: str | Callable = "identity",
    clip: bool = True,
) -> npt.NDArray[np.float64]:
    """Map raw model scores into [0, 1].

    Parameters
    ----------
    scores
        Array-like of floats.
    mode
        ``"identity"`` (no-op), ``"sigmoid"`` (logistic), or a callable.
    clip
        If *True*, clip output to [0, 1] and warn when values were
        outside that range.
    """
    arr = np.asarray(scores, dtype=float)

    if not np.all(np.isfinite(arr)):
        raise ValueError("scores contain non-finite values (NaN / Inf)")

    if callable(mode):
        arr = mode(arr)
    elif mode == "sigmoid":
        arr = 1.0 / (1.0 + np.exp(-arr))
    elif mode != "identity":
        raise ValueError(f"Unknown mode '{mode}'. Use 'identity', 'sigmoid', or a callable.")

    if clip:
        out_of_range = (arr < 0.0) | (arr > 1.0)
        if np.any(out_of_range):
            warnings.warn(
                f"{int(out_of_range.sum())} score(s) were outside [0, 1] and have been clipped.",
                stacklevel=2,
            )
            arr = np.clip(arr, 0.0, 1.0)

    return arr


# ── 4. Patient-level aggregation ───────────────────────────────────────


def aggregate_patient_scores(
    df: pd.DataFrame,
    *,
    id_col: str = "patient_id",
    score_col: str = "y_score",
    label_col: str = "y_raw",
    method: str = "mean",
) -> pd.DataFrame:
    """Aggregate multi-row-per-patient data to one row per patient.

    Parameters
    ----------
    df
        Must contain *id_col*, *score_col*, and *label_col*.
    method
        Aggregation for *score_col*: ``"mean"``, ``"median"``, or ``"max"``.

    Returns
    -------
    pandas.DataFrame
        With columns ``[id_col, label_col, score_col]``, one row per patient.

    Raises
    ------
    ValueError
        If a patient has inconsistent ground-truth labels.
    """
    if id_col not in df.columns:
        raise ValueError(f"Column '{id_col}' not found. Skipping aggregation.")

    # Check label consistency per patient
    label_nunique = df.groupby(id_col)[label_col].nunique()
    inconsistent = label_nunique[label_nunique > 1]
    if len(inconsistent) > 0:
        raise ValueError(
            f"Patients with inconsistent labels: {list(inconsistent.index[:5])}"
        )

    agg_fn = {"mean": "mean", "median": "median", "max": "max"}
    if method not in agg_fn:
        raise ValueError(f"method must be one of {list(agg_fn)}, got '{method}'")

    grouped = df.groupby(id_col, sort=False)
    out = pd.DataFrame(
        {
            id_col: grouped[label_col].first().index,
            label_col: grouped[label_col].first().values,
            score_col: grouped[score_col].agg(agg_fn[method]).values,
        }
    )
    return out


# ── 5. Threshold strategies ────────────────────────────────────────────


def get_thresholds(
    y_true: npt.NDArray[np.int_],
    y_score: npt.NDArray[np.float64],
    *,
    include: tuple[str, ...] = ("youden", "mean_score", "median_score"),
    fixed_thresholds: tuple[float, ...] = (0.5,),
    quantiles: tuple[float, ...] = (0.25, 0.75),
) -> OrderedDict[str, float]:
    """Compute a set of candidate thresholds.

    Parameters
    ----------
    y_true
        Binary ground-truth array (0/1).
    y_score
        Continuous scores in [0, 1].
    include
        Which built-in strategies to include.
    fixed_thresholds
        Literal threshold values.
    quantiles
        Score quantiles to include.

    Returns
    -------
    OrderedDict[str, float]
        ``{strategy_name: threshold_value}``.
    """
    thresholds: OrderedDict[str, float] = OrderedDict()

    if "youden" in include:
        fpr, tpr, thr = roc_curve(y_true, y_score)
        youden_j = tpr - fpr
        # Among ties, pick the smallest threshold (documented behavior)
        best_idx = int(np.where(youden_j == youden_j.max())[0][0])
        thresholds["youden"] = float(np.clip(thr[best_idx], 0.0, 1.0))

    if "mean_score" in include:
        thresholds["mean_score"] = float(np.clip(np.mean(y_score), 0.0, 1.0))

    if "median_score" in include:
        thresholds["median_score"] = float(np.clip(np.median(y_score), 0.0, 1.0))

    for t in fixed_thresholds:
        thresholds[f"fixed_{t:.2f}"] = float(np.clip(t, 0.0, 1.0))

    for q in quantiles:
        thresholds[f"q{int(q * 100):02d}"] = float(
            np.clip(np.quantile(y_score, q), 0.0, 1.0)
        )

    return thresholds


# ── 6. Single-threshold evaluation ─────────────────────────────────────


def evaluate_at_threshold(
    y_true: npt.NDArray[np.int_],
    y_score: npt.NDArray[np.float64],
    threshold: float,
    *,
    auroc: float | None = None,
) -> dict[str, Any]:
    """Compute binary classification metrics at a single threshold.

    The prediction rule is ``y_pred = (y_score > threshold)``.

    Parameters
    ----------
    y_true, y_score
        Ground-truth (0/1) and continuous scores.
    threshold
        Decision boundary.
    auroc
        Pre-computed AUROC (threshold-independent); computed here if *None*.
    """
    y_pred = (y_score > threshold).astype(int)

    if auroc is None:
        try:
            auroc = float(roc_auc_score(y_true, y_score))
        except ValueError:
            auroc = float("nan")

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    return {
        "threshold": threshold,
        "accuracy": acc,
        "f1": f1,
        "auroc": auroc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


# ── 7. Multi-threshold evaluation ──────────────────────────────────────


def evaluate_thresholds(
    df: pd.DataFrame,
    *,
    y_col: str = "y_true",
    score_col: str = "y_score",
    id_col: str | None = None,
    thresholds: OrderedDict[str, float] | dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate multiple threshold strategies.

    Parameters
    ----------
    df
        Must contain *y_col* (binary 0/1) and *score_col*.
    thresholds
        ``{strategy_name: threshold_value}`` from :func:`get_thresholds`.
        Computed automatically when *None*.
    id_col
        Optional patient/sample identifier column to include in preds_df.

    Returns
    -------
    summary_df
        One row per strategy with columns:
        ``strategy, threshold, accuracy, f1, auroc, sensitivity, specificity, tp, tn, fp, fn``.
    preds_df
        Original data with an extra ``y_pred_<strategy>`` column per strategy.
    """
    y_true = np.asarray(df[y_col], dtype=int)
    y_score = np.asarray(df[score_col], dtype=float)

    if thresholds is None:
        thresholds = get_thresholds(y_true, y_score)

    # Pre-compute AUROC once (threshold-independent)
    try:
        auroc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auroc = float("nan")

    rows: list[dict[str, Any]] = []
    preds_df = df[[c for c in [id_col, y_col, score_col] if c is not None and c in df.columns]].copy()

    for name, thr in thresholds.items():
        metrics = evaluate_at_threshold(y_true, y_score, thr, auroc=auroc)
        metrics["strategy"] = name
        rows.append(metrics)
        preds_df[f"y_pred_{name}"] = (y_score > thr).astype(int)

    summary_df = pd.DataFrame(rows)[
        ["strategy", "threshold", "accuracy", "f1", "auroc",
         "sensitivity", "specificity", "tp", "tn", "fp", "fn"]
    ]
    return summary_df, preds_df
