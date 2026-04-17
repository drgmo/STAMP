"""Out-of-fold (OOF) aggregation of cross-validation predictions.

For K-fold cross-validation, each patient appears in exactly one test
fold.  Concatenating all fold predictions yields an OOF prediction set
covering the full dataset, on which a single global evaluation can be
performed — typically more stable and interpretable than averaging
fold-level metrics.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from stamp.statistics.extended_categorical import (
    _compute_confusion_matrix,
    _compute_fold_metrics,
    _compute_per_class_metrics,
    _extract_prob_columns,
    _FOLD_METRICS,
)
from stamp.statistics.calibration import _compute_calibration_for_fold

__author__ = "STAMP contributors"
__license__ = "MIT"


def compute_oof_stats_(
    *,
    preds_csvs: Sequence[Path],
    outpath: Path,
    ground_truth_label: str,
    n_bins: int = 10,
) -> None:
    """Concatenate all fold predictions and compute global metrics.

    Outputs:
        - ``{label}_oof_predictions.csv``  (all per-sample rows with added ``fold`` column)
        - ``{label}_oof_stats.csv``        (global discrimination metrics)
        - ``{label}_oof_per_class_stats.csv``
        - ``{label}_oof_confusion_matrix.csv``
        - ``{label}_oof_calibration_stats.csv``
    """
    outpath.mkdir(parents=True, exist_ok=True)

    dfs: list[pd.DataFrame] = []
    for csv_path in preds_csvs:
        df = pd.read_csv(csv_path, dtype={ground_truth_label: str})
        fold_name = Path(csv_path).parent.name
        df.insert(0, "fold", fold_name)
        dfs.append(df)

    if not dfs:
        return

    oof = pd.concat(dfs, ignore_index=True)
    oof.to_csv(outpath / f"{ground_truth_label}_oof_predictions.csv", index=False)

    # Keep only rows with a valid ground truth for metrics
    oof_valid = oof.dropna(subset=[ground_truth_label])
    if len(oof_valid) == 0:
        return

    prob_cols = _extract_prob_columns(oof_valid, ground_truth_label)
    if not prob_cols:
        return

    categories = [c[len(ground_truth_label) + 1:] for c in prob_cols]
    y_true = oof_valid[ground_truth_label].to_numpy()
    y_pred_probs = oof_valid[prob_cols].astype(float).to_numpy()

    # Global discrimination metrics
    fold_metrics = _compute_fold_metrics(y_true, y_pred_probs, categories)
    stats_df = pd.DataFrame(
        [fold_metrics], index=pd.Index(["oof"], name="source")
    )
    stats_df = stats_df[_FOLD_METRICS]
    stats_df.to_csv(outpath / f"{ground_truth_label}_oof_stats.csv")

    # Per-class precision/recall/F1
    per_class_df = _compute_per_class_metrics(y_true, y_pred_probs, categories)
    per_class_df.to_csv(
        outpath / f"{ground_truth_label}_oof_per_class_stats.csv"
    )

    # Confusion matrix
    cm_df = _compute_confusion_matrix(y_true, y_pred_probs, categories)
    cm_df.to_csv(outpath / f"{ground_truth_label}_oof_confusion_matrix.csv")

    # Calibration on the OOF set
    calib_summary, calib_per_class = _compute_calibration_for_fold(
        y_true, y_pred_probs, categories, n_bins=n_bins
    )
    calib_rows = []
    for cls, data in calib_per_class.items():
        calib_rows.append({
            "class": cls,
            "brier": data["brier"],
            "ece": data["ece"],
            "mce": data["mce"],
        })
    calib_rows.append({
        "class": "__overall__",
        "brier": calib_summary["multiclass_brier"],
        "ece": calib_summary["macro_ece"],
        "mce": calib_summary["macro_mce"],
    })
    pd.DataFrame(calib_rows).to_csv(
        outpath / f"{ground_truth_label}_oof_calibration_stats.csv", index=False
    )
