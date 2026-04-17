"""Calibration metrics for multiclass classification.

Computes Brier Score (multiclass), Expected Calibration Error (ECE),
Maximum Calibration Error (MCE), and reliability diagrams.  Calibration
is evaluated one-vs-rest per class and also as an overall multiclass score.
"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from stamp.statistics.extended_categorical import _read_pred_csv

__author__ = "STAMP contributors"
__license__ = "MIT"


_CALIB_METRICS = [
    "multiclass_brier",
    "macro_ece",
    "macro_mce",
    "n_samples",
]


def _multiclass_brier(y_true_oh: np.ndarray, y_pred_probs: np.ndarray) -> float:
    """Multiclass Brier score: mean squared error between one-hot and predicted probs."""
    return float(np.mean(np.sum((y_pred_probs - y_true_oh) ** 2, axis=1)))


def _ece_mce(
    y_true_binary: np.ndarray, y_pred_probs: np.ndarray, n_bins: int = 10
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Expected and maximum calibration error for a binary (one-vs-rest) problem.

    Returns:
        (ece, mce, fraction_of_positives, mean_predicted_value, bin_counts)
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(y_pred_probs, bin_edges[1:-1], right=True)

    ece = 0.0
    mce = 0.0
    fraction_pos = np.full(n_bins, np.nan)
    mean_pred = np.full(n_bins, np.nan)
    bin_counts = np.zeros(n_bins, dtype=int)

    n = len(y_pred_probs)
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(mask.sum())
        bin_counts[b] = count
        if count == 0:
            continue
        fp = float(y_true_binary[mask].mean())
        mp = float(y_pred_probs[mask].mean())
        fraction_pos[b] = fp
        mean_pred[b] = mp
        gap = abs(fp - mp)
        ece += (count / n) * gap
        mce = max(mce, gap)
    return ece, mce, fraction_pos, mean_pred, bin_counts


def _one_hot(labels: np.ndarray, categories: Sequence[str]) -> np.ndarray:
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    oh = np.zeros((len(labels), len(categories)), dtype=np.int32)
    for i, lbl in enumerate(labels):
        if lbl in cat_to_idx:
            oh[i, cat_to_idx[lbl]] = 1
    return oh


def _compute_calibration_for_fold(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    categories: Sequence[str],
    n_bins: int = 10,
) -> tuple[dict, dict[str, dict]]:
    """Return (fold_summary, per_class_details).

    per_class_details[class] = {
        "brier": float, "ece": float, "mce": float,
        "fraction_pos": np.ndarray, "mean_pred": np.ndarray, "bin_counts": np.ndarray,
    }
    """
    y_true_oh = _one_hot(y_true, categories)
    brier = _multiclass_brier(y_true_oh, y_pred_probs)

    per_class: dict[str, dict] = {}
    eces: list[float] = []
    mces: list[float] = []
    for i, cls in enumerate(categories):
        y_bin = y_true_oh[:, i]
        probs = y_pred_probs[:, i]
        ece, mce, fp, mp, counts = _ece_mce(y_bin, probs, n_bins=n_bins)
        per_class[cls] = {
            "brier": brier_score_loss(y_bin, probs),
            "ece": ece,
            "mce": mce,
            "fraction_pos": fp,
            "mean_pred": mp,
            "bin_counts": counts,
        }
        eces.append(ece)
        mces.append(mce)

    fold_summary = {
        "multiclass_brier": brier,
        "macro_ece": float(np.mean(eces)) if eces else np.nan,
        "macro_mce": float(np.max(mces)) if mces else np.nan,
        "n_samples": int(len(y_true)),
    }
    return fold_summary, per_class


def _plot_reliability_diagram(
    per_class_details: dict,
    categories: Sequence[str],
    output_path: Path,
    title: str,
) -> None:
    """Plot calibration curves for a single class across all folds."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    for fold_name, fold_data in per_class_details.items():
        mp = fold_data["mean_pred"]
        fp = fold_data["fraction_pos"]
        valid = ~np.isnan(mp)
        if valid.any():
            ax.plot(mp[valid], fp[valid], marker="o", linewidth=1,
                    label=f"{fold_name} (ECE={fold_data['ece']:.3f})")

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def compute_calibration_stats_(
    *,
    preds_csvs: Sequence[Path],
    outpath: Path,
    ground_truth_label: str,
    n_bins: int = 10,
) -> None:
    """Write calibration metrics + reliability diagrams.

    Outputs:
        - ``{label}_calibration-stats_individual.csv``  (per-fold Brier/ECE/MCE)
        - ``{label}_calibration-stats_aggregated.csv``  (mean/std/95% CI)
        - ``{label}_calibration-stats_per_class_{fold}.csv``
        - ``{label}_calibration_plot_{class}.svg``
    """
    outpath.mkdir(parents=True, exist_ok=True)

    per_fold_summary: dict[str, dict] = {}
    # {class_name: {fold_name: {brier/ece/mce/fraction_pos/mean_pred/bin_counts}}}
    per_class_per_fold: dict[str, dict[str, dict]] = {}
    reference_categories: list[str] = []

    for csv_path in preds_csvs:
        fold_name = Path(csv_path).parent.name
        result = _read_pred_csv(Path(csv_path), ground_truth_label)
        if result is None:
            continue
        y_true, y_pred_probs, categories = result
        if not reference_categories:
            reference_categories = list(categories)

        summary, per_class = _compute_calibration_for_fold(
            y_true, y_pred_probs, categories, n_bins=n_bins
        )
        per_fold_summary[fold_name] = summary

        # Store for plotting + per-class CSV
        per_class_rows = []
        for cls, data in per_class.items():
            per_class_per_fold.setdefault(cls, {})[fold_name] = data
            per_class_rows.append({
                "class": cls,
                "brier": data["brier"],
                "ece": data["ece"],
                "mce": data["mce"],
            })
        pd.DataFrame(per_class_rows).to_csv(
            outpath / f"{ground_truth_label}_calibration-stats_per_class_{fold_name}.csv",
            index=False,
        )

    if not per_fold_summary:
        return

    # Per-fold individual CSV
    individual_df = pd.DataFrame.from_dict(per_fold_summary, orient="index")
    individual_df.index.name = "fold"
    individual_df = individual_df[_CALIB_METRICS]
    individual_df.to_csv(
        outpath / f"{ground_truth_label}_calibration-stats_individual.csv"
    )

    # Aggregated stats — calibration metrics (Brier, ECE, MCE) are all
    # bounded to [0, 1], so use a logit-transformed CI to keep bounds valid.
    from stamp.statistics.categorical import _bounded_ci

    score_cols = [c for c in _CALIB_METRICS if c != "n_samples"]
    scores = individual_df[score_cols]
    n_folds = len(scores)
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    lowers: dict[str, float] = {}
    uppers: dict[str, float] = {}
    for col in score_cols:
        values = scores[col].dropna()
        if len(values) == 0:
            means[col] = np.nan
            stds[col] = np.nan
            lowers[col] = np.nan
            uppers[col] = np.nan
            continue
        m = float(values.mean())
        s = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        stds[col] = s
        if len(values) >= 2:
            _, lo, hi = _bounded_ci(values)
            means[col] = m
            lowers[col] = lo
            uppers[col] = hi
        else:
            means[col] = m
            lowers[col] = m
            uppers[col] = m
    aggregated = pd.DataFrame(
        {
            "mean": pd.Series(means),
            "std": pd.Series(stds),
            "95%_low": pd.Series(lowers),
            "95%_high": pd.Series(uppers),
            "n_folds": n_folds,
        }
    )
    aggregated.to_csv(
        outpath / f"{ground_truth_label}_calibration-stats_aggregated.csv"
    )

    # Reliability plots per class (one per class, all folds overlaid)
    for cls, folds_data in per_class_per_fold.items():
        safe_cls = cls.replace("/", "_").replace("\\", "_").replace(" ", "_")
        plot_path = outpath / f"{ground_truth_label}_calibration_plot_{safe_cls}.svg"
        _plot_reliability_diagram(
            folds_data,
            categories=reference_categories,
            output_path=plot_path,
            title=f"Reliability diagram — {ground_truth_label} = {cls}",
        )
