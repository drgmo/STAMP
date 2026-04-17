"""Calculate statistics for deployments on categorical targets."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn import metrics


def _bounded_ci(
    values: pd.Series, alpha: float = 0.95
) -> tuple[float, float, float]:
    """95% CI for a metric bounded to [0, 1] using a logit transformation.

    Using the t-distribution on raw means can produce CI bounds outside
    [0, 1] when the sampling distribution is skewed or truncated (e.g.
    F1 near 0 with only a few folds).  Transforming to logit space before
    computing the CI and mapping back via sigmoid guarantees bounds in
    (0, 1) while remaining asymptotically equivalent to the t-CI for
    interior values.

    Returns:
        (mean, lower, upper) — all in the original [0, 1] scale.
    """
    values = values.dropna()
    mean = float(values.mean())
    n = len(values)
    if n < 2:
        return mean, mean, mean

    # Logit with epsilon to avoid infinities when values hit 0 or 1
    eps = 1e-6
    clipped = np.clip(values.to_numpy(), eps, 1 - eps)
    logits = np.log(clipped / (1 - clipped))
    logit_mean = float(logits.mean())
    logit_sem = float(logits.std(ddof=1) / np.sqrt(n))
    t_crit = st.t.ppf(1 - (1 - alpha) / 2, df=n - 1)
    lo_logit = logit_mean - t_crit * logit_sem
    hi_logit = logit_mean + t_crit * logit_sem

    # Sigmoid back to [0, 1]
    lower = 1.0 / (1.0 + np.exp(-lo_logit))
    upper = 1.0 / (1.0 + np.exp(-hi_logit))
    return mean, float(lower), float(upper)

__author__ = "Marko van Treeck"
__copyright__ = "Copyright (C) 2022-2025 Marko van Treeck"
__license__ = "MIT"


_score_labels = [
    "roc_auc_score",
    "average_precision_score",
    "f1_score",
    "p_value",
    "count",
]


def _detect_targets_from_columns(columns: Sequence[str]) -> list[str]:
    """Detect target columns from CSV column names.

    Assumes multi-target format where each target has:
    - A ground truth column (target name)
    - A prediction column (pred_{target})
    - Probability columns ({target}_{class1}, {target}_{class2}, ...)

    Returns:
        List of target names detected.
    """
    # Convert to list to handle pandas Index
    columns = list(columns)
    targets = []
    for col in columns:
        # Look for columns that start with "pred_"
        if col.startswith("pred_"):
            target_name = col[5:]  # Remove "pred_" prefix
            # Verify the target column exists
            if target_name in columns:
                targets.append(target_name)
    return sorted(targets)


def _categorical(preds_df: pd.DataFrame, target_label: str) -> pd.DataFrame:
    """Calculates some stats for categorical prediction tables.

    This will calculate the number of items, the AUROC, AUPRC and p value
    for a prediction file.
    """
    categories = preds_df[target_label].unique()
    y_true = preds_df[target_label]
    y_pred = (
        preds_df[[f"{target_label}_{cat}" for cat in categories]].astype(float).values
    )

    stats_df = pd.DataFrame(index=categories)

    # class counts
    stats_df["count"] = y_true.value_counts()

    # roc_auc
    stats_df["roc_auc_score"] = [
        metrics.roc_auc_score(y_true == cat, y_pred[:, i])
        for i, cat in enumerate(categories)
    ]

    # average_precision
    stats_df["average_precision_score"] = [
        metrics.average_precision_score(y_true == cat, y_pred[:, i])
        for i, cat in enumerate(categories)
    ]

    # f1 score
    y_pred_labels = categories[y_pred.argmax(axis=1)]
    stats_df["f1_score"] = [
        metrics.f1_score(y_true == cat, y_pred_labels == cat) for cat in categories
    ]

    # p values
    p_values = []
    for i, cat in enumerate(categories):
        pos_scores = y_pred[:, i][y_true == cat]
        neg_scores = y_pred[:, i][y_true != cat]
        _, p_value = st.ttest_ind(pos_scores, neg_scores)
        p_values.append(p_value)
    stats_df["p_value"] = p_values

    assert set(_score_labels) & set(stats_df.columns) == set(_score_labels)

    return stats_df


def _aggregate_categorical_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = {}
    score_cols = ["roc_auc_score", "average_precision_score", "f1_score"]
    for cat, data in df.groupby("level_1"):
        scores_df = data[score_cols]
        means: dict[str, float] = {}
        lowers: dict[str, float] = {}
        uppers: dict[str, float] = {}
        for col in score_cols:
            m, lo, hi = _bounded_ci(scores_df[col])
            means[col] = m
            lowers[col] = lo
            uppers[col] = hi
        cat_stats_df = (
            pd.DataFrame.from_dict(
                {
                    "mean": pd.Series(means),
                    "95%_low": pd.Series(lowers),
                    "95%_high": pd.Series(uppers),
                }
            )
            .transpose()
            .unstack()
        )
        cat_stats_df[("count", "sum")] = data["count"].sum()
        stats[cat] = cat_stats_df

    return pd.DataFrame.from_dict(stats, orient="index")


def categorical_aggregated_(
    *, preds_csvs: Sequence[Path], outpath: Path, ground_truth_label: str
) -> None:
    """Calculate statistics for categorical deployments.

    Args:
        preds_csvs:  CSV files containing predictions.
        outpath:  Path to save the results to.
        target_label:  Label to compute the predictions for.

    This will apply `categorical` to all of the given `preds_csvs` and
    calculate the mean and 95% confidence interval for all the scores as
    well as sum the total instane count for each class.
    """
    preds_dfs = {
        Path(p).parent.name: _categorical(
            pd.read_csv(p, dtype=str).dropna(subset=[ground_truth_label]),
            ground_truth_label,
        )
        for p in preds_csvs
    }
    preds_df = pd.concat(preds_dfs).sort_index()
    preds_df.to_csv(outpath / f"{ground_truth_label}_categorical-stats_individual.csv")
    stats_df = _aggregate_categorical_stats(preds_df.reset_index())
    stats_df.to_csv(outpath / f"{ground_truth_label}_categorical-stats_aggregated.csv")


def categorical_aggregated_multitarget_(
    *,
    preds_csvs: Sequence[Path],
    outpath: Path,
    target_labels: Sequence[str],
) -> None:
    """Calculate statistics for multi-target categorical deployments.

    Args:
        preds_csvs:  CSV files containing predictions.
        outpath:  Path to save the results to.
        target_labels:  List of target labels to compute statistics for.

    This will apply `_categorical` to each target in the multi-target setup,
    calculate statistics per target, and save both individual and aggregated results.
    """
    outpath.mkdir(parents=True, exist_ok=True)

    all_target_stats = {}

    # Read each CSV once and cache so we don't re-read N×M times.
    csv_cache: dict[str, pd.DataFrame] = {
        Path(p).parent.name: pd.read_csv(p, dtype=str) for p in preds_csvs
    }

    for target_label in target_labels:
        # Process each target separately
        preds_dfs = {}
        for fold_name, df in csv_cache.items():
            # Drop rows where this target's ground truth is missing
            df_clean = df.dropna(subset=[target_label])
            if len(df_clean) > 0:
                preds_dfs[fold_name] = _categorical(df_clean, target_label)

        if not preds_dfs:
            continue

        # Concatenate and save individual stats for this target
        preds_df = pd.concat(preds_dfs).sort_index()
        preds_df.to_csv(outpath / f"{target_label}_categorical-stats_individual.csv")

        # Aggregate stats for this target
        stats_df = _aggregate_categorical_stats(preds_df.reset_index())
        stats_df.to_csv(outpath / f"{target_label}_categorical-stats_aggregated.csv")

        # Store for summary
        all_target_stats[target_label] = stats_df

    # Create a combined summary across all targets
    if all_target_stats:
        summary_dfs = []
        for target_name, stats_df in all_target_stats.items():
            stats_copy = stats_df.copy()
            stats_copy.index = pd.MultiIndex.from_product(
                [[target_name], stats_copy.index], names=["target", "class"]
            )
            summary_dfs.append(stats_copy)

        combined_summary = pd.concat(summary_dfs)
        combined_summary.to_csv(outpath / "multitarget_categorical-stats_summary.csv")
