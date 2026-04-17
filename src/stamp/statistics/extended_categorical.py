"""Extended categorical statistics (macro/weighted aggregation, confusion matrix, etc.).

Reads the same ``patient-preds.csv`` files as :mod:`stamp.statistics.categorical`
but computes additional metrics that are standard in medical AI reporting
(TRIPOD+AI): macro/weighted AUROC/F1/AP, balanced accuracy, per-class
precision/recall, MCC, Cohen's Kappa, log loss, confusion matrix, and
per-fold class distributions.

This module does not modify existing behaviour — it only adds extra output
files alongside the existing categorical stats.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn import metrics

__author__ = "STAMP contributors"
__license__ = "MIT"


_FOLD_METRICS = [
    "macro_auroc",
    "weighted_auroc",
    "macro_f1",
    "weighted_f1",
    "macro_average_precision",
    "weighted_average_precision",
    "balanced_accuracy",
    "accuracy",
    "mcc",
    "cohens_kappa",
    "log_loss",
    "n_samples",
]


def _one_hot(labels: np.ndarray, categories: Sequence[str]) -> np.ndarray:
    """Build a one-hot matrix with shape (N, n_classes)."""
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    oh = np.zeros((len(labels), len(categories)), dtype=np.int32)
    for i, lbl in enumerate(labels):
        if lbl in cat_to_idx:
            oh[i, cat_to_idx[lbl]] = 1
    return oh


def _compute_fold_metrics(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    categories: Sequence[str],
) -> dict:
    """Compute discrimination/aggregation metrics for one fold."""
    y_true_oh = _one_hot(y_true, categories)
    y_pred_labels = np.array([categories[i] for i in y_pred_probs.argmax(axis=1)])

    # AUROC (one-vs-rest)
    try:
        macro_auroc = metrics.roc_auc_score(y_true_oh, y_pred_probs, average="macro")
    except ValueError:
        macro_auroc = np.nan
    try:
        weighted_auroc = metrics.roc_auc_score(
            y_true_oh, y_pred_probs, average="weighted"
        )
    except ValueError:
        weighted_auroc = np.nan

    # F1
    macro_f1 = metrics.f1_score(
        y_true, y_pred_labels, average="macro", zero_division=0
    )
    weighted_f1 = metrics.f1_score(
        y_true, y_pred_labels, average="weighted", zero_division=0
    )

    # Average precision (one-vs-rest)
    try:
        macro_ap = metrics.average_precision_score(
            y_true_oh, y_pred_probs, average="macro"
        )
    except ValueError:
        macro_ap = np.nan
    try:
        weighted_ap = metrics.average_precision_score(
            y_true_oh, y_pred_probs, average="weighted"
        )
    except ValueError:
        weighted_ap = np.nan

    # Accuracy-like
    balanced_accuracy = metrics.balanced_accuracy_score(y_true, y_pred_labels)
    accuracy = metrics.accuracy_score(y_true, y_pred_labels)

    # Agreement
    mcc = metrics.matthews_corrcoef(y_true, y_pred_labels)
    kappa = metrics.cohen_kappa_score(y_true, y_pred_labels)

    # Log loss (NLL)
    try:
        ll = metrics.log_loss(y_true, y_pred_probs, labels=list(categories))
    except ValueError:
        ll = np.nan

    return {
        "macro_auroc": macro_auroc,
        "weighted_auroc": weighted_auroc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_average_precision": macro_ap,
        "weighted_average_precision": weighted_ap,
        "balanced_accuracy": balanced_accuracy,
        "accuracy": accuracy,
        "mcc": mcc,
        "cohens_kappa": kappa,
        "log_loss": ll,
        "n_samples": len(y_true),
    }


def _compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    categories: Sequence[str],
) -> pd.DataFrame:
    """Per-class precision, recall, F1, support, predicted count."""
    y_pred_labels = np.array([categories[i] for i in y_pred_probs.argmax(axis=1)])

    precision, recall, f1, support = metrics.precision_recall_fscore_support(
        y_true, y_pred_labels, labels=list(categories), zero_division=0
    )
    predicted_count = [int((y_pred_labels == c).sum()) for c in categories]

    return pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "predicted_count": predicted_count,
        },
        index=list(categories),
    )


def _compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    categories: Sequence[str],
) -> pd.DataFrame:
    """Confusion matrix as DataFrame with named rows (true) and columns (predicted)."""
    y_pred_labels = np.array([categories[i] for i in y_pred_probs.argmax(axis=1)])
    cm = metrics.confusion_matrix(y_true, y_pred_labels, labels=list(categories))
    return pd.DataFrame(
        cm,
        index=pd.Index(list(categories), name="true"),
        columns=pd.Index(list(categories), name="predicted"),
    )


def _extract_prob_columns(df: pd.DataFrame, target_label: str) -> list[str]:
    """Return the list of probability column names in canonical order."""
    prefix = f"{target_label}_"
    return [
        c for c in df.columns
        if c.startswith(prefix) and c != target_label and c != f"pred_{target_label}"
    ]


def _read_pred_csv(
    csv_path: Path, target_label: str
) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    """Read a patient-preds.csv and return (y_true, y_pred_probs, categories).

    Returns ``None`` if the CSV doesn't contain usable rows (e.g. all GT missing).
    """
    df = pd.read_csv(csv_path, dtype={target_label: str})
    df = df.dropna(subset=[target_label])
    if len(df) == 0:
        return None

    prob_cols = _extract_prob_columns(df, target_label)
    if not prob_cols:
        return None

    # Derive category order from column names: "{target}_{class}"
    categories = [c[len(target_label) + 1:] for c in prob_cols]
    y_true = df[target_label].to_numpy()
    y_pred_probs = df[prob_cols].astype(float).to_numpy()
    return y_true, y_pred_probs, categories


def compute_extended_stats_(
    *,
    preds_csvs: Sequence[Path],
    outpath: Path,
    ground_truth_label: str,
) -> None:
    """Write extended categorical stats next to the existing ones.

    Outputs:
        - ``{label}_extended-stats_individual.csv``  (per-fold metrics)
        - ``{label}_extended-stats_aggregated.csv``  (mean/std/95% CI across folds)
        - ``{label}_per_class_stats_split-{i}.csv``  (precision/recall/F1 per class)
        - ``{label}_confusion_matrix_split-{i}.csv``
        - ``{label}_fold_class_distribution.csv``    (true/predicted counts per fold)
    """
    outpath.mkdir(parents=True, exist_ok=True)

    per_fold_rows: dict[str, dict] = {}
    class_distribution_rows: list[dict] = []
    all_categories: list[str] = []

    for csv_path in preds_csvs:
        fold_name = Path(csv_path).parent.name
        result = _read_pred_csv(Path(csv_path), ground_truth_label)
        if result is None:
            continue
        y_true, y_pred_probs, categories = result

        # Remember the first non-empty category order we see
        if not all_categories:
            all_categories = list(categories)

        # Main fold metrics
        per_fold_rows[fold_name] = _compute_fold_metrics(
            y_true, y_pred_probs, categories
        )

        # Per-class metrics → own CSV per fold
        per_class_df = _compute_per_class_metrics(y_true, y_pred_probs, categories)
        per_class_df.to_csv(
            outpath / f"{ground_truth_label}_per_class_stats_{fold_name}.csv"
        )

        # Confusion matrix → own CSV per fold
        cm_df = _compute_confusion_matrix(y_true, y_pred_probs, categories)
        cm_df.to_csv(
            outpath / f"{ground_truth_label}_confusion_matrix_{fold_name}.csv"
        )

        # Fold class distribution row
        y_pred_labels = np.array(
            [categories[i] for i in y_pred_probs.argmax(axis=1)]
        )
        row = {"fold": fold_name, "n_samples": len(y_true)}
        for c in categories:
            row[f"true_count_{c}"] = int((y_true == c).sum())
            row[f"predicted_count_{c}"] = int((y_pred_labels == c).sum())
        class_distribution_rows.append(row)

    if not per_fold_rows:
        return

    # Per-fold individual CSV
    individual_df = pd.DataFrame.from_dict(per_fold_rows, orient="index")
    individual_df.index.name = "fold"
    individual_df = individual_df[_FOLD_METRICS]
    individual_df.to_csv(
        outpath / f"{ground_truth_label}_extended-stats_individual.csv"
    )

    # Aggregated (mean, std, 95% CI)
    score_cols = [c for c in _FOLD_METRICS if c != "n_samples"]
    scores = individual_df[score_cols]
    means = scores.mean()
    stds = scores.std(ddof=1)
    sems = scores.sem()
    n_folds = len(scores)
    if n_folds >= 2:
        lower, upper = st.t.interval(
            0.95, df=n_folds - 1, loc=means, scale=sems
        )
    else:
        lower = means
        upper = means
    aggregated = pd.DataFrame(
        {
            "mean": means,
            "std": stds,
            "95%_low": lower,
            "95%_high": upper,
            "n_folds": n_folds,
        }
    )
    aggregated.loc["n_samples_total"] = [
        int(individual_df["n_samples"].sum()),
        np.nan,
        np.nan,
        np.nan,
        n_folds,
    ]
    aggregated.to_csv(
        outpath / f"{ground_truth_label}_extended-stats_aggregated.csv"
    )

    # Fold class distribution CSV
    if class_distribution_rows:
        dist_df = pd.DataFrame(class_distribution_rows).set_index("fold")
        dist_df.to_csv(
            outpath / f"{ground_truth_label}_fold_class_distribution.csv"
        )
