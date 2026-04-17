"""Extended categorical statistics (macro/weighted aggregation, confusion matrix, etc.).

Reads the same ``patient-preds.csv`` files as :mod:`stamp.statistics.categorical`
but computes additional metrics that are standard in medical AI reporting
(TRIPOD+AI): macro/weighted AUROC/F1/AP, balanced accuracy, per-class
precision/recall, MCC, Cohen's Kappa, log loss, confusion matrix, and
per-fold class distributions.

This module does not modify existing behaviour — it only adds extra output
files alongside the existing categorical stats.
"""

import logging
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn import metrics

_logger = logging.getLogger("stamp")

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


def _count_tiles_in_h5(h5_path: Path) -> int | None:
    """Count the number of tiles in a STAMP feature H5 file.

    Returns None if the file doesn't exist or the dataset can't be read.
    Returns 1 for slide-level features (single vector per file).
    """
    if not h5_path.exists():
        return None
    try:
        with h5py.File(str(h5_path), "r") as f:
            # slide/patient-level features: feat_type attribute says so
            feat_type = f.attrs.get("feat_type")
            if feat_type in ("slide", "patient"):
                return 1
            # tile-level: /feats or /patch_embeddings dataset
            for key in ("feats", "patch_embeddings"):
                if key in f:
                    return int(f[key].shape[0])
            # Fallback: check /tile/features (WSIVL-style layout)
            if "tile" in f and "features" in f["tile"]:
                return int(f["tile/features"].shape[0])
    except (OSError, KeyError):
        return None
    return None


def _tile_counts_per_patient(
    *,
    feature_dir: Path,
    slide_table: Path,
    patient_label: str,
    filename_label: str,
) -> dict[str, int]:
    """Sum tile counts across all slides of each patient.

    Returns ``{patient_id: total_tile_count}``.  Patients without any
    readable H5 file are omitted.
    """
    if slide_table.suffix == ".xlsx":
        df = pd.read_excel(slide_table, dtype=str)
    else:
        df = pd.read_csv(slide_table, dtype=str)

    result: dict[str, int] = {}
    missing = 0
    for _, row in df.iterrows():
        pid = row[patient_label]
        filename = row[filename_label]
        h5_path = feature_dir / filename
        n = _count_tiles_in_h5(h5_path)
        if n is None:
            missing += 1
            continue
        result[pid] = result.get(pid, 0) + n

    if missing:
        _logger.info(
            "Tile counting: %d slide files not found in %s", missing, feature_dir
        )
    return result


def _compute_tile_distribution_row(
    *,
    fold_name: str,
    csv_path: Path,
    target_label: str,
    categories: Sequence[str],
    tile_counts_per_patient: dict[str, int],
) -> dict | None:
    """Build one row of tile-count-per-class statistics for a single fold.

    Reads the patient_id / target column from patient-preds.csv, maps each
    patient to their tile count, and sums per class.
    """
    df = pd.read_csv(csv_path, dtype=str)
    # Find patient ID column — use the first column that isn't a stats column
    # (patient-preds.csv writes patient_label first).  Heuristic: the column
    # that contains keys present in tile_counts_per_patient.
    candidate_cols = [
        c for c in df.columns
        if c != target_label
        and not c.startswith(f"{target_label}_")
        and not c.startswith("pred")
        and c != "loss"
    ]
    pid_col = None
    for c in candidate_cols:
        if df[c].astype(str).isin(set(tile_counts_per_patient.keys())).any():
            pid_col = c
            break
    if pid_col is None:
        _logger.warning(
            "Could not match patient IDs from %s to tile counts. "
            "Skipping tile-count distribution for fold %s.",
            csv_path, fold_name,
        )
        return None

    df = df.dropna(subset=[target_label])
    row: dict = {"fold": fold_name}
    for c in categories:
        mask = df[target_label] == c
        pids = df.loc[mask, pid_col].astype(str)
        total_tiles = sum(tile_counts_per_patient.get(p, 0) for p in pids)
        row[f"tile_count_{c}"] = int(total_tiles)
    # Sum of all tiles in fold
    all_pids = df[pid_col].astype(str)
    row["tile_count_total"] = int(
        sum(tile_counts_per_patient.get(p, 0) for p in all_pids)
    )
    return row


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
    # Renormalize so rows sum to 1 (fixes float precision rounding)
    row_sums = y_pred_probs.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    y_pred_probs = y_pred_probs / row_sums
    return y_true, y_pred_probs, categories


def compute_extended_stats_(
    *,
    preds_csvs: Sequence[Path],
    outpath: Path,
    ground_truth_label: str,
    feature_dir: Path | None = None,
    slide_table: Path | None = None,
    patient_label: str = "PATIENT",
    filename_label: str = "FILENAME",
) -> None:
    """Write extended categorical stats next to the existing ones.

    Outputs:
        - ``{label}_extended-stats_individual.csv``  (per-fold metrics)
        - ``{label}_extended-stats_aggregated.csv``  (mean/std/95% CI across folds)
        - ``{label}_per_class_stats_split-{i}.csv``  (precision/recall/F1 per class)
        - ``{label}_confusion_matrix_split-{i}.csv``
        - ``{label}_fold_class_distribution.csv``    (true/predicted counts per fold)
        - ``{label}_fold_tile_distribution.csv``     (tile counts per class per fold,
          only if ``feature_dir`` and ``slide_table`` are provided)
    """
    outpath.mkdir(parents=True, exist_ok=True)

    # Optionally pre-compute tile counts per patient
    tile_counts_per_patient: dict[str, int] | None = None
    if feature_dir is not None and slide_table is not None:
        try:
            tile_counts_per_patient = _tile_counts_per_patient(
                feature_dir=feature_dir,
                slide_table=slide_table,
                patient_label=patient_label,
                filename_label=filename_label,
            )
            _logger.info(
                "Loaded tile counts for %d patients from %s",
                len(tile_counts_per_patient), feature_dir,
            )
        except Exception as e:
            _logger.warning("Failed to compute tile counts: %s", e)
            tile_counts_per_patient = None

    per_fold_rows: dict[str, dict] = {}
    class_distribution_rows: list[dict] = []
    tile_distribution_rows: list[dict] = []
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

        # Tile counts per class per fold (if tile counts available)
        if tile_counts_per_patient is not None:
            tile_row = _compute_tile_distribution_row(
                fold_name=fold_name,
                csv_path=Path(csv_path),
                target_label=ground_truth_label,
                categories=categories,
                tile_counts_per_patient=tile_counts_per_patient,
            )
            if tile_row is not None:
                tile_distribution_rows.append(tile_row)

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
    # For metrics bounded to [0, 1] we use a logit-transformed CI so that
    # bounds stay in (0, 1).  For unbounded metrics (MCC, kappa, log_loss)
    # we fall back to a raw t-based CI.
    from stamp.statistics.categorical import _bounded_ci

    bounded_metrics = {
        "macro_auroc", "weighted_auroc",
        "macro_f1", "weighted_f1",
        "macro_average_precision", "weighted_average_precision",
        "balanced_accuracy", "accuracy",
    }
    score_cols = [c for c in _FOLD_METRICS if c != "n_samples"]
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
        if col in bounded_metrics and len(values) >= 2:
            _, lo, hi = _bounded_ci(values)
            means[col] = m
            lowers[col] = lo
            uppers[col] = hi
        elif len(values) >= 2:
            sem = float(values.sem())
            t_crit = st.t.ppf(0.975, df=len(values) - 1)
            means[col] = m
            lowers[col] = m - t_crit * sem
            uppers[col] = m + t_crit * sem
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

    # Fold tile distribution CSV (tile counts per class per fold)
    if tile_distribution_rows:
        tile_df = pd.DataFrame(tile_distribution_rows).set_index("fold")
        tile_df.to_csv(
            outpath / f"{ground_truth_label}_fold_tile_distribution.csv"
        )
