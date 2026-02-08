"""Survival risk scoring, stratification and evaluation.

Minimal helpers that take an existing model score (or small set of
covariates) and:

1. Fit a Cox proportional-hazards model on top to produce a continuous
   **risk score** per patient.
2. Choose one or more **thresholds** to split patients into high- vs
   low-risk groups (``HRDpos`` / ``HRDneg``).
3. Evaluate the stratification via **C-index**, **log-rank test**, and
   **Kaplan–Meier** curves.

The functions are designed to be used *after* existing inference — they
wrap around a DataFrame that already contains a score column.  No model
retraining is required.

Typical usage::

    from stamp.statistics.survival_risk import (
        fit_cox_model,
        get_survival_thresholds,
        assign_risk_groups,
        evaluate_survival_stratification,
    )

    # df_surv has columns: patient_id, time, event, y_score
    result = fit_cox_model(df_surv, score_col="y_score")
    df_surv["risk_score"] = result.risk_scores

    thresholds = get_survival_thresholds(
        df_surv, risk_col="risk_score", time_col="time", event_col="event",
    )
    summary = evaluate_survival_stratification(
        df_surv, risk_col="risk_score", time_col="time", event_col="event",
        thresholds=thresholds,
    )
"""

from __future__ import annotations

import logging
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from matplotlib import pyplot as plt

__author__ = "STAMP contributors"
__license__ = "MIT"

_logger = logging.getLogger("stamp")


# ── A. Cox model fitting ───────────────────────────────────────────────


@dataclass
class CoxResult:
    """Container returned by :func:`fit_cox_model`."""

    fitter: CoxPHFitter
    risk_scores: npt.NDArray[np.float64]
    summary: pd.DataFrame
    concordance_index: float
    covariates: list[str] = field(default_factory=list)


def fit_cox_model(
    df: pd.DataFrame,
    *,
    score_col: str = "y_score",
    time_col: str = "time",
    event_col: str = "event",
    extra_covariates: list[str] | None = None,
    penalizer: float = 0.01,
) -> CoxResult:
    """Fit a Cox proportional-hazards model and compute risk scores.

    Parameters
    ----------
    df
        Patient-level DataFrame.
    score_col
        Column with the continuous model score to use as the primary
        covariate.
    time_col, event_col
        Survival duration and event indicator (1 = event, 0 = censored).
    extra_covariates
        Optional additional clinical columns to include as covariates.
    penalizer
        L2 penalizer for ``CoxPHFitter`` (stabilises small-sample fits).

    Returns
    -------
    CoxResult
        Fitted model, risk scores (linear predictor), summary table, and
        concordance index.
    """
    covariates = [score_col]
    if extra_covariates:
        covariates = covariates + list(extra_covariates)

    cols_needed = covariates + [time_col, event_col]
    work = df[cols_needed].dropna().copy()

    if len(work) < 5:
        raise ValueError(
            f"Too few valid rows ({len(work)}) after dropping NaN. "
            "Need at least 5 for a Cox model."
        )

    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(work, duration_col=time_col, event_col=event_col)

    # Risk score = partial hazard (log scale) = linear predictor X·β
    # Higher value → higher risk → shorter expected survival.
    risk_scores = cph.predict_partial_hazard(work).values.ravel()

    c_index = float(
        concordance_index(
            work[time_col],
            -risk_scores,  # lifelines convention: predicted *survival* time
            work[event_col],
        )
    )

    return CoxResult(
        fitter=cph,
        risk_scores=risk_scores,
        summary=cph.summary,
        concordance_index=c_index,
        covariates=covariates,
    )


# ── B. Threshold strategies ────────────────────────────────────────────


def _max_logrank_threshold(
    risk: npt.NDArray[np.float64],
    time: npt.NDArray[np.float64],
    event: npt.NDArray[np.int_],
    *,
    lower_quantile: float = 0.10,
    upper_quantile: float = 0.90,
) -> float:
    """Find the threshold that maximises the log-rank test statistic.

    Candidate cutpoints are the unique risk-score values between the
    *lower_quantile* and *upper_quantile* of the distribution (to avoid
    degenerate splits near the extremes).
    """
    lo = float(np.quantile(risk, lower_quantile))
    hi = float(np.quantile(risk, upper_quantile))

    candidates = np.unique(risk)
    candidates = candidates[(candidates >= lo) & (candidates <= hi)]

    if len(candidates) == 0:
        warnings.warn(
            "No valid candidate cutpoints for max_logrank; "
            "falling back to median.",
            stacklevel=2,
        )
        return float(np.median(risk))

    best_stat = -np.inf
    best_thr = float(np.median(risk))

    for thr in candidates:
        low_mask = risk <= thr
        high_mask = risk > thr
        if low_mask.sum() < 2 or high_mask.sum() < 2:
            continue
        try:
            res = logrank_test(
                time[low_mask],
                time[high_mask],
                event_observed_A=event[low_mask],
                event_observed_B=event[high_mask],
            )
            stat = float(res.test_statistic)
        except Exception:
            continue

        if stat > best_stat:
            best_stat = stat
            best_thr = float(thr)

    return best_thr


def get_survival_thresholds(
    df: pd.DataFrame,
    *,
    risk_col: str = "risk_score",
    time_col: str = "time",
    event_col: str = "event",
    include: tuple[str, ...] = ("median_risk", "mean_risk", "max_logrank"),
    fixed_thresholds: tuple[float, ...] = (),
) -> OrderedDict[str, float]:
    """Compute candidate thresholds for risk-score stratification.

    Parameters
    ----------
    df
        Must contain *risk_col*, *time_col*, *event_col*.
    include
        Built-in strategies: ``"median_risk"``, ``"mean_risk"``,
        ``"max_logrank"``.
    fixed_thresholds
        Literal cutpoints (e.g. from a training set).

    Returns
    -------
    OrderedDict[str, float]
        ``{strategy_name: threshold_value}``.
    """
    work = df[[risk_col, time_col, event_col]].dropna()
    risk = work[risk_col].values.astype(float)
    time = work[time_col].values.astype(float)
    event = work[event_col].values.astype(int)

    thresholds: OrderedDict[str, float] = OrderedDict()

    if "median_risk" in include:
        thresholds["median_risk"] = float(np.median(risk))

    if "mean_risk" in include:
        thresholds["mean_risk"] = float(np.mean(risk))

    if "max_logrank" in include:
        thresholds["max_logrank"] = _max_logrank_threshold(risk, time, event)

    for t in fixed_thresholds:
        thresholds[f"fixed_{t:.4f}"] = float(t)

    return thresholds


# ── C. Risk-group assignment ───────────────────────────────────────────


def assign_risk_groups(
    risk_scores: npt.ArrayLike,
    threshold: float,
    *,
    high_label: str = "high_risk",
    low_label: str = "low_risk",
) -> np.ndarray:
    """Assign patients to high/low risk groups.

    Prediction rule: ``risk_score > threshold → high_label``.
    """
    arr = np.asarray(risk_scores, dtype=float)
    return np.where(arr > threshold, high_label, low_label)


# ── D. Per-threshold evaluation ────────────────────────────────────────


def _evaluate_one_threshold(
    risk: npt.NDArray[np.float64],
    time: npt.NDArray[np.float64],
    event: npt.NDArray[np.int_],
    threshold: float,
    *,
    c_index: float | None = None,
) -> dict[str, Any]:
    """Compute survival-stratification metrics at one threshold."""
    low_mask = risk <= threshold
    high_mask = risk > threshold

    n_low = int(low_mask.sum())
    n_high = int(high_mask.sum())

    # C-index (threshold-independent, compute once)
    if c_index is None:
        try:
            c_index = float(concordance_index(time, -risk, event))
        except Exception:
            c_index = float("nan")

    # Log-rank test
    if n_low >= 2 and n_high >= 2:
        try:
            res = logrank_test(
                time[low_mask],
                time[high_mask],
                event_observed_A=event[low_mask],
                event_observed_B=event[high_mask],
            )
            logrank_p = float(res.p_value)
            logrank_stat = float(res.test_statistic)
        except Exception:
            logrank_p = float("nan")
            logrank_stat = float("nan")
    else:
        logrank_p = float("nan")
        logrank_stat = float("nan")

    return {
        "threshold": threshold,
        "c_index": c_index,
        "logrank_p": logrank_p,
        "logrank_stat": logrank_stat,
        "n_low_risk": n_low,
        "n_high_risk": n_high,
        "n_total": n_low + n_high,
    }


def evaluate_survival_stratification(
    df: pd.DataFrame,
    *,
    risk_col: str = "risk_score",
    time_col: str = "time",
    event_col: str = "event",
    thresholds: OrderedDict[str, float] | dict[str, float] | None = None,
) -> pd.DataFrame:
    """Evaluate risk-score stratification across multiple thresholds.

    Parameters
    ----------
    df
        Must contain *risk_col*, *time_col*, *event_col*.
    thresholds
        ``{strategy_name: threshold_value}`` from
        :func:`get_survival_thresholds`.  Computed automatically when
        *None*.

    Returns
    -------
    pandas.DataFrame
        One row per strategy with columns: ``strategy, threshold,
        c_index, logrank_p, logrank_stat, n_low_risk, n_high_risk,
        n_total``.
    """
    work = df[[risk_col, time_col, event_col]].dropna()
    risk = work[risk_col].values.astype(float)
    time = work[time_col].values.astype(float)
    event = work[event_col].values.astype(int)

    if thresholds is None:
        thresholds = get_survival_thresholds(
            df, risk_col=risk_col, time_col=time_col, event_col=event_col
        )

    # Pre-compute C-index once (threshold-independent)
    try:
        c_index = float(concordance_index(time, -risk, event))
    except Exception:
        c_index = float("nan")

    rows: list[dict[str, Any]] = []
    for name, thr in thresholds.items():
        metrics = _evaluate_one_threshold(risk, time, event, thr, c_index=c_index)
        metrics["strategy"] = name
        rows.append(metrics)

    return pd.DataFrame(rows)[
        [
            "strategy",
            "threshold",
            "c_index",
            "logrank_p",
            "logrank_stat",
            "n_low_risk",
            "n_high_risk",
            "n_total",
        ]
    ]


# ── E. Kaplan–Meier plot ──────────────────────────────────────────────


def plot_km_by_group(
    df: pd.DataFrame,
    *,
    group_col: str = "risk_group",
    time_col: str = "time",
    event_col: str = "event",
    title: str = "Kaplan–Meier Survival Curve",
    c_index: float | None = None,
    logrank_p: float | None = None,
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot Kaplan–Meier survival curves for two risk groups.

    Parameters
    ----------
    df
        Must contain *group_col*, *time_col*, *event_col*.
    output_path
        If provided, save the figure to this path (parent dirs created).

    Returns
    -------
    matplotlib.figure.Figure
    """
    work = df.dropna(subset=[group_col, time_col, event_col])
    groups = work[group_col].unique()

    if len(groups) != 2:
        raise ValueError(
            f"Expected exactly 2 risk groups, got {len(groups)}: {list(groups)}"
        )

    fig, ax = plt.subplots(figsize=(8, 6))
    kmfs = []
    colors = ["blue", "red"]

    for i, grp in enumerate(sorted(groups)):
        mask = work[group_col] == grp
        sub = work[mask]
        kmf = KaplanMeierFitter()
        kmf.fit(sub[time_col], event_observed=sub[event_col], label=str(grp))
        kmf.plot_survival_function(ax=ax, ci_show=False, color=colors[i % 2])
        kmfs.append(kmf)

    add_at_risk_counts(*kmfs, ax=ax)

    # Annotation box
    parts = []
    if logrank_p is not None:
        parts.append(f"Log-rank p = {logrank_p:.4e}")
    if c_index is not None:
        parts.append(f"C-index = {c_index:.3f}")
    if parts:
        ax.text(
            0.6,
            0.08,
            "\n".join(parts),
            transform=ax.transAxes,
            fontsize=11,
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
        )

    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_ylim(0, 1)
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close(fig)
    return fig
