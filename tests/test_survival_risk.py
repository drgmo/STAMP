"""Tests for survival_risk scoring and stratification helpers."""

import numpy as np
import pandas as pd
import pytest

from stamp.statistics.survival_risk import (
    CoxResult,
    assign_risk_groups,
    evaluate_survival_stratification,
    fit_cox_model,
    get_survival_thresholds,
    plot_km_by_group,
)


# ── Fixtures ───────────────────────────────────────────────────────────


def _make_surv_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic survival DataFrame with informative score."""
    rng = np.random.RandomState(seed)
    # Higher score → shorter survival (positive association with risk)
    score = rng.uniform(0, 1, n)
    time = np.maximum(1.0, 100 * (1 - score) + rng.normal(0, 10, n))
    event = rng.binomial(1, 0.7, n)
    return pd.DataFrame(
        {
            "patient_id": [f"P{i:03d}" for i in range(n)],
            "time": time,
            "event": event,
            "y_score": score,
        }
    )


# ── fit_cox_model ──────────────────────────────────────────────────────


class TestFitCoxModel:
    def test_basic_fit(self) -> None:
        df = _make_surv_df()
        result = fit_cox_model(df, score_col="y_score")
        assert isinstance(result, CoxResult)
        assert len(result.risk_scores) > 0
        assert result.concordance_index > 0.0
        assert "y_score" in result.covariates

    def test_returns_risk_scores_same_length_as_valid_rows(self) -> None:
        df = _make_surv_df(n=50)
        result = fit_cox_model(df, score_col="y_score")
        n_valid = df[["y_score", "time", "event"]].dropna().shape[0]
        assert len(result.risk_scores) == n_valid

    def test_c_index_reasonable(self) -> None:
        # With an informative score, C-index should be > 0.5 (random)
        df = _make_surv_df(n=200)
        result = fit_cox_model(df, score_col="y_score")
        assert result.concordance_index > 0.5

    def test_extra_covariates(self) -> None:
        df = _make_surv_df()
        df["age"] = np.random.RandomState(0).uniform(40, 80, len(df))
        result = fit_cox_model(
            df, score_col="y_score", extra_covariates=["age"]
        )
        assert "age" in result.covariates
        assert "y_score" in result.covariates

    def test_too_few_rows_raises(self) -> None:
        df = _make_surv_df(n=3)
        with pytest.raises(ValueError, match="Too few"):
            fit_cox_model(df, score_col="y_score")

    def test_summary_is_dataframe(self) -> None:
        df = _make_surv_df()
        result = fit_cox_model(df, score_col="y_score")
        assert isinstance(result.summary, pd.DataFrame)


# ── get_survival_thresholds ────────────────────────────────────────────


class TestGetSurvivalThresholds:
    def test_returns_expected_keys(self) -> None:
        df = _make_surv_df()
        df["risk_score"] = np.random.RandomState(0).uniform(0, 1, len(df))
        result = get_survival_thresholds(df, risk_col="risk_score")
        assert "median_risk" in result
        assert "mean_risk" in result
        assert "max_logrank" in result

    def test_median_is_correct(self) -> None:
        df = _make_surv_df()
        df["risk_score"] = np.arange(len(df), dtype=float)
        result = get_survival_thresholds(
            df, risk_col="risk_score", include=("median_risk",)
        )
        expected = float(np.median(df["risk_score"]))
        assert abs(result["median_risk"] - expected) < 1e-10

    def test_fixed_thresholds(self) -> None:
        df = _make_surv_df()
        df["risk_score"] = np.random.RandomState(0).uniform(0, 1, len(df))
        result = get_survival_thresholds(
            df,
            risk_col="risk_score",
            include=(),
            fixed_thresholds=(0.5, 1.0),
        )
        assert "fixed_0.5000" in result
        assert "fixed_1.0000" in result

    def test_max_logrank_returns_float(self) -> None:
        df = _make_surv_df(n=80)
        df["risk_score"] = np.random.RandomState(0).uniform(0, 1, len(df))
        result = get_survival_thresholds(
            df, risk_col="risk_score", include=("max_logrank",)
        )
        assert isinstance(result["max_logrank"], float)


# ── assign_risk_groups ─────────────────────────────────────────────────


class TestAssignRiskGroups:
    def test_basic_split(self) -> None:
        scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        groups = assign_risk_groups(scores, 0.5)
        expected = np.array(["HRDneg", "HRDneg", "HRDneg", "HRDpos", "HRDpos"])
        np.testing.assert_array_equal(groups, expected)

    def test_custom_labels(self) -> None:
        scores = [0.2, 0.8]
        groups = assign_risk_groups(
            scores, 0.5, high_label="High", low_label="Low"
        )
        np.testing.assert_array_equal(groups, ["Low", "High"])

    def test_boundary_goes_to_low(self) -> None:
        # Exactly at threshold → low risk (rule: > threshold → high)
        groups = assign_risk_groups([0.5], 0.5)
        assert groups[0] == "HRDneg"


# ── evaluate_survival_stratification ───────────────────────────────────


class TestEvaluateSurvivalStratification:
    def test_returns_dataframe(self) -> None:
        df = _make_surv_df()
        result = fit_cox_model(df, score_col="y_score")
        df_valid = df.iloc[: len(result.risk_scores)].copy()
        df_valid["risk_score"] = result.risk_scores

        summary = evaluate_survival_stratification(df_valid, risk_col="risk_score")
        assert isinstance(summary, pd.DataFrame)
        assert "strategy" in summary.columns
        assert "c_index" in summary.columns
        assert "logrank_p" in summary.columns
        assert len(summary) > 0

    def test_c_index_is_consistent(self) -> None:
        df = _make_surv_df()
        result = fit_cox_model(df, score_col="y_score")
        df_valid = df.iloc[: len(result.risk_scores)].copy()
        df_valid["risk_score"] = result.risk_scores

        summary = evaluate_survival_stratification(df_valid, risk_col="risk_score")
        # C-index is threshold-independent, should be the same across rows
        c_indices = summary["c_index"].unique()
        assert len(c_indices) == 1

    def test_custom_thresholds(self) -> None:
        df = _make_surv_df()
        df["risk_score"] = np.random.RandomState(0).uniform(0, 1, len(df))
        summary = evaluate_survival_stratification(
            df,
            risk_col="risk_score",
            thresholds={"my_cut": 0.5},
        )
        assert summary["strategy"].iloc[0] == "my_cut"

    def test_expected_columns(self) -> None:
        df = _make_surv_df()
        df["risk_score"] = np.random.RandomState(0).uniform(0, 1, len(df))
        summary = evaluate_survival_stratification(df, risk_col="risk_score")
        expected_cols = {
            "strategy", "threshold", "c_index", "logrank_p",
            "logrank_stat", "n_low_risk", "n_high_risk", "n_total",
        }
        assert expected_cols == set(summary.columns)


# ── plot_km_by_group ───────────────────────────────────────────────────


class TestPlotKmByGroup:
    def test_basic_plot(self, tmp_path) -> None:
        df = _make_surv_df()
        df["risk_group"] = assign_risk_groups(
            np.random.RandomState(0).uniform(0, 1, len(df)), 0.5
        )
        fig = plot_km_by_group(
            df,
            output_path=tmp_path / "km.svg",
            c_index=0.65,
            logrank_p=0.001,
        )
        assert (tmp_path / "km.svg").exists()

    def test_wrong_number_of_groups_raises(self) -> None:
        df = _make_surv_df()
        df["risk_group"] = "same"
        with pytest.raises(ValueError, match="exactly 2"):
            plot_km_by_group(df)
