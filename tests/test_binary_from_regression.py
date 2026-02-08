"""Tests for binary_from_regression evaluation helpers."""

import numpy as np
import pandas as pd
import pytest

from stamp.statistics.binary_from_regression import (
    aggregate_patient_scores,
    binarize_labels,
    evaluate_at_threshold,
    evaluate_thresholds,
    extract_score,
    get_thresholds,
    postprocess_score,
)


# ── binarize_labels ────────────────────────────────────────────────────


class TestBinarizeLabels:
    def test_bool_input(self) -> None:
        y = np.array([True, False, True, False])
        result = binarize_labels(y)
        np.testing.assert_array_equal(result, [1, 0, 1, 0])

    def test_numeric_01(self) -> None:
        result = binarize_labels([0, 1, 1, 0])
        np.testing.assert_array_equal(result, [0, 1, 1, 0])

    def test_numeric_float_01(self) -> None:
        result = binarize_labels([0.0, 1.0, 1.0, 0.0])
        np.testing.assert_array_equal(result, [0, 1, 1, 0])

    def test_numeric_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="outside"):
            binarize_labels([0, 1, 2])

    def test_string_pos_neg(self) -> None:
        result = binarize_labels(["pos", "neg", "pos", "neg"])
        np.testing.assert_array_equal(result, [1, 0, 1, 0])

    def test_string_yes_no(self) -> None:
        result = binarize_labels(["yes", "no", "yes"])
        np.testing.assert_array_equal(result, [1, 0, 1])

    def test_string_with_positive_class(self) -> None:
        result = binarize_labels(
            ["cat", "dog", "cat"], positive_class="dog"
        )
        np.testing.assert_array_equal(result, [0, 1, 0])

    def test_string_with_negative_class(self) -> None:
        result = binarize_labels(
            ["cat", "dog", "cat"], negative_class="cat"
        )
        np.testing.assert_array_equal(result, [0, 1, 0])

    def test_string_case_insensitive(self) -> None:
        result = binarize_labels(["POS", " Neg ", "pos"])
        np.testing.assert_array_equal(result, [1, 0, 1])

    def test_three_classes_raises(self) -> None:
        with pytest.raises(ValueError, match="exactly 2"):
            binarize_labels(["a", "b", "c"])

    def test_positive_class_not_found_raises(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            binarize_labels(["a", "b"], positive_class="c")

    def test_unknown_convention_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot infer"):
            binarize_labels(["alpha", "beta"])


# ── extract_score ──────────────────────────────────────────────────────


class TestExtractScore:
    def test_scalar(self) -> None:
        result = extract_score(0.7)
        np.testing.assert_array_almost_equal(result, [0.7])

    def test_1d_array(self) -> None:
        result = extract_score(np.array([0.1, 0.9]))
        assert result.shape == (2,)

    def test_dict_single_key(self) -> None:
        result = extract_score({"score": [0.5, 0.6]})
        np.testing.assert_array_almost_equal(result, [0.5, 0.6])

    def test_dict_with_task_name(self) -> None:
        result = extract_score({"A": [0.1], "B": [0.9]}, task_name="B")
        np.testing.assert_array_almost_equal(result, [0.9])

    def test_dict_multi_key_no_name_raises(self) -> None:
        with pytest.raises(ValueError, match="task_name"):
            extract_score({"A": [0.1], "B": [0.9]})

    def test_list_single_element(self) -> None:
        result = extract_score([[0.1, 0.2]])
        assert len(result) == 2

    def test_list_with_index(self) -> None:
        result = extract_score([[0.1], [0.9]], task_index=1)
        np.testing.assert_array_almost_equal(result, [0.9])

    def test_list_multi_no_index_raises(self) -> None:
        with pytest.raises(ValueError, match="task_index"):
            extract_score([[0.1], [0.9]])

    def test_2d_array_single_col(self) -> None:
        arr = np.array([[0.1], [0.9]])
        result = extract_score(arr)
        np.testing.assert_array_almost_equal(result, [0.1, 0.9])

    def test_2d_array_with_index(self) -> None:
        arr = np.array([[0.1, 0.2], [0.9, 0.8]])
        result = extract_score(arr, task_index=1)
        np.testing.assert_array_almost_equal(result, [0.2, 0.8])

    def test_2d_array_multi_col_no_index_raises(self) -> None:
        with pytest.raises(ValueError, match="task_index"):
            extract_score(np.array([[0.1, 0.2], [0.9, 0.8]]))


# ── postprocess_score ──────────────────────────────────────────────────


class TestPostprocessScore:
    def test_identity(self) -> None:
        result = postprocess_score([0.3, 0.7])
        np.testing.assert_array_almost_equal(result, [0.3, 0.7])

    def test_sigmoid(self) -> None:
        result = postprocess_score([0.0], mode="sigmoid")
        np.testing.assert_array_almost_equal(result, [0.5])

    def test_clip_warning(self) -> None:
        with pytest.warns(match="outside"):
            result = postprocess_score([1.5, -0.1])
        np.testing.assert_array_almost_equal(result, [1.0, 0.0])

    def test_callable_mode(self) -> None:
        result = postprocess_score([4.0], mode=lambda x: x / 10.0)
        np.testing.assert_array_almost_equal(result, [0.4])

    def test_nan_raises(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            postprocess_score([float("nan")])

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown mode"):
            postprocess_score([0.5], mode="softmax")


# ── aggregate_patient_scores ───────────────────────────────────────────


class TestAggregatePatientScores:
    def test_basic_mean(self) -> None:
        df = pd.DataFrame(
            {
                "patient_id": ["A", "A", "B", "B"],
                "y_raw": ["pos", "pos", "neg", "neg"],
                "y_score": [0.8, 0.6, 0.2, 0.4],
            }
        )
        result = aggregate_patient_scores(df)
        assert len(result) == 2
        a_row = result[result["patient_id"] == "A"]
        np.testing.assert_almost_equal(a_row["y_score"].values[0], 0.7)

    def test_median(self) -> None:
        df = pd.DataFrame(
            {
                "patient_id": ["A", "A", "A"],
                "y_raw": ["pos", "pos", "pos"],
                "y_score": [0.1, 0.5, 0.9],
            }
        )
        result = aggregate_patient_scores(df, method="median")
        np.testing.assert_almost_equal(result["y_score"].values[0], 0.5)

    def test_inconsistent_labels_raises(self) -> None:
        df = pd.DataFrame(
            {
                "patient_id": ["A", "A"],
                "y_raw": ["pos", "neg"],
                "y_score": [0.8, 0.6],
            }
        )
        with pytest.raises(ValueError, match="inconsistent"):
            aggregate_patient_scores(df)

    def test_missing_id_col_raises(self) -> None:
        df = pd.DataFrame({"y_raw": [1], "y_score": [0.5]})
        with pytest.raises(ValueError, match="not found"):
            aggregate_patient_scores(df)


# ── get_thresholds ─────────────────────────────────────────────────────


class TestGetThresholds:
    def test_returns_ordered_dict(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.4, 0.6, 0.9])
        result = get_thresholds(y_true, y_score)
        assert isinstance(result, dict)
        assert "youden" in result
        assert "mean_score" in result
        assert "median_score" in result
        assert "fixed_0.50" in result
        assert "q25" in result
        assert "q75" in result

    def test_all_values_in_01(self) -> None:
        y_true = np.array([0, 1])
        y_score = np.array([0.2, 0.8])
        result = get_thresholds(y_true, y_score)
        for v in result.values():
            assert 0.0 <= v <= 1.0

    def test_youden_is_reasonable(self) -> None:
        # Perfect separation: youden threshold should sit between the two groups
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        result = get_thresholds(y_true, y_score, include=("youden",),
                                fixed_thresholds=(), quantiles=())
        assert 0.3 <= result["youden"] <= 0.7


# ── evaluate_at_threshold ──────────────────────────────────────────────


class TestEvaluateAtThreshold:
    def test_perfect_separation(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        result = evaluate_at_threshold(y_true, y_score, 0.5)
        assert result["accuracy"] == 1.0
        assert result["f1"] == 1.0
        assert result["sensitivity"] == 1.0
        assert result["specificity"] == 1.0
        assert result["tp"] == 2
        assert result["tn"] == 2
        assert result["fp"] == 0
        assert result["fn"] == 0

    def test_all_wrong(self) -> None:
        y_true = np.array([1, 1, 0, 0])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        result = evaluate_at_threshold(y_true, y_score, 0.5)
        assert result["accuracy"] == 0.0
        assert result["sensitivity"] == 0.0
        assert result["specificity"] == 0.0


# ── evaluate_thresholds ────────────────────────────────────────────────


class TestEvaluateThresholds:
    def test_returns_two_dataframes(self) -> None:
        df = pd.DataFrame(
            {
                "patient_id": ["A", "B", "C", "D"],
                "y_true": [0, 0, 1, 1],
                "y_score": [0.1, 0.4, 0.6, 0.9],
            }
        )
        summary, preds = evaluate_thresholds(df, id_col="patient_id")
        assert isinstance(summary, pd.DataFrame)
        assert isinstance(preds, pd.DataFrame)
        assert "strategy" in summary.columns
        assert "threshold" in summary.columns
        assert "accuracy" in summary.columns
        assert "auroc" in summary.columns
        assert len(summary) > 0
        # preds should have y_pred_* columns
        pred_cols = [c for c in preds.columns if c.startswith("y_pred_")]
        assert len(pred_cols) == len(summary)

    def test_custom_thresholds(self) -> None:
        df = pd.DataFrame({"y_true": [0, 1], "y_score": [0.3, 0.7]})
        thresholds = {"low": 0.2, "high": 0.8}
        summary, preds = evaluate_thresholds(df, thresholds=thresholds)
        assert set(summary["strategy"]) == {"low", "high"}

    def test_auroc_consistent_across_strategies(self) -> None:
        df = pd.DataFrame(
            {
                "y_true": [0, 0, 1, 1],
                "y_score": [0.1, 0.4, 0.6, 0.9],
            }
        )
        summary, _ = evaluate_thresholds(df)
        aurocs = summary["auroc"].unique()
        # AUROC is threshold-independent, should be same for all rows
        assert len(aurocs) == 1
