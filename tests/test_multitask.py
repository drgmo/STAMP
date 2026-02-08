"""Tests for multitask training and cross-validation."""

import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from stamp.modeling.config import MultitaskCrossvalConfig, MultitaskTrainingConfig
from stamp.modeling.multitask import (
    LitAttnMILMultiTask,
    MultitaskDataset,
    crossval_multitask_,
    train_multitask_,
)
from stamp.seed import Seed


def _create_multitask_dataset(
    tmp_path: Path,
    n_patients: int = 30,
    feat_dim: int = 8,
    n_tiles: int = 16,
    target_names: list[str] | None = None,
) -> tuple[Path, Path, Path]:
    """Helper: create a synthetic multitask regression dataset.

    Args:
        target_names: column names for the regression targets.
            Defaults to ``["target_A", "target_B"]``.
    """
    if target_names is None:
        target_names = ["target_A", "target_B"]

    feat_dir = tmp_path / "feats"
    feat_dir.mkdir()
    clini_path = tmp_path / "clini.csv"

    rows = []
    for i in range(n_patients):
        pid = f"PAT_{i:04d}"
        fpath = feat_dir / f"{pid}.h5"
        with h5py.File(fpath, "w") as h5:
            h5["feats"] = np.random.randn(n_tiles, feat_dim).astype(np.float32)
            h5.attrs["stamp_version"] = "2.4.0"
            h5.attrs["feat_type"] = "patient"
        row: dict[str, object] = {"PATIENT": pid}
        for name in target_names:
            row[name] = float(np.random.uniform(-10, 10))
        rows.append(row)
    pd.DataFrame(rows).to_csv(clini_path, index=False)
    return clini_path, feat_dir, tmp_path / "output"


def _create_multitask_dataset_in_subdirs(
    tmp_path: Path,
    n_patients: int = 30,
    feat_dim: int = 8,
    n_tiles: int = 16,
    target_names: list[str] | None = None,
) -> tuple[Path, Path, Path]:
    """Like ``_create_multitask_dataset`` but places feature files in subdirectories.

    Patients are distributed across ``sub_a/`` and ``sub_b/`` subdirectories
    inside the feature directory.
    """
    if target_names is None:
        target_names = ["target_A", "target_B"]

    feat_dir = tmp_path / "feats"
    feat_dir.mkdir()
    sub_a = feat_dir / "sub_a"
    sub_b = feat_dir / "sub_b"
    sub_a.mkdir()
    sub_b.mkdir()
    clini_path = tmp_path / "clini.csv"

    rows = []
    for i in range(n_patients):
        pid = f"PAT_{i:04d}"
        # Alternate between subdirectories
        sub = sub_a if i % 2 == 0 else sub_b
        fpath = sub / f"{pid}.h5"
        with h5py.File(fpath, "w") as h5:
            h5["feats"] = np.random.randn(n_tiles, feat_dim).astype(np.float32)
            h5.attrs["stamp_version"] = "2.4.0"
            h5.attrs["feat_type"] = "patient"
        row: dict[str, object] = {"PATIENT": pid}
        for name in target_names:
            row[name] = float(np.random.uniform(-10, 10))
        rows.append(row)
    pd.DataFrame(rows).to_csv(clini_path, index=False)
    return clini_path, feat_dir, tmp_path / "output"


class TestMultitaskDataset:
    def test_basic_loading(self, tmp_path: Path) -> None:
        clini_path, feat_dir, _ = _create_multitask_dataset(tmp_path, n_patients=5)
        df = pd.read_csv(clini_path)

        feature_files = [feat_dir / f"{pid}.h5" for pid in df["PATIENT"]]
        targets = torch.tensor(
            df[["target_A", "target_B"]].values, dtype=torch.float32
        )
        ds = MultitaskDataset(feature_files=feature_files, targets=targets, bag_size=8)

        assert len(ds) == 5
        feats, tgt = ds[0]
        assert feats.shape == (8, 8)  # bag_size=8, feat_dim=8
        assert tgt.shape == (2,)

    def test_no_bag_size(self, tmp_path: Path) -> None:
        clini_path, feat_dir, _ = _create_multitask_dataset(
            tmp_path, n_patients=3, n_tiles=10
        )
        df = pd.read_csv(clini_path)
        feature_files = [feat_dir / f"{pid}.h5" for pid in df["PATIENT"]]
        targets = torch.tensor(
            df[["target_A", "target_B"]].values, dtype=torch.float32
        )
        ds = MultitaskDataset(feature_files=feature_files, targets=targets, bag_size=None)

        feats, _ = ds[0]
        assert feats.shape[0] == 10  # all tiles returned


class TestLitAttnMILMultiTask:
    def test_forward(self) -> None:
        model = LitAttnMILMultiTask(
            target_names=["A", "B"],
            target_dims=[1, 1],
            in_dim=8,
            emb_dim=16,
            total_steps=10,
        )
        x = torch.randn(4, 12, 8)  # batch=4, tiles=12, feat_dim=8
        out = model(x)
        assert set(out.keys()) == {"A", "B"}
        assert out["A"].shape == (4, 1)

    def test_loss_computation(self) -> None:
        model = LitAttnMILMultiTask(
            target_names=["A", "B"],
            target_dims=[1, 1],
            loss_weights={"A": 2.0, "B": 0.5},
            in_dim=8,
            emb_dim=16,
            total_steps=10,
        )
        x = torch.randn(4, 12, 8)
        targets = torch.randn(4, 2)
        preds = model(x)
        loss, per_head = model._compute_loss(preds, targets)
        assert loss.ndim == 0  # scalar
        assert "A" in per_head and "B" in per_head


@pytest.mark.slow
def test_train_multitask_single(tmp_path: Path) -> None:
    """Integration test: single train/val split."""
    Seed.set(42)
    clini_path, feat_dir, output_dir = _create_multitask_dataset(tmp_path, n_patients=20)

    config = MultitaskTrainingConfig(
        output_dir=output_dir,
        clini_table=clini_path,
        feature_dir=feat_dir,
        patient_label="PATIENT",
        target_labels={"target_A": 1, "target_B": 1},
        emb_dim=16,
        bag_size=8,
        batch_size=4,
        max_epochs=2,
        patience=1,
        accelerator="cpu",
        seed=42,
    )
    train_multitask_(config)
    assert (output_dir / "model.ckpt").exists()


@pytest.mark.slow
def test_crossval_multitask(tmp_path: Path) -> None:
    """Integration test: K-fold cross-validation."""
    Seed.set(42)
    clini_path, feat_dir, output_dir = _create_multitask_dataset(tmp_path, n_patients=20)

    config = MultitaskTrainingConfig(
        output_dir=output_dir,
        clini_table=clini_path,
        feature_dir=feat_dir,
        patient_label="PATIENT",
        target_labels={"target_A": 1, "target_B": 1},
        emb_dim=16,
        bag_size=8,
        batch_size=4,
        max_epochs=2,
        patience=1,
        accelerator="cpu",
        seed=42,
        crossval=MultitaskCrossvalConfig(
            enabled=True,
            n_splits=2,
            shuffle=True,
            random_state=42,
        ),
    )
    crossval_multitask_(config)

    # Verify outputs
    assert (output_dir / "crossval_summary.json").exists()
    summary = json.loads((output_dir / "crossval_summary.json").read_text())
    assert summary["n_splits"] == 2
    assert len(summary["folds"]) == 2
    assert summary["mean_val_loss"] is not None

    # Check fold directories
    for fold_i in range(2):
        fold_dir = output_dir / f"fold_{fold_i}"
        assert (fold_dir / "model.ckpt").exists()
        assert (fold_dir / "fold_split.json").exists()
        assert (fold_dir / "metrics.json").exists()
        assert (fold_dir / "patient-preds.csv").exists()

        # Verify patient-preds.csv content
        preds_df = pd.read_csv(fold_dir / "patient-preds.csv")
        assert config.patient_label in preds_df.columns
        for target_name in config.target_labels:
            assert f"{target_name}_true" in preds_df.columns
            assert f"{target_name}_pred" in preds_df.columns
        assert len(preds_df) > 0

        # Verify predicted patients match the val set in the split
        split = json.loads((fold_dir / "fold_split.json").read_text())
        val_set = set(split["val_patients"])
        pred_pids = set(preds_df[config.patient_label].tolist())
        assert pred_pids == val_set, "Predicted patients should match val set exactly!"

    # Verify no patient overlap between train/val in each fold
    for fold_i in range(2):
        split = json.loads((output_dir / f"fold_{fold_i}" / "fold_split.json").read_text())
        train_set = set(split["train_patients"])
        val_set = set(split["val_patients"])
        assert train_set.isdisjoint(val_set), "Train/val patient overlap detected!"


@pytest.mark.slow
def test_crossval_disabled_falls_back_to_single(tmp_path: Path) -> None:
    """When crossval is disabled, crossval_multitask_ should fall back to single train."""
    Seed.set(42)
    clini_path, feat_dir, output_dir = _create_multitask_dataset(tmp_path, n_patients=20)

    config = MultitaskTrainingConfig(
        output_dir=output_dir,
        clini_table=clini_path,
        feature_dir=feat_dir,
        patient_label="PATIENT",
        target_labels={"target_A": 1, "target_B": 1},
        emb_dim=16,
        bag_size=8,
        batch_size=4,
        max_epochs=2,
        patience=1,
        accelerator="cpu",
        seed=42,
        crossval=MultitaskCrossvalConfig(enabled=False),
    )
    crossval_multitask_(config)
    assert (output_dir / "model.ckpt").exists()


@pytest.mark.slow
def test_crossval_multitask_many_targets(tmp_path: Path) -> None:
    """Verify that an arbitrary number of targets (>2) works end-to-end."""
    Seed.set(42)
    targets = ["scarHRD", "TMB", "CLOVAR_D", "CLOVAR_I", "CLOVAR_M"]
    clini_path, feat_dir, output_dir = _create_multitask_dataset(
        tmp_path, n_patients=20, target_names=targets
    )

    config = MultitaskTrainingConfig(
        output_dir=output_dir,
        clini_table=clini_path,
        feature_dir=feat_dir,
        patient_label="PATIENT",
        target_labels={t: 1 for t in targets},
        emb_dim=16,
        bag_size=8,
        batch_size=4,
        max_epochs=2,
        patience=1,
        accelerator="cpu",
        seed=42,
        crossval=MultitaskCrossvalConfig(
            enabled=True,
            n_splits=2,
            shuffle=True,
            random_state=42,
        ),
    )
    crossval_multitask_(config)

    # Every fold must have patient-preds.csv with columns for all 5 targets
    for fold_i in range(2):
        fold_dir = output_dir / f"fold_{fold_i}"
        assert (fold_dir / "patient-preds.csv").exists()
        preds_df = pd.read_csv(fold_dir / "patient-preds.csv")
        assert config.patient_label in preds_df.columns
        for target_name in targets:
            assert f"{target_name}_true" in preds_df.columns
            assert f"{target_name}_pred" in preds_df.columns
        assert len(preds_df) > 0


@pytest.mark.slow
def test_crossval_multitask_features_in_subdirs(tmp_path: Path) -> None:
    """Feature files in subdirectories of feature_dir should be found."""
    Seed.set(42)
    clini_path, feat_dir, output_dir = _create_multitask_dataset_in_subdirs(
        tmp_path, n_patients=20
    )

    config = MultitaskTrainingConfig(
        output_dir=output_dir,
        clini_table=clini_path,
        feature_dir=feat_dir,
        patient_label="PATIENT",
        target_labels={"target_A": 1, "target_B": 1},
        emb_dim=16,
        bag_size=8,
        batch_size=4,
        max_epochs=2,
        patience=1,
        accelerator="cpu",
        seed=42,
        crossval=MultitaskCrossvalConfig(
            enabled=True,
            n_splits=2,
            shuffle=True,
            random_state=42,
        ),
    )
    crossval_multitask_(config)

    for fold_i in range(2):
        fold_dir = output_dir / f"fold_{fold_i}"
        assert (fold_dir / "patient-preds.csv").exists()
        preds_df = pd.read_csv(fold_dir / "patient-preds.csv")
        assert len(preds_df) > 0
        for target_name in config.target_labels:
            assert f"{target_name}_true" in preds_df.columns
            assert f"{target_name}_pred" in preds_df.columns


def _create_multitask_dataset_with_slide_table(
    tmp_path: Path,
    n_patients: int = 20,
    feat_dim: int = 8,
    n_tiles: int = 16,
    target_names: list[str] | None = None,
) -> tuple[Path, Path, Path, Path]:
    """Create a dataset where feature files are in subfolders and named by slide ID.

    Returns (clini_path, slide_path, feat_dir, output_dir).
    Feature files are named ``SLIDE_xxxx.h5`` in subfolder ``encoder/``
    and a slide table maps FILENAME → PATIENT.
    """
    if target_names is None:
        target_names = ["target_A", "target_B"]

    feat_dir = tmp_path / "feats"
    feat_dir.mkdir()
    encoder_dir = feat_dir / "encoder"
    encoder_dir.mkdir()
    clini_path = tmp_path / "clini.csv"
    slide_path = tmp_path / "slide.csv"

    clini_rows = []
    slide_rows = []
    for i in range(n_patients):
        pid = f"PAT_{i:04d}"
        slide_name = f"SLIDE_{i:04d}"
        fname = f"encoder/{slide_name}.h5"
        fpath = feat_dir / fname
        with h5py.File(fpath, "w") as h5:
            h5["feats"] = np.random.randn(n_tiles, feat_dim).astype(np.float32)
            h5.attrs["stamp_version"] = "2.4.0"
            h5.attrs["feat_type"] = "patient"
        clini_row: dict[str, object] = {"PATIENT": pid}
        for name in target_names:
            clini_row[name] = float(np.random.uniform(-10, 10))
        clini_rows.append(clini_row)
        slide_rows.append({"PATIENT": pid, "FILENAME": fname})

    pd.DataFrame(clini_rows).to_csv(clini_path, index=False)
    pd.DataFrame(slide_rows).to_csv(slide_path, index=False)
    return clini_path, slide_path, feat_dir, tmp_path / "output"


@pytest.mark.slow
def test_crossval_multitask_with_slide_table(tmp_path: Path) -> None:
    """Feature files named by slide ID in subfolders, mapped via slide_table."""
    Seed.set(42)
    clini_path, slide_path, feat_dir, output_dir = (
        _create_multitask_dataset_with_slide_table(tmp_path, n_patients=20)
    )

    config = MultitaskTrainingConfig(
        output_dir=output_dir,
        clini_table=clini_path,
        slide_table=slide_path,
        feature_dir=feat_dir,
        patient_label="PATIENT",
        filename_label="FILENAME",
        target_labels={"target_A": 1, "target_B": 1},
        emb_dim=16,
        bag_size=8,
        batch_size=4,
        max_epochs=2,
        patience=1,
        accelerator="cpu",
        seed=42,
        crossval=MultitaskCrossvalConfig(
            enabled=True,
            n_splits=2,
            shuffle=True,
            random_state=42,
        ),
    )
    crossval_multitask_(config)

    for fold_i in range(2):
        fold_dir = output_dir / f"fold_{fold_i}"
        assert (fold_dir / "patient-preds.csv").exists()
        preds_df = pd.read_csv(fold_dir / "patient-preds.csv")
        assert len(preds_df) > 0
        for target_name in config.target_labels:
            assert f"{target_name}_true" in preds_df.columns
            assert f"{target_name}_pred" in preds_df.columns
