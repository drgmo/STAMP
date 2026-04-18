"""Unit tests for stamp.modeling.prompt_masks and BagDataset mask ingestion."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from stamp.modeling.prompt_masks import (
    MaskMode,
    group_for_fold,
    open_prompt_masks,
)


def _make_per_fold_h5(path: Path, n_folds: int = 3, slides=("s1", "s2")) -> None:
    with h5py.File(path, "w") as f:
        f.attrs["mask_mode"] = "per_fold"
        f.attrs["n_folds"] = n_folds
        f.attrs["random_state"] = 0
        f.attrs["judge_mode"] = "direction"
        f.attrs["encoder_name"] = "plip"
        for i in range(n_folds):
            grp = f.create_group(f"fold_{i}")
            for s in slides:
                grp.create_dataset(
                    s, data=np.linspace(0.0, 1.0, 16, dtype=np.float32)
                )


def _make_shared_h5(path: Path, slides=("s1",)) -> None:
    with h5py.File(path, "w") as f:
        f.attrs["mask_mode"] = "shared"
        f.attrs["judge_mode"] = "text_judge"
        grp = f.create_group("shared")
        for s in slides:
            grp.create_dataset(
                s, data=np.ones(16, dtype=np.float32) * 0.5
            )


def test_open_prompt_masks_per_fold(tmp_path: Path) -> None:
    p = tmp_path / "pm.h5"
    _make_per_fold_h5(p, n_folds=4)
    h = open_prompt_masks(p)
    assert h.mode is MaskMode.PER_FOLD
    assert h.n_folds == 4
    assert h.judge_mode == "direction"
    assert group_for_fold(h, 0) == "fold_0"
    assert group_for_fold(h, 3) == "fold_3"
    with pytest.raises(RuntimeError):
        group_for_fold(h, 4)


def test_open_prompt_masks_shared(tmp_path: Path) -> None:
    p = tmp_path / "pm.h5"
    _make_shared_h5(p)
    h = open_prompt_masks(p)
    assert h.mode is MaskMode.SHARED
    assert h.n_folds is None
    assert group_for_fold(h, 0) == "shared"
    assert group_for_fold(h, 99) == "shared"


def test_per_fold_without_n_folds_raises(tmp_path: Path) -> None:
    p = tmp_path / "pm.h5"
    with h5py.File(p, "w") as f:
        f.attrs["mask_mode"] = "per_fold"
        # deliberately no n_folds
    with pytest.raises(RuntimeError, match="n_folds"):
        open_prompt_masks(p)


def test_bag_dataset_feature_weight(tmp_path: Path) -> None:
    """BagDataset applies mask as elementwise feature weight when configured."""
    from stamp.modeling.data import BagDataset

    # Synthetic bag: one slide with 4 patches × 3-dim features.
    bag_path = tmp_path / "s1.h5"
    feats_np = np.arange(12, dtype=np.float32).reshape(4, 3)
    coords_um = np.zeros((4, 2), dtype=np.float32)
    with h5py.File(bag_path, "w") as f:
        f.create_dataset("feats", data=feats_np)
        ds = f.create_dataset("coords", data=coords_um)
        ds.attrs["unit"] = "um"
        f.attrs["tile_size_um"] = 224.0

    # Mask = [1, 0, 1, 0] under fold_0.
    mask_path = tmp_path / "pm.h5"
    mask = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    with h5py.File(mask_path, "w") as f:
        f.attrs["mask_mode"] = "per_fold"
        f.attrs["n_folds"] = 1
        grp = f.create_group("fold_0")
        grp.create_dataset("s1", data=mask)

    ds = BagDataset(
        bags=[[bag_path]],
        ground_truths=torch.zeros((1, 1)),
        transform=None,
        prompt_masks_path=mask_path,
        mask_group="fold_0",
        mask_application="feature_weight",
    )
    feats, _coords, _bs, _gt = ds[0]
    got = feats.numpy()
    expected = feats_np * mask[:, None]
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_bag_dataset_without_mask_is_identity(tmp_path: Path) -> None:
    from stamp.modeling.data import BagDataset

    bag_path = tmp_path / "s1.h5"
    feats_np = np.random.default_rng(0).normal(size=(5, 4)).astype(np.float32)
    coords_um = np.zeros((5, 2), dtype=np.float32)
    with h5py.File(bag_path, "w") as f:
        f.create_dataset("feats", data=feats_np)
        ds = f.create_dataset("coords", data=coords_um)
        ds.attrs["unit"] = "um"
        f.attrs["tile_size_um"] = 224.0

    ds = BagDataset(
        bags=[[bag_path]],
        ground_truths=torch.zeros((1, 1)),
        transform=None,
    )
    feats, _c, _bs, _gt = ds[0]
    np.testing.assert_allclose(feats.numpy(), feats_np, atol=1e-6)
