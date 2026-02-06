"""Smoke tests for multi-task AttnMIL model and dataset."""

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from stamp.modeling.models.attn_mil_multitask import AttnMILMultiTask
from stamp.modeling.multitask.dataset import MultitaskBagDataset
from stamp.modeling.multitask.lightning_module import (
    LitAttnMILMultiTask,
    ZScoreNormalizer,
)


class TestAttnMILMultiTask:
    """Test the AttnMILMultiTask model."""

    def test_forward_batched(self):
        """Test forward pass with batched input (B, n, d)."""
        model = AttnMILMultiTask(
            in_dim=512,
            emb_dim=128,
            head_dims={"hrd": 1, "tmb": 1, "clovar": 4},
        )
        x = torch.randn(4, 100, 512)
        outputs = model(x)

        assert "hrd" in outputs
        assert "tmb" in outputs
        assert "clovar" in outputs
        assert "attn" in outputs

        assert outputs["hrd"].shape == (4, 1)
        assert outputs["tmb"].shape == (4, 1)
        assert outputs["clovar"].shape == (4, 4)
        assert outputs["attn"].shape == (4, 100)

        # Attention should sum to ~1 per sample
        attn_sum = outputs["attn"].sum(dim=1)
        assert torch.allclose(attn_sum, torch.ones(4), atol=1e-5)

    def test_forward_single(self):
        """Test forward pass with single bag (n, d)."""
        model = AttnMILMultiTask(
            in_dim=256,
            emb_dim=64,
            head_dims={"hrd": 1, "tmb": 1},
        )
        x = torch.randn(50, 256)
        outputs = model(x)

        assert outputs["hrd"].shape == (1, 1)
        assert outputs["tmb"].shape == (1, 1)
        assert outputs["attn"].shape == (1, 50)


class TestZScoreNormalizer:
    """Test z-score normalization."""

    def test_fit_transform(self):
        targets = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        normalizer = ZScoreNormalizer().fit(targets)

        transformed = normalizer.transform(targets)
        # Mean should be ~0, std ~1
        assert torch.allclose(transformed.mean(dim=0), torch.zeros(2), atol=1e-5)

        # Inverse transform should recover original
        recovered = normalizer.inverse_transform(transformed)
        assert torch.allclose(recovered, targets, atol=1e-5)

    def test_save_load(self, tmp_path):
        targets = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        normalizer = ZScoreNormalizer().fit(targets)
        normalizer.save(tmp_path / "stats.json")

        loaded = ZScoreNormalizer.load(tmp_path / "stats.json")
        assert torch.allclose(normalizer.mean, loaded.mean)
        assert torch.allclose(normalizer.std, loaded.std)


class TestMultitaskBagDataset:
    """Test the MultitaskBagDataset."""

    @pytest.fixture
    def sample_data(self, tmp_path):
        """Create minimal sample data for testing."""
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()

        # Create H5 feature files
        for i in range(5):
            fname = f"slide_{i}.h5"
            with h5py.File(feature_dir / fname, "w") as f:
                f.create_dataset("feats", data=np.random.randn(100, 256).astype(np.float32))

        # Create slide table
        slide_df = pd.DataFrame(
            {
                "PATIENT": [f"P{i // 2}" for i in range(5)],
                "FILENAME": [f"slide_{i}.h5" for i in range(5)],
            }
        )

        # Create clini table (only 3 patients: P0, P1, P2)
        clini_df = pd.DataFrame(
            {
                "PATIENT": ["P0", "P1", "P2"],
                "scarHRD": [10.5, 20.3, 15.1],
                "TMB": [5.0, 12.0, 8.0],
            }
        ).set_index("PATIENT")

        return slide_df, clini_df, str(feature_dir)

    def test_dataset_len(self, sample_data):
        slide_df, clini_df, feature_dir = sample_data
        ds = MultitaskBagDataset(
            slide_table=slide_df,
            clini_df=clini_df,
            feature_dir=feature_dir,
            target_labels=["scarHRD", "TMB"],
            bag_size=64,
        )
        assert len(ds) == 3  # P0, P1, P2

    def test_dataset_getitem(self, sample_data):
        slide_df, clini_df, feature_dir = sample_data
        ds = MultitaskBagDataset(
            slide_table=slide_df,
            clini_df=clini_df,
            feature_dir=feature_dir,
            target_labels=["scarHRD", "TMB"],
            bag_size=64,
        )
        X, target = ds[0]
        assert X.shape == (64, 256)
        assert target.shape == (2,)
        assert X.dtype == torch.float32

    def test_h5_key_autodetect(self, tmp_path):
        """Test that H5 key auto-detection works."""
        feature_dir = tmp_path / "features2"
        feature_dir.mkdir()

        # Create H5 with 'patch_embeddings' key
        with h5py.File(feature_dir / "s1.h5", "w") as f:
            f.create_dataset(
                "patch_embeddings", data=np.random.randn(50, 128).astype(np.float32)
            )

        slide_df = pd.DataFrame({"PATIENT": ["P0"], "FILENAME": ["s1.h5"]})
        clini_df = pd.DataFrame({"PATIENT": ["P0"], "val": [1.0]}).set_index("PATIENT")

        ds = MultitaskBagDataset(
            slide_table=slide_df,
            clini_df=clini_df,
            feature_dir=str(feature_dir),
            target_labels=["val"],
            bag_size=32,
        )
        X, target = ds[0]
        assert X.shape == (32, 128)


class TestLitAttnMILMultiTask:
    """Test the Lightning module."""

    def test_training_step(self):
        model = LitAttnMILMultiTask(
            in_dim=128,
            emb_dim=64,
            target_labels=["hrd", "tmb"],
            head_dims={"hrd": 1, "tmb": 1},
            loss_weights={"hrd": 1.0, "tmb": 0.1},
            total_steps=100,
        )
        batch = (torch.randn(4, 50, 128), torch.randn(4, 2))
        loss = model.training_step(batch, 0)
        assert loss.ndim == 0  # scalar
        assert loss.item() > 0
