"""Lightning module for multi-task Attention MIL training.

Provides:
- Weighted multi-task Huber/MSE loss
- Per-fold z-score normalization (fit on train, apply on val/test)
- Per-head metric logging
"""

import json
import logging
from pathlib import Path

import lightning
import torch
from torch import Tensor, nn, optim

from stamp.modeling.models.attn_mil_multitask import AttnMILMultiTask

_logger = logging.getLogger("stamp")


class ZScoreNormalizer:
    """Z-score normalizer that fits on training data and transforms any split.

    Stores per-target mean and std, with safeguards against zero std.
    """

    def __init__(self) -> None:
        self.mean: Tensor | None = None
        self.std: Tensor | None = None

    def fit(self, targets: Tensor) -> "ZScoreNormalizer":
        """Fit on training targets. targets: (N, D)."""
        self.mean = targets.mean(dim=0)
        self.std = targets.std(dim=0).clamp(min=1e-8)
        return self

    def transform(self, targets: Tensor) -> Tensor:
        """Apply z-score normalization."""
        if self.mean is None or self.std is None:
            raise RuntimeError("ZScoreNormalizer not fitted yet.")
        return (targets - self.mean.to(targets.device)) / self.std.to(targets.device)

    def inverse_transform(self, targets: Tensor) -> Tensor:
        """Reverse z-score normalization."""
        if self.mean is None or self.std is None:
            raise RuntimeError("ZScoreNormalizer not fitted yet.")
        return targets * self.std.to(targets.device) + self.mean.to(targets.device)

    def save(self, path: Path) -> None:
        """Save mean/std to JSON."""
        if self.mean is None or self.std is None:
            raise RuntimeError("ZScoreNormalizer not fitted yet.")
        data = {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ZScoreNormalizer":
        """Load mean/std from JSON."""
        with open(path) as f:
            data = json.load(f)
        normalizer = cls()
        normalizer.mean = torch.tensor(data["mean"])
        normalizer.std = torch.tensor(data["std"])
        return normalizer


class LitAttnMILMultiTask(lightning.LightningModule):
    """Lightning module for multi-task AttnMIL regression.

    Args:
        in_dim: Input feature dimension per tile.
        emb_dim: Embedding dimension for attention MIL.
        target_labels: Ordered list of target column names.
        head_dims: Dict mapping head name to output dimension.
        loss_weights: Dict mapping target name to loss weight.
        loss_type: 'huber' or 'mse'.
        huber_delta: Delta parameter for Huber loss.
        total_steps: Total optimizer steps (for OneCycleLR).
        max_lr: Max learning rate.
        div_factor: LR scheduler div factor.
        normalizer: Optional z-score normalizer (applied to targets).
    """

    def __init__(
        self,
        *,
        in_dim: int,
        emb_dim: int = 256,
        target_labels: list[str],
        head_dims: dict[str, int],
        loss_weights: dict[str, float],
        loss_type: str = "huber",
        huber_delta: float = 1.0,
        total_steps: int = 1000,
        max_lr: float = 1e-4,
        div_factor: float = 25.0,
        normalizer: ZScoreNormalizer | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["normalizer"])

        self.model = AttnMILMultiTask(
            in_dim=in_dim, emb_dim=emb_dim, head_dims=head_dims
        )
        self.target_labels = target_labels
        self.loss_weights = loss_weights
        self.loss_type = loss_type
        self.huber_delta = huber_delta
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.div_factor = div_factor
        self.normalizer = normalizer

        # Build head → target index mapping
        # Each head maps to a slice of the target vector
        self._head_slices: dict[str, tuple[int, int]] = {}
        offset = 0
        for name, dim in head_dims.items():
            self._head_slices[name] = (offset, offset + dim)
            offset += dim

    def _compute_head_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute loss for a single head."""
        if self.loss_type == "huber":
            return nn.functional.huber_loss(
                pred, target, delta=self.huber_delta, reduction="mean"
            )
        else:
            return nn.functional.mse_loss(pred, target, reduction="mean")

    def _step(
        self, batch: tuple[Tensor, Tensor], step_name: str
    ) -> Tensor:
        X, targets = batch  # X: (B, n, d), targets: (B, D)

        # Optionally normalize targets
        if self.normalizer is not None:
            targets = self.normalizer.transform(targets)

        outputs = self.model(X)

        total_loss = torch.tensor(0.0, device=X.device)
        for name, (start, end) in self._head_slices.items():
            pred = outputs[name]  # (B, dim)
            target_slice = targets[:, start:end]  # (B, dim)
            head_loss = self._compute_head_loss(pred, target_slice)
            weight = self.loss_weights.get(name, 1.0)
            total_loss = total_loss + weight * head_loss

            self.log(
                f"{step_name}_loss_{name}",
                head_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

        self.log(
            f"{step_name}_loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return total_loss

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        return self._step(batch, "training")

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        return self._step(batch, "validation")

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            total_steps=self.total_steps,
            max_lr=self.max_lr,
            div_factor=self.div_factor,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def on_train_batch_end(self, outputs, batch, batch_idx):
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log(
            "learning_rate",
            current_lr,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
