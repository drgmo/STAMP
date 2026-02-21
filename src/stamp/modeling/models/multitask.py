"""Multitask Lightning modules for training multiple targets simultaneously.

Each multitask module wraps a shared backbone (VisionTransformer, AttentionMIL, etc.)
and adds multiple classification heads — one per target. The combined loss is a
weighted sum of per-target cross-entropy losses. Missing targets (NaN) are masked
so that a patient missing a label for a specific target does not contribute to
that target's loss.
"""

import logging
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torchmetrics.classification import MulticlassAUROC

from stamp.modeling.models import Base, Loss
from stamp.types import (
    Bags,
    BagSizes,
    CoordinatesBatch,
)

_logger = logging.getLogger("stamp")


class LitMultitaskTileClassifier(Base):
    """Multitask tile-level classifier with shared backbone and per-target heads.

    Args:
        model_class: Backbone model class (e.g., VisionTransformer, AttentionMIL).
        target_configs: List of dicts, each with:
            - ground_truth_label: str
            - categories: list[str]
            - category_weights: Tensor
            - weight: float (loss weight)
        dim_input: Feature dimensionality per tile.
    """

    supported_features = ["tile"]

    def __init__(
        self,
        *,
        model_class: type[nn.Module],
        target_configs: list[dict],
        dim_input: int,
        **kwargs,
    ) -> None:
        super().__init__(
            model_class=model_class,
            target_configs=target_configs,
            dim_input=dim_input,
            **kwargs,
        )

        self.target_configs = target_configs
        self.n_targets = len(target_configs)

        # Total output dimension = sum of categories across all targets
        total_dim_output = sum(len(tc["categories"]) for tc in target_configs)

        # Build shared backbone (outputs total_dim_output)
        self.model: nn.Module = self._build_backbone(
            model_class, dim_input, total_dim_output, kwargs
        )

        # Store per-target metadata
        self.target_labels: list[str] = []
        self.target_categories: list[list[str]] = []
        self.target_weights: list[float] = []
        self.target_class_weights: list[Tensor] = []
        self.target_offsets: list[int] = []  # Start index into logits
        self.target_sizes: list[int] = []  # Number of classes per target
        self.valid_aurocs: nn.ModuleList = nn.ModuleList()

        offset = 0
        for tc in target_configs:
            cats = list(tc["categories"])
            self.target_labels.append(tc["ground_truth_label"])
            self.target_categories.append(cats)
            self.target_weights.append(tc.get("weight", 1.0))
            self.target_class_weights.append(tc["category_weights"])
            self.target_offsets.append(offset)
            self.target_sizes.append(len(cats))
            self.valid_aurocs.append(MulticlassAUROC(len(cats)))
            offset += len(cats)

        self.hparams.update({"task": "multitask_classification"})

    def _get_target_logits(
        self, logits: Tensor, target_idx: int
    ) -> Tensor:
        """Extract logits for a specific target from the combined output."""
        start = self.target_offsets[target_idx]
        end = start + self.target_sizes[target_idx]
        return logits[:, start:end]

    def _step(
        self,
        *,
        batch: tuple[Bags, CoordinatesBatch, BagSizes, Tensor],
        step_name: str,
    ) -> Loss:
        bags, coords, bag_sizes, targets = batch
        # targets shape: (B, total_categories) — concatenated one-hots for all targets
        # with NaN for missing targets

        logits = self.model(bags, coords=coords, mask=None)

        total_loss = torch.tensor(0.0, device=logits.device)
        offset_target = 0

        for i in range(self.n_targets):
            n_cats = self.target_sizes[i]
            target_logits = self._get_target_logits(logits, i)
            target_gt = targets[:, offset_target : offset_target + n_cats]
            offset_target += n_cats

            # Mask out samples with missing ground truth (all NaN in one-hot)
            valid_mask = ~torch.isnan(target_gt).any(dim=-1)
            if not valid_mask.any():
                continue

            valid_logits = target_logits[valid_mask]
            valid_gt = target_gt[valid_mask]

            loss = nn.functional.cross_entropy(
                valid_logits,
                valid_gt.type_as(valid_logits),
                weight=self.target_class_weights[i].type_as(valid_logits),
            )
            total_loss = total_loss + self.target_weights[i] * loss

            label = self.target_labels[i]
            self.log(
                f"{step_name}_loss_{label}",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

            if step_name == "validation":
                auroc: MulticlassAUROC = self.valid_aurocs[i]
                auroc.update(valid_logits, valid_gt.long().argmax(dim=-1))
                self.log(
                    f"{step_name}_auroc_{label}",
                    auroc,
                    on_step=False,
                    on_epoch=True,
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

    def training_step(self, batch, batch_idx) -> Loss:
        return self._step(batch=batch, step_name="training")

    def validation_step(self, batch, batch_idx) -> Loss:
        return self._step(batch=batch, step_name="validation")

    def test_step(self, batch, batch_idx) -> Loss:
        return self._step(batch=batch, step_name="test")

    def predict_step(self, batch, batch_idx):
        bags, coords, bag_sizes, _ = batch
        return self.model(bags, coords=coords, mask=None)
