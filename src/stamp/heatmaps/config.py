from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict, Field

from stamp.types import SlideMPP


class HeatmapConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: Path = Field(description="Directory to save heatmap outputs")

    feature_dir: Path = Field(description="Directory containing extracted features")
    wsi_dir: Path = Field(description="Directory containing whole slide images")
    checkpoint_path: Path | None = Field(
        default=None,
        description=(
            "Path to a single model checkpoint. Required for the "
            "`heatmaps` subcommand; ignored by `heatmaps-crossval`, which "
            "discovers per-fold checkpoints under ``crossval_dir``."
        ),
    )

    # Fold-aware heatmap rendering driven by WSIVL prompt-mask artefacts.
    crossval_dir: Path | None = Field(
        default=None,
        description=(
            "Root of a STAMP crossval run (contains split-<i>/model.ckpt). "
            "Required by the `heatmaps-crossval` subcommand; ignored otherwise."
        ),
    )
    n_folds: int | None = Field(default=None, ge=2)
    splits_path: Path | None = Field(
        default=None,
        description=(
            "Optional external splits.json. If unset, STAMP reads "
            "``crossval_dir / 'splits.json'``."
        ),
    )
    prompt_masks_path: Path | None = Field(
        default=None,
        description=(
            "HDF5 with prompt-derived tile masks (see "
            "stamp.modeling.prompt_masks). When set, features are "
            "weighted by the per-fold mask before inference."
        ),
    )

    slide_paths: list[Path] | None = Field(
        default=None,
        description="Specific slide paths to process. If None, processes all slides in wsi_dir",
    )

    device: str = Field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu",
        description="Device to use for computation",
    )

    opacity: float = Field(
        default=0.6,
        description="Overlay plot opacity. A value of 0 means transparent and 1 opaque.",
        ge=0,
        le=1,
    )

    topk: int = Field(
        default=0,
        description="Number of top patches to highlight. 0 means no highlighting.",
        ge=0,
    )

    bottomk: int = Field(
        default=0,
        description="Number of bottom patches to highlight. 0 means no highlighting.",
        ge=0,
    )

    default_slide_mpp: SlideMPP | None = Field(
        default=None,
        description="MPP of the slide to use if none can be inferred from the WSI",
    )
