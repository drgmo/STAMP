"""Configuration for multi-task Attention MIL training."""

from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict, Field


class MultitaskHeadConfig(BaseModel):
    """Configuration for a single regression head."""

    model_config = ConfigDict(extra="forbid")
    dim: int = Field(1, description="Output dimension for this head")
    weight: float = Field(1.0, description="Loss weight for this head")


class MultitaskTrainConfig(BaseModel):
    """Configuration for multi-task AttnMIL training."""

    model_config = ConfigDict(extra="forbid")

    output_dir: Path = Field(description="Directory to save results to")
    clini_table: Path = Field(description="CSV with clinical data (indexed by PATIENT)")
    slide_table: Path = Field(description="CSV with PATIENT and FILENAME columns")
    feature_dir: Path = Field(description="Directory containing .h5 feature files")

    # Column names
    patient_label: str = "PATIENT"
    filename_label: str = "FILENAME"

    # Target column names in clini_table
    target_labels: dict[str, int] = Field(
        default={
            "scarHRD": 1,
            "TMB": 1,
            "CLOVAR_D": 1,
            "CLOVAR_I": 1,
            "CLOVAR_M": 1,
            "CLOVAR_P": 1,
        },
        description="Mapping of target column name -> output dimension",
    )

    # Loss weights per target head (matched by name)
    loss_weights: dict[str, float] = Field(
        default={
            "scarHRD": 1.0,
            "TMB": 0.1,
            "CLOVAR_D": 0.3,
            "CLOVAR_I": 0.3,
            "CLOVAR_M": 0.3,
            "CLOVAR_P": 0.3,
        },
        description="Loss weight per target head",
    )

    # H5 feature key (auto-detected if None)
    h5_feature_key: str | None = Field(
        default=None,
        description="Key in H5 file for features. If None, auto-detect from 'feats' or 'patch_embeddings'.",
    )

    # Model hyperparameters
    emb_dim: int = Field(256, description="Embedding dimension for attention MIL")
    bag_size: int = Field(512, description="Number of tiles per bag")
    batch_size: int = Field(16, description="Batch size for training")
    num_workers: int = Field(4, description="DataLoader workers")
    max_epochs: int = Field(64, description="Maximum training epochs")
    patience: int = Field(16, description="Early stopping patience")
    max_lr: float = Field(1e-4, description="Max learning rate (OneCycleLR)")
    div_factor: float = Field(25.0, description="LR scheduler div factor")

    # Loss type
    loss_type: str = Field("huber", description="Loss type: 'huber' or 'mse'")
    huber_delta: float = Field(1.0, description="Huber loss delta parameter")

    # Hardware
    accelerator: str = "gpu" if torch.cuda.is_available() else "cpu"

    # Reproducibility
    seed: int | None = Field(42, description="Random seed")
