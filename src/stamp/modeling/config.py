import os
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, Field

from stamp.modeling.registry import ModelName
from stamp.types import Category, PandasLabel, Task


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task: Task | None = Field(default="classification")

    output_dir: Path = Field(description="The directory to save the results to")

    clini_table: Path = Field(description="Excel or CSV to read clinical data from")
    slide_table: Path | None = Field(
        default=None, description="Excel or CSV to read patient-slide associations from"
    )
    feature_dir: Path = Field(description="Directory containing feature files")

    ground_truth_label: PandasLabel | Sequence[PandasLabel] | None = Field(
        default=None,
        description="Name of categorical column in clinical table to train on",
    )
    categories: Sequence[Category] | None = None

    status_label: PandasLabel | None = Field(
        default=None,
        description="Column in the clinical table indicating patient status (e.g. alive, dead, censored).",
    )

    time_label: PandasLabel | None = Field(
        default=None,
        description="Column in the clinical table indicating follow-up or survival time (e.g. days).",
    )

    patient_label: PandasLabel = "PATIENT"
    filename_label: PandasLabel = "FILENAME"

    params_path: Path | None = Field(
        default=None,
        description="Optional: Path to a YAML file with advanced training parameters.",
    )

    # Experimental features
    use_vary_precision_transform: bool = False

    # Prompt-derived tile masks (authored by WSIVL).
    prompt_masks_path: Path | None = Field(
        default=None,
        description=(
            "HDF5 with prompt-derived tile masks. Layout: /fold_<i>/{slide} "
            "or /shared/{slide}, detected via attrs.mask_mode. "
            "See stamp.modeling.prompt_masks."
        ),
    )
    prompt_mask_application: Literal[
        "feature_weight", "feature_gate", "none"
    ] = Field(
        default="feature_weight",
        description=(
            "How the mask modulates tile features before MIL. "
            "'feature_weight' multiplies features by the mask (broadcast "
            "over the feature axis). 'feature_gate' keeps only tiles whose "
            "mask value is >= prompt_mask_threshold. 'none' ignores the mask."
        ),
    )
    prompt_mask_threshold: float = Field(default=0.0, ge=0.0, le=1.0)


class CrossvalConfig(TrainConfig):
    n_splits: int = Field(5, ge=2)
    task: Task | None = Field(default="classification")

    splits_path: Path | None = Field(
        default=None,
        description=(
            "Optional external splits.json (typically authored by WSIVL). "
            "If set, STAMP loads it instead of regenerating, and verifies "
            "the locally computed splits match patient-set-wise."
        ),
    )
    require_prompt_masks: bool = Field(
        default=False,
        description="If true, abort when prompt_masks_path is not set.",
    )


class DeploymentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: Path

    checkpoint_paths: list[Path]
    clini_table: Path | None = None
    slide_table: Path
    feature_dir: Path

    ground_truth_label: PandasLabel | Sequence[PandasLabel] | None = None
    patient_label: PandasLabel = "PATIENT"
    filename_label: PandasLabel = "FILENAME"

    # For survival prediction
    status_label: PandasLabel | None = None
    time_label: PandasLabel | None = None

    num_workers: int = min(os.cpu_count() or 1, 16)
    accelerator: str = "gpu" if torch.cuda.is_available() else "cpu"


class VitModelParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dim_model: int = 512
    dim_feedforward: int = 512
    n_heads: int = 8
    n_layers: int = 2
    dropout: float = 0.0
    # Experimental feature: Use ALiBi positional embedding
    use_alibi: bool = False


class MlpModelParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dim_hidden: int = 512
    num_layers: int = 2
    dropout: float = 0.25


class TransMILModelParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dim_hidden: int = 512


class BarspoonParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    d_model: int = 512
    num_encoder_heads: int = 8
    num_decoder_heads: int = 8
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    dim_feedforward: int = 2048
    positional_encoding: bool = True
    # Other hparams
    learning_rate: float = 1e-4


class LinearModelParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    num_encoder_heads: int = 8
    num_decoder_heads: int = 8
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    dim_feedforward: int = 2048
    positional_encoding: bool = True
    # Other hparams
    learning_rate: float = 1e-4


class ModelParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    vit: VitModelParams = Field(default_factory=VitModelParams)
    trans_mil: TransMILModelParams = Field(default_factory=TransMILModelParams)
    mlp: MlpModelParams = Field(default_factory=MlpModelParams)
    linear: LinearModelParams = Field(default_factory=LinearModelParams)
    barspoon: BarspoonParams = Field(default_factory=BarspoonParams)


class AdvancedConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bag_size: int = 512
    num_workers: int = min(os.cpu_count() or 1, 16)
    batch_size: int = 64
    max_epochs: int = 32
    patience: int = 16
    accelerator: str = "gpu" if torch.cuda.is_available() else "cpu"
    max_lr: float = 1e-4
    div_factor: float = 25.0
    model_name: ModelName | None = Field(
        default=None,
        description='Optional. "vit" or "mlp" are defaults based on feature type.',
    )
    model_params: ModelParams
    seed: int | None = None
    checkpoint_metric: str | None = Field(
        default=None,
        description=(
            "Metric to monitor for checkpoint selection and early stopping. "
            "If not set, defaults to 'validation_loss' for classification/regression "
            "and 'val_cindex' for survival. "
            "Available metrics per task — "
            "classification: 'validation_loss', 'validation_auroc'; "
            "regression: 'validation_loss', 'validation_mae'; "
            "survival: 'val_cox_loss', 'val_cindex'."
        ),
    )
