import os
from collections.abc import Sequence
from pathlib import Path

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

    ground_truth_label: PandasLabel | None = Field(
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


class CrossvalConfig(TrainConfig):
    n_splits: int = Field(5, ge=2)
    task: Task | None = Field(default="classification")


class MultitaskTarget(BaseModel):
    """Configuration for a single target in multitask training."""

    model_config = ConfigDict(extra="forbid")

    ground_truth_label: PandasLabel = Field(
        description="Name of the column in the clinical table for this target."
    )
    task: Task = Field(
        default="classification",
        description="Task type for this target.",
    )
    categories: Sequence[Category] | None = Field(
        default=None,
        description="Categories for classification. Inferred if not set.",
    )
    weight: float = Field(
        default=1.0,
        description="Loss weight for this target (relative to other targets).",
    )


class MultitaskTrainConfig(BaseModel):
    """Configuration for multitask training with multiple targets."""

    model_config = ConfigDict(extra="forbid")

    output_dir: Path = Field(description="The directory to save the results to")

    clini_table: Path = Field(description="Excel or CSV to read clinical data from")
    slide_table: Path | None = Field(
        default=None, description="Excel or CSV to read patient-slide associations from"
    )
    feature_dir: Path = Field(description="Directory containing feature files")

    targets: list[MultitaskTarget] = Field(
        description="List of targets to train on simultaneously."
    )

    patient_label: PandasLabel = "PATIENT"
    filename_label: PandasLabel = "FILENAME"

    params_path: Path | None = Field(
        default=None,
        description="Optional: Path to a YAML file with advanced training parameters.",
    )

    use_vary_precision_transform: bool = False


class MultitaskCrossvalConfig(MultitaskTrainConfig):
    """Configuration for multitask cross-validation."""

    n_splits: int = Field(5, ge=2)


class DeploymentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: Path

    checkpoint_paths: list[Path]
    clini_table: Path | None = None
    slide_table: Path
    feature_dir: Path

    ground_truth_label: PandasLabel | None = None
    patient_label: PandasLabel = "PATIENT"
    filename_label: PandasLabel = "FILENAME"

    # For survival prediction
    status_label: PandasLabel | None = None
    time_label: PandasLabel | None = None

    num_workers: int = min(os.cpu_count() or 1, 16)
    accelerator: str = "gpu" if torch.cuda.is_available() else "cpu"


class AttentionExtractionConfig(BaseModel):
    """Configuration for extracting attention scores from a trained model."""

    model_config = ConfigDict(extra="forbid")

    output_dir: Path = Field(
        description="Directory to save attention score CSVs"
    )
    feature_dir: Path = Field(description="Directory containing feature files")
    checkpoint_path: Path = Field(description="Path to model checkpoint file")

    slide_table: Path | None = Field(
        default=None,
        description="Slide table for mapping slides to patients",
    )
    patient_label: PandasLabel = "PATIENT"
    filename_label: PandasLabel = "FILENAME"

    device: str = Field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu",
        description="Device to use for computation",
    )

    topk: int = Field(
        default=0,
        ge=0,
        description="If > 0, only include the top-k tiles by attention score per slide.",
    )


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


class LinearModelParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AttMILModelParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dim_hidden: int = 256
    dropout: float = 0.25


class ModelParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    vit: VitModelParams = Field(default_factory=VitModelParams)
    trans_mil: TransMILModelParams = Field(default_factory=TransMILModelParams)
    mlp: MlpModelParams = Field(default_factory=MlpModelParams)
    linear: LinearModelParams = Field(default_factory=LinearModelParams)
    attmil: AttMILModelParams = Field(default_factory=AttMILModelParams)


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
