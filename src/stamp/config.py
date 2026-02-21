from pydantic import BaseModel, ConfigDict

from stamp.encoding.config import PatientEncodingConfig, SlideEncodingConfig
from stamp.heatmaps.config import HeatmapConfig
from stamp.modeling.config import (
    AdvancedConfig,
    AttentionExtractionConfig,
    CrossvalConfig,
    DeploymentConfig,
    MultitaskCrossvalConfig,
    MultitaskTrainConfig,
    TrainConfig,
)
from stamp.preprocessing.config import PreprocessingConfig
from stamp.statistics import StatsConfig


class StampConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preprocessing: PreprocessingConfig | None = None

    training: TrainConfig | None = None
    crossval: CrossvalConfig | None = None
    deployment: DeploymentConfig | None = None

    multitask_training: MultitaskTrainConfig | None = None
    multitask_crossval: MultitaskCrossvalConfig | None = None

    attention_extraction: AttentionExtractionConfig | None = None

    statistics: StatsConfig | None = None

    heatmaps: HeatmapConfig | None = None

    slide_encoding: SlideEncodingConfig | None = None

    patient_encoding: PatientEncodingConfig | None = None

    advanced_config: AdvancedConfig | None = None
