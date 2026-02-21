"""Multitask training and cross-validation.

Supports training a single model with multiple classification heads on
different targets simultaneously. The backbone is shared, and each target
gets its own classification head. Missing labels per target are handled
via NaN masking in the loss.
"""

import logging
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import lightning
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.accelerators.accelerator import Accelerator
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from pydantic import BaseModel
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

from stamp.modeling.config import (
    AdvancedConfig,
    MultitaskCrossvalConfig,
    MultitaskTarget,
    MultitaskTrainConfig,
)
from stamp.modeling.data import (
    BagDataset,
    _collate_to_tuple,
    detect_feature_type,
    read_table,
    slide_to_patient_from_slide_table_,
)
from stamp.modeling.models.multitask import LitMultitaskTileClassifier
from stamp.modeling.registry import ModelName, load_model_class
from stamp.modeling.transforms import VaryPrecisionTransform
from stamp.seed import Seed
from stamp.types import (
    Category,
    FeaturePath,
    GroundTruth,
    PandasLabel,
    PatientId,
)

_logger = logging.getLogger("stamp")


# ---------- Data structures ----------


class MultitaskPatientData:
    """Patient data with multiple ground truths (one per target)."""

    def __init__(
        self,
        *,
        ground_truths: dict[PandasLabel, GroundTruth | None],
        feature_files: list[FeaturePath],
    ) -> None:
        self.ground_truths = ground_truths
        self.feature_files = feature_files


# ---------- Data loading ----------


def _load_multitask_ground_truths(
    *,
    clini_table_path: Path,
    patient_label: PandasLabel,
    targets: list[MultitaskTarget],
) -> dict[PatientId, dict[PandasLabel, GroundTruth | None]]:
    """Load ground truths for multiple targets from the clinical table.

    Returns a dict mapping patient_id -> {label: ground_truth_value}.
    Missing values are stored as None.
    """
    columns = [patient_label] + [t.ground_truth_label for t in targets]
    clini_df = read_table(clini_table_path, usecols=columns, dtype=str)

    result: dict[PatientId, dict[PandasLabel, GroundTruth | None]] = {}
    for _, row in clini_df.iterrows():
        pid = row[patient_label]
        if pd.isna(pid):
            continue
        gts: dict[PandasLabel, GroundTruth | None] = {}
        for target in targets:
            val = row.get(target.ground_truth_label)
            gts[target.ground_truth_label] = (
                None if pd.isna(val) else str(val)
            )
        result[str(pid)] = gts

    return result


def _build_multitask_patient_data(
    *,
    patient_to_ground_truths: dict[PatientId, dict[PandasLabel, GroundTruth | None]],
    slide_to_patient: dict[FeaturePath, PatientId],
) -> dict[PatientId, MultitaskPatientData]:
    """Build MultitaskPatientData from ground truths and slide mapping.

    Patients must have at least one feature file and at least one non-None
    ground truth to be included.
    """
    patient_to_slides: dict[PatientId, set[FeaturePath]] = {}
    for slide, patient in slide_to_patient.items():
        patient_to_slides.setdefault(patient, set()).add(slide)

    result: dict[PatientId, MultitaskPatientData] = {}
    for pid, gts in patient_to_ground_truths.items():
        slides = patient_to_slides.get(pid)
        if slides is None:
            continue
        existing = [s for s in slides if s.exists()]
        if not existing:
            continue
        # Include patient if at least one target has a value
        if all(v is None for v in gts.values()):
            continue
        result[pid] = MultitaskPatientData(
            ground_truths=gts,
            feature_files=existing,
        )

    _logger.info(
        f"Multitask: {len(result)} usable patients "
        f"(from {len(patient_to_ground_truths)} in clinical table)"
    )
    return result


def _create_multitask_dataloader(
    *,
    patient_data: list[MultitaskPatientData],
    targets: list[MultitaskTarget],
    resolved_categories: dict[str, list[Category]],
    bag_size: int | None,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    transform=None,
) -> DataLoader:
    """Create a dataloader for multitask tile-level training.

    Encodes ground truths as concatenated one-hot vectors across all targets.
    Missing targets are encoded as NaN.
    """
    all_ground_truths = []
    for pd_item in patient_data:
        row_parts = []
        for target in targets:
            label = target.ground_truth_label
            cats = resolved_categories[label]
            gt_val = pd_item.ground_truths.get(label)
            if gt_val is None or gt_val not in cats:
                # Missing target → NaN vector
                row_parts.append(torch.full((len(cats),), float("nan")))
            else:
                # One-hot encode
                one_hot = torch.zeros(len(cats))
                one_hot[cats.index(gt_val)] = 1.0
                row_parts.append(one_hot)
        all_ground_truths.append(torch.cat(row_parts))

    ground_truths_tensor = torch.stack(all_ground_truths)

    ds = BagDataset(
        bags=[pd_item.feature_files for pd_item in patient_data],
        bag_size=bag_size,
        ground_truths=ground_truths_tensor,
        transform=transform,
    )

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate_to_tuple,
        worker_init_fn=Seed.get_loader_worker_init() if Seed._is_set() else None,
    )
    return dl


# ---------- Training ----------


def _resolve_categories(
    targets: list[MultitaskTarget],
    patient_data: dict[PatientId, MultitaskPatientData],
) -> dict[str, list[Category]]:
    """Resolve categories for each target (use provided or infer from data)."""
    resolved: dict[str, list[Category]] = {}
    for target in targets:
        label = target.ground_truth_label
        if target.categories:
            resolved[label] = list(target.categories)
        else:
            # Infer from data
            values = sorted(
                {
                    pd_item.ground_truths[label]
                    for pd_item in patient_data.values()
                    if pd_item.ground_truths.get(label) is not None
                }
            )
            resolved[label] = values
        _logger.info(f"Target '{label}': categories = {resolved[label]}")
    return resolved


def _compute_category_weights(
    patient_data: list[MultitaskPatientData],
    label: PandasLabel,
    categories: list[Category],
) -> torch.Tensor:
    """Compute inverse-frequency class weights for a single target."""
    counts = np.zeros(len(categories))
    for pd_item in patient_data:
        gt = pd_item.ground_truths.get(label)
        if gt is not None and gt in categories:
            counts[categories.index(gt)] += 1

    # Avoid division by zero
    counts = np.maximum(counts, 1)
    reciprocal = counts.sum() / counts
    weights = reciprocal / reciprocal.sum()
    return torch.tensor(weights, dtype=torch.float32)


def train_multitask_model_(
    *,
    config: MultitaskTrainConfig,
    advanced: AdvancedConfig,
) -> None:
    """Train a multitask model."""
    feature_type = detect_feature_type(config.feature_dir)
    _logger.info(f"Detected feature type: {feature_type}")

    if feature_type != "tile":
        raise ValueError(
            "Multitask training currently only supports tile-level features."
        )
    if config.slide_table is None:
        raise ValueError("A slide table is required for multitask training.")

    # Load data
    patient_to_gts = _load_multitask_ground_truths(
        clini_table_path=config.clini_table,
        patient_label=config.patient_label,
        targets=config.targets,
    )

    slide_to_patient = slide_to_patient_from_slide_table_(
        slide_table_path=config.slide_table,
        feature_dir=config.feature_dir,
        patient_label=config.patient_label,
        filename_label=config.filename_label,
    )

    patient_data = _build_multitask_patient_data(
        patient_to_ground_truths=patient_to_gts,
        slide_to_patient=slide_to_patient,
    )

    resolved_cats = _resolve_categories(config.targets, patient_data)

    # Stratified split on first target
    first_label = config.targets[0].ground_truth_label
    stratify = [
        pd_item.ground_truths.get(first_label, "MISSING")
        for pd_item in patient_data.values()
    ]
    patient_ids = list(patient_data.keys())

    train_pids, valid_pids = cast(
        tuple[list[PatientId], list[PatientId]],
        train_test_split(patient_ids, stratify=stratify, shuffle=True, random_state=0),
    )

    train_data = [patient_data[pid] for pid in train_pids]
    valid_data = [patient_data[pid] for pid in valid_pids]

    model, train_dl, valid_dl = _setup_multitask_model(
        train_data=train_data,
        valid_data=valid_data,
        targets=config.targets,
        resolved_cats=resolved_cats,
        advanced=advanced,
        train_pids=train_pids,
        valid_pids=valid_pids,
        train_transform=(
            VaryPrecisionTransform(min_fraction_bits=1)
            if config.use_vary_precision_transform
            else None
        ),
    )

    _train_multitask_(
        output_dir=config.output_dir,
        model=model,
        train_dl=train_dl,
        valid_dl=valid_dl,
        max_epochs=advanced.max_epochs,
        patience=advanced.patience,
        accelerator=advanced.accelerator,
    )


def _setup_multitask_model(
    *,
    train_data: list[MultitaskPatientData],
    valid_data: list[MultitaskPatientData],
    targets: list[MultitaskTarget],
    resolved_cats: dict[str, list[Category]],
    advanced: AdvancedConfig,
    train_pids: list[PatientId],
    valid_pids: list[PatientId],
    train_transform=None,
) -> tuple[lightning.LightningModule, DataLoader, DataLoader]:
    """Set up the multitask model and dataloaders."""

    train_dl = _create_multitask_dataloader(
        patient_data=train_data,
        targets=targets,
        resolved_categories=resolved_cats,
        bag_size=advanced.bag_size,
        batch_size=advanced.batch_size,
        shuffle=True,
        num_workers=advanced.num_workers,
        transform=train_transform,
    )

    valid_dl = _create_multitask_dataloader(
        patient_data=valid_data,
        targets=targets,
        resolved_categories=resolved_cats,
        bag_size=None,
        batch_size=1,
        shuffle=False,
        num_workers=advanced.num_workers,
        transform=None,
    )

    # Infer feature dim
    batch = next(iter(train_dl))
    bags, _, _, _ = batch
    dim_feats = bags.shape[-1]

    # Model name
    if advanced.model_name is None:
        advanced.model_name = ModelName.VIT
        _logger.info(
            f"No model specified, defaulting to '{advanced.model_name.value}' for multitask"
        )

    # Load model class (use tile classifier registry for backbone)
    _, ModelClass = load_model_class("classification", "tile", advanced.model_name)

    model_specific_params = (
        advanced.model_params.model_dump().get(advanced.model_name.value) or {}
    )

    # Build target configs for Lightning module
    target_configs = []
    for target in targets:
        label = target.ground_truth_label
        cats = resolved_cats[label]
        weights = _compute_category_weights(train_data, label, cats)
        target_configs.append({
            "ground_truth_label": label,
            "categories": cats,
            "category_weights": weights,
            "weight": target.weight,
        })

    steps_per_epoch = len(train_dl)
    total_steps = steps_per_epoch * advanced.max_epochs

    model = LitMultitaskTileClassifier(
        model_class=ModelClass,
        target_configs=target_configs,
        dim_input=dim_feats,
        total_steps=total_steps,
        max_lr=advanced.max_lr,
        div_factor=advanced.div_factor,
        train_patients=train_pids,
        valid_patients=valid_pids,
        model_name=advanced.model_name.value,
        **model_specific_params,
    )

    return model, train_dl, valid_dl


def _train_multitask_(
    *,
    output_dir: Path,
    model: lightning.LightningModule,
    train_dl: DataLoader,
    valid_dl: DataLoader,
    max_epochs: int,
    patience: int,
    accelerator: str | Accelerator,
) -> lightning.LightningModule:
    """Train the multitask model."""
    torch.set_float32_matmul_precision("high")

    monitor_metric, mode = "validation_loss", "min"

    model_checkpoint = ModelCheckpoint(
        monitor=monitor_metric,
        mode=mode,
        filename=f"checkpoint-{{epoch:02d}}-{{{monitor_metric}:0.3f}}",
    )

    trainer = lightning.Trainer(
        default_root_dir=output_dir,
        callbacks=[
            EarlyStopping(monitor=monitor_metric, mode=mode, patience=patience),
            model_checkpoint,
        ],
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        logger=CSVLogger(save_dir=output_dir),
        log_every_n_steps=len(train_dl),
        num_sanity_val_steps=0,
    )

    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=valid_dl)
    shutil.copy(model_checkpoint.best_model_path, output_dir / "model.ckpt")

    ModelClass = type(model)
    return ModelClass.load_from_checkpoint(model_checkpoint.best_model_path)


# ---------- Cross-validation ----------


class _Split(BaseModel):
    train_patients: set[PatientId]
    test_patients: set[PatientId]


class _Splits(BaseModel):
    splits: Sequence[_Split]


def crossval_multitask_(
    config: MultitaskCrossvalConfig,
    advanced: AdvancedConfig,
) -> None:
    """Run multitask cross-validation."""
    feature_type = detect_feature_type(config.feature_dir)
    _logger.info(f"Detected feature type: {feature_type}")

    if feature_type != "tile":
        raise ValueError(
            "Multitask cross-validation currently only supports tile-level features."
        )
    if config.slide_table is None:
        raise ValueError("A slide table is required for multitask cross-validation.")

    # Load data
    patient_to_gts = _load_multitask_ground_truths(
        clini_table_path=config.clini_table,
        patient_label=config.patient_label,
        targets=config.targets,
    )

    slide_to_patient = slide_to_patient_from_slide_table_(
        slide_table_path=config.slide_table,
        feature_dir=config.feature_dir,
        patient_label=config.patient_label,
        filename_label=config.filename_label,
    )

    patient_data = _build_multitask_patient_data(
        patient_to_ground_truths=patient_to_gts,
        slide_to_patient=slide_to_patient,
    )

    resolved_cats = _resolve_categories(config.targets, patient_data)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    splits_file = config.output_dir / "splits.json"

    # Generate or load splits
    if not splits_file.exists():
        splits = _get_multitask_splits(
            patient_data=patient_data,
            targets=config.targets,
            n_splits=config.n_splits,
        )
        with open(splits_file, "w") as fp:
            fp.write(splits.model_dump_json(indent=4))
    else:
        _logger.debug(f"reading splits from {splits_file}")
        with open(splits_file, "r") as fp:
            splits = _Splits.model_validate_json(fp.read())

    # Train and evaluate each split
    for split_i, split in enumerate(splits.splits):
        split_dir = config.output_dir / f"split-{split_i}"

        if (split_dir / "patient-preds.csv").exists():
            _logger.info(
                f"Skipping split {split_i}, predictions already exist."
            )
            continue

        train_pids = [pid for pid in split.train_patients if pid in patient_data]
        test_pids = [pid for pid in split.test_patients if pid in patient_data]

        train_data = [patient_data[pid] for pid in train_pids]
        test_data = [patient_data[pid] for pid in test_pids]

        # Train
        if not (split_dir / "model.ckpt").exists():
            model, train_dl, valid_dl = _setup_multitask_model(
                train_data=train_data,
                valid_data=test_data,  # Use test as validation in crossval
                targets=config.targets,
                resolved_cats=resolved_cats,
                advanced=advanced,
                train_pids=train_pids,
                valid_pids=test_pids,
                train_transform=(
                    VaryPrecisionTransform(min_fraction_bits=1)
                    if config.use_vary_precision_transform
                    else None
                ),
            )
            model = _train_multitask_(
                output_dir=split_dir,
                model=model,
                train_dl=train_dl,
                valid_dl=valid_dl,
                max_epochs=advanced.max_epochs,
                patience=advanced.patience,
                accelerator=advanced.accelerator,
            )
        else:
            ckpt = torch.load(split_dir / "model.ckpt", map_location="cpu", weights_only=False)
            # Reload model from checkpoint
            model = LitMultitaskTileClassifier.load_from_checkpoint(
                split_dir / "model.ckpt",
                model_class=_get_model_class(advanced),
            )

        # Predict on test set
        test_dl = _create_multitask_dataloader(
            patient_data=test_data,
            targets=config.targets,
            resolved_categories=resolved_cats,
            bag_size=None,
            batch_size=1,
            shuffle=False,
            num_workers=advanced.num_workers,
            transform=None,
        )

        model = model.eval()
        torch.set_float32_matmul_precision("medium")
        trainer = lightning.Trainer(
            accelerator=advanced.accelerator,
            devices=1,
            logger=False,
        )
        raw_preds = torch.concat(
            cast(list[torch.Tensor], trainer.predict(model, test_dl))
        )

        # Build prediction DataFrame
        _save_multitask_predictions(
            raw_preds=raw_preds,
            test_pids=test_pids,
            test_data=test_data,
            targets=config.targets,
            resolved_cats=resolved_cats,
            patient_label=config.patient_label,
            output_path=split_dir / "patient-preds.csv",
        )


def _get_model_class(advanced: AdvancedConfig):
    """Get the backbone model class from advanced config."""
    model_name = advanced.model_name or ModelName.VIT
    _, ModelClass = load_model_class("classification", "tile", model_name)
    return ModelClass


def _get_multitask_splits(
    *,
    patient_data: dict[PatientId, MultitaskPatientData],
    targets: list[MultitaskTarget],
    n_splits: int,
) -> _Splits:
    """Generate stratified k-fold splits for multitask.

    Stratifies on the first target's ground truth.
    """
    patients = np.array(list(patient_data.keys()))
    first_label = targets[0].ground_truth_label

    # Stratify on first target
    stratify_labels = np.array([
        patient_data[pid].ground_truths.get(first_label, "MISSING")
        for pid in patients
    ])

    try:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        iterator = skf.split(patients, stratify_labels)
    except ValueError:
        _logger.warning(
            "StratifiedKFold failed (too few samples per class?), "
            "falling back to regular KFold."
        )
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        iterator = kf.split(patients)

    splits = _Splits(
        splits=[
            _Split(
                train_patients=set(patients[train_idx]),
                test_patients=set(patients[test_idx]),
            )
            for train_idx, test_idx in iterator
        ]
    )
    return splits


def _save_multitask_predictions(
    *,
    raw_preds: torch.Tensor,
    test_pids: list[PatientId],
    test_data: list[MultitaskPatientData],
    targets: list[MultitaskTarget],
    resolved_cats: dict[str, list[Category]],
    patient_label: PandasLabel,
    output_path: Path,
) -> None:
    """Save multitask predictions to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, (pid, pd_item) in enumerate(zip(test_pids, test_data)):
        row: dict[str, Any] = {patient_label: pid}

        offset = 0
        for target in targets:
            label = target.ground_truth_label
            cats = resolved_cats[label]
            n_cats = len(cats)

            target_logits = raw_preds[i, offset : offset + n_cats]
            target_probs = torch.softmax(target_logits, dim=0)
            offset += n_cats

            # Ground truth
            gt = pd_item.ground_truths.get(label)
            row[f"{label}_true"] = gt

            # Predicted class
            row[f"{label}_pred"] = cats[int(target_probs.argmax())]

            # Per-class probabilities
            for j, cat in enumerate(cats):
                row[f"{label}_{cat}"] = target_probs[j].item()

        rows.append(row)

    pd.DataFrame(rows).to_csv(output_path, index=False)
    _logger.info(f"Saved multitask predictions to {output_path}")
