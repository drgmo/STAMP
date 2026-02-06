"""Training entry point for multi-task Attention MIL.

This module provides the training pipeline for the multi-task AttnMIL model,
independent of the standard STAMP training pipeline.
"""

import logging
import shutil
from pathlib import Path

import lightning
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from stamp.modeling.multitask.config import MultitaskTrainConfig
from stamp.modeling.multitask.dataset import MultitaskBagDataset
from stamp.modeling.multitask.lightning_module import (
    LitAttnMILMultiTask,
    ZScoreNormalizer,
)
from stamp.seed import Seed

_logger = logging.getLogger("stamp")


def train_multitask_model_(config: MultitaskTrainConfig) -> None:
    """Train the multi-task AttnMIL model.

    1. Load clini + slide tables
    2. Build MultitaskBagDataset for train/val
    3. Fit z-score normalizer on training targets
    4. Create LitAttnMILMultiTask Lightning module
    5. Train with Lightning Trainer
    """
    if config.seed is not None:
        Seed.set(config.seed)

    torch.set_float32_matmul_precision("high")

    # Load tables
    clini_df = pd.read_csv(config.clini_table, dtype=str)
    clini_df = clini_df.set_index(config.patient_label)

    slide_df = pd.read_csv(config.slide_table, dtype=str)
    slide_df = slide_df[[config.patient_label, config.filename_label]]

    # Build ordered target labels list
    target_labels = list(config.target_labels.keys())

    # Convert target columns to float
    for col in target_labels:
        if col not in clini_df.columns:
            raise ValueError(
                f"Target column '{col}' not found in clini table. "
                f"Available columns: {list(clini_df.columns)}"
            )
        clini_df[col] = pd.to_numeric(clini_df[col], errors="coerce")

    # Build full dataset to identify valid patients
    full_ds = MultitaskBagDataset(
        slide_table=slide_df,
        clini_df=clini_df,
        feature_dir=str(config.feature_dir),
        target_labels=target_labels,
        bag_size=config.bag_size,
        h5_feature_key=config.h5_feature_key,
    )

    if len(full_ds) == 0:
        raise ValueError(
            "No valid patients found. Check that clini_table has matching patients "
            "in slide_table with existing H5 files and non-NaN target values."
        )

    # Train/val split (80/20)
    all_patients = full_ds.patients
    train_patients, val_patients = train_test_split(
        all_patients, test_size=0.2, random_state=config.seed or 0
    )

    _logger.info(
        f"Train/val split: {len(train_patients)} train, {len(val_patients)} val"
    )

    # Build train/val datasets
    train_slide_df = slide_df[
        slide_df[config.patient_label].isin(train_patients)
    ]
    val_slide_df = slide_df[
        slide_df[config.patient_label].isin(val_patients)
    ]

    train_ds = MultitaskBagDataset(
        slide_table=train_slide_df,
        clini_df=clini_df,
        feature_dir=str(config.feature_dir),
        target_labels=target_labels,
        bag_size=config.bag_size,
        h5_feature_key=config.h5_feature_key,
    )
    val_ds = MultitaskBagDataset(
        slide_table=val_slide_df,
        clini_df=clini_df,
        feature_dir=str(config.feature_dir),
        target_labels=target_labels,
        bag_size=config.bag_size,
        h5_feature_key=config.h5_feature_key,
    )

    # Fit z-score normalizer on training targets
    train_targets = torch.stack(
        [torch.from_numpy(train_ds.targets[pid]) for pid in train_ds.patients]
    )
    normalizer = ZScoreNormalizer().fit(train_targets)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    normalizer.save(config.output_dir / "zscore_stats.json")
    _logger.info(
        f"Z-score stats saved: mean={normalizer.mean}, std={normalizer.std}"
    )

    # DataLoaders
    worker_init = Seed.get_loader_worker_init() if Seed._is_set() else None
    train_dl = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        worker_init_fn=worker_init,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        worker_init_fn=worker_init,
    )

    # Infer input dimension from first sample
    sample_X, _ = train_ds[0]
    in_dim = sample_X.shape[-1]
    _logger.info(f"Detected feature dimension: {in_dim}")

    # Build head_dims from config
    head_dims: dict[str, int] = {}
    offset = 0
    for name, dim in config.target_labels.items():
        head_dims[name] = dim
        offset += dim

    # Build loss weights
    loss_weights = {name: config.loss_weights.get(name, 1.0) for name in head_dims}

    # Total steps for scheduler
    steps_per_epoch = len(train_dl)
    total_steps = steps_per_epoch * config.max_epochs

    # Create model
    model = LitAttnMILMultiTask(
        in_dim=in_dim,
        emb_dim=config.emb_dim,
        target_labels=target_labels,
        head_dims=head_dims,
        loss_weights=loss_weights,
        loss_type=config.loss_type,
        huber_delta=config.huber_delta,
        total_steps=total_steps,
        max_lr=config.max_lr,
        div_factor=config.div_factor,
        normalizer=normalizer,
    )

    _logger.info(
        f"AttnMILMultiTask model: in_dim={in_dim}, emb_dim={config.emb_dim}, "
        f"heads={head_dims}, loss_weights={loss_weights}, "
        f"loss_type={config.loss_type}, huber_delta={config.huber_delta}"
    )

    # Trainer
    model_checkpoint = ModelCheckpoint(
        monitor="validation_loss",
        mode="min",
        filename="checkpoint-{epoch:02d}-{validation_loss:0.4f}",
    )
    trainer = lightning.Trainer(
        default_root_dir=config.output_dir,
        callbacks=[
            EarlyStopping(
                monitor="validation_loss", mode="min", patience=config.patience
            ),
            model_checkpoint,
        ],
        max_epochs=config.max_epochs,
        accelerator=config.accelerator,
        devices=1,
        logger=CSVLogger(save_dir=config.output_dir),
        log_every_n_steps=steps_per_epoch,
        num_sanity_val_steps=0,
    )

    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    # Copy best checkpoint
    if model_checkpoint.best_model_path:
        shutil.copy(model_checkpoint.best_model_path, config.output_dir / "model.ckpt")
        _logger.info(
            f"Best model saved to {config.output_dir / 'model.ckpt'} "
            f"(val_loss={model_checkpoint.best_model_score:.4f})"
        )
