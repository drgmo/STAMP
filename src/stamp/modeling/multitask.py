"""Multitask training and cross-validation for STAMP.

This module provides:
  - ``LitAttnMILMultiTask``: a Lightning module wrapping an attention MIL backbone
    with per-target regression heads.
  - ``MultitaskDataset``: a patient-level dataset that returns multiple regression
    targets per patient.
  - ``train_multitask_()``: single train/val split training.
  - ``crossval_multitask_()``: K-fold cross-validation with patient-level GroupKFold.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Sequence
from pathlib import Path

import h5py
import lightning
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from sklearn.model_selection import GroupKFold, KFold
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from stamp.modeling.attn_mil import AttnMIL
from stamp.modeling.config import MultitaskTrainingConfig
from stamp.seed import Seed

__author__ = "STAMP contributors"
__license__ = "MIT"

_logger = logging.getLogger("stamp")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class MultitaskDataset(Dataset):
    """Patient-level dataset for multitask regression.

    Each item returns ``(features, targets)`` where *features* has shape
    ``(bag_size, feat_dim)`` and *targets* has shape ``(n_targets,)``.
    """

    def __init__(
        self,
        *,
        feature_files: Sequence[Path],
        targets: Tensor,
        bag_size: int | None,
    ) -> None:
        assert len(feature_files) == len(targets)
        self.feature_files = list(feature_files)
        self.targets = targets  # (N, T)
        self.bag_size = bag_size

    def __len__(self) -> int:
        return len(self.feature_files)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        with h5py.File(self.feature_files[idx], "r") as h5:
            feats = torch.from_numpy(h5["feats"][:]).float()  # type: ignore[index]

        # feats: (n_tiles, d) or (1, d)
        if feats.ndim == 1:
            feats = feats.unsqueeze(0)

        if self.bag_size is not None:
            n = feats.shape[0]
            if n >= self.bag_size:
                sel = torch.randperm(n)[: self.bag_size]
                feats = feats[sel]
            else:
                # pad with zeros
                pad = torch.zeros(self.bag_size - n, feats.shape[1])
                feats = torch.cat([feats, pad], dim=0)

        return feats, self.targets[idx]


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------


class LitAttnMILMultiTask(lightning.LightningModule):
    """Attention-MIL backbone with independent linear heads per target.

    Parameters
    ----------
    target_names : list[str]
        Ordered names of the regression targets.
    target_dims : list[int]
        Output dimension for each head (typically 1 for regression).
    loss_weights : dict[str, float] | None
        Per-target loss weight. Defaults to uniform weight.
    in_dim : int
        Feature dimensionality of the input.
    emb_dim : int
        Embedding dimension of the AttnMIL backbone.
    max_lr, div_factor, total_steps : float / int
        OneCycleLR scheduler parameters.
    loss_type : str
        ``"mse"`` or ``"huber"``.
    huber_delta : float
        Delta for Huber loss.
    """

    def __init__(
        self,
        *,
        target_names: list[str],
        target_dims: list[int],
        loss_weights: dict[str, float] | None = None,
        in_dim: int,
        emb_dim: int = 256,
        max_lr: float = 1e-4,
        div_factor: float = 25.0,
        total_steps: int = 1000,
        loss_type: str = "mse",
        huber_delta: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.target_names = target_names
        self.target_dims = target_dims
        self.loss_type = loss_type
        self.huber_delta = huber_delta
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.div_factor = div_factor

        # Build backbone + heads
        self.backbone = AttnMIL(in_dim=in_dim, emb_dim=emb_dim)
        self.heads = nn.ModuleDict(
            {
                name: nn.Linear(emb_dim, dim)
                for name, dim in zip(target_names, target_dims)
            }
        )

        # Loss weights
        if loss_weights is not None:
            self._loss_weights = {
                k: loss_weights.get(k, 1.0) for k in target_names
            }
        else:
            self._loss_weights = {k: 1.0 for k in target_names}

    # ---- forward -----------------------------------------------------------
    def forward(self, x: Tensor) -> dict[str, Tensor]:
        z, _attn = self.backbone(x)
        return {name: head(z) for name, head in self.heads.items()}

    # ---- loss --------------------------------------------------------------
    def _compute_loss(self, preds: dict[str, Tensor], targets: Tensor) -> tuple[Tensor, dict[str, float]]:
        """Return total weighted loss and a dict of per-head losses."""
        per_head: dict[str, float] = {}
        total = torch.tensor(0.0, device=targets.device)
        for i, name in enumerate(self.target_names):
            y_hat = preds[name].squeeze(-1)
            y = targets[:, i]

            if self.loss_type == "huber":
                head_loss = nn.functional.huber_loss(y_hat, y, delta=self.huber_delta)
            else:
                head_loss = nn.functional.mse_loss(y_hat, y)

            w = self._loss_weights[name]
            total = total + w * head_loss
            per_head[name] = head_loss.item()

        return total, per_head

    # ---- steps -------------------------------------------------------------
    def _step(
        self,
        batch: tuple[Tensor, ...] | list[Tensor],
        step_name: str,
    ) -> Tensor:
        feats, targets = batch[0], batch[1]
        preds = self(feats)
        loss, per_head = self._compute_loss(preds, targets)

        self.log(
            f"{step_name}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        for name, val in per_head.items():
            self.log(
                f"{step_name}_loss_{name}",
                val,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        return loss

    def training_step(self, batch, batch_idx):  # type: ignore[override]
        return self._step(batch, "training")

    def validation_step(self, batch, batch_idx):  # type: ignore[override]
        return self._step(batch, "validation")

    # ---- optimizer ---------------------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            total_steps=self.total_steps,
            max_lr=self.max_lr,
            div_factor=self.div_factor,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_clini_table(config: MultitaskTrainingConfig) -> pd.DataFrame:
    """Read clinical table and index by patient label."""
    df = pd.read_csv(config.clini_table)
    if config.patient_label not in df.columns:
        raise ValueError(
            f"patient_label '{config.patient_label}' not found in {config.clini_table}"
        )
    return df


def _build_feature_index(feature_dir: Path) -> dict[str, Path]:
    """Build a mapping from filename stem to path for all ``.h5`` files.

    Searches *feature_dir* recursively so that feature files stored in
    subdirectories (e.g. ``feature_dir/encoder_name/PAT_001.h5``) are found.
    """
    index: dict[str, Path] = {}
    for fpath in feature_dir.rglob("*.h5"):
        index[fpath.stem] = fpath
    return index


def _filter_valid_patients(
    *,
    patient_ids: Sequence[str],
    clini_df: pd.DataFrame,
    config: MultitaskTrainingConfig,
) -> tuple[list[str], list[Path], list[list[float]]]:
    """Filter patient IDs to those with valid features and target values.

    Returns:
        valid_pids: patient IDs that passed filtering.
        feature_files: corresponding feature file paths.
        target_rows: corresponding target value rows.
    """
    target_names = list(config.target_labels.keys())
    feature_dir = Path(config.feature_dir)
    clini_indexed = clini_df.set_index(config.patient_label)

    # Build an index so we find .h5 files even inside subdirectories
    feat_index = _build_feature_index(feature_dir)

    valid_pids: list[str] = []
    feature_files: list[Path] = []
    target_rows: list[list[float]] = []

    for pid in patient_ids:
        fpath = feat_index.get(pid)
        if fpath is None:
            _logger.warning(f"Feature file not found for patient {pid}, skipping")
            continue
        if pid not in clini_indexed.index:
            continue
        row = clini_indexed.loc[pid]
        try:
            vals = [float(row[t]) for t in target_names]
        except (KeyError, ValueError) as e:
            _logger.warning(f"Skipping patient {pid}: {e}")
            continue
        valid_pids.append(pid)
        feature_files.append(fpath)
        target_rows.append(vals)

    return valid_pids, feature_files, target_rows


def _build_dataset(
    *,
    patient_ids: Sequence[str],
    clini_df: pd.DataFrame,
    config: MultitaskTrainingConfig,
    bag_size: int | None,
) -> MultitaskDataset:
    """Build a MultitaskDataset for the given patient IDs."""
    _, feature_files, target_rows = _filter_valid_patients(
        patient_ids=patient_ids,
        clini_df=clini_df,
        config=config,
    )
    targets = torch.tensor(target_rows, dtype=torch.float32)
    return MultitaskDataset(
        feature_files=feature_files, targets=targets, bag_size=bag_size
    )


def _build_dataloaders(
    *,
    train_pids: Sequence[str],
    val_pids: Sequence[str],
    clini_df: pd.DataFrame,
    config: MultitaskTrainingConfig,
) -> tuple[DataLoader, DataLoader, int]:
    """Create train and validation dataloaders; return (train_dl, val_dl, feat_dim)."""
    train_ds = _build_dataset(
        patient_ids=train_pids,
        clini_df=clini_df,
        config=config,
        bag_size=config.bag_size,
    )
    val_ds = _build_dataset(
        patient_ids=val_pids,
        clini_df=clini_df,
        config=config,
        bag_size=None,
    )

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
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        worker_init_fn=worker_init,
    )

    # Infer feature dimension from the first sample
    sample_feats, _ = train_ds[0]
    feat_dim = sample_feats.shape[-1]

    return train_dl, val_dl, feat_dim


def _create_model(
    config: MultitaskTrainingConfig,
    total_steps: int,
    feat_dim: int,
) -> LitAttnMILMultiTask:
    target_names = list(config.target_labels.keys())
    target_dims = list(config.target_labels.values())
    return LitAttnMILMultiTask(
        target_names=target_names,
        target_dims=target_dims,
        loss_weights=config.loss_weights,
        in_dim=feat_dim,
        emb_dim=config.emb_dim,
        max_lr=config.max_lr,
        div_factor=config.div_factor,
        total_steps=total_steps,
        loss_type=config.loss_type,
        huber_delta=config.huber_delta,
    )


def _train_and_save(
    *,
    model: LitAttnMILMultiTask,
    train_dl: DataLoader,
    val_dl: DataLoader,
    output_dir: Path,
    config: MultitaskTrainingConfig,
) -> Path:
    """Run Trainer.fit, save best checkpoint, return path."""
    torch.set_float32_matmul_precision("high")
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_cb = ModelCheckpoint(
        monitor="validation_loss",
        mode="min",
        filename="checkpoint-{epoch:02d}-{validation_loss:.3f}",
    )
    trainer = lightning.Trainer(
        default_root_dir=output_dir,
        callbacks=[
            EarlyStopping(
                monitor="validation_loss",
                mode="min",
                patience=config.patience,
            ),
            ckpt_cb,
        ],
        max_epochs=config.max_epochs,
        accelerator=config.accelerator,
        devices=1,
        logger=CSVLogger(save_dir=output_dir),
        log_every_n_steps=max(1, len(train_dl)),
        num_sanity_val_steps=0,
    )
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    best = ckpt_cb.best_model_path
    dst = output_dir / "model.ckpt"
    if best:
        shutil.copy(best, dst)
    return dst


def _predict_multitask(
    *,
    model: LitAttnMILMultiTask,
    patient_ids: Sequence[str],
    clini_df: pd.DataFrame,
    config: MultitaskTrainingConfig,
    output_path: Path,
) -> None:
    """Run inference on *patient_ids* and write ``patient-preds.csv``.

    Columns: patient_label, <target>_true, <target>_pred  for each target.
    """
    target_names = list(config.target_labels.keys())

    # Get the valid patient IDs (same filtering as _build_dataset)
    valid_pids, _, _ = _filter_valid_patients(
        patient_ids=patient_ids,
        clini_df=clini_df,
        config=config,
    )

    # Build a dataset with bag_size=None (use all tiles at inference)
    ds = _build_dataset(
        patient_ids=patient_ids,
        clini_df=clini_df,
        config=config,
        bag_size=None,
    )

    worker_init = Seed.get_loader_worker_init() if Seed._is_set() else None
    dl = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        worker_init_fn=worker_init,
    )

    model.eval()
    device = next(model.parameters()).device

    all_preds: list[dict[str, float]] = []
    all_trues: list[dict[str, float]] = []

    with torch.no_grad():
        for feats, targets in dl:
            feats = feats.to(device)
            preds = model(feats)
            pred_row = {
                name: preds[name].squeeze().cpu().item() for name in target_names
            }
            true_row = {
                name: targets[0, i].item() for i, name in enumerate(target_names)
            }
            all_preds.append(pred_row)
            all_trues.append(true_row)

    rows = []
    for pid, true_row, pred_row in zip(valid_pids, all_trues, all_preds):
        row: dict[str, object] = {config.patient_label: pid}
        for name in target_names:
            row[f"{name}_true"] = true_row[name]
            row[f"{name}_pred"] = pred_row[name]
        rows.append(row)

    pd.DataFrame(rows).to_csv(output_path, index=False)
    _logger.info("Saved patient predictions to %s", output_path)


# ---------------------------------------------------------------------------
# Public API: single train/val
# ---------------------------------------------------------------------------


def train_multitask_(config: MultitaskTrainingConfig) -> None:
    """Train a multitask model with a single train/val split."""
    Seed.set(config.seed)
    clini_df = _load_clini_table(config)
    all_pids = clini_df[config.patient_label].unique().tolist()

    # Deterministic shuffle for the split
    rng = np.random.RandomState(config.seed)
    rng.shuffle(all_pids)
    split = int(0.8 * len(all_pids))
    train_pids, val_pids = all_pids[:split], all_pids[split:]

    train_dl, val_dl, feat_dim = _build_dataloaders(
        train_pids=train_pids,
        val_pids=val_pids,
        clini_df=clini_df,
        config=config,
    )
    total_steps = len(train_dl) * config.max_epochs
    model = _create_model(config, total_steps, feat_dim)

    _train_and_save(
        model=model,
        train_dl=train_dl,
        val_dl=val_dl,
        output_dir=Path(config.output_dir),
        config=config,
    )
    _logger.info("Multitask training complete. Output: %s", config.output_dir)


# ---------------------------------------------------------------------------
# Public API: cross-validation
# ---------------------------------------------------------------------------


def crossval_multitask_(config: MultitaskTrainingConfig) -> None:
    """K-fold cross-validation for multitask regression.

    Uses ``GroupKFold`` keyed by patient ID for patient-level splitting.
    If ``crossval.stratify_target`` is set, a stratified approach based on
    quantile-binned values is used (falls back to ``KFold`` with shuffle
    if too few samples for stratification).

    Outputs per fold:
        - ``model.ckpt``
        - ``fold_split.json`` (train/val patient IDs)
        - ``patient-preds.csv`` (per-target ground truth and predictions for val patients)
        - ``metrics.json`` (best validation loss)

    A ``crossval_summary.json`` is written at the end with aggregate
    mean/std of validation losses.
    """
    cv = config.crossval
    if cv is None or not cv.enabled:
        _logger.info("Cross-validation is disabled; running single train/val split.")
        train_multitask_(config)
        return

    clini_df = _load_clini_table(config)
    all_pids = np.array(clini_df[config.patient_label].unique().tolist())
    n_splits = cv.n_splits
    random_state = cv.random_state if cv.random_state is not None else config.seed

    # Build fold indices
    if cv.stratify_target and cv.stratify_target in clini_df.columns:
        # Stratified by quantile-binned target
        clini_indexed = clini_df.set_index(config.patient_label)
        vals = np.array(
            [float(clini_indexed.loc[p][cv.stratify_target]) for p in all_pids]
        )
        bins = min(cv.stratify_bins, len(np.unique(vals)))
        strat_labels = pd.qcut(vals, q=bins, labels=False, duplicates="drop")
        kf = KFold(n_splits=n_splits, shuffle=cv.shuffle, random_state=random_state)
        fold_iter = list(kf.split(all_pids, strat_labels))
    else:
        # Patient-level GroupKFold (each patient is its own group)
        groups = np.arange(len(all_pids))
        gkf = GroupKFold(n_splits=n_splits)
        fold_iter = list(gkf.split(all_pids, groups=groups))

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics: list[dict] = []

    for fold_i, (train_idx, val_idx) in enumerate(fold_iter):
        # Optionally run only one fold
        if cv.fold_index is not None and fold_i != cv.fold_index:
            continue

        fold_dir = output_dir / f"fold_{fold_i}"

        if (fold_dir / "patient-preds.csv").exists():
            _logger.info(f"Fold {fold_i}: patient-preds.csv exists, skipping")
            # Collect metrics if present
            mf = fold_dir / "metrics.json"
            if mf.exists():
                fold_metrics.append(json.loads(mf.read_text()))
            continue

        fold_seed = config.seed + fold_i
        Seed.set(fold_seed)

        train_pids = all_pids[train_idx].tolist()
        val_pids = all_pids[val_idx].tolist()

        # Save split information
        fold_dir.mkdir(parents=True, exist_ok=True)
        (fold_dir / "fold_split.json").write_text(
            json.dumps(
                {"fold": fold_i, "train_patients": train_pids, "val_patients": val_pids},
                indent=2,
            )
        )

        _logger.info(
            f"Fold {fold_i}/{n_splits}: train={len(train_pids)}, val={len(val_pids)}"
        )

        # Train the model (skip if checkpoint already exists)
        if not (fold_dir / "model.ckpt").exists():
            train_dl, val_dl, feat_dim = _build_dataloaders(
                train_pids=train_pids,
                val_pids=val_pids,
                clini_df=clini_df,
                config=config,
            )
            total_steps = len(train_dl) * config.max_epochs
            model = _create_model(config, total_steps, feat_dim)

            _train_and_save(
                model=model,
                train_dl=train_dl,
                val_dl=val_dl,
                output_dir=fold_dir,
                config=config,
            )

        # Generate patient-preds.csv for the validation set
        best_ckpt = fold_dir / "model.ckpt"
        if best_ckpt.exists():
            best_model = LitAttnMILMultiTask.load_from_checkpoint(str(best_ckpt))
            _predict_multitask(
                model=best_model,
                patient_ids=val_pids,
                clini_df=clini_df,
                config=config,
                output_path=fold_dir / "patient-preds.csv",
            )

        # Extract best val_loss from trainer logs
        best_val = float("inf")
        csv_dir = fold_dir / "lightning_logs"
        for metrics_csv in csv_dir.rglob("metrics.csv"):
            mdf = pd.read_csv(metrics_csv)
            if "validation_loss" in mdf.columns:
                min_val = mdf["validation_loss"].dropna().min()
                if min_val < best_val:
                    best_val = float(min_val)

        fold_result = {"fold": fold_i, "best_val_loss": best_val}
        (fold_dir / "metrics.json").write_text(json.dumps(fold_result, indent=2))
        fold_metrics.append(fold_result)

    # Write summary
    if fold_metrics:
        losses = [m["best_val_loss"] for m in fold_metrics if np.isfinite(m["best_val_loss"])]
        summary = {
            "n_splits": n_splits,
            "seed": config.seed,
            "folds": fold_metrics,
            "mean_val_loss": float(np.mean(losses)) if losses else None,
            "std_val_loss": float(np.std(losses)) if losses else None,
        }
        (output_dir / "crossval_summary.json").write_text(
            json.dumps(summary, indent=2)
        )
        _logger.info(
            "Cross-validation complete. mean_val_loss=%.4f ± %.4f",
            summary["mean_val_loss"] or float("nan"),
            summary["std_val_loss"] or float("nan"),
        )
