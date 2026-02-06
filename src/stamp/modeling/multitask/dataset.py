"""Multi-task bag dataset for H5 tile features.

Groups slides per patient, concatenates features, samples bags, and returns
multi-target regression vectors.
"""

import logging

import h5py
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

_logger = logging.getLogger("stamp")


def _autodetect_h5_key(h5: h5py.File) -> str:
    """Auto-detect the feature key in an H5 file."""
    for key in ("feats", "patch_embeddings", "features"):
        if key in h5:
            return key
    available = list(h5.keys())
    raise KeyError(
        f"Could not auto-detect feature key in H5 file. "
        f"Available keys: {available}. "
        f"Set h5_feature_key in config to specify explicitly."
    )


class MultitaskBagDataset(Dataset):
    """Dataset that groups slides per patient, loads H5 tile features,
    samples fixed-size bags, and returns multi-target vectors.

    Args:
        slide_table: DataFrame with at least patient_label and filename_label columns.
        clini_df: DataFrame with clinical targets, indexed by patient_label.
        feature_dir: Path to directory containing .h5 feature files.
        target_labels: Ordered list of column names in clini_df to use as targets.
        bag_size: Number of tiles to sample per bag.
        h5_feature_key: Key in H5 file for features. If None, auto-detect.
    """

    def __init__(
        self,
        slide_table: pd.DataFrame,
        clini_df: pd.DataFrame,
        feature_dir: str,
        target_labels: list[str],
        bag_size: int = 512,
        h5_feature_key: str | None = None,
    ) -> None:
        self.feature_dir = feature_dir
        self.bag_size = bag_size
        self.h5_feature_key = h5_feature_key
        self.target_labels = target_labels

        # Build patient → list of filenames mapping
        self.patients: list[str] = []
        self.patient_files: dict[str, list[str]] = {}
        self.targets: dict[str, np.ndarray] = {}

        for pid, group in slide_table.groupby(slide_table.columns[0]):
            pid = str(pid)
            if pid not in clini_df.index:
                continue
            filenames = group.iloc[:, 1].tolist()
            row = clini_df.loc[pid]

            # Check all target columns exist and are numeric
            try:
                target_values = np.array(
                    [float(row[col]) for col in target_labels], dtype=np.float32
                )
            except (KeyError, ValueError, TypeError) as e:
                _logger.debug(f"Skipping patient {pid}: {e}")
                continue

            if np.any(np.isnan(target_values)):
                _logger.debug(f"Skipping patient {pid}: NaN in targets")
                continue

            self.patients.append(pid)
            self.patient_files[pid] = filenames
            self.targets[pid] = target_values

        _logger.info(
            f"MultitaskBagDataset: {len(self.patients)} patients, "
            f"{len(target_labels)} targets, bag_size={bag_size}"
        )

    def _load_features(self, filename: str) -> np.ndarray:
        """Load features from an H5 file."""
        path = f"{self.feature_dir}/{filename}"
        with h5py.File(path, "r") as f:
            if self.h5_feature_key is not None:
                key = self.h5_feature_key
            else:
                key = _autodetect_h5_key(f)
            arr = f[key][:]  # type: ignore[index]
        return np.asarray(arr)

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        pid = self.patients[idx]
        filenames = self.patient_files[pid]

        # Concatenate features from all slides for this patient
        feature_arrays = [self._load_features(fn) for fn in filenames]
        X = np.concatenate(feature_arrays, axis=0)  # (N_total, d)

        # Bag sampling
        n_tiles = X.shape[0]
        if n_tiles >= self.bag_size:
            sel = np.random.choice(n_tiles, self.bag_size, replace=False)
        else:
            sel = np.random.choice(n_tiles, self.bag_size, replace=True)
        X_bag = torch.from_numpy(X[sel]).float()  # (bag_size, d)

        target_vec = torch.from_numpy(self.targets[pid]).float()

        return X_bag, target_vec
