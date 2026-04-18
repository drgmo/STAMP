"""Read-side helpers for prompt-derived tile masks produced by WSIVL.

WSIVL writes a single HDF5 file, ``prompt_masks.h5``, with a top-level
attribute ``mask_mode`` that selects the layout:

* ``per_fold``: one group per CV fold, ``/fold_<i>/{slide_name}``.
* ``shared``:   a single ``/shared/{slide_name}`` group, used when the
  prompts were chosen without any dataset access (e.g. label-free
  text-judge mode); the same mask is reused for every fold.

STAMP consumes the file fold-by-fold via :func:`group_for_fold`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import h5py

_logger = logging.getLogger("stamp")


class MaskMode(str, Enum):
    PER_FOLD = "per_fold"
    SHARED = "shared"


@dataclass(frozen=True)
class PromptMaskHandle:
    path: Path
    mode: MaskMode
    n_folds: int | None
    splits_sha256: str | None
    judge_mode: str | None
    encoder_name: str | None


def open_prompt_masks(path: Path) -> PromptMaskHandle:
    """Inspect ``prompt_masks.h5`` and return a lightweight handle.

    The file is closed before returning; workers reopen it lazily.
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        raw_mode = f.attrs.get("mask_mode", "shared")
        if isinstance(raw_mode, bytes):
            raw_mode = raw_mode.decode("utf-8")
        try:
            mode = MaskMode(raw_mode)
        except ValueError as e:
            raise RuntimeError(
                f"{path}: unknown mask_mode attr {raw_mode!r}"
            ) from e

        n_folds: int | None = None
        if mode is MaskMode.PER_FOLD:
            if "n_folds" not in f.attrs:
                raise RuntimeError(
                    f"{path}: per_fold mask without n_folds attr"
                )
            n_folds = int(f.attrs["n_folds"])

        def _opt(key: str) -> str | None:
            v = f.attrs.get(key)
            if v is None:
                return None
            return v.decode("utf-8") if isinstance(v, bytes) else str(v)

        return PromptMaskHandle(
            path=path,
            mode=mode,
            n_folds=n_folds,
            splits_sha256=_opt("splits_sha256"),
            judge_mode=_opt("judge_mode"),
            encoder_name=_opt("encoder_name"),
        )


def group_for_fold(handle: PromptMaskHandle, fold_index: int) -> str:
    """Return the HDF5 group name for ``fold_index`` given the layout."""
    if handle.mode is MaskMode.SHARED:
        return "shared"
    if handle.n_folds is None:
        raise RuntimeError(
            f"{handle.path}: per_fold mode without n_folds"
        )
    if fold_index < 0 or fold_index >= handle.n_folds:
        raise RuntimeError(
            f"fold_index {fold_index} out of range for n_folds={handle.n_folds}"
        )
    return f"fold_{fold_index}"
