"""HeatmapTopK encoder: select top-K tiles by semantic heatmap score, mean-pool.

Like EAGLE but uses pre-computed heatmap scores instead of learned attention.
No GPU or neural network needed — purely score-based selection + aggregation.

Heatmap H5 files (from WSIVL direction-benchmark) have structure::

    /x              int32 (N,)
    /y              int32 (N,)
    /pos            float32 (N,)
    /neg            float32 (N,)
    /scores/{label} float32 (N,)
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from stamp.encoding.config import EncoderName
from stamp.encoding.encoder import Encoder
from stamp.modeling.heatmap_scores import load_heatmap_scores
from stamp.types import DeviceLikeType, PandasLabel

_logger = logging.getLogger("stamp")


class HeatmapTopK(Encoder):
    """Select top-K tiles by semantic heatmap score, mean-pool to slide embedding."""

    def __init__(self, k: int = 25, score_key: str = "pos") -> None:
        super().__init__(
            model=nn.Identity(),
            identifier=EncoderName.HEATMAP_TOPK,
            precision=torch.float32,
            required_extractors=[],
        )
        self.k = k
        self.score_key = score_key

    def _generate_slide_embedding(
        self,
        feats: torch.Tensor,
        device: DeviceLikeType,
        **kwargs,
    ) -> np.ndarray:
        scores = kwargs["scores"]
        scores_t = (
            torch.from_numpy(scores) if isinstance(scores, np.ndarray) else scores
        )
        k = min(self.k, len(scores_t))
        topk_indices = torch.topk(scores_t.float(), k).indices
        topk_feats = feats[topk_indices]
        return topk_feats.mean(dim=0).to(torch.float32).detach().cpu().numpy()

    def _generate_patient_embedding(
        self,
        feats_list: list[torch.Tensor],
        device: DeviceLikeType,
        **kwargs,
    ) -> np.ndarray:
        scores_list = kwargs.get("scores_list", [])
        all_feats = torch.cat(feats_list, dim=0)
        all_scores = np.concatenate(scores_list)
        return self._generate_slide_embedding(all_feats, device, scores=all_scores)

    def encode_slides_(
        self,
        output_dir: Path,
        feat_dir: Path,
        device: DeviceLikeType,
        generate_hash: bool,
        **kwargs,
    ) -> None:
        heatmap_dir: Path | None = kwargs.get("heatmap_dir")
        if not heatmap_dir:
            raise ValueError(
                "heatmap_dir is required for HeatmapTopK encoder. "
                "Set it in slide_encoding config."
            )
        heatmap_dir = Path(heatmap_dir)

        encode_dir = output_dir / f"{self.identifier}-slide"
        os.makedirs(encode_dir, exist_ok=True)

        h5_files = sorted(f for f in os.listdir(feat_dir) if f.endswith(".h5"))
        for tile_feats_filename in (progress := tqdm(h5_files)):
            slide_name = Path(tile_feats_filename).stem
            progress.set_description(slide_name)

            output_path = encode_dir / tile_feats_filename
            if output_path.exists():
                _logger.debug(
                    "skipping %s because %s already exists", slide_name, output_path
                )
                continue

            h5_path = os.path.join(feat_dir, tile_feats_filename)
            try:
                feats, coords_info, _extractor = self._read_h5(h5_path)
            except (ValueError, FileNotFoundError) as e:
                tqdm.write(str(e))
                continue

            try:
                scores = load_heatmap_scores(
                    slide_name=slide_name,
                    stamp_coords_um=coords_info.coords_um,
                    tile_size_um=float(coords_info.tile_size_um),
                    tile_size_px=int(coords_info.tile_size_px) if coords_info.tile_size_px else None,
                    heatmap_dir=heatmap_dir,
                    score_key=self.score_key,
                    feature_h5_path=Path(h5_path),
                    normalize=False,  # top-K ranking works on raw scores
                )
            except (FileNotFoundError, KeyError) as e:
                tqdm.write(f"[{slide_name}] skipping — {e}")
                continue

            slide_embedding = self._generate_slide_embedding(
                feats, device, scores=scores
            )
            self._save_features_(
                output_path=output_path, feats=slide_embedding, feat_type="slide"
            )

    def encode_patients_(
        self,
        output_dir: Path,
        feat_dir: Path,
        slide_table_path: Path,
        patient_label: PandasLabel,
        filename_label: PandasLabel,
        device: DeviceLikeType,
        generate_hash: bool,
        **kwargs,
    ) -> None:
        heatmap_dir: Path | None = kwargs.get("heatmap_dir")
        if not heatmap_dir:
            raise ValueError(
                "heatmap_dir is required for HeatmapTopK encoder. "
                "Set it in patient_encoding config."
            )
        heatmap_dir = Path(heatmap_dir)

        encode_dir = output_dir / f"{self.identifier}-pat"
        os.makedirs(encode_dir, exist_ok=True)

        slide_table = pd.read_csv(slide_table_path)
        patient_groups = slide_table.groupby(patient_label)

        for patient_id, group in (progress := tqdm(patient_groups)):
            progress.set_description(str(patient_id))

            output_path = (encode_dir / str(patient_id)).with_suffix(".h5")
            if output_path.exists():
                _logger.debug(
                    "skipping %s because %s already exists", patient_id, output_path
                )
                continue

            feats_list: list[torch.Tensor] = []
            scores_list: list[np.ndarray] = []

            for _, row in group.iterrows():
                slide_filename = row[filename_label]
                slide_name = Path(slide_filename).stem
                h5_path = os.path.join(feat_dir, slide_filename)

                try:
                    feats, coords_info, _extractor = self._read_h5(h5_path)
                except (FileNotFoundError, ValueError) as e:
                    tqdm.write(
                        f"[{patient_id}] skip slide {slide_name}: {e}"
                    )
                    continue

                try:
                    scores = load_heatmap_scores(
                        slide_name=slide_name,
                        stamp_coords_um=coords_info.coords_um,
                        tile_size_um=float(coords_info.tile_size_um),
                        tile_size_px=int(coords_info.tile_size_px) if coords_info.tile_size_px else None,
                        heatmap_dir=heatmap_dir,
                        score_key=self.score_key,
                        feature_h5_path=Path(h5_path),
                        normalize=False,
                    )
                except (FileNotFoundError, KeyError) as e:
                    tqdm.write(
                        f"[{patient_id}] skip slide {slide_name} — {e}"
                    )
                    continue

                feats_list.append(feats)
                scores_list.append(scores)

            if not feats_list:
                tqdm.write(f"No features for patient {patient_id}")
                continue

            patient_embedding = self._generate_patient_embedding(
                feats_list, device, scores_list=scores_list
            )
            self._save_features_(
                output_path=output_path, feats=patient_embedding, feat_type="patient"
            )
