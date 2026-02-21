"""Attention score extraction for trained models.

Extracts per-tile attention scores from a trained model (particularly
AttentionMIL, but also supports attention rollout for VisionTransformer)
and saves them as CSV files — one per slide.

Each CSV contains:
    - tile_index: Sequential index of the tile in the feature file
    - coord_x_um: X coordinate of the tile in micrometers
    - coord_y_um: Y coordinate of the tile in micrometers
    - attention_score: Attention weight assigned to this tile
    - attention_rank: Rank of the tile by attention score (1 = highest)

Additionally, a summary CSV is created listing the top-k tiles per slide
across all processed slides.
"""

import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch import Tensor

from stamp.modeling.data import get_coords
from stamp.modeling.deploy import load_model_from_ckpt
from stamp.types import DeviceLikeType

_logger = logging.getLogger("stamp")


def extract_attention_scores_(
    *,
    feature_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    device: DeviceLikeType = "cpu",
    topk: int = 0,
) -> None:
    """Extract attention scores for all slides in feature_dir.

    For AttentionMIL: directly uses the gated attention weights.
    For VisionTransformer: uses attention rollout across transformer layers.

    Args:
        feature_dir: Directory containing .h5 feature files.
        checkpoint_path: Path to the trained model checkpoint.
        output_dir: Directory to save attention CSVs.
        device: Device for inference.
        topk: If > 0, include a summary of top-k tiles per slide.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model_from_ckpt(checkpoint_path).eval().to(device)
    model_name = getattr(model.hparams, "model_name", "unknown")

    _logger.info(f"Loaded model: {model_name}")
    _logger.info(f"Extracting attention scores from {feature_dir}")

    h5_files = sorted(feature_dir.glob("*.h5"))
    if not h5_files:
        _logger.warning(f"No .h5 files found in {feature_dir}")
        return

    summary_rows: list[dict] = []

    for h5_path in h5_files:
        slide_name = h5_path.stem
        _logger.info(f"Processing {slide_name}...")

        try:
            with h5py.File(h5_path, "r") as h5:
                feat_type = h5.attrs.get("feat_type", None)
                if feat_type is not None and feat_type != "tile":
                    _logger.warning(
                        f"Skipping {slide_name}: not tile-level features (type={feat_type})"
                    )
                    continue

                if "feats" in h5:
                    feats = torch.from_numpy(h5["feats"][:]).float().to(device)
                else:
                    feats = torch.from_numpy(h5["patch_embeddings"][:]).float().to(device)

                coords_info = get_coords(h5)
                coords_um = torch.from_numpy(coords_info.coords_um).float()

        except Exception as e:
            _logger.warning(f"Skipping {slide_name}: {e}")
            continue

        # Extract attention scores based on model type
        attention = _extract_attention(
            model=model,
            feats=feats,
            coords_um=coords_um.to(device),
            model_name=model_name,
            device=device,
        )

        if attention is None:
            _logger.warning(
                f"Could not extract attention for {slide_name} "
                f"(model type '{model_name}' may not support attention extraction)"
            )
            continue

        # Build per-tile DataFrame
        n_tiles = len(attention)
        ranks = torch.argsort(torch.argsort(attention, descending=True)) + 1

        df = pd.DataFrame({
            "tile_index": np.arange(n_tiles),
            "coord_x_um": coords_um[:n_tiles, 0].numpy(),
            "coord_y_um": coords_um[:n_tiles, 1].numpy(),
            "attention_score": attention.cpu().numpy(),
            "attention_rank": ranks.cpu().numpy(),
        })
        df = df.sort_values("attention_rank")

        # Save per-slide CSV
        csv_path = output_dir / f"{slide_name}_attention.csv"
        df.to_csv(csv_path, index=False)
        _logger.info(
            f"  Saved {n_tiles} tiles to {csv_path} "
            f"(top attention: {attention.max():.4f}, "
            f"min: {attention.min():.4f})"
        )

        # Collect summary rows
        if topk > 0:
            top_df = df.head(topk).copy()
            top_df.insert(0, "slide", slide_name)
            summary_rows.extend(top_df.to_dict("records"))

    # Save summary
    if topk > 0 and summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = output_dir / "attention_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        _logger.info(f"Saved top-{topk} summary to {summary_path}")

    _logger.info(f"Attention extraction complete. Results in {output_dir}")


def _extract_attention(
    *,
    model: torch.nn.Module,
    feats: Tensor,
    coords_um: Tensor,
    model_name: str,
    device: DeviceLikeType,
) -> Tensor | None:
    """Extract attention scores from a model.

    Supports:
        - attmil: Direct attention weights from gated attention mechanism
        - vit: Attention rollout across transformer layers
        - trans_mil: Nystrom attention approximation

    Returns:
        1D tensor of attention scores per tile, or None if unsupported.
    """
    with torch.no_grad():
        if model_name == "attmil":
            return _extract_attmil_attention(model, feats, coords_um)
        elif model_name == "vit":
            return _extract_vit_attention(model, feats, coords_um, device)
        elif model_name == "trans_mil":
            return _extract_transmil_attention(model, feats)
        else:
            _logger.warning(
                f"Attention extraction not implemented for model type '{model_name}'. "
                "Supported: attmil, vit, trans_mil."
            )
            return None


def _extract_attmil_attention(
    model: torch.nn.Module,
    feats: Tensor,
    coords_um: Tensor,
) -> Tensor:
    """Extract attention from AttentionMIL model."""
    backbone = model.model  # Access the backbone from the Lightning wrapper

    # Forward pass to populate attention weights
    _ = backbone(feats.unsqueeze(0), coords=coords_um.unsqueeze(0), mask=None)

    attention = backbone.get_last_attention()
    if attention is None:
        raise RuntimeError("AttentionMIL did not store attention weights")

    return attention.squeeze(0).cpu()  # (T,)


def _extract_vit_attention(
    model: torch.nn.Module,
    feats: Tensor,
    coords_um: Tensor,
    device: DeviceLikeType,
) -> Tensor:
    """Extract attention from VisionTransformer via attention rollout.

    Hooks into nn.MultiheadAttention layers to capture attention weights,
    then performs attention rollout to get CLS → tile attention.
    """
    backbone = model.model
    attn_weights_list: list[Tensor] = []

    # Register hooks to capture attention weights
    hooks = []
    for layer in backbone.transformer.layers:
        sa_module = layer[0]  # SelfAttention
        mhsa = sa_module.mhsa

        if isinstance(mhsa, torch.nn.MultiheadAttention):
            def _make_hook():
                def hook_fn(module, input, output):
                    # output is (attn_output, attn_weights)
                    # We need to call with need_weights=True
                    pass
                return hook_fn

            # Re-do forward with need_weights=True by monkey-patching temporarily
            pass

    # Alternative: compute attention rollout via direct forward
    # For VisionTransformer, we use a gradient-free approach
    bags = feats.unsqueeze(0)
    coords = coords_um.unsqueeze(0)
    batch_size = 1
    n_tiles = feats.shape[0]

    # Project features
    projected = backbone.project_features(bags)

    # Prepend CLS token
    from einops import repeat
    cls_tokens = repeat(backbone.class_token, "d -> b 1 d", b=batch_size)
    tokens = torch.cat([cls_tokens, projected], dim=1)
    coords_with_cls = torch.cat(
        [torch.zeros(batch_size, 1, 2).type_as(coords), coords], dim=1
    )

    # Manually run through transformer layers, capturing attention
    x = tokens
    attn_rollout = None

    for layer_modules in backbone.transformer.layers:
        attn_module = layer_modules[0]  # SelfAttention
        ff_module = layer_modules[1]

        # Norm
        x_normed = attn_module.norm(x)

        mhsa = attn_module.mhsa
        if isinstance(mhsa, torch.nn.MultiheadAttention):
            # Call with need_weights=True
            attn_output, attn_w = mhsa(
                x_normed, x_normed, x_normed,
                need_weights=True,
                average_attn_weights=True,
            )
            # attn_w: (B, seq, seq) or (B*heads, seq, seq)
            if attn_w.dim() == 3 and attn_w.shape[0] == batch_size:
                layer_attn = attn_w[0]  # (seq, seq)
            else:
                # Average over heads
                layer_attn = attn_w.view(
                    mhsa.num_heads, batch_size, -1, attn_w.shape[-1]
                ).mean(0)[0]
        else:
            # ALiBi or other — fall back to uniform
            seq_len = x_normed.shape[1]
            layer_attn = torch.ones(seq_len, seq_len, device=device) / seq_len
            attn_output = mhsa(
                q=x_normed, k=x_normed, v=x_normed,
                coords_q=coords_with_cls, coords_k=coords_with_cls,
                attn_mask=None, alibi_mask=None,
            )

        # Residual
        x = attn_output + x
        x = ff_module(x) + x

        # Normalize rows
        layer_attn = layer_attn / (layer_attn.sum(dim=-1, keepdim=True) + 1e-8)

        # Accumulate rollout
        if attn_rollout is None:
            attn_rollout = layer_attn
        else:
            attn_rollout = attn_rollout @ layer_attn

    if attn_rollout is None:
        raise RuntimeError("No attention layers found in transformer")

    # CLS token → tile attention (skip CLS→CLS)
    cls_attn = attn_rollout[0, 1:]  # (T,)

    # Normalize to [0, 1]
    cls_attn = cls_attn - cls_attn.min()
    cls_attn = cls_attn / (cls_attn.max().clamp(min=1e-8))

    return cls_attn.cpu()


def _extract_transmil_attention(
    model: torch.nn.Module,
    feats: Tensor,
) -> Tensor:
    """Extract attention from TransMIL model.

    Uses the Nystrom attention approximation to compute approximate
    CLS → tile attention weights.
    """
    backbone = model.model
    h = backbone._fc1(feats.unsqueeze(0))

    # Pad to square
    H = h.shape[1]
    _H = _W = int(np.ceil(np.sqrt(H)))
    add_length = _H * _W - H
    h = torch.cat([h, h[:, :add_length, :]], dim=1)

    # Add CLS token
    B = h.shape[0]
    cls_tokens = backbone.cls_token.expand(B, -1, -1).to(h.device)
    h = torch.cat((cls_tokens, h), dim=1)

    # First transformer layer with attention extraction
    h_normed = backbone.layer1.norm(h)
    out, attn = backbone.layer1.attn(h_normed, return_attn=True)

    # attn shape: (B, heads, seq, seq) or (B, seq, seq)
    if attn.dim() == 4:
        attn = attn.mean(1)  # Average over heads

    # CLS → tile attention (skip CLS and padding)
    cls_attn = attn[0, 0, 1 : H + 1]  # (T,)

    # Normalize
    cls_attn = cls_attn - cls_attn.min()
    cls_attn = cls_attn / (cls_attn.max().clamp(min=1e-8))

    return cls_attn.cpu()
