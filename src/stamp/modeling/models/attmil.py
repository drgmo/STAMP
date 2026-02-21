"""
Attention-based Multiple Instance Learning (ATTMIL)

Gated Attention MIL based on:
> Ilse, M., Tomczak, J. M., & Welling, M. (2018).
> Attention-based Deep Multiple Instance Learning.
> Proceedings of the 35th International Conference on Machine Learning (ICML).

This model uses a gated attention mechanism to aggregate tile-level features
into a single slide-level representation, which is then classified.
The attention weights are interpretable and indicate which tiles the model
considers most relevant for the prediction.
"""

import torch
from beartype import beartype
from jaxtyping import Bool, Float, jaxtyped
from torch import Tensor, nn


class AttentionMIL(nn.Module):
    """Gated Attention-based MIL pooling.

    Architecture:
        1. Project tile features to hidden dimension
        2. Compute gated attention scores per tile
        3. Weighted aggregation of tile features
        4. Classify the aggregated representation

    The gated attention mechanism uses element-wise multiplication
    of tanh and sigmoid branches, allowing the model to learn
    which features are relevant (tanh) and how much to attend (sigmoid).

    Args:
        dim_input: Dimensionality of input tile features.
        dim_hidden: Hidden dimension for attention and projection.
        dim_output: Number of output classes / regression targets.
        dropout: Dropout rate applied after the feature projection.
    """

    def __init__(
        self,
        *,
        dim_input: int,
        dim_hidden: int = 256,
        dim_output: int = 1,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()

        # Feature projection
        self.feature_projection = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Gated attention branches
        self.attention_V = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.Tanh(),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.Sigmoid(),
        )
        self.attention_w = nn.Linear(dim_hidden, 1)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(dim_hidden, dim_output),
        )

        # Store last attention weights for extraction
        self._last_attention: Tensor | None = None

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        bags: Float[Tensor, "batch tiles features"],
        *,
        coords: Float[Tensor, "batch tiles 2"] | None = None,
        mask: Bool[Tensor, "batch tiles"] | None = None,
    ) -> Float[Tensor, "batch dim_output"]:
        """Forward pass.

        Args:
            bags: Tile feature bags, shape (B, T, F).
            coords: Tile coordinates (unused, kept for API compatibility).
            mask: Boolean mask where True indicates padding tiles to ignore.

        Returns:
            Logits of shape (B, dim_output).
        """
        # Project features: (B, T, F) -> (B, T, H)
        h = self.feature_projection(bags)

        # Gated attention: (B, T, H) -> (B, T, 1) -> (B, T)
        a_v = self.attention_V(h)
        a_u = self.attention_U(h)
        attention_logits = self.attention_w(a_v * a_u).squeeze(-1)  # (B, T)

        # Mask padded tiles
        if mask is not None:
            attention_logits = attention_logits.masked_fill(mask, float("-inf"))

        # Softmax over tiles
        attention_weights = torch.softmax(attention_logits, dim=1)  # (B, T)

        # Store for extraction (detached, on CPU to save memory)
        self._last_attention = attention_weights.detach()

        # Weighted aggregation: (B, 1, T) @ (B, T, H) -> (B, 1, H) -> (B, H)
        slide_repr = torch.bmm(
            attention_weights.unsqueeze(1), h
        ).squeeze(1)

        # Classification
        logits = self.classifier(slide_repr)  # (B, dim_output)
        return logits

    def get_last_attention(self) -> Tensor | None:
        """Return the attention weights from the last forward pass.

        Returns:
            Attention weights of shape (B, T), or None if no forward pass
            has been performed yet.
        """
        return self._last_attention
