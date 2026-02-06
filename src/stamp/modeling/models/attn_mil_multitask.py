"""Multi-task Attention MIL model.

AttnMILMultiTask: attention-based MIL aggregator with multiple regression heads.
Builds on the existing AttnMIL architecture in stamp.modeling.attn_mil.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnMILMultiTask(nn.Module):
    """Attention MIL with multiple regression heads.

    Architecture:
        phi:  Linear(in_dim → emb_dim) + ReLU
        attn: Linear(emb_dim → emb_dim) + Tanh + Linear(emb_dim → 1) → softmax
        z = weighted sum of tile embeddings
        Separate linear heads for each target.

    Args:
        in_dim: Input feature dimension per tile.
        emb_dim: Embedding dimension after phi.
        head_dims: Dictionary mapping head name to output dimension.
            E.g. {"hrd": 1, "tmb": 1, "clovar": 4}
    """

    def __init__(
        self,
        in_dim: int,
        emb_dim: int = 256,
        head_dims: dict[str, int] | None = None,
    ) -> None:
        super().__init__()

        if head_dims is None:
            head_dims = {"hrd": 1, "tmb": 1, "clovar": 4}

        self.phi = nn.Sequential(nn.Linear(in_dim, emb_dim), nn.ReLU())
        self.attn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Tanh(),
            nn.Linear(emb_dim, 1),
        )

        # Regression heads
        self.heads = nn.ModuleDict(
            {name: nn.Linear(emb_dim, dim) for name, dim in head_dims.items()}
        )
        self.head_names = list(head_dims.keys())

    def forward(
        self, x: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Tensor of shape (B, n_tiles, d) or (n_tiles, d).

        Returns:
            Dictionary with keys for each head name (Tensor values) plus 'attn'.
        """
        if x.ndim == 2:
            x = x.unsqueeze(0)  # (n, d) → (1, n, d)

        h = self.phi(x)  # (B, n, emb)
        a_logits = self.attn(h).squeeze(-1)  # (B, n)
        a = F.softmax(a_logits, dim=1)  # (B, n)
        z = (h * a.unsqueeze(-1)).sum(dim=1)  # (B, emb)

        outputs: dict[str, torch.Tensor] = {}
        for name in self.head_names:
            outputs[name] = self.heads[name](z)
        outputs["attn"] = a

        return outputs
