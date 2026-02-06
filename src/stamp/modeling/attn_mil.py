# stamp/models/attn_mil.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnMIL(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int = 256) -> None:
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(in_dim, emb_dim), nn.ReLU())
        self.attn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Tanh(),
            nn.Linear(emb_dim, 1),
        )

        # ein gemeinsamer „backbone“-Head; einzelne Tasks handled STAMP über Heads
        self.out_dim = emb_dim  # wichtig: für STAMP-Head-Definition

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, n, d)
        h = self.phi(x)  # (B, n, emb)
        a = self.attn(h).squeeze(-1)  # (B, n)
        a = F.softmax(a, dim=1)
        z = (h * a.unsqueeze(-1)).sum(dim=1)  # (B, emb)
        return z, a  # z geht in STAMP-Heads, a kannst du loggen
