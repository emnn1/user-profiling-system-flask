"""简化版训练器。"""
from __future__ import annotations

from typing import Any

import torch
from torch import optim

from ..ml_models.hgt_model import HGTModel
from ..ml_models.feature_store import HeteroFeatureEncoder


class HGTTrainer:
    def __init__(self, *, model: HGTModel, feature_encoder: HeteroFeatureEncoder, device: torch.device) -> None:
        self.model = model
        self.encoder = feature_encoder
        self.device = device

    def train_on_graph(self, hetero_graph, epochs: int = 2) -> list[float]:
        self.model.train()
        x = self.encoder.build_features_for_graph(hetero_graph)
        opt = optim.Adam(self.model.parameters(), lr=1e-3)
        losses: list[float] = []
        for _ in range(epochs):
            opt.zero_grad(set_to_none=True)
            out = self.model(x, getattr(hetero_graph, "edge_index_dict", {}))
            loss = sum((v.pow(2).mean() for v in out.values()), start=torch.tensor(0.0, device=self.device))
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        return losses


__all__ = ["HGTTrainer"]
