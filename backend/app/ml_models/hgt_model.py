"""简化版 HGT 模型占位实现。

为了保证整体流程可运行，这里实现一个按节点类型分别映射的 MLP，
接口与真实 HGT 模型保持一致：
- HGTModelConfig: 聚合元信息与维度；
- HGTModel.forward(x_dict, edge_index_dict): 返回每类节点的嵌入字典。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn


@dataclass(slots=True)
class HGTModelConfig:
    metadata: Tuple[tuple[list[str], list[tuple[str, str, str]]], ...] | object
    input_dims: dict[str, int]
    hidden_dim: int = 128
    out_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4  # 未使用，仅做兼容
    dropout: float = 0.1


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout)]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


class HGTModel(nn.Module):
    def __init__(self, cfg: HGTModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.mlps = nn.ModuleDict({
            nt: _MLP(in_dim, cfg.hidden_dim, cfg.out_dim, cfg.dropout)
            for nt, in_dim in cfg.input_dims.items()
        })

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict | None = None) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        out: Dict[str, torch.Tensor] = {}
        for nt, x in x_dict.items():
            mlp = self.mlps.get(nt)
            if mlp is None:
                continue
            out[nt] = mlp(x)
        return out


__all__ = ["HGTModel", "HGTModelConfig"]
