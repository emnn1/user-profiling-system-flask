"""HGT 模型实现及训练/推理工具。

模块对外提供：

- :class:`HGTModel`：核心的异构图 Transformer 网络；
- 训练/推理便捷函数 (:func:`train_epoch`, :func:`inference` 等)，
    供 :class:`~app.services.incremental_learner.IncrementalLearner` 调用；
- 邻居采样工具 :func:`create_neighbor_loader`，与增量训练流程相衔接。

依赖关系：需要 :mod:`torch_geometric` 支持，在 ``main.py`` 中统一检测 GPU 与依赖。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

try:  # pragma: no cover - 运行环境缺少依赖时提示
    from torch_geometric.data import HeteroData  # type: ignore[import]
    from torch_geometric.loader import NeighborLoader  # type: ignore[import]
    from torch_geometric.nn import HGTConv  # type: ignore[import]
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "HGT 模块依赖 torch-geometric，请先安装后再使用。"
    ) from exc


@dataclass(slots=True)
class HGTModelConfig:
    """HGT 模型配置。"""

    metadata: tuple[list[str], list[tuple[str, str, str]]]
    input_dims: Dict[str, int]
    hidden_dim: int = 128
    out_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1


class HGTModel(nn.Module):
    """多层 HGT 模型，输出每种节点的嵌入向量。"""

    def __init__(self, config: HGTModelConfig) -> None:
        """根据配置构建 HGT 层叠与输入投影。"""
        super().__init__()
        self.config = config
        self.metadata = config.metadata
        self.node_types = config.metadata[0]

        self.input_proj = nn.ModuleDict(
            {
                node_type: nn.Linear(config.input_dims[node_type], config.hidden_dim)
                for node_type in self.node_types
            }
        )

        self.layers = nn.ModuleList()
        in_channels: Dict[str, int] = {nt: config.hidden_dim for nt in self.node_types}
        for layer_idx in range(config.num_layers):
            out_dim = config.hidden_dim if layer_idx < config.num_layers - 1 else config.out_dim
            self.layers.append(
                HGTConv(
                    in_channels=in_channels,
                    out_channels=out_dim,
                    metadata=config.metadata,
                    heads=config.num_heads,
                    dropout=config.dropout,
                )
            )
            in_channels = {nt: out_dim for nt in self.node_types}

        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.ELU()

    def forward(self, data: HeteroData, x_dict: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """执行前向传播，返回每种节点类型的嵌入。"""
        edge_index_dict = data.edge_index_dict

        # 先通过线性层将不同类型输入映射到统一隐藏维度
        x_dict = {
            node_type: self.input_proj[node_type](x_dict[node_type])
            for node_type in self.node_types
        }

        for layer_idx, conv in enumerate(self.layers):
            # 逐层执行 HGTConv；中间层追加激活与 Dropout 提升泛化能力
            x_dict = conv(x_dict, edge_index_dict)
            if layer_idx < len(self.layers) - 1:
                x_dict = {
                    node_type: self.dropout(self.activation(x))
                    for node_type, x in x_dict.items()
                }
        return x_dict

    @property
    def output_dims(self) -> Dict[str, int]:
        """返回最终输出层各节点类型的通道数。"""
        last_layer: HGTConv = self.layers[-1]
        return {node_type: last_layer.out_channels for node_type in self.node_types}


def contrastive_cmc_loss(
    anchor: Tensor,
    positive: Tensor,
    *,
    temperature: float = 0.2,
) -> Tensor:
    """对比学习损失，使用 InfoNCE / CMC 形式。"""

    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    logits = anchor @ positive.t() / temperature
    labels = torch.arange(anchor.size(0), device=anchor.device)
    return F.cross_entropy(logits, labels)


def train_epoch(
    model: HGTModel,
    data: HeteroData,
    x_dict: Mapping[str, Tensor],
    optimizer: torch.optim.Optimizer,
    positive_edges: Tensor,
    source_type: str,
    target_type: str,
    *,
    device: str | torch.device | None = None,
    temperature: float = 0.2,
) -> float:
    """简化版训练流程，基于全图前向与对比损失。"""

    model.train()
    optimizer.zero_grad()

    # 若调用方未显式指定设备，默认使用模型当前所在设备（通常为 GPU）
    resolved_device = device if device is not None else next(model.parameters()).device

    data = data.to(resolved_device)
    x_dict = {nt: feat.to(resolved_device) for nt, feat in x_dict.items()}

    embeddings = model(data, x_dict)
    src_idx, tgt_idx = positive_edges
    anchor = embeddings[source_type][src_idx]
    positive = embeddings[target_type][tgt_idx]

    loss = contrastive_cmc_loss(anchor, positive, temperature=temperature)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def inference(
    model: HGTModel,
    data: HeteroData,
    x_dict: Mapping[str, Tensor],
    *,
    device: str | torch.device | None = None,
    target_type: str | None = None,
    target_index: Tensor | None = None,
) -> dict[str, Tensor] | Tensor:
    """推理接口，对全图或指定节点生成嵌入。"""

    model.eval()
    with torch.no_grad():
        resolved_device = device if device is not None else next(model.parameters()).device

        data = data.to(resolved_device)
        x_dict = {nt: feat.to(resolved_device) for nt, feat in x_dict.items()}
        embeddings = model(data, x_dict)

        if target_type is None:
            return embeddings

        result = embeddings[target_type]
        if target_index is not None:
            result = result[target_index.to(result.device)]
        return result


def create_neighbor_loader(
    data: HeteroData,
    input_nodes: tuple[str, Tensor] | None,
    *,
    num_neighbors: int | list[int] = 10,
    batch_size: int = 1024,
    shuffle: bool = True,
) -> NeighborLoader:
    """构建异构 NeighborLoader，便于大图采样训练。"""

    return NeighborLoader(
        data,
        input_nodes=input_nodes,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=shuffle,
    )


__all__ = [
    "HGTModelConfig",
    "HGTModel",
    "contrastive_cmc_loss",
    "train_epoch",
    "inference",
    "create_neighbor_loader",
]
