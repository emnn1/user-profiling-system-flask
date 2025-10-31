"""轻量级异构特征编码器。

基于 GraphBuilder 的节点映射为每类节点生成固定维度的稠密特征，
用于 HGT 模型输入。这里实现为确定性可复现的哈希初始化，便于
无训练或小样本时也能稳定运行端到端流程。
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from typing import Dict

import torch
from torch import nn

from ..graph_services.graph_builder import GraphBuilder


@dataclass(slots=True)
class FeatureDims:
    user: int = 32
    product: int = 16
    app: int = 16

    def as_dict(self) -> dict[str, int]:
        return {"user": self.user, "product": self.product, "app": self.app}


class HeteroFeatureEncoder(nn.Module):
    """为不同节点类型生成固定维度的稠密特征。

    - 通过对节点 ID 做哈希得到稳定向量；
    - 不依赖外部词表或统计，冷启动友好；
    - 提供 ensure_node 以满足刷新后新节点的声明（本实现为 no-op）。
    """

    def __init__(self, builder: GraphBuilder, dims: FeatureDims | None = None, device: torch.device | None = None) -> None:
        super().__init__()
        self.builder = builder
        self._dims = (dims or FeatureDims()).as_dict()
        self._device = device or torch.device("cpu")
        self._ensured: set[tuple[str, str]] = set()

    @property
    def output_dims(self) -> dict[str, int]:
        return dict(self._dims)

    def to(self, device: torch.device | str):  # type: ignore[override]
        if isinstance(device, str):
            device = torch.device(device)
        self._device = device
        return super().to(device)

    @staticmethod
    def _id_to_vec(node_id: str, dim: int) -> torch.Tensor:
        h = blake2b(node_id.encode("utf-8"), digest_size=32).digest()
        # 展开为 32 字节，重复/截断到需要的维度
        base = torch.tensor(list(h), dtype=torch.float32)
        reps = (dim + base.numel() - 1) // base.numel()
        vec = base.repeat(reps)[:dim]
        # 归一化
        return (vec - vec.mean()) / (vec.std() + 1e-6)

    def build_features_for_graph(self, hetero_data: "HeteroData") -> dict[str, torch.Tensor]:
        x: Dict[str, torch.Tensor] = {}
        mapping = self.builder.reverse_node_mapping
        for nt, dim in self._dims.items():
            ids = mapping.get(nt, [])
            if not ids:
                x[nt] = torch.zeros((0, dim), dtype=torch.float32, device=self._device)
                continue
            mat = torch.stack([self._id_to_vec(_id, dim) for _id in ids]).to(self._device)
            x[nt] = mat
        return x

    def ensure_node(self, node_type: str, node_id: str) -> None:
        # 本实现无需维护词表，仅记录声明以便调试
        self._ensured.add((node_type, node_id))


__all__ = ["HeteroFeatureEncoder", "FeatureDims"]
