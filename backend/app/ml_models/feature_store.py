"""多模态特征缓冲与编码管线。

模块桥接 :class:`~app.graph_services.graph_builder.GraphBuilder` 输出的节点属性
与 :mod:`app.ml_models.encoders` 的多模态编码器，向
:class:`~app.ml_models.hgt_model.HGTModel` 提供张量化特征。

职责：

- 维护类别词表 (:class:`CategoricalVocabulary`)；
- 根据节点类型构造输入字典；
- 支持前端增量新增节点时的特征补全。
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch
from torch import Tensor, nn

from .encoders import AdaptiveFeatureEncoder, FeatureConfig
from ..graph_services.graph_builder import GraphBuilder


class CategoricalVocabulary:
    """简单的类别词表，包含一个保留的未知标记。"""

    def __init__(self, values: Iterable[Any], *, unk_token: str = "<UNK>") -> None:
        """收集取值集合并建立映射，首位保留未知标记。"""
        self._unk_token = unk_token
        self._value_to_index: Dict[str, int] = {unk_token: 0}
        for value in values:
            key = self._normalize(value)
            if key and key not in self._value_to_index:
                self._value_to_index[key] = len(self._value_to_index)

    @staticmethod
    def _normalize(value: Any) -> str:
        """将输入标准化为字符串索引键。"""
        if value is None:
            return ""
        text = str(value).strip()
        return text

    @property
    def size(self) -> int:
        """返回词表大小（含未知标记）。"""
        return len(self._value_to_index)

    def lookup(self, value: Any) -> int:
        """将取值映射为索引，未知值返回 0。"""
        key = self._normalize(value)
        if not key:
            return 0
        return self._value_to_index.get(key, 0)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


class HeteroFeatureEncoder(nn.Module):
    """根据图构建器信息生成可训练的多模态节点特征。"""

    def __init__(self, graph_builder: GraphBuilder) -> None:
        """根据图构建器的节点属性初始化词表与编码器。"""
        super().__init__()
        self.graph_builder = graph_builder

        self.plan_vocab = CategoricalVocabulary(
            value
            for _, attrs in self.graph_builder.iter_node_attributes("user")
            if (value := attrs.get("plan_type")) is not None
        )
        self.level_vocab = CategoricalVocabulary(
            value
            for _, attrs in self.graph_builder.iter_node_attributes("user")
            if (value := attrs.get("user_level")) is not None
        )
        self.brand_vocab = CategoricalVocabulary(
            value
            for _, attrs in self.graph_builder.iter_node_attributes("user")
            if (value := attrs.get("device_brand")) is not None
        )
        self.product_name_vocab = CategoricalVocabulary(
            value
            for _, attrs in self.graph_builder.iter_node_attributes("product")
            if (value := attrs.get("product_name")) is not None
        )
        self.app_name_vocab = CategoricalVocabulary(
            value
            for _, attrs in self.graph_builder.iter_node_attributes("app")
            if (value := attrs.get("app_name")) is not None
        )

        self.user_encoder = AdaptiveFeatureEncoder(
            [
                FeatureConfig(name="monthly_fee", type="numerical", output_dim=32),
                FeatureConfig(name="tenure_months", type="numerical", output_dim=32),
                FeatureConfig(
                    name="plan_type",
                    type="categorical",
                    vocab_size=max(self.plan_vocab.size, 1),
                    embedding_dim=32,
                ),
                FeatureConfig(
                    name="user_level",
                    type="categorical",
                    vocab_size=max(self.level_vocab.size, 1),
                    embedding_dim=16,
                ),
                FeatureConfig(
                    name="device_brand",
                    type="categorical",
                    vocab_size=max(self.brand_vocab.size, 1),
                    embedding_dim=16,
                ),
            ]
        )

        self.product_encoder = AdaptiveFeatureEncoder(
            [
                FeatureConfig(name="price", type="numerical", output_dim=32),
                FeatureConfig(
                    name="product_name",
                    type="categorical",
                    vocab_size=max(self.product_name_vocab.size, 1),
                    embedding_dim=32,
                ),
            ]
        )

        self.app_encoder = AdaptiveFeatureEncoder(
            [
                FeatureConfig(
                    name="app_name",
                    type="categorical",
                    vocab_size=max(self.app_name_vocab.size, 1),
                    embedding_dim=32,
                )
            ]
        )

        self.encoders = nn.ModuleDict(
            {
                "user": self.user_encoder,
                "product": self.product_encoder,
                "app": self.app_encoder,
            }
        )

    @property
    def output_dims(self) -> Dict[str, int]:
        """返回每种节点类型的编码后维度。"""
        return {node_type: encoder.output_dim for node_type, encoder in self.encoders.items()}

    def forward(self, node_type: str, node_indices: Sequence[int], device: torch.device) -> Tensor:
        """根据节点全局索引编码特征，返回批量张量。"""
        encoder = self.encoders[node_type]
        feature_payload = self._build_feature_payload(node_type, node_indices, device)
        return encoder(feature_payload)

    def encode_subgraph(self, batch: Mapping[str, Any], device: torch.device) -> dict[str, Tensor]:
        """为 NeighborLoader 采样出的子图生成节点特征。"""

        features: dict[str, Tensor] = {}
        for node_type in batch.metadata()[0]:
            hetero_store = batch[node_type]
            if not hasattr(hetero_store, "n_id"):
                indices = torch.arange(hetero_store.num_nodes)
            else:
                indices = hetero_store.n_id
            global_indices = indices.tolist()
            features[node_type] = self.forward(node_type, global_indices, device)
        return features

    def ensure_node(self, node_type: str, node_id: str) -> None:
        """触发一次特征缓存更新，确保新增节点拥有可用特征。"""
        # 当前实现按需构建特征，无需显式缓存，但保留接口便于未来扩展。
        _ = self.graph_builder.get_node_attributes(node_type, node_id)

    def _build_feature_payload(
        self,
        node_type: str,
        node_indices: Sequence[int],
        device: torch.device,
    ) -> dict[str, Tensor]:
        """构造编码器输入字典，自动处理缺失节点。"""
        node_ids: List[str] = []
        for index in node_indices:
            node_id = self.graph_builder.resolve_node_id(node_type, int(index))
            if node_id is None:
                # 处理稀疏采样带来的空洞索引
                node_ids.append("<UNK>")
            else:
                node_ids.append(node_id)

        if node_type == "user":
            return self._tensorize_user(node_ids, device)
        if node_type == "product":
            return self._tensorize_product(node_ids, device)
        if node_type == "app":
            return self._tensorize_app(node_ids, device)
        raise KeyError(f"未知节点类型: {node_type}")

    def _tensorize_user(self, node_ids: Sequence[str], device: torch.device) -> dict[str, Tensor]:
        """将用户节点属性转为张量字典。"""
        monthly_fee: List[float] = []
        tenure: List[float] = []
        plan_idx: List[int] = []
        level_idx: List[int] = []
        brand_idx: List[int] = []

        for node_id in node_ids:
            attrs = self.graph_builder.get_node_attributes("user", node_id) if node_id != "<UNK>" else {}
            monthly_fee.append(_safe_float(attrs.get("monthly_fee"), 0.0))
            tenure.append(float(_safe_int(attrs.get("tenure_months"), 0)))
            plan_idx.append(self.plan_vocab.lookup(attrs.get("plan_type")))
            level_idx.append(self.level_vocab.lookup(attrs.get("user_level")))
            brand_idx.append(self.brand_vocab.lookup(attrs.get("device_brand")))

        return {
            "monthly_fee": torch.tensor(monthly_fee, dtype=torch.float32, device=device),
            "tenure_months": torch.tensor(tenure, dtype=torch.float32, device=device),
            "plan_type": torch.tensor(plan_idx, dtype=torch.long, device=device),
            "user_level": torch.tensor(level_idx, dtype=torch.long, device=device),
            "device_brand": torch.tensor(brand_idx, dtype=torch.long, device=device),
        }

    def _tensorize_product(self, node_ids: Sequence[str], device: torch.device) -> dict[str, Tensor]:
        """将产品节点属性转为张量字典。"""
        price: List[float] = []
        name_idx: List[int] = []
        for node_id in node_ids:
            attrs = self.graph_builder.get_node_attributes("product", node_id) if node_id != "<UNK>" else {}
            price.append(_safe_float(attrs.get("price"), 0.0))
            name_idx.append(self.product_name_vocab.lookup(attrs.get("product_name")))
        return {
            "price": torch.tensor(price, dtype=torch.float32, device=device),
            "product_name": torch.tensor(name_idx, dtype=torch.long, device=device),
        }

    def _tensorize_app(self, node_ids: Sequence[str], device: torch.device) -> dict[str, Tensor]:
        """将应用节点属性转为张量字典。"""
        name_idx: List[int] = []
        for node_id in node_ids:
            attrs = self.graph_builder.get_node_attributes("app", node_id) if node_id != "<UNK>" else {}
            name_idx.append(self.app_name_vocab.lookup(attrs.get("app_name")))
        return {
            "app_name": torch.tensor(name_idx, dtype=torch.long, device=device),
        }


__all__ = ["HeteroFeatureEncoder", "CategoricalVocabulary"]
