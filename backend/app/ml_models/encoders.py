"""多模态特征编码器集合。

模块职责：

- 为 :class:`~app.ml_models.feature_store.HeteroFeatureEncoder` 提供基础编码组件；
- 与 :mod:`app.graph_services.graph_builder` 中缓存的节点属性配合，
    生成 GNN 模型 :class:`~app.ml_models.hgt_model.HGTModel` 的输入特征；
- 支持数值、类别、多值类别与序列特征的统一封装，便于未来扩展新逻辑。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(slots=True)
class FeatureConfig:
    name: str
    type: str  # "numerical", "categorical", "sequential"
    output_dim: int | None = None
    bins: Sequence[float] | None = None
    vocab_size: int | None = None
    embedding_dim: int | None = None
    multi_valued: bool = False
    channels: int | None = None
    kernel_size: int = 3
    num_layers: int = 2


class BaseEncoder(nn.Module):
    """编码器基类，提供统一的输出维度接口。"""

    def __init__(self, output_dim: int) -> None:
        """初始化基础编码器并记录输出维度。"""
        super().__init__()
        self._output_dim = output_dim

    @property
    def output_dim(self) -> int:
        """返回编码后的特征维度。"""
        return self._output_dim


class NumericalEncoder(BaseEncoder):
    """数值特征编码器：可选分箱与标准化。"""

    def __init__(self, *, bins: Sequence[float] | None = None, output_dim: int = 16) -> None:
        """构造数值特征编码器。

        :param bins: 若提供则使用 ``torch.bucketize`` 将数值离散化；
        :param output_dim: 输出向量维度。
        """
        super().__init__(output_dim)
        self._bins = torch.tensor(bins, dtype=torch.float32) if bins else None
        self._linear = nn.Linear(1 if self._bins is None else len(bins) + 1, output_dim)
        self._norm = nn.BatchNorm1d(self._linear.in_features)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """对数值张量进行归一化/分箱后投影到低维空间。"""
        x = values.float().unsqueeze(-1)
        if self._bins is not None:
            bucket_indices = torch.bucketize(x.squeeze(-1), self._bins)
            x = F.one_hot(bucket_indices, num_classes=len(self._bins) + 1).float()
        x = self._norm(x)
        return F.relu(self._linear(x))


class AttentionPooling(nn.Module):
    """注意力池化，将多个嵌入向量聚合为一个。"""

    def __init__(self, input_dim: int) -> None:
        """初始化注意力得分线性层。"""
        super().__init__()
        self.attn = nn.Linear(input_dim, 1, bias=False)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """根据注意力权重对多值嵌入进行加权求和。"""
        scores = self.attn(embeddings)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(weights * embeddings, dim=1)


class CategoricalEncoder(BaseEncoder):
    """类别特征编码器，支持多类别注意力池化。"""

    def __init__(
        self,
        *,
        vocab_size: int,
        embedding_dim: int,
    multi_valued: bool = False,
        output_dim: int | None = None,
    ) -> None:
        output_dim = output_dim or embedding_dim
        super().__init__(output_dim)

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.multi_valued = multi_valued
        self.pooling = AttentionPooling(embedding_dim) if multi_valued else None
        self.proj = nn.Linear(embedding_dim, output_dim) if output_dim != embedding_dim else nn.Identity()

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """将类别索引编码为密集向量，支持多值聚合。"""
        embedded = self.embedding(indices)
        if self.multi_valued:
            if embedded.dim() != 3:
                raise ValueError("多值类别特征输入需为 [batch, num_values]")
            embedded = self.pooling(embedded)
        else:
            if embedded.dim() == 2:
                pass
            elif embedded.dim() == 3:
                embedded = embedded.mean(dim=1)
            else:  # pragma: no cover - 输入异常
                raise ValueError("类别特征张量维度不受支持")
        return self.proj(embedded)


class SequentialEncoder(BaseEncoder):
    """时序特征编码器，使用TCN结构。"""

    def __init__(
        self,
        *,
        input_channels: int = 1,
        hidden_channels: int = 32,
        kernel_size: int = 3,
        num_layers: int = 2,
        output_dim: int = 64,
    ) -> None:
        super().__init__(output_dim)

        layers: list[nn.Module] = []
        channels = input_channels
        for layer_idx in range(num_layers):
            dilation = 2**layer_idx
            layers.append(
                nn.Conv1d(
                    channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) * dilation,
                    dilation=dilation,
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_channels))
            channels = hidden_channels

        self.tcn = nn.Sequential(*layers)
        self.proj = nn.Linear(hidden_channels, output_dim)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """对时间序列执行一维卷积堆叠并池化为固定长度向量。"""
        if sequences.dim() != 3:
            raise ValueError("时序输入需为 [batch, channels, time_steps]")
        features = self.tcn(sequences)
        features = torch.max(features, dim=-1).values
        return self.proj(features)


class AdaptiveFeatureEncoder(nn.Module):
    """根据配置组合多模态编码器。"""

    def __init__(self, feature_configs: Iterable[FeatureConfig]) -> None:
        """构建特征编码器集合，根据配置选择具体实现。"""
        super().__init__()
        self.feature_configs = list(feature_configs)
        self.encoders = nn.ModuleDict()

        for cfg in self.feature_configs:
            if cfg.type == "numerical":
                encoder = NumericalEncoder(bins=cfg.bins, output_dim=cfg.output_dim or 16)
            elif cfg.type == "categorical":
                if cfg.vocab_size is None or cfg.embedding_dim is None:
                    raise ValueError("categorical 特征需要 vocab_size 与 embedding_dim")
                encoder = CategoricalEncoder(
                    vocab_size=cfg.vocab_size,
                    embedding_dim=cfg.embedding_dim,
                    multi_valued=cfg.multi_valued,
                    output_dim=cfg.output_dim,
                )
            elif cfg.type == "sequential":
                encoder = SequentialEncoder(
                    input_channels=cfg.channels or 1,
                    hidden_channels=cfg.output_dim or 32,
                    kernel_size=cfg.kernel_size,
                    num_layers=cfg.num_layers,
                    output_dim=cfg.output_dim or 64,
                )
            else:
                raise ValueError(f"未知的特征类型: {cfg.type}")
            self.encoders[cfg.name] = encoder

    @property
    def output_dim(self) -> int:
        """返回组合编码器的总输出维度。"""
        return sum(self.encoders[name].output_dim for name in self.encoders)

    def forward(self, features: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """将各特征张量送入对应编码器并拼接结果。"""
        representations = []
        for cfg in self.feature_configs:
            if cfg.name not in features:
                raise KeyError(f"缺少特征输入: {cfg.name}")
            tensor = features[cfg.name]
            representations.append(self.encoders[cfg.name](tensor))
        return torch.cat(representations, dim=-1)


__all__ = [
    "FeatureConfig",
    "BaseEncoder",
    "NumericalEncoder",
    "CategoricalEncoder",
    "SequentialEncoder",
    "AdaptiveFeatureEncoder",
]
