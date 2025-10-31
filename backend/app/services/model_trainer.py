"""HGT 训练服务：负责全量或离线阶段的联合训练。"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import torch

try:  # pragma: no cover - 运行环境缺少依赖时提示
    from torch_geometric.data import HeteroData  # type: ignore[import]
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "HGTTrainer 依赖 torch-geometric，请先安装后再使用。"
    ) from exc

from ..ml_models.feature_store import HeteroFeatureEncoder
from ..ml_models.hgt_model import HGTModel, contrastive_cmc_loss

EdgeType = Tuple[str, str, str]


class HGTTrainer:
    """封装 HGT 模型与多模态编码器的联合训练逻辑。"""

    def __init__(
        self,
        *,
        model: HGTModel,
        feature_encoder: HeteroFeatureEncoder,
        device: torch.device | str,
        optimizer: torch.optim.Optimizer | None = None,
        lr: float = 1e-3,
    ) -> None:
        self.model = model
        self.feature_encoder = feature_encoder
        self.device = torch.device(device)
        self.model.to(self.device)
        self.feature_encoder.to(self.device)

        params = list(self.model.parameters()) + list(self.feature_encoder.parameters())
        if optimizer is None:
            self.optimizer = torch.optim.Adam(params, lr=lr)
        else:
            self.optimizer = optimizer

    def train_on_graph(
        self,
        data: HeteroData,
        *,
        epochs: int = 1,
        edge_types: Iterable[EdgeType] | None = None,
        temperature: float = 0.2,
    ) -> List[float]:
        """在给定异构图上执行若干 epoch 的联合训练，返回损失历史。"""

        edge_plan: Sequence[EdgeType]
        if edge_types is None:
            edge_plan = list(data.edge_types)
        else:
            edge_plan = list(edge_types)
        if not edge_plan:
            return []

        loss_history: List[float] = []
        data_device = data.to(self.device)

        for _ in range(epochs):
            self.model.train()
            self.feature_encoder.train()
            self.optimizer.zero_grad(set_to_none=True)

            feature_inputs = self._build_feature_inputs(data_device)
            embeddings = self.model(data_device, feature_inputs)

            total_loss = torch.tensor(0.0, device=self.device)
            edge_count = 0
            for edge_type in edge_plan:
                if edge_type not in data_device.edge_types:
                    continue
                edge_index = data_device[edge_type].edge_index
                if edge_index is None or edge_index.numel() == 0:
                    continue
                src_idx = edge_index[0]
                tgt_idx = edge_index[1]
                anchor = embeddings[edge_type[0]][src_idx]
                positive = embeddings[edge_type[2]][tgt_idx]
                total_loss = total_loss + contrastive_cmc_loss(anchor, positive, temperature=temperature)
                edge_count += 1

            if edge_count == 0:
                self.optimizer.zero_grad(set_to_none=True)
                break

            loss = total_loss / edge_count
            loss.backward()
            self.optimizer.step()
            loss_history.append(float(loss.item()))

        self.model.eval()
        self.feature_encoder.eval()
        return loss_history

    def _build_feature_inputs(self, data: HeteroData) -> dict[str, torch.Tensor]:
        feature_inputs: dict[str, torch.Tensor] = {}
        for node_type in data.node_types:
            num_nodes = data[node_type].num_nodes
            if num_nodes == 0:
                continue
            indices = list(range(num_nodes))
            feature_inputs[node_type] = self.feature_encoder.forward(node_type, indices, self.device)
        return feature_inputs


__all__ = ["HGTTrainer"]
