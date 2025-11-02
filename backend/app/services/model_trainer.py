"""HGT 训练服务：负责全量或离线阶段的联合训练。"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Literal

import torch
from torch import Tensor

try:  # pragma: no cover - 运行环境缺少依赖时提示
    from torch_geometric.data import HeteroData  # type: ignore[import]
    from torch_geometric.utils import negative_sampling  # type: ignore[import]
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "HGTTrainer 依赖 torch-geometric，请先安装后再使用。"
    ) from exc

from ..ml_models.feature_store import HeteroFeatureEncoder
from ..ml_models.hgt_model import HGTModel, contrastive_cmc_loss
from ..graph_services.graph_sampler import GraphSampler, MetisSamplingConfig, SamplingStatistics
from .training_config import TrainingConfig

EdgeType = Tuple[str, str, str]


@dataclass(slots=True)
class EdgeSplit:
    """存储对单一关系拆分后的边索引与对应的负样本。"""

    train_pos: Tensor
    val_pos: Tensor
    test_pos: Tensor
    val_neg: Tensor
    test_neg: Tensor


def split_hetero_data_edges(
    data: HeteroData,
    *,
    edge_types: Iterable[EdgeType] | None = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    negative_ratio: float = 1.0,
    seed: int | None = None,
) -> tuple[HeteroData, Dict[EdgeType, EdgeSplit]]:
    """对异构图中的边执行遮蔽拆分，返回训练图与验证/测试集合。

    该函数会按照给定比例随机划分指定关系类型的边，将验证、测试部分
    从返回的训练图中移除，并为评估阶段生成负样本。
    """

    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio 必须处于 (0, 1) 区间内。")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio 必须处于 [0, 1) 区间内。")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio 需小于 1.0，以保留测试集。")
    if negative_ratio < 0.0:
        raise ValueError("negative_ratio 不可为负数。")

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    training_data = data.clone()
    mask_plan = list(edge_types) if edge_types is not None else list(data.edge_types)
    splits: Dict[EdgeType, EdgeSplit] = {}

    for edge_type in mask_plan:
        if edge_type not in data.edge_types:
            continue

        edge_store = data[edge_type]
        edge_index = edge_store.edge_index
        if edge_index is None or edge_index.numel() == 0:
            empty = edge_index.new_empty((2, 0)) if edge_index is not None else torch.empty((2, 0), dtype=torch.long)
            splits[edge_type] = EdgeSplit(empty, empty, empty, empty, empty)
            training_data[edge_type].edge_index = empty
            continue

        # 始终在 CPU 上生成随机序列，再迁移到目标设备，避免生成器设备不匹配导致的错误。
        perm = torch.randperm(edge_index.size(1), generator=generator).to(edge_index.device)

        train_count = max(int(edge_index.size(1) * train_ratio), 1)
        val_count = int(edge_index.size(1) * val_ratio)
        remaining = edge_index.size(1) - train_count
        # 确保验证与测试至少保留 1 条正样本（若剩余为 0，则调整分配）。
        if remaining <= 1:
            train_count = edge_index.size(1) - 2 if edge_index.size(1) >= 2 else edge_index.size(1) - 1
            remaining = edge_index.size(1) - train_count
        val_count = min(val_count, max(remaining - 1, 0))
        test_count = edge_index.size(1) - train_count - val_count

        train_indices = perm[:train_count] if train_count > 0 else perm[:0]
        val_indices = perm[train_count : train_count + val_count] if val_count > 0 else perm[:0]
        test_indices = perm[train_count + val_count :] if test_count > 0 else perm[:0]

        train_edges = edge_index[:, train_indices]
        val_edges = edge_index[:, val_indices]
        test_edges = edge_index[:, test_indices]

        _subset_edge_store(training_data[edge_type], train_indices)

        num_src = data[edge_type[0]].num_nodes
        num_dst = data[edge_type[2]].num_nodes

        val_neg = _sample_negative_edges(
            train_edges,
            num_src,
            num_dst,
            math.ceil(val_edges.size(1) * negative_ratio),
            device=edge_index.device,
        )
        test_neg = _sample_negative_edges(
            train_edges,
            num_src,
            num_dst,
            math.ceil(test_edges.size(1) * negative_ratio),
            device=edge_index.device,
        )

        splits[edge_type] = EdgeSplit(train_edges, val_edges, test_edges, val_neg, test_neg)

    return training_data, splits


def _subset_edge_store(edge_store, train_indices: Tensor) -> None:
    """按照给定索引子集化 edge_store，保持属性对齐。"""

    num_edges = edge_store.edge_index.size(1)
    device = edge_store.edge_index.device
    if train_indices.numel() == num_edges:
        return

    edge_store.edge_index = edge_store.edge_index[:, train_indices]

    for key, value in list(edge_store.items()):
        if key == "edge_index":
            continue
        if torch.is_tensor(value) and value.size(0) == num_edges:
            if train_indices.numel() == 0:
                empty_shape = (0, *value.shape[1:])
                edge_store[key] = value.new_empty(empty_shape)
            else:
                edge_store[key] = value.index_select(0, train_indices.to(value.device))


def _sample_negative_edges(
    reference_edges: Tensor,
    num_src: int,
    num_dst: int,
    num_samples: int,
    *,
    device: torch.device | torch.dtype | str,
) -> Tensor:
    """为评估阶段提供负样本边。"""

    if num_samples <= 0 or num_src == 0 or num_dst == 0:
        return reference_edges.new_empty((2, 0))

    neg_edge_index = negative_sampling(
        edge_index=reference_edges,
        num_nodes=(num_src, num_dst),
        num_neg_samples=num_samples,
        method="sparse",
    )
    return neg_edge_index.to(device)


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

        self._default_lr = next(iter(self.optimizer.param_groups))['lr'] if self.optimizer.param_groups else lr

    def set_learning_rate(self, lr: float | None) -> None:
        """动态调整优化器学习率。"""

        if lr is None or lr <= 0:
            return
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def reset_learning_rate(self) -> None:
        """恢复到初始化时的学习率。"""

        self.set_learning_rate(self._default_lr)

    def train_on_graph(
        self,
        data: HeteroData,
        *,
        epochs: int = 1,
        edge_types: Iterable[EdgeType] | None = None,
        temperature: float = 0.2,
        learning_rate: float | None = None,
        progress_cb: Callable[[Dict[str, Any]], None] | None = None,
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

        original_lr = next(iter(self.optimizer.param_groups))['lr'] if self.optimizer.param_groups else None
        self.set_learning_rate(learning_rate)

        for epoch_idx in range(epochs):
            epoch_start = time.perf_counter()
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
            loss_value = float(loss.item())
            loss_history.append(loss_value)

            if progress_cb is not None:
                try:
                    current_lr = next(iter(self.optimizer.param_groups)).get("lr") if self.optimizer.param_groups else None
                except (StopIteration, AttributeError):
                    current_lr = None
                progress_cb(
                    {
                        "event": "epoch",
                        "epoch": epoch_idx + 1,
                        "total_epochs": epochs,
                        "loss": loss_value,
                        "duration_seconds": time.perf_counter() - epoch_start,
                        "learning_rate": current_lr,
                    }
                )

        self.model.eval()
        self.feature_encoder.eval()
        if original_lr is not None and learning_rate is not None and learning_rate > 0:
            self.set_learning_rate(original_lr)
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


    def evaluate_edge_splits(
        self,
        data: HeteroData,
        splits: Dict[EdgeType, EdgeSplit],
        *,
        split: Literal["val", "test"] = "val",
        full_graph: HeteroData | None = None,
    ) -> Dict[EdgeType, Dict[str, float]]:
        """对遮蔽的验证/测试边计算简单的连接预测指标。
        
        Args:
            data: 训练图（已遮蔽验证/测试边）
            splits: 边拆分信息
            split: 评估哪个拆分（val 或 test）
            full_graph: 完整图（包含所有边），用于生成节点嵌入。如果为 None，使用 data
        """

        if split not in {"val", "test"}:
            raise ValueError("split 仅支持 'val' 或 'test'.")

        # 使用完整图生成嵌入，如果没有提供则使用训练图
        graph_for_embedding = full_graph if full_graph is not None else data
        data_device = graph_for_embedding.to(self.device)
        feature_inputs = self._build_feature_inputs(data_device)

        model_was_training = self.model.training
        encoder_was_training = self.feature_encoder.training
        self.model.eval()
        self.feature_encoder.eval()
        with torch.no_grad():
            embeddings = self.model(data_device, feature_inputs)
        if model_was_training:
            self.model.train()
        if encoder_was_training:
            self.feature_encoder.train()

        metrics: Dict[EdgeType, Dict[str, float]] = {}
        for edge_type, edge_split in splits.items():
            pos_edges = getattr(edge_split, f"{split}_pos")
            neg_edges = getattr(edge_split, f"{split}_neg")
            if pos_edges.numel() == 0:
                metrics[edge_type] = {
                    "auc": _to_float_or_none(float("nan")),
                    "pos_mean": _to_float_or_none(float("nan")),
                    "neg_mean": _to_float_or_none(float("nan")),
                }
                continue

            src_type, _, tgt_type = edge_type
            pos_scores = _edge_scores(embeddings, src_type, tgt_type, pos_edges)
            neg_scores = _edge_scores(embeddings, src_type, tgt_type, neg_edges)
            auc = _binary_auc(pos_scores, neg_scores)
            metrics[edge_type] = {
                "auc": _to_float_or_none(auc),
                "pos_mean": _to_float_or_none(float(pos_scores.mean().item())),
                "neg_mean": _to_float_or_none(float(neg_scores.mean().item())) if neg_scores.numel() else None,
            }

        return metrics

    def run_automated_training(
        self,
        data: HeteroData,
        *,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        negative_ratio: float = 1.0,
        epochs: int = 5,
        temperature: float = 0.2,
        learning_rate: float | None = None,
        seed: int | None = None,
        progress_cb: Callable[[Dict[str, Any]], None] | None = None,
        training_config: TrainingConfig | None = None,
    ) -> Dict[str, Any]:
        """执行遮蔽划分 + 训练 + 评估的自动化流水线。
        
        支持两种训练模式：
        1. 完整图训练（full_graph）：使用所有数据进行训练
        2. METIS 采样训练（metis_sampling）：对图进行 METIS 分割并使用子图训练
        
        Args:
            data: 输入的异构图
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            negative_ratio: 负样本比例
            epochs: 训练轮数
            temperature: 对比学习温度参数
            learning_rate: 学习率
            seed: 随机种子
            progress_cb: 进度回调函数
            training_config: 训练配置（包含训练模式和 METIS 采样参数）
            
        Returns:
            包含损失历史、评估指标、边统计等信息的字典
        """
        # 默认使用完整图训练
        if training_config is None:
            training_config = TrainingConfig()
        
        # 根据训练模式准备训练数据
        training_data_source = data
        sampling_stats: SamplingStatistics | None = None
        
        if training_config.mode == "metis_sampling":
            # 使用 METIS 采样模式
            sampler = GraphSampler()
            metis_config = MetisSamplingConfig(
                num_parts=training_config.metis_num_parts,
                imbalance_factor=training_config.metis_imbalance_factor,
                seed=training_config.metis_seed,
                recursive=training_config.metis_recursive,
            )
            
            if progress_cb is not None:
                progress_cb({
                    "event": "sampling_start",
                    "mode": "metis",
                    "num_parts": training_config.metis_num_parts,
                })
            
            training_data_source, sampling_stats = sampler.metis_sample(
                data,
                metis_config,
                partition_id=training_config.metis_partition_id,
            )
            
            if progress_cb is not None:
                progress_cb({
                    "event": "sampling_complete",
                    "sampling_stats": {
                        "original_nodes": sampling_stats.original_nodes,
                        "original_edges": {" / ".join(k): v for k, v in sampling_stats.original_edges.items()},
                        "sampled_nodes": sampling_stats.sampled_nodes,
                        "sampled_edges": {" / ".join(k): v for k, v in sampling_stats.sampled_edges.items()},
                        "partition_sizes": sampling_stats.partition_sizes,
                        "selected_partition": sampling_stats.selected_partition,
                        "edge_cut": sampling_stats.edge_cut,
                    }
                })

        training_data, splits = split_hetero_data_edges(
            training_data_source,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            negative_ratio=negative_ratio,
            seed=seed,
        )

        def _progress(event: Dict[str, Any]) -> None:
            if progress_cb is None:
                return
            progress_cb(event)

        loss_history = self.train_on_graph(
            training_data,
            epochs=epochs,
            temperature=temperature,
            learning_rate=learning_rate,
            progress_cb=_progress if progress_cb is not None else None,
        )

        if progress_cb is not None:
            progress_cb(
                {
                    "event": "complete",
                    "final_loss": loss_history[-1] if loss_history else None,
                    "loss_history": loss_history,
                }
            )

        # 评估时使用原始完整图生成嵌入
        val_metrics = self.evaluate_edge_splits(training_data, splits, split="val", full_graph=training_data_source)
        test_metrics = self.evaluate_edge_splits(training_data, splits, split="test", full_graph=training_data_source)

        edge_stats: Dict[str, Dict[str, int]] = {}
        for edge_type, edge_split in splits.items():
            key = " / ".join(edge_type)
            edge_stats[key] = {
                "train_pos": int(edge_split.train_pos.size(1)),
                "val_pos": int(edge_split.val_pos.size(1)),
                "test_pos": int(edge_split.test_pos.size(1)),
                "val_neg": int(edge_split.val_neg.size(1)),
                "test_neg": int(edge_split.test_neg.size(1)),
            }

        result: Dict[str, Any] = {
            "loss_history": loss_history,
            "val_metrics": _serialize_metrics(val_metrics),
            "test_metrics": _serialize_metrics(test_metrics),
            "edge_statistics": edge_stats,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "negative_ratio": negative_ratio,
            "epochs": epochs,
            "temperature": temperature,
            "learning_rate": learning_rate,
            "seed": seed,
            "training_mode": training_config.mode,
        }
        
        # 添加采样统计信息（如果使用了 METIS 采样）
        if sampling_stats is not None:
            result["sampling_stats"] = {
                "original_nodes": sampling_stats.original_nodes,
                "original_edges": {" / ".join(k): v for k, v in sampling_stats.original_edges.items()},
                "sampled_nodes": sampling_stats.sampled_nodes,
                "sampled_edges": {" / ".join(k): v for k, v in sampling_stats.sampled_edges.items()},
                "partition_sizes": sampling_stats.partition_sizes,
                "selected_partition": sampling_stats.selected_partition,
                "edge_cut": sampling_stats.edge_cut,
            }
        
        return result


def _edge_scores(
    embeddings: Mapping[str, Tensor],
    source_type: str,
    target_type: str,
    edges: Tensor,
) -> Tensor:
    if edges.numel() == 0:
        return edges.new_empty((0,), dtype=embeddings[source_type].dtype)

    src_idx, tgt_idx = edges[0], edges[1]
    anchor = embeddings[source_type][src_idx]
    positive = embeddings[target_type][tgt_idx]
    return (anchor * positive).sum(dim=-1)


def _binary_auc(pos_scores: Tensor, neg_scores: Tensor) -> float:
    if pos_scores.numel() == 0 or neg_scores.numel() == 0:
        return float("nan")

    scores = torch.cat([pos_scores, neg_scores])
    labels = torch.cat([
        torch.ones(pos_scores.size(0), device=pos_scores.device),
        torch.zeros(neg_scores.size(0), device=neg_scores.device),
    ])

    order = torch.argsort(scores, descending=True)
    sorted_labels = labels[order]

    pos_total = pos_scores.size(0)
    neg_total = neg_scores.size(0)
    if pos_total == 0 or neg_total == 0:
        return float("nan")

    tp = torch.cumsum(sorted_labels, dim=0)
    fp = torch.cumsum(1.0 - sorted_labels, dim=0)

    tpr = tp / pos_total
    fpr = fp / neg_total

    auc = torch.trapz(tpr, fpr)
    return float(auc.item())


def _serialize_metrics(metrics: Dict[EdgeType, Dict[str, float | None]]) -> Dict[str, Dict[str, float | None]]:
    serialized: Dict[str, Dict[str, float | None]] = {}
    for edge_type, values in metrics.items():
        key = " / ".join(edge_type)
        serialized[key] = {metric: value for metric, value in values.items()}
    return serialized


def _to_float_or_none(value: float) -> float | None:
    if math.isnan(value):
        return None
    return float(value)


__all__ = [
    "HGTTrainer",
    "split_hetero_data_edges",
    "EdgeSplit",
]
