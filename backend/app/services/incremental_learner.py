"""动态增量学习模块，用于模拟课题三的核心思路。

该模块坐落在若干组件之间：

- :class:`~app.services.data_ingestion.DataIngestionService` 通过 ``drain_events``
    将实时事件批次交给 :meth:`IncrementalLearner.register_events`；
- :class:`~app.graph_services.graph_builder.GraphBuilder` 提供增量更新后的
    PyG ``HeteroData`` 与节点映射；
- :class:`~app.ml_models.feature_store.HeteroFeatureEncoder` 与
    :class:`~app.ml_models.hgt_model.HGTModel` 提供特征编码和 GNN 模型；
- :class:`~app.services.hybrid_profiling_service.HybridProfilingService` 调用
    :meth:`IncrementalLearner.get_latest_embedding` 获取最新用户嵌入。

核心能力：

1. 根据事件优先级调度触发局部图更新；
2. 通过 PyG 的 :func:`NeighborLoader` 对受影响节点采样局部子图并前向刷新嵌入；
3. 缓存历史嵌入以支持快速读取与对比分析。
"""
from __future__ import annotations

import heapq
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import torch

from ..data_source.mock_data_provider import Event, EventType
from ..graph_services.graph_builder import GraphBuilder
from ..ml_models.feature_store import HeteroFeatureEncoder
from ..ml_models.hgt_model import HGTModel, create_neighbor_loader


@dataclass(slots=True)
class PriorityNode:
    """优先级队列节点。"""

    priority: float
    user_id: str
    node_type: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def __lt__(self, other: "PriorityNode") -> bool:  # pragma: no cover - heap比较
        return self.priority > other.priority  # 大顶堆表现


class ChangeDetector:
    """监听事件流以识别受影响节点。"""

    def __init__(self, window: timedelta = timedelta(minutes=10)) -> None:
        """初始化滑动窗口，用于追踪近期事件。"""
        self.window = window
        self.recent_events: Deque[Event] = deque(maxlen=1000)

    def ingest_event(self, event: Event) -> List[Tuple[str, str, datetime]]:
        """记录事件并返回受影响节点集合。"""
        self.recent_events.append(event)
        affected: List[Tuple[str, str, datetime]] = []

        affected.append((event.user_id, "user", event.timestamp))
        if event.event_type is EventType.ORDER and event.product_id:
            affected.append((event.product_id, "product", event.timestamp))
        elif event.event_type is EventType.APP_USAGE and event.app_id:
            affected.append((event.app_id, "app", event.timestamp))
        elif event.event_type is EventType.CALL and event.target_user_id:
            affected.append((event.target_user_id, "user", event.timestamp))
        elif event.event_type is EventType.CLICK:
            if event.product_id:
                affected.append((event.product_id, "product", event.timestamp))
            if event.app_id:
                affected.append((event.app_id, "app", event.timestamp))
        return affected


class IncrementalLearner:
    """结合图更新与节点嵌入增量刷新的服务。"""

    def __init__(
        self,
        *,
        graph_builder: GraphBuilder,
        model: HGTModel,
        feature_encoder: HeteroFeatureEncoder,
        neighbor_sizes: Sequence[int] = (15, 10),
        loader_batch_size: int = 32,
        priority_weights: Tuple[float, float, float] = (0.6, 0.2, 0.2),
    ) -> None:
        """创建增量学习器。

        :param graph_builder: 图构建器，用于维护节点/边结构；
        :param model: HGT 模型，输出节点嵌入；
        :param feature_encoder: 特征编码器，将原始属性转换为张量；
        :param neighbor_sizes: 分层采样邻居数；
        :param loader_batch_size: NeighborLoader 的批大小；
        :param priority_weights: 计算事件优先级的权重 (时间新鲜度、度数、嵌入变化)。
        """
        self.graph_builder = graph_builder
        self.model = model
        self.feature_encoder = feature_encoder
        self.priority_weights = priority_weights
        self.change_detector = ChangeDetector()
        self.priority_queue: List[PriorityNode] = []
        self.node_degrees: Dict[str, int] = defaultdict(int)
        self.last_embeddings: Dict[str, torch.Tensor] = {}
        self.latest_data = None
        self.neighbor_sizes = list(neighbor_sizes)
        self.loader_batch_size = max(1, loader_batch_size)

        # 确保特征编码器与模型在同一设备上
        device = next(self.model.parameters()).device
        self.feature_encoder.to(device)
        self.model.eval()
        self.feature_encoder.eval()

    def register_events(self, events: Iterable[Event]) -> None:
        """消费事件批次并驱动一次增量更新流程。"""
        new_events = list(events)
        if not new_events:
            return

        # 先将事件写入图构建器，产生最新的异构图结构
        hetero_data = self.graph_builder.update_graph_from_events(new_events)
        self.latest_data = hetero_data
        for event in new_events:
            affected_nodes = self.change_detector.ingest_event(event)
            for node_id, node_type, ts in affected_nodes:
                self.node_degrees[node_id] += 1
                self.feature_encoder.ensure_node(node_type, node_id)
                # 根据时间、度数与嵌入变化量计算动态优先级
                priority = self._calculate_priority(node_id, node_type, ts)
                heapq.heappush(
                    self.priority_queue,
                    PriorityNode(priority=priority, user_id=node_id, node_type=node_type, timestamp=ts),
                )

        # 消费优先级队列并触发一次局部增量更新
        self._run_incremental_updates(hetero_data)

    def _calculate_priority(self, node_id: str, node_type: str, event_time: datetime) -> float:
        """计算节点被重新刷新的优先级，用于堆排序。"""
        a, b, c = self.priority_weights
        recency = 1.0 / max((datetime.utcnow() - event_time).total_seconds(), 1.0)
        degree = self.node_degrees.get(node_id, 1)
        embed = self.last_embeddings.get(f"{node_type}:{node_id}")
        delta = torch.norm(embed).item() if embed is not None else 0.1
        return a * recency + b * degree + c * delta

    def _run_incremental_updates(self, hetero_data) -> None:
        """弹出有限数量的高优队列节点并执行局部嵌入刷新。"""
        batch_nodes: List[PriorityNode] = []
        while self.priority_queue and len(batch_nodes) < 32:
            batch_nodes.append(heapq.heappop(self.priority_queue))

        if not batch_nodes:
            return

        node_mappings = self.graph_builder.node_mapping
        grouped: Dict[str, List[Tuple[PriorityNode, int]]] = defaultdict(list)
        for node in batch_nodes:
            idx = node_mappings.get(node.node_type, {}).get(node.user_id)
            if idx is None:
                continue
            grouped[node.node_type].append((node, idx))

        if not grouped:
            return

        device = next(self.model.parameters()).device
        cpu_data = hetero_data.to("cpu")

        self.model.eval()
        self.feature_encoder.eval()

        refreshed_keys: Set[str] = set()
        target_keys = {f"{node.node_type}:{node.user_id}" for node in batch_nodes}

        for node_type, items in grouped.items():
            indices_tensor = torch.tensor([idx for (_, idx) in items], dtype=torch.long)
            if indices_tensor.numel() == 0:
                continue

            loader = create_neighbor_loader(
                cpu_data,
                input_nodes=(node_type, indices_tensor),
                num_neighbors=list(self.neighbor_sizes),
                batch_size=min(self.loader_batch_size, indices_tensor.numel()),
                shuffle=False,
            )
            for sampled in loader:
                sampled = sampled.to(device)
                with torch.no_grad():
                    feature_inputs: Dict[str, torch.Tensor] = {}
                    for nt in sampled.node_types:
                        global_indices = sampled[nt].n_id.tolist()
                        feature_inputs[nt] = self.feature_encoder.forward(nt, global_indices, device)

                    embeddings = self.model(sampled, feature_inputs)

                refreshed = self._cache_sampled_embeddings(sampled, embeddings)
                refreshed_keys.update(refreshed)

        missing = target_keys - refreshed_keys
        if missing:
            self.refresh_all_embeddings()

    def get_latest_embedding(
        self,
        node_type: str,
        node_id: str,
        *,
        refresh: bool = True,
    ) -> Optional[torch.Tensor]:
        """获取指定节点最新的嵌入向量，必要时触发一次懒更新。"""

        key = f"{node_type}:{node_id}"
        cached = self.last_embeddings.get(key)
        if cached is not None:
            return cached.clone()

        if not refresh or self.latest_data is None:
            return None

        hetero_data = self.latest_data
        node_idx = self.graph_builder.node_mapping.get(node_type, {}).get(node_id)
        if node_idx is None:
            return None

        device = next(self.model.parameters()).device
        feature_inputs: Dict[str, torch.Tensor] = {}
        for nt in hetero_data.node_types:
            num_nodes = hetero_data[nt].num_nodes
            if num_nodes == 0:
                continue
            indices = list(range(num_nodes))
            feature_inputs[nt] = self.feature_encoder.forward(nt, indices, device)

        hetero_data_device = hetero_data.to(device)
        model_was_training = self.model.training
        encoder_was_training = self.feature_encoder.training
        self.model.eval()
        self.feature_encoder.eval()
        with torch.no_grad():
            embeddings = self.model(hetero_data_device, feature_inputs)
        if model_was_training:
            self.model.train()
        if encoder_was_training:
            self.feature_encoder.train()

        embedding = embeddings[node_type][node_idx].detach().cpu()
        self.last_embeddings[key] = embedding
        return embedding.clone()

    def refresh_all_embeddings(self) -> None:
        """基于最新图为全部节点重新计算嵌入。"""

        hetero_data = self.latest_data
        if hetero_data is None:
            return
        
        # 检查图中是否有边，如果没有边则跳过嵌入刷新
        has_edges = False
        for edge_type in hetero_data.edge_types:
            edge_store = hetero_data[edge_type]
            if edge_store.edge_index is not None and edge_store.edge_index.numel() > 0:
                has_edges = True
                break
        
        if not has_edges:
            print("警告：图中没有边信息，跳过嵌入刷新")
            return

        device = next(self.model.parameters()).device
        feature_inputs: Dict[str, torch.Tensor] = {}
        for node_type in hetero_data.node_types:
            num_nodes = hetero_data[node_type].num_nodes
            if num_nodes == 0:
                continue
            indices = list(range(num_nodes))
            feature_inputs[node_type] = self.feature_encoder.forward(node_type, indices, device)

        hetero_data_device = hetero_data.to(device)
        model_was_training = self.model.training
        encoder_was_training = self.feature_encoder.training
        self.model.eval()
        self.feature_encoder.eval()
        with torch.no_grad():
            embeddings = self.model(hetero_data_device, feature_inputs)
        if model_was_training:
            self.model.train()
        if encoder_was_training:
            self.feature_encoder.train()

        for node_type in hetero_data.node_types:
            mapping = self.graph_builder.node_mapping.get(node_type, {})
            for node_id, idx in mapping.items():
                if idx >= embeddings[node_type].size(0):
                    continue
                # 缓存全量节点最新嵌入，便于后续读取
                self.last_embeddings[f"{node_type}:{node_id}"] = embeddings[node_type][idx].detach().cpu()

    def reset_with_graph(self, hetero_data) -> None:
        """在全量图重建后，重置内部缓存与优先队列。"""

        self.priority_queue.clear()
        self.node_degrees.clear()
        self.last_embeddings.clear()
        self.latest_data = hetero_data

    def _cache_sampled_embeddings(
        self,
        sampled,
        embeddings: Mapping[str, torch.Tensor],
    ) -> Set[str]:
        """将采样子图中的节点嵌入写入缓存，返回被更新的键集合。"""

        updated: Set[str] = set()
        for node_type in sampled.node_types:
            hetero_store = sampled[node_type]
            if not hasattr(hetero_store, "n_id"):
                indices = list(range(hetero_store.num_nodes))
            else:
                indices = hetero_store.n_id.tolist()
            if not indices:
                continue
            for local_idx, gid in enumerate(indices):
                node_id = self.graph_builder.resolve_node_id(node_type, int(gid))
                if node_id is None:
                    continue
                key = f"{node_type}:{node_id}"
                self.last_embeddings[key] = embeddings[node_type][local_idx].detach().cpu()
                updated.add(key)
        return updated

__all__ = [
    "IncrementalLearner",
    "ChangeDetector",
]
