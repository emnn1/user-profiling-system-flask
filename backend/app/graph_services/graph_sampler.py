"""图采样模块：支持 METIS 分割采样策略。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

try:
    from torch_geometric.data import HeteroData  # type: ignore[import]
    from torch_geometric.utils import to_scipy_sparse_matrix  # type: ignore[import]
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "GraphSampler 依赖 torch-geometric，请先安装该库后再使用。"
    ) from exc

try:
    import pymetis  # type: ignore[import]
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "GraphSampler 依赖 pymetis，请先安装该库后再使用 METIS 采样功能。"
    ) from exc


@dataclass(slots=True)
class MetisSamplingConfig:
    """METIS 图分割采样配置。
    
    Attributes:
        num_parts: 分区数量，默认为 4
        imbalance_factor: 不平衡因子，允许分区大小不均衡的程度（0.0-1.0），默认 0.01
        seed: 随机种子，用于可重复的分区结果
        recursive: 是否使用递归二分法，默认 True
    """
    num_parts: int = 4
    imbalance_factor: float = 0.01
    seed: int | None = None
    recursive: bool = True
    
    def __post_init__(self):
        if self.num_parts < 2:
            raise ValueError("num_parts 必须 >= 2")
        if not 0.0 <= self.imbalance_factor <= 1.0:
            raise ValueError("imbalance_factor 必须在 [0, 1] 区间内")


@dataclass(slots=True)
class SamplingStatistics:
    """采样统计信息。
    
    Attributes:
        original_nodes: 原始图节点数（按类型）
        original_edges: 原始图边数（按类型）
        sampled_nodes: 采样后节点数（按类型）
        sampled_edges: 采样后边数（按类型）
        partition_sizes: 每个分区的节点数
        selected_partition: 选中的分区 ID
        edge_cut: 边切割数量
    """
    original_nodes: Dict[str, int]
    original_edges: Dict[Tuple[str, str, str], int]
    sampled_nodes: Dict[str, int]
    sampled_edges: Dict[Tuple[str, str, str], int]
    partition_sizes: List[int]
    selected_partition: int
    edge_cut: int


class GraphSampler:
    """异构图采样器，支持 METIS 分割策略。"""
    
    def __init__(self):
        """初始化图采样器。"""
        pass
    
    def metis_sample(
        self,
        data: HeteroData,
        config: MetisSamplingConfig,
        *,
        partition_id: int | None = None,
    ) -> Tuple[HeteroData, SamplingStatistics]:
        """使用 METIS 分割算法对异构图进行采样。
        
        该方法将异构图转换为同构表示，使用 METIS 进行分区，
        然后选择一个分区并提取对应的子图。
        
        Args:
            data: 输入的异构图
            config: METIS 采样配置
            partition_id: 指定选择的分区 ID，若为 None 则随机选择
            
        Returns:
            采样后的子图和统计信息
        """
        # 统计原始图信息
        original_node_counts = {
            node_type: data[node_type].num_nodes
            for node_type in data.node_types
        }
        original_edge_counts = {
            edge_type: data[edge_type].edge_index.size(1)
            for edge_type in data.edge_types
            if data[edge_type].edge_index is not None
        }
        
        # 构建节点类型到全局索引的映射
        node_type_offset = {}
        total_nodes = 0
        for node_type in sorted(data.node_types):
            node_type_offset[node_type] = total_nodes
            total_nodes += data[node_type].num_nodes
        
        if total_nodes == 0:
            raise ValueError("输入图为空，无法进行采样")
        
        # 构建全局邻接表（合并所有异构边为同构图）
        adjacency = [[] for _ in range(total_nodes)]
        
        for edge_type in data.edge_types:
            edge_index = data[edge_type].edge_index
            if edge_index is None or edge_index.numel() == 0:
                continue
            
            src_type, _, dst_type = edge_type
            src_offset = node_type_offset[src_type]
            dst_offset = node_type_offset[dst_type]
            
            # 转换为全局索引并构建邻接表
            src_nodes = edge_index[0].cpu().numpy() + src_offset
            dst_nodes = edge_index[1].cpu().numpy() + dst_offset
            
            for src, dst in zip(src_nodes, dst_nodes):
                if dst not in adjacency[src]:
                    adjacency[src].append(dst)
                # 确保无向图（METIS 要求）
                if src not in adjacency[dst]:
                    adjacency[dst].append(src)
        
        # 执行 METIS 分区
        n_cuts, membership = pymetis.part_graph(
            nparts=config.num_parts,
            adjacency=adjacency,
            recursive=config.recursive,
        )
        
        # 选择分区
        if partition_id is None:
            rng = np.random.default_rng(config.seed)
            partition_id = rng.integers(0, config.num_parts)
        else:
            if not 0 <= partition_id < config.num_parts:
                raise ValueError(f"partition_id 必须在 [0, {config.num_parts}) 区间内")
        
        # 统计每个分区的节点数
        membership_array = np.array(membership)
        partition_sizes = [int(np.sum(membership_array == i)) for i in range(config.num_parts)]
        
        # 提取选中分区的节点（全局索引）
        selected_global_nodes = np.where(membership_array == partition_id)[0]
        selected_nodes_set = set(selected_global_nodes)
        
        # 为每个节点类型提取子集
        sampled_data = HeteroData()
        node_type_mapping = {}  # 全局索引 -> 采样后索引
        
        for node_type in sorted(data.node_types):
            offset = node_type_offset[node_type]
            num_nodes = data[node_type].num_nodes
            
            # 找出该类型中属于选中分区的节点
            type_global_indices = np.arange(offset, offset + num_nodes)
            mask = np.isin(type_global_indices, selected_global_nodes)
            selected_type_indices = np.where(mask)[0]
            
            # 复制节点特征
            if hasattr(data[node_type], 'x') and data[node_type].x is not None:
                sampled_data[node_type].x = data[node_type].x[selected_type_indices]
            
            sampled_data[node_type].num_nodes = len(selected_type_indices)
            
            # 构建映射：旧全局索引 -> 新局部索引
            for new_idx, old_local_idx in enumerate(selected_type_indices):
                old_global_idx = offset + old_local_idx
                node_type_mapping[old_global_idx] = (node_type, new_idx)
        
        # 提取边子集
        sampled_edge_counts = {}
        
        for edge_type in data.edge_types:
            edge_index = data[edge_type].edge_index
            if edge_index is None or edge_index.numel() == 0:
                sampled_data[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)
                sampled_edge_counts[edge_type] = 0
                continue
            
            src_type, _, dst_type = edge_type
            src_offset = node_type_offset[src_type]
            dst_offset = node_type_offset[dst_type]
            
            # 转换为全局索引
            src_global = edge_index[0].cpu().numpy() + src_offset
            dst_global = edge_index[1].cpu().numpy() + dst_offset
            
            # 筛选两端都在选中分区的边
            valid_edges = []
            for i, (src_g, dst_g) in enumerate(zip(src_global, dst_global)):
                if src_g in selected_nodes_set and dst_g in selected_nodes_set:
                    # 转换为新的局部索引
                    _, new_src = node_type_mapping[src_g]
                    _, new_dst = node_type_mapping[dst_g]
                    valid_edges.append([new_src, new_dst])
            
            if valid_edges:
                new_edge_index = torch.tensor(valid_edges, dtype=torch.long).t().contiguous()
                sampled_data[edge_type].edge_index = new_edge_index.to(edge_index.device)
                sampled_edge_counts[edge_type] = len(valid_edges)
            else:
                sampled_data[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
                sampled_edge_counts[edge_type] = 0
            
            # 复制边属性
            if hasattr(data[edge_type], 'edge_attr') and data[edge_type].edge_attr is not None:
                if valid_edges:
                    edge_indices = [i for i, (src_g, dst_g) in enumerate(zip(src_global, dst_global))
                                    if src_g in selected_nodes_set and dst_g in selected_nodes_set]
                    sampled_data[edge_type].edge_attr = data[edge_type].edge_attr[edge_indices]
        
        # 统计采样后节点数
        sampled_node_counts = {
            node_type: sampled_data[node_type].num_nodes
            for node_type in data.node_types
        }
        
        # 构建统计信息
        stats = SamplingStatistics(
            original_nodes=original_node_counts,
            original_edges=original_edge_counts,
            sampled_nodes=sampled_node_counts,
            sampled_edges=sampled_edge_counts,
            partition_sizes=partition_sizes,
            selected_partition=partition_id,
            edge_cut=n_cuts,
        )
        
        return sampled_data, stats
