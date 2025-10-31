"""异构图构建与增量更新模块。"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Sequence

import numpy as np

import pandas as pd
import torch

try:  # pragma: no cover - 若未安装 PyG 会抛出清晰错误
    from torch_geometric.data import HeteroData  # type: ignore[import]
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "GraphBuilder 依赖 torch-geometric，请先安装该库后再使用。"
    ) from exc

from ..data_source.mock_data_provider import Event, EventType

FILE_FORMAT = Literal["csv", "parquet"]
NODE_TYPES = {"user", "product", "app"}
EDGE_TYPE_CALL = ("user", EventType.CALL.value, "user")
EDGE_TYPE_ORDER = ("user", EventType.ORDER.value, "product")
EDGE_TYPE_APP = ("user", EventType.APP_USAGE.value, "app")
EDGE_TYPE_CLICK_PRODUCT = ("user", f"{EventType.CLICK.value}_product", "product")
EDGE_TYPE_CLICK_APP = ("user", f"{EventType.CLICK.value}_app", "app")


@dataclass(slots=True)
class GraphBuildConfig:
    data_dir: Path
    file_format: FILE_FORMAT = "csv"
    device: str = "cpu"
    include_edge_timestamp: bool = True


class GraphBuilder:
    """负责将离线快照与增量事件转换为 PyG ``HeteroData``。"""

    def __init__(self, *, config: GraphBuildConfig) -> None:
        """初始化构建器并为各类节点与属性建立缓存容器。"""
        self.config = config
        self._user_id_map: Dict[str, int] = {}
        self._product_id_map: Dict[str, int] = {}
        self._app_id_map: Dict[str, int] = {}

        self._user_reverse: list[str] = []
        self._product_reverse: list[str] = []
        self._app_reverse: list[str] = []
        self._user_attributes: Dict[str, dict[str, Any]] = {}
        self._product_attributes: Dict[str, dict[str, Any]] = {}
        self._app_attributes: Dict[str, dict[str, Any]] = {}

        self._edge_store: dict[tuple[str, str, str], list[tuple[int, int, float | None]]] = defaultdict(list)
        self._hetero_data: HeteroData | None = None

    @property
    def hetero_data(self) -> HeteroData:
        """返回最近一次构建的 :class:`HeteroData` 实例。"""
        if self._hetero_data is None:
            raise RuntimeError("图尚未构建，请先调用 build_graph_from_snapshot 或 update_graph_from_events。")
        return self._hetero_data

    @property
    def node_mapping(self) -> dict[str, dict[str, int]]:
        """导出节点 ID 到索引的映射，用于增量学习组件。"""
        return {
            "user": dict(self._user_id_map),
            "product": dict(self._product_id_map),
            "app": dict(self._app_id_map),
        }

    @property
    def reverse_node_mapping(self) -> dict[str, list[str]]:
        """导出索引到原始 ID 的反向映射，辅助特征编码器恢复属性。"""
        return {
            "user": list(self._user_reverse),
            "product": list(self._product_reverse),
            "app": list(self._app_reverse),
        }

    def get_node_attributes(self, node_type: str, node_id: str) -> dict[str, Any]:
        """查询指定节点的属性字典。"""
        if node_type == "user":
            return dict(self._user_attributes.get(node_id, {}))
        if node_type == "product":
            return dict(self._product_attributes.get(node_id, {}))
        if node_type == "app":
            return dict(self._app_attributes.get(node_id, {}))
        raise KeyError(f"未知节点类型: {node_type}")

    def get_attributes_by_index(self, node_type: str, index: int) -> dict[str, Any]:
        """根据节点索引读取属性，用于构建采样子图时的映射。"""
        mapping = self.reverse_node_mapping.get(node_type)
        if mapping is None or index >= len(mapping):
            return {}
        node_id = mapping[index]
        return self.get_node_attributes(node_type, node_id)

    def resolve_node_id(self, node_type: str, index: int) -> str | None:
        """将节点索引解析为原始 ID；若越界返回 ``None``。"""
        if node_type == "user":
            return self._user_reverse[index] if index < len(self._user_reverse) else None
        if node_type == "product":
            return self._product_reverse[index] if index < len(self._product_reverse) else None
        if node_type == "app":
            return self._app_reverse[index] if index < len(self._app_reverse) else None
        return None

    def iter_node_attributes(self, node_type: str) -> Iterable[tuple[str, dict[str, Any]]]:
        """遍历指定类型的节点属性，常用于构建词表或统计。"""
        if node_type == "user":
            return self._user_attributes.items()
        if node_type == "product":
            return self._product_attributes.items()
        if node_type == "app":
            return self._app_attributes.items()
        raise KeyError(f"未知节点类型: {node_type}")

    def build_graph_from_snapshot(
        self,
        *,
        sample_ratio: float = 1.0,
        random_state: int | None = None,
    ) -> HeteroData:
        """从静态快照文件构建异构图，可选按比例采样。"""
        if not 0 < sample_ratio <= 1.0:
            raise ValueError("sample_ratio 必须位于 (0, 1] 区间")

        rng = np.random.default_rng(random_state)

        users_df = self._load_dataframe("users")
        products_df = self._load_dataframe("products")
        apps_df = self._load_dataframe("apps")
        events_df = self._load_dataframe("events")

        if sample_ratio < 1.0:
            users_df = self._sample_frame(users_df, sample_ratio, rng)
            products_df = self._sample_frame(products_df, sample_ratio, rng)
            apps_df = self._sample_frame(apps_df, sample_ratio, rng)

            user_ids = set(users_df["user_id"].astype(str))
            product_ids = set(products_df["product_id"].astype(str))
            app_ids = set(apps_df["app_id"].astype(str))

            events_df = events_df[
                events_df["user_id"].astype(str).isin(user_ids)
            ]

            def _target_filter(row: pd.Series) -> bool:
                if row["event_type"] == EventType.ORDER.value:
                    return pd.isna(row.get("product_id")) or str(row["product_id"]) in product_ids
                if row["event_type"] == EventType.APP_USAGE.value:
                    return pd.isna(row.get("app_id")) or str(row["app_id"]) in app_ids
                if row["event_type"] == EventType.CLICK.value:
                    pid = row.get("product_id")
                    aid = row.get("app_id")
                    product_ok = pd.isna(pid) or str(pid) in product_ids
                    app_ok = pd.isna(aid) or str(aid) in app_ids
                    return product_ok and app_ok
                if row["event_type"] == EventType.CALL.value:
                    target = row.get("target_user_id")
                    return pd.isna(target) or str(target) in user_ids
                return True

            events_df = events_df[events_df.apply(_target_filter, axis=1)]

            if not events_df.empty:
                events_df = self._sample_frame(events_df, sample_ratio, rng).sort_values("timestamp")

        self._reset_state()
        self._register_nodes(users_df, products_df, apps_df)
        self._append_edges_from_events(events_df.itertuples(index=False))

        self._hetero_data = self._build_hetero_data()
        return self._hetero_data

    @staticmethod
    def _sample_frame(df: pd.DataFrame, ratio: float, rng: np.random.Generator) -> pd.DataFrame:
        if df.empty:
            return df
        count = max(1, int(round(len(df) * ratio)))
        count = min(count, len(df))
        indices = rng.choice(len(df), size=count, replace=False)
        return df.iloc[sorted(indices)].reset_index(drop=True)

    def update_graph_from_events(self, events: Sequence[Event] | Iterable[Event]) -> HeteroData:
        """根据增量事件更新边列表，并在必要时重建 ``HeteroData``。"""
        if isinstance(events, Sequence):
            iterator = iter(events)
        else:
            iterator = events

        new_edges = False
        for event in iterator:
            new_edges |= self._append_event(event)

        if new_edges:
            self._hetero_data = self._build_hetero_data()
        elif self._hetero_data is None:
            self._hetero_data = self._build_hetero_data()
        return self._hetero_data

    def _register_nodes(self, users_df: pd.DataFrame, products_df: pd.DataFrame, apps_df: pd.DataFrame) -> None:
        """将 CSV/Parquet 的实体表注册为节点并缓存原始属性。"""
        for _, row in users_df.iterrows():
            user_id = str(row["user_id"])
            attrs = row.to_dict()
            self._ensure_user(user_id, attrs)
        for _, row in products_df.iterrows():
            product_id = str(row["product_id"])
            attrs = row.to_dict()
            self._ensure_product(product_id, attrs)
        for _, row in apps_df.iterrows():
            app_id = str(row["app_id"])
            attrs = row.to_dict()
            self._ensure_app(app_id, attrs)

    def _append_edges_from_events(self, events: Iterable[pd.Series | object]) -> None:
        """将历史事件记录转换为边并写入缓存。"""
        for row in events:
            if isinstance(row, pd.Series):
                event = self._row_to_event(row)
            elif hasattr(row, "_asdict"):
                event = self._row_to_event(pd.Series(row._asdict()))
            elif isinstance(row, Event):
                event = row
            elif isinstance(row, dict):
                event = self._row_to_event(pd.Series(row))
            else:  # pragma: no cover - 非预期数据格式
                raise TypeError(f"无法识别的事件记录类型: {type(row)!r}")
            self._append_event(event)

    def _append_event(self, event: Event) -> bool:
        """解析事件并向不同类型的边列表追加边，返回是否新增边。"""
        source = self._ensure_user(event.user_id)
        timestamp = event.timestamp.timestamp() if isinstance(event.timestamp, datetime) else None
        has_new_edge = False

        if event.event_type is EventType.CALL:
            if event.target_user_id:
                target = self._ensure_user(event.target_user_id)
                # 通话事件写入 user-user 边，并记录时间戳
                self._edge_store[EDGE_TYPE_CALL].append((source, target, timestamp))
                has_new_edge = True
        elif event.event_type is EventType.ORDER and event.product_id:
            target = self._ensure_product(event.product_id)
            self._edge_store[EDGE_TYPE_ORDER].append((source, target, timestamp))
            has_new_edge = True
        elif event.event_type is EventType.APP_USAGE and event.app_id:
            target = self._ensure_app(event.app_id)
            self._edge_store[EDGE_TYPE_APP].append((source, target, timestamp))
            has_new_edge = True
        elif event.event_type is EventType.CLICK:
            if event.product_id:
                target_product = self._ensure_product(event.product_id)
                # 区分点击产品与 APP 的边类型，便于后续建模
                self._edge_store[EDGE_TYPE_CLICK_PRODUCT].append((source, target_product, timestamp))
                has_new_edge = True
            if event.app_id:
                target_app = self._ensure_app(event.app_id)
                self._edge_store[EDGE_TYPE_CLICK_APP].append((source, target_app, timestamp))
                has_new_edge = True
        return has_new_edge

    def _build_hetero_data(self) -> HeteroData:
        """根据缓存的节点与边构建 PyG ``HeteroData`` 对象。"""
        data = HeteroData()

        data["user"].num_nodes = len(self._user_id_map)
        data["product"].num_nodes = len(self._product_id_map)
        data["app"].num_nodes = len(self._app_id_map)

        for edge_type, edges in self._edge_store.items():
            if not edges:
                continue
            sources, targets, timestamps = zip(*edges)
            # 将源、目标索引拼成 PyG 要求的 edge_index
            edge_index = torch.tensor([sources, targets], dtype=torch.long)
            data[edge_type].edge_index = edge_index
            if self.config.include_edge_timestamp:
                ts_tensor = torch.tensor(
                    [ts if ts is not None else 0.0 for ts in timestamps],
                    dtype=torch.float32,
                )
                data[edge_type].edge_timestamp = ts_tensor

        return data.to(self.config.device)

    def _ensure_user(self, user_id: str, attributes: dict[str, Any] | None = None) -> int:
        """确保用户节点存在，并返回其索引。"""
        return self._ensure_node(
            node_id=user_id,
            target_map=self._user_id_map,
            reverse_map=self._user_reverse,
            attr_store=self._user_attributes,
            attributes=attributes,
        )

    def _ensure_product(self, product_id: str, attributes: dict[str, Any] | None = None) -> int:
        """确保产品节点存在，并返回其索引。"""
        return self._ensure_node(
            node_id=product_id,
            target_map=self._product_id_map,
            reverse_map=self._product_reverse,
            attr_store=self._product_attributes,
            attributes=attributes,
        )

    def _ensure_app(self, app_id: str, attributes: dict[str, Any] | None = None) -> int:
        """确保应用节点存在，并返回其索引。"""
        return self._ensure_node(
            node_id=app_id,
            target_map=self._app_id_map,
            reverse_map=self._app_reverse,
            attr_store=self._app_attributes,
            attributes=attributes,
        )

    @staticmethod
    def _ensure_node(
        node_id: str,
        target_map: Dict[str, int],
        *,
        reverse_map: list[str],
        attr_store: Dict[str, dict[str, Any]],
        attributes: dict[str, Any] | None = None,
    ) -> int:
        """通用节点注册逻辑，维护映射与属性缓存。"""
        if node_id not in target_map:
            target_map[node_id] = len(target_map)
            reverse_map.append(node_id)
        if node_id not in attr_store:
            attr_store[node_id] = attributes or {}
        elif attributes is not None:
            attr_store[node_id].update({k: v for k, v in attributes.items() if v is not None})
        return target_map[node_id]

    def _load_dataframe(self, name: str) -> pd.DataFrame:
        """从配置目录中加载指定表格。"""
        path = self.config.data_dir / f"{name}.{self.config.file_format}"
        if not path.exists():
            raise FileNotFoundError(f"未找到数据文件: {path}")
        if self.config.file_format == "csv":
            df = pd.read_csv(path)
        else:
            df = pd.read_parquet(path)
        return df

    @staticmethod
    def _row_to_event(row: pd.Series) -> Event:
        """将 DataFrame 行转换为 :class:`Event` 对象。"""
        timestamp = row["timestamp"]
        if not isinstance(timestamp, datetime):
            timestamp = pd.to_datetime(timestamp)
        return Event(
            event_id=str(row["event_id"]),
            timestamp=timestamp.to_pydatetime(),
            user_id=str(row["user_id"]),
            event_type=EventType(str(row["event_type"])),
            target_user_id=row.get("target_user_id") if pd.notna(row.get("target_user_id")) else None,
            product_id=row.get("product_id") if pd.notna(row.get("product_id")) else None,
            app_id=row.get("app_id") if pd.notna(row.get("app_id")) else None,
            duration_seconds=int(row["duration_seconds"]) if pd.notna(row.get("duration_seconds")) else None,
        )

    def _reset_state(self) -> None:
        """清空缓存，确保重新构建时状态一致。"""
        self._user_id_map.clear()
        self._product_id_map.clear()
        self._app_id_map.clear()
        self._user_reverse.clear()
        self._product_reverse.clear()
        self._app_reverse.clear()
        self._user_attributes.clear()
        self._product_attributes.clear()
        self._app_attributes.clear()
        self._edge_store.clear()
        self._hetero_data = None


__all__ = [
    "GraphBuildConfig",
    "GraphBuilder",
]
