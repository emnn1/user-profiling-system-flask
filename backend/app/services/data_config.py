"""数据生成配置管理服务。

提供数据生成参数的存储、更新与持久化能力，
并支持通过 API 动态调整用户数、产品数、事件密度等参数。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict


@dataclass(slots=True)
class DataGenerationConfig:
    """数据生成配置参数。
    
    该类封装了模拟数据生成的所有可调参数，包括：
    - 实体规模：用户数、商品数、APP数
    - 事件密度：每用户平均事件数
    - 时间跨度：历史数据天数
    - 初始随机边：当无历史事件时，为每个用户按范围生成的随机边数量
    
    Attributes:
        user_count (int): 模拟用户总数，范围 10 - 100,000
        product_count (int): 商品总数，范围 5 - 1,000
        app_count (int): APP 总数，范围 5 - 500
        avg_events_per_user (int): 每用户平均生成的事件数，范围 1 - 1,000
        history_days (int): 历史数据时间跨度（天），范围 1 - 365
        min/max_* (int): 各边类型每用户生成的最小/最大数量
    """
    
    user_count: int = 1_000              # 默认生成 1000 个用户
    product_count: int = 25              # 默认 25 种商品
    app_count: int = 30                  # 默认 30 个 APP
    avg_events_per_user: int = 20        # 默认每用户 20 个事件
    history_days: int = 30               # 默认 30 天历史数据

    # 初始随机边（每个用户）的数量范围（当 events 为空时生效）
    min_orders_per_user: int = 1
    max_orders_per_user: int = 3
    min_app_usages_per_user: int = 1
    max_app_usages_per_user: int = 3
    min_calls_per_user: int = 0
    max_calls_per_user: int = 2
    min_click_products_per_user: int = 0
    max_click_products_per_user: int = 5
    min_click_apps_per_user: int = 0
    max_click_apps_per_user: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式。
        
        用于 JSON 序列化和 API 响应。
        
        Returns:
            Dict[str, Any]: 配置参数的字典表示
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DataGenerationConfig:
        """从字典构建配置对象。
        
        支持从 JSON 或 API 请求体反序列化配置对象。
        缺失的字段将使用默认值。
        
        Args:
            data (Dict[str, Any]): 包含配置参数的字典
            
        Returns:
            DataGenerationConfig: 配置对象实例
        """
        return cls(
            user_count=int(data.get("user_count", 1_000)),
            product_count=int(data.get("product_count", 25)),
            app_count=int(data.get("app_count", 30)),
            avg_events_per_user=int(data.get("avg_events_per_user", 20)),
            history_days=int(data.get("history_days", 30)),
            min_orders_per_user=int(data.get("min_orders_per_user", 1)),
            max_orders_per_user=int(data.get("max_orders_per_user", 3)),
            min_app_usages_per_user=int(data.get("min_app_usages_per_user", 1)),
            max_app_usages_per_user=int(data.get("max_app_usages_per_user", 3)),
            min_calls_per_user=int(data.get("min_calls_per_user", 0)),
            max_calls_per_user=int(data.get("max_calls_per_user", 2)),
            min_click_products_per_user=int(data.get("min_click_products_per_user", 0)),
            max_click_products_per_user=int(data.get("max_click_products_per_user", 5)),
            min_click_apps_per_user=int(data.get("min_click_apps_per_user", 0)),
            max_click_apps_per_user=int(data.get("max_click_apps_per_user", 5)),
        )
    
    def validate(self) -> None:
        """验证配置参数的有效性。
        
        检查所有参数是否在允许的范围内，超出范围时抛出异常。
        该方法在配置更新前应被调用，防止无效配置导致系统异常。
        
        Raises:
            ValueError: 当任何参数超出允许范围时
            
        Examples:
            >>> config = DataGenerationConfig(user_count=500000)
            >>> config.validate()  # 抛出 ValueError: 用户数量不能超过 100,000
        """
        # 验证用户数量范围
        if self.user_count < 10:
            raise ValueError("用户数量不能少于 10")
        if self.user_count > 100_000:
            raise ValueError("用户数量不能超过 100,000")
        
        # 验证商品数量范围
        if self.product_count < 5:
            raise ValueError("产品数量不能少于 5")
        if self.product_count > 1_000:
            raise ValueError("产品数量不能超过 1,000")
        
        # 验证 APP 数量范围
        if self.app_count < 5:
            raise ValueError("应用数量不能少于 5")
        if self.app_count > 500:
            raise ValueError("应用数量不能超过 500")
        
        # 验证事件密度范围
        if self.avg_events_per_user < 1:
            raise ValueError("平均事件数不能少于 1")
        if self.avg_events_per_user > 1000:
            raise ValueError("平均事件数不能超过 1000")
        
        # 验证历史天数范围
        if self.history_days < 1:
            raise ValueError("历史天数不能少于 1")
        if self.history_days > 365:
            raise ValueError("历史天数不能超过 365")

        # 验证各边类型的数量范围（0-100，且 min <= max）
        def _check_range(name_min: int, name_max: int, label: str) -> None:
            if name_min < 0 or name_max < 0:
                raise ValueError(f"{label} 不能为负数")
            if name_min > name_max:
                raise ValueError(f"{label} 的最小值不能大于最大值")
            if name_max > 100:
                raise ValueError(f"{label} 的最大值不能超过 100（请谨慎设置）")

        _check_range(self.min_orders_per_user, self.max_orders_per_user, "每用户订购边数")
        _check_range(self.min_app_usages_per_user, self.max_app_usages_per_user, "每用户APP使用边数")
        _check_range(self.min_calls_per_user, self.max_calls_per_user, "每用户通话边数")
        _check_range(self.min_click_products_per_user, self.max_click_products_per_user, "每用户点击商品边数")
        _check_range(self.min_click_apps_per_user, self.max_click_apps_per_user, "每用户点击APP边数")


class DataConfigStore:
    """数据配置持久化存储。
    
    负责配置的磁盘 I/O 操作，采用 JSON 格式存储。
    提供配置加载和保存功能，支持配置的持久化管理。
    
    Attributes:
        storage_path (Path): 配置文件的存储路径
    """
    
    def __init__(self, storage_path: str | Path) -> None:
        """初始化配置存储。
        
        创建存储实例并确保父目录存在。
        如果指定的目录不存在，将自动创建。
        
        Args:
            storage_path (str | Path): 配置文件的完整路径
            
        Examples:
            >>> store = DataConfigStore("data/config.json")
            >>> # 自动创建 data 目录（如果不存在）
        """
        self.storage_path = Path(storage_path)
        # 确保父目录存在，parents=True 递归创建，exist_ok=True 避免已存在时报错
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
    
    def load_config(self) -> DataGenerationConfig:
        """从磁盘加载配置，不存在时返回默认配置。
        
        该方法尝试从指定路径读取配置文件：
        1. 如果文件不存在，返回默认配置
        2. 如果文件存在但解析失败（格式错误、编码问题），返回默认配置
        3. 如果文件存在且解析成功，返回解析后的配置对象
        
        Returns:
            DataGenerationConfig: 配置对象实例
            
        Note:
            该方法不会抛出异常，任何错误都将导致返回默认配置
        """
        # 配置文件不存在时，返回默认配置
        if not self.storage_path.exists():
            return DataGenerationConfig()
        
        try:
            # 读取 JSON 文件内容
            content = self.storage_path.read_text(encoding="utf-8")
            # 解析 JSON 为字典
            data = json.loads(content)
            # 从字典构建配置对象
            return DataGenerationConfig.from_dict(data)
        except (OSError, json.JSONDecodeError, ValueError):
            # 任何错误（文件读取失败、JSON 格式错误、类型转换失败）都返回默认配置
            return DataGenerationConfig()
    
    def save_config(self, config: DataGenerationConfig) -> None:
        """保存配置到磁盘。
        
        将配置对象序列化为 JSON 格式并写入文件。
        使用 UTF-8 编码和 2 空格缩进，确保可读性。
        
        Args:
            config (DataGenerationConfig): 要保存的配置对象
            
        Raises:
            OSError: 文件写入失败时（如磁盘已满、权限不足）
            
        Examples:
            >>> config = DataGenerationConfig(user_count=5000)
            >>> store.save_config(config)
            >>> # 配置已保存到文件
        """
        # 将配置对象转换为 JSON 字符串
        # ensure_ascii=False 允许中文字符，indent=2 格式化输出
        payload = json.dumps(config.to_dict(), ensure_ascii=False, indent=2)
        # 写入文件
        self.storage_path.write_text(payload, encoding="utf-8")


__all__ = [
    "DataGenerationConfig",
    "DataConfigStore",
]
