"""训练模式配置管理服务。

提供 HGT 和编码器训练时的模式选择与采样参数配置，
支持完整图训练和 METIS 采样子图训练两种模式。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Literal


TrainingMode = Literal["full_graph", "metis_sampling"]


@dataclass(slots=True)
class TrainingConfig:
    """训练配置参数。
    
    该类封装了 HGT 和编码器训练的所有可调参数，包括：
    - 训练模式：完整图训练 vs METIS 采样训练
    - METIS 采样参数：分区数、不平衡因子、随机种子等
    
    Attributes:
        mode: 训练模式，"full_graph"（完整图）或 "metis_sampling"（METIS 采样）
        metis_num_parts: METIS 分区数，仅在 metis_sampling 模式下有效，范围 2-100
        metis_imbalance_factor: METIS 不平衡因子，范围 0.0-1.0
        metis_seed: METIS 随机种子，用于可重复的分区结果
        metis_recursive: 是否使用递归二分法
        metis_partition_id: 指定使用的分区 ID，None 表示随机选择
    """
    
    mode: TrainingMode = "full_graph"           # 默认使用完整图训练
    metis_num_parts: int = 4                    # 默认分为 4 个分区
    metis_imbalance_factor: float = 0.01        # 默认 1% 的不平衡容忍度
    metis_seed: int | None = None               # 默认不设置随机种子
    metis_recursive: bool = True                # 默认使用递归二分法
    metis_partition_id: int | None = None       # 默认随机选择分区
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式。
        
        用于 JSON 序列化和 API 响应。
        
        Returns:
            Dict[str, Any]: 配置参数的字典表示
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TrainingConfig:
        """从字典构建配置对象。
        
        支持从 JSON 或 API 请求体反序列化配置对象。
        缺失的字段将使用默认值。
        
        Args:
            data (Dict[str, Any]): 包含配置参数的字典
            
        Returns:
            TrainingConfig: 配置对象实例
        """
        mode = data.get("mode", "full_graph")
        if mode not in ("full_graph", "metis_sampling"):
            mode = "full_graph"
        
        return cls(
            mode=mode,  # type: ignore[arg-type]
            metis_num_parts=int(data.get("metis_num_parts", 4)),
            metis_imbalance_factor=float(data.get("metis_imbalance_factor", 0.01)),
            metis_seed=data.get("metis_seed"),
            metis_recursive=bool(data.get("metis_recursive", True)),
            metis_partition_id=data.get("metis_partition_id"),
        )
    
    def validate(self) -> None:
        """验证配置参数的有效性。
        
        检查所有参数是否在允许的范围内，超出范围时抛出异常。
        该方法在配置更新前应被调用，防止无效配置导致系统异常。
        
        Raises:
            ValueError: 当任何参数超出允许范围时
        """
        # 验证训练模式
        if self.mode not in ("full_graph", "metis_sampling"):
            raise ValueError(f"训练模式必须是 'full_graph' 或 'metis_sampling'，当前值: {self.mode}")
        
        # 仅在 METIS 模式下验证采样参数
        if self.mode == "metis_sampling":
            # 验证分区数
            if self.metis_num_parts < 2:
                raise ValueError("METIS 分区数不能少于 2")
            if self.metis_num_parts > 100:
                raise ValueError("METIS 分区数不能超过 100")
            
            # 验证不平衡因子
            if not 0.0 <= self.metis_imbalance_factor <= 1.0:
                raise ValueError("METIS 不平衡因子必须在 [0.0, 1.0] 区间内")
            
            # 验证分区 ID（如果指定）
            if self.metis_partition_id is not None:
                if not 0 <= self.metis_partition_id < self.metis_num_parts:
                    raise ValueError(
                        f"METIS 分区 ID 必须在 [0, {self.metis_num_parts}) 区间内，"
                        f"当前值: {self.metis_partition_id}"
                    )


class TrainingConfigStore:
    """训练配置持久化存储。
    
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
        """
        self.storage_path = Path(storage_path)
        # 确保父目录存在，parents=True 递归创建，exist_ok=True 避免已存在时报错
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
    
    def load_config(self) -> TrainingConfig:
        """从磁盘加载配置，不存在时返回默认配置。
        
        该方法尝试从指定路径读取配置文件：
        1. 如果文件不存在，返回默认配置
        2. 如果文件存在但解析失败（格式错误、编码问题），返回默认配置
        3. 如果文件存在且解析成功，返回解析后的配置对象
        
        Returns:
            TrainingConfig: 配置对象实例
            
        Note:
            该方法不会抛出异常，任何错误都将导致返回默认配置
        """
        # 配置文件不存在时，返回默认配置
        if not self.storage_path.exists():
            return TrainingConfig()
        
        try:
            # 读取 JSON 文件内容
            content = self.storage_path.read_text(encoding="utf-8")
            # 解析 JSON 为字典
            data = json.loads(content)
            # 从字典构建配置对象
            return TrainingConfig.from_dict(data)
        except (OSError, json.JSONDecodeError, ValueError):
            # 任何错误（文件读取失败、JSON 格式错误、类型转换失败）都返回默认配置
            return TrainingConfig()
    
    def save_config(self, config: TrainingConfig) -> None:
        """保存配置到磁盘。
        
        将配置对象序列化为 JSON 格式并写入文件。
        使用 UTF-8 编码和 2 空格缩进，确保可读性。
        
        Args:
            config (TrainingConfig): 要保存的配置对象
            
        Raises:
            OSError: 文件写入失败时（如磁盘已满、权限不足）
        """
        # 将配置对象转换为 JSON 字符串
        # ensure_ascii=False 允许中文字符，indent=2 格式化输出
        payload = json.dumps(config.to_dict(), ensure_ascii=False, indent=2)
        # 写入文件
        self.storage_path.write_text(payload, encoding="utf-8")


__all__ = [
    "TrainingMode",
    "TrainingConfig",
    "TrainingConfigStore",
]
