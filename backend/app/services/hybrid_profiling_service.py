"""混合画像服务：结合规则引擎与图神经网络嵌入的画像打分。

模块间关系：

- 与 :class:`~app.services.data_ingestion.DataIngestionService` 协作，读取用户画像与行为统计；
- 从 :class:`~app.services.incremental_learner.IncrementalLearner` 获取 GNN 嵌入；
- 对接 FastAPI 路由 (:mod:`app.api.profiling`) 暴露查询/规则 CRUD；
- 由 :class:`~app.services.explainer.ExplainerService` 调用以生成 SHAP 解释。

主要组成：规则引擎 (:class:`RuleEngine`)、融合核心 (:class:`FusionCore`)、
混合服务主体 (:class:`HybridProfilingService`) 以及规则存储 (:class:`RuleStore`)。
"""
from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

import torch
from torch import nn
from torch.nn import functional as F

from .data_ingestion import DataIngestionService
from .incremental_learner import IncrementalLearner


@dataclass(slots=True)
class RuleEngineResult:
    """规则引擎输出结果。"""

    score: float
    feature_vector: torch.Tensor
    details: Dict[str, float]


@dataclass(slots=True)
class RuleDefinition:
    name: str
    weight: float
    description: str
    condition: str
    predicate: Callable[[Mapping[str, Any]], bool]


class RuleEngine:
    """支持动态增删改的业务规则引擎。"""

    def __init__(self, *, rules: Optional[Iterable[Mapping[str, Any]]] = None) -> None:
        """初始化规则引擎，必要时加载已有规则集合。"""
        self._rules: List[RuleDefinition] = []
        if rules:
            for rule in rules:
                self.add_rule(
                    name=rule["name"],
                    weight=rule.get("weight", 1.0),
                    description=rule.get("description", ""),
                    condition=rule["condition"],
                )
        else:
            self._seed_default_rules()

    @property
    def feature_dim(self) -> int:
        """返回规则向量的维度，即当前规则数量。"""
        return len(self._rules)

    @property
    def rule_names(self) -> list[str]:
        """返回规则名称列表，供解释器或前端使用。"""
        return [rule.name for rule in self._rules]

    def run(self, user_raw_features: Mapping[str, Any]) -> RuleEngineResult:
        """执行所有规则并返回得分、特征向量与详情。"""
        feature_vector = torch.zeros(self.feature_dim, dtype=torch.float32)
        score = 0.0
        details: Dict[str, float] = {}

        for idx, rule in enumerate(self._rules):
            # 逐条规则执行 predicate，统计命中情况与得分贡献
            triggered = rule.predicate(user_raw_features)
            contribution = rule.weight if triggered else 0.0
            feature_vector[idx] = 1.0 if triggered else 0.0
            score += contribution
            details[rule.name] = float(contribution)

        return RuleEngineResult(score=score, feature_vector=feature_vector, details=details)

    # --- 动态管理 API -----------------------------------------------------------------
    def list_rules(self) -> list[dict[str, Any]]:
        """以字典列表形式返回规则元数据，便于序列化。"""
        return [
            {
                "name": rule.name,
                "description": rule.description,
                "weight": rule.weight,
                "condition": rule.condition,
            }
            for rule in self._rules
        ]

    def add_rule(self, *, name: str, weight: float, description: str, condition: str) -> None:
        """新增规则并编译条件表达式。"""
        if any(rule.name == name for rule in self._rules):
            raise ValueError(f"规则 {name} 已存在")
        predicate = self._compile_condition(condition)
        self._rules.append(
            RuleDefinition(
                name=name,
                weight=float(weight),
                description=description,
                condition=condition,
                predicate=predicate,
            )
        )

    def update_rule(
        self,
        name: str,
        *,
        weight: Optional[float] = None,
        description: Optional[str] = None,
        condition: Optional[str] = None,
    ) -> None:
        """更新规则属性或条件，若条件变化则重新编译。"""
        for idx, rule in enumerate(self._rules):
            if rule.name == name:
                new_weight = float(weight) if weight is not None else rule.weight
                new_description = description if description is not None else rule.description
                new_condition = condition if condition is not None else rule.condition
                predicate = rule.predicate if condition is None else self._compile_condition(new_condition)
                self._rules[idx] = RuleDefinition(
                    name=name,
                    weight=new_weight,
                    description=new_description,
                    condition=new_condition,
                    predicate=predicate,
                )
                return
        raise ValueError(f"未找到规则 {name}")

    def delete_rule(self, name: str) -> None:
        """删除指定名称的规则，不存在时抛出错误。"""
        for idx, rule in enumerate(self._rules):
            if rule.name == name:
                del self._rules[idx]
                return
        raise ValueError(f"未找到规则 {name}")

    # --- 内部工具 -------------------------------------------------------------------
    def _seed_default_rules(self) -> None:
        """在无外部规则时注入默认规则集。"""
        defaults = [
            {
                "name": "high_value_user",
                "description": "月话费高且入网时长长",
                "weight": 1.0,
                "condition": "monthly_fee > 300 and tenure_months >= 24",
            },
            {
                "name": "loyal_user",
                "description": "长期使用5G或家庭套餐的忠实用户",
                "weight": 0.8,
                "condition": "tenure_months >= 36 and ( '5G' in str(plan_type) or '家庭' in str(plan_type) )",
            },
            {
                "name": "digital_enthusiast",
                "description": "线上活跃度高",
                "weight": 0.6,
                "condition": "(event_counts.get('APP使用', 0) or 0) + (event_counts.get('点击', 0) or 0) >= 10",
            },
            {
                "name": "at_risk_churn",
                "description": "可能流失的低活跃用户",
                "weight": -0.7,
                "condition": "total_events <= 2 and tenure_months >= 12 and monthly_fee < 80",
            },
        ]
        for rule in defaults:
            try:
                self.add_rule(**rule)
            except ValueError:
                continue

    @staticmethod
    def _compile_condition(condition: str) -> Callable[[Mapping[str, Any]], bool]:
        """将规则条件表达式编译成可执行谓词，并进行语法白名单校验。"""
        import ast

        allowed_nodes = (
            ast.Expression,
            ast.BoolOp,
            ast.Compare,
            ast.Name,
            ast.Load,
            ast.BinOp,
            ast.UnaryOp,
            ast.And,
            ast.Or,
            ast.Mod,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Pow,
            ast.USub,
            ast.Constant,
            ast.Call,
            ast.keyword,
            ast.Attribute,
            ast.List,
            ast.Tuple,
            ast.Subscript,
            ast.Index,
            ast.Gt,
            ast.GtE,
            ast.Lt,
            ast.LtE,
            ast.Eq,
            ast.NotEq,
            ast.Str,
        )

        tree = ast.parse(condition, mode="eval")
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                raise ValueError(f"规则条件含有不支持的语法: {node.__class__.__name__}")
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr != "get":
                        raise ValueError("仅允许调用 dict.get")
                elif isinstance(node.func, ast.Name):
                    if node.func.id not in {"int", "float", "len", "max", "min"}:
                        raise ValueError("仅允许调用部分安全函数: int/float/len/max/min")
                else:
                    raise ValueError("不支持的函数调用")

        code = compile(tree, "<rule>", "eval")

        def predicate(features: Mapping[str, Any]) -> bool:
            local_vars = dict(features)
            # 允许 event_counts 等嵌套字典直接访问
            for key, value in list(features.items()):
                if isinstance(value, Mapping):
                    local_vars[key] = value
            try:
                return bool(eval(code, {"__builtins__": {}}, local_vars))
            except Exception:
                return False

        return predicate


class FusionCore(nn.Module):
    """规则向量与 GNN 嵌入的融合模块，带动态门控。"""

    def __init__(self, *, rule_dim: int, embedding_dim: int, hidden_dim: int = 64) -> None:
        """初始化融合核心并构建门控与评分网络。"""
        super().__init__()
        self.rule_dim = rule_dim
        self.embedding_dim = embedding_dim
        self.gate_linear = nn.Linear(rule_dim + embedding_dim, 1)
        self.nn_score = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        h_rule: torch.Tensor,
        h_nn: torch.Tensor,
        f_rule: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """融合规则得分与神经网络嵌入，返回最终分数/门控/NN 分数。"""
        if h_rule.dim() == 1:
            h_rule = h_rule.unsqueeze(0)
        if h_nn.dim() == 1:
            h_nn = h_nn.unsqueeze(0)
        if f_rule.dim() == 0:
            f_rule = f_rule.view(1, 1)
        elif f_rule.dim() == 1:
            f_rule = f_rule.unsqueeze(-1)

        fusion_input = torch.cat([h_rule, h_nn], dim=-1)
        gate = torch.sigmoid(self.gate_linear(fusion_input))
        f_nn = self.nn_score(h_nn)
        score = gate * f_rule + (1 - gate) * f_nn
        return score, gate, f_nn


class HybridProfilingService:
    """混合画像主服务，组合规则与 GNN 嵌入得出最终画像分数。"""

    def __init__(
        self,
        *,
        data_ingestion: DataIngestionService,
        incremental_learner: IncrementalLearner | None = None,
        rule_engine: RuleEngine | None = None,
        fusion_core: FusionCore | None = None,
        device: str | torch.device = "cpu",
        rule_store: "RuleStore" | None = None,
    ) -> None:
        """构造混合画像服务，并根据依赖初始化融合核心。"""
        self.data_ingestion = data_ingestion
        self.incremental_learner = incremental_learner
        self.rule_engine = rule_engine or RuleEngine()
        self.rule_store = rule_store
        self.device = torch.device(device)
        self._rule_refresh_task: asyncio.Task[None] | None = None

        if incremental_learner is not None:
            embedding_dim = incremental_learner.model.output_dims.get("user", 128)
        else:
            embedding_dim = 128
        self.fusion_core = fusion_core or FusionCore(
            rule_dim=self.rule_engine.feature_dim,
            embedding_dim=embedding_dim,
        )
        self.fusion_core.to(self.device)

    async def train_fusion_core(
        self,
        *,
        sample_size: int = 256,
        epochs: int = 3,
        lr: float = 1e-3,
        batch_size: int = 64,
        progress_cb: Callable[[Dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """使用缓存用户构造数据集，训练融合核心参数。"""

        if self.incremental_learner is None:
            return {"trained": False, "reason": "incremental learner unavailable"}

        await self.data_ingestion.ensure_cache_warm()
        user_ids = await self.data_ingestion.list_cached_user_ids()
        if not user_ids:
            return {"trained": False, "reason": "no cached users"}

        random.shuffle(user_ids)
        selected = user_ids[:sample_size]

        samples: list[dict[str, torch.Tensor]] = []
        for user_id in selected:
            profile = await self.data_ingestion.get_user_profile(user_id)
            if profile is None:
                continue

            raw_features = self._to_raw_features(profile)
            rule_result = self.rule_engine.run(raw_features)
            embedding = self.incremental_learner.get_latest_embedding("user", user_id)
            if embedding is None:
                continue

            h_rule = rule_result.feature_vector.to(self.device)
            h_nn = embedding.to(self.device)
            f_rule = torch.tensor([rule_result.score], dtype=torch.float32, device=self.device)
            target_score = (
                float(rule_result.score)
                + float(torch.linalg.norm(embedding).item()) * 0.1
            )
            target = torch.tensor([target_score], dtype=torch.float32, device=self.device)

            samples.append({
                "h_rule": h_rule,
                "h_nn": h_nn,
                "f_rule": f_rule,
                "target": target,
            })

        if not samples:
            return {"trained": False, "reason": "insufficient samples"}

        optimizer = torch.optim.Adam(self.fusion_core.parameters(), lr=lr)
        self.fusion_core.train()

        actual_epochs = max(1, epochs)
        if progress_cb is not None:
            progress_cb(
                {
                    "event": "start",
                    "total_epochs": actual_epochs,
                    "samples": len(samples),
                    "learning_rate": lr,
                }
            )

        last_loss: float = 0.0
        for epoch_idx in range(actual_epochs):
            epoch_start = time.perf_counter()
            random.shuffle(samples)
            epoch_losses: list[float] = []
            for idx in range(0, len(samples), max(1, batch_size)):
                batch = samples[idx : idx + max(1, batch_size)]
                h_rule_batch = torch.stack([item["h_rule"] for item in batch])
                h_nn_batch = torch.stack([item["h_nn"] for item in batch])
                f_rule_batch = torch.cat([item["f_rule"] for item in batch], dim=0).unsqueeze(-1)
                target_batch = torch.cat([item["target"] for item in batch], dim=0)

                optimizer.zero_grad(set_to_none=True)
                score_pred, *_ = self.fusion_core(h_rule_batch, h_nn_batch, f_rule_batch)
                loss = F.mse_loss(score_pred, target_batch)
                loss.backward()
                optimizer.step()
                last_loss = float(loss.item())
                epoch_losses.append(last_loss)

            if progress_cb is not None and epoch_losses:
                progress_cb(
                    {
                        "event": "epoch",
                        "epoch": epoch_idx + 1,
                        "total_epochs": actual_epochs,
                        "loss": sum(epoch_losses) / len(epoch_losses),
                        "duration_seconds": time.perf_counter() - epoch_start,
                        "samples": len(samples),
                        "learning_rate": lr,
                    }
                )

        self.fusion_core.eval()
        if progress_cb is not None:
            progress_cb(
                {
                    "event": "complete",
                    "epochs": actual_epochs,
                    "final_loss": last_loss,
                }
            )
        return {
            "trained": True,
            "samples": len(samples),
            "epochs": max(1, epochs),
            "final_loss": last_loss,
        }

    async def profile_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """综合规则得分与 GNN 嵌入生成指定用户的画像。"""
        profile = await self.data_ingestion.get_user_profile(user_id)
        if profile is None:
            return None

        # 预处理原始画像，供规则与模型共同使用
        raw_features = self._to_raw_features(profile)
        rule_result = self.rule_engine.run(raw_features)

        embedding: Optional[torch.Tensor] = None
        if self.incremental_learner is not None:
            embedding = self.incremental_learner.get_latest_embedding("user", user_id)
        if embedding is None:
            # 若尚未有嵌入，使用零向量占位
            embedding = torch.zeros(self.fusion_core.embedding_dim, dtype=torch.float32)
        h_nn = embedding.to(self.device)

        h_rule = rule_result.feature_vector.to(self.device)
        f_rule_tensor = torch.tensor([rule_result.score], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            # 动态门控融合规则与向量得分
            score_tensor, gate_tensor, f_nn_tensor = self.fusion_core(h_rule, h_nn, f_rule_tensor)

        score = float(score_tensor.squeeze().item())
        gate = float(gate_tensor.squeeze().item())
        f_nn = float(f_nn_tensor.squeeze().item())

        return {
            "user_id": user_id,
            "final_score": score,
            "fusion_weight": gate,
            "f_rule": float(rule_result.score),
            "f_nn": f_nn,
            "rule_details": rule_result.details,
            "h_rule": rule_result.feature_vector.tolist(),
            "h_nn": embedding.tolist(),
            "raw_features": raw_features,
        }

    @staticmethod
    def _to_raw_features(profile: Mapping[str, Any]) -> Dict[str, Any]:
        """将摄取服务返回的数据转换为规则可读的扁平结构。"""
        user_info = profile.get("user", {}) or {}
        event_counts = profile.get("event_counts", {}) or {}
        total_events = int(sum((event_counts or {}).values()))
        # 将摄取服务返回的嵌套结构整理成平铺字典，便于规则表达式访问
        return {
            "user_id": user_info.get("user_id"),
            "plan_type": user_info.get("plan_type"),
            "monthly_fee": user_info.get("monthly_fee", 0.0) or 0.0,
            "user_level": user_info.get("user_level"),
            "tenure_months": user_info.get("tenure_months", 0) or 0,
            "device_brand": user_info.get("device_brand"),
            "event_counts": event_counts,
            "total_events": total_events,
            "last_event_at": profile.get("last_event_at"),
        }

    def refresh_rule_structure(self) -> None:
        """当规则集合发生变化时，重建融合核心以匹配新的维度。"""

        rule_dim = self.rule_engine.feature_dim
        if rule_dim == self.fusion_core.rule_dim:
            return
        # 规则数量变化后，需要重新初始化融合层以匹配输入维度
        new_core = FusionCore(
            rule_dim=rule_dim,
            embedding_dim=self.fusion_core.embedding_dim,
        )
        new_core.to(self.device)
        self.fusion_core = new_core

        if self.rule_store is not None:
            self.rule_store.save(self.rule_engine.list_rules())

        if self.incremental_learner is not None:
            # 规则结构变化也可能影响 SHAP 结果，触发嵌入刷新
            self._schedule_embedding_refresh()

    def _schedule_embedding_refresh(self) -> None:
        """在规则结构变化时调度一次嵌入刷新任务。"""
        if self.incremental_learner is None:
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # 在无事件循环上下文中直接同步执行
            self.incremental_learner.refresh_all_embeddings()
            return

        if self._rule_refresh_task is not None and not self._rule_refresh_task.done():
            self._rule_refresh_task.cancel()

        async def _refresh_task() -> None:
            # 将刷新计算放到线程池，避免阻塞事件循环
            await asyncio.sleep(0)
            await asyncio.to_thread(self.incremental_learner.refresh_all_embeddings)

        self._rule_refresh_task = loop.create_task(_refresh_task())


class RuleStore:
    """负责画像规则的持久化管理。"""

    def __init__(self, *, storage_path: str | Path) -> None:
        """初始化规则存储，确保目录存在。"""
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[dict[str, Any]]:
        """从磁盘读取规则列表，解析失败时返回空列表。"""
        if not self.storage_path.exists():
            return []
        try:
            # 从磁盘读取 JSON 并解析到结构化规则列表
            content = self.storage_path.read_text(encoding="utf-8")
            data = json.loads(content)
            if isinstance(data, list):
                return data
        except (OSError, json.JSONDecodeError):
            return []
        return []

    def save(self, rules: list[dict[str, Any]]) -> None:
        """将规则集合保存为 JSON 文件。"""
        # 使用 UTF-8 编码持久化规则，确保中文信息不丢失
        payload = json.dumps(rules, ensure_ascii=False, indent=2)
        self.storage_path.write_text(payload, encoding="utf-8")


__all__ = [
    "RuleEngine",
    "RuleStore",
    "FusionCore",
    "HybridProfilingService",
    "RuleEngineResult",
]
