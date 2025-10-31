"""可解释性服务：基于 SHAP 对混合画像得分提供解释。

模块职责链条：

- 依赖 :class:`~app.services.hybrid_profiling_service.HybridProfilingService`
    的 :meth:`profile_user` 获取融合后的规则/向量特征；
- 将解释结果回传给 FastAPI 路由 (:mod:`app.api.profiling`)，
    最终在 Streamlit 页面 ``01_User_Profile_Query.py`` 中可视化；
- 通过 :meth:`ExplainerService.clear_cache` 在规则更新后刷新 SHAP 基线。

特性概览：

1. 使用 ``KernelExplainer`` 以模型无关方式估计特征贡献；
2. 可选择背景用户作为基线，提高解释稳定性；
3. 支持缓存 SHAP 对象，避免重复初始化开销。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

try:  # pragma: no cover - SHAP 可能未安装
    import shap
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError("ExplainerService 依赖 shap，请先安装该库后再使用。") from exc

from .hybrid_profiling_service import HybridProfilingService


@dataclass(slots=True)
class ExplanationResult:
    """解释结果数据结构。"""

    user_id: str
    final_score: float
    shap_values: Dict[str, float]
    top_positive: List[tuple[str, float]]
    top_negative: List[tuple[str, float]]
    gate: float
    f_rule: float
    f_nn: float
    rule_details: Dict[str, float]


class ExplainerService:
    """封装 SHAP 解释流程，面向混合画像服务。"""

    def __init__(
        self,
        *,
        profiling_service: HybridProfilingService,
        background_user_ids: Sequence[str] | None = None,
        background_size: int = 20,
    ) -> None:
        """初始化解释器。

        :param profiling_service: 混合画像服务实例，用于拿到 ``h_rule`` / ``h_nn`` 等特征；
        :param background_user_ids: 作为 SHAP baseline 的历史用户列表；
        :param background_size: baseline 中随机选取的最大用户数。
        """
        self.profiling_service = profiling_service
        self.background_user_ids = tuple(background_user_ids or ())
        self.background_size = background_size
        self.rule_engine = profiling_service.rule_engine
        self.device = profiling_service.device

        self._baseline: torch.Tensor | None = None
        self._explainer: shap.KernelExplainer | None = None

    async def explain(self, user_id: str, *, top_k: int = 5) -> Optional[ExplanationResult]:
        """解释指定用户的画像得分，返回详细贡献信息。"""
        profile = await self.profiling_service.profile_user(user_id)
        if profile is None:
            return None

        # 将规则向量、嵌入向量与规则得分拼接成 SHAP 输入
        h_rule = torch.tensor(profile["h_rule"], dtype=torch.float32)
        h_nn = torch.tensor(profile["h_nn"], dtype=torch.float32)
        f_rule = torch.tensor([profile["f_rule"]], dtype=torch.float32)

        feature_vector = torch.cat([h_rule, h_nn, f_rule])
        feature_names = self._describe_features(h_rule.size(0), h_nn.size(0))
        baseline = await self._get_or_build_baseline()
        explainer = await self._get_or_build_explainer(baseline)

        # KernelExplainer 计算特征贡献，支持 SHAP 可视化
        shap_values = explainer.shap_values(feature_vector.numpy(), nsamples="auto")
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_dict = {name: float(value) for name, value in zip(feature_names, shap_values)}
        top_positive = sorted(shap_dict.items(), key=lambda item: item[1], reverse=True)[:top_k]
        top_negative = sorted(shap_dict.items(), key=lambda item: item[1])[:top_k]

        return ExplanationResult(
            user_id=user_id,
            final_score=float(profile["final_score"]),
            shap_values=shap_dict,
            top_positive=top_positive,
            top_negative=top_negative,
            gate=float(profile["fusion_weight"]),
            f_rule=float(profile["f_rule"]),
            f_nn=float(profile["f_nn"]),
            rule_details=profile["rule_details"],
        )

    async def _get_or_build_baseline(self) -> torch.Tensor:
        """构造 SHAP baseline 矩阵，必要时从缓存中读取。"""
        if self._baseline is not None:
            return self._baseline

        feature_dim = self.rule_engine.feature_dim + self.profiling_service.fusion_core.embedding_dim + 1

        if not self.background_user_ids:
            self._baseline = torch.zeros((1, feature_dim), dtype=torch.float32)
            return self._baseline

        features: List[torch.Tensor] = []
        for user in self.background_user_ids[: self.background_size]:
            # 将历史用户的画像拼成背景集，提高解释稳定性
            profile = await self.profiling_service.profile_user(user)
            if profile is None:
                continue
            h_rule = torch.tensor(profile["h_rule"], dtype=torch.float32)
            h_nn = torch.tensor(profile["h_nn"], dtype=torch.float32)
            f_rule = torch.tensor([profile["f_rule"]], dtype=torch.float32)
            features.append(torch.cat([h_rule, h_nn, f_rule]))

        if not features:
            self._baseline = torch.zeros((1, feature_dim), dtype=torch.float32)
        else:
            self._baseline = torch.stack(features)
        return self._baseline

    async def _get_or_build_explainer(self, baseline: torch.Tensor) -> shap.KernelExplainer:
        """惰性创建 ``KernelExplainer`` 并缓存。"""
        if self._explainer is not None:
            return self._explainer

        def predict(batch: np.ndarray) -> np.ndarray:
            tensors = torch.from_numpy(batch.astype(np.float32))
            return self._predict_from_features(tensors)

        self._explainer = shap.KernelExplainer(predict, baseline.numpy())
        return self._explainer
    
    def clear_cache(self) -> None:
        """在规则或模型变更后重置 SHAP 基线与解释器。"""
        
        self._baseline = None
        self._explainer = None

    def _predict_from_features(self, features: torch.Tensor) -> np.ndarray:
        """调用融合核心获得最终分数，供 SHAP 内部使用。"""
        if features.dim() == 1:
            features = features.unsqueeze(0)
        features = features.to(self.device)
        rule_dim = self.rule_engine.feature_dim
        h_rule = features[:, :rule_dim]
        h_nn = features[:, rule_dim:-1]
        f_rules = features[:, -1].unsqueeze(-1)

        scores: List[float] = []
        for h_r, h_n, f_rule in zip(h_rule, h_nn, f_rules):
            with torch.no_grad():
                # 直接复用融合核心，保持解释与线上推理一致
                score_tensor, *_ = self.profiling_service.fusion_core(h_r, h_n, f_rule)
            scores.append(float(score_tensor.squeeze().item()))
        return np.array(scores, dtype=np.float32)

    def _describe_features(self, rule_dim: int, embedding_dim: int) -> List[str]:
        """生成特征名称列表，匹配 ``h_rule`` 与 ``h_nn`` 的维度。"""
        rule_names = self.rule_engine.rule_names
        if len(rule_names) < rule_dim:
            rule_names = [*rule_names, *[f"rule_{idx}" for idx in range(len(rule_names), rule_dim)]]
        embedding_names = [f"embedding_{idx}" for idx in range(embedding_dim)]
        return [*rule_names[:rule_dim], *embedding_names, "f_rule"]


__all__ = [
    "ExplainerService",
    "ExplanationResult",
]
