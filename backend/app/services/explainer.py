"""画像可解释性服务（简化实现）。"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(slots=True)
class ExplanationResult:
    user_id: str
    top_features: list[dict[str, Any]]


class ExplainerService:
    def __init__(self, *, profiling_service, background_user_ids: list[str] | None = None) -> None:
        self._profiling = profiling_service
        self._cache: dict[str, ExplanationResult] = {}

    def clear_cache(self) -> dict[str, str]:
        self._cache.clear()
        return {"status": "cleared"}

    async def explain(self, user_id: str) -> ExplanationResult | None:
        if user_id in self._cache:
            return self._cache[user_id]
        fusion = await self._profiling.fusion_score(user_id)
        if fusion is None:
            return None
        feats = [
            {"feature": f"rule:{r['name']}", "value": 1.0, "contribution": r["weight"]}
            for r in fusion.get("fired_rules", [])
        ] or [{"feature": "bias", "value": 1.0, "contribution": fusion.get("score", 0.0)}]
        res = ExplanationResult(user_id=user_id, top_features=feats[:5])
        self._cache[user_id] = res
        return res


__all__ = ["ExplainerService", "ExplanationResult"]
