"""规则融合画像服务（简化实现）。"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Rule:
    name: str
    description: str
    weight: float
    condition: str  # Python 表达式，变量命名以 profile 字段为准


class RuleStore:
    def __init__(self, storage_path: Path) -> None:
        self.path = storage_path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[dict[str, Any]] | None:
        if not self.path.exists():
            return None
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def save(self, rules: list[dict[str, Any]]) -> None:
        self.path.write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")


class RuleEngine:
    def __init__(self, rules: list[dict[str, Any]] | None = None) -> None:
        self._rules: dict[str, Rule] = {}
        if rules:
            for r in rules:
                self._rules[r["name"]] = Rule(
                    name=r["name"],
                    description=r.get("description", ""),
                    weight=float(r.get("weight", 1.0)),
                    condition=str(r.get("condition", "True")),
                )
        if not self._rules:
            # 默认 2 条演示规则
            self._rules = {
                "高价值套餐": Rule("高价值套餐", "月费较高的用户", 1.5, "monthly_fee >= 129"),
                "资深用户": Rule("资深用户", "在网时长 24+ 个月", 1.2, "tenure_months >= 24"),
            }

    def list_rules(self) -> list[dict[str, Any]]:
        return [asdict(r) for r in self._rules.values()]

    def add_rule(self, *, name: str, description: str, weight: float, condition: str) -> None:
        if name in self._rules:
            raise ValueError("规则已存在")
        self._rules[name] = Rule(name, description, float(weight), condition)

    def update_rule(self, name: str, *, description: str | None, weight: float | None, condition: str | None) -> None:
        if name not in self._rules:
            raise ValueError("规则不存在")
        rule = self._rules[name]
        if description is not None:
            rule.description = description
        if weight is not None:
            rule.weight = float(weight)
        if condition is not None:
            rule.condition = condition

    def delete_rule(self, name: str) -> None:
        if name not in self._rules:
            raise ValueError("规则不存在")
        del self._rules[name]

    def score(self, profile: dict[str, Any]) -> tuple[float, list[tuple[str, float]]]:
        score = 0.0
        fired: list[tuple[str, float]] = []
        safe_locals = dict(profile)
        for rule in self._rules.values():
            try:
                ok = bool(eval(rule.condition, {"__builtins__": {}}, safe_locals))
            except Exception:
                ok = False
            if ok:
                score += rule.weight
                fired.append((rule.name, rule.weight))
        return score, fired


class HybridProfilingService:
    def __init__(self, *, data_ingestion, incremental_learner, rule_engine: RuleEngine, rule_store: RuleStore, device) -> None:
        self.data_ingestion = data_ingestion
        self.incremental_learner = incremental_learner
        self.rule_engine = rule_engine
        self.rule_store = rule_store
        self.device = device

    def refresh_rule_structure(self) -> None:
        # 规则更新后写回磁盘
        self.rule_store.save(self.rule_engine.list_rules())

    async def fusion_score(self, user_id: str) -> dict[str, Any] | None:
        profile = await self.data_ingestion.get_user_profile(user_id)
        if profile is None:
            return None
        score, fired = self.rule_engine.score(profile)
        return {
            "user_id": user_id,
            "score": score,
            "fired_rules": [{"name": n, "weight": w} for n, w in fired],
        }


__all__ = ["RuleStore", "RuleEngine", "HybridProfilingService", "Rule"]
