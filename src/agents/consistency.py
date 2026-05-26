"""ConsistencyCheck —— 对多 temperature 的诊断候选做投票 + 置信度评估。

规则：
1. 严重度多数投票（3 选 1）。
2. 置信度 = 多数票数 / 总票数。
3. 若 3 个候选严重度跨度 ≥ 2 级（如 mild 与 severe 同时出现）→ requires_human_review=True。
4. 最终 reasoning_chain 取多数票中 temperature 最低（最确定）的那次。
5. diagnosis_label 同样按多数投票（按归一化标签）。
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from loguru import logger

from src.config import settings
from src.state import GraphState

AGENT = "consistency"

_SEV_ORDER = {"none": 0, "mild": 1, "moderate": 2, "severe": 3}


def _norm_label(s: str) -> str:
    return (s or "").strip().replace(" ", "")


def run(state: GraphState) -> GraphState:
    candidates = list(state.get("diagnosis_candidates") or [])
    new_state = dict(state)

    if not candidates:
        msg = "diagnosis_candidates 为空"
        logger.error(f"[consistency] {msg}")
        new_state.setdefault("errors", {})[AGENT] = msg
        return new_state  # type: ignore[return-value]

    if settings.ablation.disable_consistency:
        c = candidates[0]
        new_state["final_diagnosis"] = {
            "diagnosis_label": c["diagnosis_label"],
            "severity_grade": c["severity_grade"],
            "confidence": 1.0,
            "reasoning_chain": c["reasoning_chain"],
            "vote_distribution": {c["severity_grade"]: 1},
        }
        new_state["requires_human_review"] = False
        logger.info("[consistency] 已禁用，直接取第一个候选")
        return new_state  # type: ignore[return-value]

    # 严重度投票
    sev_votes = Counter(c["severity_grade"] for c in candidates)
    sev_majority, sev_count = sev_votes.most_common(1)[0]
    sev_indices = [_SEV_ORDER[c["severity_grade"]] for c in candidates if c["severity_grade"] in _SEV_ORDER]
    sev_spread = max(sev_indices) - min(sev_indices) if sev_indices else 0
    requires_review = sev_spread >= 2

    # label 投票
    label_votes = Counter(_norm_label(c["diagnosis_label"]) for c in candidates)
    label_majority, _ = label_votes.most_common(1)[0]

    # 选 reasoning：在严重度 = 多数票的候选里，挑 temperature 最低的
    majority_candidates = [c for c in candidates if c["severity_grade"] == sev_majority]
    chosen = min(majority_candidates, key=lambda c: c["temperature"])

    confidence = sev_count / len(candidates)

    final: dict[str, Any] = {
        "diagnosis_label": chosen["diagnosis_label"] if _norm_label(chosen["diagnosis_label"]) == label_majority else label_majority,
        "severity_grade": sev_majority,
        "confidence": round(confidence, 3),
        "reasoning_chain": chosen["reasoning_chain"],
        "vote_distribution": dict(sev_votes),
    }
    new_state["final_diagnosis"] = final
    new_state["requires_human_review"] = bool(requires_review)
    logger.info(
        f"[consistency] severity={sev_majority} confidence={confidence:.2f} "
        f"review={requires_review} spread={sev_spread}"
    )
    return new_state  # type: ignore[return-value]
