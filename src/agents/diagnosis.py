"""DiagnosisAgent —— 用 3 个 temperature 各跑一次 CoT 推理。

输出存进 state["diagnosis_candidates"]（list[DiagnosisCandidate]），
ConsistencyCheck 节点再做投票 / 合并。
"""
from __future__ import annotations

import json
from typing import Any

from loguru import logger

from src.config import settings
from src.llm import chat, parse_json_loose, prompt
from src.state import GraphState

AGENT = "diagnosis"

_SEVERITIES = ("none", "mild", "moderate", "severe")


def _build_user_msg(state: GraphState) -> str:
    payload: dict[str, Any] = {
        "intake_structured": (state.get("intake") or {}).get("structured", {}),
        "intake_raw": {
            "chief_complaint": (state.get("intake") or {}).get("chief_complaint"),
            "cvs_q_total": (state.get("intake") or {}).get("cvs_q_total"),
            "osdi_total": (state.get("intake") or {}).get("osdi_total"),
            "demographics": (state.get("intake") or {}).get("demographics", {}),
        },
        "behavior_metrics": {
            "raw": state.get("vision_metrics", {}),
            "normalized": (state.get("aggregated") or {}).get("normalized", {}),
            "anomalies": (state.get("aggregated") or {}).get("anomalies", []),
            "composite_score": (state.get("aggregated") or {}).get("composite_score"),
        },
        "guideline_hits": (state.get("retrieval") or {}).get("guideline", []),
        "similar_cases": (state.get("retrieval") or {}).get("case", []),
    }
    return (
        "请基于以下 JSON 输入进行诊断推理：\n"
        f"```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```\n"
        "严格按系统指令输出 JSON。"
    )


def _normalize_severity(s: Any) -> str:
    if isinstance(s, str) and s in _SEVERITIES:
        return s
    s_low = (str(s) or "").lower().strip()
    mapping = {
        "无": "none", "0": "none",
        "轻": "mild", "轻度": "mild", "1": "mild",
        "中": "moderate", "中度": "moderate", "2": "moderate",
        "重": "severe", "重度": "severe", "严重": "severe", "3": "severe",
    }
    return mapping.get(s_low, "moderate")


def run(state: GraphState) -> GraphState:
    user_msg = _build_user_msg(state)
    system = prompt("diagnosis_system")

    temps = settings.diagnosis_temperatures
    if settings.ablation.disable_consistency:
        temps = temps[:1]
        logger.info(f"[diagnosis] consistency 已禁用，仅跑 1 次 T={temps[0]}")

    candidates = []
    for t in temps:
        try:
            text = chat(
                agent=AGENT,
                system=system,
                user=user_msg,
                state=state,
                temperature=float(t),
            )
            raw = parse_json_loose(text)
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"[diagnosis] T={t} 失败")
            state.setdefault("errors", {})[f"{AGENT}:T={t}"] = repr(exc)
            raw = {"_error": repr(exc)}

        candidates.append(
            {
                "temperature": float(t),
                "diagnosis_label": str(raw.get("diagnosis_label", "")).strip(),
                "severity_grade": _normalize_severity(raw.get("severity_grade")),
                "reasoning_chain": str(raw.get("reasoning_chain", "")).strip(),
                "evidence": list(raw.get("evidence") or []),
                "raw_json": raw,
            }
        )

    new_state = dict(state)
    new_state["diagnosis_candidates"] = candidates
    return new_state  # type: ignore[return-value]
