"""ReportAgent —— 生成医生版 + 患者版 Markdown 双报告。

两个 LLM 调用，分别用 doctor / patient 的系统提示词。
"""
from __future__ import annotations

import json
from typing import Any

from loguru import logger

from src.llm import chat, prompt
from src.state import GraphState

AGENT = "report"


def _bundle(state: GraphState) -> dict[str, Any]:
    return {
        "patient_id": state.get("patient_id"),
        "session_id": state.get("session_id"),
        "timestamp": state.get("timestamp"),
        "intake": state.get("intake", {}),
        "vision_metrics": state.get("vision_metrics", {}),
        "aggregated": state.get("aggregated", {}),
        "retrieval": state.get("retrieval", {}),
        "final_diagnosis": state.get("final_diagnosis", {}),
        "requires_human_review": state.get("requires_human_review", False),
    }


def _render(state: GraphState, *, kind: str) -> str:
    """kind: 'doctor' or 'patient'."""
    system = prompt(f"report_{kind}_system")
    payload = _bundle(state)
    user_msg = (
        "下面是本次诊断的完整 state（已剔除日志字段）：\n"
        f"```json\n{json.dumps(payload, ensure_ascii=False, indent=2, default=str)}\n```\n"
        "请按系统指令输出 Markdown 报告（纯字符串，不要包成 JSON）。"
    )
    try:
        text = chat(
            agent=f"{AGENT}:{kind}",
            system=system,
            user=user_msg,
            state=state,
            temperature=0.3,
            format_json=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception(f"[report] {kind} 失败")
        state.setdefault("errors", {})[f"{AGENT}:{kind}"] = repr(exc)
        text = f"# 报告生成失败\n\n错误: {exc!r}"
    return text


def run(state: GraphState) -> GraphState:
    doctor_md = _render(state, kind="doctor")
    patient_md = _render(state, kind="patient")

    new_state = dict(state)
    new_state["reports"] = {"doctor_md": doctor_md, "patient_md": patient_md}
    return new_state  # type: ignore[return-value]
