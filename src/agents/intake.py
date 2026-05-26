"""IntakeAgent —— 把主诉自由文本 + 量表总分整理成结构化字段。"""
from __future__ import annotations

import json
from typing import Any

from loguru import logger

from src.llm import chat, parse_json_loose, prompt
from src.state import GraphState

AGENT = "intake"


def run(state: GraphState) -> GraphState:
    intake = dict(state.get("intake") or {})
    chief = intake.get("chief_complaint", "")
    demo = intake.get("demographics", {})
    cvs = intake.get("cvs_q_total")
    osdi = intake.get("osdi_total")

    if not chief and not demo:
        logger.warning("[intake] 主诉与人口学信息均为空，跳过 LLM 结构化")
        intake.setdefault("structured", {"symptoms": [], "missing_info": ["未提供主诉"]})
        new_state = dict(state)
        new_state["intake"] = intake
        return new_state  # type: ignore[return-value]

    user_payload: dict[str, Any] = {
        "chief_complaint": chief,
        "demographics": demo,
        "cvs_q_total": cvs,
        "osdi_total": osdi,
        "scale_extras": intake.get("scale_extras", {}),
    }
    user_msg = (
        "下面是患者本次就诊信息（JSON）：\n"
        f"```json\n{json.dumps(user_payload, ensure_ascii=False, indent=2)}\n```\n"
        "请按系统指令输出结构化 JSON。"
    )

    try:
        text = chat(
            agent=AGENT,
            system=prompt("intake_system"),
            user=user_msg,
            state=state,
            temperature=0.1,
        )
        structured = parse_json_loose(text)
    except Exception as exc:  # noqa: BLE001
        logger.exception("[intake] LLM 调用失败")
        state.setdefault("errors", {})[AGENT] = repr(exc)
        structured = {"_error": repr(exc)}

    intake["structured"] = structured
    new_state = dict(state)
    new_state["intake"] = intake
    return new_state  # type: ignore[return-value]
