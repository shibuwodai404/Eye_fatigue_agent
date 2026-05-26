"""CaseLibraryAgent —— 把本次诊断写入病例库。

JSONL fallback / PG 后端由 src.tools.case_db 内部判断。
入库前会生成两类向量：
- symptom_embedding：主诉文本的 embedding
- multimodal_embedding：主诉 + 行为指标拼接文本的 embedding（用于相似检索）
"""
from __future__ import annotations

from datetime import datetime, timezone

from loguru import logger

from src.config import settings
from src.state import GraphState
from src.tools import case_db, rag_retriever

AGENT = "case_library"


def run(state: GraphState) -> GraphState:
    new_state = dict(state)

    if settings.ablation.disable_case_library:
        logger.info("[case_library] 已禁用，跳过入库")
        new_state["case_library_record_id"] = None
        return new_state  # type: ignore[return-value]

    final = state.get("final_diagnosis") or {}
    intake = state.get("intake") or {}
    metrics = state.get("vision_metrics") or {}
    chief = intake.get("chief_complaint") or ""

    multimodal_query = state.get("_retrieval_query_text") or chief
    try:
        symptom_emb = rag_retriever.embed_text(chief) if chief else []
        multimodal_emb = rag_retriever.embed_text(multimodal_query) if multimodal_query else symptom_emb
    except Exception as exc:  # noqa: BLE001
        logger.exception("[case_library] embedding 失败")
        new_state.setdefault("errors", {})[AGENT + ":embedding"] = repr(exc)
        symptom_emb, multimodal_emb = [], []

    record = {
        "patient_id": state.get("patient_id"),
        "captured_at": state.get("timestamp") or datetime.now(timezone.utc).isoformat(),
        "chief_complaint": chief,
        "scale_scores": {
            "cvs_q_total": intake.get("cvs_q_total"),
            "osdi_total": intake.get("osdi_total"),
            **(intake.get("scale_extras") or {}),
        },
        "vision_metrics": metrics,
        "final_diagnosis": final.get("diagnosis_label"),
        "severity_grade": final.get("severity_grade"),
        "reasoning_chain": final.get("reasoning_chain"),
        "doctor_review": None,
        "video_path": state.get("video_path"),
        "symptom_embedding": symptom_emb,
        "multimodal_embedding": multimodal_emb,
    }

    try:
        rid = case_db.insert_case(record)
    except Exception as exc:  # noqa: BLE001
        logger.exception("[case_library] 写库失败")
        new_state.setdefault("errors", {})[AGENT] = repr(exc)
        rid = None

    new_state["case_library_record_id"] = rid
    return new_state  # type: ignore[return-value]
