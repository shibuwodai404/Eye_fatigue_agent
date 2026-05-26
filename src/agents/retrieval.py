"""RetrievalAgent —— RAG 指南 + 相似历史病例。

不走 LLM，纯工具组合：
- 指南：LlamaIndex 检索器
- 相似病例：用 BGE-M3 把 (主诉 + 关键指标) 拼接成 query embedding，
            到病例库做向量检索（pgvector / JSONL fallback）
"""
from __future__ import annotations

import json
from typing import Any

from loguru import logger

from src.config import settings
from src.state import GraphState
from src.tools import case_db, rag_retriever

AGENT = "retrieval"


def _build_query_text(state: GraphState) -> str:
    """把主诉 + 关键归一化指标拼成一段供 embedding 的描述文本。"""
    intake = state.get("intake") or {}
    structured = intake.get("structured") or {}
    agg = state.get("aggregated") or {}

    parts: list[str] = []
    if intake.get("chief_complaint"):
        parts.append(f"主诉：{intake['chief_complaint']}")
    if structured.get("symptoms"):
        parts.append("症状：" + json.dumps(structured["symptoms"], ensure_ascii=False))
    if intake.get("cvs_q_total") is not None:
        parts.append(f"CVS-Q 总分 {intake['cvs_q_total']}")
    if intake.get("osdi_total") is not None:
        parts.append(f"OSDI 总分 {intake['osdi_total']}")
    if agg.get("anomalies"):
        anomalies_brief = [
            f"{a['metric']}({a['severity']},z={a['z']})" for a in agg["anomalies"]
        ]
        parts.append("行为指标异常：" + "; ".join(anomalies_brief))
    if agg.get("composite_score") is not None:
        parts.append(f"综合疲劳分 {agg['composite_score']}")

    return " | ".join(parts) if parts else "视疲劳就诊一般查询"


def _build_multimodal_embedding(state: GraphState, query_text: str) -> list[float]:
    """主诉文本 + 关键归一化指标 → 单一向量。

    第一版：直接 embed 拼接后的文本（已包含数值化描述）。
    后续可换成 text_embed + 指标向量的早融合 / 晚融合策略。
    """
    return rag_retriever.embed_text(query_text)


def run(state: GraphState) -> GraphState:
    if settings.ablation.disable_rag and settings.ablation.disable_similar_cases:
        logger.info("[retrieval] RAG 与相似病例均关闭，跳过")
        new_state = dict(state)
        new_state["retrieval"] = {"guideline": [], "case": []}
        return new_state  # type: ignore[return-value]

    query_text = _build_query_text(state)
    logger.info(f"[retrieval] query: {query_text[:120]}...")

    guideline_hits: list[dict[str, Any]] = []
    case_hits: list[dict[str, Any]] = []

    if not settings.ablation.disable_rag:
        try:
            guideline_hits = rag_retriever.retrieve_guideline(query_text)
        except Exception as exc:  # noqa: BLE001
            logger.exception("[retrieval] guideline 检索失败")
            state.setdefault("errors", {})[AGENT + ":guideline"] = repr(exc)

    if not settings.ablation.disable_similar_cases:
        try:
            emb = _build_multimodal_embedding(state, query_text)
            raw_hits = case_db.search_similar(query_embedding=emb, top_k=5)
            for h in raw_hits:
                case_hits.append(
                    {
                        "source": "case",
                        "doc_id": h.get("id"),
                        "title": h.get("final_diagnosis") or "历史病例",
                        "score": float(h.get("_similarity") or 0.0),
                        "snippet": (h.get("reasoning_chain") or "")[:600],
                        "metadata": {
                            "severity_grade": h.get("severity_grade"),
                            "patient_id": h.get("patient_id"),
                            "captured_at": h.get("captured_at"),
                        },
                    }
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("[retrieval] 相似病例检索失败")
            state.setdefault("errors", {})[AGENT + ":cases"] = repr(exc)

    new_state = dict(state)
    new_state["retrieval"] = {"guideline": guideline_hits, "case": case_hits}
    # 顺便把本次的多模态 query 文本和 embedding 暂存，方便 CaseLibraryAgent 入库复用
    new_state["_retrieval_query_text"] = query_text  # type: ignore[typeddict-unknown-key]
    return new_state  # type: ignore[return-value]
