"""病例库读写。

两种后端：
- JSONL（默认）：一行一条 record，向量字段也存进 JSON；相似度计算用 numpy
- POSTGRES：PG + pgvector，schema 见 scripts/init_db.sql

切换方式：.env 里改 CASE_DB_BACKEND。
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from loguru import logger

from src.config import settings


# ============================================================
# 公共接口
# ============================================================

def insert_case(record: dict[str, Any]) -> str:
    """写入一条病例。返回 record_id。

    record 至少要包含：
      patient_id, captured_at, chief_complaint, scale_scores (dict),
      vision_metrics (dict), final_diagnosis (str), severity_grade (str),
      reasoning_chain (str), video_path (str), symptom_embedding (list[float])
    可选：doctor_review (str|None), multimodal_embedding (list[float])
    """
    record = {**record}
    record.setdefault("id", str(uuid.uuid4()))
    record.setdefault("doctor_review", None)

    backend = settings.case_db_backend.upper()
    if backend == "JSONL":
        return _jsonl_insert(record)
    if backend == "POSTGRES":
        return _pg_insert(record)
    raise ValueError(f"unknown case_db_backend: {settings.case_db_backend}")


def search_similar(
    *,
    query_embedding: list[float] | np.ndarray,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """按 cosine 相似度返回 top_k 条历史病例。"""
    if settings.ablation.disable_similar_cases:
        return []

    backend = settings.case_db_backend.upper()
    if backend == "JSONL":
        return _jsonl_search(query_embedding, top_k)
    if backend == "POSTGRES":
        return _pg_search(query_embedding, top_k)
    raise ValueError(f"unknown case_db_backend: {settings.case_db_backend}")


# ============================================================
# JSONL fallback 实现
# ============================================================

def _jsonl_path() -> Path:
    p = Path(settings.case_db_jsonl_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _jsonl_insert(record: dict[str, Any]) -> str:
    p = _jsonl_path()
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(f"[case_db/jsonl] inserted id={record['id']} -> {p}")
    return record["id"]


def _iter_jsonl() -> Iterable[dict[str, Any]]:
    p = _jsonl_path()
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _jsonl_search(query_embedding, top_k: int) -> list[dict[str, Any]]:
    q = np.asarray(query_embedding, dtype=np.float32)
    qn = np.linalg.norm(q) + 1e-9
    out: list[tuple[float, dict[str, Any]]] = []
    for row in _iter_jsonl():
        emb = row.get("multimodal_embedding") or row.get("symptom_embedding")
        if not emb:
            continue
        v = np.asarray(emb, dtype=np.float32)
        if v.shape != q.shape:
            continue
        sim = float(q @ v / (np.linalg.norm(v) * qn + 1e-9))
        out.append((sim, row))
    out.sort(key=lambda x: x[0], reverse=True)
    return [{**row, "_similarity": s} for s, row in out[:top_k]]


# ============================================================
# PostgreSQL + pgvector 实现（懒导入）
# ============================================================

_PG_INSERT_SQL = """
INSERT INTO cases (
    id, patient_id, captured_at, chief_complaint, scale_scores,
    vision_metrics, final_diagnosis, severity_grade, reasoning_chain,
    doctor_review, video_path, symptom_embedding, multimodal_embedding
) VALUES (
    %(id)s, %(patient_id)s, %(captured_at)s, %(chief_complaint)s, %(scale_scores)s,
    %(vision_metrics)s, %(final_diagnosis)s, %(severity_grade)s, %(reasoning_chain)s,
    %(doctor_review)s, %(video_path)s, %(symptom_embedding)s, %(multimodal_embedding)s
)
"""

_PG_SEARCH_SQL = """
SELECT id, patient_id, captured_at, chief_complaint, scale_scores,
       vision_metrics, final_diagnosis, severity_grade, reasoning_chain,
       doctor_review, video_path,
       1 - (multimodal_embedding <=> %(q)s::vector) AS similarity
FROM cases
WHERE multimodal_embedding IS NOT NULL
ORDER BY multimodal_embedding <=> %(q)s::vector
LIMIT %(k)s
"""


def _pg_connect():
    import psycopg
    from pgvector.psycopg import register_vector

    conn = psycopg.connect(settings.postgres_dsn)
    register_vector(conn)
    return conn


def _pg_insert(record: dict[str, Any]) -> str:
    from psycopg.types.json import Json

    params = {
        "id": record["id"],
        "patient_id": record["patient_id"],
        "captured_at": record["captured_at"],
        "chief_complaint": record.get("chief_complaint"),
        "scale_scores": Json(record.get("scale_scores") or {}),
        "vision_metrics": Json(record.get("vision_metrics") or {}),
        "final_diagnosis": record.get("final_diagnosis"),
        "severity_grade": record.get("severity_grade"),
        "reasoning_chain": record.get("reasoning_chain"),
        "doctor_review": record.get("doctor_review"),
        "video_path": record.get("video_path"),
        "symptom_embedding": record.get("symptom_embedding"),
        "multimodal_embedding": record.get("multimodal_embedding") or record.get("symptom_embedding"),
    }
    with _pg_connect() as conn, conn.cursor() as cur:
        cur.execute(_PG_INSERT_SQL, params)
        conn.commit()
    logger.info(f"[case_db/pg] inserted id={record['id']}")
    return record["id"]


def _pg_search(query_embedding, top_k: int) -> list[dict[str, Any]]:
    import numpy as np

    q = np.asarray(query_embedding, dtype=np.float32).tolist()
    with _pg_connect() as conn, conn.cursor() as cur:
        cur.execute(_PG_SEARCH_SQL, {"q": q, "k": top_k})
        cols = [d.name for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    for row in rows:
        if "similarity" in row:
            row["_similarity"] = float(row.pop("similarity"))
    return rows
