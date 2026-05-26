"""LlamaIndex + BGE-M3 RAG 检索。

第一版用本地 SimpleDirectoryReader 读 scripts/knowledge_docs/，
落盘到 logs/rag_index/ 作为持久化向量库。
"""
from __future__ import annotations

import threading
from typing import Any

from loguru import logger

from src.config import settings


_INDEX = None
_EMBED = None
_LOCK = threading.Lock()


def _get_embed_model():
    """懒加载 BGE-M3 embedding 模型，进程内共享。"""
    global _EMBED
    if _EMBED is not None:
        return _EMBED
    with _LOCK:
        if _EMBED is None:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding

            logger.info(f"[rag] loading embedding: {settings.embedding_model_name} on {settings.embedding_device}")
            _EMBED = HuggingFaceEmbedding(
                model_name=settings.embedding_model_name,
                device=settings.embedding_device,
            )
    return _EMBED


def embed_text(text: str) -> list[float]:
    """单文本 embedding，供病例库相似度检索 / state.symptom_embedding 使用。"""
    model = _get_embed_model()
    vec = model.get_text_embedding(text)
    return list(map(float, vec))


def _load_or_build_index():
    global _INDEX
    if _INDEX is not None:
        return _INDEX
    with _LOCK:
        if _INDEX is not None:
            return _INDEX

        from llama_index.core import (
            Settings as LIDSettings,
            SimpleDirectoryReader,
            StorageContext,
            VectorStoreIndex,
            load_index_from_storage,
        )

        LIDSettings.embed_model = _get_embed_model()
        # 关闭 LlamaIndex 默认对 OpenAI LLM 的依赖：我们只用它做检索
        LIDSettings.llm = None  # type: ignore[assignment]

        index_dir = settings.rag_index_dir
        index_dir.mkdir(parents=True, exist_ok=True)

        if any(index_dir.iterdir()):
            logger.info(f"[rag] loading existing index from {index_dir}")
            storage = StorageContext.from_defaults(persist_dir=str(index_dir))
            _INDEX = load_index_from_storage(storage)
        else:
            docs_dir = settings.rag_docs_dir
            docs_dir.mkdir(parents=True, exist_ok=True)
            if not any(docs_dir.iterdir()):
                logger.warning(f"[rag] {docs_dir} 为空，索引将为空；先跑 scripts/ingest_knowledge.py")
                docs = []
            else:
                docs = SimpleDirectoryReader(str(docs_dir), recursive=True).load_data()
            _INDEX = VectorStoreIndex.from_documents(docs)
            _INDEX.storage_context.persist(persist_dir=str(index_dir))
            logger.info(f"[rag] built index ({len(docs)} docs) -> {index_dir}")
    return _INDEX


def retrieve_guideline(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """返回 top_k 条指南文献片段。

    返回结构：[{doc_id, title, score, snippet, metadata}, ...]
    """
    if settings.ablation.disable_rag:
        return []

    k = top_k or settings.rag_top_k
    try:
        index = _load_or_build_index()
        retriever = index.as_retriever(similarity_top_k=k)
        nodes = retriever.retrieve(query)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"[rag] retrieve failed: {exc!r}")
        return []

    out: list[dict[str, Any]] = []
    for n in nodes:
        meta = dict(getattr(n.node, "metadata", {}) or {})
        out.append(
            {
                "source": "guideline",
                "doc_id": meta.get("file_name") or n.node.node_id,
                "title": meta.get("file_name", "unknown"),
                "score": float(n.score or 0.0),
                "snippet": (n.node.get_content() or "")[:600],
                "metadata": meta,
            }
        )
    return out


def rebuild_index() -> int:
    """供 scripts/ingest_knowledge.py 调用 —— 强制重建。返回索引条目数。"""
    global _INDEX
    from llama_index.core import (
        Settings as LIDSettings,
        SimpleDirectoryReader,
        VectorStoreIndex,
    )

    LIDSettings.embed_model = _get_embed_model()
    LIDSettings.llm = None  # type: ignore[assignment]

    docs_dir = settings.rag_docs_dir
    docs = SimpleDirectoryReader(str(docs_dir), recursive=True).load_data()
    index = VectorStoreIndex.from_documents(docs)

    index_dir = settings.rag_index_dir
    # 清空旧索引
    for p in index_dir.glob("*"):
        if p.is_file():
            p.unlink()
    index.storage_context.persist(persist_dir=str(index_dir))
    _INDEX = index
    logger.info(f"[rag] rebuilt index, {len(docs)} docs -> {index_dir}")
    return len(docs)
