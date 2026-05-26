"""把 scripts/knowledge_docs/ 下的文档灌入 RAG 索引。

用法：
  uv run python -m scripts.ingest_knowledge
"""
from __future__ import annotations

import sys

from loguru import logger

from src.config import settings
from src.tools import rag_retriever


def main() -> int:
    logger.info(f"docs_dir = {settings.rag_docs_dir}")
    logger.info(f"index_dir = {settings.rag_index_dir}")
    n = rag_retriever.rebuild_index()
    logger.info(f"done; {n} docs indexed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
