"""RAG 检索增强生成模块 — 面试知识库检索

技术栈：Chroma 向量数据库 + SentenceTransformer Embedding

用法：
    from src.rag import search_knowledge
    context = search_knowledge("LangGraph Send 并行")
    # → 返回最相关的面经条目，可直接拼入 Prompt
"""

from __future__ import annotations

import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "chroma_db"
KNOWLEDGE_FILE = ROOT / "data" / "interview_knowledge.md"

# Embedding 模型：all-MiniLM-L6-v2，384维，轻量快速，本地运行不需要 API key
_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

_client = chromadb.PersistentClient(path=str(DB_PATH))
_collection = _client.get_or_create_collection(
    name="interview_knowledge",
    embedding_function=_ef,
    metadata={"description": "AI Agent / 大模型应用开发面试知识库"},
)


def is_indexed() -> bool:
    """知识库是否已索引"""
    return _collection.count() > 0


def build_index(filepath: str | None = None, force: bool = False) -> int:
    """索引面经知识库（项目启动时调用一次）

    把 interview_knowledge.md 按 ## 标题分块，
    每块转为 Embedding 向量存入 Chroma。
    """
    path = Path(filepath) if filepath else KNOWLEDGE_FILE
    if not path.exists():
        return 0

    if is_indexed() and not force:
        return _collection.count()

    text = path.read_text(encoding="utf-8")

    # 按 ## 标题分块，保留标题作为上下文
    chunks = []
    current_title = ""
    current_content = []

    for line in text.split("\n"):
        if line.startswith("## "):
            if current_content:
                chunks.append(current_title + "\n" + "\n".join(current_content))
            current_title = line
            current_content = []
        elif line.strip():
            current_content.append(line)

    if current_content:
        chunks.append(current_title + "\n" + "\n".join(current_content))

    if not chunks:
        return 0

    # 清空旧数据（如果 force）
    if force:
        try:
            _client.delete_collection("interview_knowledge")
        except Exception:
            pass

    # 存入向量库
    ids = [f"chunk-{i}" for i in range(len(chunks))]
    _collection.upsert(documents=chunks, ids=ids)

    return len(chunks)


def search_knowledge(query: str, n_results: int = 3) -> list[str]:
    """检索与 query 最相关的面经条目

    Args:
        query: 搜索查询（如 "LangGraph 并行分发"）
        n_results: 返回条数

    Returns:
        相关文档片段列表，每条是一道面试题的完整 Q&A
    """
    if _collection.count() == 0:
        return []

    results = _collection.query(query_texts=[query], n_results=n_results)
    docs = results.get("documents", [[]])[0]
    return [d for d in docs if d]
