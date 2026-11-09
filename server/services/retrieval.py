"""检索编排（02 §3.2）：问题 → 稠密召回 + 关键词召回 → RRF 融合。

分工：`rag/` 只提供纯算法（`embed_query` / `tokenize_query` / `rrf_fuse`），
"从库里取哪些片段、两路各召回多少、融合后留几条"属于编排，放在这里，
参数一律走 `Settings`（recall_k / top_k / rrf_k），不散落在各处。

关键词一路自 V003 起走 PostgreSQL 全文索引（tsvector + GIN），
进程内不再持有索引，因此没有 rebuild / invalidate 生命周期。
"""
from __future__ import annotations

import asyncio
from typing import Any

from server.rag.embedder import embed_query
from server.rag.keyword_search import tokenize_query
from server.rag.rrf import rrf_fuse


def retrieve(
    container: Any,
    question: str,
    top_k: int | None = None,
    recall_k: int | None = None,
    rrf_k: int | None = None,
) -> list[dict]:
    """两路召回 + RRF 融合，返回 top_k 条（每条形如 04 的 citations 元素原料）。"""
    settings = container.settings
    top_k = settings.top_k if top_k is None else top_k
    recall_k = settings.recall_k if recall_k is None else recall_k
    rrf_k = settings.rrf_k if rrf_k is None else rrf_k

    repo = container.chunks
    # 嵌入与分词不占数据库连接，先算好
    vector = embed_query(question)
    query_text = tokenize_query(question)
    # 两路查询共用一个连接：池每次出池都要探活，分两次取就是两次往返（实测 ~5 ms/次）
    with repo.session() as conn:
        dense = repo.search(vector, top_k=recall_k, conn=conn)
        keyword = repo.search_keyword(query_text, top_k=recall_k, conn=conn)
    return rrf_fuse([("dense", dense), ("keyword", keyword)], k=rrf_k, top_k=top_k)


async def aretrieve(container: Any, question: str, **kwargs: Any) -> list[dict]:
    """异步包装：检索是同步阻塞的（psycopg + 模型），放进线程池，别堵事件循环。"""
    return await asyncio.to_thread(retrieve, container, question, **kwargs)
