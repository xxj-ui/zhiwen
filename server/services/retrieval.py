"""检索编排（02 §3.2）：问题 → 稠密召回 + 关键词召回 → RRF 融合。

分工：`rag/` 只提供纯算法（`embed_query` / `tokenize_query` / `rrf_fuse`），
"从库里取哪些片段、两路各召回多少、融合后留几条"属于编排，放在这里，
参数一律走 `Settings`（recall_k / top_k / rrf_k），不散落在各处。

关键词一路自 V003 起走 PostgreSQL 全文索引（tsvector + GIN），
进程内不再持有索引，因此没有 rebuild / invalidate 生命周期。
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from server.rag.embedder import EMBED_CONCURRENCY, TEI_URL, embed_query
from server.rag.keyword_search import tokenize_query
from server.rag.rrf import rrf_fuse

# 编码专用线程池。
#
# 进程内编码：大小 = 编码并发上限（默认 4）。4 × 4 线程 = 16 = 核数，多开只会抢核。
#
# 外置编码（RAG_TEI_URL）：**必须放大**。编码不占本进程 CPU 了，而远端服务靠攒批
# 提升吞吐 —— 这里限成 4 就等于把攒批的上限钉死在 4。实测：worker=4 时远端
# max_batch 只有 4、109 批才处理 289 条；放大到 32 后一次能攒 32 条（实测 batch=32
# 比逐条快 5.3 倍）。
_EMBED_WORKERS = 32 if TEI_URL else max(1, EMBED_CONCURRENCY)
_embed_pool = ThreadPoolExecutor(
    max_workers=_EMBED_WORKERS, thread_name_prefix="embed"
)


def _resolve(
    container: Any, top_k: int | None, recall_k: int | None, rrf_k: int | None
) -> tuple[int, int, int]:
    settings = container.settings
    return (
        settings.top_k if top_k is None else top_k,
        settings.recall_k if recall_k is None else recall_k,
        settings.rrf_k if rrf_k is None else rrf_k,
    )


def _prepare(question: str) -> tuple[list[float], str]:
    """编码 + 分词。两者都是纯 CPU（jieba 持 GIL），必须一起待在专用池里。

    为什么分词不能留在 `_fuse` 里：`_fuse` 跑在默认池（20 worker），而 jieba 是
    纯 Python 计算、全程持 GIL —— 20 个线程同时分词等于串行 + 高频线程切换。
    实测 32 并发下 `tokenize_query` P50 从 1 ms 膨胀到 **1686 ms**（×1686），
    一度被误判成"检索慢""LLM 抢 GIL"。挪进 4 线程的专用池后回到毫秒级。
    """
    return embed_query(question), tokenize_query(question)


def _fuse(
    container: Any,
    vector: list[float],
    query_text: str,
    top_k: int,
    recall_k: int,
    rrf_k: int,
) -> list[dict]:
    """召回 + 融合（纯 IO：2 条 SQL + 纯函数 RRF，不含编码与分词）。"""
    repo = container.chunks
    # 两路查询共用一个连接：池每次出池都要探活，分两次取就是两次往返（实测 ~5 ms/次）
    with repo.session() as conn:
        dense = repo.search(vector, top_k=recall_k, conn=conn)
        keyword = repo.search_keyword(query_text, top_k=recall_k, conn=conn)
    return rrf_fuse([("dense", dense), ("keyword", keyword)], k=rrf_k, top_k=top_k)


def retrieve(
    container: Any,
    question: str,
    top_k: int | None = None,
    recall_k: int | None = None,
    rrf_k: int | None = None,
) -> list[dict]:
    """同步版（CLI / 脚本）：编码 + 分词 + 召回 + 融合，返回 top_k 条。"""
    top_k, recall_k, rrf_k = _resolve(container, top_k, recall_k, rrf_k)
    vector, query_text = _prepare(question)
    return _fuse(container, vector, query_text, top_k, recall_k, rrf_k)


async def aretrieve(container: Any, question: str, **kwargs: Any) -> list[dict]:
    """异步版（HTTP 路径）。

    分工：
      1. **编码 + 分词** → 专用池（`_embed_pool`，大小 = RAG_EMBED_CONCURRENCY）。
         两者都是纯 CPU，jieba 还持 GIL，必须共用一个**小**池，否则线程数一多
         就互相拖死（实测 32 并发下分词 ×1686 膨胀）。
      2. **2 条 SQL + RRF** → 默认池（IO，与读历史、落库共享）。

    事件循环夹在两个 await 之间保持可调度，LLM 的流式响应不会被 CPU 任务压住。
    """
    top_k, recall_k, rrf_k = _resolve(
        container, kwargs.get("top_k"), kwargs.get("recall_k"), kwargs.get("rrf_k")
    )
    loop = asyncio.get_running_loop()
    vector, query_text = await loop.run_in_executor(_embed_pool, _prepare, question)
    return await asyncio.to_thread(
        _fuse, container, vector, query_text, top_k, recall_k, rrf_k
    )
