"""独立的嵌入服务（P0-2 的落地形态）。

**为什么要有它**：编码原本跑在 uvicorn 进程里，4 并发 × 4 线程 = 16 线程正好占满
16 核，事件循环（负责转发 DeepSeek 的流式包）抢不到 CPU，表现为「首字节从 1.0 s
涨到 2.8 s」。搬到独立进程后，编码只吃这个进程的 CPU，Web 进程始终有余量。

**核心是攒批（continuous batching），不是「搬到另一个进程」本身。**
朴素实现（每个 HTTP 请求自己 encode）实测更糟：64 并发 → 40 个 anyio 线程各开
12 个 torch 线程 = 480 线程抢 16 核，吞吐从 6.6 掉到 2.6 问/秒。
这里改成一个批处理调度器：请求进队列 → worker 等 10 ms 攒成一批 → 一次前向编码
→ 按请求切片分发。实测 64 并发能攒成 2~3 批，每批 32 条约 150 ms。

**接口刻意做成 TEI 兼容**（`POST /embed`，`{"inputs": [...], "normalize": true}`），
这样以后换成真正的 TEI / Infinity 容器时，`server/rag/embedder.py` 一行都不用改。

启动（必须用装了 torch 的解释器，不能用 Hermes 自带的 3.14）：
    PYTHONIOENCODING=utf-8 D:/develop/python/python.exe -m uvicorn \
        scripts.embed_server:app --host 127.0.0.1 --port 8088 --workers 1

主应用侧启用：export RAG_TEI_URL=http://127.0.0.1:8088
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import threading
import time

from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger("zhiwen.embed_server")

# 服务侧可以用满 CPU：这里没有事件循环要照顾
WORKER_THREADS = int(os.getenv("RAG_EMBED_SERVER_THREADS", "12"))
BATCH_SIZE = int(os.getenv("RAG_EMBED_SERVER_BATCH", "32"))
# 攒批窗口：请求入队后最多等这么久，把同时到达的合成一批。
# 10 ms 对首字节的影响可忽略（问答链路 1~3 s），但能把 64 个并发压成 2~3 批。
BATCH_WINDOW = float(os.getenv("RAG_EMBED_SERVER_WINDOW", "0.01"))
MAX_BATCH = int(os.getenv("RAG_EMBED_SERVER_MAX_BATCH", "64"))
# 编码线程数。必须与每线程的 torch 线程数相乘 = 核数，否则要么用不满、要么超订。
# 4 worker × 4 torch 线程 = 16 = 本机核数。
WORKERS = int(os.getenv("RAG_EMBED_SERVER_WORKERS", "4"))

app = FastAPI(title="zhiwen embed server", version="1.1")


class EmbedRequest(BaseModel):
    inputs: list[str] | str
    normalize: bool = True


# ----------------------------------------------------------------------
# 批处理调度器
# ----------------------------------------------------------------------
_pending: list[tuple[list[str], concurrent.futures.Future]] = []
_cv = threading.Condition()
_started = False
_stats = {"batches": 0, "items": 0, "max_batch": 0}


def _worker(slot: int) -> None:
    """编码线程：从队列取一批 → 一次前向 → 按请求切片回填 Future。

    多个 worker 并存时，每个都独立取批次，于是 N 个批可以真正并行编码
    （N × WORKER_THREADS 个 torch 线程）。只有 1 个 worker 时，无论多少并发
    编码都是串行的 —— 实测 32 并发吞吐 12.8/s，反而不如进程内 4 并发的 24.2/s。
    """
    from server.rag import embedder

    model = embedder._get_model()          # 加载模型（懒加载 + torch 线程设置）
    logger.info("编码线程 #%s 就绪（共 %s 个）：threads=%s batch=%s window=%.0fms",
                slot, WORKERS, WORKER_THREADS, BATCH_SIZE, BATCH_WINDOW * 1000)

    while True:
        with _cv:
            while not _pending:
                _cv.wait()
            # 再等一个短窗口，把「几乎同时到达」的请求攒到一起
            if BATCH_WINDOW > 0:
                _cv.wait(timeout=BATCH_WINDOW)
            items, _pending[:] = _pending[:MAX_BATCH], _pending[MAX_BATCH:]

        if not items:
            continue

        texts: list[str] = []
        slices: list[tuple[concurrent.futures.Future, int, int]] = []
        for req_texts, fut in items:
            slices.append((fut, len(texts), len(req_texts)))
            texts.extend(req_texts)

        began = time.perf_counter()
        try:
            vectors = model.encode(
                texts,
                batch_size=BATCH_SIZE,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        except Exception as exc:                    # noqa: BLE001
            logger.exception("编码失败，%s 个请求一起失败", len(items))
            for _, fut in items:
                if not fut.done():
                    fut.set_exception(exc)
            continue

        for fut, start, n in slices:
            if not fut.done():
                fut.set_result(vectors[start : start + n].tolist())

        _stats["batches"] += 1
        _stats["items"] += len(texts)
        _stats["max_batch"] = max(_stats["max_batch"], len(texts))
        cost = (time.perf_counter() - began) * 1000
        logger.info("批 %s 条 / %s 请求 / %.0f ms", len(texts), len(items), cost)


def _ensure_worker() -> None:
    global _started
    if _started:
        return
    with _cv:
        if _started:
            return
        _started = True
        for slot in range(WORKERS):
            threading.Thread(target=_worker, args=(slot,),
                             name=f"embed-batcher-{slot}", daemon=True).start()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", **_stats}


@app.get("/info")
def info() -> dict:
    return {
        "model": "BAAI/bge-small-zh-v1.5",
        "dim": 512,
        "threads": WORKER_THREADS,
        "workers": WORKERS,
        "batch": BATCH_SIZE,
        "window_ms": BATCH_WINDOW * 1000,
        **_stats,
    }


@app.post("/embed")
async def embed(req: EmbedRequest) -> list[list[float]]:
    """批量编码：把请求塞进队列等 worker 成形，异步等待自己的那一片结果。"""
    _ensure_worker()

    texts = [req.inputs] if isinstance(req.inputs, str) else list(req.inputs)
    if not texts:
        return []

    fut: concurrent.futures.Future = concurrent.futures.Future()
    with _cv:
        _pending.append((texts, fut))
        _cv.notify()
    return await asyncio.wrap_future(fut)
