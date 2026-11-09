"""启动预热（M8 / 06 A6）

**为什么要有这个模块**（2026-11-04 实测）：第一问的耗时与"检索"无关，全是进程级一次性成本——

| 环节 | 冷启动实测 |
| --- | --- |
| 首次 import torch / transformers | 88 s（磁盘冷缓存；热缓存 17.6 s） |
| bge-small-zh 权重加载 + 首次编码 | 0.3 ~ 23 s |
| BM25 索引构建（4,576 块；V003 后已移除该步骤） | 10.7 s |
| 解析库 import（pdfplumber / python-docx / openpyxl） | 每个 0.3~1 s（推断） |

这些成本如果留给"第一个提问 / 第一次上传"，界面就是无提示的等待。本模块在 lifespan
里**同步**做掉：**返回即代表服务可干活**。

为什么是同步而不是后台线程（2026-11-05 改）：
    后台预热会让服务在"端口已监听、但模型还没加载"的窗口里接请求 —— 请求会静默阻塞在
    `embed_query` 上几十秒，前端只能靠 `warmup` 状态禁用发送来兜，脚本 / curl 直接卡死。
    改成同步后，uvicorn 在 startup 完成前不把请求交给应用，"能连上" = "能干活"。

设计约束：
1. **失败不阻断启动**：预热失败只置状态位，服务照常启动，第一个请求会自己再试一遍；
2. **幂等**：重复调用只跑一次；
3. **可关**：`RAG_WARMUP=0` 时保持旧行为（不做任何预热）。
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger("zhiwen.warmup")

# 预热用的探针句：走和检索完全相同的 `embed_query` 路径（含 BGE 指令前缀），
# 这样预热掉的是真实路径，而不是另一条捷径。
PROBE_TEXT = "预热：为这个句子生成表示以用于检索相关文章。"

# loaders.py 里延迟 import 的三个解析库：首次上传对应类型时才 import。
PARSER_MODULES = ("pdfplumber", "docx", "openpyxl")

_started = False
_lock = threading.Lock()


def _preload_parsers() -> float:
    """预加载解析库，返回耗时（秒）。失败只告警——首次上传该类型时会自动重试。"""
    started = time.perf_counter()
    for name in PARSER_MODULES:
        try:
            __import__(name)
        except Exception:  # noqa: BLE001
            logger.warning("解析库 %s 预加载失败（首次上传该类型时会重试）", name)
    return time.perf_counter() - started


def run_blocking(container: Any) -> None:
    """同步预热：lifespan startup 阶段调用，**返回即代表可服务**（幂等）。

    在 `asyncio.to_thread` 里跑，事件循环不被阻塞；但 lifespan 尚未 yield，
    所以 uvicorn 不会把请求交给应用——客户端连上来只会排队，不会拿到半成品。
    """
    global _started
    with _lock:
        if _started:
            return
        _started = True

    container.warmup_state = "warming"
    t0 = time.perf_counter()
    logger.info("预热开始（服务在预热完成前不接收请求）")

    t_parsers = _preload_parsers()
    logger.info("  解析库就绪 %.1fs（%s）", t_parsers, " / ".join(PARSER_MODULES))

    try:
        from server.rag import embedder

        embedder.embed_query(PROBE_TEXT)
        container.model_state = "loaded"
    except Exception:  # noqa: BLE001
        container.model_state = "failed"
        container.warmup_state = "failed"
        logger.exception("预热失败：向量模型加载出错（服务仍启动，首个请求会重试）")
        return
    logger.info("  向量模型就绪 %.1fs", time.perf_counter() - t0 - t_parsers)

    # 关键词索引由 PG GIN 倒排维护（V003），无需在启动时构建；
    # 这里只做一次轻量探测，确认 content_tsv 列与索引可用（迁移未执行时尽早暴露）。
    try:
        n = container.chunks.count()
        logger.info("  关键词索引就绪（PG tsvector），%s 块", n)
    except Exception:  # noqa: BLE001
        container.warmup_state = "failed"
        logger.exception("预热失败：关键词索引不可用（检查 V003 迁移是否已执行）")
        return

    container.warmup_state = "ready"
    logger.info("预热完成：合计 %.1fs，服务开始接收请求", time.perf_counter() - t0)
