"""依赖装配（02 §4.2 / §3.3）

容器只负责"把依赖接好"，不做业务判断：仓储、连接池、进程内索引、模型状态位。
请求级与进程级都从这里取，`app.py` 只管装配与横切。
"""
from __future__ import annotations

import asyncio
from typing import Any

from server.core import storage
from server.core.config import Settings, load_settings
from server.db import pool as db_pool
from server.db.chunks import PgChunkRepository
from server.db.conversations import ConversationRepository
from server.db.documents import DocumentRepository


class Container:
    """进程内单例集合（单 worker，02 S-01）。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._pool: Any = None
        self.documents = DocumentRepository(settings.db_dsn)
        self.conversations = ConversationRepository(settings.db_dsn)
        # 检索走连接池（02 S-09）：pool 惰性创建，首次检索时才建，不拖慢启动
        self.chunks = PgChunkRepository(settings.db_dsn, pool_factory=self.pool)
        # 关键词检索走 PG 全文索引（V003）：进程内不再持有 BM25 索引
        # 原文件存储后端（local / minio）：进程内建一次，上传、覆盖、删除共用
        self.store = storage.build_store(settings)
        # 写锁（04 §3.7 并发写 / 02 S-01 单 worker）：上传、覆盖、删除串行，后到者等待而不是 500
        self.write_lock = asyncio.Lock()
        # 会话级锁（10.多轮会话设计 §9）：同一会话的提问串行，避免两轮交错写库
        self.chat_locks: dict[int, asyncio.Lock] = {}
        # 进程内状态位，供 /api/health 读取（02 §2.6）
        self.model_state: str = "loading"
        # 启动预热状态（M8）：idle | warming | ready | failed
        self.warmup_state: str = "idle"

    def conversation_lock(self, conversation_id: int) -> asyncio.Lock:
        """取某个会话的锁（惰性创建）。会话数量级小（本机单用户），先不做回收。"""
        lock = self.chat_locks.get(conversation_id)
        if lock is None:
            lock = asyncio.Lock()
            self.chat_locks[conversation_id] = lock
        return lock

    def pool(self) -> Any:
        if self._pool is None:
            self._pool = db_pool.create_pool(
                self.settings.db_dsn,
                self.settings.db_pool_min,
                self.settings.db_pool_max,
                self.settings.db_pool_timeout,
            )
        return self._pool

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None


_container: Container | None = None


def get_container() -> Container:
    global _container
    if _container is None:
        _container = Container(load_settings())
    return _container


def set_container(container: Container | None) -> None:
    """测试用：注入/清空容器。"""
    global _container
    _container = container
