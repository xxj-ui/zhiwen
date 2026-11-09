"""第 4 步：向量存储（pgvector 数据访问层）

职责和老项目里的 Mapper / Repository 一致：
    只负责和数据库打交道，不关心向量是怎么算出来的。

连接串只从显式参数或环境变量读取（SRS 2.4.3：不得硬编码密码）：
    RAG_DB_DSN=postgresql://<user>:<password>@127.0.0.1:54328/zhiwen
缺失即抛错，不做任何默认值兜底。
"""

import os
from contextlib import contextmanager

from server.rag.keyword_search import tokenize

TABLE_NAME = "rag_chunks"

def _require_dsn(explicit: str | None = None) -> str:
    """返回数据库连接串；缺失即抛错（P-3：RAG_DB_DSN 必填）。"""
    dsn = explicit or os.getenv("RAG_DB_DSN")
    if not dsn:
        raise RuntimeError(
            "未配置数据库连接串：请设置环境变量 RAG_DB_DSN，"
            "形如 postgresql://<user>:<password>@127.0.0.1:54328/zhiwen"
        )
    return dsn




def _to_vector_literal(vector: list[float]) -> str:
    """把 Python 列表转成 pgvector 接受的字符串格式：[1,2,3]。"""
    return "[" + ",".join(f"{x:.7f}" for x in vector) + "]"


class PgChunkRepository:
    """rag_chunks 表的数据访问对象。

    连接来源有两种（02 S-09）：
      - 传了 `pool_factory`（Web 链路）：检索/统计走进程内连接池，复用连接；
      - 没传（CLI / 一次性脚本）：每次 `psycopg.connect` 直连。
    两条路径都用 `dict_row` 行工厂，返回值形状一致。
    """

    def __init__(self, dsn: str | None = None, pool_factory=None) -> None:
        self.dsn = _require_dsn(dsn)
        self._pool_factory = pool_factory

    def _connect(self):
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as e:
            raise ImportError(
                "缺少数据库驱动，请先安装：pip install 'psycopg[binary]'"
            ) from e
        return psycopg.connect(self.dsn, row_factory=dict_row)

    @contextmanager
    def _reader(self):
        """只读查询取连接：有池用池（省掉 TCP + 认证），无池直连。"""
        if self._pool_factory is not None:
            with self._pool_factory().connection() as conn:
                yield conn
            return
        with self._connect() as own:
            yield own

    @contextmanager
    def session(self):
        """一次检索内复用同一个连接（同样走池）。

        为什么要这个：池配置了 `check_connection` 探活，每次出池都多一次往返
        （实测 ~5 ms）。检索要跑 dense + keyword 两条 SQL，各取一次连接就是
        两次探活；复用后只付一次。纯内存索引时代没有这笔开销，改造后必须补上。
        """
        with self._reader() as conn:
            yield conn

    # ------------------------------------------------------------------
    # 建表
    # ------------------------------------------------------------------

    def delete_by_source(self, source: str) -> int:
        """删除某个文件的旧数据（重新建库时用，保证幂等）。"""
        with self._reader() as conn:
            cur = conn.execute(
                f"DELETE FROM {TABLE_NAME} WHERE source = %s", (source,)
            )
            return cur.rowcount

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    def search(self, query_vector: list[float], top_k: int = 5, conn=None) -> list[dict]:
        """按余弦相似度返回 top_k 个 chunk（相似度从高到低）。

        注意：
            - 查询向量必须和建库时用同一个模型、同样归一化
            - `<=>` 是余弦距离（越小越近），1 - 距离 = 相似度
        """
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        literal = _to_vector_literal(query_vector)
        # filename 统一取 documents.filename（04 §7 数据映射），source 只作兜底
        sql = f"""
            SELECT c.id, c.source, c.page, c.chunk_index, c.content,
                   d.filename AS filename,
                   1 - (c.embedding <=> %s::vector) AS score
            FROM {TABLE_NAME} c
            LEFT JOIN documents d ON d.id = c.document_id
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
        """
        if conn is not None:
            rows = conn.execute(sql, (literal, literal, top_k)).fetchall()
        else:
            with self._reader() as c:
                rows = c.execute(sql, (literal, literal, top_k)).fetchall()

        return [
            {
                "id": row["id"],
                "source": row["source"],
                "page": row["page"],
                "chunk_index": row["chunk_index"],
                "content": row["content"],
                "filename": row["filename"],
                "score": float(row["score"]),
            }
            for row in rows
        ]

    def search_keyword(self, query_text: str, top_k: int = 5, conn=None) -> list[dict]:
        """关键词检索（PG 全文索引 + ts_rank_cd），返回结构同 `search()`。

        `query_text` 是 `to_tsquery('simple', ...)` 的 **OR 表达式**
        （由 `rag/keyword_search.tokenize_query` 生成，形如 `'线程' | '池' | '参数'`）。

        为什么不用 `plainto_tsquery`：它生成 AND 语义，长问题会退化成
        "所有词都必须出现"，实测只命中 0~1 条。BM25 是 OR 语义，必须手工拼 `|`。

        用 `ts_rank_cd`（cover density）而不是 `ts_rank`：额外考虑词距，
        词挨得越近分越高，对中文短语（"三次握手"）比纯词频更贴合语义。
        """
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        if not query_text or not query_text.strip():
            return []

        sql = f"""
            WITH q AS (SELECT to_tsquery('simple', %s) AS tsq)
            SELECT c.id, c.source, c.page, c.chunk_index, c.content,
                   d.filename AS filename,
                   ts_rank_cd(c.content_tsv, q.tsq) AS score
            FROM {TABLE_NAME} c
            LEFT JOIN documents d ON d.id = c.document_id
            CROSS JOIN q
            WHERE c.content_tsv @@ q.tsq
            ORDER BY score DESC, c.id
            LIMIT %s
        """
        if conn is not None:
            rows = conn.execute(sql, (query_text, top_k)).fetchall()
        else:
            with self._reader() as c:
                rows = c.execute(sql, (query_text, top_k)).fetchall()

        return [
            {
                "id": row["id"],
                "source": row["source"],
                "page": row["page"],
                "chunk_index": row["chunk_index"],
                "content": row["content"],
                "filename": row["filename"],
                "score": float(row["score"]),
            }
            for row in rows
        ]

    def fetch_all(self) -> list[dict]:
        """读出全部 chunk（不带向量）。CLI 遍历与回填脚本使用。"""
        sql = f"""
            SELECT c.id, c.source, c.page, c.chunk_index, c.content,
                   d.filename AS filename
            FROM {TABLE_NAME} c
            LEFT JOIN documents d ON d.id = c.document_id
            ORDER BY c.id
        """
        with self._reader() as conn:
            rows = conn.execute(sql).fetchall()

        return [
            {
                "id": row["id"],
                "source": row["source"],
                "page": row["page"],
                "chunk_index": row["chunk_index"],
                "content": row["content"],
            }
            for row in rows
        ]

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------

    def count(self) -> int:
        with self._reader() as conn:
            row = conn.execute(f"SELECT count(*) AS n FROM {TABLE_NAME}").fetchone()
        return row["n"]


    # ------------------------------------------------------------------
    # 按文档维度写入（M1：documents 表引入 document_id 之后）
    # ------------------------------------------------------------------
    @contextmanager
    def _using(self, conn=None):
        """复用外部事务连接；未传入则自开连接。"""
        if conn is not None:
            yield conn
            return
        with self._connect() as own:
            yield own

    def insert_for_document(
        self,
        document_id: int,
        source: str,
        rows: list[dict],
        vectors: list[list[float]],
        conn=None,
    ) -> int:
        """给某个文档批量写入片段；rows 元素形如 {"text","page","chunk_index"}。"""
        if len(rows) != len(vectors):
            raise ValueError(f"片段数({len(rows)})与向量数({len(vectors)})不一致")
        if not rows:
            return 0
        params = [
            (
                document_id,
                source,
                None if row.get("page") is None else str(row["page"]),
                row["chunk_index"],
                row["text"],
                _to_vector_literal(vector),
                # 分词在应用层做（jieba），PG 侧只按空白切 + 小写化（simple 配置）。
                # 索引与查询必须同一套规则，见 rag/keyword_search.tokenize_query。
                " ".join(tokenize(row["text"])),
            )
            for row, vector in zip(rows, vectors)
        ]
        sql = f"""
            INSERT INTO {TABLE_NAME}
                (document_id, source, page, chunk_index, content, embedding, content_tsv)
            VALUES (%s, %s, %s, %s, %s, %s::vector, to_tsvector('simple', %s))
        """
        with self._using(conn) as c:
            with c.cursor() as cur:
                cur.executemany(sql, params)
        return len(params)

    def delete_by_document(self, document_id: int, conn=None) -> int:
        """删除某文档的全部片段（BR-3：先删片段，再删文档行）。"""
        with self._using(conn) as c:
            cur = c.execute(f"DELETE FROM {TABLE_NAME} WHERE document_id = %s", (document_id,))
            return cur.rowcount

    # ------------------------------------------------------------------
    # 按文档维度只读（文档预览，FR-8）
    # ------------------------------------------------------------------

    def count_for_document(self, document_id: int, conn=None) -> int:
        with self._using(conn) as c:
            row = c.execute(
                f"SELECT count(*) AS n FROM {TABLE_NAME} WHERE document_id = %s", (document_id,)
            ).fetchone()
        # 连接池给的是 dict_row，自开连接是元组行
        return row["n"] if isinstance(row, dict) else row[0]

    def list_for_document(
        self, document_id: int, offset: int = 0, limit: int = 20, conn=None
    ) -> list[dict]:
        """按 chunk_index 顺序分页读片段（不带向量，预览用）。"""
        sql = f"""
            SELECT id, chunk_index, page, content
            FROM {TABLE_NAME}
            WHERE document_id = %s
            ORDER BY chunk_index
            OFFSET %s LIMIT %s
        """
        with self._using(conn) as c:
            rows = c.execute(sql, (document_id, offset, limit)).fetchall()
        # 外部传入的连接来自连接池（dict_row），自开的连接是元组行——两种都要吃得下
        def pick(row, idx, key):
            return row[key] if isinstance(row, dict) else row[idx]

        return [
            {
                "id": pick(row, 0, "id"),
                "chunk_index": pick(row, 1, "chunk_index"),
                "page": pick(row, 2, "page"),
                "content": pick(row, 3, "content"),
            }
            for row in rows
        ]
