"""回填 rag_chunks.content_tsv（V003 迁移后的存量数据）。

用法（项目根目录执行，连接串从 RAG_DB_DSN 读）：
    python scripts/backfill_tsv.py          # 只填 content_tsv IS NULL 的行，可重复执行
    python scripts/backfill_tsv.py --all    # 强制全量重算（改过分词规则时用）

分词复用 server.rag.keyword_search.tokenize —— 必须和写入路径同一套规则，
否则同义词在索引和查询侧会切得不一样，导致检索不到。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg

from server.rag.keyword_search import tokenize


def main() -> int:
    dsn = os.getenv("RAG_DB_DSN")
    if not dsn:
        print("未设置 RAG_DB_DSN", file=sys.stderr)
        return 1

    force_all = "--all" in sys.argv
    where = "" if force_all else "WHERE content_tsv IS NULL"

    with psycopg.connect(dsn) as conn:
        rows = conn.execute(f"SELECT id, content FROM rag_chunks {where} ORDER BY id").fetchall()
        total = conn.execute("SELECT count(*) FROM rag_chunks").fetchone()[0]
        print(f"库中共 {total} 块，本次待回填 {len(rows)} 块{'（全量重算）' if force_all else ''}")
        if not rows:
            print("无需回填")
            return 0

        params = [(" ".join(tokenize(content or "")), chunk_id) for chunk_id, content in rows]
        with conn.cursor() as cur:
            cur.executemany(
                "UPDATE rag_chunks SET content_tsv = to_tsvector('simple', %s) WHERE id = %s",
                params,
            )
        conn.commit()

        left = conn.execute(
            "SELECT count(*) FROM rag_chunks WHERE content_tsv IS NULL"
        ).fetchone()[0]
        empty = conn.execute(
            "SELECT count(*) FROM rag_chunks WHERE content_tsv IS NOT NULL "
            "AND content_tsv = ''::tsvector"
        ).fetchone()[0]
        print(f"完成：已填 {len(rows)} 块；仍为空 {left} 块；分词后无词 {empty} 块")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
