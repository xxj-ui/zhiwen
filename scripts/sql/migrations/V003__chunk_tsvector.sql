-- v003: 关键词检索改用 PostgreSQL 全文索引（tsvector + GIN）
-- depends: V002__multiturn_conversations
-- 设计：docs/design/02.整体架构设计.md、06.遗留问题清单.md §17 B20
--
-- 为什么换：
--   原实现用 bm25s 在进程内建「doc × term 稀疏矩阵」，两个代价随块数线性增长——
--   每次上传全量重建（0.73 ms/块）、每次查询全量打分（3.5 µs/(块·14词)）。
--   改成数据库倒排后：写入由索引自动维护（不再重建），查询只扫命中的 posting。
--
-- 分词在哪做：
--   不用 zhparser / pg_jieba 扩展（要编译，容器里没有），改由应用层用现有 jieba
--   分词后以空格拼接，PG 侧用内置的 `simple` 配置（按空白切、小写化）。
--   好处：分词规则与改造前完全一致（同为 jieba 精确模式），且零新依赖。
--
-- 注意：
--   content_tsv 是普通列而非生成列 —— 生成列要求表达式 IMMUTABLE，而分词在应用层，
--   所以由写入方显式填充；存量数据见 scripts/backfill_tsv.py。

ALTER TABLE rag_chunks ADD COLUMN IF NOT EXISTS content_tsv tsvector;

-- GIN 是倒排索引：term → 出现的行，查询 O(命中) 而非 O(全表)
CREATE INDEX IF NOT EXISTS idx_rag_chunks_tsv
    ON rag_chunks USING GIN (content_tsv);
