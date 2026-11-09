-- V003 回滚：删除全文索引与列
-- 注意：回滚后代码需同步切回 bm25s 实现，否则关键词检索会报列不存在。
DROP INDEX IF EXISTS idx_rag_chunks_tsv;
ALTER TABLE rag_chunks DROP COLUMN IF EXISTS content_tsv;
