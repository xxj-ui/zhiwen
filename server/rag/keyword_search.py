"""第 5 步：关键词检索 —— 分词与查询式构造（打分交给数据库）

职责两件：中文怎么切（jieba），以及怎么把切好的词变成 PG 查询式。
倒排与打分在 PostgreSQL 侧完成（tsvector + GIN + ts_rank_cd），见 V003 迁移。

改造背景（V003，2026-11）：
    原实现用 bm25s 在进程内建「doc × term 稀疏矩阵」，两个代价随块数线性增长——
    每次上传全量重建（0.73 ms/块）、每次查询全量打分（3.5 µs/(块·14 词)）。
    换成 GIN 倒排后：写入由索引自动维护（不再重建），查询只扫命中的 posting list。

两个必须记住的坑（都踩过）：

  1. `plainto_tsquery` 生成的是 **AND** 语义，长问题会退化成"所有词都得出现"，
     实测「Java 线程池的核心参数有哪些？」只命中 1 条且是文档标题页。
     BM25 是 OR 语义（任意词命中即计分），所以这里手工拼 `'a' | 'b' | 'c'`。

  2. jieba 会把「的 / 有 / 哪些 / ？」这类词一并切出来。BM25 靠 IDF 自动压制它们
     （全库都有的词 IDF≈0），但 ts_rank_cd 不做这层加权，必须由停用词表挡掉，
     否则「的」几乎命中所有块，把真正的关键词淹没。

为什么用 `simple` 配置：中文分词已在应用层做完，PG 只需按空白切、英文小写化。

为什么不用 lcut_for_search（搜索模式）：它会产出子词（"计算机网络" → 计算/算机/网络/…），
实测这些子词会去匹配无关内容。代价是查"计算机"匹配不到"计算机网络"，
但语义召回由稠密向量负责。
"""
import re

_jieba = None

# 只挡虚词与标点，不挡实词 —— 单字中文（池、树、锁）必须保留
_STOPWORDS = frozenset(
    """的 了 是 在 和 与 或 有 就 都 也 很 更 最 这 那 哪 哪些 什么 怎么 如何 为什么
    是否 可以 需要 应该 以及 但是 因为 所以 如果 虽然 一个 一些 一种 我们 你们 他们
    我 你 他 她 它 其 之 而 则 被 把 让 给 从 对 为 以 于 由 等 吧 吗 呢 啊""".split()
)

_HAS_WORD = re.compile(r"[\w\u4e00-\u9fff]")


def _get_jieba():
    """延迟导入 jieba，并屏蔽它首次加载词典时的日志。"""
    global _jieba
    if _jieba is None:
        import logging
        import warnings

        # jieba 内部还在用 pkg_resources，新版 setuptools 会告警，和用法无关。
        warnings.filterwarnings(
            "ignore", message="pkg_resources is deprecated", category=UserWarning
        )

        import jieba

        jieba.setLogLevel(logging.ERROR)
        _jieba = jieba
    return _jieba


def tokenize(text: str) -> list[str]:
    """把文本切成词（精确模式 lcut），英文统一转小写。"""
    if not text or not text.strip():
        return []
    return [t for t in _get_jieba().lcut(text.lower()) if t.strip()]


def to_tsquery_text(tokens: list[str]) -> str:
    """分词结果 → `to_tsquery('simple', ...)` 的 OR 表达式：`'a' | 'b' | 'c'`。

    词用单引号包裹（一个词 = 一个 lexeme），内部的单引号双写转义。
    过滤顺序：空 → 停用词 → 纯标点 → **去重**（保持首次出现顺序）。
    全被过滤掉时返回空串，调用方应视为"无有效查询词"。

    为什么要去重：问题里重复的词（「三次握手为什么是三次」→ 三次 ×2）会让 tsquery
    出现重复 lexeme，`ts_rank_cd` 对同一词重复计数，分数被无谓抬高；语义上也没有增量。

    写入侧用的是 `to_tsvector('simple', 空格拼接的原词)`,两侧必须同源，
    否则切法不一致会静默检索不到。
    """
    parts = []
    seen: set[str] = set()
    for t in tokens:
        if not t or t in _STOPWORDS:
            continue
        if not _HAS_WORD.search(t):
            continue
        if t in seen:
            continue
        seen.add(t)
        parts.append("'" + t.replace("'", "''") + "'")
    return " | ".join(parts)


def tokenize_query(question: str) -> str:
    """问题 → 分词 → PG 查询表达式（`to_tsquery('simple', ...)` 的入参）。"""
    return to_tsquery_text(tokenize(question))
