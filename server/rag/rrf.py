"""第 6 步：多路召回融合（RRF）

把稠密检索和关键词检索的结果按「排名」融合，不看原始分数。

RRF（Reciprocal Rank Fusion）公式:

    score(d) = Σ 1 / (k + rank_i(d))
               i

    rank_i(d)  文档 d 在第 i 路里的名次（从 1 开始）
    k          平滑常数，通常取 60

为什么只用排名：余弦相似度是 0~1，关键词打分是无界的，两者量纲不同，
不能相加也不能取平均；排名是唯一可比的量。

k=60 的作用是把排名带来的差距压平：
    第 1 名  1/61 ≈ 0.01639
    第 10 名 1/70 ≈ 0.01429
    第 20 名 1/80 ≈ 0.01250
这样一路的第 1 名不会轻易压过另一路的第 2、3 名，
两路都命中的文档则能明显胜出。
"""

DEFAULT_K = 60


def rrf_fuse(
    named_results: list[tuple[str, list[dict]]],
    k: int = DEFAULT_K,
    top_k: int = 5,
) -> list[dict]:
    """按 RRF 融合多路结果。

    参数:
        named_results: [(路由名, 结果列表), ...]，例如
                       [("dense", dense_results), ("keyword", keyword_results)]
        k:             RRF 平滑常数
        top_k:         融合后保留多少条

    返回:
        [{"id", "source", "page", "chunk_index", "content",
          "rrf_score", "routes": [...]}, ...]，按 rrf_score 降序
    """
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")
    if k < 0:
        raise ValueError("k 不能为负数")

    scores: dict = {}
    docs: dict = {}
    routes: dict = {}

    for name, results in named_results:
        for rank, item in enumerate(results, 1):
            cid = item.get("id")
            if cid is None:          # 没有主键就没法对齐两路结果，直接跳过
                continue
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            docs.setdefault(cid, item)
            routes.setdefault(cid, []).append(name)

    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [
        {
            "id": cid,
            "source": docs[cid].get("source"),
            "filename": docs[cid].get("filename"),
            "page": docs[cid].get("page"),
            "chunk_index": docs[cid].get("chunk_index"),
            "content": docs[cid].get("content"),
            "rrf_score": score,
            "routes": routes[cid],
        }
        for cid, score in ordered[:top_k]
    ]
