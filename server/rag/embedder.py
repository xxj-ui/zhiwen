"""第 3 步：向量化（只负责 文字 → 向量）

使用本地缓存的 BAAI/bge-small-zh-v1.5（512 维），不联网。
模型懒加载：第一次调用时才加载，之后复用同一个实例。
"""

import os
import threading

# 模型缓存位置由 HF_HOME 决定；未设置时回退到 HuggingFace 标准缓存目录。
# setdefault：如果系统里已经设了 HF_HOME，以系统值为准。
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
EMBED_DIM = 512

# BGE 官方建议：检索时给 query 加指令前缀，文档侧不加
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

# 单条编码允许的内部并行线程数（默认 4，可用 RAG_EMBED_THREADS 覆盖）。
#
# 为什么不用 torch 的默认值（本机 12）：查询路径是"一次一条短文本"，几 MFLOPs 的活
# 拆给 12 个线程，fork-join 的分活/汇合开销超过计算本身。更要命的是并发下——
# 检索走 asyncio.to_thread（默认池 20 worker），20 × 12 = 240 个线程抢 16 个核，
# 15 倍超订，实测吞吐从 44.8/s（2 线程）掉到 17.5/s（12 线程）。
# 4 线程 × 20 并发 = 80 线程（5 倍超订），实测落在无损区（8 并发下 1~12 线程
# 差异在 ±10% 噪声内，三轮中位数）。
#
# 这里用环境变量而不是进 core/config.py：rag/ 层刻意不依赖 core/（保持纯算法，
# 02 §2.4 的依赖方向），HF_HOME 也是同一处理方式。
#
# 查询编码的并发上限（默认 4，可用 RAG_EMBED_CONCURRENCY 覆盖）。
#
# 为什么需要它：检索跑在 asyncio.to_thread 的默认线程池里（20 worker），
# 20 个 encode 同时进来 = 20 × EMBED_THREADS 个计算线程。即使每任务只有 4 线程
# （80 线程 / 16 核，5 倍超订），CPU 仍被打满 —— 而**同一个进程里还跑着 asyncio
# 事件循环**，它负责接收 DeepSeek 的流式响应并转发给客户端。CPU 饱和时事件循环
# 抢不到时间片，表现为"LLM 首字节从 1.2s 涨到 2.7s"，看起来像 API 变慢。
#
# 4 × 4 = 16 线程 = 核数，正好喂满 CPU 又不打满，给事件循环留出余量。
# 只限制 embed_query（查询路径）；上传路径的 embed_documents 是批量编码，天然有
# batch_size 控制，且不在请求热路径上，不套这个闸。
EMBED_THREADS = int(os.getenv("RAG_EMBED_THREADS", "4"))
EMBED_CONCURRENCY = int(os.getenv("RAG_EMBED_CONCURRENCY", "4"))

_embed_slots = threading.BoundedSemaphore(max(1, EMBED_CONCURRENCY))

_model = None


def _get_model():
    """延迟加载模型。import 也放在函数里，避免没用到时拖慢导入。"""
    global _model
    if _model is None:
        import torch
        from sentence_transformers import SentenceTransformer

        # 必须在首次 encode 之前设置：torch 的 intra-op 线程数只在算子尚未
        # 初始化时生效，模型跑起来之后再改会被静默忽略。
        torch.set_num_threads(EMBED_THREADS)
        _model = SentenceTransformer(MODEL_NAME, local_files_only=True)
        # 模型最大输入 512 token，超出的部分会被截断（不会报错）
        _model.max_seq_length = 512
    return _model


def embed_documents(
    texts: list[str],
    batch_size: int = 32,
    show_progress: bool = False,
) -> list[list[float]]:
    """把一批文档块编码成向量。

    参数:
        texts:         文档块文本列表
        batch_size:    每批编码多少条（CPU 上 32 左右比较稳）
        show_progress: 是否显示进度条（文档多时建议开）

    返回:
        list[list[float]]，每条 512 维，已归一化（点积 = 余弦相似度）
    """
    if not texts:
        return []

    vectors = _get_model().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,   # 归一化：后续点积直接等于余弦相似度
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    """把用户问题编码成查询向量。

    和 embed_documents 的区别：问题前面会加 BGE 的检索指令，
    这样它和库里文档向量的匹配才能达到模型训练时的效果。
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("查询内容不能为空")

    # 闸门放在最外层：连模型懒加载一起串行，避免冷启动瞬间 20 个线程同时
    # 触发 _get_model() 抢着加载模型。
    with _embed_slots:
        vector = _get_model().encode(
            QUERY_INSTRUCTION + text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
    return vector.tolist()
