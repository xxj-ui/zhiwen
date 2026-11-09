"""集中配置（NFR-6）

所有可调参数只在这里出现：检索/切分/上传上限/超时/连接池。
本模块**不导入**任何 Web 或数据库库，保持可被 CLI 复用。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# 本文件在 server/core/config.py → 上溯 3 层才是项目根（server/core → server → 根）
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 目录一挪就会静默指错地方（上传文件会写进 server/ 里），启动即失败更安全
if not (PROJECT_ROOT / "requirements.txt").exists():
    raise RuntimeError(f"PROJECT_ROOT 解析错误：{PROJECT_ROOT} 下没有 requirements.txt")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw in (None, "") else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw in (None, "") else float(raw)


@dataclass(frozen=True)
class Settings:
    """运行期配置。凡是"魔法数字"都应该是这里的字段。"""

    db_dsn: str

    # 服务（SRS 2.3 / D-8：只监听本机）
    host: str = "127.0.0.1"
    port: int = 8000

    # 数据库连接池（02 S-09 / 06 A4）
    db_pool_min: int = 1
    db_pool_max: int = 4
    db_pool_timeout: float = 5.0

    # 多轮上下文（SRS FR-9 / 10.多轮会话设计 §6.1）
    history_max_turns: int = 5
    history_char_budget: int = 4000

    # 检索（SRS 5.2 / NFR-6）
    recall_k: int = 20
    top_k: int = 5
    rrf_k: int = 60

    # 切分（SRS FR-1）
    chunk_size: int = 500
    chunk_overlap: int = 100
    chunk_min_size: int = 80

    # 上传（SRS BR-8 / 04 V-8）
    max_file_size: int = 50 * 1024 * 1024
    max_request_size: int = 55 * 1024 * 1024
    max_files_per_request: int = 20  # SRS FR-1 / BR-8：单请求 1~20 个文件，超量整批拒绝
    upload_dir: Path = PROJECT_ROOT / "data" / "uploads"

    # 原文件存储后端（02 §4.2 / 06 B14）
    #   local：写 upload_dir，stored_path 为相对项目根路径
    #   minio：先落 spool_dir 再上送对象存储，stored_path 为 s3://bucket/key
    storage_backend: str = "local"
    spool_dir: Path = PROJECT_ROOT / "data" / "tmp"
    s3_endpoint: str = ""
    s3_bucket: str = "zhiwen"
    s3_access_key: str = ""
    s3_secret_key: str = ""
    s3_prefix: str = "documents"
    s3_region: str = "us-east-1"

    # 大模型（04 §3.7 / 06 A1）
    llm_model: str = "deepseek-chat"
    llm_timeout_first_byte: float = 15.0
    llm_timeout_total: float = 60.0

    # 健康检查
    # 与 db_pool_timeout 取同一量级：数据库重启/重连的那几秒里，池还在补连接，
    # 探针若用 2 s 会抢在池恢复之前判 error，出现"health 说挂但请求能通"的窗口。
    # 5 s 仍满足 04 §4.8 注 3「秒级返回」。
    health_probe_timeout: float = 5.0

    # 启动预热（M8）：把首次 import / 模型加载挪到启动后台，
    # 否则这三笔一次性成本全压在第一问上（实测冷启动约 105 s）。可用 RAG_WARMUP=0 关掉。
    warmup_enabled: bool = True

    # 嵌入模型（SRS 2.4.4）
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_dim: int = 512

    # 扫描件 OCR（FR-1 解析增强）：PDF 某页抽不到文字时，渲染成图走 RapidOCR
    # 实测（本机 CPU）：引擎初始化 0.6 s、120 dpi 首屏 2.9 s、之后 1.1~1.6 s/页、字符相似度 0.99
    # 用 RAG_OCR=0 可整体关闭；RAG_OCR_MAX_PAGES>0 时限制最多 OCR 多少页，防超长扫描件拖死导入
    ocr_enabled: bool = True
    ocr_dpi: int = 120
    ocr_min_chars: int = 10
    ocr_max_pages: int = 0


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def load_settings() -> Settings:
    """从环境变量装配配置；缺 RAG_DB_DSN 直接抛错（SRS 2.4.3 / P-3）。"""
    dsn = os.getenv("RAG_DB_DSN")
    if not dsn:
        raise RuntimeError(
            "未配置数据库连接串：请设置环境变量 RAG_DB_DSN，"
            "形如 postgresql://<user>:***@127.0.0.1:54328/zhiwen"
        )

    # 存储后端：minio 时缺凭证直接启动失败，不要等到第一次上传才报错
    backend = (os.getenv("RAG_STORAGE_BACKEND") or "local").strip().lower()
    s3_endpoint = (os.getenv("RAG_S3_ENDPOINT") or "").strip()
    s3_bucket = (os.getenv("RAG_S3_BUCKET") or "zhiwen").strip()
    s3_access_key = (os.getenv("RAG_S3_ACCESS_KEY") or "").strip()
    s3_secret_key = (os.getenv("RAG_S3_SECRET_KEY") or "").strip()
    if backend == "minio":
        missing = [
            name
            for name, value in (
                ("RAG_S3_ENDPOINT", s3_endpoint),
                ("RAG_S3_BUCKET", s3_bucket),
                ("RAG_S3_ACCESS_KEY", s3_access_key),
                ("RAG_S3_SECRET_KEY", s3_secret_key),
            )
            if not value
        ]
        if missing:
            raise RuntimeError("RAG_STORAGE_BACKEND=minio 时以下环境变量必填：" + "、".join(missing))

    return Settings(
        db_dsn=dsn,
        port=_env_int("RAG_PORT", 8000),
        db_pool_min=_env_int("RAG_DB_POOL_MIN", 1),
        db_pool_max=_env_int("RAG_DB_POOL_MAX", 4),
        db_pool_timeout=_env_float("RAG_DB_POOL_TIMEOUT", 5.0),
        history_max_turns=_env_int("RAG_HISTORY_MAX_TURNS", 5),
        history_char_budget=_env_int("RAG_HISTORY_CHAR_BUDGET", 4000),
        llm_timeout_first_byte=_env_float("RAG_LLM_TIMEOUT_FIRST_BYTE", 15.0),
        llm_timeout_total=_env_float("RAG_LLM_TIMEOUT_TOTAL", 60.0),
        warmup_enabled=_env_bool("RAG_WARMUP", True),
        storage_backend=backend,
        s3_endpoint=s3_endpoint,
        s3_bucket=s3_bucket,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
        s3_prefix=(os.getenv("RAG_S3_PREFIX") or "documents").strip("/"),
        s3_region=(os.getenv("RAG_S3_REGION") or "us-east-1").strip(),
        ocr_enabled=_env_bool("RAG_OCR", True),
        ocr_dpi=_env_int("RAG_OCR_DPI", 120),
        ocr_min_chars=_env_int("RAG_OCR_MIN_CHARS", 10),
        ocr_max_pages=_env_int("RAG_OCR_MAX_PAGES", 0),
    )
