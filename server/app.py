"""应用装配（02 §4.2）：lifespan、异常、路由、日志、请求体上限。

骨架原则：只搭骨架与横切能力，业务在 service 层（当前全部 raise NotImplementedError）。
"""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from server.api.deps import get_container, set_container
from server.core import migrations_check
from server.core.config import PROJECT_ROOT
from server.core.errors import AppError, error_body
from server.core.logging import new_request_id, request_id_ctx, setup_logging
from server.services import warmup
from server.api.routes import chat, conversations, documents, health

logger = logging.getLogger("zhiwen.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    # P-3：RAG_DB_DSN 缺失即启动失败
    container = get_container()
    logger.info("startup: host=%s port=%s", container.settings.host, container.settings.port)
    # 03 §6.7 规则 4：只校验迁移版本并告警，不在启动时隐式改库
    migrations_check.warn_if_behind(
        container.settings.db_dsn, PROJECT_ROOT / "scripts" / "sql" / "migrations"
    )
    # M8 启动预热：**同步**做掉模型加载 / 解析库 import，
    # lifespan 返回前 uvicorn 不接管请求 → "能连上" 就等于 "能干活"。
    if container.settings.warmup_enabled:
        await asyncio.to_thread(warmup.run_blocking, container)
    try:
        yield
    finally:
        container.close()
        set_container(None)
        logger.info("shutdown: pool closed")


def create_app() -> FastAPI:
    app = FastAPI(
        title="知问 · 个人知识库问答",
        version="0.1.0-skeleton",
        description="本机单用户知识库问答 API（骨架版：接口已定义，实现待 M1/M2）",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def _request_context(request: Request, call_next):
        rid = new_request_id()
        token = request_id_ctx.set(rid)
        started = time.perf_counter()
        response = None
        try:
            # 04 V-8：读 body 之前按 Content-Length 拦截超大请求
            if request.url.path.endswith("/documents/upload"):
                limit = get_container().settings.max_request_size
                raw = request.headers.get("content-length")
                if raw and raw.isdigit() and int(raw) > limit:
                    response = JSONResponse(status_code=413, content=error_body("PAYLOAD_TOO_LARGE"))
                    return response
            response = await call_next(request)
            return response
        finally:
            cost = (time.perf_counter() - started) * 1000
            status = getattr(response, "status_code", "ERR")
            logger.info("%s %s -> %s %.1fms", request.method, request.url.path, status, cost)
            if response is not None:
                response.headers["X-Request-Id"] = rid
            request_id_ctx.reset(token)

    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError):
        return JSONResponse(status_code=exc.http_status, content=error_body(exc.code, exc.message))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(x) for x in first.get("loc", []) if x != "body")
        return JSONResponse(
            status_code=422,
            content=error_body("VALIDATION_ERROR", f"参数校验失败：{loc} {first.get('msg', '')}".strip()),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException):
        code = {404: "NOT_FOUND", 405: "VALIDATION_ERROR"}.get(exc.status_code, "INTERNAL_ERROR")
        return JSONResponse(status_code=exc.status_code, content=error_body(code, str(exc.detail)))

    @app.exception_handler(NotImplementedError)
    async def _not_implemented(_: Request, exc: NotImplementedError):
        return JSONResponse(status_code=501, content=error_body("NOT_IMPLEMENTED", str(exc) or None))

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        logger.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content=error_body("INTERNAL_ERROR"))

    for module in (documents, chat, conversations, health):
        app.include_router(module.router, prefix="/api")
    return app


app = create_app()
