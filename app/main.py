"""FastAPI 项目启动入口：统一挂载所有路由

职责：
1. 创建 FastAPI 应用，配置 CORS（允许 Streamlit 前端跨域）
2. 挂载 RAG 求职匹配模块与简历评分 Agent 模块路由
3. 全局异常处理（统一响应格式，禁止返回堆栈信息）
4. 健康检查接口（服务 / Ollama / 向量库 / MySQL 状态）
5. 启动时初始化数据目录与数据库表

启动方式：
E:\简历项目\.venv\Scripts\python.exe -m app.main
    python app/main.py
    或 uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import APP_CORS_ORIGINS, ensure_data_dirs
from app.dao.chroma_db import ChromaDB
from app.dao.mysql_db import MySQLDB
from app.modules.agent_module import api as agent_api
from app.modules.rag_module import api as rag_api
from app.utils.common_tools import AppError, error_response, get_logger, success_response
from app.utils.ollama_client import OllamaClient

logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化数据目录与数据库表"""
    ensure_data_dirs()
    try:
        MySQLDB().init_tables()
        logger.info("MySQL 数据表初始化完成")
    except Exception as exc:
        logger.warning("MySQL 初始化失败（请检查 .env 数据库配置）：%s", exc)
    try:
        ChromaDB().get_collection()
        logger.info("Chroma 向量库初始化完成")
    except Exception as exc:
        logger.warning("Chroma 初始化失败：%s", exc)
    yield
    logger.info("服务已关闭")


app = FastAPI(
    title="求职匹配 AI 系统",
    description=(
        "基于本地 Ollama 的 RAG 求职匹配 + 简历评分优化 Agent 系统。"
        "全本地私有化部署，无任何云端大模型 API。"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS：允许 Streamlit 前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=APP_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载业务模块路由
app.include_router(rag_api.router)
app.include_router(agent_api.router)


# ============ 全局异常处理 ============


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    """业务异常：HTTP 200 + body.code 统一响应"""
    return JSONResponse(status_code=200, content=error_response(exc.message, code=exc.code))


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """参数校验失败：返回 400，标注错误字段与原因（英文 pydantic 消息中文化）"""
    errors = exc.errors()
    detail = errors[0] if errors else {}
    loc = ".".join(str(x) for x in detail.get("loc", [])).replace("body.", "")
    raw_msg = detail.get("msg", "格式错误")
    # 常见 pydantic 消息中文化
    cn_map = {
        "String should have at least": "内容过短",
        "String should have at most": "内容过长",
        "Input should be a valid integer": "请输入整数",
        "Input should be greater than or equal to": "数值过小",
        "Field required": "缺少必填字段",
        "Input should be a valid dictionary or instance of": "参数格式错误",
    }
    cn_msg = raw_msg
    for en, cn in cn_map.items():
        if en in raw_msg:
            cn_msg = cn
            break
    msg = f"参数 {loc} 校验失败:{cn_msg}"
    return JSONResponse(status_code=200, content=error_response(msg, code=400))


@app.exception_handler(Exception)
async def global_error_handler(request: Request, exc: Exception):
    """兜底异常：记录日志，返回统一错误响应，不透传堆栈"""
    logger.error("未捕获异常 url=%s error=%s", request.url, exc, exc_info=True)
    return JSONResponse(status_code=200, content=error_response("服务内部错误", code=500))


# ============ 健康检查 ============


@app.get("/api/health", tags=["系统"])
async def health_check():
    """健康检查：返回服务 / Ollama / 向量库 / MySQL 状态"""
    ollama_ok = OllamaClient().check_health()
    chroma_ok = ChromaDB().check_health()
    mysql_ok = MySQLDB().check_health()
    data = {
        "service": "ok",
        "ollama": "ok" if ollama_ok else "error",
        "chroma": "ok" if chroma_ok else "error",
        "mysql": "ok" if mysql_ok else "error",
        "models": OllamaClient().list_models() if ollama_ok else [],
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return success_response(data)


@app.get("/", tags=["系统"])
async def root():
    """服务根路径：返回系统信息"""
    return success_response(
        {"name": "求职匹配 AI 系统", "docs": "/docs", "health": "/api/health"},
        message="服务运行中",
    )


if __name__ == "__main__":
    import uvicorn

    from app.config import APP_HOST, APP_PORT

    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
