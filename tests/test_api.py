"""API 层集成测试(使用 FastAPI TestClient + mock 外部服务)

注意:项目统一响应设计 = HTTP 200 + body{code: 400/500}(不是 HTTP 状态码)。
外部服务(Ollama/Chroma/MySQL)全部 mock,测试无需真实后端。
"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture()
def client(monkeypatch):
    """构建 TestClient,替换外部服务依赖"""
    import app.main as main
    from app.utils.ollama_client import OllamaClient
    from app.dao.chroma_db import ChromaDB
    from app.dao.mysql_db import MySQLDB

    monkeypatch.setattr(OllamaClient, "check_health", lambda self: True)
    monkeypatch.setattr(OllamaClient, "list_models", lambda self: ["qwen2:7b"])
    monkeypatch.setattr(ChromaDB, "check_health", lambda self: True)
    monkeypatch.setattr(MySQLDB, "check_health", lambda self: True)

    # 替换 RAG service 的关键方法(不连真实库)
    from app.modules.rag_module import api as rag_api
    rag_api.service.chat = lambda query, top_k=3: {
        "answer": "测试回答", "sources": [{"file_name": "x.pdf", "score": 0.9}]
    }
    rag_api.service.query_jobs = lambda nl_query, limit=20: {"items": [], "count": 0}

    # 替换 Agent service 的 evaluate(签名与真实一致,不调 LLM)
    from app.modules.agent_module import api as agent_api
    agent_api.service.evaluate = lambda content, filename, target_job, session_id=None, jd_text="": {
        "total_score": 80, "dimensions": {}, "weaknesses": [], "suggestions": [],
        "mode": "通用评估", "pass_line": 80, "pass": True,
    }

    return TestClient(main.app)


def _body(resp):
    return resp.json()


# ---------- 健康检查 ----------

def test_health_ok(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = _body(resp)
    assert body["code"] == 200
    assert body["data"]["service"] == "ok"


# ---------- 参数校验(统一响应:HTTP 200 + body.code 400) ----------

def test_chat_empty_query_rejected(client):
    resp = client.post("/api/rag/chat", json={"query": ""})
    assert _body(resp)["code"] == 400  # 空 query → 业务 400


def test_chat_long_query_rejected(client):
    resp = client.post("/api/rag/chat", json={"query": "x" * 501})
    assert _body(resp)["code"] == 400  # 超长 → 业务 400


def test_match_resume_too_short(client):
    """简历文本过短(<10字)校验拒绝"""
    resp = client.post(
        "/api/rag/match/by-jd",
        json={"resume_text": "短", "jd_text": "JD JD JD JD JD JD JD JD JD JD"},
    )
    assert _body(resp)["code"] == 400


def test_agent_evaluate_requires_file(client):
    """Agent evaluate 未传文件 → 参数校验失败(非 200 业务成功)"""
    resp = client.post("/api/agent/evaluate", data={"target_job": "Java"})
    # 缺 UploadFile 会触发 422 或 400,但不会是业务 200
    assert resp.status_code in (400, 422) or _body(resp)["code"] in (400, 422)


# ---------- 路由可达(mock 后返回固定值) ----------

def test_rag_chat_route(client):
    resp = client.post("/api/rag/chat", json={"query": "什么是Java", "top_k": 3})
    assert _body(resp)["code"] == 200
    assert "测试回答" in _body(resp)["data"]["answer"]


def test_agent_evaluate_route(client):
    """Agent 评估 multipart 上传,mock 后返回固定结果"""
    resp = client.post(
        "/api/agent/evaluate",
        data={"target_job": "Java", "jd_text": "JD JD JD JD JD JD JD JD JD JD JD JD"},
        files={"file": ("resume.txt", "个人简历 张三 本科 Java".encode("utf-8"), "text/plain")},
    )
    assert _body(resp)["code"] == 200
    assert _body(resp)["data"]["total_score"] == 80


def test_404_not_found(client):
    """不存在路由返回 404"""
    resp = client.get("/api/not-exist")
    assert resp.status_code == 404
