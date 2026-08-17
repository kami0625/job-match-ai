"""简历评分优化 Agent 模块 - 接口层

职责：仅负责参数接收、校验、结果封装，禁止编写业务逻辑与底层调用。
业务逻辑委托给 app/modules/agent_module/service.py 的 ResumeAgentService。
统一响应格式：{"code": 200/400/404/500, "message": "...", "data": {...}}

接口清单：
- POST /api/agent/evaluate        简历评估入口（简历文件 + 目标岗位 -> 结构化打分结果）
- POST /api/agent/evaluate/stream 简历评估（SSE 流式，展示 Agent 思考过程）
- POST /api/agent/clear           会话清除（指定 session_id 或全部）
- POST /api/agent/suggestion      单独获取优化建议（可选）
- POST /api/agent/parse           简历解析（结构化提取，兼容板块 A 简历匹配页）
- POST /api/agent/score           简历-JD 匹配打分（规则）
"""

import json
from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.modules.agent_module.service import ResumeAgentService
from app.utils.common_tools import error_response, success_response

router = APIRouter(prefix="/api/agent", tags=["简历评分 Agent"])

service = ResumeAgentService()


class ClearRequest(BaseModel):
    """会话清除请求体"""

    session_id: Optional[str] = Field(None, description="指定清除的会话ID")
    all: bool = Field(False, description="是否清空全部会话")


class SuggestionRequest(BaseModel):
    """单独优化建议请求体"""

    resume_text: str = Field(..., min_length=10, max_length=20000, description="简历纯文本")
    target_job: Optional[str] = Field(None, max_length=100, description="目标岗位名称")
    score_detail: dict = Field(default_factory=dict, description="评分详情（可选）")


class ScoreRequest(BaseModel):
    """简历打分请求体"""

    resume_text: str = Field(..., min_length=10, max_length=20000, description="简历纯文本")
    job_desc: Optional[str] = Field(None, max_length=5000, description="岗位JD，为空则通用评分")


def _agent_error(exc: Exception):
    """统一 Agent 模块错误响应：汉化 + 不泄露内部路径 + 客户端错误归类为 400"""
    msg = str(getattr(exc, "message", exc) or "")
    # 文件解析类错误 -> 400 友好提示
    if "未提取到文本内容" in msg:
        return error_response("文件中未提取到可识别的文字内容,请上传包含文字层的 PDF/DOCX(扫描件或纯图片 PDF 不支持)", code=400)
    if "解析失败" in msg or "Cannot open" in msg or "not a PDF" in msg or "Failed to open" in msg:
        return error_response("简历文件解析失败,请检查文件是否完整或尝试重新保存后再上传", code=400)
    if "仅支持" in msg or "格式" in msg:
        return error_response("仅支持 PDF / DOCX 格式简历文件", code=400)
    if "过少" in msg or "不是有效简历" in msg or "内容过少" in msg:
        return error_response(msg, code=400)
    # 后端服务类错误 -> 通用提示
    if "Ollama" in msg or "11434" in msg or "connection" in msg.lower() or "timeout" in msg.lower():
        return error_response("本地大模型服务暂不可用,请确认 Ollama 已启动后重试", code=503)
    if "RAG" in msg or "rag" in msg or "岗位" in msg:
        return error_response("岗位检索服务异常,请稍后重试,或手动粘贴目标 JD 文本后重新评估", code=503)
    # 兜底：不暴露堆栈/路径
    code = getattr(exc, "code", 500)
    return error_response("评估失败,请稍后重试", code=code)


@router.post("/evaluate", summary="简历评估入口（ReAct Agent 多轮工具调用）")
async def evaluate(
    file: UploadFile = File(..., description="简历 PDF/DOCX"),
    target_job: str = Form("", description="目标岗位名称，Agent 自动调用 RAG 系统检索岗位 JD"),
    jd_text: str = Form("", description="用户上传的 JD 文本（可选；提供则跳过搜索直接用）"),
    session_id: str = Form("", description="会话ID（可选，复用中间结果）"),
):
    """驱动 ReAct Agent 完整评估链路，返回结构化打分结果（总分/分项/弱点/建议/修改示例）"""
    content = await file.read()
    try:
        data = service.evaluate(
            content,
            file.filename or "resume",
            target_job or "",
            session_id=session_id or None,
            jd_text=jd_text or "",
        )
        return success_response(data)
    except Exception as exc:
        return _agent_error(exc)


@router.post("/evaluate/stream", summary="简历评估（SSE 流式，展示 Agent 思考过程）")
async def evaluate_stream(
    file: UploadFile = File(..., description="简历 PDF/DOCX"),
    target_job: str = Form("", description="目标岗位名称"),
    jd_text: str = Form("", description="用户上传的 JD 文本（可选；提供则跳过搜索直接用）"),
    session_id: str = Form("", description="会话ID（可选）"),
    temperature: Optional[float] = Form(None, description="LLM 温度(0-2),留空用默认"),
    max_iterations: Optional[int] = Form(None, description="最大推理轮数(1-10),留空用默认"),
):
    """流式事件：start / step(thought,action,observation) / fallback / result / error"""
    content = await file.read()

    def event_stream():
        for event in service.evaluate_stream(
            content,
            file.filename or "resume",
            target_job or "",
            session_id=session_id or None,
            jd_text=jd_text or "",
            temperature=temperature,
            max_iterations=max_iterations,
        ):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/clear", summary="Agent 会话清除")
async def clear_session(request: ClearRequest):
    """清空指定会话或全部会话上下文（含中间打分结果缓存）"""
    try:
        if request.all:
            data = service.clear_all_sessions()
            return success_response(data, message="已清空全部会话")
        if request.session_id:
            data = service.clear_session(request.session_id)
            return success_response(data, message="会话已清除" if data["cleared"] else "会话不存在")
        return error_response("请提供 session_id 或设置 all=true", code=400)
    except Exception as exc:
        return error_response(
            str(getattr(exc, "message", exc)),
            code=getattr(exc, "code", 500),
        )


@router.post("/suggestion", summary="单独获取优化建议")
async def get_suggestion(request: SuggestionRequest):
    """针对简历生成逐段优化建议与修改示例（可选接口）"""
    try:
        data = service.get_suggestion(
            request.resume_text,
            target_job=request.target_job,
            score_detail=request.score_detail,
        )
        return success_response(data)
    except Exception as exc:
        return error_response(
            str(getattr(exc, "message", exc)),
            code=getattr(exc, "code", 500),
        )


@router.post("/parse", summary="简历上传解析（结构化提取）")
async def parse_resume(file: UploadFile = File(..., description="简历 PDF/DOCX")):
    """解析简历文件，输出结构化简历信息（个人信息/技能/项目经历/教育经历）"""
    content = await file.read()
    try:
        data = service.parse_resume(content, file.filename or "resume")
        # 内容过短校验:可能不是有效简历(乱码/非简历文档)
        raw_len = len(data.get("raw_text") or "")
        has_struct = bool(data.get("personal_info") or data.get("skills") or data.get("projects"))
        if raw_len < 100 or (raw_len < 500 and not has_struct):
            return error_response(
                "解析后内容过短或未识别出简历特征(姓名/技能/项目经历),请确认上传的是完整简历 PDF/DOCX,而非扫描件或图片",
                code=400,
            )
        return success_response(data)
    except Exception as exc:
        msg = str(getattr(exc, "message", exc))
        if "未提取到文本内容" in msg:
            return error_response("文件中未提取到可识别的文字内容,请上传包含文字层的 PDF/DOCX(扫描件或纯图片不支持)", code=400)
        return error_response(
            str(getattr(exc, "message", exc)),
            code=getattr(exc, "code", 500),
        )


@router.post("/score", summary="简历-JD 多维度匹配打分（规则）")
async def score(request: ScoreRequest):
    """四维打分（技能匹配/项目经验/学历要求/业务关键词），规则计算稳定可复现"""
    try:
        data = service.score(request.resume_text, request.job_desc)
        return success_response(data)
    except Exception as exc:
        return error_response(
            str(getattr(exc, "message", exc)),
            code=getattr(exc, "code", 500),
        )
