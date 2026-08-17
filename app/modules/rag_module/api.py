"""RAG 求职匹配模块 - 接口层

职责：仅负责参数接收、校验、结果封装，禁止编写业务逻辑与底层调用。
所有业务逻辑委托给 app/modules/rag_module/service.py 的 RagService。
统一响应格式：{"code": 200/400/404/500, "message": "...", "data": {...}}
"""

import json
from typing import Optional

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.modules.rag_module.service import RagService
from app.utils.common_tools import error_response, success_response

router = APIRouter(prefix="/api/rag", tags=["RAG 求职匹配"])

service = RagService()


class ChatRequest(BaseModel):
    """RAG 智能问答请求体"""

    query: str = Field(..., min_length=1, max_length=500, description="用户问题")
    top_k: int = Field(3, ge=1, le=10, description="引用片段数量")


class Nl2SqlRequest(BaseModel):
    """自然语言岗位查询请求体"""

    query: str = Field(..., min_length=1, max_length=300, description="自然语言查询条件")
    limit: int = Field(20, ge=1, le=100, description="返回条数上限")


class MatchRequest(BaseModel):
    """简历匹配计算请求体（按 job_id）"""

    resume_text: str = Field(..., min_length=10, max_length=20000, description="简历纯文本")
    job_id: int = Field(..., gt=0, description="目标岗位ID")


class MatchByTargetRequest(BaseModel):
    """简历匹配请求体（按目标岗位名实时多源查 JD）

    用于 Tab4 实时匹配场景:用户输入目标岗位名,系统自动调外部 API 查 JD,跟简历比对
    """

    resume_text: str = Field(..., min_length=10, max_length=20000, description="简历纯文本")
    target_job: str = Field(..., min_length=1, max_length=100, description="目标岗位名称,如「Agent 应用开发」")
    city: str = Field("", max_length=50, description="城市（可选）")
    salary_min: int = Field(0, ge=0, le=200, description="最低薪资 K/月（严格大于，0=不限）")
    top_n: int = Field(5, ge=3, le=10, description="候选 JD 数")


class MatchByJDRequest(BaseModel):
    """简历匹配请求体（用户上传 JD 文本）

    用于 Boss/拉勾等场景:用户把看到的目标岗位 JD 复制粘贴进来,系统直接评分
    """
    resume_text: str = Field(..., min_length=10, max_length=20000, description="简历纯文本")
    jd_text: str = Field(..., min_length=20, max_length=10000, description="目标岗位 JD 文本")
    target_job: str = Field("", max_length=100, description="目标岗位名称（用于结果展示）")


@router.post("/upload", summary="文档上传入库")
async def upload_document(file: UploadFile = File(..., description="PDF/DOCX 文档")):
    """上传岗位说明书 / 行业知识文档，自动解析分块并写入向量库"""
    filename = file.filename or "unnamed"
    content = await file.read()
    try:
        data = service.upload_document(content, filename)
        return success_response(data, message="文档入库成功")
    except Exception as exc:
        # 友好错误信息(不暴露内部路径)
        msg = str(getattr(exc, "message", exc))
        if "PDF 解析失败" in msg or "解析失败" in msg or "上传文件" in msg:
            return error_response("简历文件解析失败,请检查文件是否完整或尝试重新保存后再上传", code=400)
        if "未提取到文本内容" in msg:
            return error_response("文件中未提取到可识别的文字内容,请上传包含文字层的 PDF/DOCX(扫描件或纯图片 PDF 不支持)", code=400)
        return error_response(msg, code=getattr(exc, "code", 500))


@router.post("/chat", summary="RAG 智能问答")
async def chat(request: ChatRequest):
    """混合检索 + 精排 + 本地大模型生成回答，返回引用来源"""
    try:
        data = service.chat(request.query, top_k=request.top_k)
        return success_response(data)
    except Exception as exc:
        return error_response(
            str(getattr(exc, "message", exc)),
            code=getattr(exc, "code", 500),
        )


@router.post("/chat/stream", summary="RAG 智能问答（流式输出，SSE）")
async def chat_stream(request: ChatRequest):
    """流式问答：SSE 事件流，事件类型 meta / source / delta / error / done

    前端可通过 requests stream=True 逐行消费 "data: {json}"。
    """

    def event_stream():
        for event in service.chat_stream(request.query, request.top_k):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/query", summary="自然语言查询岗位数据库（NL2SQL）")
async def query_jobs(request: Nl2SqlRequest):
    """大模型生成 SQL 并执行查询，返回结果列表与生成的 SQL"""
    try:
        data = service.query_jobs(request.query, limit=request.limit)
        return success_response(data)
    except Exception as exc:
        return error_response(
            str(getattr(exc, "message", exc)),
            code=getattr(exc, "code", 500),
        )


@router.post("/match", summary="简历与岗位匹配度计算（按 job_id）")
async def match_resume(request: MatchRequest):
    """多维度匹配评分，返回总分、分项得分与匹配图表（base64 PNG）"""
    try:
        data = service.match_resume(request.resume_text, request.job_id)
        return success_response(data)
    except Exception as exc:
        return error_response(
            str(getattr(exc, "message", exc)),
            code=getattr(exc, "code", 500),
        )


@router.post("/match/by-jd", summary="简历与用户上传 JD 文本精准匹配")
async def match_resume_by_jd(request: MatchByJDRequest):
    """用户粘 JD 原文 + 简历 → 直接评分(不依赖多源搜索)

    适用场景:用户在 Boss/拉勾/智联等网站看到具体岗位,把 JD 原文复制进来,
    想知道自己的简历是否匹配 + 差在哪里。返回总分、4 维度、调整建议(总分<85时)。
    """
    try:
        # 简历内容基础校验
        if len(request.resume_text.strip()) < 100:
            return error_response(
                "简历内容过短（少于100字）,可能不是有效简历。请检查是否完整复制,或尝试上传 PDF/DOCX 文件。",
                code=400,
            )
        data = service.match_resume_with_jd(
            request.resume_text, request.jd_text, request.target_job
        )
        return success_response(data)
    except Exception as exc:
        return error_response(
            str(getattr(exc, "message", exc)),
            code=getattr(exc, "code", 500),
        )


@router.post("/match/by-target", summary="简历与目标岗位实时多源匹配（按岗位名）")
async def match_resume_to_target(request: MatchByTargetRequest):
    """输入目标岗位名（如「Agent 应用开发」）,系统实时多源查该岗位 JD,自动比对简历打分。

    数据流:
      1. service.search_jobs 实时检索 N 个候选 JD(本地优先 + 腾讯公开招聘 + Adzuna 兜底)
      2. service._llm_match_score 跟简历逐个匹配打分
      3. 按总分排序返回 top 1 详细(top 3 摘要作对比)
    """
    try:
        data = service.match_resume_to_target(
            resume_text=request.resume_text,
            target_job=request.target_job,
            city=request.city,
            salary_min=request.salary_min,
            top_n=request.top_n,
        )
        return success_response(data)
    except Exception as exc:
        return error_response(
            str(getattr(exc, "message", exc)),
            code=getattr(exc, "code", 500),
        )


@router.get("/jobs", summary="岗位数据库列表")
async def list_jobs(limit: int = 20, offset: int = 0):
    """分页查询岗位列表（供前端表格展示与 Agent 检索 JD）"""
    try:
        data = service.list_jobs(limit=limit, offset=offset)
        return success_response(data)
    except Exception as exc:
        return error_response(
            str(getattr(exc, "message", exc)),
            code=getattr(exc, "code", 500),
        )


class JobImportRequest(BaseModel):
    """外部岗位导入请求体"""

    source: str = Field("tencent", description="数据源标识（当前支持 tencent / adzuna）")
    limit: int = Field(30, ge=1, le=100, description="导入条数上限")


class PasteJDRequest(BaseModel):
    """批量粘贴 JD 入库请求体（面向国内的真实岗位入口）"""

    jd_text: str = Field(..., min_length=10, max_length=50000, description="JD 文本,多段用 --- 分隔")


class JobSearchRequest(BaseModel):
    """实时多源岗位检索请求体（Tab2 主入口）"""

    keywords: str = Field(..., min_length=1, max_length=100, description="岗位关键词，必填")
    city: str = Field("", max_length=50, description="城市（如 '杭州' / '北京'），留空不限")
    salary_min: int = Field(0, ge=0, le=200, description="最低薪资 K/月（严格大于），0=不限")
    sources: Optional[list] = Field(None, description="启用的数据源（None=全部可用）；如 ['local','adzuna']")
    limit: int = Field(30, ge=1, le=100, description="返回条数上限")


@router.post("/jobs/import", summary="从公开 API 导入岗位（合规数据源）")
async def import_external_jobs(request: JobImportRequest):
    """从合规公开职位 API 拉取岗位并导入 job_info 表（按 岗位名+公司+城市 去重）"""
    try:
        data = service.import_external_jobs(request.source, request.limit)
        return success_response(data)
    except Exception as exc:
        return error_response(
            str(getattr(exc, "message", exc)),
            code=getattr(exc, "code", 500),
        )


@router.post("/jobs/paste", summary="批量粘贴 JD 入库（面向国内的真实岗位入口）")
async def paste_jd(request: PasteJDRequest):
    """用户从招聘网站复制 JD 原文,粘贴到前端 → LLM 自动解析为结构化字段入库

    支持多段 JD: 用 --- 分隔,系统逐段解析,按 公司+岗位名+城市 去重
    """
    try:
        data = service.parse_and_import_pasted_jd(request.jd_text)
        return success_response(data, message=f"解析 {data['parsed']} 条 | 新增 {data['inserted']} 条 | 跳过 {data['skipped']} 条")
    except Exception as exc:
        return error_response(
            str(getattr(exc, "message", exc)),
            code=getattr(exc, "code", 500),
        )


@router.get("/jobs/sources", summary="外部数据源列表")
async def list_external_sources():
    """列出全部可用的外部岗位数据源"""
    try:
        data = service.list_external_sources()
        return success_response(data)
    except Exception as exc:
        return error_response(
            str(getattr(exc, "message", exc)),
            code=getattr(exc, "code", 500),
        )


@router.post("/jobs/search", summary="实时多源岗位检索（Tab2 主入口）")
async def search_jobs(request: JobSearchRequest):
    """根据关键词+城市+薪资实时检索岗位

    数据流：本地 MySQL job_info 优先 → 命中不足时按 sources 列表并发拉取外部合规 API
    （腾讯公开招聘 / Adzuna）→ 入库缓存 → 合并去重返回。

    返回字段:
      - items: 岗位列表（每条带 data_source 字段标识来源）
      - sources_used: 本次查询实际用到的数据源
      - local_count / external_count / external_skipped: 命中与新增统计
      - sources_failed: 失败的源及原因（用于前端展示/降级提示）
    """
    try:
        data = service.search_jobs(
            keywords=request.keywords,
            city=request.city,
            salary_min=request.salary_min,
            sources=request.sources,
            limit=request.limit,
        )
        return success_response(data)
    except Exception as exc:
        return error_response(
            str(getattr(exc, "message", exc)),
            code=getattr(exc, "code", 500),
        )


@router.get("/jobs/{job_id}", summary="岗位详情")
async def get_job(job_id: int):
    """按 ID 查询岗位详情"""
    try:
        data = service.get_job(job_id)
        return success_response(data)
    except Exception as exc:
        return error_response(
            str(getattr(exc, "message", exc)),
            code=getattr(exc, "code", 500),
        )
