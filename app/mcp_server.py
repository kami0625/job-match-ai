"""MCP Server：将求职 AI 助手核心工具暴露为标准 MCP 工具

能力：
- 把 3 个核心能力(查 JD / 简历解析 / 锚定评分)封装为 MCP 工具
- 支持双传输：stdio(本地调试/桌面客户端) + SSE(网络调用/前端直连)
- 工具失败自动降级，接口始终返回结构化结果

用法：
    # stdio 模式(默认,Claude Desktop 等本地客户端)
    python -m app.mcp_server

    # SSE 模式(网络调用,暴露 8001 端口)
    python -m app.mcp_server sse

    # 注册到 docker-compose 可作为独立服务供外部 Agent 调用
"""
from typing import Any, Dict

import sys as _sys

# Windows 下强制 UTF-8 输出:防止 GBK 编码的启动 banner 污染 stdio 管道(MCP 协议要求 UTF-8)
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(_sys.stderr, "reconfigure"):
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from fastmcp import FastMCP

from app.modules.agent_module.tools import (
    tool_benchmark_score,
    tool_get_job_requirements,
    tool_resume_parser,
)
from app.utils.common_tools import get_logger

logger = get_logger("mcp_server")

mcp = FastMCP(
    "job-match-mcp",
    instructions=(
        "求职 AI 助手工具集：提供岗位 JD 检索、简历结构化解析、简历-岗位锚定评分三个工具。"
        "所有工具均本地运行，不依赖云端 API。"
    ),
)


@mcp.tool()
def get_job_requirements(target_job: str = "", jd_text: str = "") -> Dict[str, Any]:
    """获取目标岗位的 JD(职位描述)。

    Args:
        target_job: 目标岗位名称(如「Java开发工程师」),系统自动检索岗位 JD
        jd_text: 用户直接粘贴的 JD 文本(可选,提供则优先使用)
    Returns:
        岗位 JD 文本与来源信息
    """
    try:
        return tool_get_job_requirements(target_job=target_job, jd_text=jd_text)
    except Exception as exc:
        logger.error("MCP get_job_requirements 失败: %s", exc)
        return {"job_requirements": jd_text or "", "source": "fallback", "error": str(exc)}


@mcp.tool()
def parse_resume(resume_text: str = "") -> Dict[str, Any]:
    """将简历文本结构化解析为个人信息/技能/项目经历/教育经历。

    Args:
        resume_text: 简历纯文本(建议全文传入)
    Returns:
        {"personal_info", "skills", "projects", "education", "raw_length"}
    """
    return tool_resume_parser(resume_text=resume_text)


@mcp.tool()
def benchmark_resume(resume_text: str, job_requirements: str = "") -> Dict[str, Any]:
    """对简历-岗位进行锚定评分(0-100)，返回总分与结构化信息。

    评分基于规则化锚定体系(基础 75 + 六维加分 + 建议数扣分)，可解释、可复现，
    用于替代 LLM 主观打分。

    Args:
        resume_text: 简历纯文本
        job_requirements: 目标岗位 JD 文本(必传以计算技能匹配)
    Returns:
        {"score", "skills", "raw_length"}
    """
    info = tool_resume_parser(resume_text=resume_text)
    score = tool_benchmark_score(
        resume_info=info,
        job_requirements=job_requirements or "",
        resume_text=resume_text,
    )
    return {
        "score": score,
        "skills": info.get("skills", []),
        "raw_length": info.get("raw_length", 0),
    }


if __name__ == "__main__":
    import sys

    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    logger.info("启动 MCP Server transport=%s", transport)
    mcp.run(transport=transport)  # "stdio" | "sse"
