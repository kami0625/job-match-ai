"""MCP Server 单元测试：工具封装正确性(不依赖 MCP 网络层)

注：MCP stdio 通信在 Windows 有编码坑(GBK banner 污染管道)，
故只测工具函数本身 + server 可导入，网络层留本地手测。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.mcp_server import benchmark_resume, get_job_requirements, parse_resume

JD = "要求 Java SpringBoot MySQL 本科"
RESUME = "个人简历 张三 本科 Java开发 项目:订单系统 SpringBoot MySQL"


def test_server_importable():
    import app.mcp_server as m

    assert hasattr(m, "mcp")
    assert m.mcp.name == "job-match-mcp"


def test_get_job_requirements_fallback():
    """未给岗位名时,传入的 JD 文本被接受返回(jd 字段)"""
    r = get_job_requirements(target_job="", jd_text="某公司 Java开发工程师 要求 Java SpringBoot MySQL 本科及以上")
    assert "Java" in r.get("jd", "")


def test_parse_resume_ok():
    """简历解析返回结构化字段"""
    r = parse_resume(resume_text=RESUME)
    assert "skills" in r
    assert "projects" in r
    assert r["raw_length"] > 0


def test_benchmark_resume_score_range():
    """锚定评分 0-100 且含技能列表"""
    r = benchmark_resume(resume_text=RESUME, job_requirements=JD)
    assert 0 <= r["score"] <= 100
    assert isinstance(r["skills"], list)


def test_benchmark_resume_empty_floor():
    """空简历评分不崩溃(封底)"""
    r = benchmark_resume(resume_text="", job_requirements=JD)
    assert 0 <= r["score"] <= 100
