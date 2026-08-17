"""简历评分 Agent - 自定义工具集（LangChain-ReAct 风格工具集合）

工具统一封装为 AgentTool（name / description / parameters / func / run），
供 ReAct 循环按工具名调度，也可被兜底编排路径直接调用。

工具清单：
1. tool_get_job_requirements  调用本项目 RAG 系统 HTTP 接口，获取岗位 JD 与岗位知识库资料
2. tool_resume_parser         复用 app/utils/doc_parser.py 解析简历 PDF/Word，结构化提取
3. tool_calc_match_score      简历-JD 多维度匹配打分（技能匹配/项目经验/学历要求/业务关键词）
4. tool_generate_suggestion   生成简历逐段优化建议与修改示例

约束：所有大模型能力复用 app/utils/ollama_client.py；调用 RAG 走 RAG_SERVICE_URL；
全程本地推理，无任何第三方云端大模型 API。
"""

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import re
import requests

from app.config import RAG_SERVICE_URL
from app.modules.agent_module.config import (
    AGENT_FALLBACK_SYSTEM,
    AGENT_SUGGESTION_SYSTEM,
    SCORE_WEIGHTS,
)
from app.utils.common_tools import (
    AppError,
    extract_json_from_text,
    get_logger,
)
from app.utils.doc_parser import parse_document
from app.utils.ollama_client import OllamaClient

logger = get_logger("agent_tools")

ollama = OllamaClient()

# 技能关键词表（技能匹配规则打分用）
SKILL_KEYWORDS = [
    "Java", "Python", "Go", "C++", "JavaScript", "TypeScript", "Vue", "React",
    "Spring", "SpringBoot", "MyBatis", "MySQL", "Redis", "MongoDB", "Elasticsearch",
    "Kafka", "RocketMQ", "Docker", "Kubernetes", "Linux", "Nginx", "Git",
    "Hadoop", "Spark", "Flink", "Hive", "机器学习", "深度学习", "TensorFlow", "PyTorch",
    "NLP", "计算机视觉", "数据分析", "SQL", "算法", "数据结构", "微服务", "分布式",
    "消息队列", "高并发", "性能优化", "JVM", "多线程", "网络编程", "自动化测试", "CI/CD",
    "云计算", "Android", "iOS", "Flutter", "小程序", "前端", "后端", "全栈",
]

# 学历等级映射（学历要求匹配打分用）
EDUCATION_LEVELS = {"博士": 100, "硕士": 90, "本科": 80, "大专": 65, "不限": 80}


# ============ 工具封装 ============


@dataclass
class AgentTool:
    """Agent 工具描述与执行封装"""

    name: str
    description: str
    func: Callable
    parameters: dict = field(default_factory=dict)

    def run(self, args: dict) -> str:
        """执行工具，返回 observation 文本（统一序列化为 JSON 字符串）"""
        try:
            result = self.func(**args)
            return json.dumps(result, ensure_ascii=False)
        except TypeError as exc:
            logger.error("工具 %s 入参错误 args=%s error=%s", self.name, args, exc)
            return json.dumps(
                {"error": f"工具入参错误：{exc}，请检查 action_input 字段名"}, ensure_ascii=False
            )
        except Exception as exc:
            logger.error("工具 %s 执行失败 error=%s", self.name, exc)
            return json.dumps(
                {"error": f"工具执行失败：{getattr(exc, 'message', exc)}"}, ensure_ascii=False
            )


# ============ 工具 1：获取岗位需求（RAG 联动）============


def tool_get_job_requirements(target_job: str = "", jd_text: str = "") -> dict:
    """获取目标岗位 JD 与岗位知识库资料

    三条数据源路径（优先级递减）:
      0. 用户上传 JD - 直接用 jd_text(精准评估,跳过搜索)
      1. 主路径 - 调用 /api/rag/jobs/search 实时多源检索（本地 + 腾讯 + Adzuna）
      2. 次路径 - 调用 /api/rag/query NL2SQL 兜底（用 LLM 生成 SQL 查询本地库）
      3. 增强 - 调 /api/rag/chat 检索知识库资料丰富上下文

    Args:
        target_job: 目标岗位名称
        jd_text: 用户上传的 JD 文本(可选,如提供则优先用)
    Returns:
        {"found", "job_name", "jd", "knowledge", "message", "sources_used", "source_type"}
    """
    result: dict = {"found": False, "job_name": target_job, "jd": "", "knowledge": "", "message": "", "sources_used": [], "source_type": "none"}

    # 0) 用户上传 JD 路径(优先级最高 - 即使 target_job 留空也能工作)
    if jd_text and jd_text.strip() and len(jd_text.strip()) >= 20:
        job_name = target_job.strip() if target_job and target_job.strip() else "（用户上传 JD）"
        result.update({
            "found": True,
            "job_name": job_name,
            "jd": jd_text.strip(),
            "source_type": "user_uploaded",
            "message": f"已使用你上传的 JD 文本(共{len(jd_text.strip())}字)做精准评估",
            "sources_used": ["用户上传 JD"],
        })
        # 仍调知识库丰富上下文(可选)
        try:
            chat_resp = requests.post(
                f"{RAG_SERVICE_URL}/api/rag/chat",
                json={"query": (target_job or "岗位") + " 岗位要求 技能栈"},
                timeout=30,
            )
            chat_resp.raise_for_status()
            ch = ((chat_resp.json() or {}).get("data") or {})
            ans = ch.get("answer", "")
            if ans:
                result["knowledge"] = ans[:1500]
        except Exception as exc:
            logger.warning("知识库补充失败(可忽略): %s", exc)
        return result

    # 0.5) 都没有时给出明确提示
    if not target_job or not target_job.strip():
        result["message"] = "未提供目标岗位名称（请在「目标岗位」输入岗位名,或在 JD 文本框粘贴 JD 原文）"
        return result

    # 1) 主路径：实时多源检索（本地 + 外部 API）
    try:
        resp = requests.post(
            f"{RAG_SERVICE_URL}/api/rag/jobs/search",
            json={
                "keywords": target_job.strip(),
                "city": "",
                "salary_min": 0,
                "sources": ["local", "tencent", "adzuna"],
                "limit": 8,
            },
            timeout=12,
        )
        resp.raise_for_status()
        data = ((resp.json() or {}).get("data") or {})
        items = data.get("items") or []
        sources_used = data.get("sources_used") or []
        if items:
            job = items[0]
            # 候选列表:供前端预览 + 用户选择(默认取第一条)
            candidates = [
                {
                    "job_name": x.get("job_name"),
                    "company": x.get("company"),
                    "city": x.get("city"),
                    "salary_min": x.get("salary_min"),
                    "salary_max": x.get("salary_max"),
                    "jd": _build_jd_text(x),
                }
                for x in items[:8]
            ]
            result.update({
                "found": True,
                "job_name": job.get("job_name"),
                "jd": _build_jd_text(job),
                "candidates": candidates,
                "sources_used": sources_used,
                "message": f"已通过实时多源检索到岗位「{job.get('job_name')}」（{job.get('company', '')} · {job.get('city', '')} · 来源: {', '.join(sources_used)}）",
            })
    except Exception as exc:
        logger.warning("调用 RAG 多源检索接口失败：%s", exc)

    # 2) 次路径：NL2SQL 兜底
    if not result["found"]:
        try:
            resp = requests.post(
                f"{RAG_SERVICE_URL}/api/rag/query",
                json={"query": f"查询岗位名称包含 {target_job} 的岗位信息", "limit": 5},
                timeout=12,
            )
            resp.raise_for_status()
            rows = ((resp.json() or {}).get("data") or {}).get("result") or []
            if rows:
                job = rows[0]
                candidates = [
                    {
                        "job_name": x.get("job_name"),
                        "company": x.get("company"),
                        "city": x.get("city"),
                        "salary_min": x.get("salary_min"),
                        "salary_max": x.get("salary_max"),
                        "jd": _build_jd_text(x),
                    }
                    for x in rows[:5]
                ]
                result.update({
                    "found": True,
                    "job_name": job.get("job_name"),
                    "jd": _build_jd_text(job),
                    "candidates": candidates,
                    "sources_used": ["local"],
                    "message": f"通过本地 NL2SQL 检索到岗位「{job.get('job_name')}」（{job.get('company', '')}）",
                })
        except Exception as exc:
            logger.warning("调用 RAG 智能查询接口失败：%s", exc)

    # 3) 增强：检索行业知识库
    if result.get("found"):
        try:
            resp = requests.post(
                f"{RAG_SERVICE_URL}/api/rag/chat",
                json={"query": f"针对岗位「{result['job_name']}」的职责要求与核心技能要求", "top_k": 2},
                timeout=60,
            )
            resp.raise_for_status()
            chat_data = ((resp.json() or {}).get("data") or {})
            answer = chat_data.get("answer", "")
            if answer and "参考资料" not in answer[:20]:
                result["knowledge"] = answer[:1500]
        except Exception as exc:
            logger.warning("调用 RAG 知识库问答失败：%s", exc)

    if not result.get("found"):
        result["message"] = f"岗位数据库中没有找到与「{target_job}」匹配的岗位，将进行通用评估"
    return result


def _build_jd_text(job: dict) -> str:
    """拼装岗位 JD 文本"""
    parts = [
        f"岗位名称：{job.get('job_name', '')}",
        f"公司：{job.get('company', '')}",
        f"城市：{job.get('city', '')}",
        f"薪资：{job.get('salary_min', 0)}K-{job.get('salary_max', 0)}K",
        f"学历要求：{job.get('education', '')}",
        f"经验要求：{job.get('experience', '')}",
        f"技能要求：{job.get('skill_require', '')}",
        f"岗位描述：{job.get('job_desc', '')}",
    ]
    return "\n".join(p for p in parts if p.split("：", 1)[-1].strip())


# ============ 工具 2：简历解析（复用 doc_parser）============


def tool_resume_parser(resume_text: str = "", resume_path: str = "") -> dict:
    """解析简历文本/文件，结构化提取：个人信息、技能、项目经历、教育经历

    优先使用已有纯文本；未提供时复用 app/utils/doc_parser.py 解析文件路径。
    结构化提取由本地大模型完成，解析失败回退规则提取（保证工具可用）。

    Args:
        resume_text: 简历纯文本（可选）
        resume_path: 简历文件路径（可选，pdf/docx）
    Returns:
        {"personal_info", "skills", "projects", "education", "raw_text", "raw_length"}
    """
    text = (resume_text or "").strip()
    if not text and resume_path:
        parsed = parse_document(resume_path)
        text = parsed["text"]

    if not text or len(text) < 10:
        return {
            "personal_info": {},
            "skills": [],
            "projects": [],
            "education": [],
            "raw_text": text or "",
            "raw_length": len(text or ""),
            "error": "简历文本为空或过短，无法解析",
        }

    structure_system = (
        "你是一个专业的简历解析器。从简历文本中提取结构化信息，严格输出如下 JSON（不要任何多余内容）：\n"
        '{"personal_info": {"name": "姓名", "phone": "电话", "email": "邮箱", '
        '"education_level": "最高学历", "years_of_experience": "工作年限"}, '
        '"education": ["教育经历"], "projects": ["项目经历"], "skills": ["技能列表"]}\n'
        "无法提取的字段填空字符串或空数组，不要编造。"
    )
    try:
        raw = ollama.chat(text[:4000], system=structure_system, temperature=0.2)
        data = extract_json_from_text(raw)
        return {
            "personal_info": data.get("personal_info") or {},
            "skills": data.get("skills") or [],
            "projects": data.get("projects") or [],
            "education": data.get("education") or [],
            "raw_text": text,
            "raw_length": len(text),
        }
    except AppError:
        logger.warning("简历 LLM 结构化解析失败，使用规则提取兜底")
        return _rule_extract_resume(text)


def _rule_extract_resume(text: str) -> dict:
    """规则提取兜底：从简历文本中抽取技能关键词"""
    lower = text.lower()
    skills = [w for w in SKILL_KEYWORDS if w.lower() in lower]
    return {
        "personal_info": {},
        "skills": skills,
        "projects": [],
        "education": [],
        "raw_text": text,
        "raw_length": len(text),
        "note": "规则提取模式（LLM 结构化失败兜底）",
    }


# ============ 工具 3：简历-JD 多维度匹配打分（规则计算，稳定可控）============


def tool_calc_match_score(
    resume_info: dict = None,
    job_requirements: str = "",
    resume_text: str = "",
) -> dict:
    """简历-JD 多维度匹配打分（技能匹配 / 项目经验 / 学历要求 / 业务关键词）

    规则计算为主，保证打分稳定可控、可复现。

    Args:
        resume_info: 简历结构化信息（tool_resume_parser 输出）
        job_requirements: 岗位 JD 文本（tool_get_job_requirements 输出）
        resume_text: 简历原始文本（备用）
    Returns:
        {"total_score", "dimensions": {维度: 分数}, "matched_skills", "missing_skills"}
    """
    info = resume_info or {}
    resume_lower = (resume_text or "").lower()
    jd_lower = (job_requirements or "").lower()

    # 技能匹配：JD 技能关键词在简历中的覆盖率(区分度优先:全匹配≈100,0匹配≈15)
    # 注意:统一小写比较(修复英文技能大小写导致匹配失败的 bug)
    jd_skills = _extract_skills(jd_lower)
    resume_skills = set(str(s).lower() for s in (info.get("skills") or []))
    if not jd_skills:
        skill_score = 75.0
        matched, missing = [], []
    else:
        matched = [s for s in jd_skills if s.lower() in resume_lower or s.lower() in resume_skills]
        missing = [s for s in jd_skills if s.lower() not in resume_lower and s.lower() not in resume_skills]
        # 模糊匹配:相似技术栈加分(如 Python↔Java 都属后端语言)
        similar_pairs = [
            ({"python", "django", "flask", "fastapi"}, {"java", "spring", "springboot", "spring cloud"}),
            ({"javascript", "typescript", "react", "vue"}, {"html", "css"}),
            ({"mysql", "redis"}, {"postgresql", "mongodb", "elasticsearch"}),
            ({"docker", "k8s"}, {"kubernetes", "containerd"}),
            ({"llm", "langchain", "llamaindex"}, {"ai", "机器学习", "深度学习"}),
        ]
        matched_lower = {s.lower() for s in matched}
        fuzzy = 0
        for m_set, r_set in similar_pairs:
            if any(s.lower() in matched_lower for s in m_set) and any(r in resume_skills for r in r_set):
                fuzzy += 1
        exact_pct = len(matched) / len(jd_skills)
        # 区分度设计:无固定保底,全匹配≈100,完全不匹配≈15(少量相邻技术栈迁移分)
        skill_score = min(100, round(exact_pct * 80 + fuzzy * 15 + (15 if exact_pct == 0 else 0)))

    # 项目经验：按项目条数 & 量化描述(区分度:0项目≈40,3项目+量化≈100)
    projects = info.get("projects") or []
    project_score = min(100, 40 + len(projects) * 20)
    has_quant = bool(re_search_quant(resume_text))
    if has_quant:
        project_score = min(100, project_score + 15)

    # 学历要求：JD 学历 vs 简历学历
    edu_score = _calc_education_score(jd_lower, info)

    # 业务关键词重合度：JD 与简历的高频业务词（非技能）重合
    keyword_score = _calc_keyword_score(jd_lower, resume_lower)

    dimensions = {
        "技能匹配": skill_score,
        "项目经验": project_score,
        "学历要求": edu_score,
        "业务关键词": keyword_score,
    }
    total = round(
        sum(SCORE_WEIGHTS.get(k, 0) * v for k, v in dimensions.items()) / sum(SCORE_WEIGHTS.values())
    )
    return {
        "total_score": total,
        "dimensions": dimensions,
        "matched_skills": matched,
        "missing_skills": missing,
    }


def _extract_skills(text_lower: str) -> list:
    """从（小写）文本中提取技能关键词"""
    return [w for w in SKILL_KEYWORDS if w.lower() in text_lower]


def tool_benchmark_score(resume_info: dict = None, job_requirements: str = "", resume_text: str = "") -> int:
    """锚定评分 v4(2026-08-14 多维度内容质量):基础 75 + 加分,90+ 只给真正顶级简历

    设计原则:不只看字数,看内容质量多维:
    - 技能覆盖(JD 技能命中率)
    - 项目经验(项目数 + 量化数据数)
    - 信息密度(每千字量化数 / 每百字行动词数) — 1600字废话 vs 600字精炼 天壤之别
    - 学历层次(985硕士 > 普通本科)
    - 业务关键词匹配
    - 废话检测(工作认真负责/吃苦耐劳 等套话重扣)

    等级分布:
    - 潦草/差: 30-70
    - 普通/合格: 75-82
    - 良好(4年+多项目+量化密度高): 83-88
    - 优秀(多年+多量化+匹配): 89-94
    - 顶级(8年+架构+985硕+高信息密度): 95+
    """
    import re
    info = resume_info or {}
    resume_lower = (resume_text or "").lower()
    jd_lower = (job_requirements or "").lower()
    length = len(resume_text or "")
    quant_count = len(re.findall(r"\d+[%％]|\d+\s*(人|年|万|K|%|个|次|倍)", resume_text or ""))

    # ---- 信息密度指标 ----
    # 每千字量化数(优质简历 ≥8-10/千字,废话多则很低)
    quant_density = quant_count / max(length, 1) * 1000
    # 行动词密度(每百字行动词数:负责/主导/优化/提升 等)
    action_words = ("负责", "主导", "优化", "设计", "搭建", "重构", "提升", "降低",
                    "实现", "构建", "落地", "建设", "开发", "完成", "上线", "推动",
                    "建立", "引入", "治理", "攻克", "改进", "缩短", "提升")
    action_count = sum(resume_text.count(w) for w in action_words)
    action_density = action_count / max(length, 1) * 100
    # 废话套话检测(无信息量)
    fluff_words = ("工作认真负责", "吃苦耐劳", "团队合作能力强", "抗压能力强", "学习能力强",
                   "善于沟通", "积极向上", "热爱学习", "踏实肯干", "具有良好的",
                   "沟通能力良好", "责任心强", "性格开朗", "乐于助人")
    fluff_count = sum(resume_text.count(w) for w in fluff_words)

    score = 75  # 基础分

    # === 加分项(最高 +24) ===
    bonus = 0

    # 1. 技能覆盖(0-5)
    jd_skills = _extract_skills(jd_lower)
    matched = []
    coverage = 0.0
    if jd_skills:
        resume_skills = set(str(s).lower() for s in (info.get("skills") or []))
        matched = [s for s in jd_skills if s.lower() in resume_lower or s.lower() in resume_skills]
        coverage = len(matched) / len(jd_skills)
        if coverage >= 0.95: bonus += 5
        elif coverage >= 0.85: bonus += 3
        elif coverage >= 0.7: bonus += 1

    # 2. 项目与量化(0-7)
    projects = info.get("projects") or []
    if len(projects) >= 4 and quant_count >= 5: bonus += 7
    elif len(projects) >= 3 and quant_count >= 4: bonus += 5
    elif len(projects) >= 2 and quant_count >= 2: bonus += 3
    elif len(projects) >= 1 and quant_count >= 1: bonus += 1

    # 3. 信息密度(0-4,不奖励纯字数)
    #    量化密度:每千字 ≥10 个 → +2;≥5 → +1
    if quant_density >= 10: bonus += 2
    elif quant_density >= 5: bonus += 1
    #    行动词密度:每百字 ≥3 个 → +2;≥1.5 → +1
    if action_density >= 3: bonus += 2
    elif action_density >= 1.5: bonus += 1

    # 4. 学历层次(0-3)
    edu = _calc_education_score(jd_lower, info)
    has_top_school = any(k in resume_text for k in ("985", "211", "C9", "清华", "北大", "复旦", "交大", "浙大"))
    is_master = "硕士" in resume_text
    if edu >= 100 and has_top_school and is_master:
        bonus += 3
    elif edu >= 100 and (has_top_school or is_master):
        bonus += 2
    elif edu >= 100:
        bonus += 1

    # 5. 业务关键词(0-3)
    kw = _calc_keyword_score(jd_lower, resume_lower)
    if kw >= 70: bonus += 3
    elif kw >= 50: bonus += 2
    elif kw >= 30: bonus += 1

    # 6. 章节完整(0-2,不奖励字数)
    sections_present = sum([
        any(k in resume_text for k in ("项目", "项目经验", "工作经历", "Work", "work")),
        any(k in resume_text for k in ("专业技能", "技能", "Skill", "skill")),
        any(k in resume_text for k in ("教育", "学历", "本科", "硕士", "大专")),
        any(k in resume_text for k in ("姓名", "求职", "意向", "@", "电话", "邮箱")),
    ])
    if sections_present >= 4: bonus += 2
    elif sections_present >= 3: bonus += 1

    score += bonus

    # === 扣分项 ===
    penalty = 0
    # 字数底线(极短必扣,但中等长度不再加分)
    if length < 100: penalty += 30
    elif length < 200: penalty += 18
    elif length < 300: penalty += 8

    # 信息密度过低(废话多)
    if quant_density < 2: penalty += 10      # 1600字废话=2个量化 → 1.25/千字
    elif quant_density < 4: penalty += 5
    if action_density < 1: penalty += 6      # 全文无行动词

    # 废话套话(每句 -3)
    if fluff_count >= 3: penalty += 9
    elif fluff_count >= 1: penalty += 3

    # 章节缺失
    if sections_present < 2: penalty += 10
    elif sections_present < 3: penalty += 5

    # 学历不达标
    if edu < 60: penalty += 12
    elif edu < 80: penalty += 6

    # 技能严重缺失
    if jd_skills:
        if coverage < 0.3: penalty += 10
        elif coverage < 0.5: penalty += 5

    score -= penalty
    return max(15, min(97, score))




def re_search_quant(text: str) -> bool:
    """判断文本是否包含量化指标（数字+单位/百分比）"""
    import re

    return bool(re.search(r"\d+[%％]|\d+\s*(人|年|万|K|%|个|次)", text or ""))


def _calc_education_score(jd_lower: str, info: dict) -> int:
    """学历要求匹配分：JD 学历等级 vs 简历学历等级"""
    import re

    jd_level = "不限"
    for level in ("博士", "硕士", "本科", "大专"):
        if level in jd_lower:
            jd_level = level
            break
    resume_edu = str((info.get("personal_info") or {}).get("education_level", "") or "")
    resume_edu = resume_edu + str(info.get("education") or "")
    resume_level = "不限"
    for level in ("博士", "硕士", "本科", "大专"):
        if level in resume_edu:
            resume_level = level
            break
    jd_v = EDUCATION_LEVELS.get(jd_level, 80)
    resume_v = EDUCATION_LEVELS.get(resume_level, 80)
    if resume_v >= jd_v:
        return 100
    return max(50, round(resume_v / jd_v * 100))


def _calc_keyword_score(jd_lower: str, resume_lower: str) -> int:
    """业务关键词匹配度：JD 业务词在简历中的覆盖率(区分度:JD 提到的业务领域简历里都有≈100)

    匹配策略(语义级宽松,避免近义表达误判):
    1. 整词命中(如「团队管理」简历里有)
    2. 4 字及以上业务词,简历含其任意核心 2 字子词即命中
       (如 JD「架构设计」,简历写「架构升级/架构演进」→ 含「架构」→ 命中)
    """
    import re

    STOP = {"岗位", "要求", "负责", "相关", "具有", "工作", "能力", "经验", "熟悉", "优先",
            "我们", "职位", "薪资", "地点", "职责", "任职", "以上", "以下", "以及", "进行", "包括",
            "具备", "参与", "支持", "能够", "需要", "整体", "相关", "方面", "进行"}

    def biz_words(text: str) -> set:
        words = set()
        for m in re.finditer(r"[\u4e00-\u9fa5]{2,6}", text):
            word = m.group()
            if word not in STOP:
                words.add(word)
        return words

    def kw_hit(word: str, text: str) -> bool:
        """语义级命中:整词 OR 4 字词的核心 2 字子词"""
        if word in text:
            return True
        if len(word) >= 4:
            for i in range(0, len(word) - 1, 2):
                if word[i:i + 2] in text:
                    return True
        return False

    jd_words = biz_words(jd_lower)
    if not jd_words:
        return 75
    hit = sum(1 for w in jd_words if kw_hit(w, resume_lower))
    # 命中率 0-100,无固定保底(0 命中=0,全部命中=100)
    return round(hit / len(jd_words) * 100)


# ============ 工具 4：生成逐段优化建议 =============


def tool_generate_suggestion(
    resume_text: str = "",
    score_detail: dict = None,
    job_requirements: str = "",
) -> dict:
    """生成简历逐段优化建议与修改示例

    Args:
        resume_text: 简历原文
        score_detail: 打分结果（含 dimensions / missing_skills / weaknesses）
        job_requirements: 岗位 JD（可选，用于定向建议）
    Returns:
        {"suggestions": [{"section","problem","suggestion","before","after"}], "summary"}
    """
    detail = score_detail or {}
    dimensions = detail.get("dimensions") or {}
    missing = detail.get("missing_skills") or []
    weaknesses = detail.get("weaknesses") or []

    hint = (
        f"各维度得分：{json.dumps(dimensions, ensure_ascii=False)}\n"
        f"缺失技能：{missing}\n"
        f"已识别弱点：{weaknesses}\n"
        f"岗位JD（可选）：{(job_requirements or '')[:1200]}"
    )
    system = AGENT_SUGGESTION_SYSTEM
    user = f"【扣分与弱点信息】\n{hint}\n\n【简历文本】\n{resume_text[:3000]}"
    try:
        raw = ollama.chat(user, system=system, temperature=0.3)
        data = extract_json_from_text(raw)
        if not isinstance(data, dict) or not data.get("suggestions"):
            raise AppError("优化建议为空")
        return data
    except AppError:
        logger.warning("优化建议 LLM 生成失败，返回基础建议")
        # 根据维度短板智能生成 3-5 条 actionable 建议(含项目描述模板/量化公式/STAR)
        weak_dims = [k for k, v in dimensions.items() if v < 80]
        has_projects = bool(detail.get("projects"))
        suggestions = []
        if "技能匹配" in weak_dims:
            suggestions.append({
                "section": "技能关键词",
                "problem": f"缺失技能:{', '.join(missing[:5]) if missing else '与岗位匹配度低'}",
                "suggestion": "在「专业技能」段补充缺失关键词;若有相邻技术栈(如 Python ↔ Java),显式标注可迁移经验",
                "before": "熟悉 Java、Spring Boot",
                "after": "熟悉 Java 8/11/17、Spring Boot / Spring Cloud Alibaba;了解 Python/Flask 后端开发(可快速迁移)",
            })
        if "项目经验" in weak_dims or not has_projects:
            suggestions.append({
                "section": "项目经历(面向结果)",
                "problem": "项目描述只写了做了什么,没有突出取得了怎样的成果,面试官抓不到亮点",
                "suggestion": "项目描述要**面向结果、注重成果**:用 **STAR 法则 + 量化公式** 重写每条项目:背景(Situation)→ 任务(Task)→ 行动(Action)→ 量化结果(Result,带数字:性能提升X%/吞吐X倍/节省X人天)",
                "before": "负责订单模块开发,使用 SpringBoot 优化性能",
                "after": (
                    "**项目一:信贷核心系统性能优化**(2023.03-2023.08)\n"
                    "- **背景**:大促期间接口超时严重,TP99 高达 1.2s,用户体验下降\n"
                    "- **方案**:热点数据本地缓存 + Redis 二级缓存,DB 读写分离 + 慢 SQL 治理\n"
                    "- **行动**:主导落地 4 个核心接口改造,接入 Caffeine + Redis Cluster\n"
                    "- **结果**:下单接口 TP99 由 **1.2s 降至 250ms**,系统吞吐 **提升 3 倍**,大促峰值扛住 50万单/日"
                ),
            })
        # 简历开头亮点(个人简介)
        suggestions.append({
            "section": "简历开头亮点",
            "problem": "简历开头没有个人亮点概括,HR 前 10 秒无法判断你是否匹配",
            "suggestion": "在简历顶部加一句话「个人亮点」:年限 + 核心技术栈 + 最大成果(带数字),让 HR 一眼看到匹配度",
            "before": "(无个人简介,直接进入教育背景)",
            "after": "**8 年 Java 后端经验 | 主导过日均千万级订单系统架构升级,系统可用性 99.99% | 精通微服务/高并发/分布式架构**",
        })
        if "业务关键词" in weak_dims:
            suggestions.append({
                "section": "业务关键词",
                "problem": "简历中缺少岗位行业/业务关键词,HR 一眼看不出行业经验",
                "suggestion": "在项目描述里嵌入 3-5 个行业关键词(从 JD 里抓取,如「信贷/支付/电商/LLM/RAG」)",
                "before": "负责系统开发,提升用户体验",
                "after": "负责**信贷核心系统**订单模块开发,涉及**支付链路**/**用户增长**/**风控决策**三大业务域",
            })
        if "学历要求" in weak_dims:
            suggestions.append({
                "section": "教育背景",
                "problem": "学历匹配度偏低",
                "suggestion": "教育背景突出与岗位相关的课程/项目/证书(如 Java 岗突出计算机专业、AI 岗突出机器学习课程)",
                "before": "某理工大学 计算机科学与技术 本科",
                "after": "某理工大学(211) 计算机科学与技术 本科 | 主修课程:数据结构、操作系统、计算机网络(均 90+) | 机器学习课程项目 Top 5%",
            })
        # 通用兜底
        if not suggestions:
            suggestions.append({
                "section": "整体优化",
                "problem": "整体匹配度尚可,建议强化项目量化成果",
                "suggestion": "每段项目经历补充「量化结果」(数字+单位),如性能提升 X%、吞吐 X 倍、节省 X 人天",
                "before": "负责电商后台系统开发,提升系统稳定性",
                "after": "**负责电商后台系统重构**,涉及 12 个核心模块,**系统稳定性从 99.5% 提升至 99.95%,月度故障数下降 70%,节省运维成本约 50 万/年**",
            })
        # 面试回答建议(额外一条)
        suggestions.append({
            "section": "面试回答模板",
            "problem": "项目描述写得太书面,面试时容易卡壳",
            "suggestion": "面试时按 **「30 秒电梯版 + 5 分钟深挖版」** 两套答案准备。电梯版讲背景+量化结果;深挖版讲技术选型理由+踩坑+反思",
            "before": "(没有准备,临场发挥)",
            "after": (
                "**30秒电梯版**:这个项目解决的是 XX 业务痛点,我主导了 XX 模块重构,**最终 TP99 从 1.2s 降到 250ms,系统吞吐提升 3 倍**。\n"
                "**5分钟深挖版**:技术选型上,为什么用 Caffeine 不直接用 Redis?→ 因为热点数据本地缓存比 Redis 减少 80% 网络 IO;踩过什么坑?→ 缓存击穿,用了分布式锁解决..."
            ),
        })
        return {
            "suggestions": suggestions,
            "summary": f"基于 {' / '.join(weak_dims) or '整体维度'} 短板生成 {len(suggestions)} 条 actionable 建议(含项目量化模板 + 面试回答示例)。",
        }


# ============ 工具注册表 ============

AGENT_TOOLS: dict = {
    tool.name: tool
    for tool in [
        AgentTool(
            name="tool_get_job_requirements",
            description="调用本项目 RAG 系统 HTTP 接口获取目标岗位 JD 与岗位知识库资料",
            func=tool_get_job_requirements,
            parameters={"target_job": "目标岗位名称（字符串）"},
        ),
        AgentTool(
            name="tool_resume_parser",
            description="解析简历文本，提取个人信息、技能、项目经历、教育经历（结构化JSON）",
            func=tool_resume_parser,
            parameters={"resume_text": "简历纯文本（字符串）"},
        ),
        AgentTool(
            name="tool_calc_match_score",
            description="计算简历与岗位 JD 的多维度匹配打分（技能匹配/项目经验/学历要求/业务关键词）",
            func=tool_calc_match_score,
            parameters={
                "resume_info": "简历结构化信息（tool_resume_parser 输出）",
                "job_requirements": "岗位JD文本（tool_get_job_requirements 输出）",
                "resume_text": "简历原始文本（可选）",
            },
        ),
        AgentTool(
            name="tool_generate_suggestion",
            description="生成简历逐段优化建议与修改示例",
            func=tool_generate_suggestion,
            parameters={
                "resume_text": "简历原文",
                "score_detail": "打分结果（含 dimensions/missing_skills/weaknesses）",
                "job_requirements": "岗位JD文本（可选）",
            },
        ),
    ]
}
