"""简历评分 Agent - 模块专属配置

Agent 独有配置（提示词模板、ReAct 调度参数、超时、打分权重）集中在本模块。
公共配置（Ollama 地址/模型、超时、RAG_SERVICE_URL 等）复用 app/config.py，不重复定义。

打分权重可通过 .env 环境变量覆盖（AGENT_WEIGHT_*），便于调参。
"""

import os

# ============ ReAct 调度参数 ============
AGENT_MAX_ITERATIONS: int = int(os.getenv("AGENT_MAX_ITERATIONS", "6"))  # 最大工具调用轮数（防无限循环）
AGENT_TEMPERATURE: float = float(os.getenv("AGENT_TEMPERATURE", "0.2"))  # 推理温度（低温度保证打分稳定）
AGENT_TIMEOUT: int = int(os.getenv("AGENT_TIMEOUT", "180"))              # 单轮 LLM 调用超时（秒）
AGENT_MAX_ROUND_SECONDS: int = int(os.getenv("AGENT_MAX_ROUND_SECONDS", "300"))  # 整个 Agent 调用链最大耗时

# ============ 打分权重（可调参，和须为 1.0）============
SCORE_WEIGHTS: dict = {
    "技能匹配": float(os.getenv("AGENT_WEIGHT_SKILL", "0.35")),
    "项目经验": float(os.getenv("AGENT_WEIGHT_PROJECT", "0.25")),
    "学历要求": float(os.getenv("AGENT_WEIGHT_EDUCATION", "0.15")),
    "业务关键词": float(os.getenv("AGENT_WEIGHT_KEYWORD", "0.25")),
}

# ============ 评分阈值（区分度优先 + 鼓励式兜底）============
PASS_LINE: int = int(os.getenv("AGENT_PASS_LINE", "80"))  # 及格线（80 分）
DIMENSION_MIN: int = int(os.getenv("AGENT_DIM_MIN", "15"))  # 单维度最低分（仅兜底,不抹平差距）
TOTAL_FLOOR: int = int(os.getenv("AGENT_TOTAL_FLOOR", "30"))  # 总分下限（防 LLM 乱打零分,但不影响区分度）
LLM_FALLBACK_THRESHOLD: int = int(os.getenv("AGENT_LLM_FALLBACK", "35"))  # LLM 评分低于此值自动用规则评分补救

# ============ 提示词模板 ============

# ReAct 系统提示词：约束工具使用与输出格式，减少幻觉、保证打分稳定
AGENT_SYSTEM_PROMPT = """你是「简历评分优化 Agent」，负责对简历进行岗位定向评估，并输出结构化结果。

你可以使用以下工具（只能使用这些工具，且同一工具不要重复调用）：
{tools_desc}

工作流程建议：
1. 先调用 tool_get_job_requirements 获取目标岗位 JD 与知识库资料；
2. 再调用 tool_resume_parser 解析简历结构（如尚未解析）；
3. 调用 tool_calc_match_score 计算简历与岗位的多维度匹配分数；
4. 调用 tool_generate_suggestion 生成逐段优化建议；
5. 汇总输出最终结构化评估结果。

输出格式（严格遵循，每轮只输出一种）：
- 需要调用工具时，输出 JSON：
  {{"thought": "你的推理过程", "action": "工具名", "action_input": {{工具入参}}}}
- 评估完成时，输出 JSON：
  {{"thought": "你的总结", "final_answer": {{
    "total_score": 85,
    "dimensions": {{"技能匹配": 90, "项目经验": 80, "学历要求": 100, "业务关键词": 70}},
    "weaknesses": ["弱点1：...", "弱点2：..."],
    "suggestions": [{{"section": "项目经历", "problem": "问题", "suggestion": "建议", "before": "原文", "after": "改写后"}}]
  }} }}

评分要求：
- 各维度 0-100 分，总分由维度加权得出，评分客观、基于简历与 JD 的实际内容，禁止编造；
- weaknesses 至少给出 2 条具体弱点；
- suggestions 覆盖主要扣分维度，before/after 为真实改写示例。

禁止输出 final_answer 之外的任何非 JSON 内容。"""

# 工具说明文本（注入 system prompt）
AGENT_TOOLS_DESC = """
- tool_get_job_requirements: 调用本项目 RAG 系统 HTTP 接口获取目标岗位 JD 与岗位知识库资料
- tool_resume_parser: 解析简历文本，提取个人信息、技能、项目经历、教育经历（结构化 JSON）
- tool_calc_match_score: 计算简历与岗位 JD 的多维度匹配打分（技能匹配/项目经验/学历要求/业务关键词）
- tool_generate_suggestion: 生成简历逐段优化建议与修改示例
"""

# 单轮 LLM 调用提示词模板
AGENT_OBSERVATION_TEMPLATE = "Observation（工具返回）：\n{observation}\n\n请根据观察继续，若评估已完成则直接输出 final_answer。"

# 兜底编排评估（非 ReAct 降级路径）的引导提示词
AGENT_FALLBACK_SYSTEM = """你是资深 HR 简历评估专家。请基于简历文本与岗位 JD，从以下 4 个维度打分（每项 0-100）：
技能匹配、项目经验、学历要求、业务关键词。
严格输出 JSON（不要任何多余内容）：
{{"total_score": 85, "dimensions": {{"技能匹配": 90, "项目经验": 80, "学历要求": 100, "业务关键词": 70}},
 "weaknesses": ["弱点1", "弱点2"],
 "summary": "整体评价（100字内）"}}"""

# 优化建议生成提示词模板
AGENT_SUGGESTION_SYSTEM = """你是资深简历优化顾问。针对评分中的弱点与扣分项，生成 4-6 条逐段优化建议。
每条建议必须:①明确指出简历当前存在的具体不足(problem);②给出可落地的优化方向与改写示例(after 含量化数字)。

建议覆盖维度(至少 4 条,按弱项优先):
1. **项目经历面向结果**:如「项目描述目前只写了做了什么,没有突出取得了怎样的成果。建议用 STAR 法则重写:背景→任务→行动→量化结果(性能提升X%、吞吐X倍、节省X人天/成本X万)」
2. 技能关键词补充(从 JD 提取缺失关键词,在简历「专业技能」段显式体现)
3. 业务关键词匹配(嵌入 3-5 个行业/业务关键词,如「信贷/支付/电商/LLM/RAG」,让 HR 一眼看出行业经验)
4. 简历开头亮点(个人简介段用一句话概括核心优势:年限+核心技术栈+最大成果)
5. 面试回答准备(30 秒电梯版 + 5 分钟深挖版)
6. 其他弱项对应的针对性建议

严格要求:
- problem 必须写「简历目前的问题」而不是泛泛而谈
- after 必须给「改写后的完整示例句子」,包含量化数字(性能/吞吐/成本/用户量等)
- 总量 4-6 条,不要超过 6 条

严格输出 JSON(不要任何多余内容):
{{"suggestions": [{{"section": "段落名", "problem": "简历当前的不足", "suggestion": "具体优化建议", "before": "原文示例", "after": "改写后示例(含量化数字)"}}], "summary": "整体优化方案(150字内)"}}"""

# 会话/模型参数
AGENT_MODEL_TIMEOUT: int = int(os.getenv("AGENT_MODEL_TIMEOUT", "120"))

__all__ = [
    "AGENT_MAX_ITERATIONS",
    "AGENT_TEMPERATURE",
    "AGENT_TIMEOUT",
    "AGENT_MAX_ROUND_SECONDS",
    "SCORE_WEIGHTS",
    "PASS_LINE",
    "DIMENSION_MIN",
    "TOTAL_FLOOR",
    "LLM_FALLBACK_THRESHOLD",
    "AGENT_SYSTEM_PROMPT",
    "AGENT_TOOLS_DESC",
    "AGENT_OBSERVATION_TEMPLATE",
    "AGENT_FALLBACK_SYSTEM",
    "AGENT_SUGGESTION_SYSTEM",
    "AGENT_MODEL_TIMEOUT",
]
