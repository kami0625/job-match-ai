"""Agent 输出解析器

解析 ReAct 思考 / 行动 / 观察输出：
1. 兼容 JSON 输出（含 ```json 代码块包裹、前后杂散文本）
2. 兼容文本形式（Thought: ... / Action: ... / Action Input: ... / Final Answer: ...）
3. 处理大模型输出格式错乱、解析失败异常（抛出 AppError 由上层降级）

支持的动作类型：
- {"thought": "...", "action": "工具名", "action_input": {...}}   -> 需要调用工具
- {"thought": "...", "final_answer": {...}}                       -> 评估完成
"""

import json
import re
from typing import Any, Optional

from app.utils.common_tools import AppError, get_logger

logger = get_logger("agent_parser")

# 允许的 Agent 工具名
ALLOWED_ACTIONS = {
    "tool_get_job_requirements",
    "tool_resume_parser",
    "tool_calc_match_score",
    "tool_generate_suggestion",
}

_ACTION_KEYWORDS = ("tool_get_job_requirements", "tool_resume_parser", "tool_calc_match_score", "tool_generate_suggestion")


def parse_react_output(text: str) -> dict:
    """解析 ReAct 模型输出

    Args:
        text: 大模型原始输出
    Returns:
        规范化结果：
        - 调用工具：{"thought": str, "action": 工具名, "action_input": dict}
        - 完成评估：{"thought": str, "final_answer": dict}
    Raises:
        AppError: 无法解析出合法动作或最终答案
    """
    if not text or not text.strip():
        raise AppError("模型输出为空")

    cleaned = _strip_code_block(text).strip()

    # ---------- 方式 1：JSON 解析 ----------
    parsed = _try_json(cleaned)
    if isinstance(parsed, dict):
        return _normalize_json_result(parsed)

    # ---------- 方式 2：文本形式解析 ----------
    text_result = _parse_text_form(cleaned)
    if text_result:
        return text_result

    # ---------- 方式 3：JSON 子串提取 ----------
    sub = _extract_json_substring(cleaned)
    if sub:
        try:
            parsed = json.loads(sub)
            if isinstance(parsed, dict):
                return _normalize_json_result(parsed)
        except json.JSONDecodeError:
            pass

    raise AppError(f"无法解析 ReAct 输出：{text[:120]}")


def _strip_code_block(text: str) -> str:
    """去除 markdown 代码块标记"""
    return re.sub(r"```(?:json)?", "", text).replace("```", "").strip("` \n")


def _try_json(cleaned: str) -> Optional[dict]:
    """尝试整体 JSON 解析"""
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def _normalize_json_result(data: dict) -> dict:
    """规范化 JSON 结果（兼容字段缺失/别名）"""
    thought = str(data.get("thought") or data.get("reasoning") or "").strip()

    # 最终答案
    final = data.get("final_answer")
    if final is not None:
        return {"thought": thought, "final_answer": final}

    # 工具调用
    action = data.get("action") or data.get("tool")
    if action:
        action_name = str(action).strip()
        if action_name not in ALLOWED_ACTIONS:
            raise AppError(f"模型选择了未知工具：{action_name}")
        action_input = data.get("action_input") or data.get("input") or data.get("parameters") or {}
        if not isinstance(action_input, dict):
            action_input = {"value": str(action_input)}
        return {"thought": thought, "action": action_name, "action_input": action_input}

    # 仅 final 内容（直接视为最终答案）
    for key in ("result", "output", "answer", "evaluation"):
        if key in data:
            return {"thought": thought, "final_answer": data[key]}

    raise AppError("JSON 中未包含 action 或 final_answer 字段")


def _parse_text_form(cleaned: str) -> Optional[dict]:
    """解析文本形式：Thought: / Action: / Action Input: / Final Answer:"""
    thought_m = re.search(r"(?:Thought|思考)\s*[:：]\s*(.+)", cleaned, re.S)
    thought = thought_m.group(1).strip() if thought_m else ""

    final_m = re.search(r"(?:Final Answer|最终答案)\s*[:：]\s*(\{.*\})", cleaned, re.S)
    if final_m:
        try:
            final = json.loads(final_m.group(1))
        except json.JSONDecodeError:
            final = {"text": final_m.group(1).strip()}
        return {"thought": thought, "final_answer": final}

    action_m = re.search(r"(?:Action|行动|动作)\s*[:：]\s*(\w+)", cleaned, re.S)
    input_m = re.search(r"(?:Action Input|行动输入)\s*[:：]\s*(\{.*\})", cleaned, re.S)
    if action_m:
        action_name = action_m.group(1).strip()
        if action_name not in ALLOWED_ACTIONS:
            raise AppError(f"模型选择了未知工具：{action_name}")
        action_input = {}
        if input_m:
            try:
                action_input = json.loads(input_m.group(1))
            except json.JSONDecodeError:
                action_input = {"value": input_m.group(1).strip()}
        return {"thought": thought, "action": action_name, "action_input": action_input}

    # 只有 Action 没 Action Input
    for kw in _ACTION_KEYWORDS:
        if kw in cleaned:
            return {"thought": thought, "action": kw, "action_input": {}}

    return None


def _extract_json_substring(cleaned: str) -> Optional[str]:
    """提取文本中的 JSON 对象子串"""
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    return cleaned[start : end + 1]
