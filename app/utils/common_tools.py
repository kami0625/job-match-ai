"""通用工具函数模块

提供统一日志、统一响应封装、业务异常、JSON 解析、向量相似度等跨模块复用能力。
"""

import json
import logging
import re
import sys
import uuid
from datetime import datetime
from typing import Any


# ============ 统一响应封装 ============


def success_response(data: Any = None, message: str = "success") -> dict:
    """成功响应，统一格式 {code, message, data}"""
    return {"code": 200, "message": message, "data": data}


def error_response(message: str = "服务内部错误", code: int = 500, data: Any = None) -> dict:
    """失败响应，统一格式 {code, message, data}"""
    return {"code": code, "message": message, "data": data}


class AppError(Exception):
    """业务异常基类，携带业务状态码（200/400/404/500）"""

    def __init__(self, message: str = "业务处理失败", code: int = 500):
        super().__init__(message)
        self.message = message
        self.code = code


# ============ 统一日志工具 ============

_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """获取统一格式的日志器

    Args:
        name: 模块名，如 "rag_service"
    Returns:
        logging.Logger 实例
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


# ============ 通用工具函数 ============


def generate_id(prefix: str = "") -> str:
    """生成短 UUID，可选业务前缀，如 generate_id("doc") -> doc_1a2b3c4d5e6f"""
    return f"{prefix}_{uuid.uuid4().hex[:12]}" if prefix else uuid.uuid4().hex[:12]


def safe_filename(filename: str) -> str:
    """清理文件名中的不安全字符，防止路径穿越"""
    name = re.sub(r"[\\/:*?\"<>|\s]+", "_", filename)
    return name.strip("_") or "unnamed"


def now_str(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """当前时间字符串"""
    return datetime.now().strftime(fmt)


def extract_json_from_text(text: str) -> Any:
    """从大模型输出中提取 JSON 对象/数组

    兼容 ```json 代码块、前后杂散文本等情况，提取失败抛出 AppError。

    Args:
        text: 大模型原始输出文本
    Returns:
        解析后的 JSON 对象（dict / list）
    """
    if not text:
        raise AppError("模型输出为空，无法解析 JSON")
    cleaned = re.sub(r"```(?:json)?", "", text).strip("` \n")
    for start_char, end_char in (("{", "}"), ("[", "]")):
        start = cleaned.find(start_char)
        if start == -1:
            continue
        end = cleaned.rfind(end_char)
        if end <= start:
            continue
        candidate = cleaned[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise AppError("无法从模型输出中解析出合法 JSON")


def cosine_similarity(vec_a: list, vec_b: list) -> float:
    """计算两个向量的余弦相似度

    Args:
        vec_a: 向量 A
        vec_b: 向量 B
    Returns:
        相似度，取值范围 [0, 1]
    """
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# SQL 中 `%` 字符的占位符（service 层用此替换 `%` 后再调 mysql.query，让 `%` 绕过
# PyMySQL 的 format 占位符识别、避免 LIKE 子句里的通配符 `%` 被当作 `%s` 处理报错）
SQL_PCT_PLACEHOLDER = "<_SQL_PCT_>"
