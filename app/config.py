"""全局配置模块

统一从 .env 文件读取所有环境变量，向业务代码暴露全部配置项。
规范约束：业务代码禁止直接读取环境变量，一律从本模块引入。
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录（app 的上一级）
BASE_DIR = Path(__file__).resolve().parent.parent

# 加载 .env 文件（不存在时静默跳过，使用默认值）
load_dotenv(BASE_DIR / ".env", override=False)


def _get_int(key: str, default: int) -> int:
    """安全读取整数类型环境变量"""
    value = os.getenv(key)
    try:
        return int(value) if value not in (None, "") else default
    except ValueError:
        return default


def _get_float(key: str, default: float) -> float:
    """安全读取浮点类型环境变量"""
    value = os.getenv(key)
    try:
        return float(value) if value not in (None, "") else default
    except ValueError:
        return default


# ============ Ollama 本地大模型配置 ============
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_CHAT_MODEL: str = os.getenv("OLLAMA_CHAT_MODEL", "qwen2:7b")
OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_RERANK_MODEL: str = os.getenv("OLLAMA_RERANK_MODEL", "")  # 精排模型（Ollama 官方库暂无可直接 pull 的 reranker 模型；留空启用自动降级为向量相似度排序；若自建 reranker 模型填对应名字）
OLLAMA_TIMEOUT: int = _get_int("OLLAMA_TIMEOUT", 120)
OLLAMA_TEMPERATURE: float = _get_float("OLLAMA_TEMPERATURE", 0.7)
OLLAMA_MAX_RETRIES: int = _get_int("OLLAMA_MAX_RETRIES", 2)

# ============ MySQL 配置 ============
MYSQL_HOST: str = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT: int = _get_int("MYSQL_PORT", 3306)
MYSQL_USER: str = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD: str = os.getenv("MYSQL_PASSWORD", "123456")
MYSQL_DATABASE: str = os.getenv("MYSQL_DATABASE", "job_match_ai")
MYSQL_CHARSET: str = os.getenv("MYSQL_CHARSET", "utf8mb4")

# ============ Chroma 向量库配置 ============
CHROMA_DB_PATH: Path = BASE_DIR / os.getenv("CHROMA_DB_PATH", "data/chroma_db")
CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "job_knowledge_base")

# ============ 本地数据目录 ============
DATA_DIR: Path = BASE_DIR / "data"
UPLOAD_DIR: Path = DATA_DIR / "upload_files"

# ============ 检索与分块参数 ============
RAG_TOP_K: int = _get_int("RAG_TOP_K", 3)           # 精排后返回给大模型的片段数
RAG_BM25_TOP_K: int = _get_int("RAG_BM25_TOP_K", 5)  # BM25 关键词召回数
RAG_VECTOR_TOP_K: int = _get_int("RAG_VECTOR_TOP_K", 5)  # 向量语义召回数
CHUNK_SIZE: int = _get_int("CHUNK_SIZE", 500)        # 分块字符数
CHUNK_OVERLAP: int = _get_int("CHUNK_OVERLAP", 50)   # 分块重叠字符数

# ============ RAG 服务地址（Agent 模块联动使用）============
RAG_SERVICE_URL: str = os.getenv("RAG_SERVICE_URL", "http://127.0.0.1:8000")

# ============ 外部合规岗位数据源配置（Adzuna 等）============
# Adzuna: 免费、覆盖 8+ 国（含 cn/gb/us/sg/in）
# 注册: https://developer.adzuna.com/  5 分钟拿 app_id + app_key
# 留空 → 自动跳过 Adzuna，仅腾讯等免 key 源生效
ADZUNA_APP_ID: str = os.getenv("ADZUNA_APP_ID", "")
ADZUNA_APP_KEY: str = os.getenv("ADZUNA_APP_KEY", "")
# 多国并行查询配置（逗号分隔），默认 cn + gb + us + sg + in（覆盖中国为主 + 海外补充）
ADZUNA_COUNTRIES: list = [
    c.strip().lower()
    for c in os.getenv("ADZUNA_COUNTRIES", "cn,gb,us,sg,in").split(",")
    if c.strip()
]  # type: ignore[assignment]

# ============ 简历评分维度（Agent 模块使用）============
SCORE_DIMENSIONS: list = [
    "技能匹配度",
    "项目完整性",
    "表述专业度",
    "格式规范度",
    "岗位契合度",
]

# ============ 服务配置 ============
APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT: int = _get_int("APP_PORT", 8000)
APP_CORS_ORIGINS: list = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
]


def ensure_data_dirs() -> None:
    """确保本地数据目录存在"""
    CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
