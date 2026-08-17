"""pytest 全局配置：路径引导 + 公共 fixtures"""
import sys
from pathlib import Path

import pytest

# 项目根目录加入 sys.path（pytest 从 tests/ 运行时仍可 import app.*）
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def no_llm_dependency(monkeypatch):
    """强制所有测试不依赖 Ollama LLM(保证 CI 离线可跑)

    tool_resume_parser 内部优先 LLM 抽取,失败降级规则提取;
    这里让 ollama.chat 直接抛异常 → 所有测试统一走规则路径,结果可复现。
    test_api 自身的 mock 不受影响(monkeypatch 按 fixture 隔离)。
    """
    from app.utils.ollama_client import OllamaClient
    from app.utils.common_tools import AppError

    def _no_llm(*args, **kwargs):
        # 抛 AppError 与真实 Ollama 失败路径一致,触发 parser 的规则提取降级
        raise AppError("CI 模式:LLM 不可用,降级规则路径", code=503)

    monkeypatch.setattr(OllamaClient, "chat", _no_llm)
