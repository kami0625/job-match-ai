"""Agent 记忆与会话状态封装

职责：
1. 保存单轮会话上下文（LLM 对话历史 messages、任务信息）
2. 缓存中间打分结果（resume_struct / job_requirements / score），
   同一会话重复请求时直接复用，避免重复调用大模型
3. 提供会话创建、获取、清除、全部清除能力

实现：内存字典存储（单机部署足够），线程安全。
"""

import threading
import time
from typing import Any, Optional

from app.utils.common_tools import generate_id, get_logger

logger = get_logger("agent_memory")


class AgentSession:
    """单个 Agent 会话状态"""

    def __init__(self, session_id: str) -> None:
        self.session_id: str = session_id
        self.created_at: float = time.time()
        self.updated_at: float = time.time()
        self.messages: list = []          # ReAct 多轮对话历史
        self.task: dict = {}              # 任务信息（filename/target_job/resume_text）
        # 中间结果缓存，避免重复调用大模型
        self.resume_struct: Optional[dict] = None
        self.job_requirements: Optional[dict] = None
        self.score: Optional[dict] = None
        self.suggestion: Optional[dict] = None
        self.final_result: Optional[dict] = None

    def touch(self) -> None:
        """刷新更新时间"""
        self.updated_at = time.time()


class AgentMemory:
    """会话记忆存储（内存字典 + 线程锁）"""

    _instance: Optional["AgentMemory"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "AgentMemory":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        self._sessions: dict = {}
        self._lock = threading.Lock()
        self._max_sessions: int = 100  # 最多保留会话数，防止内存膨胀

    def create_session(self, session_id: Optional[str] = None) -> AgentSession:
        """创建新会话（指定 id 或自动生成）"""
        sid = session_id or generate_id("agent")
        with self._lock:
            session = AgentSession(sid)
            self._sessions[sid] = session
            # 简单淘汰：超出上限时清理最旧的会话
            if len(self._sessions) > self._max_sessions:
                oldest = min(self._sessions.values(), key=lambda s: s.updated_at)
                self._sessions.pop(oldest.session_id, None)
            logger.info("创建 Agent 会话 session_id=%s", sid)
            return session

    def get_session(self, session_id: str) -> Optional[AgentSession]:
        """获取会话，不存在返回 None"""
        with self._lock:
            return self._sessions.get(session_id)

    def get_or_create(self, session_id: Optional[str] = None) -> AgentSession:
        """获取或创建会话（新请求默认新建，保证单轮独立）"""
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session
        return self.create_session(session_id)

    def clear_session(self, session_id: str) -> bool:
        """清空指定会话，返回是否存在"""
        with self._lock:
            existed = self._sessions.pop(session_id, None) is not None
        if existed:
            logger.info("清除 Agent 会话 session_id=%s", session_id)
        return existed

    def clear_all(self) -> int:
        """清空全部会话，返回清理数量"""
        with self._lock:
            count = len(self._sessions)
            self._sessions.clear()
        logger.info("清空全部 Agent 会话，共 %d 个", count)
        return count

    def session_count(self) -> int:
        """当前会话数"""
        with self._lock:
            return len(self._sessions)
