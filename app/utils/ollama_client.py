"""Ollama 本地大模型统一调用封装

所有大模型能力（对话 / 嵌入 / 精排）的唯一入口。
规范约束：业务代码禁止直接 requests 调用 11434 端口，必须复用本模块。

实现能力：
1. 单轮 / 多轮对话生成（支持流式输出）
2. 文本嵌入向量生成（/api/embed，兼容旧版 /api/embeddings）
3. 文本重排序（/api/rerank，失败自动降级为向量相似度排序）
4. 失败自动重试（默认 2 次），重试失败抛出统一 AppError
"""

import json
import time
from typing import Any, Iterator, Optional

import requests

from app.config import (
    OLLAMA_BASE_URL,
    OLLAMA_CHAT_MODEL,
    OLLAMA_EMBED_MODEL,
    OLLAMA_MAX_RETRIES,
    OLLAMA_RERANK_MODEL,
    OLLAMA_TEMPERATURE,
    OLLAMA_TIMEOUT,
)
from app.utils.common_tools import AppError, cosine_similarity, get_logger

logger = get_logger("ollama_client")


class OllamaClient:
    """Ollama 本地大模型客户端（单例）"""

    _instance: Optional["OllamaClient"] = None

    def __new__(cls) -> "OllamaClient":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        self.base_url = OLLAMA_BASE_URL.rstrip("/")
        self.chat_model = OLLAMA_CHAT_MODEL
        self.embed_model = OLLAMA_EMBED_MODEL
        self.rerank_model = OLLAMA_RERANK_MODEL
        self.timeout = OLLAMA_TIMEOUT
        self.temperature = OLLAMA_TEMPERATURE
        self.max_retries = OLLAMA_MAX_RETRIES

    # ---------- 基础请求 ----------

    def _request(self, path: str, payload: dict, retries: Optional[int] = None) -> dict:
        """POST 请求，带重试机制"""
        url = f"{self.base_url}{path}"
        last_error: Optional[Exception] = None
        times = retries if retries is not None else self.max_retries
        for attempt in range(times + 1):
            try:
                resp = requests.post(url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                last_error = exc
                logger.warning(
                    "Ollama 请求失败 path=%s attempt=%d/%d error=%s",
                    path, attempt + 1, times + 1, exc,
                )
                if attempt < times:
                    time.sleep(1)
        raise AppError(f"Ollama 服务调用失败（{path}）：{last_error}")

    def _get(self, path: str) -> dict:
        """GET 请求"""
        url = f"{self.base_url}{path}"
        try:
            resp = requests.get(url, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.error("Ollama GET 请求失败 path=%s error=%s", path, exc)
            raise AppError(f"Ollama 服务不可用：{exc}") from exc

    # ---------- 对话能力 ----------

    def chat(
        self,
        prompt: str,
        system: Optional[str] = None,
        messages: Optional[list] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
    ) -> Any:
        """对话生成（单轮 / 多轮 / 流式）

        Args:
            prompt: 当前用户消息
            system: 系统提示词（可选）
            messages: 历史多轮消息 [{"role": "user"/"assistant", "content": "..."}]
            temperature: 温度参数，默认取全局配置
            stream: 是否流式输出
        Returns:
            stream=False 返回完整回答字符串；stream=True 返回逐块文本迭代器
        """
        msgs: list = []
        if system:
            msgs.append({"role": "system", "content": system})
        if messages:
            msgs.extend(messages)
        # 多轮场景下 messages 已包含完整上下文时，允许 prompt 传空（不追加空用户消息）
        if prompt and prompt.strip():
            msgs.append({"role": "user", "content": prompt})
        payload = {
            "model": self.chat_model,
            "messages": msgs,
            "stream": stream,
            "options": {
                "temperature": temperature if temperature is not None else self.temperature
            },
        }
        if stream:
            return self._chat_stream(payload)
        data = self._request("/api/chat", payload)
        return data.get("message", {}).get("content", "")

    def _chat_stream(self, payload: dict) -> Iterator[str]:
        """流式对话，逐块产出文本"""
        url = f"{self.base_url}/api/chat"
        try:
            with requests.post(url, json=payload, timeout=self.timeout, stream=True) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        yield content
        except requests.RequestException as exc:
            logger.error("Ollama 流式对话失败 error=%s", exc)
            raise AppError(f"Ollama 流式对话失败：{exc}") from exc

    # ---------- 嵌入能力 ----------

    def embed(self, texts: Any) -> list:
        """生成文本嵌入向量

        Args:
            texts: 单条字符串或字符串列表
        Returns:
            向量列表，与输入文本一一对应
        """
        if isinstance(texts, str):
            texts = [texts]
        text_list = list(texts)
        if not text_list:
            return []
        try:
            data = self._request("/api/embed", {"model": self.embed_model, "input": text_list})
            embeddings = data.get("embeddings") or []
        except AppError:
            # 兼容旧版 /api/embeddings 接口（逐条调用）
            logger.warning("新版嵌入接口调用失败，降级为旧版 /api/embeddings")
            embeddings = [
                self._request("/api/embeddings", {"model": self.embed_model, "prompt": t})
                .get("embedding", [])
                for t in text_list
            ]
        if len(embeddings) != len(text_list):
            raise AppError("嵌入向量数量与输入文本数量不一致")
        return embeddings

    def embed_one(self, text: str) -> list:
        """生成单条文本嵌入向量"""
        return self.embed(text)[0]

    # ---------- 精排能力 ----------

    def rerank(self, query: str, documents: list) -> list:
        """对召回文档重排序

        Args:
            query: 查询语句
            documents: 待排序文档列表
        Returns:
            按相关度降序的 [(原始下标, 相关度分数), ...]
        """
        if not documents:
            return []
        # 未配置 rerank 模型 → 直接走向量相似度降级（Ollama 官方库暂无 reranker 模型，跳过精排）
        if not (self.rerank_model and self.rerank_model.strip()):
            logger.debug("未配置 rerank 模型，使用向量相似度降级排序")
            return self._rerank_by_embedding(query, documents)
        try:
            data = self._request(
                "/api/rerank",
                {"model": self.rerank_model, "query": query, "documents": list(documents)},
            )
            results = data.get("results") or []
            scored = [
                (int(item.get("index", 0)), float(item.get("relevance_score", 0.0)))
                for item in results
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored
        except AppError as exc:
            logger.warning("rerank 接口调用失败，降级为向量相似度排序：%s", exc)
            return self._rerank_by_embedding(query, documents)

    def _rerank_by_embedding(self, query: str, documents: list) -> list:
        """降级方案：基于嵌入向量余弦相似度排序"""
        query_vec = self.embed_one(query)
        doc_vecs = self.embed(documents)
        scores = [(i, cosine_similarity(query_vec, doc_vecs[i])) for i in range(len(documents))]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    # ---------- 服务状态 ----------

    def check_health(self) -> bool:
        """检查 Ollama 服务是否可用"""
        try:
            data = self._get("/api/tags")
            return isinstance(data, dict)
        except AppError:
            return False

    def list_models(self) -> list:
        """列出本地已安装的模型名"""
        try:
            data = self._get("/api/tags")
            return [m.get("name", "") for m in data.get("models", [])]
        except AppError:
            return []
