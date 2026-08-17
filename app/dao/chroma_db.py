"""Chroma 向量库连接与基础操作封装

基于 chromadb.PersistentClient 持久化存储，提供：
1. 集合获取 / 创建（默认 job_knowledge_base，余弦距离）
2. 文档批量写入（含元数据：doc_id、file_name、chunk_id、create_time）
3. 向量语义检索
4. 按 doc_id 删除（增量更新支持）
5. 集合统计与健康检查

嵌入能力统一复用 app/utils/ollama_client.py 的 OllamaClient，全程本地推理。
"""

from typing import Optional

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from app.config import CHROMA_COLLECTION_NAME, CHROMA_DB_PATH, RAG_VECTOR_TOP_K
from app.utils.common_tools import AppError, get_logger
from app.utils.ollama_client import OllamaClient

logger = get_logger("chroma_db")


class OllamaEmbeddingFunction(EmbeddingFunction):
    """对接本地 Ollama 的嵌入函数，供 Chroma 集合使用"""

    def __init__(self) -> None:
        self._client = OllamaClient()

    def __call__(self, input: Documents) -> Embeddings:
        texts = list(input) if isinstance(input, list) else [input]
        return self._client.embed(texts)


class ChromaDB:
    """Chroma 向量库访问封装（单例）"""

    _instance: Optional["ChromaDB"] = None

    def __new__(cls) -> "ChromaDB":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        self._client: Optional[chromadb.ClientAPI] = None
        self._collection = None
        self._embedding_fn = OllamaEmbeddingFunction()

    def get_client(self) -> chromadb.ClientAPI:
        """获取持久化客户端"""
        if self._client is None:
            self._client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        return self._client

    def get_collection(self, force_create: bool = False):
        """获取集合，不存在时自动创建"""
        if self._collection is None:
            client = self.get_client()
            try:
                self._collection = client.get_collection(CHROMA_COLLECTION_NAME)
            except Exception:
                logger.info("集合 %s 不存在，创建中...", CHROMA_COLLECTION_NAME)
                self._collection = client.create_collection(
                    name=CHROMA_COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=self._embedding_fn,
                )
        return self._collection

    def add_documents(self, ids: list, documents: list, metadatas: list) -> None:
        """批量写入文档（含元数据）"""
        if not documents:
            return
        try:
            self.get_collection().add(ids=ids, documents=documents, metadatas=metadatas)
            logger.info("向量库写入 %d 条文档", len(documents))
        except Exception as exc:
            logger.error("向量库写入失败 error=%s", exc)
            raise AppError(f"向量库写入失败：{exc}") from exc

    def semantic_search(self, query: str, top_k: Optional[int] = None) -> list:
        """向量语义检索

        Args:
            query: 查询语句
            top_k: 返回条数，默认取全局配置
        Returns:
            [{"id","document","metadata","distance"}, ...]，distance 越小越相似
        """
        k = top_k or RAG_VECTOR_TOP_K
        try:
            result = self.get_collection().query(
                query_texts=[query],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )
            ids = (result.get("ids") or [[]])[0]
            docs = (result.get("documents") or [[]])[0]
            metas = (result.get("metadatas") or [[]])[0]
            dists = (result.get("distances") or [[]])[0]
            items = []
            for i in range(len(ids)):
                items.append(
                    {
                        "id": ids[i],
                        "document": docs[i],
                        "metadata": metas[i] or {},
                        "distance": dists[i] if i < len(dists) else None,
                    }
                )
            return items
        except Exception as exc:
            logger.error("向量检索失败 error=%s", exc)
            raise AppError(f"向量检索失败：{exc}") from exc

    def delete_by_doc_id(self, doc_id: str) -> None:
        """按文档 ID 删除（增量更新支持）"""
        try:
            self.get_collection().delete(where={"doc_id": doc_id})
            logger.info("已删除文档 doc_id=%s", doc_id)
        except Exception as exc:
            logger.error("向量库删除失败 error=%s", exc)
            raise AppError(f"向量库删除失败：{exc}") from exc

    def exists_by_doc_id(self, doc_id: str) -> bool:
        """判断指定 doc_id 是否已存在于向量库（重复入库去重用）"""
        try:
            result = self.get_collection().get(where={"doc_id": doc_id}, limit=1)
            return bool(result.get("ids"))
        except Exception as exc:
            logger.warning("向量库查重失败 doc_id=%s error=%s", doc_id, exc)
            return False

    def count(self) -> int:
        """集合内文档总数"""
        try:
            return self.get_collection().count()
        except Exception:
            return 0

    def check_health(self) -> bool:
        """连接健康检查"""
        try:
            self.get_collection()
            return True
        except Exception:
            return False
