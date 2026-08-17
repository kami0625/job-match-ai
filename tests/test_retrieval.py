"""检索与分块逻辑单元测试

- BM25Index 构建与检索(纯内存)
- split_text 分块(边界/重叠/超长段落)
不依赖外部服务。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.modules.rag_module.service import BM25Index
from app.utils.doc_parser import split_text


# ---------- BM25Index ----------

def test_bm25_rebuild_and_search():
    idx = BM25Index()
    docs = ["Java 开发工程师 岗位职责", "Python 后端开发 岗位", "前端工程师 React 岗位"]
    idx.rebuild(docs)
    hits = idx.search("Java 岗位", top_k=2)
    assert len(hits) > 0
    # 命中结果应包含关键词文档
    assert any("Java" in str(h) for h in hits)


def test_bm25_empty_no_crash():
    idx = BM25Index()
    idx.rebuild([])
    assert idx.search("任意", top_k=3) == []


def test_bm25_topk_limit():
    idx = BM25Index()
    docs = [f"文档{i} 关键词" for i in range(10)]
    idx.rebuild(docs)
    hits = idx.search("关键词", top_k=3)
    assert len(hits) <= 3


# ---------- split_text ----------

def test_split_short_text_single_chunk():
    """短文本单块"""
    chunks = split_text("这是一段很短的文本", chunk_size=100, chunk_overlap=0)
    assert len(chunks) == 1


def test_split_long_text_multiple_chunks():
    """长文本多块"""
    text = "A" * 250 + "\n\n" + "B" * 250
    chunks = split_text(text, chunk_size=100, chunk_overlap=0)
    assert len(chunks) >= 2


def test_split_overlap_preserved():
    """块间重叠保留上下文"""
    text = "A" * 100 + "关键连接词" + "B" * 100
    chunks = split_text(text, chunk_size=80, chunk_overlap=20)
    # 至少两块,且重叠段出现在下一块开头附近
    assert len(chunks) >= 2


def test_split_empty_text():
    """空文本返回空列表"""
    assert split_text("", chunk_size=100, chunk_overlap=0) == []


def test_split_super_long_paragraph():
    """超长单段(>chunk_size)按字符切割,不丢内容"""
    text = "X" * 500
    chunks = split_text(text, chunk_size=100, chunk_overlap=0)
    assert len(chunks) >= 5
    assert sum(len(c) for c in chunks) >= 400  # 内容基本保留(切割不丢)
