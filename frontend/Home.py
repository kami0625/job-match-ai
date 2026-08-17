"""求职匹配 AI 系统 - 首页

启动方式：
E:\简历项目\.venv\Scripts\python.exe -m streamlit run frontend\Home.py
"""

import sys
from pathlib import Path

import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ui import load_css, render_status_cards, render_sidebar_nav

st.set_page_config(
    page_title="求职匹配 AI 系统",
    page_icon="",
    layout="centered",
    initial_sidebar_state="expanded",
)
load_css()
render_sidebar_nav(current="Home.py")

if "backend_url" not in st.session_state:
    st.session_state.backend_url = "http://127.0.0.1:8000"

st.markdown('<h1 class="jm-page-title">求职匹配 AI 系统</h1>', unsafe_allow_html=True)
st.caption("基于本地大模型的全栈求职助手 · 岗位匹配 + 简历评分优化 · 数据不出本机")

try:
    resp = requests.get(f"{st.session_state.backend_url}/api/health", timeout=10)
    body = resp.json()
    if body.get("code") == 200:
        render_status_cards(body["data"])
    else:
        st.warning(f"后端返回异常：{body.get('message')}")
except Exception:
    st.error("无法连接后端服务，请先启动 FastAPI 后端并确认地址。")

st.markdown('<div class="jm-divider"></div>', unsafe_allow_html=True)


def _module_card(title: str, sub: str, features: list, links: list) -> None:
    """板块卡片：title + 副标签 + 功能点 + 多个按钮入口

    用 st.button + switch_page 跳转(避免 a 链接导致整页 reload 空白)
    """
    st.markdown(f'<div class="jm-module-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="jm-module-sub">{sub}</div>', unsafe_allow_html=True)
    for f in features:
        st.markdown(
            f'<div class="jm-module-feat"><span class="jm-feat-marker"></span>'
            f"<span>{f}</span></div>",
            unsafe_allow_html=True,
        )
    st.markdown('<div class="jm-enter-wrap">', unsafe_allow_html=True)
    for i, (label, page) in enumerate(links):
        if st.button(label, key=f"enter_{page}", use_container_width=True):
            st.switch_page(page)
    st.markdown("</div>", unsafe_allow_html=True)


col_a, col_b = st.columns(2, gap="medium")

with col_a:
    _module_card(
        title="行业知识 + 岗位匹配",
        sub="RAG · KNOWLEDGE BASE",
        features=[
            "上传岗位说明书 / 行业知识文档，自动分块入库向量库",
            "知识问答：混合检索（BM25 + 向量）本地大模型回答",
            "岗位检索：自然语言查询（NL2SQL），多条件筛选",
            "简历匹配：多维度评分，展示综合匹配度",
        ],
        links=[
            ("进入行业知识", "pages/1_行业知识.py"),
            ("进入岗位匹配", "pages/2_岗位匹配.py"),
        ],
    )

with col_b:
    _module_card(
        title="简历评分优化",
        sub="AGENT · RESUME SCORING",
        features=[
            "上传 PDF / Word 简历，自动结构化提取",
            "岗位定向评估：自动检索 JD 或使用粘贴 JD",
            "锚定评分：优秀简历 90 分标杆，多维扣分可解释",
            "优化建议：面向结果的重写示例 + 一键导出报告",
        ],
        links=[
            ("进入简历评分优化", "pages/3_简历评分优化.py"),
        ],
    )

st.html(
    '<div class="jm-footer">FastAPI · Streamlit · Ollama (qwen2:7b / nomic-embed-text) · '
    "Chroma · MySQL 8.0 · 全本地推理，不调用任何云端大模型 API</div>"
)