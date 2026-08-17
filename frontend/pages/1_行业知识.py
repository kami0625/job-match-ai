"""板块 A：岗位智能匹配（RAG 系统页面）

用户在本页可独立操作：上传岗位文档、查岗位数据库、行业知识问答、简历岗位匹配。
所有功能底层走 FastAPI 的 /api/rag/* 接口，不经过 Agent，是独立功能。
"""

import base64
import json
from pathlib import Path
import sys

import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ui import load_css, render_sidebar_nav

st.set_page_config(
    page_title="行业知识",
    page_icon="",
    layout="centered",
    initial_sidebar_state="expanded",
)
load_css()
render_sidebar_nav(current="pages/1_行业知识.py")

if "backend_url" not in st.session_state:
    st.session_state.backend_url = "http://127.0.0.1:8000"
backend_url = st.session_state.backend_url

st.markdown('<h1 class="jm-page-title">行业知识</h1>', unsafe_allow_html=True)


def post_backend(url: str, **kwargs):
    """POST 请求后端，统一处理连接异常"""
    timeout = kwargs.pop('timeout', 300)
    try:
        resp = requests.post(url, timeout=timeout, **kwargs)
        return resp.json()
    except Exception as exc:
        st.error(f"后端服务连接失败：{exc}")
        return None


def format_salary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """为 DataFrame 中存在的 salary_min / salary_max 列加上 K 单位（前端展示用，DB 仍存 INT）；
    值为 0（面议/未标注）时显示"面议"。"""
    for col in ("salary_min", "salary_max"):
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: f"{int(v)}K" if pd.notna(v) and int(v) > 0 else "面议"
            )
    return df


def get_backend(url: str, **kwargs):
    """GET 请求后端，统一处理连接异常"""
    try:
        resp = requests.get(url, timeout=60, **kwargs)
        return resp.json()
    except Exception as exc:
        st.error(f"后端服务连接失败：{exc}")
        return None


st.title("📚 行业知识")

tab_upload, tab_chat = st.tabs(
    ["📚 行业知识库", "💬 知识问答"]
)

# ==================== Tab1 行业知识库上传 ====================

with tab_upload:
    st.subheader("📚 行业知识库文档上传")
    st.info(
        "**本模块用途：把行业知识文档（岗位说明书 / 面试真题 / 行业报告 / 技能教程）入库到向量库，"
        "供右下「知识问答 RAG」做检索问答用。**\n\n"
        "这里不是岗位数据库,岗位查询请切换到「岗位智能检索」"
        "（自动多源实时检索真实岗位）。"
    )
    uploaded = st.file_uploader("选择文档", type=["pdf", "docx"], key="rag_upload")
    if uploaded is not None and st.button("📥 上传入库", key="btn_upload"):
        with st.spinner("正在解析并写入向量库..."):
            body = post_backend(
                f"{backend_url}/api/rag/upload",
                files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
            )
        if body and body.get("code") == 200:
            data = body["data"]
            st.success(
                f"入库成功：文档ID {data['doc_id']}，分块数 {data['chunk_count']}，"
                f"时间 {data['create_time']}"
            )
            st.json(data)
        elif body:
            st.error(body.get("message"))

# ==================== Tab2 岗位智能检索 ====================

with tab_chat:
    st.subheader("行业知识智能问答")
    st.caption("基于向量知识库的混合检索问答")
    question = st.text_area(
        "输入你的问题",
        placeholder="例如：Java 高级工程师通常需要具备哪些核心能力？",
        height=100,
        key="rag_question",
    )
    if st.button("💬 提问", key="btn_chat"):
        if question.strip():
            # 流式渲染：SSE 事件逐条消费，打字机效果
            answer_placeholder = st.empty()
            full_answer = ""
            query_rewritten = None
            rewrite_failed = False
            sources: list = []
            error_msg = None
            try:
                resp = requests.post(
                    f"{backend_url}/api/rag/chat/stream",
                    json={"query": question, "top_k": 3},
                    stream=True,
                    timeout=600,
                )
                resp.raise_for_status()
                for line in resp.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                    try:
                        event = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
                    etype = event.get("type")
                    if etype == "meta":
                        query_rewritten = event.get("query_rewritten")
                        rewrite_failed = bool(event.get("rewrite_failed"))
                    elif etype == "source":
                        sources = event.get("sources", [])
                    elif etype == "delta":
                        full_answer += event.get("content", "")
                        answer_placeholder.markdown(full_answer + "▌")
                    elif etype == "error":
                        error_msg = event.get("message")
                        break
                    elif etype == "done":
                        break
            except Exception as exc:
                error_msg = f"后端服务连接失败：{exc}"

            if error_msg:
                st.error(error_msg)
            else:
                answer_placeholder.markdown(full_answer or "（无回答内容，请调整问题或先上传知识文档）")
                if query_rewritten:
                    if rewrite_failed:
                        st.caption(f"🔍 查询改写不可用,已使用原文检索:{query_rewritten[:50]}")
                    else:
                        st.caption(f"🔍 查询改写:{query_rewritten}")
                if sources:
                    with st.expander(f"📎 引用来源（{len(sources)} 条）"):
                        for i, src in enumerate(sources, 1):
                            st.markdown(f"**来源 {i}** · {src['file_name']} · 相关度 {src['score']}")
                            st.text(src["content"][:200])
        else:
            st.warning("请输入问题")

# ==================== Tab4 简历岗位匹配 ====================
