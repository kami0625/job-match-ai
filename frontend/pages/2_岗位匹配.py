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
    page_title="岗位匹配",
    page_icon="",
    layout="centered",
    initial_sidebar_state="expanded",
)
load_css()
render_sidebar_nav(current="pages/2_岗位匹配.py")

if "backend_url" not in st.session_state:
    st.session_state.backend_url = "http://127.0.0.1:8000"
backend_url = st.session_state.backend_url

st.markdown('<h1 class="jm-page-title">岗位匹配</h1>', unsafe_allow_html=True)


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


st.title("🎯 岗位匹配")

tab_query, tab_match = st.tabs(
    ["🗄️ 岗位智能检索", "📊 简历岗位匹配"]
)

# ==================== Tab1 行业知识库上传 ====================

with tab_query:
    st.subheader("🎯 国内岗位智能检索")
    st.caption(
        "面向国内用户：本地缓存 + Adzuna CN 实时拉取真实岗位。\n\n"
        "💡 **数据流**：本地库（精选国内岗位）→ 命中不足时自动调 Adzuna cn 域（合规公开聚合）→ 入库缓存。"
        "无需手动喂数据，打开就能搜。"
    )

    st.divider()
    st.markdown("##### 🚀 关键词检索（本地库优先）")
    col1, col2, col3 = st.columns([3, 1.5, 1])
    with col1:
        kw = st.text_input("岗位关键词", placeholder="如: Java开发 / 数据分析师 / Agent开发", key="ms_kw")
    with col2:
        city = st.text_input("城市", placeholder="如: 杭州 / 北京", key="ms_city")
    with col3:
        salary_min = st.number_input("薪资下限K/月(>)", min_value=0, max_value=200, value=0, step=1, key="ms_sal")

    # 拉取数据源列表
    srcs_body = get_backend(f"{backend_url}/api/rag/jobs/sources")
    available_sources = [
        s for s in (srcs_body or {}).get("data", {}).get("sources", [])
        if s.get("available")
    ]
    # 默认勾选 local + tencent + adzuna(面向国内:本地缓存 + 腾讯官网公开招聘 + Adzuna 国际)
    # 每个数据源按是否配置自动判断 available,缺 token/key 自动跳过
    # 数据源按配置自动判断可用,缺 key 自动跳过
    all_source_names = ["local"] + [s["name"] for s in available_sources]
    default_chosen = [x for x in ("local", "tencent", "adzuna") if x in all_source_names]

    chosen = st.multiselect(
        "启用数据源",
        options=all_source_names,
        default=default_chosen,
        key="ms_sources",
    )

    if st.button("🔍 实时检索", key="btn_ms_search", type="primary"):
        if not kw.strip():
            st.warning("请输入岗位关键词")
        else:
            with st.spinner("查询中（本地+外部 API）..."):
                body = post_backend(
                    f"{backend_url}/api/rag/jobs/search",
                    json={
                        "keywords": kw,
                        "city": city,
                        "salary_min": int(salary_min),
                        "sources": chosen,
                        "limit": 20,
                    },
                    timeout=120,
                )
            if body and body.get("code") == 200:
                data = body["data"]
                # 来源统计
                cols_info = st.columns([2, 2, 2, 2])
                cols_info[0].metric("📊 总数", data["total"])
                cols_info[1].metric("📌 本地命中", data["local_count"])
                cols_info[2].metric("🌐 外部新增", data["external_count"])
                cols_info[3].metric("⏱️ 耗时", f"{data['elapsed_seconds']}s")
                sources_used = ", ".join(data.get("sources_used", [])) or "(无)"
                st.caption(f"本次使用数据源: [{sources_used}]")
    
                if data["items"]:
                    st.dataframe(format_salary_columns(pd.DataFrame(data["items"])), use_container_width=True)
                else:
                    st.info("💡 本地与外部数据源均无匹配，可调整关键词/薪资再试")
            elif body:
                st.error(body.get("message"))


# ==================== Tab3 行业知识问答 ====================

with tab_match:
    st.subheader("🎯 简历与岗位实时匹配（目标岗位自动多源查 JD）")
    st.caption(
        "**输入目标岗位名称，系统自动从本地 + 外部 API 实时检索 JD**（无须选择库里已有岗位）。"
        "接着上传简历，系统对每个候选 JD 跑大模型打分，排序出 top 1 详细结果 + top 3 对比。"
    )

    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown("**① 简历文本**")
        resume_text = st.text_area(
            "粘贴简历文本",
            value=st.session_state.get("parsed_resume_text", ""),
            placeholder="粘贴简历纯文本（或在上传区上传文件自动解析）",
            height=220,
            key="match_resume_input",
        )
        resume_file = st.file_uploader(
            "或上传简历文件（PDF/DOCX/TXT，自动解析）",
            type=["pdf", "docx", "txt"],
            key="match_file",
        )
        if resume_file is not None and st.button("📄 解析简历", key="btn_parse"):
            if resume_file.name.lower().endswith(".txt"):
                resume_text = resume_file.getvalue().decode("utf-8", errors="ignore")
                st.session_state.parsed_resume_text = resume_text
                st.success(f"TXT 读取成功，共 {len(resume_text)} 字")
                st.rerun()
            else:
                with st.spinner("正在解析简历..."):
                    body = post_backend(
                        f"{backend_url}/api/agent/parse",
                        files={"file": (resume_file.name, resume_file.getvalue(), resume_file.type)},
                    )
                if body and body.get("code") == 200:
                    resume_text = body["data"].get("raw_text", "")
                    st.session_state.parsed_resume_text = resume_text
                    st.success(f"解析成功，共 {len(resume_text)} 字")
                    st.rerun()
                elif body:
                    st.error(body.get("message"))

    with col_r:
        st.markdown("**② 评估方式（选一种）**")
        # 模式选择
        match_mode = st.radio(
            "评估方式",
            options=["🔍 自动搜 JD（按岗位名）", "📋 上传 JD 文本（精准评估）"],
            key="match_mode",
            horizontal=True,
            help="自动搜:系统调多源查候选 JD;上传 JD:用你提供的具体 JD 精准评分",
        )

        if match_mode == "📋 上传 JD 文本（精准评估）":
            st.caption("💡 适用场景:在 Boss/拉勾/智联等看到具体岗位,粘 JD 原文进来,系统直接给你打分+调整建议")
            pasted_jd = st.text_area(
                "目标岗位 JD 文本（从招聘网站复制）",
                placeholder="【字节跳动 - Python Agent 开发工程师】\n工作地点:北京\n薪资:30-50K·15薪\n职位要求:\n1. 熟练 Python/Flask/FastAPI\n2. 有 LLM/RAG/Agent 应用经验\n3. 熟悉 LangChain/LlamaIndex\n...",
                height=200,
                key="match_pasted_jd",
            )
            jd_target = st.text_input(
                "目标岗位名称（用于结果展示）",
                placeholder="如: Python Agent 开发",
                key="match_jd_target",
            )
            if st.button("🎯 评估我的简历", key="btn_match_by_jd", type="primary"):
                if not resume_text or len(resume_text.strip()) < 10:
                    st.warning("请先输入或解析简历文本（至少 10 字）")
                elif not pasted_jd or len(pasted_jd.strip()) < 20:
                    st.warning("请粘贴完整的 JD 文本（至少 20 字）")
                else:
                    with st.spinner("本地大模型正在按你的 JD 精准评分..."):
                        body = post_backend(
                            f"{backend_url}/api/rag/match/by-jd",
                            json={
                                "resume_text": resume_text,
                                "jd_text": pasted_jd,
                                "target_job": jd_target,
                            },
                            timeout=180,
                        )
                    if body and body.get("code") == 200:
                        data = body["data"]
                        st.session_state.by_jd_result = data
                        st.rerun()
                    elif body:
                        st.error(body.get("message"))
        else:
            st.caption("💡 系统自动调多源查该岗位的候选 JD,跟简历比对打分")
            target_job = st.text_input(
                "目标岗位名称",
                placeholder="如: Java开发工程师 / Agent应用开发 / 数据分析师",
                key="match_target_job",
            )
            c1, c2 = st.columns(2)
            with c1:
                target_city = st.text_input("城市（可选）", placeholder="如: 杭州", key="match_city")
            with c2:
                target_salary = st.number_input(
                    "最低薪资 K/月(>)", min_value=0, max_value=200, value=0, step=1, key="match_salary",
                )
            if st.button("🎯 实时匹配", key="btn_match_target", type="primary"):
                if not resume_text or len(resume_text.strip()) < 10:
                    st.warning("请先输入或解析简历文本（至少 10 字）")
                elif not target_job.strip():
                    st.warning("请输入目标岗位名称")
                else:
                    with st.spinner(
                        f"实时检索「{target_job}」的 5 个候选 JD，并对每个跑匹配打分..."
                    ):
                        body = post_backend(
                            f"{backend_url}/api/rag/match/by-target",
                            json={
                                "resume_text": resume_text,
                                "target_job": target_job,
                                "city": target_city,
                                "salary_min": int(target_salary),
                                "top_n": 5,
                            },
                            timeout=180,
                        )
                    if body and body.get("code") == 200:
                        st.session_state.by_target_result = body["data"]
                        st.rerun()
                    elif body:
                        st.error(body.get("message"))
                    st.error(body.get("message"))


# ==================== 结果展示(从 session_state 读取) ====================

# 实时匹配(by-target)结果
if "by_target_result" in st.session_state and st.session_state.by_target_result:
    data = st.session_state.by_target_result
    st.divider()
    st.subheader("📊 实时匹配结果")
    if data.get("candidates_count", 0) == 0:
        st.info(
            f"💡 **未找到「{data.get('target_job', '该岗位')}」的候选 JD**。\n\n"
            "可以尝试：\n"
            "- 换更通用的关键词（如「Java开发」代替「高级Java架构师」）\n"
            "- 放宽城市限制（留空城市字段）\n"
            "- 降低薪资下限\n"
            "- 或切到「📋 上传 JD 文本」模式，手动粘 JD 进来做精准评估"
        )
    else:
        cA, cB, cC = st.columns([2, 2, 2])
        cA.metric("📌 候选 JD 数", data.get("candidates_count", 0))
        cB.metric("评分成功", data.get("scored_count", 0))
        cC.metric("🌐 数据源", "、".join(data.get("sources_used", [])) or "无")
        st.info(data.get("overall_suggestion", ""))
        matched = data.get("matched") or {}
        if matched:
            job = matched.get("job", {})
            m1, m2 = st.columns([1.2, 2])
            with m1:
                st.markdown(
                    f"**🏆 最佳匹配**\n\n"
                    f"- 岗位: **{job.get('job_name', '-')}**\n"
                    f"- 公司: {job.get('company', '-')}\n"
                    f"- 城市: {job.get('city', '-')}\n"
                    f"- 薪资: {job.get('salary_min', 0)}-{job.get('salary_max', 0)}K\n"
                    f"- 来源: {job.get('data_source', '-')}"
                )
                if job.get("source_url"):
                    st.markdown(f"[原始链接]({job['source_url']})")
            with m2:
                score = matched.get("total_score", 0)
                pass_line = data.get("pass_line", 85)
                st.metric("综合匹配度", f"{score} %", delta=f"及格线 {pass_line} %" if score >= pass_line else f"未达及格线 {pass_line} %")
                st.caption(matched.get("description", ""))
                # RAG 模块只展示综合匹配度(文本),分项柱状图留给 Agent 评分模块
                # 这里改用进度条 + 文字标签代替柱状图
                dims = matched.get("dimensions", {}) or {}
                for k, v in dims.items():
                    st.progress(min(max(int(v), 0), 100) / 100, text=f"{k}: {int(v)}%")
            if matched.get("chart_base64"):
                try:
                    st.image(
                        base64.b64decode(matched["chart_base64"]),
                        caption="匹配度分项柱状图（Matplotlib）",
                        use_container_width=True,
                    )
                except Exception:
                    pass  # Streamlit 1.x 内部 width 参数偶发问题,降级隐藏
            # 调整建议
            suggestions = matched.get("suggestions", [])
            if suggestions:
                st.markdown("**💡 调整建议**")
                for s in suggestions:
                    st.markdown(f"- {s}")
            # 其他候选人
            alts = data.get("alternatives", [])
            if alts:
                st.divider()
                st.markdown("**📋 其他候选人对比**")
                for i, alt in enumerate(alts, 2):
                    aj = alt.get("job", {})
                    st.markdown(
                        f"`#{i}` **{alt.get('total_score', 0)} 分** · "
                        f"{aj.get('job_name', '')} · {aj.get('company', '')} · "
                        f"{aj.get('city', '')} · {aj.get('salary_min', 0)}-{aj.get('salary_max', 0)}K · "
                        f"[{aj.get('data_source', '-')}]"
                    )
                    st.caption(alt.get("description", ""))
        if data.get("sources_failed"):
            with st.expander("部分数据源失败(不影响本次结果)"):
                for k, v in data["sources_failed"].items():
                    st.write(f"  - **{k}**: {v}")

# JD 上传(by-jd)结果
if "by_jd_result" in st.session_state and st.session_state.by_jd_result:
    data = st.session_state.by_jd_result
    st.divider()
    st.subheader("📋 JD 精准评估结果")
    job = data.get("job", {})
    score = data.get("total_score", 0)
    pass_line = 85
    m1, m2 = st.columns([1.2, 2])
    with m1:
        st.markdown(
            f"**📋 目标岗位**\n\n"
            f"- 岗位: **{job.get('job_name', '-')}**\n"
            f"- 数据来源: 用户上传的 JD"
        )
    with m2:
        st.metric(
            "综合匹配度",
            f"{score} 分",
            delta=f"及格线 {pass_line} 分" if score >= pass_line else f"未达及格线 {pass_line} 分"
        )
        st.caption(data.get("description", ""))
        # RAG 模块只展示综合匹配度(文本),分项柱状图留给 Agent 评分模块
        dims_bj = data.get("dimensions", {}) or {}
        for k, v in dims_bj.items():
            st.progress(min(max(int(v), 0), 100) / 100, text=f"{k}: {int(v)}%")
    if data.get("chart_base64"):
        st.image(
            base64.b64decode(data["chart_base64"]),
            caption="匹配度分项柱状图（Matplotlib）",
            use_container_width=True,
        )
    suggestions = data.get("suggestions", [])
    if suggestions:
        st.markdown("**💡 调整建议**" if score < pass_line else "**💡 投递建议**")
        for s in suggestions:
            st.markdown(f"- {s}")
