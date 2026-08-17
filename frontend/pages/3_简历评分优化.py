"""板块 B：简历评分优化（Agent 页面）

用户上传简历、输入目标岗位名称，点击「开始评估」。
该页面底层运行简历 ReAct Agent 智能体，Agent 内部自动调用 RAG 系统接口获取岗位 JD，
驱动多轮工具调用（获取岗位要求 → 解析简历 → 匹配打分 → 生成建议）后输出结构化结果。

功能：
- 简历拖拽上传、目标岗位输入、执行按钮
- Agent 思考过程流式输出（可开关，调试用）
- 总分仪表盘、四维分项柱状图、弱点清单、可复制修改示例
- 重置会话、导出优化报告
"""

import json
from pathlib import Path
import sys

import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ui import load_css, render_sidebar_nav

st.set_page_config(
    page_title="简历评分优化",
    page_icon="",
    layout="centered",
    initial_sidebar_state="expanded",
)
load_css()
render_sidebar_nav(current="pages/3_简历评分优化.py")

if "backend_url" not in st.session_state:
    st.session_state.backend_url = "http://127.0.0.1:8000"
backend_url = st.session_state.backend_url
if "agent_result" not in st.session_state:
    st.session_state.agent_result = None

st.markdown('<h1 class="jm-page-title">简历评分优化</h1>', unsafe_allow_html=True)


def call_clear(session_id: str = "") -> None:
    """调用后端清空会话"""
    try:
        body = {"session_id": session_id} if session_id else {"all": True}
        requests.post(f"{backend_url}/api/agent/clear", json=body, timeout=15)
    except Exception as exc:
        st.warning(f"会话清理请求失败（不影响页面重置）：{exc}")


def build_markdown_report(result: dict, target_job: str) -> str:
    """组装优化报告 Markdown 文本"""
    lines = ["# 简历评分优化报告", ""]
    lines.append(f"- 目标岗位：{target_job or '（未指定）'}")
    lines.append(f"- 评估模式：{result.get('mode', '')}")
    lines.append(f"- JD 来源：{result.get('jd_message', '')}")
    lines.append("")
    lines.append(f"## 综合得分：**{result.get('total_score')} 分**")
    lines.append("")
    lines.append("## 一、分项维度得分")
    lines.append("")
    lines.append("| 评分维度 | 得分 |")
    lines.append("| ---- | ---- |")
    for name, val in (result.get("dimensions") or {}).items():
        lines.append(f"| {name} | {val} |")
    lines.append("")
    if result.get("matched_skills"):
        lines.append(f"- 命中技能：{'、'.join(result['matched_skills'])}")
    if result.get("missing_skills"):
        lines.append(f"- 缺失技能：{'、'.join(result['missing_skills'])}")
    lines.append("")
    lines.append("## 二、弱点清单")
    lines.append("")
    for w in result.get("weaknesses") or []:
        lines.append(f"- {w}")
    lines.append("")
    lines.append("## 三、优化建议与修改示例")
    lines.append("")
    for item in result.get("suggestions") or []:
        lines.append(f"### {item.get('section', '')}：{item.get('problem', '')}")
        lines.append("")
        lines.append(f"- 建议：{item.get('suggestion', '')}")
        if item.get("before"):
            lines.append(f"- 改写前：{item.get('before')}")
        if item.get("after"):
            lines.append(f"- 改写后：{item.get('after')}")
        lines.append("")
    if result.get("summary"):
        lines.append(f"**整体评价：** {result['summary']}")
        lines.append("")
    return "\n".join(lines)


st.title("📝 简历评分优化助手")

col_input, col_result = st.columns([1, 2], gap="large")

# ==================== 左侧：输入区 ====================

with col_input:
    st.subheader("① 简历上传")
    resume_file = st.file_uploader(
        "拖拽或点击上传简历（PDF / Word）",
        type=["pdf", "docx"],
        key="agent_file",
    )

    st.subheader("② 目标岗位评估方式")
    st.radio(
        "选择一种评估方式:",
        options=["从岗位库自动查 JD", "用我复制的 JD 文本"],
        index=st.session_state.get("agent_eval_idx", 0),
        horizontal=True,
        key="agent_eval_mode",
        help="两种方式二选一,不要同时使用",
    )
    is_kw_mode = st.session_state.get("agent_eval_mode", "从岗位库自动查 JD") == "从岗位库自动查 JD"
    st.session_state["agent_eval_idx"] = 0 if is_kw_mode else 1

    if is_kw_mode:
        target_job = st.text_input(
            "目标岗位名称 / 岗位关键词",
            placeholder="例如:Java 开发工程师",
            key="agent_job",
        )
    else:
        pasted_jd = st.text_area(
            "JD 原文(从 Boss/拉勾等复制的具体岗位描述)",
            placeholder="【字节跳动 - Python Agent 开发工程师】\n薪资:30-50K·15薪\n职位要求:\n1. 熟练 Python/Flask/FastAPI\n2. LLM/RAG/Agent 应用经验\n...",
            height=140,
            key="agent_pasted_jd",
        )

    show_thoughts = st.checkbox("显示 Agent 思考过程", value=True, key="agent_thoughts")

    kw_input = (st.session_state.get("agent_job") or "").strip()
    jd_input = (st.session_state.get("agent_pasted_jd") or "").strip()
    if is_kw_mode:
        input_ok = bool(resume_file) and len(kw_input) >= 2
    else:
        input_ok = bool(resume_file) and len(jd_input) >= 20

    start = st.button(
        "开始评估",
        type="primary",
        use_container_width=True,
        disabled=not input_ok,
    )
    if resume_file is None:
        st.caption("请先上传简历文件")
    elif not input_ok:
        st.caption("请输入目标岗位关键词(至少 2 字)或粘贴 JD(至少 20 字)")

    st.divider()
    if st.button("🔄 重置会话", use_container_width=True, key="agent_reset"):
        call_clear(st.session_state.get("agent_session_id", ""))
        for key in ("agent_result", "agent_session_id", "agent_file", "agent_job",
                    "agent_pasted_jd", "agent_log", "agent_running"):
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# ==================== 右侧：结果区 ====================

with col_result:
    if start and resume_file is not None:
        # loading 防重复点击:评估期间禁用开始按钮
        st.session_state.agent_running = True
        # Agent 思考过程容器（流式更新）
        thought_holder = st.empty()
        log_lines: list = []

        def render_log() -> None:
            if show_thoughts and log_lines:
                thought_holder.markdown("**🤖 Agent 思考过程**\n\n" + "\n\n".join(log_lines))

        # 发起 SSE 流式评估
        try:
            # 读取当前 session_state(单选模式已互斥清空另一项)
            target_job_now = st.session_state.get("agent_job", "")
            jd_text_input = st.session_state.get("agent_pasted_jd", "")
            data = {"target_job": target_job_now or ""}
            if jd_text_input and jd_text_input.strip() and len(jd_text_input.strip()) >= 20:
                data["jd_text"] = jd_text_input.strip()
            resp = requests.post(
                f"{backend_url}/api/agent/evaluate/stream",
                files={"file": (resume_file.name, resume_file.getvalue(), resume_file.type)},
                data=data,
                stream=True,
                timeout=900,
            )
            resp.raise_for_status()
            final_result = None
            error_msg = None
            status_holder = st.empty()
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                try:
                    event = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                etype = event.get("type")

                if etype == "start":
                    st.session_state.agent_session_id = event.get("session_id")
                    status_holder.info(f"🔄 Agent 评估中,请稍候...")
                elif etype == "step":
                    log_lines.append(
                        f"**Round {event.get('iteration')}**\n\n"
                        f"💭 思考：{event.get('thought', '')}\n\n"
                        f"🔧 调用：`{event.get('action')}`\n\n"
                        f"📥 观察：{event.get('observation', '')[:400]}"
                    )
                    render_log()
                elif etype == "fallback":
                    st.info(event.get("message", ""))
                elif etype == "result":
                    final_result = event.get("data")
                    status_holder.empty()
                elif etype == "error":
                    error_msg = event.get("message")
                    status_holder.empty()

            if error_msg:
                st.error(error_msg)
            elif final_result is None:
                st.error("Agent 未返回有效结果，请稍后重试")
            else:
                st.session_state.agent_result = final_result
                render_log()
                if show_thoughts:
                    thought_holder.empty()
        except Exception as exc:
            st.error(f"后端服务连接失败:{exc}\n\n💡 若使用的是岗位关键词检索模式,可尝试手动粘贴 JD 文本后重新评估")
        finally:
            st.session_state.agent_running = False

    result = st.session_state.get("agent_result")
    if result is not None:
        st.divider()

        # 边界保护:如果总分 0 或 dimensions 空(说明 target_job 空且 JD 太短,Agent 走了通用评估兜底)
        total = result.get("total_score", 0)
        dims = result.get("dimensions", {}) or {}
        if total == 0 or not dims:
            st.error("评估结果为空,通常是 target_job 和 JD 文本都没提供。请:")
            st.markdown(
                "1. 在**「② 目标岗位」**填岗位名(如 `Java 开发工程师`)\n"
                "2. 或在**「📋 上传 JD 文本」**里粘具体 JD 原文(至少 20 字)\n"
                "3. 重新点 **🚀 开始评估**"
            )
            if st.button("🔄 重新评估", key="btn_re_evaluate"):
                st.session_state.agent_result = None
                st.rerun()
            st.stop()

        # 1) 总分仪表盘
        c1, c2, c3 = st.columns([1, 1, 1])
        c1.metric("综合得分", f"{result.get('total_score', 0)} 分",
            delta=f"{'达到' if result.get('pass') else '未达'}{result.get('pass_line', 80)} 分及格线",
            delta_color="normal" if result.get('pass') else "inverse")
        c2.metric("评估模式", result.get("mode", ""))
        c3.metric("评分维度", "技能/项目/学历/业务")
        st.caption("评分基准:以「优秀简历 = 95 分」为标杆,内容质量+匹配度综合扣分")
        st.progress(min(max(int(result.get("total_score", 0)), 0), 100) / 100, text="综合匹配度")

        # 1.5) 显性告知本次评估使用的岗位(容错透明度)
        used_job = result.get("used_job_name") or ""
        if used_job:
            st.info(f"📌 **本次评估使用的岗位：{used_job}**")
        if result.get("jd_message"):
            st.caption(f"📎 {result['jd_message']}")

        # 2) 分项柱状图(强制 clip 到 0-100,避免超界)
        st.markdown("**📊 分项维度得分**")
        clipped_dims = {k: max(0, min(100, v)) for k, v in dims.items()}
        dim_df = pd.DataFrame([clipped_dims]).T.rename(columns={0: "得分"})
        st.bar_chart(dim_df, height=280)

        # 技能命中/缺失
        if result.get("matched_skills") or result.get("missing_skills"):
            st.caption(
                f"命中技能：{'、'.join(result['matched_skills']) or '无'}"
                f"缺失技能：{'、'.join(result['missing_skills']) or '无'}"
            )
        if result.get("jd_message"):
            st.info(f"📎 {result['jd_message']}")

        # 3) 弱点清单
        st.markdown("**弱点清单**")
        for w in result.get("weaknesses") or []:
            st.markdown(f"- {w}")

        # 4) 优化建议 + 可复制修改示例
        st.markdown("**💡 优化建议与修改示例**")
        suggestions = result.get("suggestions") or []
        if suggestions:
            for i, item in enumerate(suggestions, 1):
                with st.expander(
                    f"{item.get('section', '建议')}：{item.get('problem', '')}",
                    expanded=(i <= 2),
                ):
                    st.markdown(f"**建议：** {item.get('suggestion', '')}")
                    if item.get("before") and item.get("after"):
                        example = f"改写前：\n{item.get('before')}\n\n改写后：\n{item.get('after')}"
                        st.code(example, language="text")
        else:
            st.info("暂无分项建议")
        if result.get("summary"):
            st.success(f"📋 整体评价：{result['summary']}")

        # 5) 导出报告
        st.divider()
        report_md = build_markdown_report(result, st.session_state.get("agent_job", ""))
        c_exp, c_txt = st.columns(2)
        c_exp.download_button(
            "⬇️ 导出 Markdown 报告",
            data=report_md.encode("utf-8"),
            file_name="简历优化报告.md",
            mime="text/markdown",
            use_container_width=True,
        )
        c_txt.download_button(
            "⬇️ 导出 TXT 报告",
            data=report_md.encode("utf-8"),
            file_name="简历优化报告.txt",
            mime="text/plain",
            use_container_width=True,
        )

        # 6) 候选岗位预览(检索模式):允许用户换一条 JD 重新评估,避免基于错误 JD 打分
        candidates = result.get("candidates") or []
        if len(candidates) > 1:
            st.divider()
            st.markdown(f"**🗂️ 检索到的候选岗位（{len(candidates)} 条）**")
            st.caption("如果上方的评估岗位不准确,可选一条更匹配的 JD 重新评估:")
            for i, cand in enumerate(candidates):
                if cand.get("job_name") == used_job:
                    continue  # 跳过已使用的岗位
                with st.container(border=True):
                    salary_txt = f"{cand.get('salary_min') or '?'}-{cand.get('salary_max') or '?'}K" if cand.get("salary_min") else "面议"
                    st.markdown(
                        f"**{cand.get('job_name')}** · {cand.get('company', '')} · {cand.get('city', '')} · {salary_txt}"
                    )
                    if st.button(f"🎯 用此 JD 重新评估", key=f"cand_{i}", use_container_width=True):
                        st.session_state.agent_pasted_jd = cand.get("jd", "")
                        st.session_state.agent_result = None
                        st.rerun()
