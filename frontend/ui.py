"""前端 UI 工具：白底极简（Notion/Linear 风格）

无渐变、无 emoji、克制专业。
"""

from pathlib import Path

import streamlit as st

_CSS_PATH = Path(__file__).parent / "assets" / "style.css"


def load_css() -> None:
    """注入全局样式（每次页面访问都注入，Streamlit 切页面会清掉 DOM）"""
    try:
        css = _CSS_PATH.read_text(encoding="utf-8")
        st.html(f"<style>{css}</style>")
    except FileNotFoundError:
        pass


def render_sidebar_nav(current: str = "") -> None:
    """侧边栏简洁文字导航（Notion 风格,无 emoji）

    Args:
        current: 当前页路径(用于高亮,如 "Home.py" 或 "pages/1_行业知识.py")
    """
    with st.sidebar:
        st.markdown(
            '<div style="padding:6px 4px 14px 4px;font-weight:700;font-size:0.95rem;letter-spacing:-0.01em;">求职匹配 AI</div>',
            unsafe_allow_html=True,
        )
        st.caption("本地推理 · v1.0")
        st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)

        items = [
            ("首页", "Home.py"),
            ("行业知识", "pages/1_行业知识.py"),
            ("岗位匹配", "pages/2_岗位匹配.py"),
            ("简历评分优化", "pages/3_简历评分优化.py"),
        ]
        # 用 st.button + switch_page(子页面不能 st.page_link 到 Home.py,统一用按钮更稳)
        for label, page in items:
            if st.button(label, key=f"nav_{page}", use_container_width=True):
                st.switch_page(page)

        st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)
        st.caption("运行环境")
        st.caption("FastAPI · Streamlit · Ollama")


def render_status_cards(health: dict) -> None:
    """四大服务状态卡片（白底极简,横向排列）"""
    items = [
        ("FastAPI", health.get("service")),
        ("Ollama", health.get("ollama")),
        ("Chroma", health.get("chroma")),
        ("MySQL", health.get("mysql")),
    ]
    cards = ""
    for name, val in items:
        ok = val == "ok"
        status_txt = "运行中" if ok else "异常"
        dot_cls = "ok" if ok else "bad"
        cards += (
            f'<div class="jm-status-card">'
            f'<span class="jm-status-dot {dot_cls}"></span>'
            f'<div><div class="jm-status-name">{name}</div>'
            f'<div class="jm-status-val">{status_txt}</div></div>'
            f'</div>'
        )
    st.html(f'<div class="jm-status-grid">{cards}</div>')
    if health.get("models"):
        st.caption("本地模型：" + "、".join(str(m) for m in health["models"]))