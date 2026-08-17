# ==================== 前端 Streamlit 镜像 ====================
FROM python:3.10-slim

WORKDIR /app

# 依赖层缓存（streamlit 单独装，与后端隔离）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 前端代码
COPY frontend/ ./frontend/
COPY .streamlit/ ./.streamlit/

EXPOSE 8501
CMD ["streamlit", "run", "frontend/Home.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
