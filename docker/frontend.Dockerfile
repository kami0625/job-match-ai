# ==================== 前端 Streamlit 镜像 ====================
FROM python:3.10-slim

WORKDIR /app

# 依赖层缓存(用官方 pypi + timeout,CI 跑避免超时)
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=180 -r requirements.txt

# 前端代码
COPY frontend/ ./frontend/
COPY .streamlit/ ./.streamlit/

EXPOSE 8501
CMD ["streamlit", "run", "frontend/Home.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
