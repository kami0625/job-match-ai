# ==================== 后端 FastAPI 镜像 ====================
FROM python:3.10-slim

WORKDIR /app

# 依赖层缓存(用官方 pypi + timeout,CI 跑避免超时)
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=180 -r requirements.txt

# 业务代码
COPY app/ ./app/
COPY .env.example ./.env.example

# 启动后端（等 MySQL 就绪后）
EXPOSE 8000
CMD ["sh", "-c", "python -c 'import time; time.sleep(5)' && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000"]
