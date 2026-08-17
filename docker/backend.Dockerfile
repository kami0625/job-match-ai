# ==================== 后端 FastAPI 镜像 ====================
FROM python:3.10-slim

WORKDIR /app

# 依赖层缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 业务代码
COPY app/ ./app/
COPY .env.example ./.env.example

# 启动后端（等 MySQL 就绪后）
EXPOSE 8000
CMD ["sh", "-c", "python -c 'import time; time.sleep(5)' && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000"]
