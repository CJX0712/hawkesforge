# HawkesForge — 轻量镜像（纯 numpy，无编译）
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 依赖锁（numpy 预编译 wheel，零编译）
COPY requirements.lock.txt requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 源码
COPY . .

# 默认运行端到端演示（落盘 benchmark.json）
CMD ["python", "examples/run_demo.py"]
