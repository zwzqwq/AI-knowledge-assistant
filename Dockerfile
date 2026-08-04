# 项目1 AI 知识库助手 —— FastAPI 服务镜像
FROM python:3.10-slim

# 减少日志缓冲，让 uvicorn 输出实时可见
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 先拷贝依赖清单并安装 —— 利用 Docker 缓存，改代码不用重装依赖
# 用清华镜像源加速国内 pip 下载（torch 体积大，不走默认源很慢）
COPY requirements.txt .
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 拷贝项目源码
COPY src ./src
COPY run_api.py .
COPY .env.example ./

# 暴露 FastAPI 端口
EXPOSE 8000

# 启动命令：常驻服务，监听 8000 端口
CMD ["uvicorn", "run_api:app", "--host", "0.0.0.0", "--port", "8000"]
