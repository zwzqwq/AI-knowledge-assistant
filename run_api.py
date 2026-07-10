"""
FastAPI 服务启动入口

运行方式:
    python run_api.py
    或
    uvicorn run_api:app --reload --port 8000
"""
import sys
import os

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.api.server import app
