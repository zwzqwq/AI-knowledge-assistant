"""
入口脚本 —— 将 src/ui/app.py 注册为 Streamlit 入口

运行方式:
    streamlit run app.py
"""
import runpy
import sys
import os

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(__file__))

runpy.run_path("src/ui/app.py", run_name="__streamlit__")
