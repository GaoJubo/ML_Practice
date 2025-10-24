#!/usr/bin/env python3
"""
机器学习算法展示平台后端启动脚本
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_python_version():
    """检查Python版本是否满足要求"""
    if sys.version_info < (3, 7):
        print("错误: 需要Python 3.7或更高版本")
        sys.exit(1)

def install_dependencies():
    """安装项目依赖"""
    backend_dir = Path(__file__).parent / "backend"
    requirements_file = backend_dir / "requirements.txt"
    
    if not requirements_file.exists():
        print(f"错误: 找不到requirements.txt文件: {requirements_file}")
        sys.exit(1)
    
    print("正在安装依赖...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("依赖安装完成")
    except subprocess.CalledProcessError:
        print("错误: 依赖安装失败")
        sys.exit(1)

def start_server(host="0.0.0.0", port=8000, reload=False):
    """启动FastAPI服务器"""
    backend_dir = Path(__file__).parent / "backend"
    app_module = "app.main:app"
    
    os.chdir(backend_dir)
    
    cmd = [
        sys.executable, "-m", "uvicorn",
        app_module,
        "--host", host,
        "--port", str(port)
    ]
    
    if reload:
        cmd.append("--reload")
    
    print(f"启动服务器: http://{host}:{port}")
    print("API文档: http://{}:{}/docs".format(host, port))
    
    try:
        subprocess.call(cmd)
    except KeyboardInterrupt:
        print("\n服务器已停止")

def main():
    parser = argparse.ArgumentParser(description="启动机器学习算法展示平台后端服务")
    parser.add_argument("--host", default="0.0.0.0", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    parser.add_argument("--reload", action="store_true", help="启用热重载")
    parser.add_argument("--install-deps", action="store_true", help="安装依赖")
    
    args = parser.parse_args()
    
    check_python_version()
    
    if args.install_deps:
        install_dependencies()
    
    start_server(args.host, args.port, args.reload)

if __name__ == "__main__":
    main()