#!/usr/bin/env python3
"""
机器学习算法展示平台前端启动脚本
"""

import os
import sys
import subprocess
import argparse
import webbrowser
from pathlib import Path
import threading
import time
import http.server
import socketserver

def start_simple_server(port=8080):
    """启动一个简单的HTTP服务器来提供前端文件"""
    os.chdir(Path(__file__).parent)
    
    class Handler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header('Access-Control-Allow-Origin', '*')
            super().end_headers()
    
    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"前端服务器启动: http://localhost:{port}")
        httpd.serve_forever()

def open_browser(port=8080, delay=1.5):
    """延迟打开浏览器"""
    time.sleep(delay)
    webbrowser.open(f"http://localhost:{port}")

def main():
    parser = argparse.ArgumentParser(description="启动机器学习算法展示平台前端")
    parser.add_argument("--port", type=int, default=8080, help="服务器端口")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    
    args = parser.parse_args()
    
    # 检查index.html是否存在
    index_file = Path(__file__).parent / "index.html"
    if not index_file.exists():
        print(f"错误: 找不到index.html文件: {index_file}")
        sys.exit(1)
    
    # 启动浏览器线程
    if not args.no_browser:
        browser_thread = threading.Thread(
            target=open_browser, 
            args=(args.port,)
        )
        browser_thread.daemon = True
        browser_thread.start()
    
    # 启动HTTP服务器
    start_simple_server(args.port)

if __name__ == "__main__":
    main()