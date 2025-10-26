#!/usr/bin/env python3
"""
机器学习算法展示平台一键启动脚本
"""

import os
import sys
import subprocess
import time
import threading
import webbrowser
import signal
import argparse
from pathlib import Path

class PlatformLauncher:
    def __init__(self):
        self.backend_process = None
        self.frontend_process = None
        self.project_root = Path(__file__).parent
        
    def check_dependencies(self):
        """检查必要的文件是否存在"""
        required_files = [
            "start_backend.py",
            "start_frontend.py", 
            "backend/app/main.py",
            "index.html"
        ]
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                print(f"错误: 找不到必要文件: {full_path}")
                return False
        return True
    
    def start_backend(self, host="0.0.0.0", port=8000, reload=False):
        """启动后端服务"""
        print("正在启动后端服务...")
        
        backend_script = self.project_root / "start_backend.py"
        cmd = [
            sys.executable, str(backend_script),
            "--host", host,
            "--port", str(port)
        ]
        
        if reload:
            cmd.append("--reload")
        
        try:
            self.backend_process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # 等待后端启动
            time.sleep(2)
            
            if self.backend_process.poll() is None:
                print(f"✅ 后端服务启动成功: http://{host}:{port}")
                return True
            else:
                print("❌ 后端服务启动失败")
                return False
                
        except Exception as e:
            print(f"❌ 启动后端服务时出错: {e}")
            return False
    
    def start_frontend(self, port=8080, open_browser=True):
        """启动前端服务"""
        print("正在启动前端服务...")
        
        frontend_script = self.project_root / "start_frontend.py"
        cmd = [
            sys.executable, str(frontend_script),
            "--port", str(port)
        ]
        
        if open_browser:
            # 延迟打开浏览器
            def delayed_browser_open():
                time.sleep(3)
                webbrowser.open(f"http://localhost:{port}")
            
            browser_thread = threading.Thread(target=delayed_browser_open)
            browser_thread.daemon = True
            browser_thread.start()
        else:
            cmd.append("--no-browser")
        
        try:
            self.frontend_process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            time.sleep(1)
            
            if self.frontend_process.poll() is None:
                print(f"✅ 前端服务启动成功: http://localhost:{port}")
                return True
            else:
                print("❌ 前端服务启动失败")
                return False
                
        except Exception as e:
            print(f"❌ 启动前端服务时出错: {e}")
            return False
    
    def monitor_processes(self):
        """监控进程输出"""
        def monitor_backend():
            if self.backend_process:
                for line in iter(self.backend_process.stdout.readline, ''):
                    if line.strip():
                        print(f"[后端] {line.strip()}")
        
        def monitor_frontend():
            if self.frontend_process:
                for line in iter(self.frontend_process.stdout.readline, ''):
                    if line.strip():
                        print(f"[前端] {line.strip()}")
        
        # 启动监控线程
        backend_monitor = threading.Thread(target=monitor_backend)
        frontend_monitor = threading.Thread(target=monitor_frontend)
        
        backend_monitor.daemon = True
        frontend_monitor.daemon = True
        
        backend_monitor.start()
        frontend_monitor.start()
    
    def stop_services(self):
        """停止所有服务"""
        print("\n正在停止服务...")
        
        if self.backend_process:
            self.backend_process.terminate()
            self.backend_process.wait()
            print("✅ 后端服务已停止")
        
        if self.frontend_process:
            self.frontend_process.terminate()
            self.frontend_process.wait()
            print("✅ 前端服务已停止")
    
    def start_all(self, backend_host="0.0.0.0", backend_port=8000, 
                  frontend_port=8080, reload=False, open_browser=True, 
                  install_deps=False):
        """启动所有服务"""
        print("🚀 机器学习算法展示平台启动中...")
        print("=" * 50)
        
        # 检查依赖
        if not self.check_dependencies():
            return False
        
        # 安装依赖（如果需要）
        if install_deps:
            print("正在安装依赖...")
            backend_script = self.project_root / "start_backend.py"
            result = subprocess.run([
                sys.executable, str(backend_script), "--install-deps"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ 依赖安装失败: {result.stderr}")
                return False
            print("✅ 依赖安装完成")
        
        # 启动后端
        if not self.start_backend(backend_host, backend_port, reload):
            return False
        
        # 启动前端
        if not self.start_frontend(frontend_port, open_browser):
            self.stop_services()
            return False
        
        # 启动监控
        self.monitor_processes()
        
        print("=" * 50)
        print("🎉 平台启动完成！")
        print(f"📊 后端API: http://{backend_host}:{backend_port}")
        print(f"🌐 前端界面: http://localhost:{frontend_port}")
        print(f"📚 API文档: http://{backend_host}:{backend_port}/docs")
        print("\n按 Ctrl+C 停止服务")
        
        # 设置信号处理
        def signal_handler(sig, frame):
            self.stop_services()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            # 保持主线程运行
            while True:
                time.sleep(1)
                
                # 检查进程是否还在运行
                if self.backend_process and self.backend_process.poll() is not None:
                    print("❌ 后端服务意外停止")
                    break
                
                if self.frontend_process and self.frontend_process.poll() is not None:
                    print("❌ 前端服务意外停止")
                    break
                    
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_services()
        
        return True

def main():
    parser = argparse.ArgumentParser(description="机器学习算法展示平台一键启动")
    parser.add_argument("--backend-host", default="0.0.0.0", help="后端主机地址")
    parser.add_argument("--backend-port", type=int, default=8000, help="后端端口")
    parser.add_argument("--frontend-port", type=int, default=8080, help="前端端口")
    parser.add_argument("--reload", action="store_true", help="启用后端热重载")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    parser.add_argument("--install-deps", action="store_true", help="启动前安装依赖")
    
    args = parser.parse_args()
    
    launcher = PlatformLauncher()
    launcher.start_all(
        backend_host=args.backend_host,
        backend_port=args.backend_port,
        frontend_port=args.frontend_port,
        reload=args.reload,
        open_browser=not args.no_browser,
        install_deps=args.install_deps
    )

if __name__ == "__main__":
    main()
