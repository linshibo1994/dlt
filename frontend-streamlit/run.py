#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
大乐透预测系统GUI启动脚本
DLT Prediction System GUI Launcher

快速启动图形用户界面的便捷脚本。
"""

import os
import sys
import subprocess
import platform

def check_dependencies():
    """检查依赖包是否安装"""
    required_packages = [
        'streamlit',
        'plotly',
        'pandas',
        'numpy',
        'psutil'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies():
    """安装缺失的依赖包"""
    print("🔧 正在安装GUI依赖包...")
    
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements_gui.txt'
        ])
        print("✅ 依赖包安装完成！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 依赖包安装失败: {e}")
        return False

def launch_gui(host='0.0.0.0', port=8501):
    """启动GUI界面"""
    print("🚀 正在启动大乐透预测系统GUI...")

    # 设置Streamlit配置
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_SERVER_ENABLE_CORS'] = 'false'
    os.environ['STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION'] = 'false'

    try:
        # 启动Streamlit应用
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'gui_app.py',
            '--server.port', str(port),
            '--server.address', host,
            '--browser.gatherUsageStats', 'false'
        ])
    except KeyboardInterrupt:
        print("\n👋 GUI已关闭")
    except Exception as e:
        print(f"❌ GUI启动失败: {e}")

def main():
    """主函数"""
    print("=" * 60)
    print("🎯 大乐透预测系统 - GUI启动器")
    print("=" * 60)

    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='大乐透预测系统GUI启动器')
    parser.add_argument('--host', default='0.0.0.0', help='服务器地址 (默认: 0.0.0.0，允许网络访问)')
    parser.add_argument('--port', type=int, default=8501, help='服务器端口 (默认: 8501)')
    parser.add_argument('--localhost-only', action='store_true', help='仅允许本地访问')

    args = parser.parse_args()

    # 如果指定了仅本地访问，则使用localhost
    if args.localhost_only:
        args.host = 'localhost'

    # 检查Python版本
    if sys.version_info < (3, 8):
        print("❌ 需要Python 3.8或更高版本")
        sys.exit(1)

    # 检查依赖包
    missing = check_dependencies()

    if missing:
        print(f"⚠️ 缺少以下依赖包: {', '.join(missing)}")

        install_choice = input("是否自动安装依赖包? (y/n): ").lower().strip()

        if install_choice in ['y', 'yes', '是']:
            if not install_dependencies():
                print("❌ 依赖包安装失败，请手动安装")
                sys.exit(1)
        else:
            print("请手动安装依赖包: pip install -r requirements_gui.txt")
            sys.exit(1)

    # 检查核心文件
    if not os.path.exists('gui_app.py'):
        print("❌ 找不到gui_app.py文件")
        sys.exit(1)

    if not os.path.exists('core_modules.py'):
        print("❌ 找不到core_modules.py文件")
        sys.exit(1)

    print("✅ 环境检查通过")

    # 显示访问地址
    if args.host == '0.0.0.0':
        print(f"\n🌐 GUI访问地址:")
        print(f"   本地访问: http://localhost:{args.port}")
        print(f"   网络访问: http://192.168.110.2:{args.port}")
        print(f"   所有接口: http://0.0.0.0:{args.port}")
    else:
        print(f"\n🌐 GUI将在浏览器中打开: http://{args.host}:{args.port}")

    print("💡 提示: 按 Ctrl+C 可以关闭GUI")
    print("\n" + "=" * 60)

    # 启动GUI
    launch_gui(args.host, args.port)

if __name__ == "__main__":
    main()
