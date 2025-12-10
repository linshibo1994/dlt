#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GUI启动器
GUI Launcher

提供便捷的GUI启动方式和环境检查
"""

import sys
import os
import subprocess
import platform
from pathlib import Path

def check_environment():
    """检查运行环境"""
    print("🔍 检查运行环境...")
    
    # 检查Python版本
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print(f"❌ Python版本过低: {python_version.major}.{python_version.minor}")
        print("   需要Python 3.8或更高版本")
        return False
    else:
        print(f"✅ Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 检查必要的包
    required_packages = [
        ('streamlit', 'streamlit'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('sklearn', 'scikit-learn')
    ]

    missing_packages = []
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
            print(f"✅ {package_name}: 已安装")
        except ImportError:
            print(f"❌ {package_name}: 未安装")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n⚠️ 缺少必要的包: {', '.join(missing_packages)}")
        print("请运行以下命令安装:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    # 检查项目文件
    required_files = [
        'gui_app.py',
        'predictor_modules.py',
        'analyzer_modules.py',
        'core_modules.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}: 存在")
        else:
            print(f"❌ {file_path}: 不存在")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️ 缺少必要的文件: {', '.join(missing_files)}")
        return False
    
    return True

def launch_gui():
    """启动GUI应用"""
    print("\n🚀 启动GUI应用...")
    
    try:
        # 设置环境变量
        env = os.environ.copy()
        env['PYTHONPATH'] = os.getcwd()
        
        # 构建启动命令
        cmd = [
            sys.executable, 
            '-m', 
            'streamlit', 
            'run', 
            'gui_app.py',
            '--server.port=8501',
            '--server.address=localhost',
            '--browser.gatherUsageStats=false'
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        print("=" * 60)
        print("🌐 GUI应用将在浏览器中打开")
        print("📍 访问地址: http://localhost:8501")
        print("⏹️ 按 Ctrl+C 停止应用")
        print("=" * 60)
        
        # 启动应用
        subprocess.run(cmd, env=env, cwd=os.getcwd())
        
    except KeyboardInterrupt:
        print("\n\n👋 GUI应用已停止")
    except Exception as e:
        print(f"\n❌ 启动GUI应用失败: {e}")
        print("\n🔧 故障排除建议:")
        print("1. 确保已安装streamlit: pip install streamlit")
        print("2. 检查端口8501是否被占用")
        print("3. 尝试手动运行: streamlit run gui_app.py")

def show_help():
    """显示帮助信息"""
    print("🎯 DLT大乐透预测系统 GUI启动器")
    print("=" * 60)
    print("用法:")
    print("  python gui_launcher.py [选项]")
    print("")
    print("选项:")
    print("  --check, -c    只检查环境，不启动GUI")
    print("  --help, -h     显示此帮助信息")
    print("  --version, -v  显示版本信息")
    print("")
    print("示例:")
    print("  python gui_launcher.py          # 检查环境并启动GUI")
    print("  python gui_launcher.py --check  # 只检查环境")
    print("")
    print("系统要求:")
    print("  - Python 3.8+")
    print("  - streamlit")
    print("  - pandas, numpy, scikit-learn")
    print("")
    print("功能特性:")
    print("  ✨ 多种预测算法 (传统算法 + 深度学习)")
    print("  📊 数据分析和可视化")
    print("  🗄️ 智能缓存系统")
    print("  ⚙️ 系统设置和优化")

def show_version():
    """显示版本信息"""
    print("🎯 DLT大乐透预测系统")
    print("版本: 2.0.0")
    print("作者: DLT Team")
    print("Python版本:", platform.python_version())
    print("操作系统:", platform.system(), platform.release())

def main():
    """主函数"""
    # 解析命令行参数
    args = sys.argv[1:]
    
    if '--help' in args or '-h' in args:
        show_help()
        return
    
    if '--version' in args or '-v' in args:
        show_version()
        return
    
    if '--check' in args or '-c' in args:
        print("🔍 环境检查模式")
        print("=" * 60)
        if check_environment():
            print("\n✅ 环境检查通过，可以启动GUI应用")
        else:
            print("\n❌ 环境检查失败，请修复问题后重试")
        return
    
    # 默认模式：检查环境并启动GUI
    print("🎯 DLT大乐透预测系统 GUI启动器")
    print("=" * 60)
    
    if not check_environment():
        print("\n❌ 环境检查失败，无法启动GUI应用")
        print("请修复上述问题后重试")
        return
    
    print("\n✅ 环境检查通过")
    
    # 询问是否启动
    try:
        response = input("\n是否启动GUI应用? (y/n): ").strip().lower()
        if response in ['y', 'yes', '是', '']:
            launch_gui()
        else:
            print("👋 已取消启动")
    except KeyboardInterrupt:
        print("\n👋 已取消启动")

if __name__ == "__main__":
    main()
