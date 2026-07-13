#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
大乐透智能预测系统 - 统一入口

支持两种使用方式：
1. 模式选择：python main.py --mode [cli|gui|api]
2. 直接命令：python main.py predict -m markov -p 800 -c 3

使用方法：
    python main.py --mode gui      # 图形界面模式
    python main.py --mode api      # API服务模式
    python main.py predict ...     # 直接执行预测命令
    python main.py evaluate ...    # 概率基线评估命令
    python main.py data status     # 直接执行数据命令
"""

import sys
import os

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 添加后端模块目录到路径，使其可以使用相对导入
BACKEND_APP_DIR = os.path.join(PROJECT_ROOT, 'backend', 'app')
if BACKEND_APP_DIR not in sys.path:
    sys.path.insert(0, BACKEND_APP_DIR)

# 设置环境变量
os.environ['DLT_PROJECT_ROOT'] = PROJECT_ROOT

# 设置 Matplotlib 缓存目录，避免不可写路径警告
if not os.environ.get('MPLCONFIGDIR'):
    mpl_cache_dir = os.path.join(PROJECT_ROOT, 'artifacts', 'mplconfig')
    try:
        os.makedirs(mpl_cache_dir, exist_ok=True)
        os.environ['MPLCONFIGDIR'] = mpl_cache_dir
    except Exception:
        # 兜底到系统临时目录
        os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

# 设置通用缓存目录，避免 Fontconfig 无可写缓存目录警告
if not os.environ.get('XDG_CACHE_HOME'):
    xdg_cache_dir = os.path.join(PROJECT_ROOT, 'artifacts', 'cache')
    try:
        os.makedirs(xdg_cache_dir, exist_ok=True)
        os.environ['XDG_CACHE_HOME'] = xdg_cache_dir
    except Exception:
        os.environ['XDG_CACHE_HOME'] = '/tmp'


def setup_paths():
    """设置路径配置"""
    try:
        import yaml
        paths_file = os.path.join(PROJECT_ROOT, 'config', 'paths.yaml')
        if os.path.exists(paths_file):
            with open(paths_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
    except ImportError:
        pass
    return None


def run_gui():
    """运行图形界面模式"""
    print("启动图形界面模式...")

    import subprocess

    # 尝试从新位置启动
    gui_paths = [
        os.path.join(PROJECT_ROOT, 'frontend', 'streamlit', 'app.py'),
        os.path.join(PROJECT_ROOT, 'gui_app.py'),
    ]

    gui_app = None
    for path in gui_paths:
        if os.path.exists(path):
            gui_app = path
            break

    if gui_app:
        cmd = [
            sys.executable, '-m', 'streamlit', 'run',
            gui_app,
            '--server.port', '8501',
            '--server.address', 'localhost'
        ]
        subprocess.run(cmd)
    else:
        print("未找到GUI应用文件")
        sys.exit(1)


def run_api():
    """运行API服务模式（预留）"""
    print("API服务模式尚未实现")
    print("计划使用FastAPI提供RESTful API接口")
    sys.exit(0)


def run_cli():
    """运行命令行模式 - 直接调用后端主程序"""
    # 从 backend/app/main.py 导入并运行
    try:
        # 由于已经将 backend/app 添加到 sys.path，可以直接导入
        from main import main as cli_main
        cli_main()
    except ImportError as e:
        print(f"CLI模块导入失败: {e}")
        print("尝试备用导入方式...")

        # 备用方式：直接执行 backend/app/main.py
        import subprocess
        main_script = os.path.join(BACKEND_APP_DIR, 'main.py')
        if os.path.exists(main_script):
            # 传递原始参数
            cmd = [sys.executable, main_script] + sys.argv[1:]
            result = subprocess.run(cmd, cwd=BACKEND_APP_DIR)
            sys.exit(result.returncode)
        else:
            print(f"找不到CLI主程序: {main_script}")
            sys.exit(1)


def main():
    """主入口函数"""
    # 检查是否是 --mode 模式
    if len(sys.argv) >= 2:
        first_arg = sys.argv[1]

        # 概率基线评估必须在加载旧 CLI 之前分流，避免非必要初始化输出
        if first_arg == 'evaluate':
            from backend.evaluation.cli import main as evaluation_main
            sys.exit(evaluation_main(sys.argv[2:]))

        # 模式选择：--mode 或 -m 作为第一个参数
        if first_arg in ['--mode', '-m'] and len(sys.argv) >= 3:
            mode = sys.argv[2]
            if mode == 'gui':
                run_gui()
                return
            elif mode == 'api':
                run_api()
                return
            elif mode == 'cli':
                # 移除 --mode cli 参数后传递给CLI
                sys.argv = [sys.argv[0]] + sys.argv[3:]
                run_cli()
                return
            else:
                print(f"未知模式: {mode}")
                print("可用模式: cli, gui, api")
                sys.exit(1)

        # 帮助信息
        elif first_arg in ['--help', '-h']:
            print_help()
            return

        # 版本信息
        elif first_arg in ['--version', '-v']:
            print("大乐透智能预测系统 v2.0.0")
            return

        # 直接命令模式：predict, data, analyze, learn, smart, optimize, backtest, system, compare, enhanced, version
        valid_commands = ['predict', 'data', 'analyze', 'learn', 'smart', 'optimize',
                         'backtest', 'system', 'compare', 'enhanced', 'version']
        if first_arg in valid_commands:
            run_cli()
            return

    # 无参数时显示帮助
    print_help()


def print_help():
    """打印帮助信息"""
    help_text = """
大乐透智能预测系统 v2.0.0

使用方法：
  python main.py [命令] [选项]
  python main.py --mode [cli|gui|api]

运行模式：
  --mode cli      命令行模式（默认）
  --mode gui      图形界面模式（Streamlit）
  --mode api      API服务模式（预留）

可用命令：
  predict         号码预测
  data            数据管理
  analyze         数据分析
  learn           自适应学习
  smart           智能预测
  optimize        参数优化
  backtest        历史回测
  system          系统管理
  compare         批量预测对比
  enhanced        增强功能
  evaluate        概率基线预测与无泄漏滚动评估
  version         显示版本信息

预测命令示例：
  python main.py predict -m markov -p 800 -c 3
  python main.py predict -m frequency -p 500 -c 5
  python main.py predict -m lstm -p 1000 -c 1

评估命令示例：
  python main.py evaluate predict --method dirichlet --periods 500 --count 5
  python main.py evaluate walk-forward --methods uniform,dirichlet --draws 30

数据命令示例：
  python main.py data status
  python main.py data update --incremental
  python main.py data latest

更多帮助：
  python main.py [命令] --help    查看命令详细帮助
  查看 README.md 或 docs/QUICK_START.md 获取完整文档
"""
    print(help_text)


if __name__ == '__main__':
    main()
