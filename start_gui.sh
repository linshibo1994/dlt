#!/bin/bash

# 大乐透预测系统GUI启动脚本
# DLT Prediction System GUI Launcher

echo "🎯 大乐透预测系统 - GUI启动器"
echo "=================================="

# 检查Python版本
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python版本: $python_version"

if [[ $(echo "$python_version < 3.8" | bc -l) -eq 1 ]]; then
    echo "❌ 需要Python 3.8或更高版本"
    exit 1
fi

# 检查依赖包
echo "🔍 检查依赖包..."
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "⚠️ 缺少Streamlit，正在安装..."
    pip3 install -r requirements_gui.txt
fi

# 检查核心文件
if [ ! -f "gui_app.py" ]; then
    echo "❌ 找不到gui_app.py文件"
    exit 1
fi

if [ ! -f "core_modules.py" ]; then
    echo "❌ 找不到core_modules.py文件"
    exit 1
fi

echo "✅ 环境检查通过"
echo ""

# 解析命令行参数
HOST="0.0.0.0"
PORT="8501"
LOCALHOST_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --localhost-only)
            LOCALHOST_ONLY=true
            HOST="localhost"
            shift
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: $0 [--host HOST] [--port PORT] [--localhost-only]"
            exit 1
            ;;
    esac
done

# 显示访问地址
if [ "$HOST" = "0.0.0.0" ]; then
    echo "🌐 GUI访问地址:"
    echo "   本地访问: http://localhost:$PORT"
    echo "   网络访问: http://192.168.110.2:$PORT"
    echo "   所有接口: http://0.0.0.0:$PORT"
else
    echo "🌐 GUI将在浏览器中打开: http://$HOST:$PORT"
fi

echo "💡 提示: 按 Ctrl+C 可以关闭GUI"
echo "=================================="

# 启动GUI
python3 -m streamlit run gui_app.py \
    --server.port $PORT \
    --server.address $HOST \
    --browser.gatherUsageStats false \
    --server.headless true
