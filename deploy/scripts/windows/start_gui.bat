@echo off
chcp 65001 >nul

REM 大乐透预测系统GUI启动脚本
REM DLT Prediction System GUI Launcher

echo 🎯 大乐透预测系统 - GUI启动器
echo ==================================

REM 检查Python版本
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 未找到Python，请先安装Python 3.8或更高版本
    pause
    exit /b 1
)

REM 检查依赖包
echo 🔍 检查依赖包...
python -c "import fastapi" >nul 2>&1
if errorlevel 1 (
    echo 缺少依赖，正在安装...
    pip install -r requirements.txt
)

REM 检查核心文件
if not exist "gui_app.py" (
    echo ❌ 找不到gui_app.py文件
    pause
    exit /b 1
)

if not exist "core_modules.py" (
    echo ❌ 找不到core_modules.py文件
    pause
    exit /b 1
)

echo ✅ 环境检查通过
echo.
echo 🌐 GUI将在浏览器中打开: http://localhost:8501
echo 💡 提示: 按 Ctrl+C 可以关闭GUI
echo ==================================

REM 启动GUI
python -m streamlit run gui_app.py --server.port 8501 --server.address localhost --browser.gatherUsageStats false --server.headless true

pause
