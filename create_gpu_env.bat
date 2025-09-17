@echo off
chcp 65001 >nul
echo ===== DLT项目GPU环境设置 =====

REM 检查conda
conda --version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Conda不可用，将使用pip安装
    goto :pip_install
)

echo [OK] 检测到Conda，创建专用GPU环境...

REM 创建conda环境
echo 创建conda环境: dlt_gpu
conda create -n dlt_gpu python=3.11 -y
if errorlevel 1 (
    echo [ERROR] Conda环境创建失败
    goto :pip_install
)

REM 激活环境并安装包
echo 安装CUDA工具包...
conda install -n dlt_gpu -c conda-forge cudatoolkit=11.7 cudnn -y

echo 安装基础科学计算包...
conda install -n dlt_gpu numpy pandas matplotlib seaborn plotly scipy scikit-learn -y

echo 安装深度学习框架...
conda run -n dlt_gpu pip install tensorflow==2.12.0
conda run -n dlt_gpu pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

echo 安装其他依赖...
conda run -n dlt_gpu pip install streamlit psutil tqdm colorama joblib pyyaml

echo.
echo [OK] Conda GPU环境创建完成！
echo.
echo 使用方法:
echo   conda activate dlt_gpu
echo   python dlt_main.py
echo.
goto :end

:pip_install
echo.
echo [INFO] 使用当前Python环境安装GPU支持包...
echo.

REM 升级pip
python -m pip install --upgrade pip

REM 安装TensorFlow（尝试最新版本）
echo 安装TensorFlow...
python -m pip install tensorflow

REM 安装PyTorch CPU版本作为备选
echo 安装PyTorch...
python -m pip install torch torchvision torchaudio

REM 安装其他必需包
echo 安装其他依赖包...
python -m pip install streamlit psutil>=5.8.0 tqdm>=4.62.0 colorama>=0.4.4 joblib>=1.1.0 pyyaml>=6.0

echo.
echo [OK] GPU支持包安装完成！
echo.
echo 注意: 当前Python版本可能不完全支持GPU加速
echo 建议使用conda环境获得最佳GPU支持
echo.

:end
echo.
echo ===== 环境设置完成 =====
echo.
echo 测试GPU支持:
echo   python gpu_accelerated_predictor.py
echo.
echo 运行项目:
echo   python dlt_main.py          (命令行版本)
echo   python gui_app.py           (GUI版本)
echo   streamlit run gui_app.py    (Web界面)
echo.
pause