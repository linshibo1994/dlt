#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPU Environment Setup Script for DLT Project
为DLT项目设置GPU环境的脚本

This script creates a proper Python environment with GPU support
这个脚本创建一个支持GPU的Python环境
"""

import os
import sys
import subprocess
import platform
import json
from pathlib import Path


def check_system():
    """检查系统要求"""
    print("=== 系统要求检查 ===")

    # 检查操作系统
    os_type = platform.system()
    print(f"操作系统: {os_type} {platform.release()}")

    # 检查Python版本
    python_version = sys.version_info
    print(f"当前Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")

    # 检查NVIDIA驱动
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("[OK] NVIDIA驱动检测成功")
            # 提取GPU信息
            lines = result.stdout.split('\n')
            for line in lines:
                if 'NVIDIA GeForce' in line or 'NVIDIA RTX' in line:
                    gpu_name = line.split('|')[1].strip() if '|' in line else line.strip()
                    print(f"GPU设备: {gpu_name}")
                    break
            return True
        else:
            print("[ERROR] NVIDIA驱动检测失败")
            return False
    except Exception as e:
        print(f"[ERROR] 无法检测NVIDIA驱动: {e}")
        return False


def check_conda():
    """检查conda是否可用"""
    try:
        result = subprocess.run(['conda', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[OK] Conda可用: {result.stdout.strip()}")
            return True
        else:
            print("[WARNING] Conda不可用")
            return False
    except Exception:
        print("[WARNING] Conda不可用")
        return False


def create_conda_environment():
    """创建conda环境"""
    print("\n=== 创建Conda环境 ===")

    env_name = "dlt_gpu"
    python_version = "3.11"

    commands = [
        # 创建环境
        f"conda create -n {env_name} python={python_version} -y",

        # 激活环境并安装CUDA工具包
        f"conda install -n {env_name} -c conda-forge cudatoolkit=11.7 cudnn -y",

        # 安装基础包
        f"conda install -n {env_name} numpy pandas matplotlib seaborn plotly scipy scikit-learn -y",
    ]

    for cmd in commands:
        print(f"执行: {cmd}")
        try:
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            print("[OK] 命令执行成功")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] 命令执行失败: {e}")
            print(f"错误输出: {e.stderr}")
            return False

    # 安装GPU版本的深度学习框架
    gpu_commands = [
        # TensorFlow GPU
        f"conda run -n {env_name} pip install tensorflow[and-cuda]==2.12.0",

        # PyTorch GPU
        f"conda run -n {env_name} pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117",

        # 其他依赖
        f"conda run -n {env_name} pip install streamlit psutil tqdm colorama joblib pyyaml",
    ]

    for cmd in gpu_commands:
        print(f"执行: {cmd}")
        try:
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            print("[OK] GPU包安装成功")
        except subprocess.CalledProcessError as e:
            print(f"[WARNING] GPU包安装失败: {e}")
            # 继续执行，不中断

    return True


def create_venv_environment():
    """创建虚拟环境（当conda不可用时）"""
    print("\n=== 创建虚拟环境 ===")

    env_dir = Path("venv_gpu")

    try:
        # 创建虚拟环境
        subprocess.run([sys.executable, '-m', 'venv', str(env_dir)], check=True)
        print(f"[OK] 虚拟环境创建成功: {env_dir}")

        # 确定激活脚本路径
        if platform.system() == "Windows":
            activate_script = env_dir / "Scripts" / "activate.bat"
            pip_executable = env_dir / "Scripts" / "pip.exe"
        else:
            activate_script = env_dir / "bin" / "activate"
            pip_executable = env_dir / "bin" / "pip"

        # 升级pip
        subprocess.run([str(pip_executable), 'install', '--upgrade', 'pip'], check=True)

        # 安装基础包
        base_packages = [
            'numpy>=1.21.0',
            'pandas>=1.3.0',
            'matplotlib>=3.5.0',
            'seaborn>=0.11.0',
            'plotly>=5.0.0',
            'scipy>=1.7.0',
            'scikit-learn>=1.0.0',
            'streamlit',
            'psutil>=5.8.0',
            'tqdm>=4.62.0',
            'colorama>=0.4.4',
            'joblib>=1.1.0',
            'pyyaml>=6.0'
        ]

        for package in base_packages:
            print(f"安装: {package}")
            subprocess.run([str(pip_executable), 'install', package], check=True)

        # 尝试安装TensorFlow（最新版本）
        try:
            print("安装TensorFlow...")
            subprocess.run([str(pip_executable), 'install', 'tensorflow>=2.12.0'], check=True)
            print("[OK] TensorFlow安装成功")
        except Exception as e:
            print(f"[WARNING] TensorFlow安装失败: {e}")

        # 尝试安装PyTorch CPU版本（作为备选）
        try:
            print("安装PyTorch...")
            subprocess.run([str(pip_executable), 'install', 'torch', 'torchvision', 'torchaudio'], check=True)
            print("[OK] PyTorch安装成功")
        except Exception as e:
            print(f"[WARNING] PyTorch安装失败: {e}")

        return env_dir, activate_script

    except Exception as e:
        print(f"[ERROR] 虚拟环境创建失败: {e}")
        return None, None


def create_activation_script(env_path=None, is_conda=False, env_name="dlt_gpu"):
    """创建环境激活脚本"""
    print("\n=== 创建激活脚本 ===")

    if platform.system() == "Windows":
        script_name = "activate_gpu_env.bat"
        if is_conda:
            script_content = f'''@echo off
echo 激活GPU环境...
call conda activate {env_name}
if errorlevel 1 (
    echo [ERROR] 环境激活失败
    pause
    exit /b 1
)
echo [OK] GPU环境已激活: {env_name}
echo.
echo 使用方法:
echo   python dlt_main.py                    # 运行命令行版本
echo   python gui_app.py                     # 运行GUI版本
echo   python -m streamlit run gui_app.py    # 运行Streamlit GUI
echo.
cmd /k
'''
        else:
            script_content = f'''@echo off
echo 激活GPU环境...
call "{env_path}\\Scripts\\activate.bat"
if errorlevel 1 (
    echo [ERROR] 环境激活失败
    pause
    exit /b 1
)
echo [OK] GPU环境已激活: {env_path}
echo.
echo 使用方法:
echo   python dlt_main.py                    # 运行命令行版本
echo   python gui_app.py                     # 运行GUI版本
echo   python -m streamlit run gui_app.py    # 运行Streamlit GUI
echo.
cmd /k
'''
    else:
        script_name = "activate_gpu_env.sh"
        if is_conda:
            script_content = f'''#!/bin/bash
echo "激活GPU环境..."
eval "$(conda shell.bash hook)"
conda activate {env_name}
if [ $? -ne 0 ]; then
    echo "[ERROR] 环境激活失败"
    exit 1
fi
echo "[OK] GPU环境已激活: {env_name}"
echo ""
echo "使用方法:"
echo "  python dlt_main.py                    # 运行命令行版本"
echo "  python gui_app.py                     # 运行GUI版本"
echo "  python -m streamlit run gui_app.py    # 运行Streamlit GUI"
echo ""
bash
'''
        else:
            script_content = f'''#!/bin/bash
echo "激活GPU环境..."
source "{env_path}/bin/activate"
if [ $? -ne 0 ]; then
    echo "[ERROR] 环境激活失败"
    exit 1
fi
echo "[OK] GPU环境已激活: {env_path}"
echo ""
echo "使用方法:"
echo "  python dlt_main.py                    # 运行命令行版本"
echo "  python gui_app.py                     # 运行GUI版本"
echo "  python -m streamlit run gui_app.py    # 运行Streamlit GUI"
echo ""
bash
'''

    with open(script_name, 'w', encoding='utf-8') as f:
        f.write(script_content)

    # 在Unix系统上设置执行权限
    if platform.system() != "Windows":
        os.chmod(script_name, 0o755)

    print(f"[OK] 激活脚本已创建: {script_name}")
    return script_name


def verify_gpu_support(env_name=None, env_path=None):
    """验证GPU支持"""
    print("\n=== 验证GPU支持 ===")

    test_script = '''
import sys
print(f"Python版本: {sys.version}")

# 测试nvidia-smi GPU检测
try:
    import subprocess
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        lines = result.stdout.strip().split("\\n")
        for i, line in enumerate(lines):
            if line.strip():
                parts = line.split(",")
                if len(parts) >= 2:
                    print(f"[OK] GPU {i}: {parts[0].strip()}, {parts[1].strip()}MB")
    else:
        print("[ERROR] nvidia-smi检测失败")
except Exception as e:
    print(f"[ERROR] GPU检测异常: {e}")

# 测试TensorFlow
try:
    import tensorflow as tf
    print(f"TensorFlow版本: {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow检测到 {len(gpus)} 个GPU设备")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.name}")
    if len(gpus) > 0:
        print("[OK] TensorFlow GPU支持正常")
    else:
        print("[WARNING] TensorFlow未检测到GPU")
except ImportError:
    print("[WARNING] TensorFlow未安装")
except Exception as e:
    print(f"[ERROR] TensorFlow测试失败: {e}")

# 测试PyTorch
try:
    import torch
    print(f"PyTorch版本: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"PyTorch CUDA可用: {cuda_available}")
    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"PyTorch检测到 {device_count} 个CUDA设备")
        for i in range(device_count):
            print(f"  设备 {i}: {torch.cuda.get_device_name(i)}")
        print("[OK] PyTorch CUDA支持正常")
    else:
        print("[WARNING] PyTorch CUDA不可用")
except ImportError:
    print("[WARNING] PyTorch未安装")
except Exception as e:
    print(f"[ERROR] PyTorch测试失败: {e}")

print("\\n=== GPU支持验证完成 ===")
'''

    with open('test_gpu_support.py', 'w', encoding='utf-8') as f:
        f.write(test_script)

    try:
        if env_name:  # conda环境
            cmd = f"conda run -n {env_name} python test_gpu_support.py"
        elif env_path:  # venv环境
            if platform.system() == "Windows":
                python_exec = f"{env_path}/Scripts/python.exe"
            else:
                python_exec = f"{env_path}/bin/python"
            cmd = f"{python_exec} test_gpu_support.py"
        else:  # 当前环境
            cmd = "python test_gpu_support.py"

        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("警告信息:")
            print(result.stderr)

        return result.returncode == 0

    except Exception as e:
        print(f"[ERROR] GPU支持验证失败: {e}")
        return False
    finally:
        # 清理测试文件
        try:
            os.remove('test_gpu_support.py')
        except:
            pass


def main():
    """主函数"""
    print("DLT项目GPU环境设置脚本")
    print("=" * 50)

    # 检查系统
    if not check_system():
        print("\n[ERROR] 系统检查失败，无法继续")
        return False

    # 检查conda
    has_conda = check_conda()

    if has_conda:
        print("\n使用Conda创建GPU环境...")
        if create_conda_environment():
            script_name = create_activation_script(is_conda=True, env_name="dlt_gpu")
            print(f"\n[OK] Conda环境创建成功")
            print(f"激活环境: {script_name}")

            # 验证GPU支持
            verify_gpu_support(env_name="dlt_gpu")

        else:
            print("\n[ERROR] Conda环境创建失败")
            return False
    else:
        print("\n使用虚拟环境创建GPU环境...")
        env_path, activate_script = create_venv_environment()
        if env_path:
            script_name = create_activation_script(env_path=env_path, is_conda=False)
            print(f"\n[OK] 虚拟环境创建成功")
            print(f"环境路径: {env_path}")
            print(f"激活脚本: {script_name}")

            # 验证GPU支持
            verify_gpu_support(env_path=env_path)
        else:
            print("\n[ERROR] 虚拟环境创建失败")
            return False

    print("\n" + "=" * 50)
    print("GPU环境设置完成!")
    print("\n下一步操作:")
    print(f"1. 运行激活脚本: {script_name}")
    print("2. 在激活的环境中运行项目:")
    print("   - 命令行版本: python dlt_main.py")
    print("   - GUI版本: python gui_app.py")
    print("   - Streamlit GUI: python -m streamlit run gui_app.py")
    print("\n注意: 所有后续的项目运行都应该在GPU环境中进行")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)