#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
跨平台安装器模块
Cross-Platform Installer Module

提供跨平台安装、依赖管理、环境配置等功能。
"""

import os
import sys
import subprocess
import shutil
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import tempfile
import urllib.request
import zipfile
import tarfile

from core_modules import logger_manager
from ..utils.exceptions import DeepLearningException
from .platform_detector import platform_detector, OSType
from .platform_adapter import platform_adapter


class InstallStatus(Enum):
    """安装状态枚举"""
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    PARTIAL = "partial"


class PackageManager(Enum):
    """包管理器枚举"""
    PIP = "pip"
    CONDA = "conda"
    APT = "apt"
    YUM = "yum"
    BREW = "brew"
    CHOCOLATEY = "chocolatey"
    WINGET = "winget"


@dataclass
class InstallConfig:
    """安装配置"""
    target_directory: str = ""
    create_virtual_env: bool = True
    virtual_env_name: str = "deep_learning_env"
    install_optional_deps: bool = True
    install_gpu_support: bool = True
    package_manager: Optional[PackageManager] = None
    custom_requirements: List[str] = field(default_factory=list)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    post_install_scripts: List[str] = field(default_factory=list)


@dataclass
class InstallResult:
    """安装结果"""
    status: InstallStatus
    message: str = ""
    installed_packages: List[str] = field(default_factory=list)
    failed_packages: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    install_log: List[str] = field(default_factory=list)
    duration: float = 0.0


class CrossPlatformInstaller:
    """跨平台安装器"""
    
    def __init__(self):
        """初始化跨平台安装器"""
        self.platform_info = platform_detector.detect_platform()
        self.adapter = platform_adapter
        self.supported_package_managers = self._detect_package_managers()
        
        logger_manager.info("跨平台安装器初始化完成")
    
    def _detect_package_managers(self) -> List[PackageManager]:
        """检测可用的包管理器"""
        try:
            managers = []
            
            # 检测pip
            if self.adapter.find_executable("pip") or self.adapter.find_executable("pip3"):
                managers.append(PackageManager.PIP)
            
            # 检测conda
            if self.adapter.find_executable("conda"):
                managers.append(PackageManager.CONDA)
            
            # 根据操作系统检测系统包管理器
            if self.platform_info.os_type == OSType.LINUX:
                if self.adapter.find_executable("apt"):
                    managers.append(PackageManager.APT)
                elif self.adapter.find_executable("yum"):
                    managers.append(PackageManager.YUM)
            
            elif self.platform_info.os_type == OSType.MACOS:
                if self.adapter.find_executable("brew"):
                    managers.append(PackageManager.BREW)
            
            elif self.platform_info.os_type == OSType.WINDOWS:
                if self.adapter.find_executable("choco"):
                    managers.append(PackageManager.CHOCOLATEY)
                if self.adapter.find_executable("winget"):
                    managers.append(PackageManager.WINGET)
            
            logger_manager.debug(f"检测到包管理器: {[m.value for m in managers]}")
            return managers
            
        except Exception as e:
            logger_manager.error(f"检测包管理器失败: {e}")
            return []
    
    def install_platform(self, config: InstallConfig = None) -> InstallResult:
        """
        安装深度学习平台
        
        Args:
            config: 安装配置
            
        Returns:
            安装结果
        """
        try:
            import time
            start_time = time.time()
            
            config = config or InstallConfig()
            result = InstallResult(status=InstallStatus.SUCCESS)
            
            logger_manager.info("开始安装深度学习平台")
            
            # 1. 检查系统要求
            if not self._check_system_requirements(result):
                result.status = InstallStatus.FAILED
                return result
            
            # 2. 创建虚拟环境
            if config.create_virtual_env:
                if not self._create_virtual_environment(config, result):
                    result.warnings.append("虚拟环境创建失败，使用系统环境")
            
            # 3. 安装核心依赖
            if not self._install_core_dependencies(config, result):
                result.status = InstallStatus.FAILED
                return result
            
            # 4. 安装可选依赖
            if config.install_optional_deps:
                self._install_optional_dependencies(config, result)
            
            # 5. 安装GPU支持
            if config.install_gpu_support:
                self._install_gpu_support(config, result)
            
            # 6. 安装自定义依赖
            if config.custom_requirements:
                self._install_custom_requirements(config, result)
            
            # 7. 配置环境变量
            if config.environment_variables:
                self._configure_environment_variables(config, result)
            
            # 8. 运行后安装脚本
            if config.post_install_scripts:
                self._run_post_install_scripts(config, result)
            
            # 9. 验证安装
            if not self._verify_installation(result):
                result.status = InstallStatus.PARTIAL
                result.warnings.append("安装验证部分失败")
            
            result.duration = time.time() - start_time
            result.message = f"安装完成，耗时 {result.duration:.1f} 秒"
            
            logger_manager.info(f"深度学习平台安装完成: {result.status.value}")
            return result
            
        except Exception as e:
            logger_manager.error(f"安装深度学习平台失败: {e}")
            return InstallResult(
                status=InstallStatus.FAILED,
                message=f"安装失败: {e}"
            )
    
    def _check_system_requirements(self, result: InstallResult) -> bool:
        """检查系统要求"""
        try:
            # 检查Python版本
            if sys.version_info < (3, 8):
                result.message = f"Python版本过低: {sys.version_info.major}.{sys.version_info.minor} < 3.8"
                return False
            
            # 检查磁盘空间
            free_space_gb = self.platform_info.hardware_info.disk_free / (1024**3)
            if free_space_gb < 5:
                result.warnings.append(f"磁盘空间不足: {free_space_gb:.1f} GB < 5 GB")
            
            # 检查内存
            available_memory_gb = self.platform_info.hardware_info.memory_available / (1024**3)
            if available_memory_gb < 2:
                result.warnings.append(f"可用内存不足: {available_memory_gb:.1f} GB < 2 GB")
            
            result.install_log.append("系统要求检查通过")
            return True
            
        except Exception as e:
            result.message = f"系统要求检查失败: {e}"
            return False
    
    def _create_virtual_environment(self, config: InstallConfig, result: InstallResult) -> bool:
        """创建虚拟环境"""
        try:
            # 选择虚拟环境工具
            if PackageManager.CONDA in self.supported_package_managers:
                return self._create_conda_env(config, result)
            else:
                return self._create_venv(config, result)
                
        except Exception as e:
            result.warnings.append(f"创建虚拟环境失败: {e}")
            return False
    
    def _create_conda_env(self, config: InstallConfig, result: InstallResult) -> bool:
        """创建Conda环境"""
        try:
            cmd = ["conda", "create", "-n", config.virtual_env_name, "python>=3.8", "-y"]
            
            process_result = self.adapter.execute_command(cmd, timeout=300)
            
            if process_result.returncode == 0:
                result.install_log.append(f"Conda环境创建成功: {config.virtual_env_name}")
                return True
            else:
                result.warnings.append(f"Conda环境创建失败: {process_result.stderr}")
                return False
                
        except Exception as e:
            result.warnings.append(f"创建Conda环境失败: {e}")
            return False
    
    def _create_venv(self, config: InstallConfig, result: InstallResult) -> bool:
        """创建Python虚拟环境"""
        try:
            venv_path = Path.home() / ".virtualenvs" / config.virtual_env_name
            
            cmd = [sys.executable, "-m", "venv", str(venv_path)]
            
            process_result = self.adapter.execute_command(cmd, timeout=120)
            
            if process_result.returncode == 0:
                result.install_log.append(f"Python虚拟环境创建成功: {venv_path}")
                return True
            else:
                result.warnings.append(f"Python虚拟环境创建失败: {process_result.stderr}")
                return False
                
        except Exception as e:
            result.warnings.append(f"创建Python虚拟环境失败: {e}")
            return False
    
    def _install_core_dependencies(self, config: InstallConfig, result: InstallResult) -> bool:
        """安装核心依赖"""
        try:
            core_packages = [
                "numpy>=1.21.0",
                "pandas>=1.3.0",
                "scikit-learn>=1.0.0",
                "torch>=1.12.0",
                "torchvision>=0.13.0",
                "matplotlib>=3.5.0",
                "seaborn>=0.11.0",
                "tqdm>=4.62.0",
                "psutil>=5.8.0"
            ]
            
            return self._install_packages(core_packages, config, result, "核心依赖")
            
        except Exception as e:
            result.message = f"安装核心依赖失败: {e}"
            return False
    
    def _install_optional_dependencies(self, config: InstallConfig, result: InstallResult):
        """安装可选依赖"""
        try:
            optional_packages = [
                "plotly>=5.0.0",
                "dash>=2.0.0",
                "jupyter>=1.0.0",
                "tensorboard>=2.8.0",
                "optuna>=2.10.0",
                "mlflow>=1.20.0"
            ]
            
            self._install_packages(optional_packages, config, result, "可选依赖", required=False)
            
        except Exception as e:
            result.warnings.append(f"安装可选依赖失败: {e}")
    
    def _install_gpu_support(self, config: InstallConfig, result: InstallResult):
        """安装GPU支持"""
        try:
            if not self.platform_info.capabilities.get('cuda_available', False):
                result.warnings.append("CUDA不可用，跳过GPU支持安装")
                return
            
            gpu_packages = []
            
            # 根据平台选择GPU包
            if self.platform_info.os_type == OSType.LINUX:
                gpu_packages = [
                    "torch[cuda]",
                    "torchvision[cuda]",
                    "torchaudio[cuda]"
                ]
            elif self.platform_info.os_type == OSType.WINDOWS:
                gpu_packages = [
                    "torch",
                    "torchvision", 
                    "torchaudio",
                    "--index-url https://download.pytorch.org/whl/cu118"
                ]
            
            if gpu_packages:
                self._install_packages(gpu_packages, config, result, "GPU支持", required=False)
            
        except Exception as e:
            result.warnings.append(f"安装GPU支持失败: {e}")
    
    def _install_custom_requirements(self, config: InstallConfig, result: InstallResult):
        """安装自定义依赖"""
        try:
            self._install_packages(config.custom_requirements, config, result, "自定义依赖", required=False)
            
        except Exception as e:
            result.warnings.append(f"安装自定义依赖失败: {e}")
    
    def _install_packages(self, packages: List[str], config: InstallConfig, 
                         result: InstallResult, category: str, required: bool = True) -> bool:
        """安装包列表"""
        try:
            if not packages:
                return True
            
            logger_manager.info(f"开始安装{category}: {len(packages)} 个包")
            
            # 选择包管理器
            package_manager = config.package_manager or self._get_preferred_package_manager()
            
            success_count = 0
            
            for package in packages:
                try:
                    if self._install_single_package(package, package_manager, result):
                        success_count += 1
                        result.installed_packages.append(package)
                    else:
                        result.failed_packages.append(package)
                        
                except Exception as e:
                    result.failed_packages.append(package)
                    result.warnings.append(f"安装 {package} 失败: {e}")
            
            result.install_log.append(f"{category}安装完成: {success_count}/{len(packages)}")
            
            if required and success_count == 0:
                result.message = f"关键{category}安装失败"
                return False
            
            return True
            
        except Exception as e:
            if required:
                result.message = f"安装{category}失败: {e}"
                return False
            else:
                result.warnings.append(f"安装{category}失败: {e}")
                return True
    
    def _install_single_package(self, package: str, package_manager: PackageManager, 
                               result: InstallResult) -> bool:
        """安装单个包"""
        try:
            if package_manager == PackageManager.PIP:
                cmd = ["pip", "install", package]
            elif package_manager == PackageManager.CONDA:
                cmd = ["conda", "install", package, "-y"]
            else:
                result.warnings.append(f"不支持的包管理器: {package_manager.value}")
                return False
            
            process_result = self.adapter.execute_command(cmd, timeout=300)
            
            if process_result.returncode == 0:
                logger_manager.debug(f"包安装成功: {package}")
                return True
            else:
                logger_manager.warning(f"包安装失败: {package}, 错误: {process_result.stderr}")
                return False
                
        except Exception as e:
            logger_manager.error(f"安装包失败: {package}, 错误: {e}")
            return False
    
    def _get_preferred_package_manager(self) -> PackageManager:
        """获取首选包管理器"""
        if PackageManager.CONDA in self.supported_package_managers:
            return PackageManager.CONDA
        elif PackageManager.PIP in self.supported_package_managers:
            return PackageManager.PIP
        else:
            return PackageManager.PIP  # 默认使用pip
    
    def _configure_environment_variables(self, config: InstallConfig, result: InstallResult):
        """配置环境变量"""
        try:
            for name, value in config.environment_variables.items():
                self.adapter.set_environment_variable(name, value)
                result.install_log.append(f"设置环境变量: {name}={value}")
                
        except Exception as e:
            result.warnings.append(f"配置环境变量失败: {e}")
    
    def _run_post_install_scripts(self, config: InstallConfig, result: InstallResult):
        """运行后安装脚本"""
        try:
            for script in config.post_install_scripts:
                try:
                    process_result = self.adapter.execute_command(script, shell=True, timeout=120)
                    
                    if process_result.returncode == 0:
                        result.install_log.append(f"后安装脚本执行成功: {script}")
                    else:
                        result.warnings.append(f"后安装脚本执行失败: {script}")
                        
                except Exception as e:
                    result.warnings.append(f"运行后安装脚本失败: {script}, 错误: {e}")
                    
        except Exception as e:
            result.warnings.append(f"运行后安装脚本失败: {e}")
    
    def _verify_installation(self, result: InstallResult) -> bool:
        """验证安装"""
        try:
            # 验证核心模块导入
            test_imports = [
                "numpy",
                "pandas", 
                "sklearn",
                "torch",
                "matplotlib"
            ]
            
            failed_imports = []
            
            for module in test_imports:
                try:
                    __import__(module)
                except ImportError:
                    failed_imports.append(module)
            
            if failed_imports:
                result.warnings.append(f"模块导入失败: {', '.join(failed_imports)}")
                return False
            
            result.install_log.append("安装验证通过")
            return True
            
        except Exception as e:
            result.warnings.append(f"安装验证失败: {e}")
            return False
    
    def uninstall_platform(self, config: InstallConfig = None) -> InstallResult:
        """
        卸载深度学习平台
        
        Args:
            config: 安装配置
            
        Returns:
            卸载结果
        """
        try:
            config = config or InstallConfig()
            result = InstallResult(status=InstallStatus.SUCCESS)
            
            logger_manager.info("开始卸载深度学习平台")
            
            # 删除虚拟环境
            if config.create_virtual_env:
                self._remove_virtual_environment(config, result)
            
            # 清理环境变量
            if config.environment_variables:
                self._cleanup_environment_variables(config, result)
            
            result.message = "卸载完成"
            logger_manager.info("深度学习平台卸载完成")
            
            return result
            
        except Exception as e:
            logger_manager.error(f"卸载深度学习平台失败: {e}")
            return InstallResult(
                status=InstallStatus.FAILED,
                message=f"卸载失败: {e}"
            )
    
    def _remove_virtual_environment(self, config: InstallConfig, result: InstallResult):
        """删除虚拟环境"""
        try:
            if PackageManager.CONDA in self.supported_package_managers:
                cmd = ["conda", "env", "remove", "-n", config.virtual_env_name, "-y"]
                process_result = self.adapter.execute_command(cmd, timeout=120)
                
                if process_result.returncode == 0:
                    result.install_log.append(f"Conda环境删除成功: {config.virtual_env_name}")
                else:
                    result.warnings.append(f"Conda环境删除失败: {process_result.stderr}")
            
            else:
                venv_path = Path.home() / ".virtualenvs" / config.virtual_env_name
                if venv_path.exists():
                    shutil.rmtree(venv_path)
                    result.install_log.append(f"Python虚拟环境删除成功: {venv_path}")
                    
        except Exception as e:
            result.warnings.append(f"删除虚拟环境失败: {e}")
    
    def _cleanup_environment_variables(self, config: InstallConfig, result: InstallResult):
        """清理环境变量"""
        try:
            for name in config.environment_variables.keys():
                if name in os.environ:
                    del os.environ[name]
                    result.install_log.append(f"清理环境变量: {name}")
                    
        except Exception as e:
            result.warnings.append(f"清理环境变量失败: {e}")
    
    def get_installation_status(self) -> Dict[str, Any]:
        """获取安装状态"""
        try:
            status = {
                "platform_info": {
                    "os": f"{self.platform_info.os_name} {self.platform_info.os_version}",
                    "architecture": self.platform_info.architecture.value,
                    "python_version": self.platform_info.python_info.version
                },
                "package_managers": [pm.value for pm in self.supported_package_managers],
                "installed_packages": {},
                "system_capabilities": self.platform_info.capabilities
            }
            
            # 检查已安装的包
            try:
                for package_name in ["numpy", "pandas", "torch", "sklearn", "matplotlib"]:
                    try:
                        module = __import__(package_name)
                        version = getattr(module, '__version__', 'unknown')
                        status["installed_packages"][package_name] = version
                    except ImportError:
                        status["installed_packages"][package_name] = "not_installed"
            except Exception:
                pass
            
            return status
            
        except Exception as e:
            logger_manager.error(f"获取安装状态失败: {e}")
            return {"error": f"获取安装状态失败: {e}"}


# 全局跨平台安装器实例
cross_platform_installer = CrossPlatformInstaller()


if __name__ == "__main__":
    # 测试跨平台安装器功能
    print("📦 测试跨平台安装器功能...")
    
    try:
        installer = CrossPlatformInstaller()
        
        # 获取安装状态
        status = installer.get_installation_status()
        print(f"✅ 安装状态获取成功")
        print(f"平台: {status['platform_info']['os']}")
        print(f"Python: {status['platform_info']['python_version']}")
        print(f"包管理器: {', '.join(status['package_managers'])}")
        
        # 检查已安装的包
        installed_count = sum(1 for v in status['installed_packages'].values() if v != 'not_installed')
        total_count = len(status['installed_packages'])
        print(f"已安装包: {installed_count}/{total_count}")
        
        # 测试系统要求检查
        result = InstallResult(status=InstallStatus.SUCCESS)
        if installer._check_system_requirements(result):
            print("✅ 系统要求检查通过")
        else:
            print(f"⚠️ 系统要求检查: {result.message}")
        
        print("✅ 跨平台安装器功能测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("跨平台安装器功能测试完成")
