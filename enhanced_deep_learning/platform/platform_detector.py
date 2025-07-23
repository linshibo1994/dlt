#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
平台检测器模块
Platform Detector Module

提供操作系统检测、硬件检测、环境检测等功能。
"""

import os
import sys
import platform
import subprocess
import psutil
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import importlib.util

from core_modules import logger_manager
from ..utils.exceptions import DeepLearningException


class OSType(Enum):
    """操作系统类型枚举"""
    WINDOWS = "windows"
    LINUX = "linux"
    MACOS = "macos"
    FREEBSD = "freebsd"
    UNKNOWN = "unknown"


class Architecture(Enum):
    """系统架构枚举"""
    X86_64 = "x86_64"
    ARM64 = "arm64"
    X86 = "x86"
    ARM = "arm"
    UNKNOWN = "unknown"


@dataclass
class HardwareInfo:
    """硬件信息"""
    cpu_count: int
    cpu_freq: float
    memory_total: int
    memory_available: int
    disk_total: int
    disk_free: int
    gpu_count: int = 0
    gpu_info: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PythonInfo:
    """Python环境信息"""
    version: str
    implementation: str
    executable: str
    prefix: str
    packages: Dict[str, str] = field(default_factory=dict)


@dataclass
class PlatformInfo:
    """平台信息"""
    os_type: OSType
    os_name: str
    os_version: str
    architecture: Architecture
    hostname: str
    python_info: PythonInfo
    hardware_info: HardwareInfo
    environment_vars: Dict[str, str] = field(default_factory=dict)
    capabilities: Dict[str, bool] = field(default_factory=dict)


class PlatformDetector:
    """平台检测器"""
    
    def __init__(self):
        """初始化平台检测器"""
        self._platform_info = None
        self._detection_cache = {}
        
        logger_manager.info("平台检测器初始化完成")
    
    def detect_platform(self, force_refresh: bool = False) -> PlatformInfo:
        """
        检测平台信息
        
        Args:
            force_refresh: 是否强制刷新缓存
            
        Returns:
            平台信息
        """
        try:
            if self._platform_info is None or force_refresh:
                self._platform_info = self._perform_detection()
            
            return self._platform_info
            
        except Exception as e:
            logger_manager.error(f"平台检测失败: {e}")
            raise DeepLearningException(f"平台检测失败: {e}")
    
    def _perform_detection(self) -> PlatformInfo:
        """执行平台检测"""
        try:
            # 检测操作系统
            os_info = self._detect_os()
            
            # 检测架构
            arch = self._detect_architecture()
            
            # 检测Python环境
            python_info = self._detect_python_info()
            
            # 检测硬件信息
            hardware_info = self._detect_hardware_info()
            
            # 检测环境变量
            env_vars = self._detect_environment_vars()
            
            # 检测系统能力
            capabilities = self._detect_capabilities()
            
            platform_info = PlatformInfo(
                os_type=os_info[0],
                os_name=os_info[1],
                os_version=os_info[2],
                architecture=arch,
                hostname=platform.node(),
                python_info=python_info,
                hardware_info=hardware_info,
                environment_vars=env_vars,
                capabilities=capabilities
            )
            
            logger_manager.info(f"平台检测完成: {os_info[1]} {os_info[2]} ({arch.value})")
            return platform_info
            
        except Exception as e:
            logger_manager.error(f"执行平台检测失败: {e}")
            raise
    
    def _detect_os(self) -> Tuple[OSType, str, str]:
        """检测操作系统"""
        try:
            system = platform.system().lower()
            
            if system == "windows":
                os_type = OSType.WINDOWS
                os_name = "Windows"
                os_version = platform.release()
            elif system == "linux":
                os_type = OSType.LINUX
                os_name = "Linux"
                os_version = platform.release()
            elif system == "darwin":
                os_type = OSType.MACOS
                os_name = "macOS"
                os_version = platform.mac_ver()[0]
            elif system == "freebsd":
                os_type = OSType.FREEBSD
                os_name = "FreeBSD"
                os_version = platform.release()
            else:
                os_type = OSType.UNKNOWN
                os_name = system.capitalize()
                os_version = platform.release()
            
            return os_type, os_name, os_version
            
        except Exception as e:
            logger_manager.error(f"检测操作系统失败: {e}")
            return OSType.UNKNOWN, "Unknown", "Unknown"
    
    def _detect_architecture(self) -> Architecture:
        """检测系统架构"""
        try:
            machine = platform.machine().lower()
            
            if machine in ["x86_64", "amd64"]:
                return Architecture.X86_64
            elif machine in ["arm64", "aarch64"]:
                return Architecture.ARM64
            elif machine in ["i386", "i686", "x86"]:
                return Architecture.X86
            elif machine.startswith("arm"):
                return Architecture.ARM
            else:
                return Architecture.UNKNOWN
                
        except Exception as e:
            logger_manager.error(f"检测系统架构失败: {e}")
            return Architecture.UNKNOWN
    
    def _detect_python_info(self) -> PythonInfo:
        """检测Python环境信息"""
        try:
            # 检测已安装的包
            packages = {}
            try:
                import pkg_resources
                for dist in pkg_resources.working_set:
                    packages[dist.project_name] = dist.version
            except Exception:
                # 如果pkg_resources不可用，尝试其他方法
                try:
                    import importlib.metadata
                    for dist in importlib.metadata.distributions():
                        packages[dist.metadata['Name']] = dist.version
                except Exception:
                    logger_manager.warning("无法获取已安装包信息")
            
            python_info = PythonInfo(
                version=platform.python_version(),
                implementation=platform.python_implementation(),
                executable=sys.executable,
                prefix=sys.prefix,
                packages=packages
            )
            
            return python_info
            
        except Exception as e:
            logger_manager.error(f"检测Python环境信息失败: {e}")
            return PythonInfo(
                version="Unknown",
                implementation="Unknown",
                executable="Unknown",
                prefix="Unknown"
            )
    
    def _detect_hardware_info(self) -> HardwareInfo:
        """检测硬件信息"""
        try:
            # CPU信息
            cpu_count = psutil.cpu_count(logical=True)
            cpu_freq = psutil.cpu_freq().current if psutil.cpu_freq() else 0.0
            
            # 内存信息
            memory = psutil.virtual_memory()
            memory_total = memory.total
            memory_available = memory.available
            
            # 磁盘信息
            disk = psutil.disk_usage('/')
            disk_total = disk.total
            disk_free = disk.free
            
            # GPU信息
            gpu_count, gpu_info = self._detect_gpu_info()
            
            hardware_info = HardwareInfo(
                cpu_count=cpu_count,
                cpu_freq=cpu_freq,
                memory_total=memory_total,
                memory_available=memory_available,
                disk_total=disk_total,
                disk_free=disk_free,
                gpu_count=gpu_count,
                gpu_info=gpu_info
            )
            
            return hardware_info
            
        except Exception as e:
            logger_manager.error(f"检测硬件信息失败: {e}")
            return HardwareInfo(
                cpu_count=1,
                cpu_freq=0.0,
                memory_total=0,
                memory_available=0,
                disk_total=0,
                disk_free=0
            )
    
    def _detect_gpu_info(self) -> Tuple[int, List[Dict[str, Any]]]:
        """检测GPU信息"""
        try:
            gpu_info = []
            gpu_count = 0
            
            # 尝试使用nvidia-ml-py检测NVIDIA GPU
            try:
                import pynvml
                pynvml.nvmlInit()
                gpu_count = pynvml.nvmlDeviceGetCount()
                
                for i in range(gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    gpu_info.append({
                        'index': i,
                        'name': name,
                        'memory_total': memory_info.total,
                        'memory_free': memory_info.free,
                        'memory_used': memory_info.used,
                        'vendor': 'NVIDIA'
                    })
                
            except Exception:
                # 如果NVIDIA检测失败，尝试其他方法
                try:
                    # 尝试使用PyTorch检测
                    import torch
                    if torch.cuda.is_available():
                        gpu_count = torch.cuda.device_count()
                        for i in range(gpu_count):
                            props = torch.cuda.get_device_properties(i)
                            gpu_info.append({
                                'index': i,
                                'name': props.name,
                                'memory_total': props.total_memory,
                                'memory_free': props.total_memory,  # PyTorch不提供空闲内存
                                'memory_used': 0,
                                'vendor': 'NVIDIA'
                            })
                except Exception:
                    logger_manager.debug("无法检测GPU信息")
            
            return gpu_count, gpu_info
            
        except Exception as e:
            logger_manager.error(f"检测GPU信息失败: {e}")
            return 0, []
    
    def _detect_environment_vars(self) -> Dict[str, str]:
        """检测重要的环境变量"""
        try:
            important_vars = [
                'PATH', 'PYTHONPATH', 'HOME', 'USER', 'USERNAME',
                'CUDA_VISIBLE_DEVICES', 'CUDA_HOME', 'CUDNN_HOME',
                'LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH',
                'OMP_NUM_THREADS', 'MKL_NUM_THREADS'
            ]
            
            env_vars = {}
            for var in important_vars:
                value = os.environ.get(var)
                if value is not None:
                    env_vars[var] = value
            
            return env_vars
            
        except Exception as e:
            logger_manager.error(f"检测环境变量失败: {e}")
            return {}
    
    def _detect_capabilities(self) -> Dict[str, bool]:
        """检测系统能力"""
        try:
            capabilities = {}
            
            # 检测CUDA支持
            capabilities['cuda_available'] = self._check_cuda_availability()
            
            # 检测OpenMP支持
            capabilities['openmp_available'] = self._check_openmp_availability()
            
            # 检测MKL支持
            capabilities['mkl_available'] = self._check_mkl_availability()
            
            # 检测Docker支持
            capabilities['docker_available'] = self._check_docker_availability()
            
            # 检测网络连接
            capabilities['internet_available'] = self._check_internet_availability()
            
            # 检测权限
            capabilities['admin_privileges'] = self._check_admin_privileges()
            
            return capabilities
            
        except Exception as e:
            logger_manager.error(f"检测系统能力失败: {e}")
            return {}
    
    def _check_cuda_availability(self) -> bool:
        """检查CUDA可用性"""
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            try:
                # 尝试运行nvidia-smi
                result = subprocess.run(['nvidia-smi'], 
                                      capture_output=True, 
                                      timeout=5)
                return result.returncode == 0
            except Exception:
                return False
    
    def _check_openmp_availability(self) -> bool:
        """检查OpenMP可用性"""
        try:
            import os
            return 'OMP_NUM_THREADS' in os.environ or self._check_library_availability('libgomp')
        except Exception:
            return False
    
    def _check_mkl_availability(self) -> bool:
        """检查MKL可用性"""
        try:
            import numpy as np
            return 'mkl' in np.__config__.show().lower()
        except Exception:
            return False
    
    def _check_docker_availability(self) -> bool:
        """检查Docker可用性"""
        try:
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, 
                                  timeout=5)
            return result.returncode == 0
        except Exception:
            return False
    
    def _check_internet_availability(self) -> bool:
        """检查网络连接"""
        try:
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except Exception:
            return False
    
    def _check_admin_privileges(self) -> bool:
        """检查管理员权限"""
        try:
            if os.name == 'nt':  # Windows
                import ctypes
                return ctypes.windll.shell32.IsUserAnAdmin() != 0
            else:  # Unix-like
                return os.geteuid() == 0
        except Exception:
            return False
    
    def _check_library_availability(self, library_name: str) -> bool:
        """检查库可用性"""
        try:
            if os.name == 'nt':  # Windows
                # 在Windows上检查DLL
                import ctypes
                try:
                    ctypes.CDLL(library_name)
                    return True
                except OSError:
                    return False
            else:  # Unix-like
                # 在Unix系统上检查共享库
                import ctypes
                try:
                    ctypes.CDLL(library_name)
                    return True
                except OSError:
                    return False
        except Exception:
            return False
    
    def get_platform_summary(self) -> str:
        """获取平台摘要信息"""
        try:
            info = self.detect_platform()
            
            summary = f"""平台信息摘要:
操作系统: {info.os_name} {info.os_version}
架构: {info.architecture.value}
主机名: {info.hostname}
Python: {info.python_info.version} ({info.python_info.implementation})
CPU: {info.hardware_info.cpu_count} 核心 @ {info.hardware_info.cpu_freq:.1f} MHz
内存: {info.hardware_info.memory_total / (1024**3):.1f} GB (可用: {info.hardware_info.memory_available / (1024**3):.1f} GB)
磁盘: {info.hardware_info.disk_total / (1024**3):.1f} GB (空闲: {info.hardware_info.disk_free / (1024**3):.1f} GB)
GPU: {info.hardware_info.gpu_count} 个设备
CUDA: {'可用' if info.capabilities.get('cuda_available', False) else '不可用'}
Docker: {'可用' if info.capabilities.get('docker_available', False) else '不可用'}
网络: {'连接正常' if info.capabilities.get('internet_available', False) else '无网络连接'}"""
            
            return summary
            
        except Exception as e:
            logger_manager.error(f"获取平台摘要失败: {e}")
            return f"获取平台摘要失败: {e}"


# 全局平台检测器实例
platform_detector = PlatformDetector()


if __name__ == "__main__":
    # 测试平台检测器功能
    print("🔍 测试平台检测器功能...")
    
    try:
        detector = PlatformDetector()
        
        # 检测平台信息
        platform_info = detector.detect_platform()
        
        print("✅ 平台检测成功")
        print(f"操作系统: {platform_info.os_name} {platform_info.os_version}")
        print(f"架构: {platform_info.architecture.value}")
        print(f"Python: {platform_info.python_info.version}")
        print(f"CPU: {platform_info.hardware_info.cpu_count} 核心")
        print(f"内存: {platform_info.hardware_info.memory_total / (1024**3):.1f} GB")
        print(f"GPU: {platform_info.hardware_info.gpu_count} 个设备")
        
        # 显示能力检测结果
        print("\n系统能力:")
        for capability, available in platform_info.capabilities.items():
            status = "✅" if available else "❌"
            print(f"  {status} {capability}: {available}")
        
        # 获取平台摘要
        summary = detector.get_platform_summary()
        print(f"\n{summary}")
        
        print("\n✅ 平台检测器功能测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("平台检测器功能测试完成")
