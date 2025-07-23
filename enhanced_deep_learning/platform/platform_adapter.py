#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
平台适配器模块
Platform Adapter Module

提供跨平台适配、路径处理、命令执行等功能。
"""

import os
import sys
import subprocess
import shutil
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
import tempfile

from core_modules import logger_manager
from ..utils.exceptions import DeepLearningException
from .platform_detector import PlatformDetector, OSType, platform_detector


@dataclass
class AdapterConfig:
    """适配器配置"""
    platform_specific_paths: Dict[str, str] = field(default_factory=dict)
    platform_specific_commands: Dict[str, str] = field(default_factory=dict)
    environment_overrides: Dict[str, str] = field(default_factory=dict)
    feature_flags: Dict[str, bool] = field(default_factory=dict)


class AdapterRegistry:
    """适配器注册表"""
    
    def __init__(self):
        """初始化适配器注册表"""
        self.adapters = {}
        self.default_adapters = {}
        
        logger_manager.debug("适配器注册表初始化完成")
    
    def register_adapter(self, os_type: OSType, adapter: 'PlatformAdapter'):
        """
        注册平台适配器
        
        Args:
            os_type: 操作系统类型
            adapter: 平台适配器
        """
        try:
            self.adapters[os_type] = adapter
            logger_manager.debug(f"平台适配器注册成功: {os_type.value}")
            
        except Exception as e:
            logger_manager.error(f"注册平台适配器失败: {e}")
    
    def get_adapter(self, os_type: OSType) -> Optional['PlatformAdapter']:
        """
        获取平台适配器
        
        Args:
            os_type: 操作系统类型
            
        Returns:
            平台适配器
        """
        return self.adapters.get(os_type)
    
    def get_current_adapter(self) -> Optional['PlatformAdapter']:
        """获取当前平台的适配器"""
        try:
            platform_info = platform_detector.detect_platform()
            return self.get_adapter(platform_info.os_type)
            
        except Exception as e:
            logger_manager.error(f"获取当前平台适配器失败: {e}")
            return None


class PlatformAdapter:
    """平台适配器"""
    
    def __init__(self, os_type: OSType, config: AdapterConfig = None):
        """
        初始化平台适配器
        
        Args:
            os_type: 操作系统类型
            config: 适配器配置
        """
        self.os_type = os_type
        self.config = config or AdapterConfig()
        self.platform_info = platform_detector.detect_platform()
        
        logger_manager.info(f"平台适配器初始化完成: {os_type.value}")
    
    def normalize_path(self, path: str) -> str:
        """
        标准化路径
        
        Args:
            path: 原始路径
            
        Returns:
            标准化后的路径
        """
        try:
            # 使用pathlib进行跨平台路径处理
            normalized = Path(path).resolve()
            return str(normalized)
            
        except Exception as e:
            logger_manager.error(f"路径标准化失败: {e}")
            return path
    
    def get_executable_extension(self) -> str:
        """获取可执行文件扩展名"""
        if self.os_type == OSType.WINDOWS:
            return ".exe"
        else:
            return ""
    
    def get_library_extension(self) -> str:
        """获取动态库扩展名"""
        if self.os_type == OSType.WINDOWS:
            return ".dll"
        elif self.os_type == OSType.MACOS:
            return ".dylib"
        else:
            return ".so"
    
    def get_path_separator(self) -> str:
        """获取路径分隔符"""
        if self.os_type == OSType.WINDOWS:
            return ";"
        else:
            return ":"
    
    def execute_command(self, command: Union[str, List[str]], 
                       shell: bool = None, timeout: int = None,
                       capture_output: bool = True) -> subprocess.CompletedProcess:
        """
        执行命令
        
        Args:
            command: 命令字符串或命令列表
            shell: 是否使用shell
            timeout: 超时时间
            capture_output: 是否捕获输出
            
        Returns:
            命令执行结果
        """
        try:
            # 平台特定的shell设置
            if shell is None:
                shell = self.os_type == OSType.WINDOWS
            
            # 适配命令
            adapted_command = self._adapt_command(command)
            
            # 执行命令
            result = subprocess.run(
                adapted_command,
                shell=shell,
                timeout=timeout,
                capture_output=capture_output,
                text=True,
                check=False
            )
            
            logger_manager.debug(f"命令执行完成: {adapted_command}")
            return result
            
        except Exception as e:
            logger_manager.error(f"命令执行失败: {e}")
            raise DeepLearningException(f"命令执行失败: {e}")
    
    def _adapt_command(self, command: Union[str, List[str]]) -> Union[str, List[str]]:
        """适配命令"""
        try:
            if isinstance(command, str):
                # 检查是否有平台特定的命令替换
                for pattern, replacement in self.config.platform_specific_commands.items():
                    if pattern in command:
                        command = command.replace(pattern, replacement)
                return command
            
            elif isinstance(command, list):
                # 适配命令列表
                adapted_command = []
                for part in command:
                    # 检查是否有平台特定的命令替换
                    for pattern, replacement in self.config.platform_specific_commands.items():
                        if pattern in part:
                            part = part.replace(pattern, replacement)
                    adapted_command.append(part)
                return adapted_command
            
            return command
            
        except Exception as e:
            logger_manager.error(f"命令适配失败: {e}")
            return command
    
    def get_temp_directory(self) -> str:
        """获取临时目录"""
        try:
            return tempfile.gettempdir()
            
        except Exception as e:
            logger_manager.error(f"获取临时目录失败: {e}")
            return "/tmp" if self.os_type != OSType.WINDOWS else "C:\\temp"
    
    def get_home_directory(self) -> str:
        """获取用户主目录"""
        try:
            return str(Path.home())
            
        except Exception as e:
            logger_manager.error(f"获取用户主目录失败: {e}")
            return os.path.expanduser("~")
    
    def get_config_directory(self, app_name: str) -> str:
        """
        获取应用配置目录
        
        Args:
            app_name: 应用名称
            
        Returns:
            配置目录路径
        """
        try:
            if self.os_type == OSType.WINDOWS:
                # Windows: %APPDATA%\AppName
                appdata = os.environ.get('APPDATA', '')
                if appdata:
                    return os.path.join(appdata, app_name)
                else:
                    return os.path.join(self.get_home_directory(), 'AppData', 'Roaming', app_name)
            
            elif self.os_type == OSType.MACOS:
                # macOS: ~/Library/Application Support/AppName
                return os.path.join(self.get_home_directory(), 'Library', 'Application Support', app_name)
            
            else:
                # Linux/Unix: ~/.config/AppName
                xdg_config = os.environ.get('XDG_CONFIG_HOME', '')
                if xdg_config:
                    return os.path.join(xdg_config, app_name)
                else:
                    return os.path.join(self.get_home_directory(), '.config', app_name)
                    
        except Exception as e:
            logger_manager.error(f"获取配置目录失败: {e}")
            return os.path.join(self.get_home_directory(), f'.{app_name}')
    
    def get_data_directory(self, app_name: str) -> str:
        """
        获取应用数据目录
        
        Args:
            app_name: 应用名称
            
        Returns:
            数据目录路径
        """
        try:
            if self.os_type == OSType.WINDOWS:
                # Windows: %LOCALAPPDATA%\AppName
                localappdata = os.environ.get('LOCALAPPDATA', '')
                if localappdata:
                    return os.path.join(localappdata, app_name)
                else:
                    return os.path.join(self.get_home_directory(), 'AppData', 'Local', app_name)
            
            elif self.os_type == OSType.MACOS:
                # macOS: ~/Library/Application Support/AppName
                return os.path.join(self.get_home_directory(), 'Library', 'Application Support', app_name)
            
            else:
                # Linux/Unix: ~/.local/share/AppName
                xdg_data = os.environ.get('XDG_DATA_HOME', '')
                if xdg_data:
                    return os.path.join(xdg_data, app_name)
                else:
                    return os.path.join(self.get_home_directory(), '.local', 'share', app_name)
                    
        except Exception as e:
            logger_manager.error(f"获取数据目录失败: {e}")
            return os.path.join(self.get_home_directory(), f'.{app_name}')
    
    def get_cache_directory(self, app_name: str) -> str:
        """
        获取应用缓存目录
        
        Args:
            app_name: 应用名称
            
        Returns:
            缓存目录路径
        """
        try:
            if self.os_type == OSType.WINDOWS:
                # Windows: %LOCALAPPDATA%\AppName\Cache
                return os.path.join(self.get_data_directory(app_name), 'Cache')
            
            elif self.os_type == OSType.MACOS:
                # macOS: ~/Library/Caches/AppName
                return os.path.join(self.get_home_directory(), 'Library', 'Caches', app_name)
            
            else:
                # Linux/Unix: ~/.cache/AppName
                xdg_cache = os.environ.get('XDG_CACHE_HOME', '')
                if xdg_cache:
                    return os.path.join(xdg_cache, app_name)
                else:
                    return os.path.join(self.get_home_directory(), '.cache', app_name)
                    
        except Exception as e:
            logger_manager.error(f"获取缓存目录失败: {e}")
            return os.path.join(self.get_home_directory(), f'.{app_name}', 'cache')
    
    def ensure_directory(self, directory: str) -> bool:
        """
        确保目录存在
        
        Args:
            directory: 目录路径
            
        Returns:
            是否成功
        """
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            return True
            
        except Exception as e:
            logger_manager.error(f"创建目录失败: {e}")
            return False
    
    def find_executable(self, name: str) -> Optional[str]:
        """
        查找可执行文件
        
        Args:
            name: 可执行文件名
            
        Returns:
            可执行文件路径
        """
        try:
            # 添加平台特定的扩展名
            if not name.endswith(self.get_executable_extension()):
                name += self.get_executable_extension()
            
            # 使用shutil.which查找
            executable_path = shutil.which(name)
            
            if executable_path:
                logger_manager.debug(f"找到可执行文件: {name} -> {executable_path}")
                return executable_path
            else:
                logger_manager.debug(f"未找到可执行文件: {name}")
                return None
                
        except Exception as e:
            logger_manager.error(f"查找可执行文件失败: {e}")
            return None
    
    def get_environment_variable(self, name: str, default: str = None) -> Optional[str]:
        """
        获取环境变量
        
        Args:
            name: 环境变量名
            default: 默认值
            
        Returns:
            环境变量值
        """
        try:
            # 检查配置覆盖
            if name in self.config.environment_overrides:
                return self.config.environment_overrides[name]
            
            # 获取系统环境变量
            return os.environ.get(name, default)
            
        except Exception as e:
            logger_manager.error(f"获取环境变量失败: {e}")
            return default
    
    def set_environment_variable(self, name: str, value: str):
        """
        设置环境变量
        
        Args:
            name: 环境变量名
            value: 环境变量值
        """
        try:
            os.environ[name] = value
            logger_manager.debug(f"设置环境变量: {name}={value}")
            
        except Exception as e:
            logger_manager.error(f"设置环境变量失败: {e}")
    
    def is_feature_enabled(self, feature: str) -> bool:
        """
        检查功能是否启用
        
        Args:
            feature: 功能名称
            
        Returns:
            是否启用
        """
        return self.config.feature_flags.get(feature, True)


# 创建平台特定的适配器
def create_windows_adapter() -> PlatformAdapter:
    """创建Windows适配器"""
    config = AdapterConfig(
        platform_specific_commands={
            'ls': 'dir',
            'cat': 'type',
            'rm': 'del',
            'cp': 'copy',
            'mv': 'move'
        }
    )
    return PlatformAdapter(OSType.WINDOWS, config)


def create_linux_adapter() -> PlatformAdapter:
    """创建Linux适配器"""
    config = AdapterConfig()
    return PlatformAdapter(OSType.LINUX, config)


def create_macos_adapter() -> PlatformAdapter:
    """创建macOS适配器"""
    config = AdapterConfig()
    return PlatformAdapter(OSType.MACOS, config)


# 全局适配器注册表和当前适配器
adapter_registry = AdapterRegistry()

# 注册默认适配器
adapter_registry.register_adapter(OSType.WINDOWS, create_windows_adapter())
adapter_registry.register_adapter(OSType.LINUX, create_linux_adapter())
adapter_registry.register_adapter(OSType.MACOS, create_macos_adapter())

# 获取当前平台适配器
platform_adapter = adapter_registry.get_current_adapter()


if __name__ == "__main__":
    # 测试平台适配器功能
    print("🔧 测试平台适配器功能...")
    
    try:
        # 获取当前平台适配器
        adapter = adapter_registry.get_current_adapter()
        
        if adapter:
            print(f"✅ 当前平台适配器: {adapter.os_type.value}")
            
            # 测试路径处理
            test_path = "test/path/file.txt"
            normalized = adapter.normalize_path(test_path)
            print(f"✅ 路径标准化: {test_path} -> {normalized}")
            
            # 测试目录获取
            home_dir = adapter.get_home_directory()
            config_dir = adapter.get_config_directory("deep_learning_platform")
            data_dir = adapter.get_data_directory("deep_learning_platform")
            cache_dir = adapter.get_cache_directory("deep_learning_platform")
            
            print(f"✅ 主目录: {home_dir}")
            print(f"✅ 配置目录: {config_dir}")
            print(f"✅ 数据目录: {data_dir}")
            print(f"✅ 缓存目录: {cache_dir}")
            
            # 测试可执行文件查找
            python_exe = adapter.find_executable("python")
            if python_exe:
                print(f"✅ Python可执行文件: {python_exe}")
            
            # 测试环境变量
            path_var = adapter.get_environment_variable("PATH")
            if path_var:
                print(f"✅ PATH环境变量长度: {len(path_var)} 字符")
            
            print("✅ 平台适配器功能测试完成")
        
        else:
            print("❌ 无法获取当前平台适配器")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("平台适配器功能测试完成")
