"""
跨平台兼容性模块
Cross-Platform Compatibility Module

提供平台检测、适配、优雅降级、环境兼容性测试等功能。
"""

from .platform_detector import (
    PlatformDetector, PlatformInfo, OSType,
    platform_detector
)
from .platform_adapter import (
    PlatformAdapter, AdapterConfig, AdapterRegistry,
    platform_adapter
)
from .graceful_degradation import (
    GracefulDegradation, FeatureAvailability, FallbackStrategy,
    graceful_degradation
)
from .compatibility_tester import (
    CompatibilityTester, TestResult, TestSuite,
    compatibility_tester
)
from .cross_platform_installer import (
    CrossPlatformInstaller, InstallConfig, InstallResult,
    cross_platform_installer
)

__all__ = [
    # 平台检测器
    'PlatformDetector',
    'PlatformInfo',
    'OSType',
    'platform_detector',
    
    # 平台适配器
    'PlatformAdapter',
    'AdapterConfig',
    'AdapterRegistry',
    'platform_adapter',
    
    # 优雅降级
    'GracefulDegradation',
    'FeatureAvailability',
    'FallbackStrategy',
    'graceful_degradation',
    
    # 兼容性测试器
    'CompatibilityTester',
    'TestResult',
    'TestSuite',
    'compatibility_tester',
    
    # 跨平台安装器
    'CrossPlatformInstaller',
    'InstallConfig',
    'InstallResult',
    'cross_platform_installer'
]
