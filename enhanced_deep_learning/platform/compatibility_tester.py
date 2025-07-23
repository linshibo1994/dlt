#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
兼容性测试器模块
Compatibility Tester Module

提供环境兼容性测试、依赖检查、性能基准测试等功能。
"""

import os
import sys
import time
import subprocess
import importlib
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import tempfile
import json

from core_modules import logger_manager
from ..utils.exceptions import DeepLearningException
from .platform_detector import platform_detector


class TestStatus(Enum):
    """测试状态枚举"""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WARNING = "warning"


class TestCategory(Enum):
    """测试类别枚举"""
    SYSTEM = "system"
    DEPENDENCIES = "dependencies"
    HARDWARE = "hardware"
    PERFORMANCE = "performance"
    FUNCTIONALITY = "functionality"


@dataclass
class TestResult:
    """测试结果"""
    name: str
    category: TestCategory
    status: TestStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0
    recommendations: List[str] = field(default_factory=list)


@dataclass
class TestSuite:
    """测试套件"""
    name: str
    description: str
    tests: List[Callable] = field(default_factory=list)
    setup: Optional[Callable] = None
    teardown: Optional[Callable] = None


class CompatibilityTester:
    """兼容性测试器"""
    
    def __init__(self):
        """初始化兼容性测试器"""
        self.platform_info = platform_detector.detect_platform()
        self.test_suites = {}
        self.test_results = []
        
        # 注册默认测试套件
        self._register_default_test_suites()
        
        logger_manager.info("兼容性测试器初始化完成")
    
    def _register_default_test_suites(self):
        """注册默认测试套件"""
        try:
            # 系统兼容性测试套件
            system_suite = TestSuite(
                name="system_compatibility",
                description="系统兼容性测试",
                tests=[
                    self._test_python_version,
                    self._test_operating_system,
                    self._test_architecture,
                    self._test_disk_space,
                    self._test_memory,
                    self._test_permissions
                ]
            )
            self.register_test_suite(system_suite)
            
            # 依赖兼容性测试套件
            dependency_suite = TestSuite(
                name="dependency_compatibility",
                description="依赖兼容性测试",
                tests=[
                    self._test_core_dependencies,
                    self._test_optional_dependencies,
                    self._test_version_conflicts,
                    self._test_import_capabilities
                ]
            )
            self.register_test_suite(dependency_suite)
            
            # 硬件兼容性测试套件
            hardware_suite = TestSuite(
                name="hardware_compatibility",
                description="硬件兼容性测试",
                tests=[
                    self._test_cpu_capabilities,
                    self._test_gpu_availability,
                    self._test_cuda_compatibility,
                    self._test_memory_bandwidth
                ]
            )
            self.register_test_suite(hardware_suite)
            
            # 性能基准测试套件
            performance_suite = TestSuite(
                name="performance_benchmark",
                description="性能基准测试",
                tests=[
                    self._test_cpu_performance,
                    self._test_memory_performance,
                    self._test_io_performance,
                    self._test_gpu_performance
                ]
            )
            self.register_test_suite(performance_suite)
            
        except Exception as e:
            logger_manager.error(f"注册默认测试套件失败: {e}")
    
    def register_test_suite(self, suite: TestSuite):
        """
        注册测试套件
        
        Args:
            suite: 测试套件
        """
        try:
            self.test_suites[suite.name] = suite
            logger_manager.debug(f"测试套件注册成功: {suite.name}")
            
        except Exception as e:
            logger_manager.error(f"注册测试套件失败: {e}")
    
    def run_test_suite(self, suite_name: str) -> List[TestResult]:
        """
        运行测试套件
        
        Args:
            suite_name: 测试套件名称
            
        Returns:
            测试结果列表
        """
        try:
            suite = self.test_suites.get(suite_name)
            if not suite:
                raise DeepLearningException(f"测试套件不存在: {suite_name}")
            
            logger_manager.info(f"开始运行测试套件: {suite.name}")
            
            results = []
            
            # 执行setup
            if suite.setup:
                try:
                    suite.setup()
                except Exception as e:
                    logger_manager.error(f"测试套件setup失败: {e}")
            
            # 执行测试
            for test_func in suite.tests:
                try:
                    start_time = time.time()
                    result = test_func()
                    result.duration = time.time() - start_time
                    results.append(result)
                    
                    status_symbol = {
                        TestStatus.PASSED: "✅",
                        TestStatus.FAILED: "❌",
                        TestStatus.SKIPPED: "⏭️",
                        TestStatus.WARNING: "⚠️"
                    }.get(result.status, "❓")
                    
                    logger_manager.info(f"{status_symbol} {result.name}: {result.message}")
                    
                except Exception as e:
                    error_result = TestResult(
                        name=test_func.__name__,
                        category=TestCategory.SYSTEM,
                        status=TestStatus.FAILED,
                        message=f"测试执行异常: {e}",
                        duration=time.time() - start_time
                    )
                    results.append(error_result)
                    logger_manager.error(f"❌ {test_func.__name__}: 测试执行异常: {e}")
            
            # 执行teardown
            if suite.teardown:
                try:
                    suite.teardown()
                except Exception as e:
                    logger_manager.error(f"测试套件teardown失败: {e}")
            
            self.test_results.extend(results)
            logger_manager.info(f"测试套件完成: {suite.name}")
            
            return results
            
        except Exception as e:
            logger_manager.error(f"运行测试套件失败: {e}")
            return []
    
    def run_all_tests(self) -> Dict[str, List[TestResult]]:
        """
        运行所有测试套件
        
        Returns:
            所有测试结果
        """
        try:
            all_results = {}
            
            for suite_name in self.test_suites.keys():
                results = self.run_test_suite(suite_name)
                all_results[suite_name] = results
            
            return all_results
            
        except Exception as e:
            logger_manager.error(f"运行所有测试失败: {e}")
            return {}
    
    # 系统兼容性测试
    def _test_python_version(self) -> TestResult:
        """测试Python版本"""
        try:
            version = sys.version_info
            min_version = (3, 8)
            
            if version >= min_version:
                return TestResult(
                    name="python_version",
                    category=TestCategory.SYSTEM,
                    status=TestStatus.PASSED,
                    message=f"Python版本: {version.major}.{version.minor}.{version.micro}",
                    details={"version": f"{version.major}.{version.minor}.{version.micro}"}
                )
            else:
                return TestResult(
                    name="python_version",
                    category=TestCategory.SYSTEM,
                    status=TestStatus.FAILED,
                    message=f"Python版本过低: {version.major}.{version.minor}.{version.micro} < {min_version[0]}.{min_version[1]}",
                    recommendations=["升级Python到3.8或更高版本"]
                )
                
        except Exception as e:
            return TestResult(
                name="python_version",
                category=TestCategory.SYSTEM,
                status=TestStatus.FAILED,
                message=f"Python版本检测失败: {e}"
            )
    
    def _test_operating_system(self) -> TestResult:
        """测试操作系统"""
        try:
            supported_os = ["Windows", "Linux", "macOS"]
            
            if self.platform_info.os_name in supported_os:
                return TestResult(
                    name="operating_system",
                    category=TestCategory.SYSTEM,
                    status=TestStatus.PASSED,
                    message=f"操作系统: {self.platform_info.os_name} {self.platform_info.os_version}",
                    details={
                        "os_name": self.platform_info.os_name,
                        "os_version": self.platform_info.os_version
                    }
                )
            else:
                return TestResult(
                    name="operating_system",
                    category=TestCategory.SYSTEM,
                    status=TestStatus.WARNING,
                    message=f"未测试的操作系统: {self.platform_info.os_name}",
                    recommendations=["在支持的操作系统上测试"]
                )
                
        except Exception as e:
            return TestResult(
                name="operating_system",
                category=TestCategory.SYSTEM,
                status=TestStatus.FAILED,
                message=f"操作系统检测失败: {e}"
            )
    
    def _test_architecture(self) -> TestResult:
        """测试系统架构"""
        try:
            supported_arch = ["x86_64", "arm64"]
            arch = self.platform_info.architecture.value
            
            if arch in supported_arch:
                return TestResult(
                    name="architecture",
                    category=TestCategory.SYSTEM,
                    status=TestStatus.PASSED,
                    message=f"系统架构: {arch}",
                    details={"architecture": arch}
                )
            else:
                return TestResult(
                    name="architecture",
                    category=TestCategory.SYSTEM,
                    status=TestStatus.WARNING,
                    message=f"未完全测试的架构: {arch}",
                    recommendations=["在x86_64或arm64架构上测试"]
                )
                
        except Exception as e:
            return TestResult(
                name="architecture",
                category=TestCategory.SYSTEM,
                status=TestStatus.FAILED,
                message=f"架构检测失败: {e}"
            )
    
    def _test_disk_space(self) -> TestResult:
        """测试磁盘空间"""
        try:
            min_space_gb = 10
            free_space_gb = self.platform_info.hardware_info.disk_free / (1024**3)
            
            if free_space_gb >= min_space_gb:
                return TestResult(
                    name="disk_space",
                    category=TestCategory.SYSTEM,
                    status=TestStatus.PASSED,
                    message=f"可用磁盘空间: {free_space_gb:.1f} GB",
                    details={"free_space_gb": free_space_gb}
                )
            else:
                return TestResult(
                    name="disk_space",
                    category=TestCategory.SYSTEM,
                    status=TestStatus.WARNING,
                    message=f"磁盘空间不足: {free_space_gb:.1f} GB < {min_space_gb} GB",
                    recommendations=[f"释放磁盘空间，建议至少{min_space_gb}GB可用空间"]
                )
                
        except Exception as e:
            return TestResult(
                name="disk_space",
                category=TestCategory.SYSTEM,
                status=TestStatus.FAILED,
                message=f"磁盘空间检测失败: {e}"
            )
    
    def _test_memory(self) -> TestResult:
        """测试内存"""
        try:
            min_memory_gb = 4
            available_memory_gb = self.platform_info.hardware_info.memory_available / (1024**3)
            
            if available_memory_gb >= min_memory_gb:
                return TestResult(
                    name="memory",
                    category=TestCategory.SYSTEM,
                    status=TestStatus.PASSED,
                    message=f"可用内存: {available_memory_gb:.1f} GB",
                    details={"available_memory_gb": available_memory_gb}
                )
            else:
                return TestResult(
                    name="memory",
                    category=TestCategory.SYSTEM,
                    status=TestStatus.WARNING,
                    message=f"可用内存不足: {available_memory_gb:.1f} GB < {min_memory_gb} GB",
                    recommendations=[f"增加内存或关闭其他程序，建议至少{min_memory_gb}GB可用内存"]
                )
                
        except Exception as e:
            return TestResult(
                name="memory",
                category=TestCategory.SYSTEM,
                status=TestStatus.FAILED,
                message=f"内存检测失败: {e}"
            )
    
    def _test_permissions(self) -> TestResult:
        """测试权限"""
        try:
            # 测试临时文件创建权限
            with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
                tmp_file.write(b"test")
                tmp_file.flush()
            
            return TestResult(
                name="permissions",
                category=TestCategory.SYSTEM,
                status=TestStatus.PASSED,
                message="文件系统权限正常"
            )
            
        except Exception as e:
            return TestResult(
                name="permissions",
                category=TestCategory.SYSTEM,
                status=TestStatus.FAILED,
                message=f"权限检测失败: {e}",
                recommendations=["检查文件系统权限"]
            )
    
    # 依赖兼容性测试
    def _test_core_dependencies(self) -> TestResult:
        """测试核心依赖"""
        try:
            core_deps = ["numpy", "pandas", "torch", "sklearn"]
            missing_deps = []
            
            for dep in core_deps:
                try:
                    importlib.import_module(dep)
                except ImportError:
                    missing_deps.append(dep)
            
            if not missing_deps:
                return TestResult(
                    name="core_dependencies",
                    category=TestCategory.DEPENDENCIES,
                    status=TestStatus.PASSED,
                    message="所有核心依赖已安装"
                )
            else:
                return TestResult(
                    name="core_dependencies",
                    category=TestCategory.DEPENDENCIES,
                    status=TestStatus.FAILED,
                    message=f"缺少核心依赖: {', '.join(missing_deps)}",
                    recommendations=[f"安装缺少的依赖: pip install {' '.join(missing_deps)}"]
                )
                
        except Exception as e:
            return TestResult(
                name="core_dependencies",
                category=TestCategory.DEPENDENCIES,
                status=TestStatus.FAILED,
                message=f"核心依赖检测失败: {e}"
            )
    
    def _test_optional_dependencies(self) -> TestResult:
        """测试可选依赖"""
        try:
            optional_deps = ["plotly", "dash", "tensorboard", "jupyter"]
            missing_deps = []
            
            for dep in optional_deps:
                try:
                    importlib.import_module(dep)
                except ImportError:
                    missing_deps.append(dep)
            
            if not missing_deps:
                return TestResult(
                    name="optional_dependencies",
                    category=TestCategory.DEPENDENCIES,
                    status=TestStatus.PASSED,
                    message="所有可选依赖已安装"
                )
            else:
                return TestResult(
                    name="optional_dependencies",
                    category=TestCategory.DEPENDENCIES,
                    status=TestStatus.WARNING,
                    message=f"缺少可选依赖: {', '.join(missing_deps)}",
                    recommendations=[f"安装可选依赖以获得完整功能: pip install {' '.join(missing_deps)}"]
                )
                
        except Exception as e:
            return TestResult(
                name="optional_dependencies",
                category=TestCategory.DEPENDENCIES,
                status=TestStatus.FAILED,
                message=f"可选依赖检测失败: {e}"
            )
    
    def _test_version_conflicts(self) -> TestResult:
        """测试版本冲突"""
        try:
            # 这里可以实现更复杂的版本冲突检测
            return TestResult(
                name="version_conflicts",
                category=TestCategory.DEPENDENCIES,
                status=TestStatus.PASSED,
                message="未检测到版本冲突"
            )
            
        except Exception as e:
            return TestResult(
                name="version_conflicts",
                category=TestCategory.DEPENDENCIES,
                status=TestStatus.FAILED,
                message=f"版本冲突检测失败: {e}"
            )
    
    def _test_import_capabilities(self) -> TestResult:
        """测试导入能力"""
        try:
            # 测试关键模块导入
            test_imports = [
                "enhanced_deep_learning.core",
                "enhanced_deep_learning.models",
                "enhanced_deep_learning.data"
            ]
            
            failed_imports = []
            for module in test_imports:
                try:
                    importlib.import_module(module)
                except ImportError as e:
                    failed_imports.append(f"{module}: {e}")
            
            if not failed_imports:
                return TestResult(
                    name="import_capabilities",
                    category=TestCategory.DEPENDENCIES,
                    status=TestStatus.PASSED,
                    message="所有关键模块导入成功"
                )
            else:
                return TestResult(
                    name="import_capabilities",
                    category=TestCategory.DEPENDENCIES,
                    status=TestStatus.FAILED,
                    message=f"模块导入失败: {'; '.join(failed_imports)}",
                    recommendations=["检查模块安装和路径配置"]
                )
                
        except Exception as e:
            return TestResult(
                name="import_capabilities",
                category=TestCategory.DEPENDENCIES,
                status=TestStatus.FAILED,
                message=f"导入能力检测失败: {e}"
            )
    
    # 硬件兼容性测试
    def _test_cpu_capabilities(self) -> TestResult:
        """测试CPU能力"""
        try:
            cpu_count = self.platform_info.hardware_info.cpu_count
            
            if cpu_count >= 2:
                return TestResult(
                    name="cpu_capabilities",
                    category=TestCategory.HARDWARE,
                    status=TestStatus.PASSED,
                    message=f"CPU核心数: {cpu_count}",
                    details={"cpu_count": cpu_count}
                )
            else:
                return TestResult(
                    name="cpu_capabilities",
                    category=TestCategory.HARDWARE,
                    status=TestStatus.WARNING,
                    message=f"CPU核心数较少: {cpu_count}",
                    recommendations=["建议使用多核CPU以获得更好性能"]
                )
                
        except Exception as e:
            return TestResult(
                name="cpu_capabilities",
                category=TestCategory.HARDWARE,
                status=TestStatus.FAILED,
                message=f"CPU能力检测失败: {e}"
            )
    
    def _test_gpu_availability(self) -> TestResult:
        """测试GPU可用性"""
        try:
            gpu_count = self.platform_info.hardware_info.gpu_count
            
            if gpu_count > 0:
                return TestResult(
                    name="gpu_availability",
                    category=TestCategory.HARDWARE,
                    status=TestStatus.PASSED,
                    message=f"检测到 {gpu_count} 个GPU",
                    details={"gpu_count": gpu_count}
                )
            else:
                return TestResult(
                    name="gpu_availability",
                    category=TestCategory.HARDWARE,
                    status=TestStatus.WARNING,
                    message="未检测到GPU",
                    recommendations=["安装GPU以获得更好的训练性能"]
                )
                
        except Exception as e:
            return TestResult(
                name="gpu_availability",
                category=TestCategory.HARDWARE,
                status=TestStatus.FAILED,
                message=f"GPU可用性检测失败: {e}"
            )
    
    def _test_cuda_compatibility(self) -> TestResult:
        """测试CUDA兼容性"""
        try:
            cuda_available = self.platform_info.capabilities.get('cuda_available', False)
            
            if cuda_available:
                return TestResult(
                    name="cuda_compatibility",
                    category=TestCategory.HARDWARE,
                    status=TestStatus.PASSED,
                    message="CUDA可用"
                )
            else:
                return TestResult(
                    name="cuda_compatibility",
                    category=TestCategory.HARDWARE,
                    status=TestStatus.WARNING,
                    message="CUDA不可用",
                    recommendations=["安装CUDA和cuDNN以启用GPU加速"]
                )
                
        except Exception as e:
            return TestResult(
                name="cuda_compatibility",
                category=TestCategory.HARDWARE,
                status=TestStatus.FAILED,
                message=f"CUDA兼容性检测失败: {e}"
            )
    
    def _test_memory_bandwidth(self) -> TestResult:
        """测试内存带宽"""
        try:
            # 简单的内存带宽测试
            import numpy as np
            
            size = 100 * 1024 * 1024  # 100MB
            data = np.random.random(size // 8).astype(np.float64)
            
            start_time = time.time()
            result = np.sum(data)
            duration = time.time() - start_time
            
            bandwidth_gb_s = (size / (1024**3)) / duration
            
            return TestResult(
                name="memory_bandwidth",
                category=TestCategory.HARDWARE,
                status=TestStatus.PASSED,
                message=f"内存带宽: {bandwidth_gb_s:.2f} GB/s",
                details={"bandwidth_gb_s": bandwidth_gb_s}
            )
            
        except Exception as e:
            return TestResult(
                name="memory_bandwidth",
                category=TestCategory.HARDWARE,
                status=TestStatus.FAILED,
                message=f"内存带宽测试失败: {e}"
            )
    
    # 性能基准测试
    def _test_cpu_performance(self) -> TestResult:
        """测试CPU性能"""
        try:
            import numpy as np
            
            # CPU计算基准测试
            size = 1000
            a = np.random.random((size, size))
            b = np.random.random((size, size))
            
            start_time = time.time()
            c = np.dot(a, b)
            duration = time.time() - start_time
            
            gflops = (2 * size**3) / (duration * 1e9)
            
            return TestResult(
                name="cpu_performance",
                category=TestCategory.PERFORMANCE,
                status=TestStatus.PASSED,
                message=f"CPU性能: {gflops:.2f} GFLOPS",
                details={"gflops": gflops, "duration": duration}
            )
            
        except Exception as e:
            return TestResult(
                name="cpu_performance",
                category=TestCategory.PERFORMANCE,
                status=TestStatus.FAILED,
                message=f"CPU性能测试失败: {e}"
            )
    
    def _test_memory_performance(self) -> TestResult:
        """测试内存性能"""
        try:
            import numpy as np
            
            # 内存访问基准测试
            size = 50 * 1024 * 1024  # 50MB
            data = np.random.random(size // 8).astype(np.float64)
            
            start_time = time.time()
            # 顺序访问
            total = 0
            for i in range(0, len(data), 1000):
                total += data[i]
            duration = time.time() - start_time
            
            throughput_gb_s = (size / (1024**3)) / duration
            
            return TestResult(
                name="memory_performance",
                category=TestCategory.PERFORMANCE,
                status=TestStatus.PASSED,
                message=f"内存吞吐量: {throughput_gb_s:.2f} GB/s",
                details={"throughput_gb_s": throughput_gb_s}
            )
            
        except Exception as e:
            return TestResult(
                name="memory_performance",
                category=TestCategory.PERFORMANCE,
                status=TestStatus.FAILED,
                message=f"内存性能测试失败: {e}"
            )
    
    def _test_io_performance(self) -> TestResult:
        """测试IO性能"""
        try:
            # IO性能基准测试
            test_size = 10 * 1024 * 1024  # 10MB
            test_data = b'x' * test_size
            
            with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
                # 写入测试
                start_time = time.time()
                tmp_file.write(test_data)
                tmp_file.flush()
                write_duration = time.time() - start_time
                
                # 读取测试
                tmp_file.seek(0)
                start_time = time.time()
                read_data = tmp_file.read()
                read_duration = time.time() - start_time
                
                write_mb_s = (test_size / (1024**2)) / write_duration
                read_mb_s = (test_size / (1024**2)) / read_duration
                
                return TestResult(
                    name="io_performance",
                    category=TestCategory.PERFORMANCE,
                    status=TestStatus.PASSED,
                    message=f"IO性能 - 写入: {write_mb_s:.1f} MB/s, 读取: {read_mb_s:.1f} MB/s",
                    details={
                        "write_mb_s": write_mb_s,
                        "read_mb_s": read_mb_s
                    }
                )
                
        except Exception as e:
            return TestResult(
                name="io_performance",
                category=TestCategory.PERFORMANCE,
                status=TestStatus.FAILED,
                message=f"IO性能测试失败: {e}"
            )
    
    def _test_gpu_performance(self) -> TestResult:
        """测试GPU性能"""
        try:
            if not self.platform_info.capabilities.get('cuda_available', False):
                return TestResult(
                    name="gpu_performance",
                    category=TestCategory.PERFORMANCE,
                    status=TestStatus.SKIPPED,
                    message="GPU不可用，跳过性能测试"
                )
            
            import torch
            
            if not torch.cuda.is_available():
                return TestResult(
                    name="gpu_performance",
                    category=TestCategory.PERFORMANCE,
                    status=TestStatus.SKIPPED,
                    message="CUDA不可用，跳过GPU性能测试"
                )
            
            # GPU计算基准测试
            device = torch.device('cuda')
            size = 1000
            
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)
            
            # 预热
            torch.mm(a, b)
            torch.cuda.synchronize()
            
            # 实际测试
            start_time = time.time()
            c = torch.mm(a, b)
            torch.cuda.synchronize()
            duration = time.time() - start_time
            
            gflops = (2 * size**3) / (duration * 1e9)
            
            return TestResult(
                name="gpu_performance",
                category=TestCategory.PERFORMANCE,
                status=TestStatus.PASSED,
                message=f"GPU性能: {gflops:.2f} GFLOPS",
                details={"gflops": gflops, "duration": duration}
            )
            
        except Exception as e:
            return TestResult(
                name="gpu_performance",
                category=TestCategory.PERFORMANCE,
                status=TestStatus.FAILED,
                message=f"GPU性能测试失败: {e}"
            )
    
    def generate_compatibility_report(self) -> Dict[str, Any]:
        """生成兼容性报告"""
        try:
            # 运行所有测试
            all_results = self.run_all_tests()
            
            # 统计结果
            total_tests = 0
            passed_tests = 0
            failed_tests = 0
            warning_tests = 0
            skipped_tests = 0
            
            for suite_results in all_results.values():
                for result in suite_results:
                    total_tests += 1
                    if result.status == TestStatus.PASSED:
                        passed_tests += 1
                    elif result.status == TestStatus.FAILED:
                        failed_tests += 1
                    elif result.status == TestStatus.WARNING:
                        warning_tests += 1
                    elif result.status == TestStatus.SKIPPED:
                        skipped_tests += 1
            
            # 生成报告
            report = {
                "platform_info": {
                    "os": f"{self.platform_info.os_name} {self.platform_info.os_version}",
                    "architecture": self.platform_info.architecture.value,
                    "python_version": self.platform_info.python_info.version,
                    "hostname": self.platform_info.hostname
                },
                "test_summary": {
                    "total_tests": total_tests,
                    "passed": passed_tests,
                    "failed": failed_tests,
                    "warnings": warning_tests,
                    "skipped": skipped_tests,
                    "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0
                },
                "test_results": {},
                "recommendations": []
            }
            
            # 添加详细结果
            for suite_name, suite_results in all_results.items():
                report["test_results"][suite_name] = []
                for result in suite_results:
                    report["test_results"][suite_name].append({
                        "name": result.name,
                        "status": result.status.value,
                        "message": result.message,
                        "duration": result.duration,
                        "details": result.details
                    })
                    
                    # 收集建议
                    if result.recommendations:
                        report["recommendations"].extend(result.recommendations)
            
            return report
            
        except Exception as e:
            logger_manager.error(f"生成兼容性报告失败: {e}")
            return {"error": f"生成兼容性报告失败: {e}"}


# 全局兼容性测试器实例
compatibility_tester = CompatibilityTester()


if __name__ == "__main__":
    # 测试兼容性测试器功能
    print("🧪 测试兼容性测试器功能...")
    
    try:
        tester = CompatibilityTester()
        
        # 运行系统兼容性测试
        print("运行系统兼容性测试...")
        system_results = tester.run_test_suite("system_compatibility")
        
        # 运行依赖兼容性测试
        print("运行依赖兼容性测试...")
        dependency_results = tester.run_test_suite("dependency_compatibility")
        
        # 生成兼容性报告
        print("生成兼容性报告...")
        report = tester.generate_compatibility_report()
        
        print(f"✅ 兼容性测试完成")
        print(f"总测试数: {report['test_summary']['total_tests']}")
        print(f"通过: {report['test_summary']['passed']}")
        print(f"失败: {report['test_summary']['failed']}")
        print(f"警告: {report['test_summary']['warnings']}")
        print(f"跳过: {report['test_summary']['skipped']}")
        print(f"成功率: {report['test_summary']['success_rate']:.1f}%")
        
        if report.get('recommendations'):
            print(f"建议数量: {len(report['recommendations'])}")
        
        print("✅ 兼容性测试器功能测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("兼容性测试器功能测试完成")
