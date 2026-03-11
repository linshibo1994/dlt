#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
增强硬件加速器模块
Enhanced Hardware Accelerator Module

提供智能硬件检测、性能评估、加速方式选择和优雅降级功能。
"""

import os
import sys
import time
import psutil
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from core_modules import logger_manager, cache_manager
from ..platform.platform_detector import PlatformDetector, HardwareInfo


class AccelerationType(Enum):
    """加速类型枚举"""
    CPU_SINGLE = "cpu_single"
    CPU_MULTI = "cpu_multi"
    GPU_CUDA = "gpu_cuda"
    GPU_OPENCL = "gpu_opencl"
    AUTO = "auto"


@dataclass
class AccelerationConfig:
    """加速配置"""
    acceleration_type: AccelerationType
    cpu_threads: int = -1  # -1表示使用所有核心
    gpu_device_id: int = 0
    gpu_memory_limit: Optional[float] = None  # GB
    batch_size_multiplier: float = 1.0
    mixed_precision: bool = False
    fallback_enabled: bool = True


@dataclass
class PerformanceBenchmark:
    """性能基准测试结果"""
    cpu_score: float
    memory_score: float
    gpu_score: float
    overall_score: float
    recommended_acceleration: AccelerationType
    benchmark_time: float


class EnhancedHardwareAccelerator:
    """增强硬件加速器"""
    
    def __init__(self):
        """初始化增强硬件加速器"""
        self.platform_detector = PlatformDetector()
        self.platform_info = self.platform_detector.detect_platform()
        self.benchmark_cache = {}
        self.acceleration_config = None
        
        # 检测可用的加速库
        self.available_accelerations = self._detect_acceleration_libraries()
        
        logger_manager.info("增强硬件加速器初始化完成")
    
    def _detect_acceleration_libraries(self) -> Dict[str, bool]:
        """检测可用的加速库"""
        libraries = {
            'tensorflow_gpu': False,
            'pytorch_cuda': False,
            'opencl': False,
            'mkl': False,
            'openblas': False,
            'joblib': False
        }
        
        try:
            # 检测TensorFlow GPU支持
            import tensorflow as tf
            libraries['tensorflow_gpu'] = len(tf.config.list_physical_devices('GPU')) > 0
        except ImportError:
            pass
        
        try:
            # 检测PyTorch CUDA支持
            import torch
            libraries['pytorch_cuda'] = torch.cuda.is_available()
        except ImportError:
            pass
        
        try:
            # 检测OpenCL
            import pyopencl
            libraries['opencl'] = True
        except ImportError:
            pass
        
        try:
            # 检测MKL
            import mkl
            libraries['mkl'] = True
        except ImportError:
            pass
        
        try:
            # 检测OpenBLAS - 使用静默方式获取numpy配置
            import numpy as np
            import io
            import sys
            
            # 捕获stdout，因为np.__config__.show()在某些numpy版本中会打印到stdout
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()
            try:
                result = np.__config__.show()
                # numpy 2.0+ 返回字符串，旧版本打印到stdout并返回None
                if result is None:
                    config_str = captured_output.getvalue()
                else:
                    config_str = str(result)
            finally:
                sys.stdout = old_stdout
            
            libraries['openblas'] = 'openblas' in config_str.lower()
        except:
            pass
        
        try:
            # 检测joblib（用于CPU并行）
            import joblib
            libraries['joblib'] = True
        except ImportError:
            pass
        
        return libraries
    
    def benchmark_hardware(self, cache_results: bool = True) -> PerformanceBenchmark:
        """
        硬件性能基准测试
        
        Args:
            cache_results: 是否缓存结果
            
        Returns:
            性能基准测试结果
        """
        cache_key = "hardware_benchmark"
        
        if cache_results:
            cached_result = cache_manager.load_cache("performance", cache_key)
            if cached_result:
                logger_manager.info("使用缓存的性能基准测试结果")
                return PerformanceBenchmark(**cached_result)
        
        logger_manager.info("开始硬件性能基准测试...")
        start_time = time.time()
        
        # CPU性能测试
        cpu_score = self._benchmark_cpu()
        
        # 内存性能测试
        memory_score = self._benchmark_memory()
        
        # GPU性能测试
        gpu_score = self._benchmark_gpu()
        
        # 计算综合得分
        overall_score = (cpu_score * 0.4 + memory_score * 0.3 + gpu_score * 0.3)
        
        # 推荐加速方式
        recommended_acceleration = self._recommend_acceleration(cpu_score, memory_score, gpu_score)
        
        benchmark_time = time.time() - start_time
        
        result = PerformanceBenchmark(
            cpu_score=cpu_score,
            memory_score=memory_score,
            gpu_score=gpu_score,
            overall_score=overall_score,
            recommended_acceleration=recommended_acceleration,
            benchmark_time=benchmark_time
        )
        
        if cache_results:
            cache_manager.save_cache("performance", cache_key, result.__dict__)
        
        logger_manager.info(f"硬件性能基准测试完成，耗时: {benchmark_time:.2f}秒")
        logger_manager.info(f"CPU得分: {cpu_score:.2f}, 内存得分: {memory_score:.2f}, GPU得分: {gpu_score:.2f}")
        logger_manager.info(f"综合得分: {overall_score:.2f}, 推荐加速: {recommended_acceleration.value}")
        
        return result
    
    def _benchmark_cpu(self) -> float:
        """CPU性能基准测试"""
        try:
            # 矩阵运算测试
            size = 1000
            iterations = 3
            
            times = []
            for _ in range(iterations):
                start = time.time()
                a = np.random.rand(size, size).astype(np.float32)
                b = np.random.rand(size, size).astype(np.float32)
                c = np.dot(a, b)
                times.append(time.time() - start)
            
            avg_time = np.mean(times)
            # 基准时间（秒），用于计算得分
            baseline_time = 2.0  # 假设基准CPU需要2秒
            cpu_score = max(0, min(100, (baseline_time / avg_time) * 50))
            
            logger_manager.debug(f"CPU基准测试: 平均时间 {avg_time:.3f}秒, 得分 {cpu_score:.2f}")
            return cpu_score
            
        except Exception as e:
            logger_manager.error(f"CPU基准测试失败: {e}")
            return 25.0  # 默认得分
    
    def _benchmark_memory(self) -> float:
        """内存性能基准测试"""
        try:
            # 内存带宽测试
            size = 100 * 1024 * 1024  # 100MB
            iterations = 3
            
            times = []
            for _ in range(iterations):
                start = time.time()
                data = np.random.bytes(size)
                copy_data = bytearray(data)
                times.append(time.time() - start)
            
            avg_time = np.mean(times)
            # 计算内存带宽 (MB/s)
            bandwidth = (size / (1024 * 1024)) / avg_time
            
            # 基准带宽 (MB/s)，用于计算得分
            baseline_bandwidth = 5000  # 5GB/s
            memory_score = max(0, min(100, (bandwidth / baseline_bandwidth) * 50))
            
            logger_manager.debug(f"内存基准测试: 带宽 {bandwidth:.2f}MB/s, 得分 {memory_score:.2f}")
            return memory_score
            
        except Exception as e:
            logger_manager.error(f"内存基准测试失败: {e}")
            return 25.0  # 默认得分
    
    def _benchmark_gpu(self) -> float:
        """GPU性能基准测试"""
        if not self.available_accelerations.get('tensorflow_gpu', False):
            logger_manager.debug("GPU不可用，跳过GPU基准测试")
            return 0.0
        
        try:
            import tensorflow as tf
            
            # 检查GPU设备
            gpus = tf.config.list_physical_devices('GPU')
            if not gpus:
                return 0.0
            
            # GPU矩阵运算测试
            with tf.device('/GPU:0'):
                size = 2000
                iterations = 3
                
                times = []
                for _ in range(iterations):
                    start = time.time()
                    a = tf.random.normal([size, size], dtype=tf.float32)
                    b = tf.random.normal([size, size], dtype=tf.float32)
                    c = tf.matmul(a, b)
                    tf.keras.backend.get_value(c)  # 确保计算完成
                    times.append(time.time() - start)
                
                avg_time = np.mean(times)
                # 基准时间（秒），用于计算得分
                baseline_time = 0.5  # 假设基准GPU需要0.5秒
                gpu_score = max(0, min(100, (baseline_time / avg_time) * 50))
                
                logger_manager.debug(f"GPU基准测试: 平均时间 {avg_time:.3f}秒, 得分 {gpu_score:.2f}")
                return gpu_score
                
        except Exception as e:
            logger_manager.error(f"GPU基准测试失败: {e}")
            return 0.0
    
    def _recommend_acceleration(self, cpu_score: float, memory_score: float, gpu_score: float) -> AccelerationType:
        """推荐最优加速方式"""
        # 如果GPU得分很高，推荐GPU加速
        if gpu_score > 60:
            return AccelerationType.GPU_CUDA
        
        # 如果CPU核心数多且得分不错，推荐CPU多线程
        if self.platform_info.hardware_info.cpu_count >= 4 and cpu_score > 40:
            return AccelerationType.CPU_MULTI
        
        # 否则使用CPU单线程
        return AccelerationType.CPU_SINGLE
    
    def configure_acceleration(self, config: Optional[AccelerationConfig] = None) -> AccelerationConfig:
        """
        配置加速方式
        
        Args:
            config: 加速配置，如果为None则使用自动配置
            
        Returns:
            最终的加速配置
        """
        if config is None:
            # 自动配置
            benchmark = self.benchmark_hardware()
            config = AccelerationConfig(
                acceleration_type=benchmark.recommended_acceleration,
                cpu_threads=self.platform_info.hardware_info.cpu_count,
                gpu_device_id=0,
                gpu_memory_limit=None,
                batch_size_multiplier=1.0,
                mixed_precision=gpu_score > 70,  # 高性能GPU启用混合精度
                fallback_enabled=True
            )
        
        self.acceleration_config = config
        logger_manager.info(f"加速配置: {config.acceleration_type.value}")
        
        return config
    
    def get_optimal_batch_size(self, base_batch_size: int, model_type: str = "default") -> int:
        """
        获取最优批次大小
        
        Args:
            base_batch_size: 基础批次大小
            model_type: 模型类型
            
        Returns:
            优化后的批次大小
        """
        if self.acceleration_config is None:
            self.configure_acceleration()
        
        multiplier = self.acceleration_config.batch_size_multiplier
        
        # 根据加速类型调整
        if self.acceleration_config.acceleration_type == AccelerationType.GPU_CUDA:
            # GPU可以处理更大的批次
            multiplier *= 2.0
        elif self.acceleration_config.acceleration_type == AccelerationType.CPU_MULTI:
            # CPU多线程适中调整
            multiplier *= 1.5
        
        # 根据可用内存调整
        available_memory_gb = self.platform_info.hardware_info.memory_available / (1024**3)
        if available_memory_gb < 4:
            multiplier *= 0.5
        elif available_memory_gb > 16:
            multiplier *= 1.5
        
        optimal_batch_size = max(1, int(base_batch_size * multiplier))
        
        logger_manager.debug(f"批次大小优化: {base_batch_size} -> {optimal_batch_size}")
        return optimal_batch_size
    
    def get_optimal_thread_count(self, task_type: str = "default") -> int:
        """
        获取最优线程数
        
        Args:
            task_type: 任务类型
            
        Returns:
            最优线程数
        """
        cpu_count = self.platform_info.hardware_info.cpu_count
        
        # 根据任务类型调整
        if task_type in ["clustering", "markov", "bayesian"]:
            # CPU密集型任务
            return max(1, cpu_count - 1)  # 保留一个核心给系统
        elif task_type in ["data_processing", "feature_extraction"]:
            # I/O密集型任务
            return min(cpu_count * 2, 16)  # 最多16个线程
        else:
            # 默认情况
            return max(1, cpu_count // 2)

    def detect_hardware(self) -> HardwareInfo:
        """
        检测硬件信息

        Returns:
            硬件信息对象
        """
        try:
            # 使用平台检测器获取硬件信息
            platform_info = self.platform_detector.detect_platform()
            hardware_info = platform_info.hardware_info

            # 检测GPU和CUDA支持
            gpu_available = False
            cuda_available = False
            cuda_version = ""

            try:
                import tensorflow as tf
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    gpu_available = True
                    cuda_available = True
                    # 尝试获取CUDA版本
                    try:
                        cuda_version = tf.sysconfig.get_build_info()['cuda_version']
                    except:
                        cuda_version = "unknown"
            except ImportError:
                pass

            # 更新硬件信息
            hardware_info.gpu_count = len(tf.config.list_physical_devices('GPU')) if 'tf' in locals() else 0
            hardware_info.cuda_available = cuda_available
            hardware_info.cuda_version = cuda_version

            # 检测其他加速库
            hardware_info.mkl_available = self.available_accelerations.get('mkl', False)
            hardware_info.opencl_available = self.available_accelerations.get('opencl', False)

            logger_manager.info(f"硬件检测完成: CPU {hardware_info.cpu_count}核, 内存 {hardware_info.memory_total//1024//1024//1024}GB")
            if hardware_info.cuda_available:
                logger_manager.info(f"GPU支持: CUDA {hardware_info.cuda_version}, {hardware_info.gpu_count}个设备")

            return hardware_info

        except Exception as e:
            logger_manager.error(f"硬件检测失败: {e}")
            # 返回默认硬件信息
            import platform
            return HardwareInfo(
                cpu_count=psutil.cpu_count(),
                cpu_freq=psutil.cpu_freq().current if psutil.cpu_freq() else 2000.0,
                cpu_brand=platform.processor() or "Unknown",
                cpu_architecture=platform.machine() or "Unknown",
                memory_total=psutil.virtual_memory().total,
                memory_available=psutil.virtual_memory().available,
                disk_total=psutil.disk_usage('/').total,
                disk_free=psutil.disk_usage('/').free,
                gpu_count=0,
                gpu_info=[],
                cuda_available=False,
                cuda_version="",
                opencl_available=False,
                mkl_available=False,
                performance_score=50.0
            )
