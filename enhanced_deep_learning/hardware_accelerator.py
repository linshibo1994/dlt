#!/usr/bin/env python3
"""
硬件加速抽象层
提供跨平台的GPU/TPU加速支持
"""

import os
import platform
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
import numpy as np

from core_modules import logger_manager, config_manager

# 尝试导入各种加速库
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


class AcceleratorType(Enum):
    """加速器类型"""
    CPU = "cpu"
    GPU_CUDA = "gpu_cuda"
    GPU_OPENCL = "gpu_opencl"
    TPU = "tpu"
    MPS = "mps"  # Apple Metal Performance Shaders


class HardwareAccelerator:
    """硬件加速抽象层"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化硬件加速器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        
        # 加速器配置
        self.preferred_accelerator = self.config.get('preferred_accelerator', 'auto')
        self.memory_limit = self.config.get('memory_limit', None)
        self.allow_growth = self.config.get('allow_growth', True)
        
        # 检测可用的加速器
        self.available_accelerators = self._detect_accelerators()
        self.current_accelerator = self._select_accelerator()
        
        # 初始化加速器
        self._initialize_accelerator()
        
        logger_manager.info(f"硬件加速器初始化完成，当前使用: {self.current_accelerator.value}")
    
    def _detect_accelerators(self) -> List[AcceleratorType]:
        """检测可用的加速器"""
        available = [AcceleratorType.CPU]  # CPU总是可用
        
        try:
            # 检测NVIDIA GPU (CUDA)
            if TF_AVAILABLE:
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    available.append(AcceleratorType.GPU_CUDA)
            
            if TORCH_AVAILABLE:
                if torch.cuda.is_available():
                    available.append(AcceleratorType.GPU_CUDA)
            
            # 检测Apple MPS
            if platform.system() == 'Darwin':  # macOS
                if TF_AVAILABLE:
                    try:
                        # 检查是否支持Metal
                        if hasattr(tf.config.experimental, 'list_physical_devices'):
                            metal_devices = tf.config.experimental.list_physical_devices('GPU')
                            if metal_devices:
                                available.append(AcceleratorType.MPS)
                    except:
                        pass
                
                if TORCH_AVAILABLE:
                    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        available.append(AcceleratorType.MPS)
            
            # 检测TPU
            if TF_AVAILABLE:
                try:
                    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
                    if tpu:
                        available.append(AcceleratorType.TPU)
                except:
                    pass
            
            # 检测OpenCL GPU
            if CUPY_AVAILABLE:
                try:
                    cp.cuda.runtime.getDeviceCount()
                    available.append(AcceleratorType.GPU_OPENCL)
                except:
                    pass
            
        except Exception as e:
            logger_manager.warning(f"检测加速器时出错: {e}")
        
        logger_manager.info(f"检测到可用加速器: {[acc.value for acc in available]}")
        
        return available
    
    def _select_accelerator(self) -> AcceleratorType:
        """选择最佳加速器"""
        if self.preferred_accelerator == 'auto':
            # 自动选择最佳加速器
            priority = [
                AcceleratorType.TPU,
                AcceleratorType.GPU_CUDA,
                AcceleratorType.MPS,
                AcceleratorType.GPU_OPENCL,
                AcceleratorType.CPU
            ]
            
            for acc_type in priority:
                if acc_type in self.available_accelerators:
                    return acc_type
            
            return AcceleratorType.CPU
        
        else:
            # 使用指定的加速器
            try:
                preferred = AcceleratorType(self.preferred_accelerator)
                if preferred in self.available_accelerators:
                    return preferred
                else:
                    logger_manager.warning(f"指定的加速器 {self.preferred_accelerator} 不可用，回退到CPU")
                    return AcceleratorType.CPU
            except ValueError:
                logger_manager.error(f"无效的加速器类型: {self.preferred_accelerator}")
                return AcceleratorType.CPU
    
    def _initialize_accelerator(self):
        """初始化选定的加速器"""
        try:
            if self.current_accelerator == AcceleratorType.GPU_CUDA:
                self._initialize_cuda()
            elif self.current_accelerator == AcceleratorType.MPS:
                self._initialize_mps()
            elif self.current_accelerator == AcceleratorType.TPU:
                self._initialize_tpu()
            elif self.current_accelerator == AcceleratorType.CPU:
                self._initialize_cpu()
            
            logger_manager.info(f"加速器 {self.current_accelerator.value} 初始化成功")
            
        except Exception as e:
            logger_manager.error(f"初始化加速器失败: {e}")
            # 回退到CPU
            self.current_accelerator = AcceleratorType.CPU
            self._initialize_cpu()
    
    def _initialize_cuda(self):
        """初始化CUDA GPU"""
        if TF_AVAILABLE:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        if self.allow_growth:
                            tf.config.experimental.set_memory_growth(gpu, True)
                        
                        if self.memory_limit:
                            tf.config.experimental.set_memory_limit(gpu, self.memory_limit)
                    
                    logger_manager.info(f"TensorFlow GPU配置完成，检测到 {len(gpus)} 个GPU")
                    
                except RuntimeError as e:
                    logger_manager.error(f"TensorFlow GPU配置失败: {e}")
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            
            logger_manager.info(f"PyTorch CUDA配置完成，当前设备: {device_name} ({device_count} 个GPU)")
    
    def _initialize_mps(self):
        """初始化Apple MPS"""
        if TORCH_AVAILABLE and hasattr(torch.backends, 'mps'):
            if torch.backends.mps.is_available():
                logger_manager.info("PyTorch MPS配置完成")
        
        if TF_AVAILABLE:
            # TensorFlow Metal配置
            logger_manager.info("TensorFlow Metal配置完成")
    
    def _initialize_tpu(self):
        """初始化TPU"""
        if TF_AVAILABLE:
            try:
                resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
                tf.config.experimental_connect_to_cluster(resolver)
                tf.tpu.experimental.initialize_tpu_system(resolver)
                
                strategy = tf.distribute.TPUStrategy(resolver)
                logger_manager.info(f"TPU配置完成，副本数: {strategy.num_replicas_in_sync}")
                
            except Exception as e:
                logger_manager.error(f"TPU配置失败: {e}")
                raise
    
    def _initialize_cpu(self):
        """初始化CPU"""
        if TF_AVAILABLE:
            # 设置CPU线程数
            cpu_count = os.cpu_count()
            tf.config.threading.set_intra_op_parallelism_threads(cpu_count)
            tf.config.threading.set_inter_op_parallelism_threads(cpu_count)
            
            logger_manager.info(f"TensorFlow CPU配置完成，使用 {cpu_count} 个线程")
        
        if TORCH_AVAILABLE:
            # 设置PyTorch CPU线程数
            cpu_count = os.cpu_count()
            torch.set_num_threads(cpu_count)
            
            logger_manager.info(f"PyTorch CPU配置完成，使用 {cpu_count} 个线程")
    
    def get_device_info(self) -> Dict[str, Any]:
        """获取设备信息"""
        info = {
            'current_accelerator': self.current_accelerator.value,
            'available_accelerators': [acc.value for acc in self.available_accelerators],
            'device_details': {}
        }
        
        try:
            if self.current_accelerator == AcceleratorType.GPU_CUDA:
                if TF_AVAILABLE:
                    gpus = tf.config.list_physical_devices('GPU')
                    info['device_details']['tensorflow_gpus'] = len(gpus)
                
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    info['device_details']['pytorch_cuda'] = {
                        'device_count': torch.cuda.device_count(),
                        'current_device': torch.cuda.current_device(),
                        'device_name': torch.cuda.get_device_name(torch.cuda.current_device()),
                        'memory_allocated': torch.cuda.memory_allocated(),
                        'memory_reserved': torch.cuda.memory_reserved()
                    }
            
            elif self.current_accelerator == AcceleratorType.CPU:
                info['device_details']['cpu_count'] = os.cpu_count()
                info['device_details']['platform'] = platform.platform()
            
        except Exception as e:
            logger_manager.warning(f"获取设备信息失败: {e}")
        
        return info
    
    def optimize_for_inference(self) -> bool:
        """优化推理性能"""
        try:
            if TF_AVAILABLE:
                # 启用混合精度
                if self.current_accelerator in [AcceleratorType.GPU_CUDA, AcceleratorType.TPU]:
                    policy = tf.keras.mixed_precision.Policy('mixed_float16')
                    tf.keras.mixed_precision.set_global_policy(policy)
                    logger_manager.info("TensorFlow混合精度已启用")
                
                # 启用XLA编译
                tf.config.optimizer.set_jit(True)
                logger_manager.info("TensorFlow XLA编译已启用")
            
            if TORCH_AVAILABLE:
                # 启用JIT编译
                torch.jit.set_fusion_strategy([('STATIC', 20), ('DYNAMIC', 20)])
                logger_manager.info("PyTorch JIT优化已启用")
            
            return True
            
        except Exception as e:
            logger_manager.error(f"推理优化失败: {e}")
            return False
    
    def optimize_for_training(self) -> bool:
        """优化训练性能"""
        try:
            if TF_AVAILABLE:
                # 启用自动混合精度
                if self.current_accelerator in [AcceleratorType.GPU_CUDA, AcceleratorType.TPU]:
                    policy = tf.keras.mixed_precision.Policy('mixed_float16')
                    tf.keras.mixed_precision.set_global_policy(policy)
                    logger_manager.info("TensorFlow训练混合精度已启用")
            
            if TORCH_AVAILABLE:
                # 启用自动混合精度
                if self.current_accelerator == AcceleratorType.GPU_CUDA:
                    torch.backends.cudnn.benchmark = True
                    logger_manager.info("PyTorch CUDNN基准测试已启用")
            
            return True
            
        except Exception as e:
            logger_manager.error(f"训练优化失败: {e}")
            return False
    
    def create_distributed_strategy(self) -> Optional[Any]:
        """创建分布式训练策略"""
        try:
            if not TF_AVAILABLE:
                return None
            
            if self.current_accelerator == AcceleratorType.TPU:
                resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
                return tf.distribute.TPUStrategy(resolver)
            
            elif self.current_accelerator == AcceleratorType.GPU_CUDA:
                gpus = tf.config.list_physical_devices('GPU')
                if len(gpus) > 1:
                    return tf.distribute.MirroredStrategy()
                else:
                    return tf.distribute.OneDeviceStrategy("/gpu:0")
            
            else:
                return tf.distribute.OneDeviceStrategy("/cpu:0")
            
        except Exception as e:
            logger_manager.error(f"创建分布式策略失败: {e}")
            return None
    
    def get_optimal_batch_size(self, model_size: int, data_size: int) -> int:
        """获取最优批大小"""
        try:
            base_batch_size = 32
            
            if self.current_accelerator == AcceleratorType.TPU:
                # TPU通常需要较大的批大小
                return min(128, data_size)
            
            elif self.current_accelerator == AcceleratorType.GPU_CUDA:
                # 根据GPU内存调整批大小
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory
                    # 简单的内存估算
                    estimated_batch_size = min(gpu_memory // (model_size * 4), data_size)
                    return max(base_batch_size, estimated_batch_size)
            
            elif self.current_accelerator == AcceleratorType.CPU:
                # CPU使用较小的批大小
                cpu_count = os.cpu_count()
                return min(base_batch_size, max(1, cpu_count * 2))
            
            return base_batch_size
            
        except Exception as e:
            logger_manager.error(f"计算最优批大小失败: {e}")
            return 32
    
    def cleanup(self):
        """清理资源"""
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger_manager.info("PyTorch GPU缓存已清理")
            
            if TF_AVAILABLE:
                tf.keras.backend.clear_session()
                logger_manager.info("TensorFlow会话已清理")
            
        except Exception as e:
            logger_manager.warning(f"清理资源时出错: {e}")
    
    def benchmark_performance(self, operation_func, *args, **kwargs) -> Dict[str, float]:
        """性能基准测试"""
        import time
        
        results = {
            'execution_time': 0.0,
            'memory_usage': 0.0,
            'throughput': 0.0
        }
        
        try:
            # 预热
            for _ in range(3):
                operation_func(*args, **kwargs)
            
            # 基准测试
            start_time = time.time()
            
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.synchronize()
            
            result = operation_func(*args, **kwargs)
            
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            
            results['execution_time'] = end_time - start_time
            
            # 内存使用情况
            if TORCH_AVAILABLE and torch.cuda.is_available():
                results['memory_usage'] = torch.cuda.memory_allocated() / 1024**2  # MB
            
            logger_manager.info(f"性能基准测试完成: {results}")
            
        except Exception as e:
            logger_manager.error(f"性能基准测试失败: {e}")
        
        return results


# 全局硬件加速器实例
hardware_accelerator = HardwareAccelerator()
