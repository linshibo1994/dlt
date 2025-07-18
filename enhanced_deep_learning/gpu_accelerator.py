#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPU加速器
提供GPU加速支持和内存管理功能
"""

import os
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union
import platform
import psutil

from core_modules import logger_manager


class GPUAccelerator:
    """GPU加速器"""
    
    def __init__(self):
        """初始化GPU加速器"""
        self.gpu_available = False
        self.gpu_devices = []
        self.gpu_memory = {}
        self.tf_gpu_available = False
        self.torch_gpu_available = False
        
        # 检查GPU可用性
        self._check_gpu_availability()
        
        if self.gpu_available:
            logger_manager.info(f"GPU加速器初始化完成，可用GPU: {len(self.gpu_devices)}")
        else:
            logger_manager.info("GPU加速器初始化完成，没有可用GPU")
    
    def _check_gpu_availability(self) -> None:
        """检查GPU可用性"""
        # 检查TensorFlow GPU
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                self.gpu_available = True
                self.tf_gpu_available = True
                self.gpu_devices.extend([f"TensorFlow GPU {i}" for i in range(len(gpus))])
                
                # 获取GPU内存信息
                for i, gpu in enumerate(gpus):
                    try:
                        gpu_memory = tf.config.experimental.get_memory_info(gpu)
                        self.gpu_memory[f"TensorFlow GPU {i}"] = {
                            'total': gpu_memory['total'] / (1024 ** 3),  # GB
                            'used': gpu_memory['used'] / (1024 ** 3)  # GB
                        }
                    except:
                        # 如果无法获取内存信息，使用默认值
                        self.gpu_memory[f"TensorFlow GPU {i}"] = {
                            'total': 0,
                            'used': 0
                        }
                
                logger_manager.info(f"检测到 {len(gpus)} 个 TensorFlow GPU")
        except Exception as e:
            logger_manager.debug(f"TensorFlow GPU检测失败: {e}")
        
        # 检查PyTorch GPU
        try:
            import torch
            if torch.cuda.is_available():
                self.gpu_available = True
                self.torch_gpu_available = True
                
                # 获取GPU数量
                device_count = torch.cuda.device_count()
                self.gpu_devices.extend([f"PyTorch GPU {i}" for i in range(device_count)])
                
                # 获取GPU内存信息
                for i in range(device_count):
                    try:
                        total_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)  # GB
                        reserved_memory = torch.cuda.memory_reserved(i) / (1024 ** 3)  # GB
                        allocated_memory = torch.cuda.memory_allocated(i) / (1024 ** 3)  # GB
                        
                        self.gpu_memory[f"PyTorch GPU {i}"] = {
                            'total': total_memory,
                            'reserved': reserved_memory,
                            'allocated': allocated_memory,
                            'free': total_memory - allocated_memory
                        }
                    except:
                        # 如果无法获取内存信息，使用默认值
                        self.gpu_memory[f"PyTorch GPU {i}"] = {
                            'total': 0,
                            'reserved': 0,
                            'allocated': 0,
                            'free': 0
                        }
                
                logger_manager.info(f"检测到 {device_count} 个 PyTorch GPU")
        except Exception as e:
            logger_manager.debug(f"PyTorch GPU检测失败: {e}")
    
    def is_gpu_available(self) -> bool:
        """
        检查是否有可用的GPU
        
        Returns:
            是否有可用的GPU
        """
        return self.gpu_available
    
    def get_gpu_devices(self) -> List[str]:
        """
        获取可用的GPU设备列表
        
        Returns:
            GPU设备列表
        """
        return self.gpu_devices
    
    def get_gpu_memory_info(self) -> Dict[str, Dict[str, float]]:
        """
        获取GPU内存信息
        
        Returns:
            GPU内存信息字典
        """
        return self.gpu_memory
    
    def enable_tensorflow_gpu(self, memory_limit: Optional[int] = None) -> bool:
        """
        启用TensorFlow GPU
        
        Args:
            memory_limit: 内存限制（MB），如果为None则不限制
            
        Returns:
            是否成功启用
        """
        if not self.tf_gpu_available:
            logger_manager.warning("TensorFlow GPU不可用")
            return False
        
        try:
            import tensorflow as tf
            
            # 获取可用的GPU设备
            gpus = tf.config.list_physical_devices('GPU')
            
            if not gpus:
                logger_manager.warning("没有可用的TensorFlow GPU")
                return False
            
            # 配置GPU内存增长
            for gpu in gpus:
                if memory_limit:
                    # 限制GPU内存使用
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                    )
                else:
                    # 允许GPU内存增长
                    tf.config.experimental.set_memory_growth(gpu, True)
            
            logger_manager.info(f"TensorFlow GPU已启用，内存限制: {memory_limit if memory_limit else '无限制'}")
            return True
        except Exception as e:
            logger_manager.error(f"启用TensorFlow GPU失败: {e}")
            return False
    
    def enable_pytorch_gpu(self, device_id: int = 0) -> bool:
        """
        启用PyTorch GPU
        
        Args:
            device_id: 设备ID
            
        Returns:
            是否成功启用
        """
        if not self.torch_gpu_available:
            logger_manager.warning("PyTorch GPU不可用")
            return False
        
        try:
            import torch
            
            # 检查设备ID是否有效
            if device_id >= torch.cuda.device_count():
                logger_manager.warning(f"无效的设备ID: {device_id}，最大ID: {torch.cuda.device_count() - 1}")
                return False
            
            # 设置默认设备
            torch.cuda.set_device(device_id)
            
            # 清空缓存
            torch.cuda.empty_cache()
            
            logger_manager.info(f"PyTorch GPU已启用，设备ID: {device_id}")
            return True
        except Exception as e:
            logger_manager.error(f"启用PyTorch GPU失败: {e}")
            return False
    
    def migrate_model_to_gpu(self, model: Any, framework: str = 'tensorflow') -> Any:
        """
        将模型迁移到GPU
        
        Args:
            model: 模型对象
            framework: 框架名称，支持'tensorflow', 'pytorch'
            
        Returns:
            迁移后的模型
        """
        if not self.gpu_available:
            logger_manager.warning("GPU不可用，无法迁移模型")
            return model
        
        if framework == 'tensorflow':
            if not self.tf_gpu_available:
                logger_manager.warning("TensorFlow GPU不可用，无法迁移模型")
                return model
            
            try:
                import tensorflow as tf
                
                # 获取可用的GPU设备
                gpus = tf.config.list_physical_devices('GPU')
                
                if not gpus:
                    logger_manager.warning("没有可用的TensorFlow GPU，无法迁移模型")
                    return model
                
                # 将模型迁移到GPU
                with tf.device('/GPU:0'):
                    # 如果是Keras模型，可以直接克隆
                    if hasattr(model, 'clone_model'):
                        gpu_model = tf.keras.models.clone_model(model)
                        gpu_model.set_weights(model.get_weights())
                    else:
                        # 否则直接返回原模型（TensorFlow会自动使用GPU）
                        gpu_model = model
                
                logger_manager.info("模型已迁移到TensorFlow GPU")
                return gpu_model
            except Exception as e:
                logger_manager.error(f"将模型迁移到TensorFlow GPU失败: {e}")
                return model
        
        elif framework == 'pytorch':
            if not self.torch_gpu_available:
                logger_manager.warning("PyTorch GPU不可用，无法迁移模型")
                return model
            
            try:
                import torch
                
                # 检查模型是否已经在GPU上
                if next(model.parameters()).is_cuda:
                    logger_manager.info("模型已经在PyTorch GPU上")
                    return model
                
                # 将模型迁移到GPU
                gpu_model = model.cuda()
                
                logger_manager.info("模型已迁移到PyTorch GPU")
                return gpu_model
            except Exception as e:
                logger_manager.error(f"将模型迁移到PyTorch GPU失败: {e}")
                return model
        
        else:
            logger_manager.warning(f"未知的框架: {framework}")
            return model
    
    def clear_gpu_memory(self, framework: str = 'all') -> bool:
        """
        清理GPU内存
        
        Args:
            framework: 框架名称，支持'tensorflow', 'pytorch', 'all'
            
        Returns:
            是否成功清理
        """
        success = True
        
        if framework in ['tensorflow', 'all']:
            if self.tf_gpu_available:
                try:
                    import tensorflow as tf
                    
                    # 清理TensorFlow GPU内存
                    tf.keras.backend.clear_session()
                    
                    logger_manager.info("TensorFlow GPU内存已清理")
                except Exception as e:
                    logger_manager.error(f"清理TensorFlow GPU内存失败: {e}")
                    success = False
        
        if framework in ['pytorch', 'all']:
            if self.torch_gpu_available:
                try:
                    import torch
                    
                    # 清理PyTorch GPU内存
                    torch.cuda.empty_cache()
                    
                    logger_manager.info("PyTorch GPU内存已清理")
                except Exception as e:
                    logger_manager.error(f"清理PyTorch GPU内存失败: {e}")
                    success = False
        
        return success
    
    def monitor_gpu_usage(self) -> Dict[str, Any]:
        """
        监控GPU使用情况
        
        Returns:
            GPU使用情况字典
        """
        usage_info = {
            'gpu_available': self.gpu_available,
            'gpu_devices': self.gpu_devices,
            'gpu_memory': {},
            'system_memory': {}
        }
        
        # 更新GPU内存信息
        if self.tf_gpu_available:
            try:
                import tensorflow as tf
                
                gpus = tf.config.list_physical_devices('GPU')
                for i, gpu in enumerate(gpus):
                    try:
                        gpu_memory = tf.config.experimental.get_memory_info(gpu)
                        usage_info['gpu_memory'][f"TensorFlow GPU {i}"] = {
                            'total': gpu_memory['total'] / (1024 ** 3),  # GB
                            'used': gpu_memory['used'] / (1024 ** 3)  # GB
                        }
                    except:
                        pass
            except:
                pass
        
        if self.torch_gpu_available:
            try:
                import torch
                
                for i in range(torch.cuda.device_count()):
                    try:
                        total_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)  # GB
                        reserved_memory = torch.cuda.memory_reserved(i) / (1024 ** 3)  # GB
                        allocated_memory = torch.cuda.memory_allocated(i) / (1024 ** 3)  # GB
                        
                        usage_info['gpu_memory'][f"PyTorch GPU {i}"] = {
                            'total': total_memory,
                            'reserved': reserved_memory,
                            'allocated': allocated_memory,
                            'free': total_memory - allocated_memory
                        }
                    except:
                        pass
            except:
                pass
        
        # 获取系统内存信息
        try:
            memory = psutil.virtual_memory()
            usage_info['system_memory'] = {
                'total': memory.total / (1024 ** 3),  # GB
                'available': memory.available / (1024 ** 3),  # GB
                'used': memory.used / (1024 ** 3),  # GB
                'percent': memory.percent
            }
        except:
            pass
        
        return usage_info


class GPUMemoryManager:
    """GPU内存管理器"""
    
    def __init__(self, accelerator: Optional[GPUAccelerator] = None):
        """
        初始化GPU内存管理器
        
        Args:
            accelerator: GPU加速器，如果为None则创建新的
        """
        self.accelerator = accelerator or GPUAccelerator()
        self.memory_limit = None
        self.auto_clear = True
        
        logger_manager.info("GPU内存管理器初始化完成")
    
    def set_memory_limit(self, limit_mb: int) -> bool:
        """
        设置内存限制
        
        Args:
            limit_mb: 内存限制（MB）
            
        Returns:
            是否成功设置
        """
        self.memory_limit = limit_mb
        
        # 应用内存限制
        if self.accelerator.tf_gpu_available:
            success = self.accelerator.enable_tensorflow_gpu(memory_limit=limit_mb)
            if not success:
                logger_manager.warning("设置TensorFlow GPU内存限制失败")
                return False
        
        logger_manager.info(f"GPU内存限制已设置为 {limit_mb} MB")
        return True
    
    def enable_auto_clear(self, enabled: bool = True) -> None:
        """
        启用自动清理
        
        Args:
            enabled: 是否启用
        """
        self.auto_clear = enabled
        logger_manager.info(f"GPU内存自动清理已{'启用' if enabled else '禁用'}")
    
    def clear_if_needed(self, threshold_percent: float = 80.0) -> bool:
        """
        根据需要清理内存
        
        Args:
            threshold_percent: 阈值百分比，超过此值时清理内存
            
        Returns:
            是否已清理
        """
        if not self.auto_clear:
            return False
        
        # 获取GPU使用情况
        usage_info = self.accelerator.monitor_gpu_usage()
        
        # 检查是否需要清理
        need_clear = False
        for device, memory in usage_info.get('gpu_memory', {}).items():
            if 'total' in memory and 'used' in memory and memory['total'] > 0:
                usage_percent = memory['used'] / memory['total'] * 100
                if usage_percent > threshold_percent:
                    need_clear = True
                    break
        
        if need_clear:
            logger_manager.info(f"GPU内存使用率超过 {threshold_percent}%，执行清理")
            return self.accelerator.clear_gpu_memory()
        
        return False
    
    def optimize_batch_size(self, initial_batch_size: int, model_size_mb: int) -> int:
        """
        优化批处理大小
        
        Args:
            initial_batch_size: 初始批处理大小
            model_size_mb: 模型大小（MB）
            
        Returns:
            优化后的批处理大小
        """
        if not self.accelerator.gpu_available:
            logger_manager.warning("GPU不可用，使用初始批处理大小")
            return initial_batch_size
        
        # 获取GPU内存信息
        usage_info = self.accelerator.monitor_gpu_usage()
        
        # 计算可用内存
        available_memory_mb = 0
        for device, memory in usage_info.get('gpu_memory', {}).items():
            if 'total' in memory and 'used' in memory:
                available = (memory['total'] - memory['used']) * 1024  # MB
                if available > available_memory_mb:
                    available_memory_mb = available
        
        if available_memory_mb == 0:
            logger_manager.warning("无法获取GPU可用内存，使用初始批处理大小")
            return initial_batch_size
        
        # 估计每个样本的内存使用量
        estimated_sample_memory = model_size_mb / initial_batch_size * 2  # 乘以2作为安全系数
        
        # 计算最大批处理大小
        max_batch_size = int(available_memory_mb / estimated_sample_memory * 0.8)  # 使用80%的可用内存
        
        # 确保批处理大小是2的幂次方
        optimized_batch_size = 2 ** int(np.log2(max_batch_size))
        
        # 限制在合理范围内
        optimized_batch_size = max(16, min(optimized_batch_size, 1024))
        
        logger_manager.info(f"批处理大小已优化: {initial_batch_size} -> {optimized_batch_size}")
        
        return optimized_batch_size


if __name__ == "__main__":
    # 测试GPU加速器
    print("🚀 测试GPU加速器...")
    
    # 创建GPU加速器
    accelerator = GPUAccelerator()
    
    # 检查GPU可用性
    gpu_available = accelerator.is_gpu_available()
    print(f"GPU可用性: {gpu_available}")
    
    # 获取GPU设备列表
    gpu_devices = accelerator.get_gpu_devices()
    print(f"GPU设备列表: {gpu_devices}")
    
    # 获取GPU内存信息
    gpu_memory = accelerator.get_gpu_memory_info()
    print(f"GPU内存信息: {gpu_memory}")
    
    # 测试GPU内存管理器
    print("\n🚀 测试GPU内存管理器...")
    
    # 创建GPU内存管理器
    memory_manager = GPUMemoryManager(accelerator)
    
    # 设置内存限制
    memory_manager.set_memory_limit(1024)
    
    # 启用自动清理
    memory_manager.enable_auto_clear(True)
    
    # 优化批处理大小
    optimized_batch_size = memory_manager.optimize_batch_size(32, 500)
    print(f"优化后的批处理大小: {optimized_batch_size}")
    
    print("GPU加速器测试完成")