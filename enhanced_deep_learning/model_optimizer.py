#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模型优化器
提供模型量化、中间结果缓存和资源使用监控功能
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional, Union, Callable
import time
import psutil
import functools
import hashlib
import pickle
import json
from pathlib import Path

from core_modules import logger_manager, cache_manager


class ModelOptimizer:
    """模型优化器"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        初始化模型优化器
        
        Args:
            cache_dir: 缓存目录，如果为None则使用默认缓存目录
        """
        self.cache_dir = cache_dir or os.path.join(cache_manager.get_cache_dir(), 'model_optimizer')
        self._ensure_cache_dir()
        
        # 资源监控
        self.resource_usage = {
            'cpu': [],
            'memory': [],
            'disk': [],
            'time': []
        }
        
        logger_manager.info("模型优化器初始化完成")
    
    def _ensure_cache_dir(self) -> None:
        """确保缓存目录存在"""
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 创建子目录
        for subdir in ['quantized_models', 'result_cache']:
            os.makedirs(os.path.join(self.cache_dir, subdir), exist_ok=True)
    
    def quantize_tensorflow_model(self, model: Any, quantization_type: str = 'float16') -> Any:
        """
        量化TensorFlow模型
        
        Args:
            model: TensorFlow模型
            quantization_type: 量化类型，支持'float16', 'int8', 'dynamic'
            
        Returns:
            量化后的模型
        """
        try:
            import tensorflow as tf
            
            # 检查模型类型
            if not isinstance(model, tf.keras.Model):
                logger_manager.warning("不是有效的TensorFlow模型，无法量化")
                return model
            
            # 根据量化类型进行量化
            if quantization_type == 'float16':
                # 使用混合精度策略
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                
                # 克隆模型并转换为float16
                quantized_model = tf.keras.models.clone_model(model)
                quantized_model.set_weights(model.get_weights())
                
                logger_manager.info("TensorFlow模型已量化为float16")
                return quantized_model
            
            elif quantization_type == 'int8':
                # 保存原始模型
                temp_model_path = os.path.join(self.cache_dir, 'temp_model')
                model.save(temp_model_path)
                
                # 使用TFLite转换器进行int8量化
                converter = tf.lite.TFLiteConverter.from_saved_model(temp_model_path)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.int8]
                
                # 转换模型
                tflite_model = converter.convert()
                
                # 保存量化后的模型
                quantized_model_path = os.path.join(self.cache_dir, 'quantized_models', 'model_int8.tflite')
                with open(quantized_model_path, 'wb') as f:
                    f.write(tflite_model)
                
                logger_manager.info(f"TensorFlow模型已量化为int8，保存到 {quantized_model_path}")
                
                # 返回原始模型（因为TFLite模型需要特殊处理）
                return model
            
            elif quantization_type == 'dynamic':
                # 保存原始模型
                temp_model_path = os.path.join(self.cache_dir, 'temp_model')
                model.save(temp_model_path)
                
                # 使用TFLite转换器进行动态范围量化
                converter = tf.lite.TFLiteConverter.from_saved_model(temp_model_path)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                
                # 转换模型
                tflite_model = converter.convert()
                
                # 保存量化后的模型
                quantized_model_path = os.path.join(self.cache_dir, 'quantized_models', 'model_dynamic.tflite')
                with open(quantized_model_path, 'wb') as f:
                    f.write(tflite_model)
                
                logger_manager.info(f"TensorFlow模型已动态量化，保存到 {quantized_model_path}")
                
                # 返回原始模型（因为TFLite模型需要特殊处理）
                return model
            
            else:
                logger_manager.warning(f"未知的量化类型: {quantization_type}")
                return model
        
        except Exception as e:
            logger_manager.error(f"TensorFlow模型量化失败: {e}")
            return model
    
    def quantize_pytorch_model(self, model: Any, quantization_type: str = 'float16') -> Any:
        """
        量化PyTorch模型
        
        Args:
            model: PyTorch模型
            quantization_type: 量化类型，支持'float16', 'int8', 'dynamic'
            
        Returns:
            量化后的模型
        """
        try:
            import torch
            
            # 检查模型类型
            if not isinstance(model, torch.nn.Module):
                logger_manager.warning("不是有效的PyTorch模型，无法量化")
                return model
            
            # 根据量化类型进行量化
            if quantization_type == 'float16':
                # 转换为float16
                model_fp16 = model.half()
                
                logger_manager.info("PyTorch模型已量化为float16")
                return model_fp16
            
            elif quantization_type == 'int8':
                # 静态量化
                model_int8 = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
                )
                
                logger_manager.info("PyTorch模型已量化为int8")
                return model_int8
            
            elif quantization_type == 'dynamic':
                # 动态量化
                model_dynamic = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
                
                logger_manager.info("PyTorch模型已动态量化")
                return model_dynamic
            
            else:
                logger_manager.warning(f"未知的量化类型: {quantization_type}")
                return model
        
        except Exception as e:
            logger_manager.error(f"PyTorch模型量化失败: {e}")
            return model
    
    def quantize_model(self, model: Any, framework: str = 'tensorflow', 
                     quantization_type: str = 'float16') -> Any:
        """
        量化模型
        
        Args:
            model: 模型对象
            framework: 框架名称，支持'tensorflow', 'pytorch'
            quantization_type: 量化类型
            
        Returns:
            量化后的模型
        """
        if framework == 'tensorflow':
            return self.quantize_tensorflow_model(model, quantization_type)
        elif framework == 'pytorch':
            return self.quantize_pytorch_model(model, quantization_type)
        else:
            logger_manager.warning(f"未知的框架: {framework}")
            return model
    
    def cache_result(self, func: Callable) -> Callable:
        """
        缓存函数结果的装饰器
        
        Args:
            func: 要缓存结果的函数
            
        Returns:
            装饰后的函数
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = self._generate_cache_key(func.__name__, args, kwargs)
            cache_file = os.path.join(self.cache_dir, 'result_cache', f"{cache_key}.pkl")
            
            # 检查缓存是否存在
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        result = pickle.load(f)
                    logger_manager.debug(f"从缓存加载结果: {func.__name__}")
                    return result
                except Exception as e:
                    logger_manager.warning(f"加载缓存结果失败: {e}")
            
            # 执行函数
            result = func(*args, **kwargs)
            
            # 缓存结果
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
                logger_manager.debug(f"缓存函数结果: {func.__name__}")
            except Exception as e:
                logger_manager.warning(f"缓存结果失败: {e}")
            
            return result
        
        return wrapper
    
    def _generate_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """
        生成缓存键
        
        Args:
            func_name: 函数名称
            args: 位置参数
            kwargs: 关键字参数
            
        Returns:
            缓存键
        """
        # 创建可哈希的参数表示
        def make_hashable(obj):
            if isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            elif isinstance(obj, (list, tuple)):
                return tuple(make_hashable(x) for x in obj)
            elif isinstance(obj, dict):
                return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
            elif isinstance(obj, np.ndarray):
                return obj.tobytes()
            else:
                return str(obj)
        
        # 组合函数名和参数
        key_parts = [func_name]
        key_parts.extend([make_hashable(arg) for arg in args])
        key_parts.extend([f"{k}={make_hashable(v)}" for k, v in sorted(kwargs.items())])
        
        # 生成哈希
        key_str = str(key_parts).encode('utf-8')
        return hashlib.md5(key_str).hexdigest()
    
    def clear_result_cache(self) -> None:
        """清除结果缓存"""
        cache_dir = os.path.join(self.cache_dir, 'result_cache')
        
        # 删除缓存文件
        for file_path in Path(cache_dir).glob('*.pkl'):
            try:
                os.remove(file_path)
            except Exception as e:
                logger_manager.warning(f"删除缓存文件失败: {e}")
        
        logger_manager.info("结果缓存已清除")
    
    def start_resource_monitoring(self) -> None:
        """开始资源监控"""
        # 清空历史记录
        self.resource_usage = {
            'cpu': [],
            'memory': [],
            'disk': [],
            'time': []
        }
        
        logger_manager.info("资源监控已开始")
    
    def record_resource_usage(self) -> Dict[str, float]:
        """
        记录资源使用情况
        
        Returns:
            当前资源使用情况
        """
        # 获取CPU使用率
        cpu_percent = psutil.cpu_percent()
        
        # 获取内存使用情况
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # 获取磁盘使用情况
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        
        # 记录时间
        current_time = time.time()
        
        # 记录资源使用情况
        self.resource_usage['cpu'].append(cpu_percent)
        self.resource_usage['memory'].append(memory_percent)
        self.resource_usage['disk'].append(disk_percent)
        self.resource_usage['time'].append(current_time)
        
        # 返回当前资源使用情况
        current_usage = {
            'cpu': cpu_percent,
            'memory': memory_percent,
            'disk': disk_percent,
            'time': current_time
        }
        
        return current_usage
    
    def get_resource_usage_summary(self) -> Dict[str, Dict[str, float]]:
        """
        获取资源使用情况摘要
        
        Returns:
            资源使用情况摘要
        """
        summary = {}
        
        for resource, values in self.resource_usage.items():
            if resource != 'time' and values:
                summary[resource] = {
                    'min': min(values),
                    'max': max(values),
                    'avg': sum(values) / len(values),
                    'current': values[-1] if values else 0
                }
        
        return summary
    
    def save_resource_usage_report(self, file_path: Optional[str] = None) -> str:
        """
        保存资源使用情况报告
        
        Args:
            file_path: 报告文件路径，如果为None则使用默认路径
            
        Returns:
            报告文件路径
        """
        # 设置默认文件路径
        if file_path is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            file_path = os.path.join(self.cache_dir, f'resource_usage_{timestamp}.json')
        
        # 获取资源使用情况摘要
        summary = self.get_resource_usage_summary()
        
        # 添加详细数据
        report = {
            'summary': summary,
            'details': {
                'cpu': self.resource_usage['cpu'],
                'memory': self.resource_usage['memory'],
                'disk': self.resource_usage['disk'],
                'time': [t - self.resource_usage['time'][0] for t in self.resource_usage['time']]
            }
        }
        
        # 保存报告
        try:
            with open(file_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger_manager.info(f"资源使用情况报告已保存到 {file_path}")
        except Exception as e:
            logger_manager.error(f"保存资源使用情况报告失败: {e}")
        
        return file_path
    
    def optimize_model_inference(self, model: Any, framework: str = 'tensorflow') -> Any:
        """
        优化模型推理
        
        Args:
            model: 模型对象
            framework: 框架名称，支持'tensorflow', 'pytorch'
            
        Returns:
            优化后的模型
        """
        if framework == 'tensorflow':
            try:
                import tensorflow as tf
                
                # 检查模型类型
                if not isinstance(model, tf.keras.Model):
                    logger_manager.warning("不是有效的TensorFlow模型，无法优化")
                    return model
                
                # 转换为SavedModel格式
                temp_model_path = os.path.join(self.cache_dir, 'temp_optimized_model')
                model.save(temp_model_path)
                
                # 加载并优化模型
                optimized_model = tf.saved_model.load(temp_model_path)
                
                # 创建优化函数
                @tf.function
                def optimized_predict(x):
                    return optimized_model(x)
                
                # 返回优化后的模型和预测函数
                logger_manager.info("TensorFlow模型推理已优化")
                return optimized_model, optimized_predict
            
            except Exception as e:
                logger_manager.error(f"优化TensorFlow模型推理失败: {e}")
                return model, None
        
        elif framework == 'pytorch':
            try:
                import torch
                
                # 检查模型类型
                if not isinstance(model, torch.nn.Module):
                    logger_manager.warning("不是有效的PyTorch模型，无法优化")
                    return model
                
                # 设置为评估模式
                model.eval()
                
                # 使用JIT编译
                example_input = torch.randn(1, *model.input_shape[1:])
                optimized_model = torch.jit.trace(model, example_input)
                
                logger_manager.info("PyTorch模型推理已优化")
                return optimized_model
            
            except Exception as e:
                logger_manager.error(f"优化PyTorch模型推理失败: {e}")
                return model
        
        else:
            logger_manager.warning(f"未知的框架: {framework}")
            return model


class CachedModel:
    """缓存模型结果"""
    
    def __init__(self, model: Any, optimizer: ModelOptimizer, cache_size: int = 100):
        """
        初始化缓存模型
        
        Args:
            model: 模型对象
            optimizer: 模型优化器
            cache_size: 缓存大小
        """
        self.model = model
        self.optimizer = optimizer
        self.cache_size = cache_size
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger_manager.info(f"缓存模型初始化完成，缓存大小: {cache_size}")
    
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        预测
        
        Args:
            inputs: 输入数据
            
        Returns:
            预测结果
        """
        # 生成缓存键
        cache_key = self._generate_cache_key(inputs)
        
        # 检查缓存
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]
        
        # 执行预测
        result = self.model.predict(inputs)
        
        # 更新缓存
        self.cache_misses += 1
        self._update_cache(cache_key, result)
        
        return result
    
    def _generate_cache_key(self, inputs: np.ndarray) -> str:
        """
        生成缓存键
        
        Args:
            inputs: 输入数据
            
        Returns:
            缓存键
        """
        # 使用输入数据的哈希作为缓存键
        return hashlib.md5(inputs.tobytes()).hexdigest()
    
    def _update_cache(self, key: str, value: np.ndarray) -> None:
        """
        更新缓存
        
        Args:
            key: 缓存键
            value: 缓存值
        """
        # 如果缓存已满，删除最早的项
        if len(self.cache) >= self.cache_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        # 添加新项
        self.cache[key] = value
    
    def get_cache_stats(self) -> Dict[str, int]:
        """
        获取缓存统计信息
        
        Returns:
            缓存统计信息
        """
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_size': len(self.cache),
            'max_cache_size': self.cache_size,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate
        }
    
    def clear_cache(self) -> None:
        """清除缓存"""
        self.cache = {}
        logger_manager.info("模型缓存已清除")


if __name__ == "__main__":
    # 测试模型优化器
    print("🚀 测试模型优化器...")
    
    # 创建模型优化器
    optimizer = ModelOptimizer()
    
    # 测试资源监控
    print("\n测试资源监控...")
    optimizer.start_resource_monitoring()
    
    # 模拟一些操作
    for _ in range(5):
        time.sleep(0.5)
        usage = optimizer.record_resource_usage()
        print(f"CPU: {usage['cpu']}%, 内存: {usage['memory']}%, 磁盘: {usage['disk']}%")
    
    # 获取资源使用情况摘要
    summary = optimizer.get_resource_usage_summary()
    print(f"\n资源使用情况摘要: {summary}")
    
    # 保存资源使用情况报告
    report_path = optimizer.save_resource_usage_report()
    print(f"资源使用情况报告已保存到: {report_path}")
    
    # 测试结果缓存装饰器
    print("\n测试结果缓存装饰器...")
    
    @optimizer.cache_result
    def expensive_calculation(x):
        print("执行昂贵计算...")
        time.sleep(1)
        return x * 2
    
    # 第一次调用（应该执行计算）
    result1 = expensive_calculation(10)
    print(f"结果1: {result1}")
    
    # 第二次调用（应该从缓存加载）
    result2 = expensive_calculation(10)
    print(f"结果2: {result2}")
    
    # 不同参数的调用（应该执行计算）
    result3 = expensive_calculation(20)
    print(f"结果3: {result3}")
    
    # 清除结果缓存
    optimizer.clear_result_cache()
    
    print("模型优化器测试完成")