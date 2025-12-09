#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
智能加速选择器模块
Intelligent Acceleration Selector Module

基于任务类型、数据量和硬件性能智能选择最优加速方式。
"""

import os
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from core_modules import logger_manager
from .enhanced_hardware_accelerator import EnhancedHardwareAccelerator, AccelerationType, AccelerationConfig


class TaskType(Enum):
    """任务类型枚举"""
    DEEP_LEARNING_TRAINING = "dl_training"
    DEEP_LEARNING_INFERENCE = "dl_inference"
    TRADITIONAL_ML = "traditional_ml"
    DATA_PROCESSING = "data_processing"
    CLUSTERING = "clustering"
    MARKOV_CHAIN = "markov_chain"
    BAYESIAN_INFERENCE = "bayesian_inference"


@dataclass
class TaskProfile:
    """任务配置文件"""
    task_type: TaskType
    data_size: int  # 数据量（样本数）
    feature_dim: int  # 特征维度
    complexity_score: float  # 复杂度评分 (0-100)
    memory_intensive: bool = False
    cpu_intensive: bool = False
    gpu_suitable: bool = True


@dataclass
class AccelerationRecommendation:
    """加速推荐结果"""
    recommended_type: AccelerationType
    config: AccelerationConfig
    estimated_speedup: float
    confidence: float
    reasoning: str


class AccelerationSelector:
    """智能加速选择器"""
    
    def __init__(self):
        """初始化加速选择器"""
        self.hardware_accelerator = EnhancedHardwareAccelerator()
        self.benchmark_result = None
        
        # 任务类型的默认配置
        self.task_configs = {
            TaskType.DEEP_LEARNING_TRAINING: {
                'gpu_preference': 0.8,
                'memory_requirement': 'high',
                'cpu_efficiency': 0.3
            },
            TaskType.DEEP_LEARNING_INFERENCE: {
                'gpu_preference': 0.6,
                'memory_requirement': 'medium',
                'cpu_efficiency': 0.5
            },
            TaskType.TRADITIONAL_ML: {
                'gpu_preference': 0.2,
                'memory_requirement': 'medium',
                'cpu_efficiency': 0.8
            },
            TaskType.DATA_PROCESSING: {
                'gpu_preference': 0.1,
                'memory_requirement': 'low',
                'cpu_efficiency': 0.9
            },
            TaskType.CLUSTERING: {
                'gpu_preference': 0.3,
                'memory_requirement': 'medium',
                'cpu_efficiency': 0.7
            },
            TaskType.MARKOV_CHAIN: {
                'gpu_preference': 0.2,
                'memory_requirement': 'low',
                'cpu_efficiency': 0.8
            },
            TaskType.BAYESIAN_INFERENCE: {
                'gpu_preference': 0.4,
                'memory_requirement': 'medium',
                'cpu_efficiency': 0.6
            }
        }
        
        logger_manager.info("智能加速选择器初始化完成")
    
    def select_optimal_acceleration(self, task_profile: TaskProfile) -> AccelerationRecommendation:
        """
        选择最优加速方式
        
        Args:
            task_profile: 任务配置文件
            
        Returns:
            加速推荐结果
        """
        logger_manager.info(f"为任务类型 {task_profile.task_type.value} 选择最优加速方式")
        
        # 获取硬件基准测试结果
        if self.benchmark_result is None:
            self.benchmark_result = self.hardware_accelerator.benchmark_hardware()
        
        # 计算各种加速方式的得分
        scores = self._calculate_acceleration_scores(task_profile)
        
        # 选择得分最高的加速方式
        best_acceleration = max(scores.keys(), key=lambda k: scores[k]['score'])
        best_score = scores[best_acceleration]
        
        # 生成配置
        config = self._generate_acceleration_config(best_acceleration, task_profile)
        
        # 估算加速比
        estimated_speedup = self._estimate_speedup(best_acceleration, task_profile)
        
        # 计算置信度
        confidence = self._calculate_confidence(best_score, scores)
        
        # 生成推理说明
        reasoning = self._generate_reasoning(best_acceleration, task_profile, best_score)
        
        recommendation = AccelerationRecommendation(
            recommended_type=best_acceleration,
            config=config,
            estimated_speedup=estimated_speedup,
            confidence=confidence,
            reasoning=reasoning
        )
        
        logger_manager.info(f"推荐加速方式: {best_acceleration.value}, 预期加速比: {estimated_speedup:.2f}x, 置信度: {confidence:.2f}")
        
        return recommendation
    
    def _calculate_acceleration_scores(self, task_profile: TaskProfile) -> Dict[AccelerationType, Dict[str, float]]:
        """计算各种加速方式的得分"""
        scores = {}
        
        # 获取任务配置
        task_config = self.task_configs.get(task_profile.task_type, {})
        gpu_preference = task_config.get('gpu_preference', 0.5)
        cpu_efficiency = task_config.get('cpu_efficiency', 0.5)
        
        # CPU单线程得分
        cpu_single_score = self._calculate_cpu_single_score(task_profile, cpu_efficiency)
        scores[AccelerationType.CPU_SINGLE] = {
            'score': cpu_single_score,
            'factors': {'cpu_performance': self.benchmark_result.cpu_score, 'efficiency': cpu_efficiency}
        }
        
        # CPU多线程得分
        cpu_multi_score = self._calculate_cpu_multi_score(task_profile, cpu_efficiency)
        scores[AccelerationType.CPU_MULTI] = {
            'score': cpu_multi_score,
            'factors': {'cpu_performance': self.benchmark_result.cpu_score, 'parallelism': True}
        }
        
        # GPU CUDA得分
        gpu_cuda_score = self._calculate_gpu_cuda_score(task_profile, gpu_preference)
        scores[AccelerationType.GPU_CUDA] = {
            'score': gpu_cuda_score,
            'factors': {'gpu_performance': self.benchmark_result.gpu_score, 'gpu_preference': gpu_preference}
        }
        
        return scores
    
    def _calculate_cpu_single_score(self, task_profile: TaskProfile, cpu_efficiency: float) -> float:
        """计算CPU单线程得分"""
        base_score = self.benchmark_result.cpu_score * cpu_efficiency
        
        # 数据量调整
        if task_profile.data_size < 1000:
            base_score *= 1.2  # 小数据量适合单线程
        elif task_profile.data_size > 10000:
            base_score *= 0.6  # 大数据量不适合单线程
        
        # 复杂度调整
        if task_profile.complexity_score < 30:
            base_score *= 1.1  # 简单任务适合单线程
        
        return max(0, min(100, base_score))
    
    def _calculate_cpu_multi_score(self, task_profile: TaskProfile, cpu_efficiency: float) -> float:
        """计算CPU多线程得分"""
        cpu_count = self.hardware_accelerator.platform_info.hardware_info.cpu_count
        base_score = self.benchmark_result.cpu_score * cpu_efficiency
        
        # 多核加成
        parallelism_bonus = min(2.0, math.log2(cpu_count))
        base_score *= parallelism_bonus
        
        # 数据量调整
        if task_profile.data_size > 5000:
            base_score *= 1.3  # 大数据量适合多线程
        
        # 任务类型调整
        if task_profile.task_type in [TaskType.CLUSTERING, TaskType.TRADITIONAL_ML]:
            base_score *= 1.2  # 这些任务适合CPU并行
        
        return max(0, min(100, base_score))
    
    def _calculate_gpu_cuda_score(self, task_profile: TaskProfile, gpu_preference: float) -> float:
        """计算GPU CUDA得分"""
        if self.benchmark_result.gpu_score == 0:
            return 0  # 没有GPU
        
        base_score = self.benchmark_result.gpu_score * gpu_preference
        
        # 数据量调整
        if task_profile.data_size > 10000:
            base_score *= 1.4  # 大数据量适合GPU
        elif task_profile.data_size < 1000:
            base_score *= 0.7  # 小数据量GPU优势不明显
        
        # 复杂度调整
        if task_profile.complexity_score > 70:
            base_score *= 1.3  # 复杂任务适合GPU
        
        # 特征维度调整
        if task_profile.feature_dim > 100:
            base_score *= 1.2  # 高维特征适合GPU
        
        # 任务类型调整
        if task_profile.task_type in [TaskType.DEEP_LEARNING_TRAINING, TaskType.DEEP_LEARNING_INFERENCE]:
            base_score *= 1.5  # 深度学习任务非常适合GPU
        
        return max(0, min(100, base_score))
    
    def _generate_acceleration_config(self, acceleration_type: AccelerationType, task_profile: TaskProfile) -> AccelerationConfig:
        """生成加速配置"""
        cpu_count = self.hardware_accelerator.platform_info.hardware_info.cpu_count
        
        if acceleration_type == AccelerationType.CPU_SINGLE:
            return AccelerationConfig(
                acceleration_type=acceleration_type,
                cpu_threads=1,
                fallback_enabled=True
            )
        elif acceleration_type == AccelerationType.CPU_MULTI:
            # 根据任务类型调整线程数
            if task_profile.task_type == TaskType.CLUSTERING:
                threads = max(2, cpu_count - 1)
            elif task_profile.task_type == TaskType.DATA_PROCESSING:
                threads = min(cpu_count * 2, 16)
            else:
                threads = max(2, cpu_count // 2)
            
            return AccelerationConfig(
                acceleration_type=acceleration_type,
                cpu_threads=threads,
                fallback_enabled=True
            )
        elif acceleration_type == AccelerationType.GPU_CUDA:
            # GPU配置
            memory_limit = None
            if task_profile.memory_intensive:
                # 为内存密集型任务预留更多GPU内存
                memory_limit = 0.8  # 使用80%的GPU内存
            
            mixed_precision = (
                task_profile.task_type in [TaskType.DEEP_LEARNING_TRAINING, TaskType.DEEP_LEARNING_INFERENCE] and
                self.benchmark_result.gpu_score > 70
            )
            
            return AccelerationConfig(
                acceleration_type=acceleration_type,
                gpu_device_id=0,
                gpu_memory_limit=memory_limit,
                mixed_precision=mixed_precision,
                fallback_enabled=True
            )
        else:
            # 默认配置
            return AccelerationConfig(
                acceleration_type=AccelerationType.CPU_SINGLE,
                fallback_enabled=True
            )
    
    def _estimate_speedup(self, acceleration_type: AccelerationType, task_profile: TaskProfile) -> float:
        """估算加速比"""
        if acceleration_type == AccelerationType.CPU_SINGLE:
            return 1.0  # 基准
        elif acceleration_type == AccelerationType.CPU_MULTI:
            cpu_count = self.hardware_accelerator.platform_info.hardware_info.cpu_count
            # 考虑并行效率损失
            efficiency = 0.7 if task_profile.task_type == TaskType.CLUSTERING else 0.6
            return min(cpu_count * efficiency, cpu_count * 0.8)
        elif acceleration_type == AccelerationType.GPU_CUDA:
            if self.benchmark_result.gpu_score > 80:
                return 5.0 if task_profile.task_type in [TaskType.DEEP_LEARNING_TRAINING] else 3.0
            elif self.benchmark_result.gpu_score > 60:
                return 3.0 if task_profile.task_type in [TaskType.DEEP_LEARNING_TRAINING] else 2.0
            else:
                return 2.0
        else:
            return 1.0
    
    def _calculate_confidence(self, best_score: Dict[str, float], all_scores: Dict[AccelerationType, Dict[str, float]]) -> float:
        """计算置信度"""
        scores = [score['score'] for score in all_scores.values()]
        max_score = max(scores)
        second_max = sorted(scores, reverse=True)[1] if len(scores) > 1 else 0
        
        # 基于得分差距计算置信度
        if max_score > 0:
            confidence = min(0.95, (max_score - second_max) / max_score)
        else:
            confidence = 0.5
        
        return max(0.1, confidence)
    
    def _generate_reasoning(self, acceleration_type: AccelerationType, task_profile: TaskProfile, score_info: Dict[str, float]) -> str:
        """生成推理说明"""
        reasons = []
        
        if acceleration_type == AccelerationType.GPU_CUDA:
            reasons.append(f"GPU性能得分: {self.benchmark_result.gpu_score:.1f}")
            if task_profile.task_type in [TaskType.DEEP_LEARNING_TRAINING, TaskType.DEEP_LEARNING_INFERENCE]:
                reasons.append("深度学习任务适合GPU加速")
            if task_profile.data_size > 10000:
                reasons.append("大数据量适合GPU并行处理")
        elif acceleration_type == AccelerationType.CPU_MULTI:
            cpu_count = self.hardware_accelerator.platform_info.hardware_info.cpu_count
            reasons.append(f"CPU核心数: {cpu_count}")
            reasons.append(f"CPU性能得分: {self.benchmark_result.cpu_score:.1f}")
            if task_profile.task_type in [TaskType.CLUSTERING, TaskType.TRADITIONAL_ML]:
                reasons.append("传统机器学习任务适合CPU多线程")
        else:
            reasons.append(f"CPU性能得分: {self.benchmark_result.cpu_score:.1f}")
            if task_profile.data_size < 1000:
                reasons.append("小数据量适合单线程处理")
        
        return "; ".join(reasons)
