#!/usr/bin/env python3
"""
元学习优化器
实现元学习算法，自动优化模型参数和架构
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Callable
import json
import os
from datetime import datetime
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score, precision_score, recall_score
import copy

from core_modules import logger_manager, cache_manager
from interfaces import OptimizerInterface


class MetaLearningOptimizer(OptimizerInterface):
    """元学习优化器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化元学习优化器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        
        # 元学习参数
        self.learning_rate = self.config.get('learning_rate', 0.01)
        self.adaptation_steps = self.config.get('adaptation_steps', 5)
        self.meta_batch_size = self.config.get('meta_batch_size', 4)
        self.max_iterations = self.config.get('max_iterations', 100)
        
        # 优化历史
        self.optimization_history = []
        self.best_parameters = {}
        self.performance_history = []
        
        # 任务池
        self.task_pool = []
        
        logger_manager.info("元学习优化器初始化完成")
    
    def optimize(self, objective_function: Callable, parameter_space: Dict[str, Any], 
                data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        执行元学习优化
        
        Args:
            objective_function: 目标函数
            parameter_space: 参数空间
            data: 训练数据
            
        Returns:
            优化结果
        """
        try:
            logger_manager.info("开始元学习优化")
            
            # 初始化
            self.best_parameters = {}
            best_score = float('-inf')
            
            # 生成任务
            tasks = self._generate_tasks(data)
            
            # 元学习循环
            for iteration in range(self.max_iterations):
                logger_manager.info(f"元学习迭代 {iteration + 1}/{self.max_iterations}")
                
                # 采样元批次
                meta_batch = self._sample_meta_batch(tasks)
                
                # 内循环：快速适应
                adapted_parameters = []
                for task in meta_batch:
                    params = self._fast_adaptation(objective_function, parameter_space, task)
                    adapted_parameters.append(params)
                
                # 外循环：元更新
                meta_gradient = self._compute_meta_gradient(
                    objective_function, adapted_parameters, meta_batch
                )
                
                # 更新元参数
                parameter_space = self._update_meta_parameters(parameter_space, meta_gradient)
                
                # 评估当前参数
                current_score = self._evaluate_parameters(objective_function, parameter_space, data)
                
                # 更新最佳参数
                if current_score > best_score:
                    best_score = current_score
                    self.best_parameters = copy.deepcopy(parameter_space)
                
                # 记录历史
                self.performance_history.append({
                    'iteration': iteration,
                    'score': current_score,
                    'best_score': best_score
                })
                
                # 早停检查
                if self._should_early_stop():
                    logger_manager.info(f"早停于第 {iteration + 1} 次迭代")
                    break
            
            result = {
                'best_parameters': self.best_parameters,
                'best_score': best_score,
                'iterations': len(self.performance_history),
                'optimization_history': self.optimization_history
            }
            
            logger_manager.info(f"元学习优化完成，最佳分数: {best_score:.4f}")
            
            return result
            
        except Exception as e:
            logger_manager.error(f"元学习优化失败: {e}")
            return {}
    
    def _generate_tasks(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """生成学习任务"""
        try:
            tasks = []
            
            if data is None or data.empty:
                return tasks
            
            # 基于时间窗口生成任务
            window_sizes = [50, 100, 200, 300]
            
            for window_size in window_sizes:
                if len(data) >= window_size:
                    # 滑动窗口任务
                    for start in range(0, len(data) - window_size, window_size // 2):
                        end = start + window_size
                        task_data = data.iloc[start:end]
                        
                        tasks.append({
                            'type': 'time_window',
                            'window_size': window_size,
                            'start': start,
                            'end': end,
                            'data': task_data
                        })
            
            # 基于数据特征生成任务
            feature_tasks = self._generate_feature_tasks(data)
            tasks.extend(feature_tasks)
            
            logger_manager.info(f"生成了 {len(tasks)} 个学习任务")
            
            return tasks
            
        except Exception as e:
            logger_manager.error(f"生成任务失败: {e}")
            return []
    
    def _generate_feature_tasks(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """基于特征生成任务"""
        tasks = []
        
        try:
            # 基于和值范围的任务
            if 'front_balls' in data.columns:
                sums = []
                for _, row in data.iterrows():
                    front_balls = [int(x) for x in str(row['front_balls']).split(',')]
                    sums.append(sum(front_balls))
                
                # 按和值分组
                sum_ranges = [(60, 90), (90, 120), (120, 150), (150, 180)]
                
                for min_sum, max_sum in sum_ranges:
                    mask = [(s >= min_sum and s <= max_sum) for s in sums]
                    if sum(mask) > 20:  # 至少20个样本
                        filtered_data = data[mask]
                        tasks.append({
                            'type': 'sum_range',
                            'min_sum': min_sum,
                            'max_sum': max_sum,
                            'data': filtered_data
                        })
            
            # 基于奇偶性的任务
            odd_even_tasks = self._generate_odd_even_tasks(data)
            tasks.extend(odd_even_tasks)
            
        except Exception as e:
            logger_manager.error(f"生成特征任务失败: {e}")
        
        return tasks
    
    def _generate_odd_even_tasks(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """生成奇偶性任务"""
        tasks = []
        
        try:
            if 'front_balls' in data.columns:
                # 按奇偶比例分组
                odd_even_ratios = []
                
                for _, row in data.iterrows():
                    front_balls = [int(x) for x in str(row['front_balls']).split(',')]
                    odd_count = sum(1 for x in front_balls if x % 2 == 1)
                    odd_even_ratios.append(odd_count / len(front_balls))
                
                # 分组：奇数多、偶数多、平衡
                for ratio_range, name in [((0.0, 0.4), 'even_heavy'), 
                                        ((0.4, 0.6), 'balanced'), 
                                        ((0.6, 1.0), 'odd_heavy')]:
                    mask = [(r >= ratio_range[0] and r <= ratio_range[1]) for r in odd_even_ratios]
                    if sum(mask) > 20:
                        filtered_data = data[mask]
                        tasks.append({
                            'type': 'odd_even',
                            'pattern': name,
                            'data': filtered_data
                        })
        
        except Exception as e:
            logger_manager.error(f"生成奇偶性任务失败: {e}")
        
        return tasks
    
    def _sample_meta_batch(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """采样元批次"""
        try:
            if len(tasks) <= self.meta_batch_size:
                return tasks
            
            # 随机采样
            indices = np.random.choice(len(tasks), self.meta_batch_size, replace=False)
            return [tasks[i] for i in indices]
            
        except Exception as e:
            logger_manager.error(f"采样元批次失败: {e}")
            return tasks[:self.meta_batch_size]
    
    def _fast_adaptation(self, objective_function: Callable, parameter_space: Dict[str, Any], 
                        task: Dict[str, Any]) -> Dict[str, Any]:
        """快速适应"""
        try:
            # 复制参数
            adapted_params = copy.deepcopy(parameter_space)
            
            # 获取任务数据
            task_data = task.get('data')
            if task_data is None or task_data.empty:
                return adapted_params
            
            # 快速适应步骤
            for step in range(self.adaptation_steps):
                # 计算梯度（简化版）
                gradient = self._compute_gradient(objective_function, adapted_params, task_data)
                
                # 更新参数
                adapted_params = self._apply_gradient(adapted_params, gradient, self.learning_rate)
            
            return adapted_params
            
        except Exception as e:
            logger_manager.error(f"快速适应失败: {e}")
            return parameter_space
    
    def _compute_gradient(self, objective_function: Callable, parameters: Dict[str, Any], 
                         data: pd.DataFrame) -> Dict[str, float]:
        """计算梯度（数值梯度）"""
        try:
            gradient = {}
            epsilon = 1e-5
            
            # 基准分数
            base_score = objective_function(parameters, data)
            
            # 对每个参数计算数值梯度
            for param_name, param_value in parameters.items():
                if isinstance(param_value, (int, float)):
                    # 扰动参数
                    perturbed_params = copy.deepcopy(parameters)
                    perturbed_params[param_name] = param_value + epsilon
                    
                    # 计算扰动后的分数
                    perturbed_score = objective_function(perturbed_params, data)
                    
                    # 计算梯度
                    gradient[param_name] = (perturbed_score - base_score) / epsilon
            
            return gradient
            
        except Exception as e:
            logger_manager.error(f"计算梯度失败: {e}")
            return {}
    
    def _apply_gradient(self, parameters: Dict[str, Any], gradient: Dict[str, float], 
                       learning_rate: float) -> Dict[str, Any]:
        """应用梯度更新"""
        try:
            updated_params = copy.deepcopy(parameters)
            
            for param_name, grad_value in gradient.items():
                if param_name in updated_params and isinstance(updated_params[param_name], (int, float)):
                    updated_params[param_name] += learning_rate * grad_value
                    
                    # 参数约束
                    updated_params[param_name] = self._constrain_parameter(param_name, updated_params[param_name])
            
            return updated_params
            
        except Exception as e:
            logger_manager.error(f"应用梯度失败: {e}")
            return parameters
    
    def _constrain_parameter(self, param_name: str, value: float) -> float:
        """约束参数值"""
        # 常见参数约束
        constraints = {
            'learning_rate': (1e-6, 1.0),
            'dropout_rate': (0.0, 0.9),
            'batch_size': (1, 1024),
            'epochs': (1, 1000),
            'hidden_units': (1, 1024)
        }
        
        if param_name in constraints:
            min_val, max_val = constraints[param_name]
            return max(min_val, min(max_val, value))
        
        return value
    
    def _compute_meta_gradient(self, objective_function: Callable, 
                              adapted_parameters: List[Dict[str, Any]], 
                              meta_batch: List[Dict[str, Any]]) -> Dict[str, float]:
        """计算元梯度"""
        try:
            meta_gradient = {}
            
            # 对每个适应后的参数计算元梯度
            for params, task in zip(adapted_parameters, meta_batch):
                task_data = task.get('data')
                if task_data is None or task_data.empty:
                    continue
                
                # 计算任务梯度
                task_gradient = self._compute_gradient(objective_function, params, task_data)
                
                # 累积元梯度
                for param_name, grad_value in task_gradient.items():
                    if param_name not in meta_gradient:
                        meta_gradient[param_name] = 0.0
                    meta_gradient[param_name] += grad_value
            
            # 平均化
            if meta_gradient:
                for param_name in meta_gradient:
                    meta_gradient[param_name] /= len(adapted_parameters)
            
            return meta_gradient
            
        except Exception as e:
            logger_manager.error(f"计算元梯度失败: {e}")
            return {}
    
    def _update_meta_parameters(self, parameter_space: Dict[str, Any], 
                               meta_gradient: Dict[str, float]) -> Dict[str, Any]:
        """更新元参数"""
        try:
            updated_space = copy.deepcopy(parameter_space)
            
            # 应用元梯度
            for param_name, grad_value in meta_gradient.items():
                if param_name in updated_space and isinstance(updated_space[param_name], (int, float)):
                    updated_space[param_name] += self.learning_rate * grad_value
                    updated_space[param_name] = self._constrain_parameter(param_name, updated_space[param_name])
            
            return updated_space
            
        except Exception as e:
            logger_manager.error(f"更新元参数失败: {e}")
            return parameter_space
    
    def _evaluate_parameters(self, objective_function: Callable, parameters: Dict[str, Any], 
                           data: pd.DataFrame) -> float:
        """评估参数"""
        try:
            if data is None or data.empty:
                return 0.0
            
            return objective_function(parameters, data)
            
        except Exception as e:
            logger_manager.error(f"评估参数失败: {e}")
            return 0.0
    
    def _should_early_stop(self, patience: int = 10) -> bool:
        """检查是否应该早停"""
        try:
            if len(self.performance_history) < patience:
                return False
            
            # 检查最近几次迭代是否没有改进
            recent_scores = [h['best_score'] for h in self.performance_history[-patience:]]
            
            # 如果最近的分数都相同，则早停
            if len(set(recent_scores)) == 1:
                return True
            
            return False
            
        except Exception as e:
            logger_manager.error(f"早停检查失败: {e}")
            return False
    
    def save_optimization_results(self, file_path: str = None) -> str:
        """保存优化结果"""
        try:
            if file_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = f"meta_learning_results_{timestamp}.json"
            
            results = {
                'timestamp': datetime.now().isoformat(),
                'best_parameters': self.best_parameters,
                'performance_history': self.performance_history,
                'optimization_history': self.optimization_history,
                'config': self.config
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger_manager.info(f"优化结果已保存到 {file_path}")
            return file_path
            
        except Exception as e:
            logger_manager.error(f"保存优化结果失败: {e}")
            return ""
    
    def load_optimization_results(self, file_path: str) -> bool:
        """加载优化结果"""
        try:
            if not os.path.exists(file_path):
                logger_manager.error(f"文件不存在: {file_path}")
                return False
            
            with open(file_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            self.best_parameters = results.get('best_parameters', {})
            self.performance_history = results.get('performance_history', [])
            self.optimization_history = results.get('optimization_history', [])
            
            logger_manager.info(f"优化结果已从 {file_path} 加载")
            return True
            
        except Exception as e:
            logger_manager.error(f"加载优化结果失败: {e}")
            return False
