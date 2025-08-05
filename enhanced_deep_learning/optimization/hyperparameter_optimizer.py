#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
超参数优化模块
Hyperparameter Optimization Module

提供贝叶斯优化、网格搜索、随机搜索、遗传算法等超参数优化方法。
"""

import os
import time
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import math

from core_modules import logger_manager
from ..utils.exceptions import DeepLearningException


class OptimizationMethod(Enum):
    """优化方法枚举"""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    GENETIC = "genetic"
    PARTICLE_SWARM = "particle_swarm"
    HYPERBAND = "hyperband"


@dataclass
class HyperParameter:
    """超参数定义"""
    name: str
    param_type: str  # 'int', 'float', 'categorical', 'bool'
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    choices: Optional[List[Any]] = None
    log_scale: bool = False
    default_value: Any = None


@dataclass
class OptimizationConfig:
    """优化配置"""
    method: OptimizationMethod
    max_trials: int = 100
    max_time: Optional[float] = None  # 最大优化时间（秒）
    n_jobs: int = 1
    cv_folds: int = 5
    scoring_metric: str = 'accuracy'
    early_stopping: bool = True
    patience: int = 10
    random_state: Optional[int] = None


@dataclass
class Trial:
    """试验记录"""
    trial_id: int
    parameters: Dict[str, Any]
    score: float
    duration: float
    status: str = 'completed'  # 'completed', 'failed', 'pruned'
    metadata: Dict[str, Any] = field(default_factory=dict)


class GridSearchOptimizer:
    """网格搜索优化器"""
    
    def __init__(self, parameters: List[HyperParameter], config: OptimizationConfig):
        """
        初始化网格搜索优化器
        
        Args:
            parameters: 超参数列表
            config: 优化配置
        """
        self.parameters = parameters
        self.config = config
        self.param_grid = self._build_param_grid()
        
        logger_manager.info(f"网格搜索优化器初始化完成，参数组合数: {len(self.param_grid)}")
    
    def _build_param_grid(self) -> List[Dict[str, Any]]:
        """构建参数网格"""
        try:
            param_values = {}
            
            for param in self.parameters:
                if param.param_type == 'categorical':
                    param_values[param.name] = param.choices
                elif param.param_type == 'bool':
                    param_values[param.name] = [True, False]
                elif param.param_type in ['int', 'float']:
                    # 为数值参数创建网格
                    if param.log_scale:
                        values = np.logspace(
                            np.log10(param.min_value),
                            np.log10(param.max_value),
                            num=10
                        )
                    else:
                        values = np.linspace(param.min_value, param.max_value, num=10)
                    
                    if param.param_type == 'int':
                        values = [int(v) for v in values]
                    
                    param_values[param.name] = values
            
            # 生成所有组合
            keys = param_values.keys()
            combinations = itertools.product(*param_values.values())
            param_grid = [dict(zip(keys, combo)) for combo in combinations]
            
            # 限制组合数量
            if len(param_grid) > self.config.max_trials:
                random.shuffle(param_grid)
                param_grid = param_grid[:self.config.max_trials]
            
            return param_grid
            
        except Exception as e:
            logger_manager.error(f"构建参数网格失败: {e}")
            return [{}]
    
    def optimize(self, objective_function: Callable) -> List[Trial]:
        """
        执行网格搜索优化
        
        Args:
            objective_function: 目标函数
            
        Returns:
            试验结果列表
        """
        try:
            trials = []
            start_time = time.time()

            # 初始化智能早停机制
            from ..utils.intelligent_early_stopping import GeneralIntelligentEarlyStopping
            early_stopping = GeneralIntelligentEarlyStopping(
                patience=20,  # 连续20次相同结果时停止
                min_delta=1e-6,
                verbose=1
            )
            early_stopping.reset()

            for i, params in enumerate(self.param_grid):
                # 检查时间限制
                if self.config.max_time and (time.time() - start_time) > self.config.max_time:
                    logger_manager.info(f"达到时间限制，停止优化")
                    break

                # 执行试验
                trial_start = time.time()
                try:
                    score = objective_function(params)
                    status = 'completed'
                except Exception as e:
                    logger_manager.error(f"试验 {i} 失败: {e}")
                    score = float('-inf')
                    status = 'failed'

                trial_duration = time.time() - trial_start

                trial = Trial(
                    trial_id=i,
                    parameters=params,
                    score=score,
                    duration=trial_duration,
                    status=status
                )

                trials.append(trial)

                logger_manager.debug(f"试验 {i}: 分数={score:.4f}, 参数={params}")

                # 智能早停检查（使用负分数，因为我们要最大化分数）
                if status == 'completed' and early_stopping.update(-score):
                    logger_manager.info(f"网格搜索智能早停，已完成 {i + 1} 次试验")
                    break
            
            # 按分数排序
            trials.sort(key=lambda x: x.score, reverse=True)
            
            logger_manager.info(f"网格搜索完成，最佳分数: {trials[0].score:.4f}")
            return trials
            
        except Exception as e:
            logger_manager.error(f"网格搜索优化失败: {e}")
            return []


class RandomSearchOptimizer:
    """随机搜索优化器"""
    
    def __init__(self, parameters: List[HyperParameter], config: OptimizationConfig):
        """
        初始化随机搜索优化器
        
        Args:
            parameters: 超参数列表
            config: 优化配置
        """
        self.parameters = parameters
        self.config = config
        
        if config.random_state:
            random.seed(config.random_state)
            np.random.seed(config.random_state)
        
        logger_manager.info("随机搜索优化器初始化完成")
    
    def _sample_parameter(self, param: HyperParameter) -> Any:
        """采样单个参数"""
        try:
            if param.param_type == 'categorical':
                return random.choice(param.choices)
            elif param.param_type == 'bool':
                return random.choice([True, False])
            elif param.param_type == 'int':
                if param.log_scale:
                    log_min = np.log10(param.min_value)
                    log_max = np.log10(param.max_value)
                    log_value = random.uniform(log_min, log_max)
                    return int(10 ** log_value)
                else:
                    return random.randint(param.min_value, param.max_value)
            elif param.param_type == 'float':
                if param.log_scale:
                    log_min = np.log10(param.min_value)
                    log_max = np.log10(param.max_value)
                    log_value = random.uniform(log_min, log_max)
                    return 10 ** log_value
                else:
                    return random.uniform(param.min_value, param.max_value)
            else:
                return param.default_value
                
        except Exception as e:
            logger_manager.error(f"参数采样失败: {e}")
            return param.default_value
    
    def _sample_parameters(self) -> Dict[str, Any]:
        """采样参数组合"""
        return {param.name: self._sample_parameter(param) for param in self.parameters}
    
    def optimize(self, objective_function: Callable) -> List[Trial]:
        """
        执行随机搜索优化
        
        Args:
            objective_function: 目标函数
            
        Returns:
            试验结果列表
        """
        try:
            trials = []
            start_time = time.time()

            # 初始化智能早停机制
            from ..utils.intelligent_early_stopping import GeneralIntelligentEarlyStopping
            early_stopping = GeneralIntelligentEarlyStopping(
                patience=20,  # 连续20次相同结果时停止
                min_delta=1e-6,
                verbose=1
            )
            early_stopping.reset()

            for i in range(self.config.max_trials):
                # 检查时间限制
                if self.config.max_time and (time.time() - start_time) > self.config.max_time:
                    logger_manager.info(f"达到时间限制，停止优化")
                    break

                # 采样参数
                params = self._sample_parameters()

                # 执行试验
                trial_start = time.time()
                try:
                    score = objective_function(params)
                    status = 'completed'
                except Exception as e:
                    logger_manager.error(f"试验 {i} 失败: {e}")
                    score = float('-inf')
                    status = 'failed'

                trial_duration = time.time() - trial_start

                trial = Trial(
                    trial_id=i,
                    parameters=params,
                    score=score,
                    duration=trial_duration,
                    status=status
                )

                trials.append(trial)

                logger_manager.debug(f"试验 {i}: 分数={score:.4f}, 参数={params}")

                # 智能早停检查（使用负分数，因为我们要最大化分数）
                if status == 'completed' and early_stopping.update(-score):
                    logger_manager.info(f"随机搜索智能早停，已完成 {i + 1} 次试验")
                    break
            
            # 按分数排序
            trials.sort(key=lambda x: x.score, reverse=True)
            
            logger_manager.info(f"随机搜索完成，最佳分数: {trials[0].score:.4f}")
            return trials
            
        except Exception as e:
            logger_manager.error(f"随机搜索优化失败: {e}")
            return []


class BayesianOptimizer:
    """贝叶斯优化器（简化实现）"""
    
    def __init__(self, parameters: List[HyperParameter], config: OptimizationConfig):
        """
        初始化贝叶斯优化器
        
        Args:
            parameters: 超参数列表
            config: 优化配置
        """
        self.parameters = parameters
        self.config = config
        self.trials_history = []
        
        logger_manager.info("贝叶斯优化器初始化完成")
    
    def _acquisition_function(self, params: Dict[str, Any]) -> float:
        """
        获取函数（简化实现）
        
        Args:
            params: 参数组合
            
        Returns:
            获取值
        """
        try:
            if not self.trials_history:
                return random.random()
            
            # 简化的期望改进计算
            best_score = max(trial.score for trial in self.trials_history if trial.status == 'completed')
            
            # 计算与历史参数的距离
            distances = []
            for trial in self.trials_history:
                distance = 0
                for param_name, param_value in params.items():
                    if param_name in trial.parameters:
                        if isinstance(param_value, (int, float)):
                            distance += abs(param_value - trial.parameters[param_name])
                        else:
                            distance += 0 if param_value == trial.parameters[param_name] else 1
                distances.append(distance)
            
            # 简化的获取函数：距离越远，获取值越高
            min_distance = min(distances) if distances else 1
            acquisition_value = min_distance + random.random() * 0.1
            
            return acquisition_value
            
        except Exception as e:
            logger_manager.error(f"计算获取函数失败: {e}")
            return random.random()
    
    def _suggest_next_parameters(self) -> Dict[str, Any]:
        """建议下一组参数"""
        try:
            # 生成候选参数
            candidates = []
            for _ in range(100):  # 生成100个候选
                candidate = {}
                for param in self.parameters:
                    if param.param_type == 'categorical':
                        candidate[param.name] = random.choice(param.choices)
                    elif param.param_type == 'bool':
                        candidate[param.name] = random.choice([True, False])
                    elif param.param_type == 'int':
                        if param.log_scale:
                            log_min = np.log10(param.min_value)
                            log_max = np.log10(param.max_value)
                            log_value = random.uniform(log_min, log_max)
                            candidate[param.name] = int(10 ** log_value)
                        else:
                            candidate[param.name] = random.randint(param.min_value, param.max_value)
                    elif param.param_type == 'float':
                        if param.log_scale:
                            log_min = np.log10(param.min_value)
                            log_max = np.log10(param.max_value)
                            log_value = random.uniform(log_min, log_max)
                            candidate[param.name] = 10 ** log_value
                        else:
                            candidate[param.name] = random.uniform(param.min_value, param.max_value)
                
                candidates.append(candidate)
            
            # 选择获取值最高的候选
            best_candidate = max(candidates, key=self._acquisition_function)
            return best_candidate
            
        except Exception as e:
            logger_manager.error(f"建议参数失败: {e}")
            # 回退到随机采样
            return {param.name: param.default_value for param in self.parameters}
    
    def optimize(self, objective_function: Callable) -> List[Trial]:
        """
        执行贝叶斯优化
        
        Args:
            objective_function: 目标函数
            
        Returns:
            试验结果列表
        """
        try:
            start_time = time.time()

            # 初始化智能早停机制
            from ..utils.intelligent_early_stopping import GeneralIntelligentEarlyStopping
            early_stopping = GeneralIntelligentEarlyStopping(
                patience=20,  # 连续20次相同结果时停止
                min_delta=1e-6,
                verbose=1
            )
            early_stopping.reset()

            for i in range(self.config.max_trials):
                # 检查时间限制
                if self.config.max_time and (time.time() - start_time) > self.config.max_time:
                    logger_manager.info(f"达到时间限制，停止优化")
                    break

                # 建议参数
                params = self._suggest_next_parameters()

                # 执行试验
                trial_start = time.time()
                try:
                    score = objective_function(params)
                    status = 'completed'
                except Exception as e:
                    logger_manager.error(f"试验 {i} 失败: {e}")
                    score = float('-inf')
                    status = 'failed'

                trial_duration = time.time() - trial_start

                trial = Trial(
                    trial_id=i,
                    parameters=params,
                    score=score,
                    duration=trial_duration,
                    status=status
                )

                self.trials_history.append(trial)

                logger_manager.debug(f"试验 {i}: 分数={score:.4f}, 参数={params}")

                # 智能早停检查（使用负分数，因为我们要最大化分数）
                if status == 'completed' and early_stopping.update(-score):
                    logger_manager.info(f"超参数优化智能早停，已完成 {i + 1} 次试验")
                    break
            
            # 按分数排序
            self.trials_history.sort(key=lambda x: x.score, reverse=True)
            
            logger_manager.info(f"贝叶斯优化完成，最佳分数: {self.trials_history[0].score:.4f}")
            return self.trials_history
            
        except Exception as e:
            logger_manager.error(f"贝叶斯优化失败: {e}")
            return []


class GeneticOptimizer:
    """遗传算法优化器"""
    
    def __init__(self, parameters: List[HyperParameter], config: OptimizationConfig):
        """
        初始化遗传算法优化器
        
        Args:
            parameters: 超参数列表
            config: 优化配置
        """
        self.parameters = parameters
        self.config = config
        self.population_size = min(50, config.max_trials // 2)
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        
        logger_manager.info(f"遗传算法优化器初始化完成，种群大小: {self.population_size}")
    
    def _create_individual(self) -> Dict[str, Any]:
        """创建个体"""
        individual = {}
        for param in self.parameters:
            if param.param_type == 'categorical':
                individual[param.name] = random.choice(param.choices)
            elif param.param_type == 'bool':
                individual[param.name] = random.choice([True, False])
            elif param.param_type == 'int':
                individual[param.name] = random.randint(param.min_value, param.max_value)
            elif param.param_type == 'float':
                individual[param.name] = random.uniform(param.min_value, param.max_value)
        
        return individual
    
    def _mutate(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """变异操作"""
        mutated = individual.copy()
        
        for param in self.parameters:
            if random.random() < self.mutation_rate:
                if param.param_type == 'categorical':
                    mutated[param.name] = random.choice(param.choices)
                elif param.param_type == 'bool':
                    mutated[param.name] = not mutated[param.name]
                elif param.param_type == 'int':
                    # 高斯变异
                    current_value = mutated[param.name]
                    mutation = int(random.gauss(0, (param.max_value - param.min_value) * 0.1))
                    new_value = current_value + mutation
                    mutated[param.name] = max(param.min_value, min(param.max_value, new_value))
                elif param.param_type == 'float':
                    # 高斯变异
                    current_value = mutated[param.name]
                    mutation = random.gauss(0, (param.max_value - param.min_value) * 0.1)
                    new_value = current_value + mutation
                    mutated[param.name] = max(param.min_value, min(param.max_value, new_value))
        
        return mutated
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """交叉操作"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1, child2 = parent1.copy(), parent2.copy()
        
        # 随机选择交叉点
        param_names = list(parent1.keys())
        crossover_point = random.randint(1, len(param_names) - 1)
        
        for i in range(crossover_point, len(param_names)):
            param_name = param_names[i]
            child1[param_name], child2[param_name] = child2[param_name], child1[param_name]
        
        return child1, child2
    
    def optimize(self, objective_function: Callable) -> List[Trial]:
        """
        执行遗传算法优化
        
        Args:
            objective_function: 目标函数
            
        Returns:
            试验结果列表
        """
        try:
            # 初始化种群
            population = [self._create_individual() for _ in range(self.population_size)]
            all_trials = []
            trial_id = 0
            
            generations = self.config.max_trials // self.population_size
            
            for generation in range(generations):
                # 评估种群
                generation_trials = []
                for individual in population:
                    try:
                        score = objective_function(individual)
                        status = 'completed'
                    except Exception as e:
                        logger_manager.error(f"个体评估失败: {e}")
                        score = float('-inf')
                        status = 'failed'
                    
                    trial = Trial(
                        trial_id=trial_id,
                        parameters=individual,
                        score=score,
                        duration=0.0,
                        status=status,
                        metadata={'generation': generation}
                    )
                    
                    generation_trials.append(trial)
                    all_trials.append(trial)
                    trial_id += 1
                
                # 选择
                generation_trials.sort(key=lambda x: x.score, reverse=True)
                elite_size = self.population_size // 4
                elite = [trial.parameters for trial in generation_trials[:elite_size]]
                
                # 生成新种群
                new_population = elite.copy()
                
                while len(new_population) < self.population_size:
                    # 锦标赛选择
                    tournament_size = 3
                    tournament = random.sample(generation_trials[:self.population_size//2], tournament_size)
                    parent1 = max(tournament, key=lambda x: x.score).parameters
                    
                    tournament = random.sample(generation_trials[:self.population_size//2], tournament_size)
                    parent2 = max(tournament, key=lambda x: x.score).parameters
                    
                    # 交叉和变异
                    child1, child2 = self._crossover(parent1, parent2)
                    child1 = self._mutate(child1)
                    child2 = self._mutate(child2)
                    
                    new_population.extend([child1, child2])
                
                population = new_population[:self.population_size]
                
                best_score = generation_trials[0].score
                logger_manager.debug(f"第 {generation} 代，最佳分数: {best_score:.4f}")
            
            # 按分数排序
            all_trials.sort(key=lambda x: x.score, reverse=True)
            
            logger_manager.info(f"遗传算法优化完成，最佳分数: {all_trials[0].score:.4f}")
            return all_trials
            
        except Exception as e:
            logger_manager.error(f"遗传算法优化失败: {e}")
            return []


class HyperparameterOptimizer:
    """超参数优化器"""
    
    def __init__(self, parameters: List[HyperParameter], config: OptimizationConfig):
        """
        初始化超参数优化器
        
        Args:
            parameters: 超参数列表
            config: 优化配置
        """
        self.parameters = parameters
        self.config = config
        self.optimizer = self._create_optimizer()
        self.optimization_history = []
        
        logger_manager.info(f"超参数优化器初始化完成，方法: {config.method.value}")
    
    def _create_optimizer(self):
        """创建具体的优化器"""
        if self.config.method == OptimizationMethod.GRID_SEARCH:
            return GridSearchOptimizer(self.parameters, self.config)
        elif self.config.method == OptimizationMethod.RANDOM_SEARCH:
            return RandomSearchOptimizer(self.parameters, self.config)
        elif self.config.method == OptimizationMethod.BAYESIAN:
            return BayesianOptimizer(self.parameters, self.config)
        elif self.config.method == OptimizationMethod.GENETIC:
            return GeneticOptimizer(self.parameters, self.config)
        else:
            logger_manager.warning(f"不支持的优化方法: {self.config.method}")
            return RandomSearchOptimizer(self.parameters, self.config)
    
    def optimize(self, objective_function: Callable) -> List[Trial]:
        """
        执行超参数优化
        
        Args:
            objective_function: 目标函数
            
        Returns:
            试验结果列表
        """
        try:
            start_time = time.time()
            
            trials = self.optimizer.optimize(objective_function)
            
            total_time = time.time() - start_time
            
            # 记录优化历史
            self.optimization_history.append({
                'method': self.config.method.value,
                'total_trials': len(trials),
                'best_score': trials[0].score if trials else 0.0,
                'total_time': total_time,
                'timestamp': time.time()
            })
            
            logger_manager.info(f"超参数优化完成，总时间: {total_time:.2f}秒")
            return trials
            
        except Exception as e:
            logger_manager.error(f"超参数优化失败: {e}")
            return []
    
    def get_best_parameters(self, trials: List[Trial]) -> Dict[str, Any]:
        """获取最佳参数"""
        if not trials:
            return {}
        
        best_trial = max(trials, key=lambda x: x.score)
        return best_trial.parameters
    
    def save_results(self, trials: List[Trial], file_path: str):
        """保存优化结果"""
        try:
            results = {
                'config': {
                    'method': self.config.method.value,
                    'max_trials': self.config.max_trials,
                    'scoring_metric': self.config.scoring_metric
                },
                'parameters': [
                    {
                        'name': p.name,
                        'type': p.param_type,
                        'min_value': p.min_value,
                        'max_value': p.max_value,
                        'choices': p.choices
                    } for p in self.parameters
                ],
                'trials': [
                    {
                        'trial_id': t.trial_id,
                        'parameters': t.parameters,
                        'score': t.score,
                        'duration': t.duration,
                        'status': t.status
                    } for t in trials
                ],
                'optimization_history': self.optimization_history
            }
            
            with open(file_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger_manager.info(f"优化结果已保存到: {file_path}")
            
        except Exception as e:
            logger_manager.error(f"保存优化结果失败: {e}")
    
    def load_results(self, file_path: str) -> List[Trial]:
        """加载优化结果"""
        try:
            with open(file_path, 'r') as f:
                results = json.load(f)
            
            trials = []
            for trial_data in results['trials']:
                trial = Trial(
                    trial_id=trial_data['trial_id'],
                    parameters=trial_data['parameters'],
                    score=trial_data['score'],
                    duration=trial_data['duration'],
                    status=trial_data['status']
                )
                trials.append(trial)
            
            logger_manager.info(f"优化结果已从 {file_path} 加载")
            return trials
            
        except Exception as e:
            logger_manager.error(f"加载优化结果失败: {e}")
            return []


# 全局超参数优化器实例
default_parameters = [
    HyperParameter('learning_rate', 'float', 0.0001, 0.1, log_scale=True),
    HyperParameter('batch_size', 'int', 16, 128),
    HyperParameter('epochs', 'int', 10, 100),
    HyperParameter('optimizer', 'categorical', choices=['adam', 'sgd', 'rmsprop'])
]

default_config = OptimizationConfig(
    method=OptimizationMethod.RANDOM_SEARCH,
    max_trials=50,
    scoring_metric='accuracy'
)

hyperparameter_optimizer = HyperparameterOptimizer(default_parameters, default_config)


if __name__ == "__main__":
    # 测试超参数优化功能
    print("🎯 测试超参数优化功能...")
    
    # 定义测试目标函数
    def test_objective(params):
        """测试目标函数"""
        # 模拟模型训练和评估
        score = random.random()
        
        # 添加一些逻辑使某些参数组合更好
        if params.get('learning_rate', 0.01) < 0.01:
            score += 0.1
        if params.get('batch_size', 32) == 32:
            score += 0.05
        
        return score
    
    # 测试不同的优化方法
    methods = [OptimizationMethod.RANDOM_SEARCH, OptimizationMethod.GRID_SEARCH, OptimizationMethod.GENETIC]
    
    for method in methods:
        print(f"\n测试 {method.value} 优化...")
        
        config = OptimizationConfig(method=method, max_trials=20)
        optimizer = HyperparameterOptimizer(default_parameters, config)
        
        trials = optimizer.optimize(test_objective)
        
        if trials:
            best_params = optimizer.get_best_parameters(trials)
            print(f"最佳参数: {best_params}")
            print(f"最佳分数: {trials[0].score:.4f}")
    
    print("超参数优化功能测试完成")
