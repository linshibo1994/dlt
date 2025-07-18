#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模型优化器
提供自动化的模型参数优化功能
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any, Callable, Union
from datetime import datetime
import json
import time
from tqdm import tqdm
from collections import defaultdict
import itertools
import random

# 尝试导入核心模块
try:
    from core_modules import logger_manager, data_manager, cache_manager
except ImportError:
    # 如果在不同目录运行，添加父目录到路径
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core_modules import logger_manager, data_manager, cache_manager

# 尝试导入模型基准测试框架
try:
    from improvements.model_benchmark import ModelBenchmark, get_model_benchmark
except ImportError:
    logger_manager.error("模型基准测试框架未找到，请确保improvements/model_benchmark.py文件存在")
    ModelBenchmark = None
    get_model_benchmark = None

# 尝试导入scikit-learn和scikit-optimize
try:
    from sklearn.model_selection import ParameterGrid, ParameterSampler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False


class ModelOptimizer:
    """模型优化器"""
    
    def __init__(self, model_creator: Callable, param_space: Dict, 
                optimization_method: str = 'grid', n_iter: int = 50):
        """初始化模型优化器
        
        Args:
            model_creator: 模型创建函数，接受参数字典，返回模型实例
            param_space: 参数空间，字典形式，键为参数名，值为参数可能的取值列表
            optimization_method: 优化方法，可选值: 'grid', 'random', 'bayesian'
            n_iter: 随机搜索或贝叶斯优化的迭代次数
        """
        self.model_creator = model_creator
        self.param_space = param_space
        self.optimization_method = optimization_method
        self.n_iter = n_iter
        
        # 优化结果
        self.best_params = None
        self.best_score = -float('inf')
        self.best_model = None
        self.results = []
        
        # 获取数据
        self.df = data_manager.get_data()
        if self.df is None:
            logger_manager.error("数据未加载")
        
        # 获取基准测试实例
        if ModelBenchmark is not None:
            self.benchmark = get_model_benchmark()
        else:
            self.benchmark = None
            logger_manager.error("模型基准测试框架未找到，性能评估功能将不可用")
    
    def optimize(self, train_periods: int = 500, val_periods: int = 50, 
                metric: str = 'accuracy', verbose: bool = True) -> Dict:
        """执行参数优化
        
        Args:
            train_periods: 训练数据期数
            val_periods: 验证数据期数
            metric: 优化指标，可选值: 'accuracy', 'hit_rate', 'roi'
            verbose: 是否显示详细信息
            
        Returns:
            Dict: 优化结果
        """
        if self.df is None or len(self.df) < train_periods + val_periods:
            logger_manager.error("数据不足，无法进行参数优化")
            return {}
        
        # 准备数据
        train_data = self.df.iloc[val_periods:val_periods+train_periods]
        val_data = self.df.iloc[:val_periods]
        
        # 根据优化方法选择参数组合
        param_combinations = self._get_param_combinations()
        
        if verbose:
            logger_manager.info(f"开始参数优化，方法: {self.optimization_method}, 参数组合数: {len(param_combinations)}")
            iterator = tqdm(param_combinations, desc="参数优化")
        else:
            iterator = param_combinations
        
        # 评估每个参数组合
        for params in iterator:
            try:
                # 使用当前参数创建模型
                model = self.model_creator(**params)
                
                # 训练模型
                if hasattr(model, 'fit'):
                    model.fit(train_data)
                elif hasattr(model, 'train'):
                    model.train(train_data)
                
                # 在验证集上评估
                score = self._evaluate_model(model, val_data, metric)
                
                # 记录结果
                result = {
                    'params': params,
                    'score': score,
                    'timestamp': datetime.now().isoformat()
                }
                self.results.append(result)
                
                # 更新最佳参数
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params
                    self.best_model = model
                    
                    if verbose:
                        logger_manager.info(f"发现更好的参数: {params}, 得分: {score:.4f}")
                
            except Exception as e:
                logger_manager.error(f"评估参数 {params} 失败: {e}")
        
        # 整理结果
        optimization_result = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'all_results': self.results,
            'optimization_method': self.optimization_method,
            'metric': metric,
            'timestamp': datetime.now().isoformat()
        }
        
        if verbose:
            logger_manager.info(f"参数优化完成，最佳参数: {self.best_params}, 最佳得分: {self.best_score:.4f}")
        
        return optimization_result
    
    def _get_param_combinations(self):
        """获取参数组合"""
        if self.optimization_method == 'grid':
            # 网格搜索
            if SKLEARN_AVAILABLE:
                return list(ParameterGrid(self.param_space))
            else:
                # 手动实现网格搜索
                keys = self.param_space.keys()
                values = self.param_space.values()
                combinations = list(itertools.product(*values))
                return [dict(zip(keys, combo)) for combo in combinations]
        
        elif self.optimization_method == 'random':
            # 随机搜索
            if SKLEARN_AVAILABLE:
                return list(ParameterSampler(self.param_space, n_iter=self.n_iter, random_state=42))
            else:
                # 手动实现随机搜索
                random.seed(42)
                
                combinations = []
                for _ in range(self.n_iter):
                    params = {}
                    for key, values in self.param_space.items():
                        params[key] = random.choice(values)
                    combinations.append(params)
                
                return combinations
        
        elif self.optimization_method == 'bayesian':
            # 贝叶斯优化
            if SKOPT_AVAILABLE:
                # 将参数空间转换为skopt格式
                skopt_space = []
                for key, values in self.param_space.items():
                    if isinstance(values[0], int):
                        skopt_space.append(Integer(min(values), max(values), name=key))
                    elif isinstance(values[0], float):
                        skopt_space.append(Real(min(values), max(values), name=key))
                    else:
                        skopt_space.append(Categorical(values, name=key))
                
                # 使用贝叶斯优化生成参数组合
                from skopt.sampler import Sobol
                from skopt.space import Space
                
                space = Space(skopt_space)
                sobol = Sobol()
                return [dict(zip(self.param_space.keys(), x)) for x in sobol.generate(space.dimensions, self.n_iter)]
            else:
                # 如果skopt不可用，退回到随机搜索
                logger_manager.warning("scikit-optimize不可用，退回到随机搜索")
                return self._get_param_combinations('random')
        
        else:
            logger_manager.error(f"未知的优化方法: {self.optimization_method}")
            return []
    
    def _evaluate_model(self, model, val_data, metric):
        """评估模型性能"""
        # 在验证集上进行预测
        predictions = []
        
        for i in range(len(val_data)):
            try:
                # 获取验证期的实际结果
                val_row = val_data.iloc[i]
                actual_front, actual_back = data_manager.parse_balls(val_row)
                
                # 使用后续数据进行预测
                historical_data = val_data.iloc[i+1:]
                
                # 调用模型的预测方法
                if hasattr(model, 'predict'):
                    pred = model.predict(historical_data)
                elif hasattr(model, 'predict_next'):
                    pred = model.predict_next(historical_data)
                else:
                    logger_manager.error("模型没有predict或predict_next方法")
                    return -float('inf')
                
                # 解析预测结果
                if isinstance(pred, list) and len(pred) > 0:
                    if isinstance(pred[0], tuple) and len(pred[0]) == 2:
                        pred_front, pred_back = pred[0]
                    elif isinstance(pred[0], dict) and 'front_balls' in pred[0] and 'back_balls' in pred[0]:
                        pred_front = pred[0]['front_balls']
                        pred_back = pred[0]['back_balls']
                    else:
                        logger_manager.warning(f"无法解析预测结果: {pred[0]}")
                        continue
                elif isinstance(pred, tuple) and len(pred) == 2:
                    pred_front, pred_back = pred
                else:
                    logger_manager.warning(f"无法解析预测结果: {pred}")
                    continue
                
                # 计算命中情况
                front_hits = len(set(pred_front) & set(actual_front))
                back_hits = len(set(pred_back) & set(actual_back))
                
                # 记录预测结果
                predictions.append({
                    'front_hits': front_hits,
                    'back_hits': back_hits,
                    'actual_front': actual_front,
                    'actual_back': actual_back,
                    'pred_front': pred_front,
                    'pred_back': pred_back
                })
                
            except Exception as e:
                logger_manager.error(f"预测验证期 {i} 失败: {e}")
        
        # 计算评估指标
        if not predictions:
            return -float('inf')
        
        if metric == 'accuracy':
            # 准确率（加权得分）
            scores = []
            for pred in predictions:
                front_hits = pred['front_hits']
                back_hits = pred['back_hits']
                
                # 前区权重
                front_weight = 0.7
                # 后区权重
                back_weight = 0.3
                
                # 前区得分（满分为5）
                front_score = front_hits / 5
                # 后区得分（满分为2）
                back_score = back_hits / 2
                
                # 加权总分
                score = front_weight * front_score + back_weight * back_score
                scores.append(score)
            
            return np.mean(scores)
        
        elif metric == 'hit_rate':
            # 命中率（任意号码命中）
            hit_count = sum(1 for pred in predictions if pred['front_hits'] > 0 or pred['back_hits'] > 0)
            return hit_count / len(predictions)
        
        elif metric == 'roi':
            # 投资回报率
            total_cost = len(predictions) * 2  # 假设每注2元
            total_prize = 0
            
            for pred in predictions:
                front_hits = pred['front_hits']
                back_hits = pred['back_hits']
                
                # 计算奖金
                if front_hits == 5 and back_hits == 2:
                    total_prize += 10000000  # 一等奖
                elif front_hits == 5 and back_hits == 1:
                    total_prize += 100000  # 二等奖
                elif front_hits == 5 and back_hits == 0:
                    total_prize += 3000  # 三等奖
                elif front_hits == 4 and back_hits == 2:
                    total_prize += 200  # 四等奖
                elif front_hits == 4 and back_hits == 1:
                    total_prize += 10  # 五等奖
                elif front_hits == 3 and back_hits == 2:
                    total_prize += 10  # 五等奖
                elif front_hits == 4 and back_hits == 0:
                    total_prize += 5  # 六等奖
                elif front_hits == 3 and back_hits == 1:
                    total_prize += 5  # 六等奖
                elif front_hits == 2 and back_hits == 2:
                    total_prize += 5  # 六等奖
                elif front_hits == 0 and back_hits == 2:
                    total_prize += 5  # 六等奖
            
            return (total_prize - total_cost) / total_cost if total_cost > 0 else 0
        
        else:
            logger_manager.error(f"未知的评估指标: {metric}")
            return -float('inf')
    
    def benchmark_best_model(self, model_name: str, test_periods: int = 50, verbose: bool = True) -> Dict:
        """使用基准测试框架评估最佳模型
        
        Args:
            model_name: 模型名称
            test_periods: 测试期数
            verbose: 是否显示详细信息
            
        Returns:
            Dict: 评估结果
        """
        if self.benchmark is None:
            logger_manager.error("模型基准测试框架未找到，无法进行基准测试")
            return {}
        
        if self.best_model is None:
            logger_manager.error("没有最佳模型可供评估，请先运行optimize方法")
            return {}
        
        # 注册最佳模型
        self.benchmark.register_model(
            model_name,
            lambda data: self.best_model.predict(data) if hasattr(self.best_model, 'predict') else self.best_model.predict_next(data),
            f"优化后的模型 (参数: {self.best_params})",
            "optimized"
        )
        
        # 评估模型
        result = self.benchmark.evaluate_model(model_name, test_periods, verbose=verbose)
        
        return result
    
    def visualize_optimization_process(self, output_path=None):
        """可视化优化过程
        
        Args:
            output_path: 图表输出路径
        """
        if not self.results:
            logger_manager.warning("没有优化结果可供可视化")
            return
        
        try:
            # 提取得分
            scores = [result['score'] for result in self.results]
            iterations = list(range(1, len(scores) + 1))
            
            # 创建图表
            plt.figure(figsize=(10, 6))
            
            # 绘制得分曲线
            plt.plot(iterations, scores, 'b-', alpha=0.6)
            plt.plot(iterations, scores, 'bo', alpha=0.5)
            
            # 标记最佳得分
            best_idx = scores.index(max(scores))
            plt.plot(best_idx + 1, scores[best_idx], 'ro', markersize=10)
            plt.annotate(f"最佳: {scores[best_idx]:.4f}", 
                        xy=(best_idx + 1, scores[best_idx]),
                        xytext=(best_idx + 1 + 5, scores[best_idx]),
                        arrowprops=dict(facecolor='black', shrink=0.05))
            
            # 添加移动平均线
            window = min(10, len(scores))
            if window > 1:
                moving_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
                plt.plot(range(window, len(scores) + 1), moving_avg, 'r-', alpha=0.8, label=f'{window}次迭代移动平均')
            
            # 设置图表属性
            plt.title('参数优化过程')
            plt.xlabel('迭代次数')
            plt.ylabel('评估得分')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # 保存或显示图表
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                plt.savefig(output_path)
                logger_manager.info(f"优化过程图表已保存到 {output_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger_manager.error(f"可视化优化过程失败: {e}")
    
    def visualize_parameter_importance(self, output_path=None):
        """可视化参数重要性
        
        Args:
            output_path: 图表输出路径
        """
        if not self.results or not self.best_params:
            logger_manager.warning("没有优化结果可供可视化")
            return
        
        try:
            # 分析每个参数的影响
            param_importance = {}
            
            for param_name in self.best_params.keys():
                # 收集该参数的所有取值和对应的得分
                param_values = {}
                
                for result in self.results:
                    param_value = result['params'].get(param_name)
                    score = result['score']
                    
                    if param_value not in param_values:
                        param_values[param_value] = []
                    
                    param_values[param_value].append(score)
                
                # 计算每个参数值的平均得分
                param_avg_scores = {value: np.mean(scores) for value, scores in param_values.items()}
                
                # 计算参数重要性（最高平均分与最低平均分的差值）
                if param_avg_scores:
                    max_score = max(param_avg_scores.values())
                    min_score = min(param_avg_scores.values())
                    importance = max_score - min_score
                    param_importance[param_name] = {
                        'importance': importance,
                        'avg_scores': param_avg_scores,
                        'best_value': self.best_params[param_name]
                    }
            
            # 按重要性排序
            sorted_params = sorted(param_importance.items(), key=lambda x: x[1]['importance'], reverse=True)
            
            # 创建图表
            fig, axes = plt.subplots(len(sorted_params), 1, figsize=(10, 4 * len(sorted_params)))
            if len(sorted_params) == 1:
                axes = [axes]
            
            for i, (param_name, info) in enumerate(sorted_params):
                ax = axes[i]
                
                # 准备数据
                values = []
                scores = []
                colors = []
                
                for value, score in info['avg_scores'].items():
                    values.append(str(value))
                    scores.append(score)
                    # 最佳值用红色标记
                    if value == info['best_value']:
                        colors.append('red')
                    else:
                        colors.append('skyblue')
                
                # 绘制条形图
                bars = ax.bar(values, scores, color=colors, alpha=0.7)
                
                # 添加数值标签
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.4f}', ha='center', va='bottom', fontsize=8)
                
                # 设置标题和标签
                ax.set_title(f'参数 {param_name} 的影响 (重要性: {info["importance"]:.4f})')
                ax.set_ylabel('平均得分')
                ax.set_ylim(bottom=0)
                
                # 标记最佳值
                best_idx = values.index(str(info['best_value']))
                ax.text(best_idx, scores[best_idx] / 2, '最佳', ha='center', color='white', fontweight='bold')
            
            plt.tight_layout()
            
            # 保存或显示图表
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                plt.savefig(output_path)
                logger_manager.info(f"参数重要性图表已保存到 {output_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger_manager.error(f"可视化参数重要性失败: {e}")
    
    def save_results(self, output_path):
        """保存优化结果
        
        Args:
            output_path: 结果输出路径
        """
        try:
            # 准备结果数据
            result_data = {
                'best_params': self.best_params,
                'best_score': self.best_score,
                'optimization_method': self.optimization_method,
                'param_space': self.param_space,
                'results': self.results,
                'timestamp': datetime.now().isoformat()
            }
            
            # 保存为JSON
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
            
            logger_manager.info(f"优化结果已保存到 {output_path}")
            
        except Exception as e:
            logger_manager.error(f"保存优化结果失败: {e}")
    
    def load_results(self, input_path):
        """加载优化结果
        
        Args:
            input_path: 结果输入路径
            
        Returns:
            bool: 是否加载成功
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(input_path):
                logger_manager.error(f"结果文件不存在: {input_path}")
                return False
            
            # 加载JSON
            with open(input_path, 'r', encoding='utf-8') as f:
                load_data = json.load(f)
            
            # 更新优化结果
            self.best_params = load_data.get('best_params')
            self.best_score = load_data.get('best_score')
            self.results = load_data.get('results', [])
            
            logger_manager.info(f"优化结果已从 {input_path} 加载")
            
            # 使用最佳参数创建模型
            if self.best_params:
                try:
                    self.best_model = self.model_creator(**self.best_params)
                    logger_manager.info(f"已使用最佳参数创建模型")
                except Exception as e:
                    logger_manager.error(f"使用最佳参数创建模型失败: {e}")
            
            return True
            
        except Exception as e:
            logger_manager.error(f"加载优化结果失败: {e}")
            return False


# 示例模型类
class SimplePredictor:
    """简单预测模型，用于测试参数优化"""
    
    def __init__(self, weight_frequency=0.5, weight_missing=0.3, weight_hot_cold=0.2):
        """初始化简单预测模型
        
        Args:
            weight_frequency: 频率权重
            weight_missing: 遗漏权重
            weight_hot_cold: 冷热权重
        """
        self.weight_frequency = weight_frequency
        self.weight_missing = weight_missing
        self.weight_hot_cold = weight_hot_cold
        
        # 统计数据
        self.frequency_stats = None
        self.missing_stats = None
        self.hot_cold_stats = None
    
    def fit(self, train_data):
        """训练模型
        
        Args:
            train_data: 训练数据
        """
        # 计算频率统计
        front_counter = {}
        back_counter = {}
        
        for _, row in train_data.iterrows():
            front_balls, back_balls = data_manager.parse_balls(row)
            
            for ball in front_balls:
                front_counter[ball] = front_counter.get(ball, 0) + 1
            
            for ball in back_balls:
                back_counter[ball] = back_counter.get(ball, 0) + 1
        
        # 归一化频率
        total_front = sum(front_counter.values()) if front_counter else 0
        total_back = sum(back_counter.values()) if back_counter else 0
        
        self.frequency_stats = {
            'front': {ball: count/total_front if total_front > 0 else 0 for ball, count in front_counter.items()},
            'back': {ball: count/total_back if total_back > 0 else 0 for ball, count in back_counter.items()}
        }
        
        # 计算遗漏统计
        front_missing = {i: 0 for i in range(1, 36)}
        back_missing = {i: 0 for i in range(1, 13)}
        
        for i, row in train_data.iterrows():
            front_balls, back_balls = data_manager.parse_balls(row)
            
            for ball in range(1, 36):
                if ball in front_balls:
                    front_missing[ball] = 0
                else:
                    front_missing[ball] += 1
            
            for ball in range(1, 13):
                if ball in back_balls:
                    back_missing[ball] = 0
                else:
                    back_missing[ball] += 1
        
        # 归一化遗漏值（倒数，遗漏越大，值越小）
        max_front_missing = max(front_missing.values()) if front_missing else 1
        max_back_missing = max(back_missing.values()) if back_missing else 1
        
        self.missing_stats = {
            'front': {ball: 1 - missing/max_front_missing if max_front_missing > 0 else 0 for ball, missing in front_missing.items()},
            'back': {ball: 1 - missing/max_back_missing if max_back_missing > 0 else 0 for ball, missing in back_missing.items()}
        }
        
        # 计算冷热统计
        front_avg = np.mean(list(self.frequency_stats['front'].values())) if self.frequency_stats['front'] else 0
        back_avg = np.mean(list(self.frequency_stats['back'].values())) if self.frequency_stats['back'] else 0
        
        self.hot_cold_stats = {
            'front': {ball: 1 if freq > front_avg else 0 for ball, freq in self.frequency_stats['front'].items()},
            'back': {ball: 1 if freq > back_avg else 0 for ball, freq in self.frequency_stats['back'].items()}
        }
    
    def predict(self, data=None):
        """预测下一期号码
        
        Args:
            data: 历史数据（可选）
            
        Returns:
            List[Tuple[List[int], List[int]]]: 预测结果列表
        """
        if self.frequency_stats is None or self.missing_stats is None or self.hot_cold_stats is None:
            return []
        
        # 计算综合得分
        front_scores = {}
        back_scores = {}
        
        for ball in range(1, 36):
            # 频率得分
            freq_score = self.frequency_stats['front'].get(ball, 0)
            # 遗漏得分
            missing_score = self.missing_stats['front'].get(ball, 0)
            # 冷热得分
            hot_cold_score = self.hot_cold_stats['front'].get(ball, 0)
            
            # 综合得分
            front_scores[ball] = self.weight_frequency * freq_score + \
                               self.weight_missing * missing_score + \
                               self.weight_hot_cold * hot_cold_score
        
        for ball in range(1, 13):
            # 频率得分
            freq_score = self.frequency_stats['back'].get(ball, 0)
            # 遗漏得分
            missing_score = self.missing_stats['back'].get(ball, 0)
            # 冷热得分
            hot_cold_score = self.hot_cold_stats['back'].get(ball, 0)
            
            # 综合得分
            back_scores[ball] = self.weight_frequency * freq_score + \
                              self.weight_missing * missing_score + \
                              self.weight_hot_cold * hot_cold_score
        
        # 选择得分最高的号码
        front_sorted = sorted(front_scores.items(), key=lambda x: x[1], reverse=True)
        back_sorted = sorted(back_scores.items(), key=lambda x: x[1], reverse=True)
        
        front_balls = [ball for ball, _ in front_sorted[:5]]
        back_balls = [ball for ball, _ in back_sorted[:2]]
        
        return [(sorted(front_balls), sorted(back_balls))]


if __name__ == "__main__":
    # 测试参数优化
    print("🔍 测试参数优化框架...")
    
    # 定义参数空间
    param_space = {
        'weight_frequency': [0.1, 0.3, 0.5, 0.7, 0.9],
        'weight_missing': [0.1, 0.2, 0.3, 0.4, 0.5],
        'weight_hot_cold': [0.1, 0.2, 0.3, 0.4, 0.5]
    }
    
    # 定义模型创建函数
    def create_model(**params):
        return SimplePredictor(**params)
    
    # 创建优化器
    optimizer = ModelOptimizer(
        model_creator=create_model,
        param_space=param_space,
        optimization_method='grid'
    )
    
    # 执行优化
    print("\n📊 执行网格搜索优化...")
    result = optimizer.optimize(train_periods=300, val_periods=50, metric='accuracy')
    
    # 打印最佳参数
    print(f"\n🏆 最佳参数: {optimizer.best_params}")
    print(f"🎯 最佳得分: {optimizer.best_score:.4f}")
    
    # 可视化优化过程
    print("\n📈 可视化优化过程...")
    optimizer.visualize_optimization_process("output/parameter_optimization.png")
    
    # 可视化参数重要性
    print("\n📊 可视化参数重要性...")
    optimizer.visualize_parameter_importance("output/parameter_importance.png")
    
    # 使用基准测试框架评估最佳模型
    if optimizer.benchmark is not None:
        print("\n🔍 使用基准测试框架评估最佳模型...")
        benchmark_result = optimizer.benchmark_best_model("优化后的简单预测器", test_periods=20)
        
        # 比较基准模型和优化模型
        print("\n🔄 比较基准模型和优化模型...")
        optimizer.benchmark.register_model(
            "基准简单预测器",
            lambda data: SimplePredictor().fit(data) or SimplePredictor().predict(data),
            "未优化的简单预测器",
            "baseline"
        )
        optimizer.benchmark.evaluate_model("基准简单预测器", test_periods=20)
        optimizer.benchmark.compare_models(categories=["baseline", "optimized"])
    
    # 保存优化结果
    print("\n💾 保存优化结果...")
    optimizer.save_results("output/parameter_optimization.json")
    
    print("\n✅ 参数优化测试完成")