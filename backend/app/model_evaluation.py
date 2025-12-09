#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模型评估框架
提供统一的模型评估、比较和优化功能
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Any, Callable, Union
from datetime import datetime
from collections import defaultdict
import json
import time
from tqdm import tqdm

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
    BENCHMARK_AVAILABLE = True
except ImportError:
    logger_manager.error("模型基准测试框架未找到，请确保improvements/model_benchmark.py文件存在")
    BENCHMARK_AVAILABLE = False
    ModelBenchmark = None
    get_model_benchmark = None

# 尝试导入模型优化器
try:
    from improvements.model_optimizer import ModelOptimizer
    OPTIMIZER_AVAILABLE = True
except ImportError:
    logger_manager.error("模型优化器未找到，请确保improvements/model_optimizer.py文件存在")
    OPTIMIZER_AVAILABLE = False
    ModelOptimizer = None


class ModelEvaluator:
    """模型评估框架"""
    
    def __init__(self):
        """初始化模型评估框架"""
        self.df = data_manager.get_data()
        if self.df is None:
            logger_manager.error("数据未加载")
        
        # 获取基准测试实例
        if BENCHMARK_AVAILABLE:
            self.benchmark = get_model_benchmark()
        else:
            self.benchmark = None
            logger_manager.error("模型基准测试框架未找到，基准测试功能将不可用")
        
        # 优化器字典
        self.optimizers = {}
        
        # 评估结果
        self.evaluation_results = {}
    
    def register_model(self, model_name: str, predict_func: Callable, 
                      description: str = "", category: str = "traditional"):
        """注册模型
        
        Args:
            model_name: 模型名称
            predict_func: 预测函数，接受期数参数，返回预测结果
            description: 模型描述
            category: 模型类别
        """
        if self.benchmark is None:
            logger_manager.error("模型基准测试框架未找到，无法注册模型")
            return False
        
        self.benchmark.register_model(model_name, predict_func, description, category)
        return True
    
    def evaluate_model(self, model_name: str, test_periods: int = 50, 
                      bet_cost: float = 2.0, verbose: bool = True) -> Dict:
        """评估单个模型
        
        Args:
            model_name: 模型名称
            test_periods: 测试期数
            bet_cost: 每注投注成本
            verbose: 是否显示详细信息
            
        Returns:
            Dict: 评估结果
        """
        if self.benchmark is None:
            logger_manager.error("模型基准测试框架未找到，无法评估模型")
            return {}
        
        result = self.benchmark.evaluate_model(model_name, test_periods, bet_cost, verbose)
        self.evaluation_results[model_name] = result
        return result
    
    def evaluate_all_models(self, test_periods: int = 50, bet_cost: float = 2.0, verbose: bool = True):
        """评估所有注册的模型
        
        Args:
            test_periods: 测试期数
            bet_cost: 每注投注成本
            verbose: 是否显示详细信息
        """
        if self.benchmark is None:
            logger_manager.error("模型基准测试框架未找到，无法评估模型")
            return
        
        self.benchmark.evaluate_all_models(test_periods, bet_cost, verbose)
        
        # 更新评估结果
        for model_name, model_info in self.benchmark.results.items():
            if model_info.get('evaluated', False):
                self.evaluation_results[model_name] = model_info
    
    def compare_models(self, metrics: List[str] = None, categories: List[str] = None, 
                      verbose: bool = True) -> Dict:
        """比较模型性能
        
        Args:
            metrics: 要比较的指标列表，默认为所有指标
            categories: 要比较的模型类别，默认为所有类别
            verbose: 是否显示详细信息
            
        Returns:
            Dict: 比较结果
        """
        if self.benchmark is None:
            logger_manager.error("模型基准测试框架未找到，无法比较模型")
            return {}
        
        return self.benchmark.compare_models(metrics, categories, verbose)
    
    def optimize_model(self, model_name: str, model_creator: Callable, param_space: Dict,
                      optimization_method: str = 'grid', n_iter: int = 50,
                      train_periods: int = 500, val_periods: int = 50,
                      metric: str = 'accuracy', verbose: bool = True) -> Dict:
        """优化模型参数
        
        Args:
            model_name: 模型名称
            model_creator: 模型创建函数，接受参数字典，返回模型实例
            param_space: 参数空间，字典形式，键为参数名，值为参数可能的取值列表
            optimization_method: 优化方法，可选值: 'grid', 'random', 'bayesian'
            n_iter: 随机搜索或贝叶斯优化的迭代次数
            train_periods: 训练数据期数
            val_periods: 验证数据期数
            metric: 优化指标，可选值: 'accuracy', 'hit_rate', 'roi'
            verbose: 是否显示详细信息
            
        Returns:
            Dict: 优化结果
        """
        if not OPTIMIZER_AVAILABLE:
            logger_manager.error("模型优化器未找到，无法优化模型")
            return {}
        
        # 创建优化器
        optimizer = ModelOptimizer(
            model_creator=model_creator,
            param_space=param_space,
            optimization_method=optimization_method,
            n_iter=n_iter
        )
        
        # 执行优化
        result = optimizer.optimize(train_periods, val_periods, metric, verbose)
        
        # 保存优化器
        self.optimizers[model_name] = optimizer
        
        # 注册优化后的模型
        if optimizer.best_model:
            optimized_model_name = f"优化后的{model_name}"
            self.register_model(
                optimized_model_name,
                lambda data: optimizer.best_model.predict(data) if hasattr(optimizer.best_model, 'predict') else optimizer.best_model.predict_next(data),
                f"优化后的{model_name} (参数: {optimizer.best_params})",
                "optimized"
            )
        
        return result
    
    def visualize_optimization(self, model_name: str, output_dir: str = "output"):
        """可视化优化过程
        
        Args:
            model_name: 模型名称
            output_dir: 输出目录
        """
        if model_name not in self.optimizers:
            logger_manager.error(f"未找到模型 {model_name} 的优化器")
            return
        
        optimizer = self.optimizers[model_name]
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 可视化优化过程
        process_path = os.path.join(output_dir, f"{model_name}_optimization_process.png")
        optimizer.visualize_optimization_process(process_path)
        
        # 可视化参数重要性
        importance_path = os.path.join(output_dir, f"{model_name}_parameter_importance.png")
        optimizer.visualize_parameter_importance(importance_path)
    
    def generate_report(self, output_path=None) -> str:
        """生成评估报告
        
        Args:
            output_path: 报告输出路径，如果为None则返回报告内容
            
        Returns:
            str: 报告内容
        """
        if self.benchmark is None:
            logger_manager.error("模型基准测试框架未找到，无法生成报告")
            return "模型基准测试框架未找到，无法生成报告"
        
        return self.benchmark.generate_report(output_path)
    
    def visualize_comparison(self, metrics=None, output_path=None, figsize=(12, 10)):
        """可视化模型比较结果
        
        Args:
            metrics: 要可视化的指标列表，默认为所有指标
            output_path: 图表输出路径
            figsize: 图表大小
        """
        if self.benchmark is None:
            logger_manager.error("模型基准测试框架未找到，无法可视化比较")
            return
        
        self.benchmark.visualize_comparison(metrics, output_path, figsize)
    
    def visualize_model_performance(self, model_name, output_path=None):
        """可视化单个模型的性能
        
        Args:
            model_name: 模型名称
            output_path: 图表输出路径
        """
        if self.benchmark is None:
            logger_manager.error("模型基准测试框架未找到，无法可视化模型性能")
            return
        
        self.benchmark.visualize_model_performance(model_name, output_path)
    
    def save_results(self, output_dir="output"):
        """保存所有评估和优化结果
        
        Args:
            output_dir: 输出目录
        """
        try:
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存评估结果
            if self.benchmark:
                benchmark_path = os.path.join(output_dir, "model_benchmark_results.json")
                self.benchmark.save_results(benchmark_path)
            
            # 保存优化结果
            for model_name, optimizer in self.optimizers.items():
                optimizer_path = os.path.join(output_dir, f"{model_name}_optimization.json")
                optimizer.save_results(optimizer_path)
            
            logger_manager.info(f"所有结果已保存到 {output_dir} 目录")
            
        except Exception as e:
            logger_manager.error(f"保存结果失败: {e}")
    
    def load_results(self, benchmark_path=None, optimizer_paths=None):
        """加载评估和优化结果
        
        Args:
            benchmark_path: 基准测试结果路径
            optimizer_paths: 优化器结果路径字典，键为模型名称，值为路径
            
        Returns:
            bool: 是否加载成功
        """
        success = True
        
        # 加载基准测试结果
        if benchmark_path and self.benchmark:
            if not self.benchmark.load_results(benchmark_path):
                logger_manager.error(f"加载基准测试结果失败: {benchmark_path}")
                success = False
        
        # 加载优化器结果
        if optimizer_paths:
            for model_name, path in optimizer_paths.items():
                if not OPTIMIZER_AVAILABLE:
                    logger_manager.error("模型优化器未找到，无法加载优化结果")
                    success = False
                    continue
                
                # 创建空优化器
                optimizer = ModelOptimizer(lambda: None, {})
                
                # 加载结果
                if optimizer.load_results(path):
                    self.optimizers[model_name] = optimizer
                else:
                    logger_manager.error(f"加载优化器结果失败: {path}")
                    success = False
        
        return success


# 全局实例
_model_evaluator = None

def get_model_evaluator() -> ModelEvaluator:
    """获取模型评估器实例"""
    global _model_evaluator
    if _model_evaluator is None:
        _model_evaluator = ModelEvaluator()
    return _model_evaluator


if __name__ == "__main__":
    # 测试模型评估框架
    print("🔍 测试模型评估框架...")
    evaluator = get_model_evaluator()
    
    # 模拟预测函数
    def mock_random_predict(historical_data):
        """随机预测"""
        return [(sorted(np.random.choice(range(1, 36), 5, replace=False)), 
                sorted(np.random.choice(range(1, 13), 2, replace=False))) for _ in range(3)]
    
    def mock_frequency_predict(historical_data):
        """频率预测"""
        # 简化版频率预测
        from collections import Counter
        
        front_counter = Counter()
        back_counter = Counter()
        
        for _, row in historical_data.iterrows():
            front_balls, back_balls = data_manager.parse_balls(row)
            front_counter.update(front_balls)
            back_counter.update(back_balls)
        
        front_most_common = [ball for ball, _ in front_counter.most_common(5)]
        back_most_common = [ball for ball, _ in back_counter.most_common(2)]
        
        return [(sorted(front_most_common), sorted(back_most_common)) for _ in range(3)]
    
    # 注册模型
    print("\n📝 注册测试模型...")
    evaluator.register_model("随机预测", mock_random_predict, "完全随机的预测方法", "baseline")
    evaluator.register_model("频率预测", mock_frequency_predict, "基于历史频率的预测方法", "traditional")
    
    # 评估模型
    print("\n📊 评估模型...")
    evaluator.evaluate_all_models(test_periods=20, verbose=True)
    
    # 比较模型
    print("\n🔄 比较模型...")
    evaluator.compare_models(verbose=True)
    
    # 定义简单模型类
    class SimplePredictor:
        """简单预测模型，用于测试参数优化"""
        
        def __init__(self, weight_frequency=0.5, weight_missing=0.3):
            self.weight_frequency = weight_frequency
            self.weight_missing = weight_missing
            self.frequency_stats = None
            self.missing_stats = None
        
        def fit(self, train_data):
            # 简化版训练
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
            
            # 归一化遗漏值
            max_front_missing = max(front_missing.values()) if front_missing else 1
            max_back_missing = max(back_missing.values()) if back_missing else 1
            
            self.missing_stats = {
                'front': {ball: missing/max_front_missing if max_front_missing > 0 else 0 for ball, missing in front_missing.items()},
                'back': {ball: missing/max_back_missing if max_back_missing > 0 else 0 for ball, missing in back_missing.items()}
            }
        
        def predict(self, data=None):
            if self.frequency_stats is None or self.missing_stats is None:
                return []
            
            # 计算综合得分
            front_scores = {}
            back_scores = {}
            
            for ball in range(1, 36):
                # 频率得分
                freq_score = self.frequency_stats['front'].get(ball, 0)
                # 遗漏得分
                missing_score = self.missing_stats['front'].get(ball, 0)
                
                # 综合得分
                front_scores[ball] = self.weight_frequency * freq_score + self.weight_missing * missing_score
            
            for ball in range(1, 13):
                # 频率得分
                freq_score = self.frequency_stats['back'].get(ball, 0)
                # 遗漏得分
                missing_score = self.missing_stats['back'].get(ball, 0)
                
                # 综合得分
                back_scores[ball] = self.weight_frequency * freq_score + self.weight_missing * missing_score
            
            # 选择得分最高的号码
            front_sorted = sorted(front_scores.items(), key=lambda x: x[1], reverse=True)
            back_sorted = sorted(back_scores.items(), key=lambda x: x[1], reverse=True)
            
            front_balls = [ball for ball, _ in front_sorted[:5]]
            back_balls = [ball for ball, _ in back_sorted[:2]]
            
            return [(sorted(front_balls), sorted(back_balls))]
    
    # 优化模型参数
    if OPTIMIZER_AVAILABLE:
        print("\n🔧 优化模型参数...")
        
        # 定义参数空间
        param_space = {
            'weight_frequency': [0.3, 0.5, 0.7, 0.9],
            'weight_missing': [0.1, 0.3, 0.5, 0.7]
        }
        
        # 定义模型创建函数
        def create_model(**params):
            return SimplePredictor(**params)
        
        # 执行优化
        evaluator.optimize_model(
            model_name="简单预测器",
            model_creator=create_model,
            param_space=param_space,
            optimization_method='grid',
            train_periods=200,
            val_periods=30,
            metric='accuracy',
            verbose=True
        )
        
        # 可视化优化过程
        evaluator.visualize_optimization("简单预测器")
        
        # 评估优化后的模型
        evaluator.evaluate_model("优化后的简单预测器", test_periods=20)
        
        # 比较优化前后的模型
        evaluator.compare_models(categories=["traditional", "optimized"])
    
    # 生成报告
    print("\n📝 生成评估报告...")
    report = evaluator.generate_report("output/model_evaluation_report.md")
    
    # 可视化比较
    print("\n📈 可视化模型比较...")
    evaluator.visualize_comparison(output_path="output/model_evaluation_comparison.png")
    
    # 保存结果
    print("\n💾 保存评估结果...")
    evaluator.save_results("output")
    
    print("\n✅ 模型评估框架测试完成")