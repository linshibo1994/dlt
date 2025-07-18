#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模型优化示例
展示如何使用模型优化器优化预测模型的参数
"""

import os
import sys
import numpy as np
from typing import List, Tuple, Dict
from collections import Counter

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入核心模块
from core_modules import logger_manager, data_manager, cache_manager

# 导入模型优化器
try:
    from improvements.model_optimizer import ModelOptimizer
    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False
    print("⚠️ 模型优化器模块未找到，请确保improvements/model_optimizer.py文件存在")

# 导入模型基准测试框架
try:
    from improvements.model_benchmark import get_model_benchmark
    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False
    print("⚠️ 模型基准测试框架未找到，将跳过基准测试部分")

# 导入预测器
try:
    from predictor_modules import get_traditional_predictor, get_advanced_predictor
    PREDICTORS_AVAILABLE = True
except ImportError:
    PREDICTORS_AVAILABLE = False
    print("⚠️ 预测器模块未找到，将使用模拟预测器")

# 导入增强马尔可夫预测器
try:
    from improvements.enhanced_markov import get_markov_predictor
    ENHANCED_MARKOV_AVAILABLE = True
except ImportError:
    ENHANCED_MARKOV_AVAILABLE = False
    print("⚠️ 增强马尔可夫模块未找到，将跳过相关测试")


class FrequencyPredictor:
    """频率预测器"""
    
    def __init__(self, recent_weight=0.7, historical_weight=0.3, recent_periods=100):
        """初始化频率预测器
        
        Args:
            recent_weight: 近期数据权重
            historical_weight: 历史数据权重
            recent_periods: 近期数据期数
        """
        self.recent_weight = recent_weight
        self.historical_weight = historical_weight
        self.recent_periods = recent_periods
        
        # 统计数据
        self.recent_freq = None
        self.historical_freq = None
    
    def fit(self, train_data):
        """训练模型
        
        Args:
            train_data: 训练数据
        """
        # 计算近期频率
        recent_data = train_data.head(self.recent_periods)
        recent_front_counter = Counter()
        recent_back_counter = Counter()
        
        for _, row in recent_data.iterrows():
            front_balls, back_balls = data_manager.parse_balls(row)
            recent_front_counter.update(front_balls)
            recent_back_counter.update(back_balls)
        
        # 计算历史频率
        historical_data = train_data.iloc[self.recent_periods:]
        historical_front_counter = Counter()
        historical_back_counter = Counter()
        
        for _, row in historical_data.iterrows():
            front_balls, back_balls = data_manager.parse_balls(row)
            historical_front_counter.update(front_balls)
            historical_back_counter.update(back_balls)
        
        # 归一化频率
        total_recent_front = sum(recent_front_counter.values()) if recent_front_counter else 0
        total_recent_back = sum(recent_back_counter.values()) if recent_back_counter else 0
        
        total_historical_front = sum(historical_front_counter.values()) if historical_front_counter else 0
        total_historical_back = sum(historical_back_counter.values()) if historical_back_counter else 0
        
        self.recent_freq = {
            'front': {ball: count/total_recent_front if total_recent_front > 0 else 0 
                     for ball, count in recent_front_counter.items()},
            'back': {ball: count/total_recent_back if total_recent_back > 0 else 0 
                    for ball, count in recent_back_counter.items()}
        }
        
        self.historical_freq = {
            'front': {ball: count/total_historical_front if total_historical_front > 0 else 0 
                     for ball, count in historical_front_counter.items()},
            'back': {ball: count/total_historical_back if total_historical_back > 0 else 0 
                    for ball, count in historical_back_counter.items()}
        }
    
    def predict(self, data=None):
        """预测下一期号码
        
        Args:
            data: 历史数据（可选）
            
        Returns:
            List[Tuple[List[int], List[int]]]: 预测结果列表
        """
        if self.recent_freq is None or self.historical_freq is None:
            return []
        
        # 计算综合频率
        front_scores = {}
        back_scores = {}
        
        for ball in range(1, 36):
            # 近期频率
            recent_score = self.recent_freq['front'].get(ball, 0)
            # 历史频率
            historical_score = self.historical_freq['front'].get(ball, 0)
            
            # 综合得分
            front_scores[ball] = self.recent_weight * recent_score + self.historical_weight * historical_score
        
        for ball in range(1, 13):
            # 近期频率
            recent_score = self.recent_freq['back'].get(ball, 0)
            # 历史频率
            historical_score = self.historical_freq['back'].get(ball, 0)
            
            # 综合得分
            back_scores[ball] = self.recent_weight * recent_score + self.historical_weight * historical_score
        
        # 选择得分最高的号码
        front_sorted = sorted(front_scores.items(), key=lambda x: x[1], reverse=True)
        back_sorted = sorted(back_scores.items(), key=lambda x: x[1], reverse=True)
        
        front_balls = [ball for ball, _ in front_sorted[:5]]
        back_balls = [ball for ball, _ in back_sorted[:2]]
        
        return [(sorted(front_balls), sorted(back_balls))]


class HotColdPredictor:
    """冷热号预测器"""
    
    def __init__(self, hot_threshold=0.6, cold_threshold=0.4, hot_ratio=0.6):
        """初始化冷热号预测器
        
        Args:
            hot_threshold: 热号阈值（高于平均频率的比例）
            cold_threshold: 冷号阈值（低于平均频率的比例）
            hot_ratio: 热号比例（在选择的号码中）
        """
        self.hot_threshold = hot_threshold
        self.cold_threshold = cold_threshold
        self.hot_ratio = hot_ratio
        
        # 统计数据
        self.front_hot = None
        self.front_cold = None
        self.back_hot = None
        self.back_cold = None
    
    def fit(self, train_data):
        """训练模型
        
        Args:
            train_data: 训练数据
        """
        # 计算频率
        front_counter = Counter()
        back_counter = Counter()
        
        for _, row in train_data.iterrows():
            front_balls, back_balls = data_manager.parse_balls(row)
            front_counter.update(front_balls)
            back_counter.update(back_balls)
        
        # 计算平均频率
        front_avg = np.mean(list(front_counter.values())) if front_counter else 0
        back_avg = np.mean(list(back_counter.values())) if back_counter else 0
        
        # 热号（高于平均频率的hot_threshold倍）
        self.front_hot = [ball for ball, count in front_counter.items() if count > front_avg * self.hot_threshold]
        self.back_hot = [ball for ball, count in back_counter.items() if count > back_avg * self.hot_threshold]
        
        # 冷号（低于平均频率的cold_threshold倍）
        self.front_cold = [ball for ball, count in front_counter.items() if count < front_avg * self.cold_threshold]
        self.back_cold = [ball for ball, count in back_counter.items() if count < front_avg * self.cold_threshold]
        
        # 如果热号或冷号不足，补充
        if len(self.front_hot) < 3:
            remaining = [ball for ball in range(1, 36) if ball not in self.front_hot]
            self.front_hot.extend(np.random.choice(remaining, min(3 - len(self.front_hot), len(remaining)), replace=False))
        
        if len(self.front_cold) < 2:
            remaining = [ball for ball in range(1, 36) if ball not in self.front_cold and ball not in self.front_hot]
            self.front_cold.extend(np.random.choice(remaining, min(2 - len(self.front_cold), len(remaining)), replace=False))
        
        if not self.back_hot:
            self.back_hot = list(np.random.choice(range(1, 13), 1, replace=False))
        
        if not self.back_cold:
            remaining = [ball for ball in range(1, 13) if ball not in self.back_hot]
            self.back_cold = list(np.random.choice(remaining, 1, replace=False))
    
    def predict(self, data=None):
        """预测下一期号码
        
        Args:
            data: 历史数据（可选）
            
        Returns:
            List[Tuple[List[int], List[int]]]: 预测结果列表
        """
        if self.front_hot is None or self.front_cold is None or self.back_hot is None or self.back_cold is None:
            return []
        
        # 计算热号数量
        hot_count = int(5 * self.hot_ratio)
        cold_count = 5 - hot_count
        
        # 选择热号
        front_selected = []
        if len(self.front_hot) >= hot_count:
            front_selected.extend(np.random.choice(self.front_hot, hot_count, replace=False))
        else:
            front_selected.extend(self.front_hot)
            remaining = [ball for ball in range(1, 36) if ball not in front_selected]
            front_selected.extend(np.random.choice(remaining, hot_count - len(self.front_hot), replace=False))
        
        # 选择冷号
        if len(self.front_cold) >= cold_count:
            cold_candidates = [ball for ball in self.front_cold if ball not in front_selected]
            if len(cold_candidates) >= cold_count:
                front_selected.extend(np.random.choice(cold_candidates, cold_count, replace=False))
            else:
                front_selected.extend(cold_candidates)
                remaining = [ball for ball in range(1, 36) if ball not in front_selected]
                front_selected.extend(np.random.choice(remaining, cold_count - len(cold_candidates), replace=False))
        else:
            remaining = [ball for ball in range(1, 36) if ball not in front_selected]
            front_selected.extend(np.random.choice(remaining, cold_count, replace=False))
        
        # 选择后区号码
        back_selected = []
        if self.back_hot:
            back_selected.append(np.random.choice(self.back_hot))
        else:
            back_selected.append(np.random.choice(range(1, 13)))
        
        if self.back_cold:
            cold_candidates = [ball for ball in self.back_cold if ball not in back_selected]
            if cold_candidates:
                back_selected.append(np.random.choice(cold_candidates))
            else:
                remaining = [ball for ball in range(1, 13) if ball not in back_selected]
                back_selected.append(np.random.choice(remaining))
        else:
            remaining = [ball for ball in range(1, 13) if ball not in back_selected]
            back_selected.append(np.random.choice(remaining))
        
        return [(sorted(front_selected), sorted(back_selected))]


class MarkovPredictor:
    """马尔可夫预测器"""
    
    def __init__(self, order=1, transition_weight=0.7, frequency_weight=0.3):
        """初始化马尔可夫预测器
        
        Args:
            order: 马尔可夫链阶数
            transition_weight: 转移概率权重
            frequency_weight: 频率权重
        """
        self.order = order
        self.transition_weight = transition_weight
        self.frequency_weight = frequency_weight
        
        # 统计数据
        self.front_transitions = None
        self.back_transitions = None
        self.front_freq = None
        self.back_freq = None
    
    def fit(self, train_data):
        """训练模型
        
        Args:
            train_data: 训练数据
        """
        # 计算频率
        front_counter = Counter()
        back_counter = Counter()
        
        for _, row in train_data.iterrows():
            front_balls, back_balls = data_manager.parse_balls(row)
            front_counter.update(front_balls)
            back_counter.update(back_balls)
        
        # 归一化频率
        total_front = sum(front_counter.values()) if front_counter else 0
        total_back = sum(back_counter.values()) if back_counter else 0
        
        self.front_freq = {ball: count/total_front if total_front > 0 else 0 
                          for ball, count in front_counter.items()}
        self.back_freq = {ball: count/total_back if total_back > 0 else 0 
                         for ball, count in back_counter.items()}
        
        # 计算转移概率
        self.front_transitions = {}
        self.back_transitions = {}
        
        # 对于n阶马尔可夫链，我们需要n个连续的状态作为条件
        for i in range(len(train_data) - self.order):
            # 构建条件状态（前n期的号码）
            condition_front = []
            condition_back = []
            
            for j in range(self.order):
                front, back = data_manager.parse_balls(train_data.iloc[i + j])
                condition_front.extend(front)
                condition_back.extend(back)
            
            # 获取下一期的号码（要预测的状态）
            next_front, next_back = data_manager.parse_balls(train_data.iloc[i + self.order])
            
            # 将条件状态转换为元组（作为字典键）
            condition_front_tuple = tuple(sorted(condition_front))
            condition_back_tuple = tuple(sorted(condition_back))
            
            # 更新转移计数
            if condition_front_tuple not in self.front_transitions:
                self.front_transitions[condition_front_tuple] = Counter()
            if condition_back_tuple not in self.back_transitions:
                self.back_transitions[condition_back_tuple] = Counter()
            
            for next_ball in next_front:
                self.front_transitions[condition_front_tuple][next_ball] += 1
            
            for next_ball in next_back:
                self.back_transitions[condition_back_tuple][next_ball] += 1
        
        # 转换为概率
        for condition, counter in self.front_transitions.items():
            total = sum(counter.values())
            if total > 0:
                self.front_transitions[condition] = {ball: count/total for ball, count in counter.items()}
        
        for condition, counter in self.back_transitions.items():
            total = sum(counter.values())
            if total > 0:
                self.back_transitions[condition] = {ball: count/total for ball, count in counter.items()}
    
    def predict(self, data=None):
        """预测下一期号码
        
        Args:
            data: 历史数据（可选）
            
        Returns:
            List[Tuple[List[int], List[int]]]: 预测结果列表
        """
        if self.front_transitions is None or self.back_transitions is None:
            return []
        
        # 获取最近n期的号码作为条件状态
        condition_front = []
        condition_back = []
        
        if data is not None and len(data) >= self.order:
            for i in range(self.order):
                front, back = data_manager.parse_balls(data.iloc[i])
                condition_front.extend(front)
                condition_back.extend(back)
        
        # 如果数据不足，使用默认值
        if not condition_front:
            condition_front = list(range(1, 6))
        if not condition_back:
            condition_back = [1, 2]
        
        # 将条件状态转换为元组
        condition_front_tuple = tuple(sorted(condition_front))
        condition_back_tuple = tuple(sorted(condition_back))
        
        # 计算综合得分
        front_scores = {}
        back_scores = {}
        
        for ball in range(1, 36):
            # 转移概率
            trans_prob = 0
            if condition_front_tuple in self.front_transitions and ball in self.front_transitions[condition_front_tuple]:
                trans_prob = self.front_transitions[condition_front_tuple][ball]
            
            # 频率
            freq = self.front_freq.get(ball, 0)
            
            # 综合得分
            front_scores[ball] = self.transition_weight * trans_prob + self.frequency_weight * freq
        
        for ball in range(1, 13):
            # 转移概率
            trans_prob = 0
            if condition_back_tuple in self.back_transitions and ball in self.back_transitions[condition_back_tuple]:
                trans_prob = self.back_transitions[condition_back_tuple][ball]
            
            # 频率
            freq = self.back_freq.get(ball, 0)
            
            # 综合得分
            back_scores[ball] = self.transition_weight * trans_prob + self.frequency_weight * freq
        
        # 选择得分最高的号码
        front_sorted = sorted(front_scores.items(), key=lambda x: x[1], reverse=True)
        back_sorted = sorted(back_scores.items(), key=lambda x: x[1], reverse=True)
        
        front_balls = [ball for ball, _ in front_sorted[:5]]
        back_balls = [ball for ball, _ in back_sorted[:2]]
        
        return [(sorted(front_balls), sorted(back_balls))]


def optimize_frequency_predictor():
    """优化频率预测器参数"""
    if not OPTIMIZER_AVAILABLE:
        print("⚠️ 模型优化器不可用，跳过频率预测器优化")
        return None
    
    print("\n🔍 开始优化频率预测器参数...")
    
    # 定义参数空间
    param_space = {
        'recent_weight': [0.5, 0.6, 0.7, 0.8, 0.9],
        'historical_weight': [0.1, 0.2, 0.3, 0.4, 0.5],
        'recent_periods': [50, 100, 150, 200, 300]
    }
    
    # 定义模型创建函数
    def create_model(**params):
        return FrequencyPredictor(**params)
    
    # 创建优化器
    optimizer = ModelOptimizer(
        model_creator=create_model,
        param_space=param_space,
        optimization_method='grid'
    )
    
    # 执行优化
    result = optimizer.optimize(train_periods=300, val_periods=50, metric='accuracy', verbose=True)
    
    # 打印最佳参数
    print(f"\n🏆 频率预测器最佳参数: {optimizer.best_params}")
    print(f"🎯 最佳得分: {optimizer.best_score:.4f}")
    
    # 可视化优化过程
    optimizer.visualize_optimization_process("output/frequency_optimization.png")
    
    # 可视化参数重要性
    optimizer.visualize_parameter_importance("output/frequency_parameter_importance.png")
    
    # 保存优化结果
    optimizer.save_results("output/frequency_optimization.json")
    
    return optimizer


def optimize_hot_cold_predictor():
    """优化冷热号预测器参数"""
    if not OPTIMIZER_AVAILABLE:
        print("⚠️ 模型优化器不可用，跳过冷热号预测器优化")
        return None
    
    print("\n🔍 开始优化冷热号预测器参数...")
    
    # 定义参数空间
    param_space = {
        'hot_threshold': [0.5, 0.6, 0.7, 0.8, 0.9],
        'cold_threshold': [0.2, 0.3, 0.4, 0.5, 0.6],
        'hot_ratio': [0.4, 0.5, 0.6, 0.7, 0.8]
    }
    
    # 定义模型创建函数
    def create_model(**params):
        return HotColdPredictor(**params)
    
    # 创建优化器
    optimizer = ModelOptimizer(
        model_creator=create_model,
        param_space=param_space,
        optimization_method='grid'
    )
    
    # 执行优化
    result = optimizer.optimize(train_periods=300, val_periods=50, metric='accuracy', verbose=True)
    
    # 打印最佳参数
    print(f"\n🏆 冷热号预测器最佳参数: {optimizer.best_params}")
    print(f"🎯 最佳得分: {optimizer.best_score:.4f}")
    
    # 可视化优化过程
    optimizer.visualize_optimization_process("output/hot_cold_optimization.png")
    
    # 可视化参数重要性
    optimizer.visualize_parameter_importance("output/hot_cold_parameter_importance.png")
    
    # 保存优化结果
    optimizer.save_results("output/hot_cold_optimization.json")
    
    return optimizer


def optimize_markov_predictor():
    """优化马尔可夫预测器参数"""
    if not OPTIMIZER_AVAILABLE:
        print("⚠️ 模型优化器不可用，跳过马尔可夫预测器优化")
        return None
    
    print("\n🔍 开始优化马尔可夫预测器参数...")
    
    # 定义参数空间
    param_space = {
        'order': [1, 2, 3],
        'transition_weight': [0.5, 0.6, 0.7, 0.8, 0.9],
        'frequency_weight': [0.1, 0.2, 0.3, 0.4, 0.5]
    }
    
    # 定义模型创建函数
    def create_model(**params):
        return MarkovPredictor(**params)
    
    # 创建优化器
    optimizer = ModelOptimizer(
        model_creator=create_model,
        param_space=param_space,
        optimization_method='grid'
    )
    
    # 执行优化
    result = optimizer.optimize(train_periods=300, val_periods=50, metric='accuracy', verbose=True)
    
    # 打印最佳参数
    print(f"\n🏆 马尔可夫预测器最佳参数: {optimizer.best_params}")
    print(f"🎯 最佳得分: {optimizer.best_score:.4f}")
    
    # 可视化优化过程
    optimizer.visualize_optimization_process("output/markov_optimization.png")
    
    # 可视化参数重要性
    optimizer.visualize_parameter_importance("output/markov_parameter_importance.png")
    
    # 保存优化结果
    optimizer.save_results("output/markov_optimization.json")
    
    return optimizer


def benchmark_optimized_models(freq_optimizer, hot_cold_optimizer, markov_optimizer):
    """使用基准测试框架评估优化后的模型"""
    if not BENCHMARK_AVAILABLE:
        print("⚠️ 模型基准测试框架不可用，跳过基准测试")
        return
    
    print("\n🔍 使用基准测试框架评估优化后的模型...")
    
    # 获取基准测试实例
    benchmark = get_model_benchmark()
    
    # 注册基准模型
    benchmark.register_model(
        "基准频率预测器",
        lambda data: FrequencyPredictor().fit(data) or FrequencyPredictor().predict(data),
        "未优化的频率预测器",
        "baseline"
    )
    
    benchmark.register_model(
        "基准冷热号预测器",
        lambda data: HotColdPredictor().fit(data) or HotColdPredictor().predict(data),
        "未优化的冷热号预测器",
        "baseline"
    )
    
    benchmark.register_model(
        "基准马尔可夫预测器",
        lambda data: MarkovPredictor().fit(data) or MarkovPredictor().predict(data),
        "未优化的马尔可夫预测器",
        "baseline"
    )
    
    # 注册优化后的模型
    if freq_optimizer and freq_optimizer.best_model:
        benchmark.register_model(
            "优化后的频率预测器",
            lambda data: freq_optimizer.best_model.predict(data),
            f"优化后的频率预测器 (参数: {freq_optimizer.best_params})",
            "optimized"
        )
    
    if hot_cold_optimizer and hot_cold_optimizer.best_model:
        benchmark.register_model(
            "优化后的冷热号预测器",
            lambda data: hot_cold_optimizer.best_model.predict(data),
            f"优化后的冷热号预测器 (参数: {hot_cold_optimizer.best_params})",
            "optimized"
        )
    
    if markov_optimizer and markov_optimizer.best_model:
        benchmark.register_model(
            "优化后的马尔可夫预测器",
            lambda data: markov_optimizer.best_model.predict(data),
            f"优化后的马尔可夫预测器 (参数: {markov_optimizer.best_params})",
            "optimized"
        )
    
    # 评估所有模型
    test_periods = 20
    print(f"\n📊 评估所有模型 (测试期数: {test_periods})...")
    benchmark.evaluate_all_models(test_periods=test_periods, verbose=True)
    
    # 比较模型
    print("\n🔄 比较基准模型和优化模型...")
    benchmark.compare_models(categories=["baseline", "optimized"], verbose=True)
    
    # 生成报告
    print("\n📝 生成评估报告...")
    report_path = "output/optimized_models_report.md"
    benchmark.generate_report(report_path)
    print(f"✅ 评估报告已保存到: {report_path}")
    
    # 可视化比较
    print("\n📈 可视化模型比较...")
    comparison_path = "output/optimized_models_comparison.png"
    benchmark.visualize_comparison(output_path=comparison_path)
    print(f"✅ 比较图表已保存到: {comparison_path}")
    
    # 可视化最佳模型性能
    if benchmark.comparison and 'overall_ranking' in benchmark.comparison:
        best_model = next(iter(benchmark.comparison['overall_ranking'].keys()), None)
        if best_model:
            print(f"\n📊 可视化最佳模型 ({best_model}) 性能...")
            performance_path = f"output/{best_model}_performance.png"
            benchmark.visualize_model_performance(best_model, output_path=performance_path)
            print(f"✅ 性能图表已保存到: {performance_path}")


def main():
    """主函数"""
    print("🚀 开始模型优化测试...")
    
    # 确保输出目录存在
    os.makedirs("output", exist_ok=True)
    
    # 优化频率预测器
    freq_optimizer = optimize_frequency_predictor()
    
    # 优化冷热号预测器
    hot_cold_optimizer = optimize_hot_cold_predictor()
    
    # 优化马尔可夫预测器
    markov_optimizer = optimize_markov_predictor()
    
    # 使用基准测试框架评估优化后的模型
    benchmark_optimized_models(freq_optimizer, hot_cold_optimizer, markov_optimizer)
    
    print("\n✅ 模型优化测试完成")


if __name__ == "__main__":
    main()