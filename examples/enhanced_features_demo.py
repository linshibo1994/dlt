#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
增强特性演示
展示如何使用模型评估框架的增强特性
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入核心模块
from core_modules import logger_manager, data_manager, cache_manager

# 导入模型评估框架
try:
    from improvements.model_evaluation import get_model_evaluator
    EVALUATOR_AVAILABLE = True
except ImportError:
    EVALUATOR_AVAILABLE = False
    print("⚠️ 模型评估框架未找到，请确保improvements/model_evaluation.py文件存在")

# 导入模型基准测试框架
try:
    from improvements.model_benchmark import get_model_benchmark
    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False
    print("⚠️ 模型基准测试框架未找到，请确保improvements/model_benchmark.py文件存在")

# 导入模型优化器
try:
    from improvements.model_optimizer import ModelOptimizer
    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False
    print("⚠️ 模型优化器未找到，请确保improvements/model_optimizer.py文件存在")

# 导入增强马尔可夫预测器
try:
    from improvements.enhanced_markov import get_markov_predictor
    ENHANCED_MARKOV_AVAILABLE = True
except ImportError:
    ENHANCED_MARKOV_AVAILABLE = False
    print("⚠️ 增强马尔可夫模块未找到，将跳过相关测试")

# 导入LSTM预测器
try:
    from advanced_lstm_predictor import AdvancedLSTMPredictor, TENSORFLOW_AVAILABLE
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ LSTM预测器模块未找到，将跳过相关测试")


def demo_model_benchmark():
    """演示模型基准测试框架"""
    if not BENCHMARK_AVAILABLE:
        print("❌ 模型基准测试框架未找到，无法演示")
        return
    
    print("\n🔍 演示模型基准测试框架...")
    
    # 获取基准测试实例
    benchmark = get_model_benchmark()
    
    # 注册模拟预测器
    def mock_random_predict(historical_data):
        """随机预测"""
        return [(sorted(np.random.choice(range(1, 36), 5, replace=False)), 
                sorted(np.random.choice(range(1, 13), 2, replace=False))) for _ in range(3)]
    
    def mock_frequency_predict(historical_data):
        """频率预测"""
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
    print("📝 注册测试模型...")
    benchmark.register_model("随机预测", mock_random_predict, "完全随机的预测方法", "baseline")
    benchmark.register_model("频率预测", mock_frequency_predict, "基于历史频率的预测方法", "traditional")
    
    # 评估模型
    print("📊 评估模型...")
    benchmark.evaluate_all_models(test_periods=10, verbose=True)
    
    # 比较模型
    print("🔄 比较模型...")
    benchmark.compare_models(verbose=True)
    
    # 生成报告
    print("📝 生成评估报告...")
    os.makedirs("output", exist_ok=True)
    report = benchmark.generate_report("output/benchmark_demo_report.md")
    
    # 可视化比较
    print("📈 可视化模型比较...")
    benchmark.visualize_comparison(output_path="output/benchmark_demo_comparison.png")
    
    # 可视化模型性能
    print("📊 可视化模型性能...")
    benchmark.visualize_model_performance("频率预测", output_path="output/benchmark_demo_performance.png")
    
    print("✅ 模型基准测试框架演示完成")


def demo_model_optimizer():
    """演示模型优化器"""
    if not OPTIMIZER_AVAILABLE:
        print("❌ 模型优化器未找到，无法演示")
        return
    
    print("\n🔧 演示模型优化器...")
    
    # 定义简单模型类
    class SimplePredictor:
        """简单预测模型，用于测试参数优化"""
        
        def __init__(self, weight_frequency=0.5, weight_missing=0.3):
            self.weight_frequency = weight_frequency
            self.weight_missing = weight_missing
            self.frequency_stats = None
            self.missing_stats = None
        
        def fit(self, train_data):
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
    
    # 定义参数空间
    param_space = {
        'weight_frequency': [0.3, 0.5, 0.7, 0.9],
        'weight_missing': [0.1, 0.3, 0.5, 0.7]
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
    print("📊 执行网格搜索优化...")
    result = optimizer.optimize(train_periods=100, val_periods=20, metric='accuracy', verbose=True)
    
    # 打印最佳参数
    print(f"🏆 最佳参数: {optimizer.best_params}")
    print(f"🎯 最佳得分: {optimizer.best_score:.4f}")
    
    # 可视化优化过程
    print("📈 可视化优化过程...")
    os.makedirs("output", exist_ok=True)
    optimizer.visualize_optimization_process("output/optimizer_demo_process.png")
    
    # 可视化参数重要性
    print("📊 可视化参数重要性...")
    optimizer.visualize_parameter_importance("output/optimizer_demo_importance.png")
    
    print("✅ 模型优化器演示完成")


def demo_enhanced_markov():
    """演示增强马尔可夫链"""
    if not ENHANCED_MARKOV_AVAILABLE:
        print("❌ 增强马尔可夫模块未找到，无法演示")
        return
    
    print("\n⛓️ 演示增强马尔可夫链...")
    
    # 获取马尔可夫预测器
    predictor = get_markov_predictor()
    
    # 测试二阶马尔可夫链
    print("📊 测试二阶马尔可夫链...")
    results = predictor.multi_order_markov_predict(count=3, periods=300, order=2)
    print("✅ 二阶马尔可夫链测试成功")
    for i, (front, back) in enumerate(results):
        print(f"  第 {i+1} 注: {sorted(front)} + {sorted(back)}")
    
    # 测试三阶马尔可夫链
    print("\n📊 测试三阶马尔可夫链...")
    results = predictor.multi_order_markov_predict(count=3, periods=300, order=3)
    print("✅ 三阶马尔可夫链测试成功")
    for i, (front, back) in enumerate(results):
        print(f"  第 {i+1} 注: {sorted(front)} + {sorted(back)}")
    
    # 测试自适应马尔可夫链
    print("\n📊 测试自适应马尔可夫链...")
    results = predictor.adaptive_order_markov_predict(count=3, periods=300)
    print("✅ 自适应马尔可夫链测试成功")
    for i, pred in enumerate(results):
        print(f"  第 {i+1} 注: {sorted(pred['front_balls'])} + {sorted(pred['back_balls'])}")
        print(f"  阶数权重: {pred['order_weights']}")
    
    print("✅ 增强马尔可夫链演示完成")


def demo_lstm_predictor():
    """演示LSTM预测器"""
    if not TENSORFLOW_AVAILABLE:
        print("❌ TensorFlow未安装，无法演示LSTM预测器")
        return
    
    print("\n🧠 演示LSTM预测器...")
    
    try:
        # 创建LSTM预测器
        predictor = AdvancedLSTMPredictor()
        
        # 测试LSTM预测
        print("📊 测试LSTM预测...")
        results = predictor.lstm_predict(count=3)
        print("✅ LSTM预测测试成功")
        for i, (front, back) in enumerate(results):
            print(f"  第 {i+1} 注: {sorted(front)} + {sorted(back)}")
        
        print("✅ LSTM预测器演示完成")
    except Exception as e:
        print(f"❌ LSTM预测器演示失败: {e}")


def demo_model_evaluation():
    """演示模型评估框架"""
    if not EVALUATOR_AVAILABLE:
        print("❌ 模型评估框架未找到，无法演示")
        return
    
    print("\n🔍 演示模型评估框架...")
    
    # 获取模型评估器实例
    evaluator = get_model_evaluator()
    
    # 注册模拟预测器
    def mock_random_predict(historical_data):
        """随机预测"""
        return [(sorted(np.random.choice(range(1, 36), 5, replace=False)), 
                sorted(np.random.choice(range(1, 13), 2, replace=False))) for _ in range(3)]
    
    def mock_frequency_predict(historical_data):
        """频率预测"""
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
    print("📝 注册测试模型...")
    evaluator.register_model("随机预测", mock_random_predict, "完全随机的预测方法", "baseline")
    evaluator.register_model("频率预测", mock_frequency_predict, "基于历史频率的预测方法", "traditional")
    
    # 评估模型
    print("📊 评估模型...")
    evaluator.evaluate_all_models(test_periods=10, verbose=True)
    
    # 比较模型
    print("🔄 比较模型...")
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
        print("🔧 优化模型参数...")
        
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
            train_periods=100,
            val_periods=20,
            metric='accuracy',
            verbose=True
        )
        
        # 可视化优化过程
        os.makedirs("output", exist_ok=True)
        evaluator.visualize_optimization("简单预测器", "output")
        
        # 评估优化后的模型
        evaluator.evaluate_model("优化后的简单预测器", test_periods=10)
        
        # 比较优化前后的模型
        evaluator.compare_models(categories=["traditional", "optimized"])
    
    # 生成报告
    print("📝 生成评估报告...")
    os.makedirs("output", exist_ok=True)
    report = evaluator.generate_report("output/evaluation_demo_report.md")
    
    # 可视化比较
    print("📈 可视化模型比较...")
    evaluator.visualize_comparison(output_path="output/evaluation_demo_comparison.png")
    
    print("✅ 模型评估框架演示完成")


def main():
    """主函数"""
    print("🚀 开始增强特性演示...")
    
    # 确保输出目录存在
    os.makedirs("output", exist_ok=True)
    
    # 演示模型基准测试框架
    demo_model_benchmark()
    
    # 演示模型优化器
    demo_model_optimizer()
    
    # 演示增强马尔可夫链
    demo_enhanced_markov()
    
    # 演示LSTM预测器
    demo_lstm_predictor()
    
    # 演示模型评估框架
    demo_model_evaluation()
    
    print("\n✅ 增强特性演示完成")


if __name__ == "__main__":
    main()