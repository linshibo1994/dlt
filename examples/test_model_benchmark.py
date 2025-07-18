#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模型基准测试示例
展示如何使用模型基准测试框架评估和比较不同的预测算法
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

# 导入模型基准测试框架
from improvements.model_benchmark import get_model_benchmark

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

# 导入LSTM预测器
try:
    from advanced_lstm_predictor import AdvancedLSTMPredictor, TENSORFLOW_AVAILABLE
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ LSTM预测器模块未找到，将跳过相关测试")

# 导入高级集成预测器
try:
    from improvements.advanced_ensemble import AdvancedEnsemblePredictor
    ADVANCED_ENSEMBLE_AVAILABLE = True
except ImportError:
    ADVANCED_ENSEMBLE_AVAILABLE = False
    print("⚠️ 高级集成预测器模块未找到，将跳过相关测试")


def mock_random_predict(historical_data) -> List[Tuple[List[int], List[int]]]:
    """随机预测（基准模型）"""
    return [(sorted(np.random.choice(range(1, 36), 5, replace=False)), 
            sorted(np.random.choice(range(1, 13), 2, replace=False))) for _ in range(3)]


def mock_frequency_predict(historical_data) -> List[Tuple[List[int], List[int]]]:
    """频率预测（模拟）"""
    front_counter = Counter()
    back_counter = Counter()
    
    for _, row in historical_data.iterrows():
        front_balls, back_balls = data_manager.parse_balls(row)
        front_counter.update(front_balls)
        back_counter.update(back_balls)
    
    front_most_common = [ball for ball, _ in front_counter.most_common(5)]
    back_most_common = [ball for ball, _ in back_counter.most_common(2)]
    
    return [(sorted(front_most_common), sorted(back_most_common)) for _ in range(3)]


def mock_hot_cold_predict(historical_data) -> List[Tuple[List[int], List[int]]]:
    """冷热号预测（模拟）"""
    # 计算频率
    front_counter = Counter()
    back_counter = Counter()
    
    # 使用最近100期数据
    recent_data = historical_data.head(100)
    for _, row in recent_data.iterrows():
        front_balls, back_balls = data_manager.parse_balls(row)
        front_counter.update(front_balls)
        back_counter.update(back_balls)
    
    # 计算平均频率
    front_avg = sum(front_counter.values()) / len(front_counter) if front_counter else 0
    back_avg = sum(back_counter.values()) / len(back_counter) if back_counter else 0
    
    # 热号（高于平均频率）
    front_hot = [ball for ball, count in front_counter.items() if count > front_avg]
    back_hot = [ball for ball, count in back_counter.items() if count > back_avg]
    
    # 冷号（低于平均频率）
    front_cold = [ball for ball, count in front_counter.items() if count <= front_avg]
    back_cold = [ball for ball, count in back_counter.items() if count <= back_avg]
    
    # 组合热号和冷号
    predictions = []
    for _ in range(3):
        # 选择3个热号和2个冷号
        front_selected = []
        if len(front_hot) >= 3:
            front_selected.extend(np.random.choice(front_hot, 3, replace=False))
        else:
            front_selected.extend(front_hot)
            front_selected.extend(np.random.choice(front_cold, 3 - len(front_hot), replace=False))
        
        # 补充冷号
        remaining_cold = [b for b in front_cold if b not in front_selected]
        if len(remaining_cold) >= 2:
            front_selected.extend(np.random.choice(remaining_cold, 2, replace=False))
        else:
            front_selected.extend(remaining_cold)
            # 如果冷号不足，从所有号码中随机选择
            all_balls = list(range(1, 36))
            remaining_balls = [b for b in all_balls if b not in front_selected]
            front_selected.extend(np.random.choice(remaining_balls, 2 - len(remaining_cold), replace=False))
        
        # 后区选择1个热号和1个冷号
        back_selected = []
        if back_hot:
            back_selected.append(np.random.choice(back_hot))
        else:
            back_selected.append(np.random.choice(range(1, 13)))
        
        remaining_cold = [b for b in back_cold if b not in back_selected]
        if remaining_cold:
            back_selected.append(np.random.choice(remaining_cold))
        else:
            remaining_balls = [b for b in range(1, 13) if b not in back_selected]
            back_selected.append(np.random.choice(remaining_balls))
        
        predictions.append((sorted(front_selected), sorted(back_selected)))
    
    return predictions


def mock_missing_predict(historical_data) -> List[Tuple[List[int], List[int]]]:
    """遗漏值预测（模拟）"""
    # 计算遗漏值
    front_missing = {i: 0 for i in range(1, 36)}
    back_missing = {i: 0 for i in range(1, 13)}
    
    # 使用最近100期数据
    recent_data = historical_data.head(100)
    for _, row in recent_data.iterrows():
        front_balls, back_balls = data_manager.parse_balls(row)
        
        # 更新遗漏值
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
    
    # 选择遗漏值最大的号码
    front_sorted = sorted(front_missing.items(), key=lambda x: x[1], reverse=True)
    back_sorted = sorted(back_missing.items(), key=lambda x: x[1], reverse=True)
    
    front_selected = [ball for ball, _ in front_sorted[:5]]
    back_selected = [ball for ball, _ in back_sorted[:2]]
    
    return [(sorted(front_selected), sorted(back_selected)) for _ in range(3)]


def register_real_predictors(benchmark):
    """注册实际预测器"""
    if not PREDICTORS_AVAILABLE:
        return False
    
    # 获取预测器实例
    traditional = get_traditional_predictor()
    advanced = get_advanced_predictor()
    
    # 注册传统预测器
    benchmark.register_model(
        "频率预测(实际)", 
        lambda data: traditional.frequency_predict(3), 
        "基于历史频率的预测方法", 
        "traditional"
    )
    
    benchmark.register_model(
        "冷热号预测(实际)", 
        lambda data: traditional.hot_cold_predict(3), 
        "基于冷热号的预测方法", 
        "traditional"
    )
    
    benchmark.register_model(
        "遗漏值预测(实际)", 
        lambda data: traditional.missing_predict(3), 
        "基于遗漏值的预测方法", 
        "traditional"
    )
    
    # 注册高级预测器
    benchmark.register_model(
        "马尔可夫预测", 
        lambda data: advanced.markov_predict(3), 
        "基于马尔可夫链的预测方法", 
        "advanced"
    )
    
    benchmark.register_model(
        "贝叶斯预测", 
        lambda data: advanced.bayesian_predict(3), 
        "基于贝叶斯分析的预测方法", 
        "advanced"
    )
    
    benchmark.register_model(
        "集成预测", 
        lambda data: advanced.ensemble_predict(3), 
        "基于多种算法集成的预测方法", 
        "advanced"
    )
    
    return True


def register_enhanced_predictors(benchmark):
    """注册增强预测器"""
    registered_count = 0
    
    # 注册增强马尔可夫预测器
    if ENHANCED_MARKOV_AVAILABLE:
        markov_predictor = get_markov_predictor()
        
        benchmark.register_model(
            "二阶马尔可夫", 
            lambda data: markov_predictor.multi_order_markov_predict(3, 300, 2), 
            "基于二阶马尔可夫链的预测方法", 
            "enhanced"
        )
        
        benchmark.register_model(
            "三阶马尔可夫", 
            lambda data: markov_predictor.multi_order_markov_predict(3, 300, 3), 
            "基于三阶马尔可夫链的预测方法", 
            "enhanced"
        )
        
        benchmark.register_model(
            "自适应马尔可夫", 
            lambda data: [tuple([pred['front_balls'], pred['back_balls']]) for pred in 
                         markov_predictor.adaptive_order_markov_predict(3, 300)], 
            "基于自适应阶数马尔可夫链的预测方法", 
            "enhanced"
        )
        
        registered_count += 3
    
    # 注册LSTM预测器
    if TENSORFLOW_AVAILABLE:
        try:
            lstm_predictor = AdvancedLSTMPredictor()
            
            benchmark.register_model(
                "LSTM深度学习", 
                lambda data: lstm_predictor.lstm_predict(3), 
                "基于LSTM深度学习的预测方法", 
                "deep_learning"
            )
            
            registered_count += 1
        except Exception as e:
            print(f"⚠️ LSTM预测器初始化失败: {e}")
    
    # 注册高级集成预测器
    if ADVANCED_ENSEMBLE_AVAILABLE and PREDICTORS_AVAILABLE:
        try:
            ensemble = AdvancedEnsemblePredictor()
            traditional = get_traditional_predictor()
            advanced = get_advanced_predictor()
            
            # 注册基础预测器
            ensemble.register_predictor('frequency', traditional, weight=0.3)
            ensemble.register_predictor('markov', advanced, weight=0.4)
            ensemble.register_predictor('bayesian', advanced, weight=0.3)
            
            benchmark.register_model(
                "Stacking集成", 
                lambda data: ensemble.stacking_predict(3), 
                "基于Stacking的集成预测方法", 
                "ensemble"
            )
            
            benchmark.register_model(
                "加权集成", 
                lambda data: ensemble.weighted_ensemble_predict(3), 
                "基于权重的集成预测方法", 
                "ensemble"
            )
            
            benchmark.register_model(
                "自适应集成", 
                lambda data: ensemble.adaptive_ensemble_predict(3), 
                "基于自适应权重的集成预测方法", 
                "ensemble"
            )
            
            registered_count += 3
        except Exception as e:
            print(f"⚠️ 高级集成预测器初始化失败: {e}")
    
    return registered_count


def main():
    """主函数"""
    print("🚀 开始模型基准测试...")
    
    # 获取基准测试实例
    benchmark = get_model_benchmark()
    
    # 注册模拟预测器
    print("\n📝 注册模拟预测器...")
    benchmark.register_model("随机预测", mock_random_predict, "完全随机的预测方法", "baseline")
    benchmark.register_model("频率预测(模拟)", mock_frequency_predict, "基于历史频率的模拟预测方法", "mock")
    benchmark.register_model("冷热号预测(模拟)", mock_hot_cold_predict, "基于冷热号的模拟预测方法", "mock")
    benchmark.register_model("遗漏值预测(模拟)", mock_missing_predict, "基于遗漏值的模拟预测方法", "mock")
    
    # 注册实际预测器
    print("\n📝 注册实际预测器...")
    real_registered = register_real_predictors(benchmark)
    if real_registered:
        print(f"✅ 成功注册实际预测器")
    else:
        print(f"⚠️ 实际预测器注册失败，将仅使用模拟预测器")
    
    # 注册增强预测器
    print("\n📝 注册增强预测器...")
    enhanced_count = register_enhanced_predictors(benchmark)
    if enhanced_count > 0:
        print(f"✅ 成功注册 {enhanced_count} 个增强预测器")
    else:
        print(f"⚠️ 增强预测器注册失败，将仅使用基础预测器")
    
    # 设置测试参数
    test_periods = 20  # 测试期数
    
    # 评估所有模型
    print(f"\n📊 评估所有模型 (测试期数: {test_periods})...")
    benchmark.evaluate_all_models(test_periods=test_periods, verbose=True)
    
    # 比较模型
    print("\n🔄 比较模型...")
    benchmark.compare_models(verbose=True)
    
    # 生成报告
    print("\n📝 生成评估报告...")
    report_path = "output/model_benchmark_report.md"
    benchmark.generate_report(report_path)
    print(f"✅ 评估报告已保存到: {report_path}")
    
    # 可视化比较
    print("\n📈 可视化模型比较...")
    comparison_path = "output/model_comparison.png"
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
    
    # 保存结果
    print("\n💾 保存评估结果...")
    results_path = "output/model_benchmark_results.json"
    benchmark.save_results(results_path)
    print(f"✅ 评估结果已保存到: {results_path}")
    
    print("\n✅ 模型基准测试完成")


if __name__ == "__main__":
    # 确保输出目录存在
    os.makedirs("output", exist_ok=True)
    
    # 运行主函数
    main()