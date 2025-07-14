#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
大乐透预测系统 Python API 使用示例
演示如何在Python代码中直接使用系统功能
"""

import os
import sys
import json
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入系统模块
from core_modules import data_manager, cache_manager, logger_manager
from analyzer_modules import basic_analyzer, advanced_analyzer, comprehensive_analyzer
from predictor_modules import get_traditional_predictor, get_advanced_predictor, get_super_predictor
from adaptive_learning_modules import enhanced_adaptive_predictor


def main():
    """主函数 - 演示API使用"""
    print("🎯 大乐透预测系统 Python API 使用示例")
    print("=" * 50)
    
    # 1. 数据管理示例
    print("\n📊 1. 数据管理示例")
    data_example()
    
    # 2. 数据分析示例
    print("\n🔍 2. 数据分析示例")
    analysis_example()
    
    # 3. 号码预测示例
    print("\n🎯 3. 号码预测示例")
    prediction_example()
    
    # 4. 自适应学习示例
    print("\n🧠 4. 自适应学习示例")
    learning_example()
    
    # 5. 缓存管理示例
    print("\n💾 5. 缓存管理示例")
    cache_example()
    
    print("\n🎉 API使用示例完成！")


def data_example():
    """数据管理示例"""
    print("获取数据...")
    
    # 获取数据
    df = data_manager.get_data()
    if df is not None:
        print(f"✅ 数据加载成功: {len(df)} 期")
        
        # 获取统计信息
        stats = data_manager.get_stats()
        print(f"📈 数据统计:")
        print(f"  总期数: {stats.get('total_periods', 0)}")
        print(f"  最新期号: {stats.get('latest_issue', 'N/A')}")
        print(f"  数据范围: {stats.get('date_range', {}).get('start', 'N/A')} 到 {stats.get('date_range', {}).get('end', 'N/A')}")
        
        # 解析最新一期号码
        latest_row = df.iloc[0]
        front_balls, back_balls = data_manager.parse_balls(latest_row)
        print(f"🎯 最新开奖: {' '.join([str(b).zfill(2) for b in front_balls])} + {' '.join([str(b).zfill(2) for b in back_balls])}")
    else:
        print("❌ 数据加载失败")


def analysis_example():
    """数据分析示例"""
    print("进行数据分析...")
    
    try:
        # 基础分析
        print("📊 基础分析:")
        freq_result = basic_analyzer.frequency_analysis(500)
        if freq_result:
            front_freq = freq_result.get('front_frequency', {})
            if front_freq:
                top_front = sorted(front_freq.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"  前区高频号码: {[f'{num}({count}次)' for num, count in top_front]}")
        
        # 高级分析
        print("🧠 高级分析:")
        markov_result = advanced_analyzer.markov_analysis(300)
        if markov_result:
            print("  马尔可夫链分析完成")
        
        bayesian_result = advanced_analyzer.bayesian_analysis(300)
        if bayesian_result:
            print("  贝叶斯分析完成")
        
        # 综合分析
        print("🔬 综合分析:")
        comprehensive_result = comprehensive_analyzer.comprehensive_analysis(500)
        if comprehensive_result:
            print("  综合分析完成")
            
            # 保存分析结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"output/reports/api_analysis_{timestamp}.json"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(comprehensive_result, f, ensure_ascii=False, indent=2, default=str)
            print(f"  分析结果已保存: {filename}")
    
    except Exception as e:
        print(f"❌ 分析失败: {e}")


def prediction_example():
    """号码预测示例"""
    print("进行号码预测...")
    
    try:
        # 传统预测
        print("📊 传统预测:")
        traditional_predictor = get_traditional_predictor()
        
        freq_pred = traditional_predictor.frequency_predict(3)
        print(f"  频率预测 ({len(freq_pred)} 注):")
        for i, (front, back) in enumerate(freq_pred):
            front_str = ' '.join([str(b).zfill(2) for b in front])
            back_str = ' '.join([str(b).zfill(2) for b in back])
            print(f"    第 {i+1} 注: {front_str} + {back_str}")
        
        # 高级预测
        print("🧠 高级预测:")
        advanced_predictor = get_advanced_predictor()
        
        ensemble_pred = advanced_predictor.ensemble_predict(2)
        print(f"  集成预测 ({len(ensemble_pred)} 注):")
        for i, (front, back) in enumerate(ensemble_pred):
            front_str = ' '.join([str(b).zfill(2) for b in front])
            back_str = ' '.join([str(b).zfill(2) for b in back])
            print(f"    第 {i+1} 注: {front_str} + {back_str}")
        
        # 混合策略预测
        print("🎯 混合策略预测:")
        mixed_pred = advanced_predictor.mixed_strategy_predict(2, strategy='balanced')
        print(f"  平衡策略 ({len(mixed_pred)} 注):")
        for pred in mixed_pred:
            front_str = ' '.join([str(b).zfill(2) for b in pred['front_balls']])
            back_str = ' '.join([str(b).zfill(2) for b in pred['back_balls']])
            print(f"    第 {pred['index']} 注: {front_str} + {back_str} (风险: {pred['risk_level']})")
        
        # 保存预测结果
        all_predictions = {
            'frequency': freq_pred,
            'ensemble': ensemble_pred,
            'mixed_strategy': mixed_pred,
            'timestamp': datetime.now().isoformat()
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"output/predictions/api_predictions_{timestamp}.json"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(all_predictions, f, ensure_ascii=False, indent=2, default=str)
        print(f"📁 预测结果已保存: {filename}")
    
    except Exception as e:
        print(f"❌ 预测失败: {e}")


def learning_example():
    """自适应学习示例"""
    print("进行自适应学习...")
    
    try:
        # 简短的学习示例（减少期数以节省时间）
        print("🔄 开始学习 (UCB1算法, 20期测试)...")
        
        results = enhanced_adaptive_predictor.enhanced_adaptive_learning(
            start_period=100,
            test_periods=20,
            algorithm='ucb1'
        )
        
        if results:
            print("✅ 学习完成!")
            print(f"  测试期数: {results.get('total_periods', 0)}")
            print(f"  中奖率: {results.get('win_rate', 0):.3f}")
            print(f"  最优预测器: {results.get('best_predictor', 'N/A')}")
            
            # 基于学习结果进行智能预测
            print("🎯 基于学习结果进行智能预测...")
            smart_pred = enhanced_adaptive_predictor.smart_predict(3)
            
            if smart_pred:
                print(f"  智能预测 ({len(smart_pred)} 注):")
                for i, pred in enumerate(smart_pred):
                    front_str = ' '.join([str(b).zfill(2) for b in pred['front_balls']])
                    back_str = ' '.join([str(b).zfill(2) for b in pred['back_balls']])
                    confidence = pred.get('confidence', 0)
                    print(f"    第 {i+1} 注: {front_str} + {back_str} (置信度: {confidence:.3f})")
            
            # 保存学习结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"output/learning/api_learning_{timestamp}.json"
            enhanced_adaptive_predictor.save_enhanced_results(filename)
            print(f"📁 学习结果已保存: {filename}")
        else:
            print("❌ 学习失败")
    
    except Exception as e:
        print(f"❌ 学习失败: {e}")


def cache_example():
    """缓存管理示例"""
    print("缓存管理...")
    
    try:
        # 获取缓存信息
        cache_info = cache_manager.get_cache_info()
        print("📊 缓存信息:")
        
        for cache_type in ['models', 'analysis', 'data']:
            info = cache_info[cache_type]
            print(f"  {cache_type}: {info['files']} 个文件, {info['size_mb']:.2f} MB")
        
        print(f"  总计: {cache_info['total']['files']} 个文件, {cache_info['total']['size_mb']:.2f} MB")
        
        # 演示缓存操作
        print("💾 缓存操作演示:")
        
        # 保存测试数据到缓存
        test_data = {'test': 'api_example', 'timestamp': datetime.now().isoformat()}
        cache_manager.save_cache('analysis', 'api_test', test_data)
        print("  保存测试数据到缓存")
        
        # 从缓存加载数据
        loaded_data = cache_manager.load_cache('analysis', 'api_test')
        if loaded_data:
            print(f"  从缓存加载数据: {loaded_data}")
        
        # 清理测试缓存
        cache_manager.clear_cache('analysis')
        print("  清理分析缓存")
    
    except Exception as e:
        print(f"❌ 缓存操作失败: {e}")


def custom_analysis_example():
    """自定义分析示例"""
    print("\n🔬 6. 自定义分析示例")
    
    try:
        # 获取数据
        df = data_manager.get_data()
        if df is None:
            print("❌ 无法获取数据")
            return
        
        print("进行自定义分析...")
        
        # 分析最近100期的号码分布
        recent_df = df.head(100)
        
        # 统计前区号码频率
        front_freq = {}
        back_freq = {}
        
        for _, row in recent_df.iterrows():
            front_balls, back_balls = data_manager.parse_balls(row)
            
            for ball in front_balls:
                front_freq[ball] = front_freq.get(ball, 0) + 1
            
            for ball in back_balls:
                back_freq[ball] = back_freq.get(ball, 0) + 1
        
        # 显示分析结果
        print("📊 最近100期分析结果:")
        
        # 前区热号
        hot_front = sorted(front_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"  前区热号: {[f'{num}({count})' for num, count in hot_front]}")
        
        # 后区热号
        hot_back = sorted(back_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"  后区热号: {[f'{num}({count})' for num, count in hot_back]}")
        
        # 计算奇偶比
        odd_count = sum(1 for num in front_freq.keys() if num % 2 == 1)
        even_count = len(front_freq) - odd_count
        print(f"  前区奇偶比: {odd_count}:{even_count}")
        
        # 保存自定义分析结果
        custom_result = {
            'analysis_type': 'custom_recent_100',
            'front_frequency': front_freq,
            'back_frequency': back_freq,
            'hot_front': hot_front,
            'hot_back': hot_back,
            'odd_even_ratio': f"{odd_count}:{even_count}",
            'timestamp': datetime.now().isoformat()
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"output/reports/custom_analysis_{timestamp}.json"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(custom_result, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"📁 自定义分析结果已保存: {filename}")
    
    except Exception as e:
        print(f"❌ 自定义分析失败: {e}")


if __name__ == "__main__":
    # 确保输出目录存在
    os.makedirs("output/reports", exist_ok=True)
    os.makedirs("output/predictions", exist_ok=True)
    os.makedirs("output/learning", exist_ok=True)
    
    # 运行示例
    main()
    
    # 运行自定义分析示例
    custom_analysis_example()
    
    print("\n💡 提示:")
    print("  - 可以根据需要修改此脚本")
    print("  - 所有生成的文件都保存在 output/ 目录下")
    print("  - 查看各模块的源代码了解更多API用法")
    print("  - 使用 logger_manager 进行日志记录")
    print("  - 使用 cache_manager 进行缓存管理")
