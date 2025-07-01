#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
高级混合分析预测器 - 简化接口
基于7种数学模型的综合预测系统
"""

import os
import sys
from advanced_hybrid_analyzer import AdvancedHybridAnalyzer


class HybridPredictor:
    """高级混合分析预测器 - 简化接口"""
    
    def __init__(self, data_file="data/dlt_data_all.csv"):
        """初始化预测器
        
        Args:
            data_file: 数据文件路径
        """
        self.data_file = data_file
        self.analyzer = None
        
        if not os.path.exists(data_file):
            print(f"❌ 数据文件不存在: {data_file}")
            print("请先运行数据爬虫获取数据")
            return
        
        self.analyzer = AdvancedHybridAnalyzer(data_file)
    
    def predict(self, periods=100, count=1, explain=True):
        """执行预测
        
        Args:
            periods: 分析期数 (建议30-200期)
            count: 预测注数 (1-10注)
            explain: 是否显示详细过程
        
        Returns:
            预测结果列表 [(前区号码列表, 后区号码列表), ...]
        """
        if not self.analyzer:
            print("❌ 分析器初始化失败")
            return []
        
        return self.analyzer.predict_with_hybrid_analysis(
            periods=periods,
            count=count,
            explain=explain
        )
    
    def predict_stable(self, periods=100, explain=True):
        """预测1注最稳定的号码
        
        Args:
            periods: 分析期数
            explain: 是否显示详细过程
        
        Returns:
            (前区号码列表, 后区号码列表)
        """
        predictions = self.predict(periods=periods, count=1, explain=explain)
        return predictions[0] if predictions else ([], [])
    
    def predict_multiple(self, periods=100, count=5, explain=True):
        """预测多注号码
        
        Args:
            periods: 分析期数
            count: 预测注数
            explain: 是否显示详细过程
        
        Returns:
            预测结果列表
        """
        return self.predict(periods=periods, count=count, explain=explain)
    
    def quick_predict(self, count=1):
        """快速预测（使用默认参数）
        
        Args:
            count: 预测注数
        
        Returns:
            预测结果列表
        """
        print("🚀 快速预测模式（基于100期数据）")
        return self.predict(periods=100, count=count, explain=False)
    
    def detailed_predict(self, periods=100, count=1):
        """详细预测（显示完整分析过程）
        
        Args:
            periods: 分析期数
            count: 预测注数
        
        Returns:
            预测结果列表
        """
        print("🔬 详细分析模式")
        return self.predict(periods=periods, count=count, explain=True)
    
    def format_predictions(self, predictions):
        """格式化预测结果
        
        Args:
            predictions: 预测结果列表
        
        Returns:
            格式化的字符串列表
        """
        formatted = []
        for i, (front_balls, back_balls) in enumerate(predictions, 1):
            front_str = ' '.join([str(b).zfill(2) for b in sorted(front_balls)])
            back_str = ' '.join([str(b).zfill(2) for b in sorted(back_balls)])
            formatted.append(f"第 {i} 注: 前区 {front_str} | 后区 {back_str}")
        return formatted
    
    def print_predictions(self, predictions):
        """打印预测结果
        
        Args:
            predictions: 预测结果列表
        """
        if not predictions:
            print("❌ 没有预测结果")
            return
        
        print(f"\n🎯 预测结果 ({len(predictions)} 注):")
        formatted = self.format_predictions(predictions)
        for line in formatted:
            print(line)


def main():
    """主函数 - 命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="高级混合分析预测器")
    parser.add_argument("-d", "--data", default="data/dlt_data_all.csv", help="数据文件路径")
    parser.add_argument("-p", "--periods", type=int, default=100, help="分析期数")
    parser.add_argument("-c", "--count", type=int, default=1, help="预测注数")
    parser.add_argument("-q", "--quick", action="store_true", help="快速预测模式")
    parser.add_argument("--detail", action="store_true", help="详细分析模式")
    parser.add_argument("--stable", action="store_true", help="预测最稳定的1注")
    
    args = parser.parse_args()
    
    # 创建预测器
    predictor = HybridPredictor(args.data)
    
    if not predictor.analyzer:
        return
    
    # 执行预测
    if args.quick:
        predictions = predictor.quick_predict(args.count)
    elif args.detail:
        predictions = predictor.detailed_predict(args.periods, args.count)
    elif args.stable:
        front_balls, back_balls = predictor.predict_stable(args.periods, explain=True)
        predictions = [(front_balls, back_balls)] if front_balls else []
    else:
        predictions = predictor.predict(args.periods, args.count, explain=False)
    
    # 显示结果
    predictor.print_predictions(predictions)


# 使用示例
def example_usage():
    """使用示例"""
    print("=" * 60)
    print("高级混合分析预测器使用示例")
    print("=" * 60)
    
    # 创建预测器
    predictor = HybridPredictor()
    
    if not predictor.analyzer:
        return
    
    print("\n1. 快速预测1注:")
    predictions = predictor.quick_predict(1)
    predictor.print_predictions(predictions)
    
    print("\n2. 预测最稳定的1注:")
    front_balls, back_balls = predictor.predict_stable(periods=50, explain=False)
    if front_balls:
        predictor.print_predictions([(front_balls, back_balls)])
    
    print("\n3. 预测5注号码:")
    predictions = predictor.predict_multiple(periods=100, count=5, explain=False)
    predictor.print_predictions(predictions)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # 如果没有命令行参数，运行示例
        example_usage()
    else:
        # 运行命令行接口
        main()
