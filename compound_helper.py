#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
大乐透复式投注助手 - 简化接口
提供常用的复式投注组合和快速预测功能
"""

import sys
from predictors import CompoundPredictor


class CompoundHelper:
    """大乐透复式投注助手"""
    
    def __init__(self, data_file="data/dlt_data_all.csv"):
        """初始化助手"""
        self.predictor = CompoundPredictor(data_file)
    
    def predict_custom(self, periods, combinations_str, method="hybrid"):
        """自定义复式预测
        
        Args:
            periods: 分析期数
            combinations_str: 组合字符串，如 "6+2,7+3"
            method: 预测方法
        
        Returns:
            预测结果
        """
        combinations = []
        try:
            for combo in combinations_str.split(','):
                front, back = combo.strip().split('+')
                combinations.append((int(front), int(back)))
        except ValueError:
            print("❌ 组合格式错误，请使用格式：6+2,7+3")
            return []
        
        return self.predictor.predict_compound_combinations(
            periods=periods,
            combinations=combinations,
            method=method,
            explain=True
        )
    
    def predict_conservative(self, periods=1000, method="hybrid"):
        """保守型复式投注（成本较低）"""
        print("🛡️ 保守型复式投注策略")
        combinations = [(6, 2), (6, 3)]  # 6+2, 6+3
        return self.predictor.predict_compound_combinations(
            periods=periods,
            combinations=combinations,
            method=method,
            explain=True
        )
    
    def predict_balanced(self, periods=2000, method="hybrid"):
        """平衡型复式投注（中等成本）"""
        print("⚖️ 平衡型复式投注策略")
        combinations = [(7, 2), (7, 3), (8, 2)]  # 7+2, 7+3, 8+2
        return self.predictor.predict_compound_combinations(
            periods=periods,
            combinations=combinations,
            method=method,
            explain=True
        )
    
    def predict_aggressive(self, periods=3000, method="hybrid"):
        """激进型复式投注（成本较高）"""
        print("🚀 激进型复式投注策略")
        combinations = [(8, 3), (9, 3), (10, 4)]  # 8+3, 9+3, 10+4
        return self.predictor.predict_compound_combinations(
            periods=periods,
            combinations=combinations,
            method=method,
            explain=True
        )
    
    def predict_mega(self, periods=5000, method="hybrid"):
        """超级复式投注（高成本高覆盖）"""
        print("💎 超级复式投注策略")
        combinations = [(12, 4), (15, 5)]  # 12+4, 15+5
        return self.predictor.predict_compound_combinations(
            periods=periods,
            combinations=combinations,
            method=method,
            explain=True
        )
    
    def show_cost_analysis(self):
        """显示不同复式组合的成本分析"""
        print("💰 复式投注成本分析表")
        print("=" * 60)
        print(f"{'组合':<8} {'总注数':<12} {'投注成本':<12} {'适用场景'}")
        print("-" * 60)
        
        combinations = [
            ("5+2", 1, 3, "单式投注"),
            ("6+2", 6, 18, "小复式"),
            ("6+3", 18, 54, "小复式"),
            ("7+2", 21, 63, "中复式"),
            ("7+3", 63, 189, "中复式"),
            ("8+2", 56, 168, "中复式"),
            ("8+3", 168, 504, "大复式"),
            ("9+3", 252, 756, "大复式"),
            ("10+3", 360, 1080, "大复式"),
            ("10+4", 2160, 6480, "超大复式"),
            ("12+4", 4950, 14850, "超大复式"),
            ("15+5", 30030, 90090, "巨型复式"),
        ]
        
        for combo, notes, cost, scenario in combinations:
            print(f"{combo:<8} {notes:<12,} {cost:<12,} {scenario}")
        
        print("-" * 60)
        print("💡 建议：根据预算选择合适的复式组合")
    
    def interactive_mode(self):
        """交互式模式"""
        print("🎯 大乐透复式投注助手 - 交互模式")
        print("=" * 50)
        
        while True:
            print("\n请选择预测策略:")
            print("1. 自定义复式组合")
            print("2. 保守型策略 (6+2, 6+3)")
            print("3. 平衡型策略 (7+2, 7+3, 8+2)")
            print("4. 激进型策略 (8+3, 9+3, 10+4)")
            print("5. 超级策略 (12+4, 15+5)")
            print("6. 成本分析表")
            print("0. 退出")
            
            choice = input("\n请输入选择 (0-6): ").strip()
            
            if choice == "0":
                print("👋 感谢使用，祝您好运！")
                break
            elif choice == "1":
                periods = int(input("请输入分析期数 (建议1000-5000): ") or "3000")
                combinations = input("请输入复式组合 (格式: 6+2,7+3): ").strip()
                method = input("请选择方法 (hybrid/markov, 默认hybrid): ").strip() or "hybrid"
                
                if combinations:
                    self.predict_custom(periods, combinations, method)
                else:
                    print("❌ 请输入有效的组合")
            elif choice == "2":
                periods = int(input("请输入分析期数 (默认1000): ") or "1000")
                method = input("请选择方法 (hybrid/markov, 默认hybrid): ").strip() or "hybrid"
                self.predict_conservative(periods, method)
            elif choice == "3":
                periods = int(input("请输入分析期数 (默认2000): ") or "2000")
                method = input("请选择方法 (hybrid/markov, 默认hybrid): ").strip() or "hybrid"
                self.predict_balanced(periods, method)
            elif choice == "4":
                periods = int(input("请输入分析期数 (默认3000): ") or "3000")
                method = input("请选择方法 (hybrid/markov, 默认hybrid): ").strip() or "hybrid"
                self.predict_aggressive(periods, method)
            elif choice == "5":
                periods = int(input("请输入分析期数 (默认5000): ") or "5000")
                method = input("请选择方法 (hybrid/markov, 默认hybrid): ").strip() or "hybrid"
                self.predict_mega(periods, method)
            elif choice == "6":
                self.show_cost_analysis()
            else:
                print("❌ 无效选择，请重新输入")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="大乐透复式投注助手")
    parser.add_argument("-d", "--data", default="data/dlt_data_all.csv", help="数据文件路径")
    parser.add_argument("-i", "--interactive", action="store_true", help="交互模式")
    parser.add_argument("-s", "--strategy", choices=["conservative", "balanced", "aggressive", "mega"], 
                       help="预设策略")
    parser.add_argument("-p", "--periods", type=int, default=3000, help="分析期数")
    parser.add_argument("-m", "--method", choices=["hybrid", "markov"], default="hybrid", help="预测方法")
    parser.add_argument("-c", "--combinations", help="自定义组合 (格式: 6+2,7+3)")
    parser.add_argument("--cost", action="store_true", help="显示成本分析")
    
    args = parser.parse_args()
    
    # 创建助手
    helper = CompoundHelper(args.data)
    
    if not helper.predictor.df is not None:
        print("数据加载失败")
        return
    
    if args.interactive:
        # 交互模式
        helper.interactive_mode()
    elif args.cost:
        # 成本分析
        helper.show_cost_analysis()
    elif args.combinations:
        # 自定义组合
        helper.predict_custom(args.periods, args.combinations, args.method)
    elif args.strategy:
        # 预设策略
        if args.strategy == "conservative":
            helper.predict_conservative(args.periods, args.method)
        elif args.strategy == "balanced":
            helper.predict_balanced(args.periods, args.method)
        elif args.strategy == "aggressive":
            helper.predict_aggressive(args.periods, args.method)
        elif args.strategy == "mega":
            helper.predict_mega(args.periods, args.method)
    else:
        # 默认显示帮助
        print("🎯 大乐透复式投注助手")
        print("=" * 40)
        print("使用示例:")
        print("  python3 compound_helper.py -i                    # 交互模式")
        print("  python3 compound_helper.py --cost                # 成本分析")
        print("  python3 compound_helper.py -s balanced           # 平衡策略")
        print("  python3 compound_helper.py -c '6+2,7+3' -p 3000  # 自定义组合")
        print("\n预设策略:")
        print("  conservative: 保守型 (6+2, 6+3)")
        print("  balanced:     平衡型 (7+2, 7+3, 8+2)")
        print("  aggressive:   激进型 (8+3, 9+3, 10+4)")
        print("  mega:         超级型 (12+4, 15+5)")


if __name__ == "__main__":
    main()
