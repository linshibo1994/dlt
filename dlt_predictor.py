#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
大乐透智能预测系统 - 主程序
整合所有功能：预测、分析、数据管理、可视化
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 导入整合后的模块
try:
    from predictors import TraditionalPredictor, AdvancedPredictor, SuperPredictor, CompoundPredictor
    PREDICTORS_AVAILABLE = True
except ImportError:
    PREDICTORS_AVAILABLE = False
    print("预测器模块不可用")

try:
    from analyzers import BasicAnalyzer, AdvancedAnalyzer, ComprehensiveAnalyzer
    ANALYZERS_AVAILABLE = True
except ImportError:
    ANALYZERS_AVAILABLE = False
    print("分析器模块不可用")

try:
    from data_manager import DataCrawler, DataProcessor, DataUtils
    DATA_MANAGER_AVAILABLE = True
except ImportError:
    DATA_MANAGER_AVAILABLE = False
    print("数据管理器模块不可用")

try:
    from visualization import VisualizationTool
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("可视化模块不可用")


class DLTPredictor:
    """大乐透智能预测系统主类"""
    
    def __init__(self, data_file="data/dlt_data_all.csv"):
        self.data_file = data_file
        self.check_data_file()
    
    def check_data_file(self):
        """检查数据文件是否存在"""
        if not os.path.exists(self.data_file):
            print(f"数据文件 {self.data_file} 不存在")
            if DATA_MANAGER_AVAILABLE:
                print("尝试自动获取数据...")
                self.auto_fetch_data()
            else:
                print("数据管理器不可用，无法自动获取数据")
                return False
        return True
    
    def auto_fetch_data(self):
        """自动获取数据"""
        try:
            data_dir = os.path.dirname(self.data_file)
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            
            crawler = DataCrawler()
            results = crawler.get_history_data(count=300)
            if results:
                crawler.save_to_csv(results, os.path.basename(self.data_file))
                print("数据获取成功")
                return True
            else:
                print("数据获取失败")
                return False
        except Exception as e:
            print(f"自动获取数据失败: {e}")
            return False
    
    def check_system_status(self):
        """检查系统状态"""
        print("🔧 系统状态检查")
        print("=" * 50)
        
        # 检查模块状态
        print("📦 预测模块状态:")
        print(f"  预测器模块: {'✅ 可用' if PREDICTORS_AVAILABLE else '❌ 不可用'}")
        print(f"  分析器模块: {'✅ 可用' if ANALYZERS_AVAILABLE else '❌ 不可用'}")
        print(f"  数据管理器: {'✅ 可用' if DATA_MANAGER_AVAILABLE else '❌ 不可用'}")
        print(f"  可视化工具: {'✅ 可用' if VISUALIZATION_AVAILABLE else '❌ 不可用'}")
        
        # 检查数据状态
        print("\n📊 数据状态:")
        if os.path.exists(self.data_file):
            try:
                df = pd.read_csv(self.data_file)
                print(f"  数据文件: ✅ {self.data_file}")
                print(f"  记录数量: {len(df)} 条")
                print(f"  数据范围: {df['issue'].min()} - {df['issue'].max()}")
            except Exception as e:
                print(f"  数据文件: ❌ 读取失败 - {e}")
        else:
            print(f"  数据文件: ❌ 不存在")
        
        # 检查可用功能
        print("\n🎮 可用功能:")
        if PREDICTORS_AVAILABLE:
            print("  ✅ 传统集成    ✅ 混合分析      ✅ 马尔可夫链")
            print("  ✅ 统计分析    ✅ 高级集成      ✅ LSTM深度学习")
            print("  ✅ 集成学习    ✅ 蒙特卡洛模拟  ✅ 聚类分析")
            print("  ✅ 终极集成    ✅ 复式投注预测")
        else:
            print("  ❌ 预测功能不可用")
        
        if ANALYZERS_AVAILABLE:
            print("  ✅ 基础分析    ✅ 高级分析      ✅ 综合分析")
        else:
            print("  ❌ 分析功能不可用")
        
        if DATA_MANAGER_AVAILABLE:
            print("  ✅ 数据爬取    ✅ 数据处理      ✅ 数据验证")
        else:
            print("  ❌ 数据管理功能不可用")
        
        if VISUALIZATION_AVAILABLE:
            print("  ✅ 可视化图表  ✅ 统计报告")
        else:
            print("  ❌ 可视化功能不可用")
    
    def predict_compound(self, combinations=None, method="hybrid", periods=150):
        """复式投注预测"""
        if not PREDICTORS_AVAILABLE:
            print("预测器模块不可用")
            return {}

        if not os.path.exists(self.data_file):
            print("数据文件不存在，无法进行预测")
            return {}

        try:
            predictor = CompoundPredictor(self.data_file)

            if combinations is None:
                combinations = [(6, 2), (7, 3)]  # 默认组合

            # 如果传入的是字符串格式，转换为元组格式
            if isinstance(combinations[0], str):
                combo_tuples = []
                for combo in combinations:
                    front_count, back_count = map(int, combo.split('+'))
                    combo_tuples.append((front_count, back_count))
                combinations = combo_tuples

            print("🎰 复式投注预测")
            print("=" * 50)
            print(f"📋 使用方法: {method}")
            print(f"📊 分析期数: {periods}")
            print()

            results = predictor.predict_compound_combinations(
                periods=periods,
                combinations=combinations,
                method=method,
                explain=True
            )

            return results

        except Exception as e:
            print(f"复式预测失败: {e}")
            return {}

    def predict(self, method="ultimate_ensemble", count=1):
        """预测号码"""
        if not PREDICTORS_AVAILABLE:
            print("预测器模块不可用")
            return []
        
        if not os.path.exists(self.data_file):
            print("数据文件不存在，无法进行预测")
            return []
        
        try:
            if method == "quick":
                predictor = SuperPredictor(self.data_file)
                predictions = predictor.quick_predict(count=count)
                
                print("⚡ 快速预测模式")
                print("=" * 50)
                print("\n🎯 快速预测结果（混合分析）:")
                
            elif method == "traditional_ensemble":
                predictor = TraditionalPredictor(self.data_file)
                predictions = predictor.ensemble_predict(count=count)
                
                print("🔬 传统方法集成预测")
                print("=" * 50)
                print("📋 使用方法: 混合分析 + 马尔可夫链 + 统计分析")
                print("\n✅ 传统方法预测完成")
                print("\n🏆 传统集成预测结果:")
                
            elif method == "advanced_ensemble":
                predictor = AdvancedPredictor(self.data_file)
                predictions = predictor.ensemble_predict(count=count)
                
                print("🧠 高级方法集成预测")
                print("=" * 50)
                print("📋 使用方法: LSTM + 集成学习 + 蒙特卡洛 + 聚类分析")
                print("\n✅ 高级方法预测完成")
                print("\n🏆 高级集成预测结果:")
                
            elif method == "ultimate_ensemble":
                predictor = SuperPredictor(self.data_file)
                predictions = predictor.ultimate_ensemble_predict(count=count)
                
                print("🌟 终极集成预测")
                print("=" * 50)
                print("📋 融合方法: 传统预测 + 高级预测")
                print("\n✅ 传统方法预测完成")
                print("✅ 高级方法预测完成")
                print("\n🏆 终极集成预测结果:")
                
            elif method == "compare":
                predictor = SuperPredictor(self.data_file)
                results = predictor.compare_all_methods(count=1)
                
                print("📊 全方法预测对比")
                print("=" * 80)
                print("\n🔬 传统预测方法:")
                print("方法              权重       状态       第1注预测")
                print("-" * 70)
                
                traditional_methods = ['hybrid', 'markov', 'statistical']
                traditional_weights = {'hybrid': 50.0, 'markov': 30.0, 'statistical': 20.0}
                
                for method_name in traditional_methods:
                    if method_name in results and results[method_name]:
                        front, back = results[method_name][0]
                        front_str = ' '.join([str(b).zfill(2) for b in front])
                        back_str = ' '.join([str(b).zfill(2) for b in back])
                        weight = traditional_weights.get(method_name.upper(), 0)
                        print(f"{method_name.upper():<12} {weight:>6.1f}%    ✅可用      前区 {front_str} | 后区 {back_str}")
                    else:
                        print(f"{method_name.upper():<12} {'0.0':>6}%    ❌不可用    预测失败")
                
                print("\n🧠 高级预测方法:")
                advanced_methods = ['lstm', 'ensemble_ml', 'monte_carlo', 'clustering']
                for method_name in advanced_methods:
                    if method_name in results and results[method_name]:
                        front, back = results[method_name][0]
                        front_str = ' '.join([str(b).zfill(2) for b in front])
                        back_str = ' '.join([str(b).zfill(2) for b in back])
                        print(f"{method_name.upper()}预测:")
                        print(f"  第 1 注: 前区 {front_str} | 后区 {back_str}")
                    else:
                        print(f"{method_name.upper()}预测: 预测失败")
                
                if 'ultimate_ensemble' in results and results['ultimate_ensemble']:
                    front, back = results['ultimate_ensemble'][0]
                    front_str = ' '.join([str(b).zfill(2) for b in front])
                    back_str = ' '.join([str(b).zfill(2) for b in back])
                    print(f"\n🌟 终极集成预测:")
                    print(f"第 1 注: 前区 {front_str} | 后区 {back_str}")
                
                return results
            
            else:
                print(f"未知的预测方法: {method}")
                return []
            
            # 显示预测结果
            for i, (front, back) in enumerate(predictions, 1):
                front_str = ' '.join([str(b).zfill(2) for b in front])
                back_str = ' '.join([str(b).zfill(2) for b in back])
                print(f"第 {i} 注: 前区 {front_str} | 后区 {back_str}")
            
            if method == "ultimate_ensemble":
                print(f"\n💡 权重配置:")
                print(f"   传统方法: 60%")
                print(f"   高级方法: 40%")
            
            return predictions
            
        except Exception as e:
            print(f"预测失败: {e}")
            return []
    
    def analyze(self, method="comprehensive", periods=0):
        """数据分析"""
        if not ANALYZERS_AVAILABLE:
            print("分析器模块不可用")
            return {}
        
        if not os.path.exists(self.data_file):
            print("数据文件不存在，无法进行分析")
            return {}
        
        try:
            if method == "basic":
                analyzer = BasicAnalyzer(self.data_file)
                results = analyzer.run_basic_analysis()
                print("基础分析完成")
            elif method == "advanced":
                analyzer = AdvancedAnalyzer(self.data_file)
                results = analyzer.run_advanced_analysis()
                print("高级分析完成")
            elif method == "comprehensive":
                analyzer = ComprehensiveAnalyzer(self.data_file)
                results = analyzer.run_all_analysis(periods=periods)
                print("综合分析完成")
            else:
                print(f"未知的分析方法: {method}")
                return {}
            
            return results
            
        except Exception as e:
            print(f"分析失败: {e}")
            return {}
    
    def generate_numbers(self, count=5, strategy="hybrid"):
        """生成号码"""
        if not PREDICTORS_AVAILABLE:
            print("预测器模块不可用，使用随机生成")
            for i in range(count):
                front, back = DataUtils.generate_random_numbers()
                print(f"[{i+1}] {DataUtils.format_numbers(front, back)}")
            return
        
        if strategy == "random":
            for i in range(count):
                front, back = DataUtils.generate_random_numbers()
                print(f"[{i+1}] {DataUtils.format_numbers(front, back)}")
        else:
            predictions = self.predict(method="quick", count=count)
            for i, (front, back) in enumerate(predictions, 1):
                print(f"[{i}] {DataUtils.format_numbers(front, back)}")
    
    def get_latest_result(self, compare=False):
        """获取最新开奖结果"""
        if not os.path.exists(self.data_file):
            print("数据文件不存在")
            return
        
        try:
            df = pd.read_csv(self.data_file)
            if len(df) == 0:
                print("数据文件为空")
                return
            
            # 获取最新一期数据
            latest_row = df.iloc[0]  # 假设数据按期号降序排列
            
            issue = latest_row['issue']
            date = latest_row['date']
            front_balls = latest_row['front_balls']
            back_balls = latest_row['back_balls']
            
            print(f"\n最新开奖结果 (期号: {issue}, 日期: {date})")
            print(f"前区号码: {front_balls}")
            print(f"后区号码: {back_balls}")
            
            # 如果需要比对
            if compare:
                self._compare_with_latest(front_balls, back_balls)
                
        except Exception as e:
            print(f"获取最新开奖结果失败: {e}")
    
    def _compare_with_latest(self, front_balls_latest, back_balls_latest):
        """将用户输入的号码与最新开奖结果进行比对"""
        try:
            print("\n请输入您的号码进行比对:")
            
            # 输入前区号码
            front_input = input("请输入前区5个号码（用空格分隔）: ")
            front_balls = [int(x.strip()) for x in front_input.split()]
            
            if len(front_balls) != 5:
                print("前区号码必须是5个")
                return
            
            # 输入后区号码
            back_input = input("请输入后区2个号码（用空格分隔）: ")
            back_balls = [int(x.strip()) for x in back_input.split()]
            
            if len(back_balls) != 2:
                print("后区号码必须是2个")
                return
            
            # 解析最新开奖号码
            latest_front = [int(x.strip()) for x in front_balls_latest.split(',')]
            latest_back = [int(x.strip()) for x in back_balls_latest.split(',')]
            
            # 排序
            front_balls.sort()
            back_balls.sort()
            
            # 比对
            front_match = len(set(front_balls) & set(latest_front))
            back_match = len(set(back_balls) & set(latest_back))
            
            print(f"\n您的号码: 前区 {front_balls}, 后区 {back_balls}")
            print(f"开奖号码: 前区 {latest_front}, 后区 {latest_back}")
            print(f"匹配结果: 前区匹配 {front_match} 个, 后区匹配 {back_match} 个")
            
            # 判断中奖等级
            prize_level = DataUtils.calculate_prize(front_balls, back_balls, latest_front, latest_back)
            if prize_level > 0:
                print(f"恭喜您中得 {DataUtils.get_prize_name(prize_level)}！")
            else:
                print("很遗憾，您未中奖")
                
        except Exception as e:
            print(f"比对失败: {e}")
    
    def crawl_data(self, count=100, get_all=False, update=False):
        """爬取数据"""
        if not DATA_MANAGER_AVAILABLE:
            print("数据管理器模块不可用")
            return
        
        try:
            crawler = DataCrawler()
            
            if update:
                crawler.update_data(self.data_file, count)
            else:
                if get_all:
                    results = crawler.get_history_data(get_all=True)
                else:
                    results = crawler.get_history_data(count=count)
                
                if results:
                    crawler.save_to_csv(results, os.path.basename(self.data_file))
                    print(f"数据爬取完成，保存到 {self.data_file}")
                else:
                    print("未获取到数据")
        except Exception as e:
            print(f"爬取数据失败: {e}")
    
    def visualize(self, output_dir="output/visualization"):
        """生成可视化图表"""
        if not VISUALIZATION_AVAILABLE:
            print("可视化模块不可用")
            return
        
        if not os.path.exists(self.data_file):
            print("数据文件不存在")
            return
        
        try:
            viz = VisualizationTool(self.data_file)
            viz.generate_all_charts(output_dir)
            print(f"可视化图表已生成到: {output_dir}")
        except Exception as e:
            print(f"生成可视化图表失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="大乐透智能预测系统")
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # 预测子命令
    predict_parser = subparsers.add_parser("predict", help="预测号码")
    predict_parser.add_argument("-m", "--method",
                               choices=['quick', 'traditional_ensemble', 'advanced_ensemble', 'ultimate_ensemble', 'compare'],
                               default='ultimate_ensemble', help="预测方法")
    predict_parser.add_argument("-n", "--num", type=int, default=1, help="预测注数")
    predict_parser.add_argument("-d", "--data", default="data/dlt_data_all.csv", help="数据文件路径")

    # 复式预测子命令
    compound_parser = subparsers.add_parser("compound", help="复式投注预测")
    compound_parser.add_argument("-c", "--combinations", nargs='+', default=["6+2", "7+3"],
                                help="复式组合，格式：6+2 7+3 表示6+2和7+3两注")
    compound_parser.add_argument("-m", "--method", choices=['hybrid', 'markov', 'advanced'],
                                default='hybrid', help="预测方法")
    compound_parser.add_argument("-p", "--periods", type=int, default=150, help="分析期数")
    compound_parser.add_argument("-d", "--data", default="data/dlt_data_all.csv", help="数据文件路径")
    
    # 分析子命令
    analyze_parser = subparsers.add_parser("analyze", help="分析数据")
    analyze_parser.add_argument("-m", "--method", choices=['basic', 'advanced', 'comprehensive'], 
                               default='comprehensive', help="分析方法")
    analyze_parser.add_argument("-p", "--periods", type=int, default=0, help="分析期数，0表示全部")
    analyze_parser.add_argument("-d", "--data", default="data/dlt_data_all.csv", help="数据文件路径")
    
    # 生成号码子命令
    generate_parser = subparsers.add_parser("generate", help="生成号码")
    generate_parser.add_argument("-c", "--count", type=int, default=5, help="生成号码注数")
    generate_parser.add_argument("-s", "--strategy", choices=["random", "hybrid"], default="hybrid", help="生成策略")
    generate_parser.add_argument("-d", "--data", default="data/dlt_data_all.csv", help="数据文件路径")
    
    # 最新开奖子命令
    latest_parser = subparsers.add_parser("latest", help="显示最新开奖结果")
    latest_parser.add_argument("-c", "--compare", action="store_true", help="与自选号码比对")
    latest_parser.add_argument("-d", "--data", default="data/dlt_data_all.csv", help="数据文件路径")
    
    # 爬虫子命令
    crawl_parser = subparsers.add_parser("crawl", help="爬取数据")
    crawl_parser.add_argument("-c", "--count", type=int, default=100, help="爬取期数")
    crawl_parser.add_argument("-a", "--all", action="store_true", help="爬取所有历史数据")
    crawl_parser.add_argument("-u", "--update", action="store_true", help="更新现有数据")
    crawl_parser.add_argument("-d", "--data", default="data/dlt_data_all.csv", help="数据文件路径")
    
    # 可视化子命令
    viz_parser = subparsers.add_parser("visualize", help="生成可视化图表")
    viz_parser.add_argument("-o", "--output", default="output/visualization", help="输出目录")
    viz_parser.add_argument("-d", "--data", default="data/dlt_data_all.csv", help="数据文件路径")
    
    # 状态检查子命令
    status_parser = subparsers.add_parser("status", help="检查系统状态")
    status_parser.add_argument("-d", "--data", default="data/dlt_data_all.csv", help="数据文件路径")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 如果没有子命令，显示帮助
    if not args.command:
        parser.print_help()
        return
    
    # 创建预测器实例
    data_file = getattr(args, 'data', 'data/dlt_data_all.csv')
    predictor = DLTPredictor(data_file)
    
    # 根据子命令执行相应的功能
    if args.command == "predict":
        predictor.predict(method=args.method, count=args.num)
    elif args.command == "compound":
        predictor.predict_compound(combinations=args.combinations, method=args.method, periods=args.periods)
    elif args.command == "analyze":
        predictor.analyze(method=args.method, periods=args.periods)
    elif args.command == "generate":
        predictor.generate_numbers(count=args.count, strategy=args.strategy)
    elif args.command == "latest":
        predictor.get_latest_result(compare=args.compare)
    elif args.command == "crawl":
        predictor.crawl_data(count=args.count, get_all=args.all, update=args.update)
    elif args.command == "visualize":
        predictor.visualize(output_dir=args.output)
    elif args.command == "status":
        predictor.check_system_status()


if __name__ == "__main__":
    main()
