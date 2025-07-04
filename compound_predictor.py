#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
大乐透复式投注预测器
支持任意期数分析，生成任意注数的任意前区+后区数量组合
"""

import os
import json
import argparse
from datetime import datetime
from advanced_hybrid_analyzer import AdvancedHybridAnalyzer
from dlt_analyzer import DLTAnalyzer


class CompoundPredictor:
    """大乐透复式投注预测器"""
    
    def __init__(self, data_file="data/dlt_data_all.csv"):
        """初始化预测器
        
        Args:
            data_file: 数据文件路径
        """
        self.data_file = data_file
        self.hybrid_analyzer = None
        self.markov_analyzer = None
        
        if not os.path.exists(data_file):
            print(f"❌ 数据文件不存在: {data_file}")
            print("请先运行数据爬虫获取数据")
            return
        
        # 初始化分析器
        self.hybrid_analyzer = AdvancedHybridAnalyzer(data_file)
        self.markov_analyzer = DLTAnalyzer(data_file)
    
    def predict_compound_combinations(self, periods=3000, combinations=None, method="hybrid", explain=True):
        """预测复式投注组合
        
        Args:
            periods: 分析期数
            combinations: 组合列表，格式：[(前区数量, 后区数量), ...]
                         例如：[(6, 2), (7, 5)] 表示第一注6+2，第二注7+5
            method: 预测方法 ("hybrid" 或 "markov")
            explain: 是否显示详细过程
        
        Returns:
            预测结果列表
        """
        if not self.hybrid_analyzer or not self.markov_analyzer:
            print("❌ 分析器初始化失败")
            return []
        
        if not combinations:
            combinations = [(6, 2), (7, 3)]  # 默认组合
        
        if explain:
            print("=" * 80)
            print("🎯 大乐透复式投注预测器")
            print("=" * 80)
            print(f"📊 分析期数: {periods} 期")
            print(f"🎲 预测方法: {'高级混合分析' if method == 'hybrid' else '马尔可夫链分析'}")
            print(f"📝 复式组合: {len(combinations)} 注")
            for i, (front, back) in enumerate(combinations, 1):
                print(f"   第 {i} 注: {front}+{back} (前区{front}个号码，后区{back}个号码)")
            print()
        
        predictions = []
        
        for i, (front_count, back_count) in enumerate(combinations, 1):
            if explain:
                print(f"🔮 生成第 {i} 注复式组合 ({front_count}+{back_count})...")
            
            # 验证参数合理性
            if not self._validate_combination(front_count, back_count):
                print(f"❌ 第 {i} 注组合参数无效: {front_count}+{back_count}")
                continue
            
            # 根据方法选择预测器
            if method == "hybrid":
                front_balls, back_balls = self._predict_hybrid_compound(
                    periods, front_count, back_count, i, explain
                )
            else:  # markov
                front_balls, back_balls = self._predict_markov_compound(
                    periods, front_count, back_count, i, explain
                )
            
            if front_balls and back_balls:
                prediction = {
                    'index': i,
                    'combination': f"{front_count}+{back_count}",
                    'front_count': front_count,
                    'back_count': back_count,
                    'front_balls': sorted(front_balls),
                    'back_balls': sorted(back_balls),
                    'total_combinations': self._calculate_total_combinations(front_count, back_count),
                    'investment_cost': self._calculate_investment_cost(front_count, back_count)
                }
                predictions.append(prediction)
                
                if explain:
                    self._print_compound_result(prediction)
            else:
                print(f"❌ 第 {i} 注预测失败")
        
        # 保存预测结果
        if predictions:
            self._save_compound_predictions(predictions, periods, method)
        
        if explain:
            print("\n" + "=" * 80)
            print("✅ 复式投注预测完成")
            print("=" * 80)
        
        return predictions
    
    def _validate_combination(self, front_count, back_count):
        """验证组合参数的合理性"""
        if front_count < 5 or front_count > 35:
            return False
        if back_count < 2 or back_count > 12:
            return False
        return True
    
    def _predict_hybrid_compound(self, periods, front_count, back_count, index, explain=True):
        """使用高级混合分析预测复式组合"""
        try:
            # 获取分析数据
            data = self.hybrid_analyzer.df.tail(periods).copy()
            
            # 运行混合分析
            hybrid_analysis = self.hybrid_analyzer._run_hybrid_analysis(data, explain=False)
            
            # 获取最近一期号码
            latest_row = data.iloc[-1]
            latest_front = self.hybrid_analyzer.parse_balls(latest_row['front_balls'])
            latest_back = self.hybrid_analyzer.parse_balls(latest_row['back_balls'])
            
            # 计算综合评分
            front_scores, back_scores = self.hybrid_analyzer._calculate_comprehensive_scores(
                hybrid_analysis, latest_front, latest_back, index-1, explain=False
            )
            
            # 选择指定数量的号码
            front_balls = self._select_top_numbers(front_scores, front_count, 35)
            back_balls = self._select_top_numbers(back_scores, back_count, 12)
            
            if explain:
                print(f"   ✅ 高级混合分析完成")
                print(f"   📊 前区评分最高的{front_count}个号码已选择")
                print(f"   📊 后区评分最高的{back_count}个号码已选择")
            
            return front_balls, back_balls
            
        except Exception as e:
            print(f"❌ 高级混合分析预测失败: {e}")
            return [], []
    
    def _predict_markov_compound(self, periods, front_count, back_count, index, explain=True):
        """使用马尔可夫链分析预测复式组合"""
        try:
            # 使用马尔可夫链分析器
            try:
                analysis_result = self.markov_analyzer.analyze_periods(periods)
            except TypeError:
                # 如果方法签名不匹配，尝试其他方式
                analysis_result = self.markov_analyzer.analyze_periods(periods, False)

            if not analysis_result:
                # 如果分析失败，使用简化的马尔可夫链分析
                return self._simple_markov_analysis(periods, front_count, back_count, explain)

            # 获取转移概率
            front_probs = analysis_result.get('front_transition_probs', {})
            back_probs = analysis_result.get('back_transition_probs', {})

            # 获取最近一期号码
            latest_data = self.markov_analyzer.df.tail(1).iloc[0]
            latest_front = [int(x.strip()) for x in str(latest_data['front_balls']).split(',')]
            latest_back = [int(x.strip()) for x in str(latest_data['back_balls']).split(',')]

            # 计算马尔可夫链评分
            front_scores = self._calculate_markov_scores(latest_front, front_probs, 35)
            back_scores = self._calculate_markov_scores(latest_back, back_probs, 12)

            # 选择指定数量的号码
            front_balls = self._select_top_numbers(front_scores, front_count, 35)
            back_balls = self._select_top_numbers(back_scores, back_count, 12)

            if explain:
                print(f"   ✅ 马尔可夫链分析完成")
                print(f"   🔗 前区转移概率最高的{front_count}个号码已选择")
                print(f"   🔗 后区转移概率最高的{back_count}个号码已选择")

            return front_balls, back_balls

        except Exception as e:
            print(f"❌ 马尔可夫链分析预测失败: {e}")
            # 使用简化分析作为备选
            return self._simple_markov_analysis(periods, front_count, back_count, explain)

    def _simple_markov_analysis(self, periods, front_count, back_count, explain=True):
        """简化的马尔可夫链分析"""
        try:
            # 获取数据
            data = self.markov_analyzer.df.tail(periods)

            # 构建简单的转移统计
            front_transitions = {}
            back_transitions = {}

            for i in range(len(data) - 1):
                current_row = data.iloc[i]
                next_row = data.iloc[i + 1]

                current_front = [int(x.strip()) for x in str(current_row['front_balls']).split(',')]
                current_back = [int(x.strip()) for x in str(current_row['back_balls']).split(',')]
                next_front = [int(x.strip()) for x in str(next_row['front_balls']).split(',')]
                next_back = [int(x.strip()) for x in str(next_row['back_balls']).split(',')]

                # 统计前区转移
                for cf in current_front:
                    if cf not in front_transitions:
                        front_transitions[cf] = {}
                    for nf in next_front:
                        front_transitions[cf][nf] = front_transitions[cf].get(nf, 0) + 1

                # 统计后区转移
                for cb in current_back:
                    if cb not in back_transitions:
                        back_transitions[cb] = {}
                    for nb in next_back:
                        back_transitions[cb][nb] = back_transitions[cb].get(nb, 0) + 1

            # 获取最近一期号码
            latest_data = data.tail(1).iloc[0]
            latest_front = [int(x.strip()) for x in str(latest_data['front_balls']).split(',')]
            latest_back = [int(x.strip()) for x in str(latest_data['back_balls']).split(',')]

            # 计算评分
            front_scores = self._calculate_simple_markov_scores(latest_front, front_transitions, 35)
            back_scores = self._calculate_simple_markov_scores(latest_back, back_transitions, 12)

            # 选择号码
            front_balls = self._select_top_numbers(front_scores, front_count, 35)
            back_balls = self._select_top_numbers(back_scores, back_count, 12)

            if explain:
                print(f"   ✅ 简化马尔可夫链分析完成")
                print(f"   🔗 前区转移概率最高的{front_count}个号码已选择")
                print(f"   🔗 后区转移概率最高的{back_count}个号码已选择")

            return front_balls, back_balls

        except Exception as e:
            print(f"❌ 简化马尔可夫链分析失败: {e}")
            return [], []

    def _calculate_simple_markov_scores(self, current_balls, transitions, max_ball):
        """计算简化马尔可夫链评分"""
        scores = {i: 0.0 for i in range(1, max_ball + 1)}

        for current_ball in current_balls:
            if current_ball in transitions:
                total_transitions = sum(transitions[current_ball].values())
                for next_ball, count in transitions[current_ball].items():
                    scores[next_ball] += count / total_transitions if total_transitions > 0 else 0

        # 如果没有转移概率，使用均匀分布
        if all(score == 0 for score in scores.values()):
            uniform_score = 1.0 / max_ball
            scores = {i: uniform_score for i in range(1, max_ball + 1)}

        return scores
    
    def _calculate_markov_scores(self, current_balls, transition_probs, max_ball):
        """计算马尔可夫链评分"""
        scores = {i: 0.0 for i in range(1, max_ball + 1)}
        
        for current_ball in current_balls:
            if current_ball in transition_probs:
                for next_ball, prob_info in transition_probs[current_ball].items():
                    if isinstance(prob_info, dict):
                        prob = prob_info.get('probability', 0)
                    else:
                        prob = prob_info
                    scores[next_ball] += prob
        
        # 如果没有转移概率，使用均匀分布
        if all(score == 0 for score in scores.values()):
            uniform_score = 1.0 / max_ball
            scores = {i: uniform_score for i in range(1, max_ball + 1)}
        
        return scores
    
    def _select_top_numbers(self, scores, count, max_ball):
        """选择评分最高的指定数量号码"""
        # 按评分排序
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # 选择前count个号码
        selected = [ball for ball, score in sorted_scores[:count]]
        
        # 如果数量不足，随机补充
        if len(selected) < count:
            remaining = [i for i in range(1, max_ball + 1) if i not in selected]
            import random
            random.shuffle(remaining)
            selected.extend(remaining[:count - len(selected)])
        
        return sorted(selected[:count])
    
    def _calculate_total_combinations(self, front_count, back_count):
        """计算总投注组合数"""
        import math
        
        # C(front_count, 5) * C(back_count, 2)
        front_combinations = math.comb(front_count, 5)
        back_combinations = math.comb(back_count, 2)
        
        return front_combinations * back_combinations
    
    def _calculate_investment_cost(self, front_count, back_count, single_cost=3):
        """计算投注成本"""
        total_combinations = self._calculate_total_combinations(front_count, back_count)
        return total_combinations * single_cost
    
    def _print_compound_result(self, prediction):
        """打印复式预测结果"""
        front_str = ' '.join([str(b).zfill(2) for b in prediction['front_balls']])
        back_str = ' '.join([str(b).zfill(2) for b in prediction['back_balls']])
        
        print(f"   第 {prediction['index']} 注 ({prediction['combination']}):")
        print(f"     前区 ({prediction['front_count']}个): {front_str}")
        print(f"     后区 ({prediction['back_count']}个): {back_str}")
        print(f"     总组合数: {prediction['total_combinations']:,} 注")
        print(f"     投注成本: {prediction['investment_cost']:,} 元")
        print()
    
    def _save_compound_predictions(self, predictions, periods, method):
        """保存复式预测结果"""
        try:
            output_dir = "output/compound"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"compound_predictions_{method}_{periods}periods_{timestamp}.json"
            filepath = os.path.join(output_dir, filename)
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'method': method,
                'periods': periods,
                'total_predictions': len(predictions),
                'predictions': predictions,
                'summary': {
                    'total_combinations': sum(p['total_combinations'] for p in predictions),
                    'total_cost': sum(p['investment_cost'] for p in predictions)
                }
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            print(f"💾 复式预测结果已保存到: {filepath}")
            
        except Exception as e:
            print(f"保存预测结果失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="大乐透复式投注预测器")
    parser.add_argument("-d", "--data", default="data/dlt_data_all.csv", help="数据文件路径")
    parser.add_argument("-p", "--periods", type=int, default=3000, help="分析期数")
    parser.add_argument("-m", "--method", choices=["hybrid", "markov"], default="hybrid", 
                       help="预测方法 (hybrid: 高级混合分析, markov: 马尔可夫链)")
    parser.add_argument("-c", "--combinations", help="复式组合，格式：6+2,7+3,8+4")
    parser.add_argument("--explain", action="store_true", help="显示详细过程")
    
    args = parser.parse_args()
    
    # 解析组合参数
    combinations = []
    if args.combinations:
        try:
            for combo in args.combinations.split(','):
                front, back = combo.strip().split('+')
                combinations.append((int(front), int(back)))
        except ValueError:
            print("❌ 组合格式错误，请使用格式：6+2,7+3,8+4")
            return
    else:
        # 默认组合
        combinations = [(6, 2), (7, 3)]
    
    # 创建预测器
    predictor = CompoundPredictor(args.data)
    
    # 执行预测
    predictions = predictor.predict_compound_combinations(
        periods=args.periods,
        combinations=combinations,
        method=args.method,
        explain=args.explain
    )
    
    if predictions:
        print(f"\n🎉 复式投注预测完成！")
        print(f"📊 基于 {args.periods} 期数据的 {len(predictions)} 注复式预测:")
        
        total_combinations = sum(p['total_combinations'] for p in predictions)
        total_cost = sum(p['investment_cost'] for p in predictions)
        
        for prediction in predictions:
            front_str = ' '.join([str(b).zfill(2) for b in prediction['front_balls']])
            back_str = ' '.join([str(b).zfill(2) for b in prediction['back_balls']])
            print(f"第 {prediction['index']} 注 ({prediction['combination']}): "
                  f"前区 {front_str} | 后区 {back_str} "
                  f"({prediction['total_combinations']:,}注, {prediction['investment_cost']:,}元)")
        
        print(f"\n💰 投注汇总:")
        print(f"   总组合数: {total_combinations:,} 注")
        print(f"   总投注额: {total_cost:,} 元")
    else:
        print("❌ 预测失败")


if __name__ == "__main__":
    main()
