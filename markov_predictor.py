#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
大乐透马尔可夫链预测器
专门用于基于马尔可夫链算法进行大乐透号码预测
"""

import os
import csv
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import random


class DLTMarkovPredictor:
    """大乐透马尔可夫链预测器"""
    
    def __init__(self, data_file):
        """初始化预测器
        
        Args:
            data_file: 数据文件路径
        """
        self.data_file = data_file
        self.df = None
        self.front_balls_lists = []
        self.back_balls_lists = []
        self.front_transition_matrix = defaultdict(lambda: defaultdict(float))
        self.back_transition_matrix = defaultdict(lambda: defaultdict(float))
        
        # 加载数据
        self.load_data()
        
    def load_data(self):
        """加载数据"""
        try:
            self.df = pd.read_csv(self.data_file)
            print(f"成功加载数据，共 {len(self.df)} 条记录")
            
            # 解析号码
            self._parse_ball_numbers()
            
            # 构建转移矩阵
            self._build_transition_matrices()
            
            return True
        except Exception as e:
            print(f"加载数据失败: {e}")
            return False
    
    def _parse_ball_numbers(self):
        """解析前区和后区号码"""
        self.front_balls_lists = []
        self.back_balls_lists = []
        
        for _, row in self.df.iterrows():
            # 解析前区号码
            front_balls_str = str(row["front_balls"])
            front_balls = [int(ball.strip()) for ball in front_balls_str.split(",")]
            self.front_balls_lists.append(sorted(front_balls))
            
            # 解析后区号码
            back_balls_str = str(row["back_balls"])
            back_balls = [int(ball.strip()) for ball in back_balls_str.split(",")]
            self.back_balls_lists.append(sorted(back_balls))
    
    def _build_transition_matrices(self):
        """构建马尔可夫链转移矩阵"""
        print("构建马尔可夫链转移矩阵...")
        
        # 构建前区转移矩阵
        for i in range(len(self.front_balls_lists) - 1):
            current_balls = self.front_balls_lists[i]
            next_balls = self.front_balls_lists[i + 1]
            
            for current_ball in current_balls:
                for next_ball in next_balls:
                    self.front_transition_matrix[current_ball][next_ball] += 1
        
        # 构建后区转移矩阵
        for i in range(len(self.back_balls_lists) - 1):
            current_balls = self.back_balls_lists[i]
            next_balls = self.back_balls_lists[i + 1]
            
            for current_ball in current_balls:
                for next_ball in next_balls:
                    self.back_transition_matrix[current_ball][next_ball] += 1
        
        # 归一化转移概率
        self._normalize_transition_matrices()
    
    def _normalize_transition_matrices(self):
        """归一化转移矩阵"""
        # 归一化前区转移矩阵
        for current_ball in self.front_transition_matrix:
            total = sum(self.front_transition_matrix[current_ball].values())
            if total > 0:
                for next_ball in self.front_transition_matrix[current_ball]:
                    self.front_transition_matrix[current_ball][next_ball] /= total
        
        # 归一化后区转移矩阵
        for current_ball in self.back_transition_matrix:
            total = sum(self.back_transition_matrix[current_ball].values())
            if total > 0:
                for next_ball in self.back_transition_matrix[current_ball]:
                    self.back_transition_matrix[current_ball][next_ball] /= total
    
    def predict_single_set(self, explain=False):
        """预测一注号码
        
        Args:
            explain: 是否显示预测过程
        
        Returns:
            (前区号码列表, 后区号码列表)
        """
        # 获取最近一期的号码作为起始状态
        latest_front = self.front_balls_lists[0]
        latest_back = self.back_balls_lists[0]
        
        if explain:
            print(f"最近一期号码: 前区 {','.join([str(b).zfill(2) for b in latest_front])}, 后区 {','.join([str(b).zfill(2) for b in latest_back])}")
            print("\n基于马尔可夫链状态转移概率预测:")
        
        # 预测前区号码
        front_candidates = defaultdict(float)
        
        for current_ball in latest_front:
            if current_ball in self.front_transition_matrix:
                for next_ball, prob in self.front_transition_matrix[current_ball].items():
                    front_candidates[next_ball] += prob
        
        # 选择概率最高的前区号码
        sorted_front_candidates = sorted(front_candidates.items(), key=lambda x: x[1], reverse=True)
        
        if explain:
            print("\n前区号码预测:")
            print("基于上期号码的转移概率，候选号码排名(前10):")
            for i, (ball, prob) in enumerate(sorted_front_candidates[:10]):
                print(f"  {ball:02d}: 概率 {prob:.4f}")
        
        # 选择前5个概率最高的号码，如果不足5个则随机补充
        predicted_front = []
        for ball, prob in sorted_front_candidates:
            if len(predicted_front) < 5:
                predicted_front.append(ball)
        
        # 如果预测的号码不足5个，随机补充
        if len(predicted_front) < 5:
            remaining_balls = [i for i in range(1, 36) if i not in predicted_front]
            random.shuffle(remaining_balls)
            predicted_front.extend(remaining_balls[:5-len(predicted_front)])
        
        predicted_front = sorted(predicted_front[:5])
        
        # 预测后区号码
        back_candidates = defaultdict(float)
        
        for current_ball in latest_back:
            if current_ball in self.back_transition_matrix:
                for next_ball, prob in self.back_transition_matrix[current_ball].items():
                    back_candidates[next_ball] += prob
        
        # 选择概率最高的后区号码
        sorted_back_candidates = sorted(back_candidates.items(), key=lambda x: x[1], reverse=True)
        
        if explain:
            print("\n后区号码预测:")
            print("基于上期号码的转移概率，候选号码排名:")
            for ball in range(1, 13):
                prob = back_candidates.get(ball, 0.0)
                print(f"  {ball:02d}: 概率 {prob:.4f}")
        
        # 选择前2个概率最高的号码，如果不足2个则随机补充
        predicted_back = []
        for ball, prob in sorted_back_candidates:
            if len(predicted_back) < 2:
                predicted_back.append(ball)
        
        # 如果预测的号码不足2个，随机补充
        if len(predicted_back) < 2:
            remaining_balls = [i for i in range(1, 13) if i not in predicted_back]
            random.shuffle(remaining_balls)
            predicted_back.extend(remaining_balls[:2-len(predicted_back)])
        
        predicted_back = sorted(predicted_back[:2])
        
        if explain:
            print(f"\n最终预测号码: 前区 {','.join([str(b).zfill(2) for b in predicted_front])}, 后区 {','.join([str(b).zfill(2) for b in predicted_back])}")
        
        return predicted_front, predicted_back
    
    def predict_multiple_sets(self, num_sets=5, explain=False):
        """预测多注号码

        Args:
            num_sets: 预测的注数
            explain: 是否显示预测过程

        Returns:
            预测结果列表，每个元素为(前区号码列表, 后区号码列表)
        """
        print(f"使用马尔可夫链预测 {num_sets} 注号码...")

        if explain:
            print("=" * 50)

        predictions = []
        seen_predictions = set()

        for i in range(num_sets):
            if explain:
                print(f"\n第 {i+1} 注预测:")
                print("-" * 30)

            # 为了增加多样性，在预测过程中加入一些随机性
            attempts = 0
            max_attempts = 20

            while attempts < max_attempts:
                if i == 0:
                    # 第一注使用标准预测
                    front_balls, back_balls = self.predict_single_set(explain=explain)
                else:
                    # 后续注数加入随机性
                    front_balls, back_balls = self._predict_with_variation(i, explain=explain)

                # 检查是否与之前的预测重复
                prediction_tuple = (tuple(front_balls), tuple(back_balls))
                if prediction_tuple not in seen_predictions:
                    predictions.append((front_balls, back_balls))
                    seen_predictions.add(prediction_tuple)
                    break

                attempts += 1

            # 如果尝试多次仍然重复，强制调整
            if attempts >= max_attempts:
                front_balls, back_balls = self._force_unique_prediction(predictions)
                predictions.append((front_balls, back_balls))
                seen_predictions.add((tuple(front_balls), tuple(back_balls)))

            if not explain:
                print(f"第 {i+1} 注: 前区 {' '.join([str(b).zfill(2) for b in front_balls])} | 后区 {' '.join([str(b).zfill(2) for b in back_balls])}")

        return predictions

    def _predict_with_variation(self, variation_level, explain=False):
        """带变化的预测方法

        Args:
            variation_level: 变化程度
            explain: 是否显示预测过程

        Returns:
            (前区号码列表, 后区号码列表)
        """
        # 获取最近一期的号码作为起始状态
        latest_front = self.front_balls_lists[0]
        latest_back = self.back_balls_lists[0]

        # 预测前区号码
        front_candidates = defaultdict(float)

        for current_ball in latest_front:
            if current_ball in self.front_transition_matrix:
                for next_ball, prob in self.front_transition_matrix[current_ball].items():
                    front_candidates[next_ball] += prob

        # 根据变化程度调整选择策略
        sorted_front_candidates = sorted(front_candidates.items(), key=lambda x: x[1], reverse=True)

        # 选择策略：混合高概率和随机选择
        predicted_front = []

        # 选择一些高概率号码
        high_prob_count = max(1, 5 - variation_level)
        for i in range(min(high_prob_count, len(sorted_front_candidates))):
            ball, prob = sorted_front_candidates[i]
            if ball not in predicted_front:
                predicted_front.append(ball)

        # 随机选择剩余号码
        remaining_balls = [i for i in range(1, 36) if i not in predicted_front]
        random.shuffle(remaining_balls)

        while len(predicted_front) < 5:
            predicted_front.append(remaining_balls.pop())

        predicted_front = sorted(predicted_front[:5])

        # 预测后区号码
        back_candidates = defaultdict(float)

        for current_ball in latest_back:
            if current_ball in self.back_transition_matrix:
                for next_ball, prob in self.back_transition_matrix[current_ball].items():
                    back_candidates[next_ball] += prob

        sorted_back_candidates = sorted(back_candidates.items(), key=lambda x: x[1], reverse=True)

        # 后区号码选择策略
        predicted_back = []

        if variation_level <= 2 and len(sorted_back_candidates) >= 2:
            # 选择高概率号码
            for i in range(min(2, len(sorted_back_candidates))):
                ball, prob = sorted_back_candidates[i]
                if ball not in predicted_back:
                    predicted_back.append(ball)

        # 随机补充
        remaining_back_balls = [i for i in range(1, 13) if i not in predicted_back]
        random.shuffle(remaining_back_balls)

        while len(predicted_back) < 2:
            predicted_back.append(remaining_back_balls.pop())

        predicted_back = sorted(predicted_back[:2])

        return predicted_front, predicted_back

    def _force_unique_prediction(self, existing_predictions):
        """强制生成唯一的预测结果

        Args:
            existing_predictions: 已有的预测结果

        Returns:
            (前区号码列表, 后区号码列表)
        """
        # 完全随机生成一个不重复的组合
        max_attempts = 100
        attempts = 0

        while attempts < max_attempts:
            front_balls = sorted(random.sample(range(1, 36), 5))
            back_balls = sorted(random.sample(range(1, 13), 2))

            prediction_tuple = (tuple(front_balls), tuple(back_balls))
            if prediction_tuple not in [(tuple(p[0]), tuple(p[1])) for p in existing_predictions]:
                return front_balls, back_balls

            attempts += 1

        # 如果还是重复，直接返回随机结果
        return sorted(random.sample(range(1, 36), 5)), sorted(random.sample(range(1, 13), 2))
    
    def _adjust_prediction(self, front_balls, back_balls, existing_predictions):
        """调整预测结果以避免重复
        
        Args:
            front_balls: 前区号码
            back_balls: 后区号码
            existing_predictions: 已有的预测结果
        
        Returns:
            调整后的(前区号码列表, 后区号码列表)
        """
        # 简单的调整策略：随机替换一个号码
        adjusted_front = front_balls.copy()
        adjusted_back = back_balls.copy()
        
        # 随机替换一个前区号码
        if random.random() < 0.7:  # 70%的概率调整前区
            replace_idx = random.randint(0, 4)
            new_ball = random.randint(1, 35)
            while new_ball in adjusted_front:
                new_ball = random.randint(1, 35)
            adjusted_front[replace_idx] = new_ball
            adjusted_front.sort()
        
        # 随机替换一个后区号码
        if random.random() < 0.5:  # 50%的概率调整后区
            replace_idx = random.randint(0, 1)
            new_ball = random.randint(1, 12)
            while new_ball in adjusted_back:
                new_ball = random.randint(1, 12)
            adjusted_back[replace_idx] = new_ball
            adjusted_back.sort()
        
        return adjusted_front, adjusted_back


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="大乐透马尔可夫链预测器")
    parser.add_argument("-d", "--data", default="data/dlt_data.csv", help="数据文件路径")
    parser.add_argument("-n", "--num", type=int, default=1, help="预测注数")
    parser.add_argument("--explain", action="store_true", help="显示预测过程")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data):
        print(f"数据文件不存在: {args.data}")
        print("请先运行爬虫获取数据:")
        print("python3 dlt_500_crawler.py -c 100 -o data/dlt_data.csv")
        return
    
    # 创建预测器
    predictor = DLTMarkovPredictor(args.data)
    
    # 进行预测
    if args.num == 1:
        front_balls, back_balls = predictor.predict_single_set(explain=args.explain)
        print(f"\n马尔可夫链预测号码: 前区 {' '.join([str(b).zfill(2) for b in front_balls])} | 后区 {' '.join([str(b).zfill(2) for b in back_balls])}")
    else:
        predictions = predictor.predict_multiple_sets(args.num, explain=args.explain)
        print(f"\n马尔可夫链预测 {args.num} 注号码:")
        for i, (front_balls, back_balls) in enumerate(predictions, 1):
            print(f"第 {i} 注: 前区 {' '.join([str(b).zfill(2) for b in front_balls])} | 后区 {' '.join([str(b).zfill(2) for b in back_balls])}")


if __name__ == "__main__":
    main()
