#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
大乐透数据分析与预测系统
整合了数据爬取、马尔可夫链分析、预测等所有功能
"""

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from datetime import datetime

# 可视化库
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from bs4 import BeautifulSoup

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class DLTCrawler:
    """大乐透数据爬虫 - 从500彩票网获取数据"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        
        self.api_url = "https://datachart.500.com/dlt/history/newinc/history.php"
    
    def get_history_data(self, count=50, get_all=False):
        """获取历史数据"""
        results = []
        
        try:
            if get_all:
                print("开始从500彩票网获取所有历史大乐透数据...")
                results = self._fetch_all_data()
                print(f"全量获取完成，共获取 {len(results)} 期大乐透开奖数据")
            else:
                print(f"开始从500彩票网获取最近{count}期大乐透数据...")
                results = self._fetch_limited_data(count)
                print(f"成功获取 {len(results)} 期大乐透开奖数据")
            
        except Exception as e:
            print(f"获取数据失败: {e}")
        
        return results
    
    def _fetch_all_data(self):
        """获取所有历史数据"""
        results = []
        
        try:
            params = {'limit': 2000, 'sort': 0}
            response = requests.get(self.api_url, headers=self.headers, params=params, timeout=60)
            response.encoding = 'gb2312'
            
            if response.status_code != 200:
                print(f"请求失败，状态码: {response.status_code}")
                return results
            
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('div', {'class': 'chart'})
            if not table:
                print("未找到开奖数据表格")
                return results
            
            rows = table.find_all('tr')
            print(f"找到 {len(rows)} 行数据，开始解析...")
            
            for i, row in enumerate(rows):
                try:
                    cells = row.find_all('td')
                    if len(cells) != 15:
                        continue
                    
                    issue = cells[0].get_text().strip()
                    date = cells[14].get_text().strip()
                    
                    front_balls = []
                    for j in range(1, 6):
                        ball = cells[j].get_text().strip()
                        if ball.isdigit():
                            front_balls.append(ball.zfill(2))
                    
                    back_balls = []
                    for j in range(6, 8):
                        ball = cells[j].get_text().strip()
                        if ball.isdigit():
                            back_balls.append(ball.zfill(2))
                    
                    if len(front_balls) == 5 and len(back_balls) == 2 and issue.isdigit():
                        result = {
                            "issue": issue,
                            "date": date,
                            "front_balls": ",".join(front_balls),
                            "back_balls": ",".join(back_balls)
                        }
                        results.append(result)
                        
                        if len(results) % 100 == 0:
                            print(f"已解析 {len(results)} 期数据...")
                    
                except Exception as e:
                    print(f"解析第{i+1}行数据失败: {e}")
                    continue
            
        except Exception as e:
            print(f"获取全量数据失败: {e}")
        
        return results
    
    def _fetch_limited_data(self, count):
        """获取指定数量的数据"""
        results = []
        
        try:
            params = {'limit': count, 'sort': 0}
            response = requests.get(self.api_url, headers=self.headers, params=params, timeout=30)
            response.encoding = 'gb2312'
            
            if response.status_code != 200:
                print(f"请求失败，状态码: {response.status_code}")
                return results
            
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('div', {'class': 'chart'})
            if not table:
                print("未找到开奖数据表格")
                return results
            
            rows = table.find_all('tr')
            
            for i, row in enumerate(rows):
                if len(results) >= count:
                    break
                    
                try:
                    cells = row.find_all('td')
                    if len(cells) != 15:
                        continue
                    
                    issue = cells[0].get_text().strip()
                    date = cells[14].get_text().strip()
                    
                    front_balls = []
                    for j in range(1, 6):
                        ball = cells[j].get_text().strip()
                        if ball.isdigit():
                            front_balls.append(ball.zfill(2))
                    
                    back_balls = []
                    for j in range(6, 8):
                        ball = cells[j].get_text().strip()
                        if ball.isdigit():
                            back_balls.append(ball.zfill(2))
                    
                    if len(front_balls) == 5 and len(back_balls) == 2 and issue.isdigit():
                        result = {
                            "issue": issue,
                            "date": date,
                            "front_balls": ",".join(front_balls),
                            "back_balls": ",".join(back_balls)
                        }
                        results.append(result)
                        print(f"获取第{issue}期数据: 前区 {','.join(front_balls)}, 后区 {','.join(back_balls)}")
                    
                except Exception as e:
                    print(f"解析第{i+1}行数据失败: {e}")
                    continue
            
        except Exception as e:
            print(f"获取限量数据失败: {e}")
        
        return results
    
    def save_to_csv(self, results, filename="dlt_data.csv"):
        """保存数据到CSV文件"""
        if not results:
            print("没有数据需要保存")
            return None
        
        try:
            if os.path.dirname(filename):
                file_path = filename
            else:
                file_path = os.path.join(self.data_dir, filename)
            
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['issue', 'date', 'front_balls', 'back_balls']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in results:
                    writer.writerow(result)
            
            print(f"数据已保存到: {file_path}")
            print(f"共保存 {len(results)} 条记录")
            
            return file_path
            
        except Exception as e:
            print(f"保存数据失败: {e}")
            return None


class DLTAnalyzer:
    """大乐透分析器 - 整合了所有分析功能"""
    
    def __init__(self, data_file):
        self.data_file = data_file
        self.df = None
        self.analysis_data = None
        self.load_data()
    
    def load_data(self):
        """加载数据"""
        try:
            self.df = pd.read_csv(self.data_file)
            self.df = self.df.sort_values('issue', ascending=False)
            print(f"成功加载数据，共 {len(self.df)} 条记录")
            print(f"数据范围: {self.df.iloc[-1]['issue']} - {self.df.iloc[0]['issue']}")
            return True
        except Exception as e:
            print(f"加载数据失败: {e}")
            return False
    
    def parse_balls(self, balls_str):
        """解析号码字符串"""
        return [int(ball.strip()) for ball in str(balls_str).split(",")]
    
    def check_duplicates(self, quiet=False):
        """检查重复数据"""
        if not quiet:
            print("检查数据重复...")
        
        duplicates = self.df[self.df.duplicated(subset=['issue'], keep=False)]
        
        if len(duplicates) > 0:
            print(f"发现 {len(duplicates)} 条重复记录:")
            for _, row in duplicates.iterrows():
                print(f"  期号: {row['issue']}, 日期: {row['date']}")
            return True
        else:
            if not quiet:
                print("未发现重复数据")
            return False
    
    def remove_duplicates(self):
        """去除重复数据"""
        original_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=['issue'], keep='first')
        removed_count = original_count - len(self.df)
        
        if removed_count > 0:
            print(f"已去除 {removed_count} 条重复记录")
            # 保存去重后的数据
            self.df.to_csv(self.data_file, index=False)
            print(f"去重后的数据已保存到: {self.data_file}")
        else:
            print("没有重复数据需要去除")
        
        return removed_count

    def analyze_periods(self, num_periods=300, verbose=True):
        """分析指定期数的数据"""
        if num_periods > len(self.df):
            print(f"警告: 请求分析 {num_periods} 期，但只有 {len(self.df)} 期数据，将使用全部数据")
            num_periods = len(self.df)

        analysis_df = self.df.head(num_periods)

        if verbose:
            print(f"\n开始分析最新 {num_periods} 期数据...")
            print(f"分析范围: {analysis_df.iloc[-1]['issue']} - {analysis_df.iloc[0]['issue']}")

        # 构建转移矩阵
        front_transitions = defaultdict(lambda: defaultdict(int))
        back_transitions = defaultdict(lambda: defaultdict(int))

        # 统计号码出现频率
        front_frequency = defaultdict(int)
        back_frequency = defaultdict(int)

        # 解析数据并构建转移关系
        for i in range(len(analysis_df) - 1):
            current_row = analysis_df.iloc[i]
            next_row = analysis_df.iloc[i + 1]

            current_front = self.parse_balls(current_row['front_balls'])
            current_back = self.parse_balls(current_row['back_balls'])
            next_front = self.parse_balls(next_row['front_balls'])
            next_back = self.parse_balls(next_row['back_balls'])

            # 统计频率
            for ball in current_front:
                front_frequency[ball] += 1
            for ball in current_back:
                back_frequency[ball] += 1

            # 构建转移关系
            for curr_ball in current_front:
                for next_ball in next_front:
                    front_transitions[curr_ball][next_ball] += 1

            for curr_ball in current_back:
                for next_ball in next_back:
                    back_transitions[curr_ball][next_ball] += 1

        # 计算转移概率
        front_probabilities = self._calculate_probabilities(front_transitions)
        back_probabilities = self._calculate_probabilities(back_transitions)

        # 计算稳定性得分
        front_stability = self._calculate_stability_scores(front_probabilities, front_frequency)
        back_stability = self._calculate_stability_scores(back_probabilities, back_frequency)

        self.analysis_data = {
            'num_periods': num_periods,
            'data_range': {
                'start': analysis_df.iloc[-1]['issue'],
                'end': analysis_df.iloc[0]['issue']
            },
            'latest_draw': {
                'issue': analysis_df.iloc[0]['issue'],
                'date': analysis_df.iloc[0]['date'],
                'front_balls': self.parse_balls(analysis_df.iloc[0]['front_balls']),
                'back_balls': self.parse_balls(analysis_df.iloc[0]['back_balls'])
            },
            'front_transitions': front_transitions,
            'back_transitions': back_transitions,
            'front_probabilities': front_probabilities,
            'back_probabilities': back_probabilities,
            'front_frequency': front_frequency,
            'back_frequency': back_frequency,
            'front_stability': front_stability,
            'back_stability': back_stability
        }

        if verbose:
            self._print_analysis_summary()

        # 保存马尔可夫链分析结果
        self._save_markov_analysis(self.analysis_data)

        return self.analysis_data

    def _calculate_probabilities(self, transitions):
        """计算转移概率"""
        probabilities = {}
        for from_ball in transitions:
            total = sum(transitions[from_ball].values())
            if total > 0:
                probabilities[from_ball] = {}
                for to_ball, count in transitions[from_ball].items():
                    probabilities[from_ball][to_ball] = count / total
        return probabilities

    def _calculate_stability_scores(self, probabilities, frequency):
        """计算稳定性得分"""
        stability = {}
        max_ball = 35 if len(frequency) > 12 else 12

        for ball in range(1, max_ball + 1):
            freq_score = frequency.get(ball, 0) / max(frequency.values()) if frequency else 0

            if ball in probabilities:
                probs = list(probabilities[ball].values())
                variance = np.var(probs) if probs else 1.0
                prob_stability = 1.0 / (variance + 0.001)
            else:
                prob_stability = 0.1

            stability[ball] = freq_score * 0.6 + prob_stability * 0.4

        return stability

    def _print_analysis_summary(self):
        """打印分析摘要"""
        data = self.analysis_data
        print(f"\n分析摘要:")
        print(f"分析期数: {data['num_periods']} 期")
        print(f"数据范围: {data['data_range']['start']} - {data['data_range']['end']}")
        print(f"最新一期: {data['latest_draw']['issue']} ({data['latest_draw']['date']})")

        latest_front = data['latest_draw']['front_balls']
        latest_back = data['latest_draw']['back_balls']
        print(f"最新号码: 前区 {' '.join([str(b).zfill(2) for b in latest_front])}, 后区 {' '.join([str(b).zfill(2) for b in latest_back])}")

        sorted_front = sorted(data['front_stability'].items(), key=lambda x: x[1], reverse=True)
        sorted_back = sorted(data['back_stability'].items(), key=lambda x: x[1], reverse=True)

        print(f"\n前区最稳定号码 (前5): {', '.join([f'{ball:02d}' for ball, _ in sorted_front[:5]])}")
        print(f"后区最稳定号码 (前3): {', '.join([f'{ball:02d}' for ball, _ in sorted_back[:3]])}")

    def predict_numbers(self, num_predictions=1, explain=False):
        """预测号码"""
        if not self.analysis_data:
            print("请先运行 analyze_periods() 进行分析")
            return []

        print(f"\n基于 {self.analysis_data['num_periods']} 期数据生成 {num_predictions} 注预测...")

        predictions = []
        used_combinations = set()

        for i in range(num_predictions):
            if explain and i == 0:
                print(f"\n第 {i+1} 注预测过程:")
                print("-" * 40)

            front_balls, back_balls = self._generate_single_prediction(i, explain and i == 0)

            combination = (tuple(sorted(front_balls)), tuple(sorted(back_balls)))
            attempts = 0
            while combination in used_combinations and attempts < 50:
                front_balls, back_balls = self._generate_single_prediction(i + attempts, False)
                combination = (tuple(sorted(front_balls)), tuple(sorted(back_balls)))
                attempts += 1

            used_combinations.add(combination)

            stability_score = self._calculate_prediction_stability(front_balls, back_balls)

            predictions.append({
                'front': sorted(front_balls),
                'back': sorted(back_balls),
                'stability_score': stability_score
            })

        predictions.sort(key=lambda x: x['stability_score'], reverse=True)

        print(f"\n预测结果 (按稳定性排序):")
        print("=" * 60)

        for i, pred in enumerate(predictions, 1):
            front_str = ' '.join([str(b).zfill(2) for b in pred['front']])
            back_str = ' '.join([str(b).zfill(2) for b in pred['back']])
            print(f"第 {i} 注: 前区 {front_str} | 后区 {back_str} (稳定性: {pred['stability_score']:.4f})")

        print(f"\n🎯 最稳定预测: 前区 {' '.join([str(b).zfill(2) for b in predictions[0]['front']])} | 后区 {' '.join([str(b).zfill(2) for b in predictions[0]['back']])}")

        return predictions

    def _generate_single_prediction(self, variation_level, explain=False):
        """生成单注预测"""
        data = self.analysis_data
        latest_front = data['latest_draw']['front_balls']
        latest_back = data['latest_draw']['back_balls']

        if explain:
            print(f"基于最新一期号码: 前区 {' '.join([str(b).zfill(2) for b in latest_front])}, 后区 {' '.join([str(b).zfill(2) for b in latest_back])}")

        # 预测前区号码
        front_candidates = defaultdict(float)

        for current_ball in latest_front:
            if current_ball in data['front_probabilities']:
                for next_ball, prob in data['front_probabilities'][current_ball].items():
                    front_candidates[next_ball] += prob * 0.7

        for ball, stability in data['front_stability'].items():
            front_candidates[ball] += stability * 0.3

        sorted_front = sorted(front_candidates.items(), key=lambda x: x[1], reverse=True)

        if explain:
            print("\n前区候选号码 (前10):")
            for i, (ball, score) in enumerate(sorted_front[:10], 1):
                print(f"  {i:2d}. {ball:02d}号 (得分: {score:.4f})")

        if variation_level == 0:
            selected_front = [ball for ball, _ in sorted_front[:5]]
        else:
            high_score = [ball for ball, _ in sorted_front[:8]]
            random.shuffle(high_score)
            selected_front = high_score[:5]

        # 预测后区号码
        back_candidates = defaultdict(float)

        for current_ball in latest_back:
            if current_ball in data['back_probabilities']:
                for next_ball, prob in data['back_probabilities'][current_ball].items():
                    back_candidates[next_ball] += prob * 0.7

        for ball, stability in data['back_stability'].items():
            back_candidates[ball] += stability * 0.3

        sorted_back = sorted(back_candidates.items(), key=lambda x: x[1], reverse=True)

        if explain:
            print("\n后区候选号码:")
            for i, (ball, score) in enumerate(sorted_back, 1):
                print(f"  {i:2d}. {ball:02d}号 (得分: {score:.4f})")

        if variation_level == 0:
            selected_back = [ball for ball, _ in sorted_back[:2]]
        else:
            high_score_back = [ball for ball, _ in sorted_back[:4]]
            random.shuffle(high_score_back)
            selected_back = high_score_back[:2]

        return selected_front, selected_back

    def _calculate_prediction_stability(self, front_balls, back_balls):
        """计算预测的稳定性得分"""
        data = self.analysis_data

        front_score = sum(data['front_stability'].get(ball, 0) for ball in front_balls) / len(front_balls)
        back_score = sum(data['back_stability'].get(ball, 0) for ball in back_balls) / len(back_balls)

        return (front_score + back_score) / 2

    def basic_analysis(self, save_results=True):
        """基础统计分析"""
        print("\n开始基础统计分析...")

        # 号码频率分析
        front_frequency = defaultdict(int)
        back_frequency = defaultdict(int)

        for _, row in self.df.iterrows():
            front_balls = self.parse_balls(row['front_balls'])
            back_balls = self.parse_balls(row['back_balls'])

            for ball in front_balls:
                front_frequency[ball] += 1
            for ball in back_balls:
                back_frequency[ball] += 1

        # 计算遗漏值
        front_missing = self._calculate_missing_values('front')
        back_missing = self._calculate_missing_values('back')

        # 热门号分析
        front_hot = sorted(front_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
        back_hot = sorted(back_frequency.items(), key=lambda x: x[1], reverse=True)[:5]

        # 冷门号分析
        front_cold = sorted(front_frequency.items(), key=lambda x: x[1])[:10]
        back_cold = sorted(back_frequency.items(), key=lambda x: x[1])[:5]

        results = {
            'total_periods': len(self.df),
            'front_frequency': dict(front_frequency),
            'back_frequency': dict(back_frequency),
            'front_missing': front_missing,
            'back_missing': back_missing,
            'front_hot_numbers': front_hot,
            'back_hot_numbers': back_hot,
            'front_cold_numbers': front_cold,
            'back_cold_numbers': back_cold
        }

        # 打印结果
        print(f"\n基础分析结果 (共{len(self.df)}期数据):")
        print("=" * 50)

        print("\n前区热门号码 (前10):")
        for i, (ball, count) in enumerate(front_hot, 1):
            freq = count / len(self.df) * 100
            print(f"  {i:2d}. {ball:02d}号: 出现{count:3d}次 (频率{freq:.1f}%)")

        print("\n后区热门号码:")
        for i, (ball, count) in enumerate(back_hot, 1):
            freq = count / len(self.df) * 100
            print(f"  {i:2d}. {ball:02d}号: 出现{count:3d}次 (频率{freq:.1f}%)")

        print(f"\n前区遗漏值最大的号码: {max(front_missing.items(), key=lambda x: x[1])}")
        print(f"后区遗漏值最大的号码: {max(back_missing.items(), key=lambda x: x[1])}")

        if save_results:
            self._save_basic_results(results)

        return results

    def _calculate_missing_values(self, ball_type):
        """计算遗漏值"""
        missing = {}
        max_ball = 35 if ball_type == 'front' else 12

        for ball in range(1, max_ball + 1):
            missing[ball] = 0

            for i, (_, row) in enumerate(self.df.iterrows()):
                balls = self.parse_balls(row[f'{ball_type}_balls'])
                if ball in balls:
                    missing[ball] = 0
                else:
                    missing[ball] += 1

                if i == 0:  # 只计算到最新一期
                    break

        return missing

    def _save_basic_results(self, results):
        """保存基础分析结果"""
        output_dir = "output/basic"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 保存JSON结果
        with open(f"{output_dir}/basic_analysis.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"基础分析结果已保存到: {output_dir}/basic_analysis.json")

    def bayesian_analysis(self, save_results=True):
        """贝叶斯分析"""
        print("\n开始贝叶斯分析...")

        # 计算先验概率
        front_prior = self._calculate_prior_probability('front')
        back_prior = self._calculate_prior_probability('back')

        # 计算条件概率
        front_conditional = self._calculate_conditional_probability('front')
        back_conditional = self._calculate_conditional_probability('back')

        # 计算后验概率
        front_posterior = self._calculate_posterior_probability(front_prior, front_conditional)
        back_posterior = self._calculate_posterior_probability(back_prior, back_conditional)

        results = {
            'front_prior': front_prior,
            'back_prior': back_prior,
            'front_conditional': front_conditional,
            'back_conditional': back_conditional,
            'front_posterior': front_posterior,
            'back_posterior': back_posterior
        }

        # 打印结果
        print("\n贝叶斯分析结果:")
        print("=" * 50)

        # 显示前区最高概率号码
        sorted_front = sorted(front_posterior.items(), key=lambda x: x[1], reverse=True)
        print("\n前区后验概率最高的号码 (前10):")
        for i, (ball, prob) in enumerate(sorted_front[:10], 1):
            print(f"  {i:2d}. {ball:02d}号: 概率 {prob:.4f}")

        # 显示后区最高概率号码
        sorted_back = sorted(back_posterior.items(), key=lambda x: x[1], reverse=True)
        print("\n后区后验概率最高的号码:")
        for i, (ball, prob) in enumerate(sorted_back, 1):
            print(f"  {i:2d}. {ball:02d}号: 概率 {prob:.4f}")

        if save_results:
            self._save_bayesian_results(results)

        return results

    def _calculate_prior_probability(self, ball_type):
        """计算先验概率"""
        frequency = defaultdict(int)
        total_count = 0

        for _, row in self.df.iterrows():
            balls = self.parse_balls(row[f'{ball_type}_balls'])
            for ball in balls:
                frequency[ball] += 1
                total_count += 1

        prior = {}
        max_ball = 35 if ball_type == 'front' else 12
        for ball in range(1, max_ball + 1):
            prior[ball] = frequency[ball] / total_count if total_count > 0 else 0

        return prior

    def _calculate_conditional_probability(self, ball_type):
        """计算条件概率"""
        conditional = defaultdict(lambda: defaultdict(int))
        ball_counts = defaultdict(int)

        for i in range(len(self.df) - 1):
            current_balls = self.parse_balls(self.df.iloc[i][f'{ball_type}_balls'])
            next_balls = self.parse_balls(self.df.iloc[i + 1][f'{ball_type}_balls'])

            for curr_ball in current_balls:
                ball_counts[curr_ball] += 1
                for next_ball in next_balls:
                    conditional[curr_ball][next_ball] += 1

        # 归一化
        normalized_conditional = {}
        for curr_ball in conditional:
            total = ball_counts[curr_ball]
            normalized_conditional[curr_ball] = {}
            max_ball = 35 if ball_type == 'front' else 12
            for next_ball in range(1, max_ball + 1):
                normalized_conditional[curr_ball][next_ball] = conditional[curr_ball][next_ball] / total if total > 0 else 0

        return normalized_conditional

    def _calculate_posterior_probability(self, prior, conditional):
        """计算后验概率"""
        # 获取最新一期号码
        latest_balls = self.parse_balls(self.df.iloc[0]['front_balls']) if 'front' in str(conditional) else self.parse_balls(self.df.iloc[0]['back_balls'])

        posterior = {}
        max_ball = 35 if len(prior) > 12 else 12

        for ball in range(1, max_ball + 1):
            # 贝叶斯公式: P(ball|evidence) = P(evidence|ball) * P(ball) / P(evidence)
            likelihood = 1.0
            for latest_ball in latest_balls:
                if latest_ball in conditional:
                    likelihood *= conditional[latest_ball].get(ball, 0.001)

            posterior[ball] = likelihood * prior.get(ball, 0.001)

        # 归一化
        total_posterior = sum(posterior.values())
        if total_posterior > 0:
            for ball in posterior:
                posterior[ball] /= total_posterior

        return posterior

    def _save_bayesian_results(self, results):
        """保存贝叶斯分析结果"""
        output_dir = "output/advanced"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 保存JSON结果
        with open(f"{output_dir}/bayesian_analysis.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"贝叶斯分析结果已保存到: {output_dir}/bayesian_analysis.json")

    def _save_markov_analysis(self, analysis_data):
        """保存马尔可夫链分析结果"""
        output_dir = "output/advanced"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 准备保存的数据
        markov_results = {
            'analysis_info': {
                'num_periods': analysis_data['num_periods'],
                'data_range': analysis_data['data_range'],
                'latest_draw': analysis_data['latest_draw'],
                'analysis_date': datetime.now().isoformat()
            },
            'front_transition_probs': {},
            'back_transition_probs': {},
            'front_stability_scores': {str(k): float(v) for k, v in analysis_data['front_stability'].items()},
            'back_stability_scores': {str(k): float(v) for k, v in analysis_data['back_stability'].items()},
            'front_frequency': {str(k): int(v) for k, v in analysis_data['front_frequency'].items()},
            'back_frequency': {str(k): int(v) for k, v in analysis_data['back_frequency'].items()}
        }

        # 转换转移概率为可序列化格式
        for from_ball, to_balls in analysis_data['front_probabilities'].items():
            markov_results['front_transition_probs'][str(from_ball)] = {}
            for to_ball in range(1, 36):
                markov_results['front_transition_probs'][str(from_ball)][str(to_ball)] = to_balls.get(to_ball, 0.0)

        for from_ball, to_balls in analysis_data['back_probabilities'].items():
            markov_results['back_transition_probs'][str(from_ball)] = {}
            for to_ball in range(1, 13):
                markov_results['back_transition_probs'][str(from_ball)][str(to_ball)] = to_balls.get(to_ball, 0.0)

        # 保存JSON结果
        with open(f"{output_dir}/markov_chain_analysis.json", 'w', encoding='utf-8') as f:
            json.dump(markov_results, f, ensure_ascii=False, indent=2, default=str)

        print(f"马尔可夫链分析结果已保存到: {output_dir}/markov_chain_analysis.json")

    def probability_analysis(self, save_results=True):
        """概率分析"""
        print("\n开始概率分析...")

        # 计算各种概率
        single_ball_probs = self._calculate_single_ball_probabilities()
        combination_probs = self._calculate_combination_probabilities()
        pattern_probs = self._calculate_pattern_probabilities()

        results = {
            'single_ball_probabilities': single_ball_probs,
            'combination_probabilities': combination_probs,
            'pattern_probabilities': pattern_probs
        }

        # 打印结果
        print("\n概率分析结果:")
        print("=" * 50)

        print("\n单球出现概率 (前区前10):")
        sorted_front = sorted(single_ball_probs['front'].items(), key=lambda x: x[1], reverse=True)
        for i, (ball, prob) in enumerate(sorted_front[:10], 1):
            print(f"  {i:2d}. {ball:02d}号: {prob:.4f} ({prob*100:.2f}%)")

        print("\n单球出现概率 (后区):")
        sorted_back = sorted(single_ball_probs['back'].items(), key=lambda x: x[1], reverse=True)
        for i, (ball, prob) in enumerate(sorted_back, 1):
            print(f"  {i:2d}. {ball:02d}号: {prob:.4f} ({prob*100:.2f}%)")

        if save_results:
            self._save_probability_results(results)

        return results

    def _calculate_single_ball_probabilities(self):
        """计算单球出现概率"""
        front_count = defaultdict(int)
        back_count = defaultdict(int)
        total_draws = len(self.df)

        for _, row in self.df.iterrows():
            front_balls = self.parse_balls(row['front_balls'])
            back_balls = self.parse_balls(row['back_balls'])

            for ball in front_balls:
                front_count[ball] += 1
            for ball in back_balls:
                back_count[ball] += 1

        # 计算概率
        front_probs = {}
        back_probs = {}

        for ball in range(1, 36):
            front_probs[ball] = front_count[ball] / total_draws

        for ball in range(1, 13):
            back_probs[ball] = back_count[ball] / total_draws

        return {'front': front_probs, 'back': back_probs}

    def _calculate_combination_probabilities(self):
        """计算组合出现概率"""
        # 计算常见组合的概率
        front_pairs = defaultdict(int)
        back_pairs = defaultdict(int)
        total_draws = len(self.df)

        for _, row in self.df.iterrows():
            front_balls = self.parse_balls(row['front_balls'])
            back_balls = self.parse_balls(row['back_balls'])

            # 前区两两组合
            for i in range(len(front_balls)):
                for j in range(i+1, len(front_balls)):
                    pair = tuple(sorted([front_balls[i], front_balls[j]]))
                    front_pairs[pair] += 1

            # 后区组合
            if len(back_balls) == 2:
                pair = tuple(sorted(back_balls))
                back_pairs[pair] += 1

        # 转换为概率
        front_pair_probs = {pair: count/total_draws for pair, count in front_pairs.items()}
        back_pair_probs = {pair: count/total_draws for pair, count in back_pairs.items()}

        return {
            'front_pairs': front_pair_probs,
            'back_pairs': back_pair_probs
        }

    def _calculate_pattern_probabilities(self):
        """计算模式概率"""
        odd_even_patterns = defaultdict(int)
        size_patterns = defaultdict(int)
        sum_ranges = defaultdict(int)
        total_draws = len(self.df)

        for _, row in self.df.iterrows():
            front_balls = self.parse_balls(row['front_balls'])
            back_balls = self.parse_balls(row['back_balls'])

            # 奇偶模式
            front_odd = sum(1 for ball in front_balls if ball % 2 == 1)
            back_odd = sum(1 for ball in back_balls if ball % 2 == 1)
            odd_even_patterns[f"前区{front_odd}奇{5-front_odd}偶_后区{back_odd}奇{2-back_odd}偶"] += 1

            # 大小模式
            front_small = sum(1 for ball in front_balls if ball <= 17)
            back_small = sum(1 for ball in back_balls if ball <= 6)
            size_patterns[f"前区{front_small}小{5-front_small}大_后区{back_small}小{2-back_small}大"] += 1

            # 和值范围
            front_sum = sum(front_balls)
            if front_sum <= 70:
                sum_range = "小和值(≤70)"
            elif front_sum <= 110:
                sum_range = "中和值(71-110)"
            else:
                sum_range = "大和值(≥111)"
            sum_ranges[sum_range] += 1

        # 转换为概率
        pattern_probs = {
            'odd_even': {pattern: count/total_draws for pattern, count in odd_even_patterns.items()},
            'size': {pattern: count/total_draws for pattern, count in size_patterns.items()},
            'sum_ranges': {range_name: count/total_draws for range_name, count in sum_ranges.items()}
        }

        return pattern_probs

    def _save_probability_results(self, results):
        """保存概率分析结果"""
        output_dir = "output/advanced"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 转换tuple键为字符串以便JSON序列化
        serializable_results = {
            'single_ball_probabilities': results['single_ball_probabilities'],
            'combination_probabilities': {
                'front_pairs': {str(k): v for k, v in results['combination_probabilities']['front_pairs'].items()},
                'back_pairs': {str(k): v for k, v in results['combination_probabilities']['back_pairs'].items()}
            },
            'pattern_probabilities': results['pattern_probabilities']
        }

        # 保存JSON结果
        with open(f"{output_dir}/probability_analysis.json", 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)

        print(f"概率分析结果已保存到: {output_dir}/probability_analysis.json")

    def frequency_pattern_analysis(self, save_results=True):
        """频率模式分析"""
        print("\n开始频率模式分析...")

        # 分析号码组合模式
        combination_patterns = self._analyze_combination_patterns()

        # 分析奇偶模式
        odd_even_patterns = self._analyze_odd_even_patterns()

        # 分析大小模式
        size_patterns = self._analyze_size_patterns()

        # 分析连号模式
        consecutive_patterns = self._analyze_consecutive_patterns()

        results = {
            'combination_patterns': combination_patterns,
            'odd_even_patterns': odd_even_patterns,
            'size_patterns': size_patterns,
            'consecutive_patterns': consecutive_patterns
        }

        # 打印结果
        print("\n频率模式分析结果:")
        print("=" * 50)

        print("\n奇偶模式分布:")
        for pattern, count in sorted(odd_even_patterns['front'].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  前区{pattern}: {count}次")

        print("\n大小模式分布:")
        for pattern, count in sorted(size_patterns['front'].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  前区{pattern}: {count}次")

        if save_results:
            self._save_frequency_results(results)

        return results

    def _analyze_combination_patterns(self):
        """分析号码组合模式"""
        patterns = {'front': defaultdict(int), 'back': defaultdict(int)}

        for _, row in self.df.iterrows():
            front_balls = self.parse_balls(row['front_balls'])
            back_balls = self.parse_balls(row['back_balls'])

            # 前区模式
            front_pattern = tuple(sorted(front_balls))
            patterns['front'][front_pattern] += 1

            # 后区模式
            back_pattern = tuple(sorted(back_balls))
            patterns['back'][back_pattern] += 1

        return patterns

    def _analyze_odd_even_patterns(self):
        """分析奇偶模式"""
        patterns = {'front': defaultdict(int), 'back': defaultdict(int)}

        for _, row in self.df.iterrows():
            front_balls = self.parse_balls(row['front_balls'])
            back_balls = self.parse_balls(row['back_balls'])

            # 前区奇偶
            odd_count = sum(1 for ball in front_balls if ball % 2 == 1)
            even_count = 5 - odd_count
            front_pattern = f"{odd_count}奇{even_count}偶"
            patterns['front'][front_pattern] += 1

            # 后区奇偶
            odd_count = sum(1 for ball in back_balls if ball % 2 == 1)
            even_count = 2 - odd_count
            back_pattern = f"{odd_count}奇{even_count}偶"
            patterns['back'][back_pattern] += 1

        return patterns

    def _analyze_size_patterns(self):
        """分析大小模式"""
        patterns = {'front': defaultdict(int), 'back': defaultdict(int)}

        for _, row in self.df.iterrows():
            front_balls = self.parse_balls(row['front_balls'])
            back_balls = self.parse_balls(row['back_balls'])

            # 前区大小 (1-17小，18-35大)
            small_count = sum(1 for ball in front_balls if ball <= 17)
            big_count = 5 - small_count
            front_pattern = f"{small_count}小{big_count}大"
            patterns['front'][front_pattern] += 1

            # 后区大小 (1-6小，7-12大)
            small_count = sum(1 for ball in back_balls if ball <= 6)
            big_count = 2 - small_count
            back_pattern = f"{small_count}小{big_count}大"
            patterns['back'][back_pattern] += 1

        return patterns

    def _analyze_consecutive_patterns(self):
        """分析连号模式"""
        patterns = {'front': defaultdict(int), 'back': defaultdict(int)}

        for _, row in self.df.iterrows():
            front_balls = sorted(self.parse_balls(row['front_balls']))
            back_balls = sorted(self.parse_balls(row['back_balls']))

            # 前区连号
            consecutive_count = 0
            for i in range(len(front_balls) - 1):
                if front_balls[i + 1] - front_balls[i] == 1:
                    consecutive_count += 1
            patterns['front'][consecutive_count] += 1

            # 后区连号
            consecutive_count = 0
            for i in range(len(back_balls) - 1):
                if back_balls[i + 1] - back_balls[i] == 1:
                    consecutive_count += 1
            patterns['back'][consecutive_count] += 1

        return patterns

    def _save_frequency_results(self, results):
        """保存频率分析结果"""
        output_dir = "output/advanced"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 转换tuple键为字符串以便JSON序列化
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serializable_results[key] = {}
                for ball_type, patterns in value.items():
                    serializable_results[key][ball_type] = {}
                    for pattern, count in patterns.items():
                        pattern_str = str(pattern) if isinstance(pattern, tuple) else pattern
                        serializable_results[key][ball_type][pattern_str] = count
            else:
                serializable_results[key] = value

        # 保存JSON结果
        with open(f"{output_dir}/frequency_analysis.json", 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)

        print(f"频率分析结果已保存到: {output_dir}/frequency_analysis.json")

    def trend_analysis(self, periods=50):
        """走势分析"""
        print(f"\n开始走势分析 (最近{periods}期)...")

        recent_data = self.df.head(periods)

        # 号码走势
        front_trends = self._calculate_number_trends(recent_data, 'front')
        back_trends = self._calculate_number_trends(recent_data, 'back')

        # 和值走势
        sum_trends = self._calculate_sum_trends(recent_data)

        # 跨度走势
        span_trends = self._calculate_span_trends(recent_data)

        results = {
            'periods': periods,
            'front_trends': front_trends,
            'back_trends': back_trends,
            'sum_trends': sum_trends,
            'span_trends': span_trends
        }

        # 打印结果
        print("\n走势分析结果:")
        print("=" * 50)

        print(f"\n最近{periods}期和值走势:")
        print(f"  平均和值: {np.mean(sum_trends['front']):.1f}")
        print(f"  最大和值: {max(sum_trends['front'])}")
        print(f"  最小和值: {min(sum_trends['front'])}")

        print(f"\n最近{periods}期跨度走势:")
        print(f"  平均跨度: {np.mean(span_trends['front']):.1f}")
        print(f"  最大跨度: {max(span_trends['front'])}")
        print(f"  最小跨度: {min(span_trends['front'])}")

        return results

    def _calculate_number_trends(self, data, ball_type):
        """计算号码走势"""
        trends = []
        for _, row in data.iterrows():
            balls = self.parse_balls(row[f'{ball_type}_balls'])
            trends.append(balls)
        return trends

    def _calculate_sum_trends(self, data):
        """计算和值走势"""
        trends = {'front': [], 'back': []}

        for _, row in data.iterrows():
            front_balls = self.parse_balls(row['front_balls'])
            back_balls = self.parse_balls(row['back_balls'])

            trends['front'].append(sum(front_balls))
            trends['back'].append(sum(back_balls))

        return trends

    def _calculate_span_trends(self, data):
        """计算跨度走势"""
        trends = {'front': [], 'back': []}

        for _, row in data.iterrows():
            front_balls = self.parse_balls(row['front_balls'])
            back_balls = self.parse_balls(row['back_balls'])

            trends['front'].append(max(front_balls) - min(front_balls))
            trends['back'].append(max(back_balls) - min(back_balls))

        return trends

    def frequency_based_prediction(self, num_predictions=1):
        """基于频率分析的预测"""
        print(f"\n基于频率分析生成 {num_predictions} 注预测...")

        # 获取频率数据
        front_frequency = defaultdict(int)
        back_frequency = defaultdict(int)

        for _, row in self.df.iterrows():
            front_balls = self.parse_balls(row['front_balls'])
            back_balls = self.parse_balls(row['back_balls'])

            for ball in front_balls:
                front_frequency[ball] += 1
            for ball in back_balls:
                back_frequency[ball] += 1

        predictions = []
        used_combinations = set()

        for i in range(num_predictions):
            # 基于频率选择号码
            front_candidates = sorted(front_frequency.items(), key=lambda x: x[1], reverse=True)
            back_candidates = sorted(back_frequency.items(), key=lambda x: x[1], reverse=True)

            # 添加随机性避免重复
            front_selected = []
            back_selected = []

            # 选择前区号码
            available_front = [ball for ball, _ in front_candidates]
            random.shuffle(available_front)
            front_selected = available_front[:5]

            # 选择后区号码
            available_back = [ball for ball, _ in back_candidates]
            random.shuffle(available_back)
            back_selected = available_back[:2]

            # 检查重复
            combination = (tuple(sorted(front_selected)), tuple(sorted(back_selected)))
            if combination not in used_combinations:
                used_combinations.add(combination)
                predictions.append({
                    'front': sorted(front_selected),
                    'back': sorted(back_selected),
                    'method': 'frequency_based'
                })

        # 显示结果
        print("\n基于频率的预测结果:")
        for i, pred in enumerate(predictions, 1):
            front_str = ' '.join([str(b).zfill(2) for b in pred['front']])
            back_str = ' '.join([str(b).zfill(2) for b in pred['back']])
            print(f"第 {i} 注: 前区 {front_str} | 后区 {back_str}")

        return predictions

    def mixed_strategy_prediction(self, num_predictions=1):
        """混合策略预测"""
        print(f"\n混合策略生成 {num_predictions} 注预测...")

        predictions = []

        # 策略1: 马尔可夫链预测
        if hasattr(self, 'analysis_data') and self.analysis_data:
            markov_preds = self.predict_numbers(max(1, num_predictions // 3), explain=False)
            for pred in markov_preds:
                predictions.append({
                    'front': pred['front'],
                    'back': pred['back'],
                    'method': 'markov_chain',
                    'score': pred['stability_score']
                })

        # 策略2: 频率预测
        freq_preds = self.frequency_based_prediction(max(1, num_predictions // 3))
        predictions.extend(freq_preds)

        # 策略3: 随机预测（基于统计特征）
        for i in range(num_predictions - len(predictions)):
            front_balls = random.sample(range(1, 36), 5)
            back_balls = random.sample(range(1, 13), 2)
            predictions.append({
                'front': sorted(front_balls),
                'back': sorted(back_balls),
                'method': 'statistical_random'
            })

        # 显示结果
        print("\n混合策略预测结果:")
        for i, pred in enumerate(predictions[:num_predictions], 1):
            front_str = ' '.join([str(b).zfill(2) for b in pred['front']])
            back_str = ' '.join([str(b).zfill(2) for b in pred['back']])
            method = pred.get('method', 'unknown')
            print(f"第 {i} 注: 前区 {front_str} | 后区 {back_str} ({method})")

        return predictions[:num_predictions]

    def check_winning_comparison(self, predictions, target_issue=None):
        """中奖对比分析"""
        print("\n开始中奖对比分析...")

        if target_issue:
            # 查找指定期号
            target_row = self.df[self.df['issue'] == str(target_issue)]
            if target_row.empty:
                print(f"未找到期号 {target_issue} 的开奖数据")
                return None
            target_row = target_row.iloc[0]
        else:
            # 使用最新一期
            target_row = self.df.iloc[0]

        winning_front = self.parse_balls(target_row['front_balls'])
        winning_back = self.parse_balls(target_row['back_balls'])

        print(f"对比期号: {target_row['issue']}")
        print(f"开奖号码: 前区 {' '.join([str(b).zfill(2) for b in winning_front])}, 后区 {' '.join([str(b).zfill(2) for b in winning_back])}")

        results = []
        for i, pred in enumerate(predictions, 1):
            front_matches = len(set(pred['front']) & set(winning_front))
            back_matches = len(set(pred['back']) & set(winning_back))

            # 判断中奖等级
            prize_level = self._determine_prize_level(front_matches, back_matches)

            result = {
                'prediction_no': i,
                'front_matches': front_matches,
                'back_matches': back_matches,
                'prize_level': prize_level,
                'front_balls': pred['front'],
                'back_balls': pred['back']
            }
            results.append(result)

            print(f"第 {i} 注: 前区中{front_matches}个, 后区中{back_matches}个 - {prize_level}")

        return results

    def _determine_prize_level(self, front_matches, back_matches):
        """判断中奖等级"""
        if front_matches == 5 and back_matches == 2:
            return "一等奖"
        elif front_matches == 5 and back_matches == 1:
            return "二等奖"
        elif front_matches == 5 and back_matches == 0:
            return "三等奖"
        elif front_matches == 4 and back_matches == 2:
            return "四等奖"
        elif front_matches == 4 and back_matches == 1:
            return "五等奖"
        elif front_matches == 3 and back_matches == 2:
            return "六等奖"
        elif front_matches == 4 and back_matches == 0:
            return "七等奖"
        elif front_matches == 3 and back_matches == 1:
            return "八等奖"
        elif front_matches == 2 and back_matches == 2:
            return "九等奖"
        else:
            return "未中奖"

    def historical_comparison(self, periods=100):
        """历史对比分析"""
        print(f"\n开始历史对比分析 (最近{periods}期)...")

        recent_data = self.df.head(periods)

        # 统计各种特征
        features = {
            'sum_distribution': {'front': [], 'back': []},
            'odd_even_distribution': {'front': [], 'back': []},
            'size_distribution': {'front': [], 'back': []},
            'span_distribution': {'front': [], 'back': []}
        }

        for _, row in recent_data.iterrows():
            front_balls = self.parse_balls(row['front_balls'])
            back_balls = self.parse_balls(row['back_balls'])

            # 和值分布
            features['sum_distribution']['front'].append(sum(front_balls))
            features['sum_distribution']['back'].append(sum(back_balls))

            # 奇偶分布
            front_odd = sum(1 for ball in front_balls if ball % 2 == 1)
            back_odd = sum(1 for ball in back_balls if ball % 2 == 1)
            features['odd_even_distribution']['front'].append(front_odd)
            features['odd_even_distribution']['back'].append(back_odd)

            # 大小分布
            front_small = sum(1 for ball in front_balls if ball <= 17)
            back_small = sum(1 for ball in back_balls if ball <= 6)
            features['size_distribution']['front'].append(front_small)
            features['size_distribution']['back'].append(back_small)

            # 跨度分布
            features['span_distribution']['front'].append(max(front_balls) - min(front_balls))
            features['span_distribution']['back'].append(max(back_balls) - min(back_balls))

        # 计算统计特征
        stats = {}
        for feature_name, feature_data in features.items():
            stats[feature_name] = {}
            for ball_type in ['front', 'back']:
                data = feature_data[ball_type]
                stats[feature_name][ball_type] = {
                    'mean': np.mean(data),
                    'std': np.std(data),
                    'min': min(data),
                    'max': max(data),
                    'median': np.median(data)
                }

        # 打印结果
        print("\n历史特征统计:")
        print("=" * 50)

        print(f"\n前区和值统计 (最近{periods}期):")
        front_sum_stats = stats['sum_distribution']['front']
        print(f"  平均值: {front_sum_stats['mean']:.1f}")
        print(f"  标准差: {front_sum_stats['std']:.1f}")
        print(f"  范围: {front_sum_stats['min']} - {front_sum_stats['max']}")

        print(f"\n前区跨度统计 (最近{periods}期):")
        front_span_stats = stats['span_distribution']['front']
        print(f"  平均值: {front_span_stats['mean']:.1f}")
        print(f"  标准差: {front_span_stats['std']:.1f}")
        print(f"  范围: {front_span_stats['min']} - {front_span_stats['max']}")

        return stats

    def update_data_append(self, new_periods=10):
        """追加最新数据到CSV文件"""
        print(f"开始获取最新 {new_periods} 期数据...")

        # 创建爬虫获取最新数据
        crawler = DLTCrawler()
        new_results = crawler.get_history_data(new_periods)

        if not new_results:
            print("未获取到新数据")
            return False

        # 读取现有数据
        existing_data = []
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                existing_data = list(reader)

        # 合并数据（去重）
        existing_issues = {row['issue'] for row in existing_data}
        new_data = [result for result in new_results if result['issue'] not in existing_issues]

        if new_data:
            all_data = new_data + existing_data
            # 按期号排序
            all_data.sort(key=lambda x: int(x['issue']), reverse=True)

            # 保存合并后的数据
            with open(self.data_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['issue', 'date', 'front_balls', 'back_balls']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_data)

            print(f"成功追加 {len(new_data)} 条新数据到 {self.data_file}")
            print(f"总数据量: {len(all_data)} 条")

            # 重新加载数据
            self.load_data()
            return True
        else:
            print("没有新数据需要追加")
            return False

    def visualization_analysis(self, save_charts=True):
        """可视化分析 - 生成各种图表"""
        print("\n开始可视化分析...")

        output_dir = "output/advanced"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 1. 生成频率分布图
        self._generate_frequency_charts(output_dir, save_charts)

        # 2. 生成转移概率热力图
        if hasattr(self, 'analysis_data') and self.analysis_data:
            self._generate_transition_heatmap(output_dir, save_charts)

        # 3. 生成网络图
        if hasattr(self, 'analysis_data') and self.analysis_data:
            self._generate_network_graph(output_dir, save_charts)

        # 4. 生成遗漏值热力图
        self._generate_missing_heatmap(output_dir, save_charts)

        # 5. 生成走势图
        self._generate_trend_charts(output_dir, save_charts)

        print(f"可视化图表已保存到: {output_dir}")

        return True

    def _generate_frequency_charts(self, output_dir, save_charts):
        """生成频率分布图"""
        # 计算频率
        front_frequency = defaultdict(int)
        back_frequency = defaultdict(int)

        for _, row in self.df.iterrows():
            front_balls = self.parse_balls(row['front_balls'])
            back_balls = self.parse_balls(row['back_balls'])

            for ball in front_balls:
                front_frequency[ball] += 1
            for ball in back_balls:
                back_frequency[ball] += 1

        # 前区频率分布图
        plt.figure(figsize=(15, 8))

        plt.subplot(2, 1, 1)
        front_balls = list(range(1, 36))
        front_counts = [front_frequency[ball] for ball in front_balls]

        bars = plt.bar(front_balls, front_counts, color='skyblue', alpha=0.7)
        plt.title('前区号码频率分布', fontsize=16, fontweight='bold')
        plt.xlabel('号码', fontsize=12)
        plt.ylabel('出现次数', fontsize=12)
        plt.grid(True, alpha=0.3)

        # 标注最高频率
        max_idx = front_counts.index(max(front_counts))
        plt.annotate(f'最高: {front_balls[max_idx]}号\n{max(front_counts)}次',
                    xy=(front_balls[max_idx], max(front_counts)),
                    xytext=(front_balls[max_idx]+3, max(front_counts)+10),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red')

        # 后区频率分布图
        plt.subplot(2, 1, 2)
        back_balls = list(range(1, 13))
        back_counts = [back_frequency[ball] for ball in back_balls]

        bars = plt.bar(back_balls, back_counts, color='lightcoral', alpha=0.7)
        plt.title('后区号码频率分布', fontsize=16, fontweight='bold')
        plt.xlabel('号码', fontsize=12)
        plt.ylabel('出现次数', fontsize=12)
        plt.grid(True, alpha=0.3)

        # 标注最高频率
        max_idx = back_counts.index(max(back_counts))
        plt.annotate(f'最高: {back_balls[max_idx]}号\n{max(back_counts)}次',
                    xy=(back_balls[max_idx], max(back_counts)),
                    xytext=(back_balls[max_idx]+1, max(back_counts)+20),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red')

        plt.tight_layout()

        if save_charts:
            plt.savefig(f"{output_dir}/frequency_distribution.png", dpi=300, bbox_inches='tight')
            print("频率分布图已保存")

        plt.close()

    def _generate_transition_heatmap(self, output_dir, save_charts):
        """生成转移概率热力图"""
        if not self.analysis_data:
            return

        # 前区转移概率热力图
        front_probs = self.analysis_data['front_probabilities']

        # 创建转移矩阵
        matrix_size = 35
        transition_matrix = np.zeros((matrix_size, matrix_size))

        for from_ball in range(1, matrix_size + 1):
            if from_ball in front_probs:
                for to_ball in range(1, matrix_size + 1):
                    if to_ball in front_probs[from_ball]:
                        transition_matrix[from_ball-1][to_ball-1] = front_probs[from_ball][to_ball]

        plt.figure(figsize=(12, 10))
        sns.heatmap(transition_matrix,
                   xticklabels=range(1, matrix_size + 1),
                   yticklabels=range(1, matrix_size + 1),
                   cmap='YlOrRd',
                   cbar_kws={'label': '转移概率'})

        plt.title('前区号码转移概率热力图', fontsize=16, fontweight='bold')
        plt.xlabel('转移到号码', fontsize=12)
        plt.ylabel('从号码', fontsize=12)

        if save_charts:
            plt.savefig(f"{output_dir}/front_transition_heatmap.png", dpi=300, bbox_inches='tight')
            print("前区转移概率热力图已保存")

        plt.close()

    def _generate_network_graph(self, output_dir, save_charts):
        """生成转移网络图"""
        if not self.analysis_data:
            return

        # 创建网络图
        G = nx.DiGraph()

        # 添加节点和边（只显示概率较高的转移）
        front_probs = self.analysis_data['front_probabilities']
        threshold = 0.1  # 只显示概率大于0.1的转移

        for from_ball in front_probs:
            for to_ball, prob in front_probs[from_ball].items():
                if prob > threshold:
                    G.add_edge(from_ball, to_ball, weight=prob)

        if len(G.nodes()) > 0:
            plt.figure(figsize=(12, 10))

            # 设置布局
            pos = nx.spring_layout(G, k=1, iterations=50)

            # 绘制节点
            nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                                 node_size=300, alpha=0.7)

            # 绘制边
            edges = G.edges()
            weights = [G[u][v]['weight'] for u, v in edges]
            nx.draw_networkx_edges(G, pos, width=[w*5 for w in weights],
                                 alpha=0.6, edge_color='gray', arrows=True)

            # 绘制标签
            nx.draw_networkx_labels(G, pos, font_size=8)

            plt.title('后区号码转移网络图 (概率>0.1)', fontsize=16, fontweight='bold')
            plt.axis('off')

            if save_charts:
                plt.savefig(f"{output_dir}/back_transition_network.png", dpi=300, bbox_inches='tight')
                print("转移网络图已保存")

        plt.close()

    def _generate_missing_heatmap(self, output_dir, save_charts):
        """生成遗漏值热力图"""
        # 计算最近50期的遗漏值变化
        recent_data = self.df.head(50)

        front_missing_history = []
        back_missing_history = []

        for i in range(len(recent_data)):
            # 计算到第i期为止的遗漏值
            subset_data = recent_data.iloc[:i+1]
            front_missing = self._calculate_missing_values_for_data(subset_data, 'front')
            back_missing = self._calculate_missing_values_for_data(subset_data, 'back')

            front_missing_history.append([front_missing.get(j, 0) for j in range(1, 36)])
            back_missing_history.append([back_missing.get(j, 0) for j in range(1, 13)])

        # 前区遗漏值热力图
        plt.figure(figsize=(15, 8))

        plt.subplot(2, 1, 1)
        front_matrix = np.array(front_missing_history).T
        sns.heatmap(front_matrix,
                   xticklabels=range(1, len(recent_data)+1),
                   yticklabels=range(1, 36),
                   cmap='Reds',
                   cbar_kws={'label': '遗漏期数'})

        plt.title('前区号码遗漏值热力图 (最近50期)', fontsize=14, fontweight='bold')
        plt.xlabel('期数', fontsize=12)
        plt.ylabel('号码', fontsize=12)

        # 后区遗漏值热力图
        plt.subplot(2, 1, 2)
        back_matrix = np.array(back_missing_history).T
        sns.heatmap(back_matrix,
                   xticklabels=range(1, len(recent_data)+1),
                   yticklabels=range(1, 13),
                   cmap='Blues',
                   cbar_kws={'label': '遗漏期数'})

        plt.title('后区号码遗漏值热力图 (最近50期)', fontsize=14, fontweight='bold')
        plt.xlabel('期数', fontsize=12)
        plt.ylabel('号码', fontsize=12)

        plt.tight_layout()

        if save_charts:
            plt.savefig(f"{output_dir}/missing_value_heatmap.png", dpi=300, bbox_inches='tight')
            print("遗漏值热力图已保存")

        plt.close()

    def _calculate_missing_values_for_data(self, data, ball_type):
        """为指定数据计算遗漏值"""
        missing = {}
        max_ball = 35 if ball_type == 'front' else 12

        for ball in range(1, max_ball + 1):
            missing[ball] = 0

            for i, (_, row) in enumerate(data.iterrows()):
                balls = self.parse_balls(row[f'{ball_type}_balls'])
                if ball in balls:
                    missing[ball] = 0
                else:
                    missing[ball] += 1

                if i == 0:  # 只计算到最新一期
                    break

        return missing

    def _generate_trend_charts(self, output_dir, save_charts):
        """生成走势图"""
        recent_data = self.df.head(100)  # 最近100期

        # 计算和值走势
        front_sums = []
        back_sums = []
        issues = []

        for _, row in recent_data.iterrows():
            front_balls = self.parse_balls(row['front_balls'])
            back_balls = self.parse_balls(row['back_balls'])

            front_sums.append(sum(front_balls))
            back_sums.append(sum(back_balls))
            issues.append(row['issue'])

        # 反转列表以按时间顺序显示
        front_sums.reverse()
        back_sums.reverse()
        issues.reverse()

        plt.figure(figsize=(15, 10))

        # 前区和值走势
        plt.subplot(3, 1, 1)
        plt.plot(range(len(front_sums)), front_sums, 'b-o', markersize=3, linewidth=1)
        plt.title('前区和值走势图 (最近100期)', fontsize=14, fontweight='bold')
        plt.ylabel('和值', fontsize=12)
        plt.grid(True, alpha=0.3)

        # 添加平均线
        avg_front = np.mean(front_sums)
        plt.axhline(y=avg_front, color='r', linestyle='--', alpha=0.7,
                   label=f'平均值: {avg_front:.1f}')
        plt.legend()

        # 后区和值走势
        plt.subplot(3, 1, 2)
        plt.plot(range(len(back_sums)), back_sums, 'r-o', markersize=3, linewidth=1)
        plt.title('后区和值走势图 (最近100期)', fontsize=14, fontweight='bold')
        plt.ylabel('和值', fontsize=12)
        plt.grid(True, alpha=0.3)

        # 添加平均线
        avg_back = np.mean(back_sums)
        plt.axhline(y=avg_back, color='b', linestyle='--', alpha=0.7,
                   label=f'平均值: {avg_back:.1f}')
        plt.legend()

        # 奇偶比例走势
        plt.subplot(3, 1, 3)
        odd_ratios = []
        for _, row in recent_data.iterrows():
            front_balls = self.parse_balls(row['front_balls'])
            odd_count = sum(1 for ball in front_balls if ball % 2 == 1)
            odd_ratios.append(odd_count / 5)

        odd_ratios.reverse()
        plt.plot(range(len(odd_ratios)), odd_ratios, 'g-o', markersize=3, linewidth=1)
        plt.title('前区奇数比例走势图 (最近100期)', fontsize=14, fontweight='bold')
        plt.ylabel('奇数比例', fontsize=12)
        plt.xlabel('期数', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)

        # 添加理论平均线
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7,
                   label='理论平均: 0.5')
        plt.legend()

        plt.tight_layout()

        if save_charts:
            plt.savefig(f"{output_dir}/trend_charts.png", dpi=300, bbox_inches='tight')
            print("走势图已保存")

        plt.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="大乐透数据分析与预测系统")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # 爬取数据命令
    crawl_parser = subparsers.add_parser("crawl", help="爬取大乐透历史数据")
    crawl_parser.add_argument("-c", "--count", type=int, default=50, help="获取的期数")
    crawl_parser.add_argument("-o", "--output", default="data/dlt_data.csv", help="输出文件名")
    crawl_parser.add_argument("-a", "--all", action="store_true", help="获取所有历史数据")

    # 检查数据命令
    check_parser = subparsers.add_parser("check", help="检查数据质量")
    check_parser.add_argument("-d", "--data", default="data/dlt_data.csv", help="数据文件路径")
    check_parser.add_argument("-q", "--quiet", action="store_true", help="静默模式")
    check_parser.add_argument("--remove-duplicates", action="store_true", help="去除重复数据")

    # 更新数据命令
    update_parser = subparsers.add_parser("update", help="追加最新数据到现有文件")
    update_parser.add_argument("-d", "--data", default="data/dlt_data.csv", help="数据文件路径")
    update_parser.add_argument("-n", "--new-periods", type=int, default=10, help="获取最新期数")

    # 基础分析命令
    basic_parser = subparsers.add_parser("basic", help="基础统计分析")
    basic_parser.add_argument("-d", "--data", default="data/dlt_data.csv", help="数据文件路径")

    # 贝叶斯分析命令
    bayesian_parser = subparsers.add_parser("bayesian", help="贝叶斯分析")
    bayesian_parser.add_argument("-d", "--data", default="data/dlt_data.csv", help="数据文件路径")

    # 概率分析命令
    prob_parser = subparsers.add_parser("probability", help="概率分析")
    prob_parser.add_argument("-d", "--data", default="data/dlt_data.csv", help="数据文件路径")

    # 频率模式分析命令
    frequency_parser = subparsers.add_parser("frequency", help="频率模式分析")
    frequency_parser.add_argument("-d", "--data", default="data/dlt_data.csv", help="数据文件路径")

    # 走势分析命令
    trend_parser = subparsers.add_parser("trend", help="走势分析")
    trend_parser.add_argument("-d", "--data", default="data/dlt_data.csv", help="数据文件路径")
    trend_parser.add_argument("-p", "--periods", type=int, default=50, help="分析期数")

    # 历史对比命令
    history_parser = subparsers.add_parser("history", help="历史对比分析")
    history_parser.add_argument("-d", "--data", default="data/dlt_data.csv", help="数据文件路径")
    history_parser.add_argument("-p", "--periods", type=int, default=100, help="对比期数")

    # 马尔可夫链分析命令
    markov_parser = subparsers.add_parser("markov", help="马尔可夫链分析和预测")
    markov_parser.add_argument("-d", "--data", default="data/dlt_data.csv", help="数据文件路径")
    markov_parser.add_argument("-p", "--periods", type=int, default=300, help="分析的期数")
    markov_parser.add_argument("-n", "--num", type=int, default=1, help="预测注数")
    markov_parser.add_argument("--explain", action="store_true", help="显示预测过程")

    # 频率预测命令
    freq_predict_parser = subparsers.add_parser("freq-predict", help="基于频率的预测")
    freq_predict_parser.add_argument("-d", "--data", default="data/dlt_data.csv", help="数据文件路径")
    freq_predict_parser.add_argument("-n", "--num", type=int, default=1, help="预测注数")

    # 混合策略预测命令
    mixed_parser = subparsers.add_parser("mixed", help="混合策略预测")
    mixed_parser.add_argument("-d", "--data", default="data/dlt_data.csv", help="数据文件路径")
    mixed_parser.add_argument("-n", "--num", type=int, default=3, help="预测注数")

    # 中奖对比命令
    compare_parser = subparsers.add_parser("compare", help="中奖对比分析")
    compare_parser.add_argument("-d", "--data", default="data/dlt_data.csv", help="数据文件路径")
    compare_parser.add_argument("-i", "--issue", help="指定期号进行对比")
    compare_parser.add_argument("-n", "--num", type=int, default=3, help="预测注数")

    # 可视化分析命令
    visual_parser = subparsers.add_parser("visual", help="可视化分析")
    visual_parser.add_argument("-d", "--data", default="data/dlt_data.csv", help="数据文件路径")
    visual_parser.add_argument("-p", "--periods", type=int, default=300, help="马尔可夫链分析期数")

    # 完整分析命令
    full_parser = subparsers.add_parser("full", help="运行完整分析")
    full_parser.add_argument("-d", "--data", default="data/dlt_data.csv", help="数据文件路径")
    full_parser.add_argument("-p", "--periods", type=int, default=300, help="马尔可夫链分析期数")
    full_parser.add_argument("-n", "--num", type=int, default=5, help="预测注数")

    args = parser.parse_args()

    if args.command == "crawl":
        crawler = DLTCrawler()
        if args.all:
            results = crawler.get_history_data(get_all=True)
        else:
            results = crawler.get_history_data(args.count)

        if results:
            crawler.save_to_csv(results, args.output)
        else:
            print("未获取到数据")

    elif args.command == "check":
        if not os.path.exists(args.data):
            print(f"数据文件不存在: {args.data}")
            return

        analyzer = DLTAnalyzer(args.data)
        has_duplicates = analyzer.check_duplicates(args.quiet)

        if has_duplicates and args.remove_duplicates:
            analyzer.remove_duplicates()

    elif args.command == "update":
        if not os.path.exists(args.data):
            print(f"数据文件不存在: {args.data}")
            return

        analyzer = DLTAnalyzer(args.data)
        analyzer.update_data_append(args.new_periods)

    elif args.command == "basic":
        if not os.path.exists(args.data):
            print(f"数据文件不存在: {args.data}")
            return

        analyzer = DLTAnalyzer(args.data)
        analyzer.basic_analysis()

    elif args.command == "bayesian":
        if not os.path.exists(args.data):
            print(f"数据文件不存在: {args.data}")
            return

        analyzer = DLTAnalyzer(args.data)
        analyzer.bayesian_analysis()

    elif args.command == "probability":
        if not os.path.exists(args.data):
            print(f"数据文件不存在: {args.data}")
            return

        analyzer = DLTAnalyzer(args.data)
        analyzer.probability_analysis()

    elif args.command == "frequency":
        if not os.path.exists(args.data):
            print(f"数据文件不存在: {args.data}")
            return

        analyzer = DLTAnalyzer(args.data)
        analyzer.frequency_pattern_analysis()

    elif args.command == "trend":
        if not os.path.exists(args.data):
            print(f"数据文件不存在: {args.data}")
            return

        analyzer = DLTAnalyzer(args.data)
        analyzer.trend_analysis(args.periods)

    elif args.command == "history":
        if not os.path.exists(args.data):
            print(f"数据文件不存在: {args.data}")
            return

        analyzer = DLTAnalyzer(args.data)
        analyzer.historical_comparison(args.periods)

    elif args.command == "markov":
        if not os.path.exists(args.data):
            print(f"数据文件不存在: {args.data}")
            return

        analyzer = DLTAnalyzer(args.data)
        analyzer.analyze_periods(args.periods, verbose=True)
        predictions = analyzer.predict_numbers(args.num, explain=args.explain)

    elif args.command == "freq-predict":
        if not os.path.exists(args.data):
            print(f"数据文件不存在: {args.data}")
            return

        analyzer = DLTAnalyzer(args.data)
        analyzer.frequency_based_prediction(args.num)

    elif args.command == "mixed":
        if not os.path.exists(args.data):
            print(f"数据文件不存在: {args.data}")
            return

        analyzer = DLTAnalyzer(args.data)
        analyzer.analyze_periods(300, verbose=False)  # 为混合策略准备马尔可夫链数据
        analyzer.mixed_strategy_prediction(args.num)

    elif args.command == "compare":
        if not os.path.exists(args.data):
            print(f"数据文件不存在: {args.data}")
            return

        analyzer = DLTAnalyzer(args.data)
        analyzer.analyze_periods(300, verbose=False)
        predictions = analyzer.predict_numbers(args.num, explain=False)
        analyzer.check_winning_comparison(predictions, args.issue)

    elif args.command == "visual":
        if not os.path.exists(args.data):
            print(f"数据文件不存在: {args.data}")
            return

        analyzer = DLTAnalyzer(args.data)
        analyzer.analyze_periods(args.periods, verbose=False)  # 为网络图准备数据
        analyzer.visualization_analysis()

    elif args.command == "full":
        if not os.path.exists(args.data):
            print(f"数据文件不存在: {args.data}")
            return

        analyzer = DLTAnalyzer(args.data)

        print("=" * 60)
        print("大乐透完整分析报告")
        print("=" * 60)

        # 运行所有分析
        analyzer.basic_analysis()
        analyzer.bayesian_analysis()
        analyzer.probability_analysis()
        analyzer.frequency_pattern_analysis()
        analyzer.trend_analysis()
        analyzer.historical_comparison()

        # 马尔可夫链预测
        analyzer.analyze_periods(args.periods, verbose=True)
        predictions = analyzer.predict_numbers(args.num, explain=True)

        # 混合策略预测
        mixed_predictions = analyzer.mixed_strategy_prediction(args.num)

        # 中奖对比
        analyzer.check_winning_comparison(predictions)

        print("\n" + "=" * 60)
        print("完整分析报告结束")
        print("=" * 60)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
