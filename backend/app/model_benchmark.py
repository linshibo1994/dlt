#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模型基准测试框架
提供全面的模型评估、比较和可视化功能
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Any, Callable, Union
from datetime import datetime
from collections import defaultdict
import json
import time
from tqdm import tqdm

# 尝试导入核心模块
try:
    from core_modules import logger_manager, data_manager, cache_manager
except ImportError:
    # 如果在不同目录运行，添加父目录到路径
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core_modules import logger_manager, data_manager, cache_manager


class ModelBenchmark:
    """模型基准测试框架"""
    
    def __init__(self):
        """初始化基准测试框架"""
        self.df = data_manager.get_data()
        if self.df is None:
            logger_manager.error("数据未加载")
            
        # 评估指标
        self.metrics = [
            'hit_rate',          # 命中率（任意号码命中）
            'accuracy',          # 准确率（加权得分）
            'consistency',       # 一致性（标准差）
            'roi',               # 投资回报率
            'adaptability',      # 适应性（性能改进趋势）
            'time_efficiency'    # 时间效率（预测速度）
        ]
        
        # 奖金设置
        self.prize_levels = {
            '一等奖': 10000000,  # 1000万（示例值）
            '二等奖': 100000,    # 10万（示例值）
            '三等奖': 3000,      # 3000元
            '四等奖': 200,       # 200元
            '五等奖': 10,        # 10元
            '六等奖': 5,         # 5元
            '未中奖': 0
        }
        
        # 评估结果
        self.results = {}
        
        # 比较结果
        self.comparison = None
    
    def register_model(self, model_name: str, predict_func: Callable, 
                      description: str = "", category: str = "traditional"):
        """注册模型
        
        Args:
            model_name: 模型名称
            predict_func: 预测函数，接受期数参数，返回预测结果
            description: 模型描述
            category: 模型类别
        """
        self.results[model_name] = {
            'name': model_name,
            'description': description,
            'category': category,
            'predict_func': predict_func,
            'evaluated': False
        }
        
        logger_manager.info(f"已注册模型: {model_name} ({category})")
    
    def evaluate_model(self, model_name: str, test_periods: int = 50, 
                      bet_cost: float = 2.0, verbose: bool = True) -> Dict:
        """评估单个模型
        
        Args:
            model_name: 模型名称
            test_periods: 测试期数
            bet_cost: 每注投注成本
            verbose: 是否显示详细信息
            
        Returns:
            Dict: 评估结果
        """
        if model_name not in self.results:
            logger_manager.error(f"未注册的模型: {model_name}")
            return {}
        
        if self.df is None or len(self.df) < test_periods:
            logger_manager.error(f"数据不足，无法评估模型 {model_name}")
            return {}
        
        if verbose:
            logger_manager.info(f"开始评估模型: {model_name}, 测试期数: {test_periods}")
        
        model_info = self.results[model_name]
        predict_func = model_info['predict_func']
        
        # 初始化结果
        result = {
            'model_name': model_name,
            'description': model_info['description'],
            'category': model_info['category'],
            'test_periods': test_periods,
            'metrics': {},
            'detailed_results': [],
            'timestamp': datetime.now().isoformat()
        }
        
        # 统计变量
        total_hits = 0
        hit_counts = defaultdict(int)  # 按命中数量统计
        prize_levels = defaultdict(int)  # 按奖级统计
        total_cost = 0
        total_prize = 0
        accuracy_scores = []
        prediction_times = []
        
        # 逐期测试
        for i in tqdm(range(test_periods), desc=f"评估 {model_name}", disable=not verbose):
            try:
                # 获取测试期的实际结果
                test_row = self.df.iloc[i]
                actual_front, actual_back = data_manager.parse_balls(test_row)
                issue = test_row.get('issue', str(i))
                
                # 使用历史数据进行预测
                historical_data = self.df.iloc[i+1:i+501]  # 使用后续500期作为历史数据
                
                # 记录预测时间
                start_time = time.time()
                predictions = predict_func(historical_data)
                end_time = time.time()
                prediction_time = end_time - start_time
                prediction_times.append(prediction_time)
                
                # 如果没有预测结果，跳过
                if not predictions:
                    continue
                
                # 评估每一注预测
                period_results = []
                period_cost = 0
                period_prize = 0
                
                for pred_idx, pred in enumerate(predictions):
                    # 解析预测结果
                    if isinstance(pred, tuple) and len(pred) == 2:
                        pred_front, pred_back = pred
                    elif isinstance(pred, dict) and 'front_balls' in pred and 'back_balls' in pred:
                        pred_front = pred['front_balls']
                        pred_back = pred['back_balls']
                    else:
                        logger_manager.warning(f"无法解析预测结果: {pred}")
                        continue
                    
                    # 计算命中情况
                    front_hits = len(set(pred_front) & set(actual_front))
                    back_hits = len(set(pred_back) & set(actual_back))
                    
                    # 判断中奖等级
                    prize_level, prize_amount = self._calculate_prize_level(front_hits, back_hits)
                    
                    # 更新统计
                    hit_key = f"{front_hits}+{back_hits}"
                    hit_counts[hit_key] += 1
                    prize_levels[prize_level] += 1
                    
                    if front_hits > 0 or back_hits > 0:
                        total_hits += 1
                    
                    # 计算成本和奖金
                    period_cost += bet_cost
                    period_prize += prize_amount
                    
                    # 计算准确率得分（加权得分）
                    accuracy_score = self._calculate_accuracy_score(front_hits, back_hits)
                    accuracy_scores.append(accuracy_score)
                    
                    # 记录详细结果
                    period_results.append({
                        'prediction_index': pred_idx + 1,
                        'predicted_front': pred_front,
                        'predicted_back': pred_back,
                        'actual_front': actual_front,
                        'actual_back': actual_back,
                        'front_hits': front_hits,
                        'back_hits': back_hits,
                        'prize_level': prize_level,
                        'prize_amount': prize_amount,
                        'accuracy_score': accuracy_score
                    })
                
                # 更新总成本和奖金
                total_cost += period_cost
                total_prize += period_prize
                
                # 记录本期结果
                result['detailed_results'].append({
                    'issue': issue,
                    'period_results': period_results,
                    'period_cost': period_cost,
                    'period_prize': period_prize,
                    'period_profit': period_prize - period_cost,
                    'prediction_time': prediction_time
                })
                
            except Exception as e:
                logger_manager.error(f"评估期 {i} 失败: {e}")
        
        # 计算评估指标
        total_predictions = sum(hit_counts.values())
        
        if total_predictions > 0:
            # 命中率
            result['metrics']['hit_rate'] = total_hits / total_predictions
            
            # 准确率（加权得分）
            result['metrics']['accuracy'] = np.mean(accuracy_scores) if accuracy_scores else 0
            
            # 一致性（标准差）
            result['metrics']['consistency'] = 1.0 / (1.0 + np.std(accuracy_scores)) if len(accuracy_scores) > 1 else 0
            
            # 投资回报率
            result['metrics']['roi'] = (total_prize - total_cost) / total_cost if total_cost > 0 else 0
            
            # 适应性（后半段vs前半段）
            if len(accuracy_scores) >= 10:
                mid_point = len(accuracy_scores) // 2
                first_half = np.mean(accuracy_scores[:mid_point])
                second_half = np.mean(accuracy_scores[mid_point:])
                result['metrics']['adaptability'] = second_half - first_half
            else:
                result['metrics']['adaptability'] = 0
            
            # 时间效率（预测速度）
            avg_time = np.mean(prediction_times) if prediction_times else 0
            result['metrics']['time_efficiency'] = 1.0 / (1.0 + avg_time)  # 转换为0-1范围，越快越高
        
        # 记录命中统计
        result['hit_statistics'] = dict(hit_counts)
        result['prize_statistics'] = dict(prize_levels)
        result['total_cost'] = total_cost
        result['total_prize'] = total_prize
        result['net_profit'] = total_prize - total_cost
        result['avg_prediction_time'] = np.mean(prediction_times) if prediction_times else 0
        
        # 更新模型评估状态
        self.results[model_name].update(result)
        self.results[model_name]['evaluated'] = True
        
        if verbose:
            logger_manager.info(f"模型 {model_name} 评估完成")
            
            # 打印关键指标
            print(f"\n📊 {model_name} 评估结果:")
            print(f"  命中率: {result['metrics'].get('hit_rate', 0):.4f}")
            print(f"  准确率: {result['metrics'].get('accuracy', 0):.4f}")
            print(f"  ROI: {result['metrics'].get('roi', 0):.4f}")
            print(f"  总投注: {total_cost:.2f} 元")
            print(f"  总奖金: {total_prize:.2f} 元")
            print(f"  净利润: {total_prize - total_cost:.2f} 元")
            print(f"  平均预测时间: {np.mean(prediction_times) if prediction_times else 0:.4f} 秒")
        
        return result
    
    def evaluate_all_models(self, test_periods: int = 50, bet_cost: float = 2.0, verbose: bool = True):
        """评估所有注册的模型
        
        Args:
            test_periods: 测试期数
            bet_cost: 每注投注成本
            verbose: 是否显示详细信息
        """
        if verbose:
            logger_manager.info(f"开始评估所有模型，测试期数: {test_periods}")
        
        for model_name in self.results.keys():
            self.evaluate_model(model_name, test_periods, bet_cost, verbose)
        
        if verbose:
            logger_manager.info("所有模型评估完成")
    
    def compare_models(self, metrics: List[str] = None, categories: List[str] = None, 
                      verbose: bool = True) -> Dict:
        """比较模型性能
        
        Args:
            metrics: 要比较的指标列表，默认为所有指标
            categories: 要比较的模型类别，默认为所有类别
            verbose: 是否显示详细信息
            
        Returns:
            Dict: 比较结果
        """
        # 使用默认指标或指定指标
        metrics = metrics or self.metrics
        
        # 筛选已评估的模型
        evaluated_models = {name: info for name, info in self.results.items() 
                           if info.get('evaluated', False)}
        
        # 筛选指定类别的模型
        if categories:
            evaluated_models = {name: info for name, info in evaluated_models.items() 
                              if info.get('category') in categories}
        
        if not evaluated_models:
            logger_manager.warning("没有已评估的模型可供比较")
            return {}
        
        comparison_result = {
            'models': list(evaluated_models.keys()),
            'metrics': {},
            'rankings': {},
            'overall_ranking': {},
            'categories': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # 按类别分组
        for model_name, model_info in evaluated_models.items():
            category = model_info.get('category', 'unknown')
            if category not in comparison_result['categories']:
                comparison_result['categories'][category] = []
            comparison_result['categories'][category].append(model_name)
        
        # 比较各指标
        for metric in metrics:
            comparison_result['metrics'][metric] = {}
            metric_values = {}
            
            for model_name, model_info in evaluated_models.items():
                if 'metrics' in model_info and metric in model_info['metrics']:
                    metric_values[model_name] = model_info['metrics'][metric]
            
            # 排序
            sorted_models = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)
            
            # 记录排名
            for rank, (model_name, value) in enumerate(sorted_models, 1):
                comparison_result['metrics'][metric][model_name] = {
                    'value': value,
                    'rank': rank
                }
        
        # 计算总体排名
        model_ranks = defaultdict(int)
        for metric in metrics:
            if metric in comparison_result['metrics']:
                for model_name, info in comparison_result['metrics'][metric].items():
                    model_ranks[model_name] += info['rank']
        
        # 排序（总排名最低的最好）
        sorted_models = sorted(model_ranks.items(), key=lambda x: x[1])
        
        # 记录总体排名
        for rank, (model_name, total_rank) in enumerate(sorted_models, 1):
            comparison_result['overall_ranking'][model_name] = {
                'rank': rank,
                'total_rank_score': total_rank,
                'category': evaluated_models[model_name].get('category', 'unknown')
            }
        
        # 记录各指标的排名
        for metric in metrics:
            if metric in comparison_result['metrics']:
                comparison_result['rankings'][metric] = [
                    model_name for model_name, _ in sorted(
                        comparison_result['metrics'][metric].items(),
                        key=lambda x: x[1]['rank']
                    )
                ]
        
        # 保存比较结果
        self.comparison = comparison_result
        
        if verbose:
            # 打印总体排名
            print("\n🏆 模型总体排名:")
            for model_name, info in comparison_result['overall_ranking'].items():
                print(f"  {info['rank']}. {model_name} (类别: {info['category']})")
            
            # 打印各指标最佳模型
            print("\n🥇 各指标最佳模型:")
            for metric in metrics:
                if metric in comparison_result['rankings'] and comparison_result['rankings'][metric]:
                    best_model = comparison_result['rankings'][metric][0]
                    best_value = comparison_result['metrics'][metric][best_model]['value']
                    print(f"  {metric}: {best_model} ({best_value:.4f})")
        
        return comparison_result
    
    def generate_report(self, output_path=None) -> str:
        """生成评估报告
        
        Args:
            output_path: 报告输出路径，如果为None则返回报告内容
            
        Returns:
            str: 报告内容
        """
        # 筛选已评估的模型
        evaluated_models = {name: info for name, info in self.results.items() 
                           if info.get('evaluated', False)}
        
        if not evaluated_models:
            return "没有评估结果可供生成报告"
        
        report = ["# 模型评估报告"]
        report.append(f"\n## 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 模型概览
        report.append("\n## 模型概览")
        report.append("\n| 模型名称 | 类别 | 命中率 | 准确率 | ROI | 平均预测时间(秒) |")
        report.append("| --- | --- | --- | --- | --- | --- |")
        
        for model_name, info in evaluated_models.items():
            metrics = info.get('metrics', {})
            hit_rate = metrics.get('hit_rate', 0)
            accuracy = metrics.get('accuracy', 0)
            roi = metrics.get('roi', 0)
            avg_time = info.get('avg_prediction_time', 0)
            category = info.get('category', 'unknown')
            
            report.append(f"| {model_name} | {category} | {hit_rate:.4f} | {accuracy:.4f} | {roi:.4f} | {avg_time:.4f} |")
        
        # 总体排名
        if self.comparison and 'overall_ranking' in self.comparison:
            report.append("\n## 总体排名")
            report.append("\n| 排名 | 模型名称 | 类别 | 总分 |")
            report.append("| --- | --- | --- | --- |")
            
            for model_name, info in self.comparison['overall_ranking'].items():
                rank = info['rank']
                category = info['category']
                score = info['total_rank_score']
                report.append(f"| {rank} | {model_name} | {category} | {score} |")
        
        # 各指标排名
        if self.comparison and 'rankings' in self.comparison:
            report.append("\n## 各指标排名")
            
            for metric, models in self.comparison['rankings'].items():
                report.append(f"\n### {metric}")
                report.append("\n| 排名 | 模型名称 | 得分 |")
                report.append("| --- | --- | --- |")
                
                for i, model_name in enumerate(models, 1):
                    value = self.comparison['metrics'][metric][model_name]['value']
                    report.append(f"| {i} | {model_name} | {value:.4f} |")
        
        # 详细评估结果
        report.append("\n## 详细评估结果")
        
        for model_name, info in evaluated_models.items():
            report.append(f"\n### {model_name}")
            
            # 基本信息
            report.append(f"\n#### 基本信息")
            report.append(f"- 类别: {info.get('category', 'unknown')}")
            report.append(f"- 描述: {info.get('description', '')}")
            report.append(f"- 测试期数: {info.get('test_periods', 0)}")
            report.append(f"- 总投注成本: {info.get('total_cost', 0):.2f}")
            report.append(f"- 总奖金: {info.get('total_prize', 0):.2f}")
            report.append(f"- 净利润: {info.get('net_profit', 0):.2f}")
            report.append(f"- 平均预测时间: {info.get('avg_prediction_time', 0):.4f} 秒")
            
            # 评估指标
            report.append(f"\n#### 评估指标")
            metrics = info.get('metrics', {})
            for metric, value in metrics.items():
                report.append(f"- {metric}: {value:.4f}")
            
            # 命中统计
            report.append(f"\n#### 命中统计")
            hit_stats = info.get('hit_statistics', {})
            for hit_key, count in sorted(hit_stats.items()):
                report.append(f"- {hit_key}: {count}次")
            
            # 奖级统计
            report.append(f"\n#### 奖级统计")
            prize_stats = info.get('prize_statistics', {})
            for level, count in sorted(prize_stats.items()):
                report.append(f"- {level}: {count}次")
        
        # 保存报告
        report_content = "\n".join(report)
        
        if output_path:
            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                logger_manager.info(f"评估报告已保存到 {output_path}")
            except Exception as e:
                logger_manager.error(f"保存评估报告失败: {e}")
        
        return report_content
    
    def visualize_comparison(self, metrics=None, output_path=None, figsize=(12, 10)):
        """可视化模型比较结果
        
        Args:
            metrics: 要可视化的指标列表，默认为所有指标
            output_path: 图表输出路径
            figsize: 图表大小
        """
        if not self.comparison:
            logger_manager.warning("没有比较结果可供可视化")
            return
        
        # 使用默认指标或指定指标
        metrics = metrics or self.metrics
        
        # 筛选有效指标
        valid_metrics = [m for m in metrics if m in self.comparison.get('metrics', {})]
        
        if not valid_metrics:
            logger_manager.warning("没有有效指标可供可视化")
            return
        
        try:
            # 创建图表
            fig, axes = plt.subplots(len(valid_metrics), 1, figsize=figsize)
            if len(valid_metrics) == 1:
                axes = [axes]
            
            # 设置颜色映射
            categories = self.comparison.get('categories', {})
            category_colors = {}
            cmap = plt.cm.tab10
            for i, category in enumerate(categories.keys()):
                category_colors[category] = cmap(i % 10)
            
            # 绘制各指标对比图
            for i, metric in enumerate(valid_metrics):
                ax = axes[i]
                
                # 准备数据
                models = []
                values = []
                colors = []
                
                metric_data = self.comparison['metrics'][metric]
                for model_name, data in sorted(metric_data.items(), key=lambda x: x[1]['value'], reverse=True):
                    models.append(model_name)
                    values.append(data['value'])
                    
                    # 获取模型类别
                    category = 'unknown'
                    for cat, model_list in categories.items():
                        if model_name in model_list:
                            category = cat
                            break
                    
                    colors.append(category_colors.get(category, 'gray'))
                
                # 绘制条形图
                bars = ax.bar(models, values, color=colors, alpha=0.7)
                
                # 添加数值标签
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.4f}', ha='center', va='bottom', fontsize=8)
                
                # 设置标题和标签
                ax.set_title(f'{metric.capitalize()} 比较')
                ax.set_ylabel(metric)
                ax.set_ylim(bottom=0)
                
                # 旋转x轴标签
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # 添加图例
            handles = [plt.Rectangle((0,0),1,1, color=color) for color in category_colors.values()]
            labels = list(category_colors.keys())
            fig.legend(handles, labels, loc='upper right', title='模型类别')
            
            plt.tight_layout()
            
            # 保存或显示图表
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger_manager.info(f"比较图表已保存到 {output_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger_manager.error(f"可视化比较失败: {e}")
    
    def visualize_model_performance(self, model_name, output_path=None):
        """可视化单个模型的性能
        
        Args:
            model_name: 模型名称
            output_path: 图表输出路径
        """
        if model_name not in self.results or not self.results[model_name].get('evaluated', False):
            logger_manager.warning(f"模型 {model_name} 未评估，无法可视化")
            return
        
        model_info = self.results[model_name]
        detailed_results = model_info.get('detailed_results', [])
        
        if not detailed_results:
            logger_manager.warning(f"模型 {model_name} 没有详细结果，无法可视化")
            return
        
        try:
            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. 准确率随时间变化
            accuracy_scores = []
            issues = []
            
            for period in detailed_results:
                for pred in period['period_results']:
                    accuracy_scores.append(pred['accuracy_score'])
                    issues.append(period['issue'])
            
            axes[0, 0].plot(accuracy_scores, marker='o', linestyle='-', alpha=0.7)
            axes[0, 0].set_title(f'{model_name} - 准确率随时间变化')
            axes[0, 0].set_xlabel('预测序号')
            axes[0, 0].set_ylabel('准确率得分')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 添加移动平均线
            window = min(10, len(accuracy_scores))
            if window > 1:
                moving_avg = np.convolve(accuracy_scores, np.ones(window)/window, mode='valid')
                axes[0, 0].plot(range(window-1, len(accuracy_scores)), moving_avg, 'r-', 
                               linewidth=2, alpha=0.8, label=f'{window}期移动平均')
                axes[0, 0].legend()
            
            # 2. 命中统计
            hit_stats = model_info.get('hit_statistics', {})
            hit_keys = []
            hit_counts = []
            
            for key, count in sorted(hit_stats.items()):
                hit_keys.append(key)
                hit_counts.append(count)
            
            axes[0, 1].bar(hit_keys, hit_counts, color='skyblue', alpha=0.7)
            axes[0, 1].set_title(f'{model_name} - 命中统计')
            axes[0, 1].set_xlabel('命中情况 (前区+后区)')
            axes[0, 1].set_ylabel('次数')
            
            # 添加数值标签
            for i, v in enumerate(hit_counts):
                axes[0, 1].text(i, v + 0.1, str(v), ha='center')
            
            # 3. 奖级统计
            prize_stats = model_info.get('prize_statistics', {})
            prize_levels = []
            prize_counts = []
            
            # 按奖级排序
            level_order = ['一等奖', '二等奖', '三等奖', '四等奖', '五等奖', '六等奖', '未中奖']
            for level in level_order:
                if level in prize_stats:
                    prize_levels.append(level)
                    prize_counts.append(prize_stats[level])
            
            axes[1, 0].bar(prize_levels, prize_counts, color='lightgreen', alpha=0.7)
            axes[1, 0].set_title(f'{model_name} - 奖级统计')
            axes[1, 0].set_xlabel('奖级')
            axes[1, 0].set_ylabel('次数')
            plt.setp(axes[1, 0].get_xticklabels(), rotation=45, ha='right')
            
            # 添加数值标签
            for i, v in enumerate(prize_counts):
                axes[1, 0].text(i, v + 0.1, str(v), ha='center')
            
            # 4. 累计收益曲线
            cumulative_cost = 0
            cumulative_prize = 0
            cumulative_profit = []
            
            for period in detailed_results:
                cumulative_cost += period['period_cost']
                cumulative_prize += period['period_prize']
                cumulative_profit.append(cumulative_prize - cumulative_cost)
            
            axes[1, 1].plot(cumulative_profit, marker='o', linestyle='-', color='orange', alpha=0.7)
            axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            axes[1, 1].set_title(f'{model_name} - 累计收益曲线')
            axes[1, 1].set_xlabel('期数')
            axes[1, 1].set_ylabel('累计收益 (元)')
            axes[1, 1].grid(True, alpha=0.3)
            
            # 添加最终收益标注
            final_profit = cumulative_profit[-1] if cumulative_profit else 0
            axes[1, 1].text(len(cumulative_profit) - 1, final_profit, 
                           f'{final_profit:.2f}元', ha='right', va='bottom')
            
            plt.tight_layout()
            
            # 保存或显示图表
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger_manager.info(f"模型性能图表已保存到 {output_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger_manager.error(f"可视化模型性能失败: {e}")
    
    def save_results(self, output_path):
        """保存评估结果
        
        Args:
            output_path: 结果输出路径
        """
        try:
            # 准备结果数据
            save_data = {
                'results': {},
                'comparison': self.comparison,
                'timestamp': datetime.now().isoformat()
            }
            
            # 移除不可序列化的预测函数
            for model_name, info in self.results.items():
                if info.get('evaluated', False):
                    model_data = info.copy()
                    if 'predict_func' in model_data:
                        del model_data['predict_func']
                    save_data['results'][model_name] = model_data
            
            # 保存为JSON
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            logger_manager.info(f"评估结果已保存到 {output_path}")
            
        except Exception as e:
            logger_manager.error(f"保存评估结果失败: {e}")
    
    def load_results(self, input_path):
        """加载评估结果
        
        Args:
            input_path: 结果输入路径
            
        Returns:
            bool: 是否加载成功
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(input_path):
                logger_manager.error(f"结果文件不存在: {input_path}")
                return False
            
            # 加载JSON
            with open(input_path, 'r', encoding='utf-8') as f:
                load_data = json.load(f)
            
            # 更新比较结果
            if 'comparison' in load_data:
                self.comparison = load_data['comparison']
            
            # 更新模型结果（保留预测函数）
            if 'results' in load_data:
                for model_name, info in load_data['results'].items():
                    if model_name in self.results:
                        # 保留预测函数
                        predict_func = self.results[model_name].get('predict_func')
                        self.results[model_name] = info
                        self.results[model_name]['predict_func'] = predict_func
                    else:
                        # 新模型，没有预测函数
                        info['predict_func'] = None
                        self.results[model_name] = info
            
            logger_manager.info(f"评估结果已从 {input_path} 加载")
            return True
            
        except Exception as e:
            logger_manager.error(f"加载评估结果失败: {e}")
            return False
    
    def _calculate_prize_level(self, front_hits: int, back_hits: int) -> Tuple[str, float]:
        """计算中奖等级和奖金（按照大乐透2019年新规则）
        
        Args:
            front_hits: 前区命中数
            back_hits: 后区命中数
            
        Returns:
            Tuple[str, float]: 奖级和奖金
        """
        if front_hits == 5 and back_hits == 2:
            return "一等奖", self.prize_levels["一等奖"]
        elif front_hits == 5 and back_hits == 1:
            return "二等奖", self.prize_levels["二等奖"]
        elif front_hits == 5 and back_hits == 0:
            return "三等奖", self.prize_levels["三等奖"]
        elif front_hits == 4 and back_hits == 2:
            return "四等奖", self.prize_levels["四等奖"]
        elif front_hits == 4 and back_hits == 1:
            return "五等奖", self.prize_levels["五等奖"]
        elif (front_hits == 4 and back_hits == 0) or (front_hits == 3 and back_hits == 2):
            return "六等奖", self.prize_levels["六等奖"]
        elif front_hits == 3 and back_hits == 1:
            return "七等奖", self.prize_levels["七等奖"]
        elif (front_hits == 3 and back_hits == 0) or (front_hits == 2 and back_hits == 2):
            return "八等奖", self.prize_levels["八等奖"]
        elif (front_hits == 2 and back_hits == 1) or (front_hits == 1 and back_hits == 2) or (front_hits == 0 and back_hits == 2):
            return "九等奖", self.prize_levels["九等奖"]
        else:
            return "未中奖", self.prize_levels["未中奖"]
    
    def _calculate_accuracy_score(self, front_hits: int, back_hits: int) -> float:
        """计算准确率得分（加权得分）
        
        Args:
            front_hits: 前区命中数
            back_hits: 后区命中数
            
        Returns:
            float: 准确率得分
        """
        # 前区权重
        front_weight = 0.7
        # 后区权重
        back_weight = 0.3
        
        # 前区得分（满分为5）
        front_score = front_hits / 5
        # 后区得分（满分为2）
        back_score = back_hits / 2
        
        # 加权总分
        return front_weight * front_score + back_weight * back_score


# 全局实例
_model_benchmark = None

def get_model_benchmark() -> ModelBenchmark:
    """获取模型基准测试实例"""
    global _model_benchmark
    if _model_benchmark is None:
        _model_benchmark = ModelBenchmark()
    return _model_benchmark


if __name__ == "__main__":
    # 测试模型基准测试框架
    print("🔍 测试模型基准测试框架...")
    benchmark = get_model_benchmark()
    
    # 模拟预测函数（仅用于测试）
    def mock_baseline_predict(historical_data):
        """基准预测（仅用于测试）"""
        # 使用简单的频率分析作为基准
        from collections import Counter

        front_counter = Counter()
        back_counter = Counter()

        for _, row in historical_data.head(50).iterrows():
            front_balls, back_balls = data_manager.parse_balls(row)
            front_counter.update(front_balls)
            back_counter.update(back_balls)

        front_most_common = [ball for ball, _ in front_counter.most_common(5)]
        back_most_common = [ball for ball, _ in back_counter.most_common(2)]

        # 确保有足够的号码
        while len(front_most_common) < 5:
            for i in range(1, 36):
                if i not in front_most_common:
                    front_most_common.append(i)
                    if len(front_most_common) >= 5:
                        break

        while len(back_most_common) < 2:
            for i in range(1, 13):
                if i not in back_most_common:
                    back_most_common.append(i)
                    if len(back_most_common) >= 2:
                        break

        return [(sorted(front_most_common[:5]), sorted(back_most_common[:2])) for _ in range(3)]
    
    def mock_frequency_predict(historical_data):
        """频率预测"""
        # 简化版频率预测
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
    
    def mock_markov_predict(historical_data):
        """马尔可夫预测（仅用于测试）"""
        # 简化版马尔可夫预测，基于历史数据
        try:
            from improvements.enhanced_markov import get_markov_predictor
            predictor = get_markov_predictor()
            results = predictor.markov_predict(count=3, periods=100)
            return [(pred['front'], pred['back']) for pred in results]
        except:
            # 如果马尔可夫预测器不可用，使用频率分析
            return mock_frequency_predict(historical_data)
    
    # 注册模型
    print("\n📝 注册测试模型...")
    benchmark.register_model("随机预测", mock_random_predict, "完全随机的预测方法", "baseline")
    benchmark.register_model("频率预测", mock_frequency_predict, "基于历史频率的预测方法", "traditional")
    benchmark.register_model("马尔可夫预测", mock_markov_predict, "基于马尔可夫链的预测方法", "advanced")
    
    # 评估模型
    print("\n📊 评估模型...")
    benchmark.evaluate_all_models(test_periods=20, verbose=True)
    
    # 比较模型
    print("\n🔄 比较模型...")
    benchmark.compare_models(verbose=True)
    
    # 生成报告
    print("\n📝 生成评估报告...")
    report = benchmark.generate_report("output/model_benchmark_report.md")
    
    # 可视化比较
    print("\n📈 可视化模型比较...")
    benchmark.visualize_comparison(output_path="output/model_comparison.png")
    
    # 可视化单个模型性能
    print("\n📊 可视化模型性能...")
    benchmark.visualize_model_performance("频率预测", output_path="output/frequency_model_performance.png")
    
    # 保存结果
    print("\n💾 保存评估结果...")
    benchmark.save_results("output/model_benchmark_results.json")
    
    print("\n✅ 模型基准测试框架测试完成")