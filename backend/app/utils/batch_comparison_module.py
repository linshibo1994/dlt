#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量预测对比功能模块
提供批量预测对比、中奖等级判定、统计分析和结果导出功能
"""

import os
import sys
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import concurrent.futures
import threading
import time

# 导入核心模块
from backend.app.core.core_modules import cache_manager, logger_manager, data_manager, task_manager

# 导入预测器
from backend.app.predictors.predictor_modules import get_traditional_predictor, get_advanced_predictor, get_super_predictor

# 导入分析器
from backend.app.analyzers.analyzer_modules import basic_analyzer, advanced_analyzer, comprehensive_analyzer

@dataclass
class BatchComparisonConfig:
    """批量对比配置类"""
    target_issue: str              # 目标期号，如 "25104"
    method: str                    # 预测方法，如 "markov", "frequency"
    analysis_periods: int          # 分析期数，如 100
    comparison_times: int          # 对比次数，如 50
    random_periods: bool = False   # 是否使用随机期数
    min_random_periods: int = 20   # 随机期数最小值
    max_random_periods: int = None # 随机期数最大值（None表示使用最大可用期数）
    export_excel: bool = False     # 是否导出Excel
    show_progress: bool = True     # 是否显示进度
    
    def __post_init__(self):
        """配置后处理"""
        if self.max_random_periods is None:
            # 获取最大可用期数
            df = data_manager.get_data()
            if df is not None:
                self.max_random_periods = len(df) - 1
            else:
                self.max_random_periods = 2000
    
    def validate(self) -> Tuple[bool, str]:
        """验证配置参数"""
        if not self.target_issue or len(self.target_issue) < 5:
            return False, "目标期号格式不正确"
        
        if self.comparison_times <= 0 or self.comparison_times > 10000:
            return False, "对比次数必须在1-10000之间"
        
        if self.random_periods:
            if self.min_random_periods >= self.max_random_periods:
                return False, "随机期数范围设置不正确"
        else:
            if self.analysis_periods <= 0 or self.analysis_periods > self.max_random_periods:
                return False, "分析期数必须在1-{}之间".format(self.max_random_periods)
        
        return True, ""

@dataclass
class PredictionRecord:
    """单次预测记录"""
    round_number: int             # 轮次编号
    analysis_periods: int         # 分析期数
    predicted_front: List[int]    # 预测前区号码
    predicted_back: List[int]     # 预测后区号码
    actual_front: List[int]       # 实际前区号码
    actual_back: List[int]        # 实际后区号码
    front_hits: int              # 前区命中数
    back_hits: int               # 后区命中数
    prize_level: int             # 中奖等级（0表示未中奖，1-9表示对应等级）
    execution_time: float        # 执行时间（秒）
    timestamp: datetime          # 预测时间

@dataclass
class PrizeStatistics:
    """中奖统计"""
    total_comparisons: int                    # 总对比次数
    prize_counts: Dict[int, int]             # 各等级中奖次数
    prize_probabilities: Dict[int, float]    # 各等级中奖概率
    total_wins: int                          # 总中奖次数
    total_win_rate: float                    # 总中奖率
    avg_execution_time: float                # 平均执行时间
    total_execution_time: float              # 总执行时间

class PrizeChecker:
    """中奖等级判定器"""
    
    # 大乐透正确的中奖等级规则（2019年新规则）
    PRIZE_RULES = {
        1: {'front': 5, 'back': 2},  # 一等奖: 5+2
        2: {'front': 5, 'back': 1},  # 二等奖: 5+1
        3: {'front': 5, 'back': 0},  # 三等奖: 5+0
        4: {'front': 4, 'back': 2},  # 四等奖: 4+2
        5: {'front': 4, 'back': 1},  # 五等奖: 4+1
        6: [{'front': 4, 'back': 0}, {'front': 3, 'back': 2}],  # 六等奖: 4+0 或 3+2
        7: {'front': 3, 'back': 1},  # 七等奖: 3+1
        8: [{'front': 3, 'back': 0}, {'front': 2, 'back': 2}],  # 八等奖: 3+0 或 2+2
        9: [{'front': 2, 'back': 1}, {'front': 1, 'back': 2}, {'front': 0, 'back': 2}]   # 九等奖: 2+1 或 1+2 或 0+2
    }
    
    # 固定奖金（元）
    FIXED_PRIZES = {
        3: 10000,  # 三等奖
        4: 3000,   # 四等奖
        5: 300,    # 五等奖
        6: 200,    # 六等奖
        7: 100,    # 七等奖
        8: 15,     # 八等奖
        9: 5       # 九等奖
    }
    
    @staticmethod
    def check_prize_level(front_hits: int, back_hits: int) -> int:
        """
        判定中奖等级
        
        Args:
            front_hits: 前区命中数量
            back_hits: 后区命中数量
            
        Returns:
            中奖等级 (0表示未中奖，1-9表示对应等级)
        """
        # 按等级从高到低检查
        for level in range(1, 10):
            rule = PrizeChecker.PRIZE_RULES[level]
            
            if isinstance(rule, dict):
                # 单一规则
                if front_hits == rule['front'] and back_hits == rule['back']:
                    return level
            else:
                # 多个规则（八等奖和九等奖）
                for sub_rule in rule:
                    if front_hits == sub_rule['front'] and back_hits == sub_rule['back']:
                        return level
        
        return 0  # 未中奖
    
    @staticmethod
    def get_prize_description(level: int) -> str:
        """获取奖级描述"""
        if level == 0:
            return "未中奖"
        elif level <= 2:
            return f"{level}等奖 (浮动奖金)"
        else:
            prize_amount = PrizeChecker.FIXED_PRIZES.get(level, 0)
            return f"{level}等奖 ({prize_amount}元)"
    
    @staticmethod
    def calculate_hits(predicted: List[int], actual: List[int]) -> int:
        """计算命中数量"""
        return len(set(predicted) & set(actual))

class BatchComparison:
    """批量对比器"""
    
    def __init__(self):
        self.predictors = {
            'traditional': None,
            'advanced': None,
            'super': None
        }
        self._load_predictors()
    
    def _load_predictors(self):
        """延迟加载预测器"""
        try:
            self.predictors['traditional'] = get_traditional_predictor()
            self.predictors['advanced'] = get_advanced_predictor()
            self.predictors['super'] = get_super_predictor()
            logger_manager.info("批量对比器：预测器加载完成")
        except Exception as e:
            logger_manager.error(f"批量对比器：预测器加载失败 - {e}")
    
    def execute(self, config: BatchComparisonConfig, progress_callback=None) -> 'ComparisonResult':
        """
        执行批量对比
        
        Args:
            config: 对比配置
            progress_callback: 进度回调函数，接收参数(current, total, message)
            
        Returns:
            对比结果
        """
        # 验证配置
        is_valid, error_msg = config.validate()
        if not is_valid:
            raise ValueError(f"配置验证失败: {error_msg}")
        
        logger_manager.info(f"开始批量对比: 期号={config.target_issue}, 方法={config.method}, 次数={config.comparison_times}")
        
        # 获取目标期号的实际开奖号码
        actual_numbers = self._get_actual_numbers(config.target_issue)
        if not actual_numbers:
            raise ValueError(f"无法获取期号 {config.target_issue} 的开奖数据")
        
        actual_front, actual_back = actual_numbers
        
        # 执行批量预测对比
        start_time = time.time()
        predictions = []
        
        for i in range(config.comparison_times):
            try:
                # 更新进度
                if progress_callback:
                    progress_callback(i + 1, config.comparison_times, 
                                    f"执行第 {i + 1}/{config.comparison_times} 次预测对比")
                
                # 单次对比
                record = self._single_comparison(config, actual_front, actual_back, i + 1)
                predictions.append(record)
                
                # 显示进度
                if config.show_progress and (i + 1) % max(1, config.comparison_times // 10) == 0:
                    progress = (i + 1) / config.comparison_times * 100
                    print(f"进度: {progress:.1f}% ({i + 1}/{config.comparison_times})")
                
            except Exception as e:
                logger_manager.error(f"第 {i + 1} 次对比失败: {e}")
                continue
        
        total_time = time.time() - start_time
        
        # 计算统计信息
        statistics = self._calculate_statistics(predictions, total_time)
        
        # 创建结果对象
        result = ComparisonResult(
            config=config,
            predictions=predictions,
            statistics=statistics,
            actual_front=actual_front,
            actual_back=actual_back,
            execution_time=total_time,
            timestamp=datetime.now()
        )
        
        logger_manager.info(f"批量对比完成: 总时间={total_time:.2f}秒, 总中奖率={statistics.total_win_rate:.2%}")
        
        return result
    
    def _get_actual_numbers(self, target_issue: str) -> Optional[Tuple[List[int], List[int]]]:
        """获取指定期号的实际开奖号码"""
        try:
            df = data_manager.get_data()
            if df is None:
                return None
            
            # 查找目标期号（转换为整数进行比较）
            try:
                target_issue_int = int(target_issue)
                target_row = df[df['issue'] == target_issue_int]
            except ValueError:
                # 如果转换失败，尝试字符串比较
                target_row = df[df['issue'] == target_issue]
            
            if target_row.empty:
                logger_manager.warning(f"未找到期号 {target_issue} 的开奖数据")
                return None
            
            # 解析号码
            front_balls, back_balls = data_manager.parse_balls(target_row.iloc[0])
            return front_balls, back_balls
            
        except Exception as e:
            logger_manager.error(f"获取开奖数据失败: {e}")
            return None
    
    def _single_comparison(self, config: BatchComparisonConfig, 
                          actual_front: List[int], actual_back: List[int], 
                          round_number: int) -> PredictionRecord:
        """执行单次预测对比"""
        start_time = time.time()
        
        # 确定分析期数
        if config.random_periods:
            analysis_periods = random.randint(config.min_random_periods, config.max_random_periods)
        else:
            analysis_periods = config.analysis_periods
        
        # 执行预测
        prediction = self._execute_prediction(config.method, analysis_periods)
        
        if not prediction:
            raise Exception("预测执行失败")
        
        predicted_front, predicted_back = prediction
        
        # 计算命中数
        front_hits = PrizeChecker.calculate_hits(predicted_front, actual_front)
        back_hits = PrizeChecker.calculate_hits(predicted_back, actual_back)
        
        # 判定中奖等级
        prize_level = PrizeChecker.check_prize_level(front_hits, back_hits)
        
        execution_time = time.time() - start_time
        
        return PredictionRecord(
            round_number=round_number,
            analysis_periods=analysis_periods,
            predicted_front=predicted_front,
            predicted_back=predicted_back,
            actual_front=actual_front,
            actual_back=actual_back,
            front_hits=front_hits,
            back_hits=back_hits,
            prize_level=prize_level,
            execution_time=execution_time,
            timestamp=datetime.now()
        )
    
    def _execute_prediction(self, method: str, periods: int) -> Optional[Tuple[List[int], List[int]]]:
        """执行预测"""
        try:
            result = None

            # 传统统计方法
            if method in ['frequency', 'hot_cold', 'missing']:
                predictor = self.predictors['traditional']
                if method == 'frequency':
                    result = predictor.frequency_predict(count=1, periods=periods)
                elif method == 'hot_cold':
                    result = predictor.hot_cold_predict(count=1, periods=periods)
                elif method == 'missing':
                    result = predictor.missing_predict(count=1, periods=periods)

            elif method == 'bayesian':
                predictor = self.predictors['traditional']
                result = predictor.bayesian_predict(count=1, periods=periods)

            # 马尔可夫链方法
            elif method in ['markov', 'markov_2nd', 'markov_3rd', 'adaptive_markov', 'markov_custom']:
                predictor = self.predictors['advanced']
                if method == 'markov':
                    result = predictor.markov_predict(count=1, periods=periods)
                elif method == 'markov_2nd':
                    result = predictor.markov_2nd_predict(count=1, periods=periods)
                elif method == 'markov_3rd':
                    result = predictor.markov_3rd_predict(count=1, periods=periods)
                elif method == 'adaptive_markov':
                    result = predictor.adaptive_markov_predict(count=1, periods=periods)
                elif method == 'markov_custom':
                    result = predictor.markov_predict_custom(count=1, analysis_periods=periods)

            # 高级算法
            elif method in ['ensemble', 'clustering']:
                predictor = self.predictors['advanced']
                if method == 'ensemble':
                    result = predictor.ensemble_predict(count=1, periods=periods)
                elif method == 'clustering':
                    result = predictor.clustering_predict(count=1, periods=periods)

            # 集成学习方法
            elif method in ['stacking', 'adaptive_ensemble', 'ultimate_ensemble', 'enhanced']:
                predictor = self.predictors['advanced']
                if method == 'stacking':
                    result = predictor.stacking_predict(count=1, periods=periods)
                elif method == 'adaptive_ensemble':
                    result = predictor.adaptive_ensemble_predict(count=1, periods=periods)
                elif method == 'ultimate_ensemble':
                    result = predictor.ultimate_ensemble_predict(count=1, periods=periods)
                elif method == 'enhanced':
                    result = predictor.enhanced_predict(count=1, periods=periods)

            # 智能预测方法（注意：这些方法定义在 AdvancedPredictor 上，不是 SuperPredictor）
            elif method in ['super', 'adaptive', 'nine_models', 'advanced_integration', 'mixed_strategy', 'highly_integrated']:
                predictor = self.predictors['advanced']
                if method == 'super':
                    result = predictor.super_predict(count=1, periods=periods)
                elif method == 'adaptive':
                    result = predictor.adaptive_predict(count=1, periods=periods)
                elif method == 'nine_models':
                    result = predictor.nine_models_predict(count=1, periods=periods)
                elif method == 'advanced_integration':
                    result = predictor.advanced_integration_predict(count=1, periods=periods)
                elif method == 'mixed_strategy':
                    result = predictor.mixed_strategy_predict(count=1, periods=periods)
                elif method == 'highly_integrated':
                    result = predictor.highly_integrated_predict(count=1, periods=periods)

            # 复式投注方法 - 提取单注预测结果
            elif method in ['markov_compound', 'nine_models_compound']:
                predictor = self.predictors['advanced']
                try:
                    if method == 'markov_compound':
                        compound_result = predictor.markov_compound_predict(
                            front_count=8, back_count=4, analysis_periods=periods
                        )
                    else:
                        compound_result = predictor.nine_models_compound_predict(
                            front_count=8, back_count=4, analysis_periods=periods
                        )
                    if isinstance(compound_result, dict):
                        front = sorted(compound_result.get('front_balls', []))[:5]
                        back = sorted(compound_result.get('back_balls', []))[:2]
                        if len(front) == 5 and len(back) == 2:
                            return (front, back)
                except Exception as e:
                    logger_manager.warning(f"复式预测 {method} 失败，回退到频率分析: {e}")
                result = self.predictors['traditional'].frequency_predict(count=1, periods=periods)

            # compound/duplex 是复式投注模式，使用频率分析作为回退
            elif method in ['compound', 'duplex']:
                result = self.predictors['traditional'].frequency_predict(count=1, periods=periods)

            # 深度学习方法
            elif method in ['lstm', 'transformer', 'gan']:
                try:
                    from backend.app.predictors.deep_learning.advanced_lstm_predictor import AdvancedLSTMPredictor
                    dl_predictor = AdvancedLSTMPredictor()
                    dl_result = dl_predictor.predict(count=1)
                    if dl_result and len(dl_result) > 0:
                        result = [dl_result[0]]
                    else:
                        result = self.predictors['advanced'].ensemble_predict(count=1, periods=periods)
                except Exception as e:
                    logger_manager.warning(f"深度学习方法 {method} 不可用，回退到集成方法: {e}")
                    result = self.predictors['advanced'].ensemble_predict(count=1, periods=periods)

            # 交集递减方法 - 使用集成方法作为代理
            elif method == 'consensus_halving':
                result = self.predictors['advanced'].ensemble_predict(count=1, periods=periods)

            else:
                logger_manager.error(f"不支持的预测方法: {method}")
                return None

            if result and len(result) > 0:
                return result[0]
            else:
                return None

        except Exception as e:
            logger_manager.error(f"预测执行失败: {e}")
            return None
    
    def _calculate_statistics(self, predictions: List[PredictionRecord], 
                             total_time: float) -> PrizeStatistics:
        """计算统计信息"""
        if not predictions:
            return PrizeStatistics(
                total_comparisons=0,
                prize_counts={},
                prize_probabilities={},
                total_wins=0,
                total_win_rate=0.0,
                avg_execution_time=0.0,
                total_execution_time=total_time
            )
        
        total_comparisons = len(predictions)
        
        # 统计各等级中奖次数
        prize_counts = defaultdict(int)
        for record in predictions:
            prize_counts[record.prize_level] += 1
        
        # 计算中奖概率
        prize_probabilities = {}
        for level, count in prize_counts.items():
            prize_probabilities[level] = count / total_comparisons
        
        # 总中奖次数和中奖率
        total_wins = sum(count for level, count in prize_counts.items() if level > 0)
        total_win_rate = total_wins / total_comparisons if total_comparisons > 0 else 0.0
        
        # 平均执行时间
        avg_execution_time = sum(r.execution_time for r in predictions) / total_comparisons
        
        return PrizeStatistics(
            total_comparisons=total_comparisons,
            prize_counts=dict(prize_counts),
            prize_probabilities=prize_probabilities,
            total_wins=total_wins,
            total_win_rate=total_win_rate,
            avg_execution_time=avg_execution_time,
            total_execution_time=total_time
        )

class ComparisonResult:
    """对比结果类"""
    
    def __init__(self, config: BatchComparisonConfig, predictions: List[PredictionRecord],
                 statistics: PrizeStatistics, actual_front: List[int], actual_back: List[int],
                 execution_time: float, timestamp: datetime):
        self.config = config
        self.predictions = predictions
        self.statistics = statistics
        self.actual_front = actual_front
        self.actual_back = actual_back
        self.execution_time = execution_time
        self.timestamp = timestamp
    
    def get_summary(self) -> Dict[str, Any]:
        """获取结果摘要"""
        return {
            'config': asdict(self.config),
            'actual_numbers': {
                'front': self.actual_front,
                'back': self.actual_back
            },
            'statistics': asdict(self.statistics),
            'execution_info': {
                'total_time': self.execution_time,
                'timestamp': self.timestamp.isoformat()
            }
        }
    
    def print_summary(self):
        """打印结果摘要"""
        print("\n" + "="*60)
        print("📊 批量预测对比结果摘要")
        print("="*60)
        
        # 基本信息
        print(f"🎯 目标期号: {self.config.target_issue}")
        print(f"🔮 预测方法: {self.config.method}")
        print(f"📈 分析期数: {self.config.analysis_periods}" + 
              (" (随机)" if self.config.random_periods else ""))
        print(f"🔄 对比次数: {self.config.comparison_times}")
        print(f"⏱️  总执行时间: {self.execution_time:.2f}秒")
        
        # 实际开奖号码
        front_str = " ".join([str(n).zfill(2) for n in self.actual_front])
        back_str = " ".join([str(n).zfill(2) for n in self.actual_back])
        print(f"🎲 实际开奖: {front_str} + {back_str}")
        
        # 中奖统计
        print(f"\n🏆 中奖统计:")
        print(f"   总中奖次数: {self.statistics.total_wins}/{self.statistics.total_comparisons}")
        print(f"   总中奖率: {self.statistics.total_win_rate:.2%}")
        print(f"   平均执行时间: {self.statistics.avg_execution_time:.3f}秒")
        
        # 各等级中奖详情
        print(f"\n🎊 各等级中奖详情:")
        for level in range(1, 10):
            count = self.statistics.prize_counts.get(level, 0)
            prob = self.statistics.prize_probabilities.get(level, 0.0)
            if count > 0:
                description = PrizeChecker.get_prize_description(level)
                print(f"   {description}: {count}次 ({prob:.2%})")
        
        # 未中奖
        no_win_count = self.statistics.prize_counts.get(0, 0)
        no_win_prob = self.statistics.prize_probabilities.get(0, 0.0)
        print(f"   未中奖: {no_win_count}次 ({no_win_prob:.2%})")
        
        print("="*60)
    
    def export_to_excel(self, filename: str = None) -> str:
        """导出到Excel文件"""
        if filename is None:
            timestamp = self.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"batch_comparison_{self.config.target_issue}_{self.config.method}_{timestamp}.xlsx"
        
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # 工作表1: 配置信息
                config_df = pd.DataFrame([{
                    '目标期号': self.config.target_issue,
                    '预测方法': self.config.method,
                    '分析期数': self.config.analysis_periods,
                    '对比次数': self.config.comparison_times,
                    '随机期数': '是' if self.config.random_periods else '否',
                    '总执行时间(秒)': round(self.execution_time, 2),
                    '生成时间': self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                }])
                config_df.to_excel(writer, sheet_name='配置信息', index=False)
                
                # 工作表2: 实际开奖号码
                actual_df = pd.DataFrame([{
                    '期号': self.config.target_issue,
                    '前区号码': ' '.join([str(n).zfill(2) for n in self.actual_front]),
                    '后区号码': ' '.join([str(n).zfill(2) for n in self.actual_back])
                }])
                actual_df.to_excel(writer, sheet_name='开奖号码', index=False)
                
                # 工作表3: 中奖统计
                stats_data = []
                for level in range(0, 10):
                    count = self.statistics.prize_counts.get(level, 0)
                    prob = self.statistics.prize_probabilities.get(level, 0.0)
                    stats_data.append({
                        '奖级': '未中奖' if level == 0 else f'{level}等奖',
                        '中奖次数': count,
                        '中奖概率': f'{prob:.4%}',
                        '描述': PrizeChecker.get_prize_description(level)
                    })
                
                stats_df = pd.DataFrame(stats_data)
                stats_df.to_excel(writer, sheet_name='中奖统计', index=False)
                
                # 工作表4: 详细记录
                records_data = []
                for record in self.predictions:
                    front_str = ' '.join([str(n).zfill(2) for n in record.predicted_front])
                    back_str = ' '.join([str(n).zfill(2) for n in record.predicted_back])
                    
                    records_data.append({
                        '轮次': record.round_number,
                        '分析期数': record.analysis_periods,
                        '预测前区': front_str,
                        '预测后区': back_str,
                        '前区命中': record.front_hits,
                        '后区命中': record.back_hits,
                        '中奖等级': '未中奖' if record.prize_level == 0 else f'{record.prize_level}等奖',
                        '执行时间(秒)': round(record.execution_time, 3),
                        '预测时间': record.timestamp.strftime("%H:%M:%S")
                    })
                
                records_df = pd.DataFrame(records_data)
                records_df.to_excel(writer, sheet_name='详细记录', index=False)
            
            logger_manager.info(f"Excel文件已导出: {filename}")
            return filename
            
        except Exception as e:
            logger_manager.error(f"Excel导出失败: {e}")
            raise Exception(f"Excel导出失败: {e}")
    
    def to_json(self) -> str:
        """转换为JSON格式"""
        data = {
            'config': asdict(self.config),
            'actual_numbers': {
                'front': self.actual_front,
                'back': self.actual_back
            },
            'statistics': asdict(self.statistics),
            'predictions': [asdict(record) for record in self.predictions],
            'execution_time': self.execution_time,
            'timestamp': self.timestamp.isoformat()
        }
        
        # 处理datetime对象
        def datetime_handler(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object {obj} is not JSON serializable")
        
        return json.dumps(data, ensure_ascii=False, indent=2, default=datetime_handler)

# 便捷函数
def quick_comparison(target_issue: str, method: str = "markov", periods: int = 100, 
                    times: int = 10, random_periods: bool = False, 
                    export: bool = False, show_progress: bool = True) -> ComparisonResult:
    """
    快速批量对比
    
    Args:
        target_issue: 目标期号
        method: 预测方法
        periods: 分析期数
        times: 对比次数
        random_periods: 是否随机期数
        export: 是否导出Excel
        show_progress: 是否显示进度
        
    Returns:
        对比结果
    """
    config = BatchComparisonConfig(
        target_issue=target_issue,
        method=method,
        analysis_periods=periods,
        comparison_times=times,
        random_periods=random_periods,
        export_excel=export,
        show_progress=show_progress
    )
    
    batch_comparison = BatchComparison()
    result = batch_comparison.execute(config)
    
    # 显示结果摘要
    result.print_summary()
    
    # 导出Excel
    if export:
        try:
            filename = result.export_to_excel()
            print(f"\n📊 Excel文件已导出: {filename}")
        except Exception as e:
            print(f"\n❌ Excel导出失败: {e}")
    
    return result

# 可用的预测方法列表
AVAILABLE_METHODS = [
    'frequency', 'hot_cold', 'missing',  # 传统统计方法
    'markov', 'markov_2nd', 'markov_3rd', 'adaptive_markov', 'bayesian', 'ensemble', 'clustering',  # 高级方法
    'super', 'adaptive', 'nine_models', 'advanced_integration', 'mixed_strategy', 'highly_integrated'  # 超级方法
]

def get_method_description(method: str) -> str:
    """获取预测方法描述"""
    descriptions = {
        'frequency': '频率分析预测',
        'hot_cold': '冷热号分析预测',
        'missing': '遗漏值分析预测',
        'markov': '1阶马尔可夫链预测',
        'markov_2nd': '2阶马尔可夫链预测',
        'markov_3rd': '3阶马尔可夫链预测',
        'adaptive_markov': '自适应马尔可夫链预测',
        'bayesian': '贝叶斯分析预测',
        'ensemble': '集成学习预测',
        'clustering': '聚类分析预测',
        'super': '超级预测',
        'adaptive': '自适应预测',
        'nine_models': '9种数学模型预测',
        'advanced_integration': '高级集成分析预测',
        'mixed_strategy': '混合策略预测',
        'highly_integrated': '高度集成预测'
    }
    return descriptions.get(method, method)

if __name__ == "__main__":
    # 测试代码
    print("🧪 批量预测对比模块测试")
    
    # 快速测试
    try:
        result = quick_comparison(
            target_issue="25103",
            method="frequency",
            periods=50,
            times=5,
            random_periods=False,
            export=False,
            show_progress=True
        )
        print("✅ 测试成功")
    except Exception as e:
        print(f"❌ 测试失败: {e}")