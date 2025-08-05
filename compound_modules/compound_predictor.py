#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
复式预测基类模块
Compound Predictor Base Module

提供统一的复式预测接口和功能。
"""

import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

from . import logger_manager


@dataclass
class CompoundConfig:
    """复式预测配置"""
    front_count: int = 8  # 前区号码数量 (6-15)
    back_count: int = 4   # 后区号码数量 (3-12)
    periods: int = 500    # 分析期数
    max_cost: int = 10000 # 最大投注成本 (元)
    min_confidence: float = 0.5  # 最小置信度
    
    def __post_init__(self):
        """验证配置参数"""
        if not (6 <= self.front_count <= 15):
            raise ValueError("前区号码数量必须在6-15之间")
        if not (3 <= self.back_count <= 12):
            raise ValueError("后区号码数量必须在3-12之间")
        if not (50 <= self.periods <= 2748):
            raise ValueError("分析期数必须在50-2748之间")


@dataclass
class CompoundResult:
    """复式预测结果"""
    front_balls: List[int]
    back_balls: List[int]
    front_count: int
    back_count: int
    total_combinations: int
    total_cost: int
    confidence: float
    method: str
    analysis_periods: int
    timestamp: str
    details: Dict[str, Any] = None
    
    def format_display(self) -> str:
        """格式化显示结果"""
        front_str = ' '.join([str(b).zfill(2) for b in self.front_balls])
        back_str = ' '.join([str(b).zfill(2) for b in self.back_balls])
        
        return f"""
复式预测结果 ({self.method}):
前区号码: {front_str} ({self.front_count}个)
后区号码: {back_str} ({self.back_count}个)
总组合数: {self.total_combinations:,} 注
投注成本: {self.total_cost:,} 元
置信度: {self.confidence:.3f}
分析期数: {self.analysis_periods}
生成时间: {self.timestamp}
"""


class CompoundPredictorMixin(ABC):
    """复式预测混入类"""
    
    def __init__(self):
        """初始化复式预测混入"""
        self.compound_config = None
    
    def configure_compound(self, config: CompoundConfig) -> None:
        """配置复式预测参数"""
        self.compound_config = config
        logger_manager.info(f"复式预测配置: {config.front_count}+{config.back_count}, 期数: {config.periods}")
    
    @abstractmethod
    def predict_compound(self, config: Optional[CompoundConfig] = None) -> CompoundResult:
        """
        复式预测抽象方法
        
        Args:
            config: 复式预测配置，如果为None则使用默认配置
            
        Returns:
            复式预测结果
        """
        pass
    
    def calculate_combinations(self, front_count: int, back_count: int) -> int:
        """
        计算复式投注组合数
        
        Args:
            front_count: 前区号码数量
            back_count: 后区号码数量
            
        Returns:
            总组合数
        """
        try:
            front_combinations = math.comb(front_count, 5)
            back_combinations = math.comb(back_count, 2)
            total_combinations = front_combinations * back_combinations
            
            logger_manager.debug(f"组合数计算: C({front_count},5) × C({back_count},2) = {front_combinations} × {back_combinations} = {total_combinations}")
            
            return total_combinations
        except Exception as e:
            logger_manager.error(f"计算组合数失败: {e}")
            return 0
    
    def calculate_cost(self, combinations: int, cost_per_bet: int = 3) -> int:
        """
        计算投注成本
        
        Args:
            combinations: 组合数
            cost_per_bet: 每注成本 (默认3元)
            
        Returns:
            总成本
        """
        return combinations * cost_per_bet
    
    def validate_compound_params(self, front_count: int, back_count: int, max_cost: int = 10000) -> bool:
        """
        验证复式参数
        
        Args:
            front_count: 前区号码数量
            back_count: 后区号码数量
            max_cost: 最大成本限制
            
        Returns:
            是否有效
        """
        # 检查号码数量范围
        if not (6 <= front_count <= 15):
            logger_manager.error(f"前区号码数量 {front_count} 超出范围 [6, 15]")
            return False
        
        if not (3 <= back_count <= 12):
            logger_manager.error(f"后区号码数量 {back_count} 超出范围 [3, 12]")
            return False
        
        # 检查成本限制
        combinations = self.calculate_combinations(front_count, back_count)
        cost = self.calculate_cost(combinations)
        
        if cost > max_cost:
            logger_manager.error(f"投注成本 {cost} 元超出限制 {max_cost} 元")
            return False
        
        logger_manager.info(f"复式参数验证通过: {front_count}+{back_count}, 成本: {cost} 元")
        return True
    
    def estimate_win_probability(self, front_count: int, back_count: int) -> Dict[str, float]:
        """
        估算中奖概率
        
        Args:
            front_count: 前区号码数量
            back_count: 后区号码数量
            
        Returns:
            各等奖中奖概率
        """
        try:
            # 大乐透总的组合数
            total_combinations = math.comb(35, 5) * math.comb(12, 2)
            
            # 用户选择的组合数
            user_combinations = self.calculate_combinations(front_count, back_count)
            
            # 一等奖概率 (5+2)
            first_prize_prob = user_combinations / total_combinations
            
            # 二等奖概率 (5+1)
            second_prize_combinations = math.comb(front_count, 5) * math.comb(back_count, 1) * math.comb(12-back_count, 1)
            second_prize_prob = second_prize_combinations / total_combinations
            
            # 三等奖概率 (5+0)
            third_prize_combinations = math.comb(front_count, 5) * math.comb(12-back_count, 2)
            third_prize_prob = third_prize_combinations / total_combinations
            
            # 四等奖概率 (4+2)
            fourth_prize_combinations = math.comb(front_count, 4) * math.comb(35-front_count, 1) * math.comb(back_count, 2)
            fourth_prize_prob = fourth_prize_combinations / total_combinations
            
            # 五等奖概率 (4+1)
            fifth_prize_combinations = (math.comb(front_count, 4) * math.comb(35-front_count, 1) * 
                                      math.comb(back_count, 1) * math.comb(12-back_count, 1))
            fifth_prize_prob = fifth_prize_combinations / total_combinations
            
            return {
                'first_prize': first_prize_prob,
                'second_prize': second_prize_prob,
                'third_prize': third_prize_prob,
                'fourth_prize': fourth_prize_prob,
                'fifth_prize': fifth_prize_prob,
                'any_prize': first_prize_prob + second_prize_prob + third_prize_prob + fourth_prize_prob + fifth_prize_prob
            }
            
        except Exception as e:
            logger_manager.error(f"计算中奖概率失败: {e}")
            return {}
    
    def optimize_compound_selection(self, candidates: List[int], target_count: int, 
                                  is_front: bool = True, strategy: str = "balanced") -> List[int]:
        """
        优化复式号码选择
        
        Args:
            candidates: 候选号码列表
            target_count: 目标数量
            is_front: 是否为前区
            strategy: 选择策略 ("balanced", "aggressive", "conservative")
            
        Returns:
            优化后的号码列表
        """
        if len(candidates) <= target_count:
            return sorted(candidates)
        
        # 根据策略选择号码
        if strategy == "aggressive":
            # 激进策略：选择得分最高的号码
            return sorted(candidates[:target_count])
        elif strategy == "conservative":
            # 保守策略：选择得分适中的号码
            mid_start = len(candidates) // 4
            mid_end = mid_start + target_count
            return sorted(candidates[mid_start:mid_end])
        else:
            # 平衡策略：混合选择
            high_count = target_count // 2
            low_count = target_count - high_count
            
            high_numbers = candidates[:high_count]
            low_numbers = candidates[-low_count:]
            
            return sorted(high_numbers + low_numbers)
    
    def generate_compound_variations(self, base_front: List[int], base_back: List[int], 
                                   variation_count: int = 3) -> List[Tuple[List[int], List[int]]]:
        """
        生成复式变化组合
        
        Args:
            base_front: 基础前区号码
            base_back: 基础后区号码
            variation_count: 变化数量
            
        Returns:
            变化组合列表
        """
        variations = [(base_front, base_back)]
        
        try:
            for i in range(variation_count - 1):
                # 前区变化：替换1-2个号码
                var_front = base_front.copy()
                replace_count = min(2, len(var_front) - 6)  # 保证至少6个号码
                
                if replace_count > 0:
                    # 简单的变化策略：调整边界号码
                    for j in range(replace_count):
                        if var_front[j] > 1:
                            var_front[j] -= 1
                        elif var_front[-(j+1)] < 35:
                            var_front[-(j+1)] += 1
                
                # 后区变化：替换1个号码
                var_back = base_back.copy()
                if len(var_back) > 3 and var_back[0] > 1:
                    var_back[0] -= 1
                elif len(var_back) > 3 and var_back[-1] < 12:
                    var_back[-1] += 1
                
                variations.append((sorted(var_front), sorted(var_back)))
            
        except Exception as e:
            logger_manager.error(f"生成复式变化失败: {e}")
        
        return variations


class CompoundPredictorBase(CompoundPredictorMixin):
    """复式预测基类"""
    
    def __init__(self, method_name: str):
        """
        初始化复式预测基类
        
        Args:
            method_name: 预测方法名称
        """
        super().__init__()
        self.method_name = method_name
        logger_manager.info(f"初始化复式预测器: {method_name}")
    
    def predict_compound(self, config: Optional[CompoundConfig] = None) -> CompoundResult:
        """
        默认复式预测实现
        
        Args:
            config: 复式预测配置
            
        Returns:
            复式预测结果
        """
        if config is None:
            config = self.compound_config or CompoundConfig()
        
        # 验证参数
        if not self.validate_compound_params(config.front_count, config.back_count, config.max_cost):
            raise ValueError("复式预测参数验证失败")
        
        # 生成基础预测（子类应该重写此方法）
        front_balls, back_balls = self._generate_base_prediction(config)
        
        # 计算组合数和成本
        combinations = self.calculate_combinations(config.front_count, config.back_count)
        cost = self.calculate_cost(combinations)
        
        # 创建结果
        from datetime import datetime
        result = CompoundResult(
            front_balls=front_balls,
            back_balls=back_balls,
            front_count=config.front_count,
            back_count=config.back_count,
            total_combinations=combinations,
            total_cost=cost,
            confidence=config.min_confidence,
            method=self.method_name,
            analysis_periods=config.periods,
            timestamp=datetime.now().isoformat()
        )
        
        logger_manager.info(f"复式预测完成: {self.method_name}, 组合数: {combinations}, 成本: {cost}")
        return result
    
    def _generate_base_prediction(self, config: CompoundConfig) -> Tuple[List[int], List[int]]:
        """
        生成基础预测（子类应该重写）
        
        Args:
            config: 复式预测配置
            
        Returns:
            (前区号码, 后区号码)
        """
        # 默认实现：随机选择
        import random
        
        front_balls = sorted(random.sample(range(1, 36), config.front_count))
        back_balls = sorted(random.sample(range(1, 13), config.back_count))
        
        logger_manager.warning(f"使用默认随机预测: {self.method_name}")
        return front_balls, back_balls
