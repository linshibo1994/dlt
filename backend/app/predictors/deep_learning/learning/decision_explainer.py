#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
决策解释器
解释模型选择和权重调整决策
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union
from datetime import datetime

from .performance_tracker import PerformanceTracker
from .weight_optimizer import WeightOptimizer
from core_modules import logger_manager


class DecisionExplainer:
    """决策解释器"""
    
    def __init__(self, performance_tracker: Optional[PerformanceTracker] = None,
               weight_optimizer: Optional[WeightOptimizer] = None):
        """
        初始化决策解释器
        
        Args:
            performance_tracker: 性能跟踪器，如果为None则创建新的
            weight_optimizer: 权重优化器，如果为None则创建新的
        """
        self.performance_tracker = performance_tracker or PerformanceTracker()
        self.weight_optimizer = weight_optimizer or WeightOptimizer(self.performance_tracker)
        
        logger_manager.info("初始化决策解释器")
    
    def explain_model_selection(self, model_id: str, metric_name: str = 'overall_score',
                              window: Optional[int] = None) -> Dict[str, Any]:
        """
        解释模型选择
        
        Args:
            model_id: 模型ID
            metric_name: 指标名称
            window: 窗口大小，如果为None则使用全部历史
            
        Returns:
            解释字典
        """
        # 获取模型性能指标
        metrics = self.performance_tracker.get_model_metrics(model_id)
        
        if not metrics:
            return {
                'model_id': model_id,
                'explanation': f"模型 {model_id} 没有性能记录"
            }
        
        # 获取平均指标
        avg_metrics = metrics.get_average_metrics(window)
        
        # 获取所有模型的比较
        all_models = list(self.performance_tracker.get_all_model_metrics().keys())
        comparison = self.performance_tracker.compare_models(all_models, metric_name, window)
        
        # 获取最佳模型
        best_model = self.performance_tracker.get_best_model(metric_name, window)
        
        # 获取模型排名
        ranking = sorted(comparison.items(), key=lambda x: x[1], reverse=True)
        rank = next((i + 1 for i, (mid, _) in enumerate(ranking) if mid == model_id), -1)
        
        # 获取模型权重
        weights = self.weight_optimizer.get_weights()
        weight = weights.get(model_id, 0.0)
        
        # 生成解释
        explanation = {
            'model_id': model_id,
            'metrics': avg_metrics,
            'comparison': comparison,
            'best_model': best_model,
            'rank': rank,
            'weight': weight,
            'explanation': self._generate_model_explanation(model_id, avg_metrics, comparison, best_model, rank, weight)
        }
        
        return explanation
    
    def _generate_model_explanation(self, model_id: str, metrics: Dict[str, float],
                                  comparison: Dict[str, float], best_model: str,
                                  rank: int, weight: float) -> str:
        """
        生成模型解释
        
        Args:
            model_id: 模型ID
            metrics: 平均指标
            comparison: 模型比较
            best_model: 最佳模型
            rank: 模型排名
            weight: 模型权重
            
        Returns:
            解释文本
        """
        explanation = f"模型 {model_id} 的性能分析:\n\n"
        
        # 添加性能指标
        explanation += "性能指标:\n"
        for metric, value in metrics.items():
            explanation += f"- {metric}: {value:.4f}\n"
        
        explanation += "\n"
        
        # 添加排名信息
        if rank > 0:
            explanation += f"在所有模型中排名第 {rank}\n"
            
            if model_id == best_model:
                explanation += "这是当前性能最好的模型\n"
            else:
                best_score = comparison.get(best_model, 0.0)
                model_score = comparison.get(model_id, 0.0)
                diff = best_score - model_score
                explanation += f"与最佳模型 {best_model} 的差距: {diff:.4f}\n"
        
        explanation += "\n"
        
        # 添加权重信息
        explanation += f"在集成中的权重: {weight:.4f}\n"
        
        # 添加建议
        explanation += "\n建议:\n"
        
        if model_id == best_model:
            explanation += "- 保持当前配置，该模型表现良好\n"
        elif rank <= 3:
            explanation += "- 考虑增加该模型在集成中的权重\n"
            explanation += "- 可以尝试微调模型参数以进一步提高性能\n"
        else:
            explanation += "- 考虑减少该模型在集成中的权重\n"
            explanation += "- 可能需要重新训练或调整模型参数\n"
        
        return explanation
    
    def explain_weight_adjustment(self) -> Dict[str, Any]:
        """
        解释权重调整
        
        Returns:
            解释字典
        """
        # 获取权重解释
        weight_explanation = self.weight_optimizer.get_weight_explanation()
        
        # 获取更新历史
        update_history = self.weight_optimizer.get_update_history()
        
        if not update_history:
            return {
                'explanation': "尚未进行权重调整",
                'weights': self.weight_optimizer.get_weights()
            }
        
        # 获取权重历史趋势
        weight_trends = {}
        for model_id in self.weight_optimizer.get_weights():
            history = self.weight_optimizer.get_weight_history(model_id)
            if len(history) >= 2:
                initial = history[0]
                current = history[-1]
                change = current - initial
                trend = "上升" if change > 0.01 else "下降" if change < -0.01 else "稳定"
                weight_trends[model_id] = {
                    'initial': initial,
                    'current': current,
                    'change': change,
                    'trend': trend
                }
        
        # 生成解释
        explanation = {
            'weights': self.weight_optimizer.get_weights(),
            'weight_explanation': weight_explanation,
            'weight_trends': weight_trends,
            'update_count': len(update_history),
            'explanation': self._generate_weight_explanation(weight_explanation, weight_trends, update_history)
        }
        
        return explanation
    
    def _generate_weight_explanation(self, weight_explanation: Dict[str, Any],
                                   weight_trends: Dict[str, Dict[str, Any]],
                                   update_history: List[Dict[str, Any]]) -> str:
        """
        生成权重解释
        
        Args:
            weight_explanation: 权重解释
            weight_trends: 权重趋势
            update_history: 更新历史
            
        Returns:
            解释文本
        """
        explanation = "权重调整分析:\n\n"
        
        # 添加当前权重
        explanation += "当前权重:\n"
        for model_id, weight in weight_explanation['weights'].items():
            explanation += f"- {model_id}: {weight:.4f}\n"
        
        explanation += "\n"
        
        # 添加权重趋势
        explanation += "权重趋势:\n"
        for model_id, trend in weight_trends.items():
            explanation += f"- {model_id}: {trend['initial']:.4f} -> {trend['current']:.4f} ({trend['trend']})\n"
        
        explanation += "\n"
        
        # 添加调整原因
        explanation += "调整原因:\n"
        explanation += weight_explanation['explanation']
        
        explanation += "\n"
        
        # 添加建议
        explanation += "建议:\n"
        
        # 找出权重增加最多和减少最多的模型
        if weight_trends:
            max_increase = max(weight_trends.items(), key=lambda x: x[1]['change'])
            max_decrease = min(weight_trends.items(), key=lambda x: x[1]['change'])
            
            if max_increase[1]['change'] > 0.05:
                explanation += f"- 模型 {max_increase[0]} 的权重显著增加，表明其性能持续提升\n"
                explanation += "  可以考虑进一步优化该模型或增加类似模型\n"
            
            if max_decrease[1]['change'] < -0.05:
                explanation += f"- 模型 {max_decrease[0]} 的权重显著减少，表明其性能可能存在问题\n"
                explanation += "  建议检查该模型或考虑替换\n"
        
        # 检查是否有权重过于集中的情况
        weights = weight_explanation['weights']
        if weights and max(weights.values()) > 0.7:
            dominant_model = max(weights.items(), key=lambda x: x[1])[0]
            explanation += f"- 模型 {dominant_model} 的权重过高 ({weights[dominant_model]:.4f})，可能导致过度依赖\n"
            explanation += "  建议引入更多多样化的模型以提高集成的鲁棒性\n"
        
        return explanation
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """
        生成综合报告
        
        Returns:
            报告字典
        """
        # 获取所有模型
        all_models = list(self.performance_tracker.get_all_model_metrics().keys())
        
        if not all_models:
            return {
                'explanation': "没有模型数据，无法生成报告"
            }
        
        # 获取各模型的解释
        model_explanations = {}
        for model_id in all_models:
            model_explanations[model_id] = self.explain_model_selection(model_id)
        
        # 获取权重调整解释
        weight_explanation = self.explain_weight_adjustment()
        
        # 获取最佳模型
        best_model = self.performance_tracker.get_best_model()
        
        # 生成报告
        report = {
            'timestamp': datetime.now().isoformat(),
            'models': all_models,
            'best_model': best_model,
            'model_explanations': model_explanations,
            'weight_explanation': weight_explanation,
            'explanation': self._generate_report_explanation(all_models, best_model, model_explanations, weight_explanation)
        }
        
        return report
    
    def _generate_report_explanation(self, all_models: List[str], best_model: str,
                                   model_explanations: Dict[str, Dict[str, Any]],
                                   weight_explanation: Dict[str, Any]) -> str:
        """
        生成报告解释
        
        Args:
            all_models: 所有模型
            best_model: 最佳模型
            model_explanations: 模型解释
            weight_explanation: 权重解释
            
        Returns:
            解释文本
        """
        explanation = "模型性能与权重综合报告\n"
        explanation += "=" * 50 + "\n\n"
        
        # 添加概述
        explanation += f"模型数量: {len(all_models)}\n"
        explanation += f"最佳模型: {best_model}\n\n"
        
        # 添加模型性能摘要
        explanation += "模型性能摘要:\n"
        explanation += "-" * 30 + "\n"
        
        for model_id in all_models:
            model_expl = model_explanations[model_id]
            metrics = model_expl.get('metrics', {})
            overall_score = metrics.get('overall_score', 0.0)
            rank = model_expl.get('rank', -1)
            weight = model_expl.get('weight', 0.0)
            
            explanation += f"模型: {model_id}\n"
            explanation += f"- 综合得分: {overall_score:.4f}\n"
            explanation += f"- 排名: {rank}\n"
            explanation += f"- 权重: {weight:.4f}\n\n"
        
        # 添加权重调整摘要
        explanation += "权重调整摘要:\n"
        explanation += "-" * 30 + "\n"
        explanation += weight_explanation.get('explanation', "无权重调整信息") + "\n\n"
        
        # 添加建议
        explanation += "综合建议:\n"
        explanation += "-" * 30 + "\n"
        
        # 检查是否有性能差距过大的情况
        if best_model and len(all_models) > 1:
            best_score = model_explanations[best_model]['metrics'].get('overall_score', 0.0)
            worst_model = min(all_models, key=lambda m: model_explanations[m]['metrics'].get('overall_score', 0.0))
            worst_score = model_explanations[worst_model]['metrics'].get('overall_score', 0.0)
            
            if best_score - worst_score > 0.3:
                explanation += f"- 模型性能差距较大，最佳模型 {best_model} 和最差模型 {worst_model} 的得分差距为 {best_score - worst_score:.4f}\n"
                explanation += "  建议考虑移除或重新训练性能较差的模型\n\n"
        
        # 检查是否有权重分布不均的情况
        weights = weight_explanation.get('weights', {})
        if weights and len(weights) > 1:
            max_weight = max(weights.values())
            min_weight = min(weights.values())
            
            if max_weight / (min_weight + 1e-10) > 5:
                explanation += "- 权重分布不均，可能导致集成效果不佳\n"
                explanation += "  建议调整学习率或手动平衡权重\n\n"
        
        # 添加总结
        explanation += "总结:\n"
        if best_model:
            explanation += f"- 当前最佳模型是 {best_model}，建议重点关注和优化该模型\n"
        else:
            explanation += "- 暂无明确的最佳模型，建议继续收集性能数据\n"
        
        explanation += "- 定期检查模型性能并更新权重，以保持集成效果\n"
        explanation += "- 考虑引入新的模型或优化现有模型，以提高整体预测准确性\n"
        
        return explanation


if __name__ == "__main__":
    # 测试决策解释器
    print("🔍 测试决策解释器...")
    
    # 创建性能跟踪器
    tracker = PerformanceTracker()
    
    # 跟踪模型性能
    tracker.track_performance("model1", {
        'overall_score': 0.6,
        'accuracy': 0.7,
        'hit_rate': 0.5
    })
    
    tracker.track_performance("model2", {
        'overall_score': 0.8,
        'accuracy': 0.9,
        'hit_rate': 0.7
    })
    
    # 创建权重优化器
    optimizer = WeightOptimizer(tracker)
    optimizer.initialize_weights(["model1", "model2"])
    optimizer.update_weights()
    
    # 创建决策解释器
    explainer = DecisionExplainer(tracker, optimizer)
    
    # 解释模型选择
    model_explanation = explainer.explain_model_selection("model1")
    print(f"模型解释:\n{model_explanation['explanation']}")
    
    # 解释权重调整
    weight_explanation = explainer.explain_weight_adjustment()
    print(f"\n权重调整解释:\n{weight_explanation['explanation']}")
    
    # 生成综合报告
    report = explainer.generate_comprehensive_report()
    print(f"\n综合报告:\n{report['explanation']}")
    
    print("决策解释器测试完成")