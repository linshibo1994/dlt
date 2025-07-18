#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
决策解释器测试
"""

import unittest
import os
import sys
import numpy as np
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from enhanced_deep_learning.decision_explainer import DecisionExplainer
from enhanced_deep_learning.performance_tracker import PerformanceTracker
from enhanced_deep_learning.weight_optimizer import WeightOptimizer


class TestDecisionExplainer(unittest.TestCase):
    """决策解释器测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建性能跟踪器
        self.tracker = PerformanceTracker()
        
        # 跟踪模型性能
        self.tracker.track_performance("model1", {
            'overall_score': 0.6,
            'accuracy': 0.7,
            'hit_rate': 0.5
        })
        
        self.tracker.track_performance("model2", {
            'overall_score': 0.8,
            'accuracy': 0.9,
            'hit_rate': 0.7
        })
        
        # 创建权重优化器
        self.optimizer = WeightOptimizer(self.tracker)
        self.optimizer.initialize_weights(["model1", "model2"])
        self.optimizer.update_weights()
        
        # 创建决策解释器
        self.explainer = DecisionExplainer(self.tracker, self.optimizer)
    
    def test_explain_model_selection(self):
        """测试解释模型选择"""
        # 解释模型选择
        explanation = self.explainer.explain_model_selection("model1")
        
        # 检查解释
        self.assertEqual(explanation['model_id'], "model1")
        self.assertIn('metrics', explanation)
        self.assertIn('comparison', explanation)
        self.assertIn('best_model', explanation)
        self.assertIn('rank', explanation)
        self.assertIn('weight', explanation)
        self.assertIn('explanation', explanation)
        
        # 检查排名
        self.assertEqual(explanation['rank'], 2)  # model1应该排第2
        
        # 检查最佳模型
        self.assertEqual(explanation['best_model'], "model2")
    
    def test_explain_weight_adjustment(self):
        """测试解释权重调整"""
        # 解释权重调整
        explanation = self.explainer.explain_weight_adjustment()
        
        # 检查解释
        self.assertIn('weights', explanation)
        self.assertIn('weight_explanation', explanation)
        self.assertIn('weight_trends', explanation)
        self.assertIn('update_count', explanation)
        self.assertIn('explanation', explanation)
        
        # 检查更新次数
        self.assertEqual(explanation['update_count'], 1)
    
    def test_generate_comprehensive_report(self):
        """测试生成综合报告"""
        # 生成综合报告
        report = self.explainer.generate_comprehensive_report()
        
        # 检查报告
        self.assertIn('timestamp', report)
        self.assertIn('models', report)
        self.assertIn('best_model', report)
        self.assertIn('model_explanations', report)
        self.assertIn('weight_explanation', report)
        self.assertIn('explanation', report)
        
        # 检查模型列表
        self.assertEqual(set(report['models']), {"model1", "model2"})
        
        # 检查最佳模型
        self.assertEqual(report['best_model'], "model2")
        
        # 检查模型解释
        self.assertIn("model1", report['model_explanations'])
        self.assertIn("model2", report['model_explanations'])
    
    def test_explanation_content(self):
        """测试解释内容"""
        # 解释模型选择
        model_explanation = self.explainer.explain_model_selection("model1")
        
        # 检查解释内容
        explanation_text = model_explanation['explanation']
        self.assertIn("模型 model1 的性能分析", explanation_text)
        self.assertIn("性能指标", explanation_text)
        self.assertIn("排名第 2", explanation_text)
        self.assertIn("与最佳模型", explanation_text)
        
        # 解释权重调整
        weight_explanation = self.explainer.explain_weight_adjustment()
        
        # 检查解释内容
        explanation_text = weight_explanation['explanation']
        self.assertIn("权重调整分析", explanation_text)
        self.assertIn("当前权重", explanation_text)
        
        # 生成综合报告
        report = self.explainer.generate_comprehensive_report()
        
        # 检查报告内容
        explanation_text = report['explanation']
        self.assertIn("模型性能与权重综合报告", explanation_text)
        self.assertIn("模型数量", explanation_text)
        self.assertIn("最佳模型", explanation_text)
        self.assertIn("模型性能摘要", explanation_text)
        self.assertIn("权重调整摘要", explanation_text)
        self.assertIn("综合建议", explanation_text)
        self.assertIn("总结", explanation_text)


if __name__ == "__main__":
    unittest.main()