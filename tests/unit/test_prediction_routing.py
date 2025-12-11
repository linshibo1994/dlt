#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试预测方法路由逻辑
验证不同预测方法是否正确路由到对应的处理器
"""

import sys
import os
from types import SimpleNamespace

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../backend/app'))

from core.method_categories import (
    DEEP_LEARNING_METHODS,
    ENSEMBLE_METHODS,
    TRADITIONAL_METHODS,
    MARKOV_METHODS,
    ADVANCED_METHODS,
    REQUIRES_DEEP_LEARNING,
    is_deep_learning_method,
    is_traditional_method,
    get_method_category
)


class TestMethodCategories:
    """测试方法分类常量"""

    def test_deep_learning_methods(self):
        """测试深度学习方法列表"""
        expected = ['lstm', 'transformer', 'gan']
        assert DEEP_LEARNING_METHODS == expected, f"期望 {expected}, 实际 {DEEP_LEARNING_METHODS}"
        print("✅ 深度学习方法列表正确")

    def test_ensemble_methods(self):
        """测试集成学习方法列表"""
        expected = ['stacking', 'adaptive_ensemble', 'ultimate_ensemble']
        assert ENSEMBLE_METHODS == expected, f"期望 {expected}, 实际 {ENSEMBLE_METHODS}"
        print("✅ 集成学习方法列表正确")

    def test_ensemble_not_in_deep_learning(self):
        """验证 ensemble 不在深度学习方法中"""
        assert 'ensemble' not in REQUIRES_DEEP_LEARNING, "ensemble 不应该在 REQUIRES_DEEP_LEARNING 中"
        assert 'ensemble' in ADVANCED_METHODS, "ensemble 应该在 ADVANCED_METHODS 中"
        print("✅ ensemble 正确分类为高级算法")

    def test_requires_deep_learning(self):
        """测试需要深度学习支持的方法"""
        expected = DEEP_LEARNING_METHODS + ENSEMBLE_METHODS
        assert REQUIRES_DEEP_LEARNING == expected
        print("✅ REQUIRES_DEEP_LEARNING 正确包含深度学习和集成方法")

    def test_is_deep_learning_method(self):
        """测试 is_deep_learning_method 函数"""
        # 深度学习方法
        assert is_deep_learning_method('lstm') == True
        assert is_deep_learning_method('transformer') == True
        assert is_deep_learning_method('gan') == True

        # 集成学习方法（需要深度学习支持）
        assert is_deep_learning_method('stacking') == True
        assert is_deep_learning_method('adaptive_ensemble') == True

        # 传统方法
        assert is_deep_learning_method('ensemble') == False
        assert is_deep_learning_method('frequency') == False
        assert is_deep_learning_method('markov') == False

        print("✅ is_deep_learning_method 函数工作正常")

    def test_get_method_category(self):
        """测试 get_method_category 函数"""
        assert get_method_category('frequency') == '传统统计'
        assert get_method_category('markov') == '马尔可夫链'
        assert get_method_category('lstm') == '深度学习'
        assert get_method_category('stacking') == '集成学习'
        assert get_method_category('ensemble') == '高级算法'
        assert get_method_category('super') == '智能增强'
        assert get_method_category('compound') == '复式投注'

        print("✅ get_method_category 函数工作正常")


class TestPredictionRouting:
    """测试预测路由逻辑"""

    def test_ensemble_routing(self):
        """测试 ensemble 方法路由"""
        # ensemble 应该走传统路径
        args = SimpleNamespace(
            method='ensemble',
            periods=100,
            count=1,
            compound=False
        )

        use_deep_learning = args.method in REQUIRES_DEEP_LEARNING
        assert use_deep_learning == False, "ensemble 不应该走深度学习路径"

        # ensemble 应该在高级方法中
        is_advanced = args.method in ADVANCED_METHODS
        assert is_advanced == True, "ensemble 应该在高级方法中"

        print("✅ ensemble 路由测试通过")

    def test_lstm_routing(self):
        """测试 LSTM 方法路由"""
        args = SimpleNamespace(
            method='lstm',
            periods=100,
            count=1,
            compound=False
        )

        use_deep_learning = args.method in REQUIRES_DEEP_LEARNING
        assert use_deep_learning == True, "lstm 应该走深度学习路径"

        is_pure_dl = args.method in DEEP_LEARNING_METHODS
        assert is_pure_dl == True, "lstm 应该在纯深度学习方法中"

        print("✅ lstm 路由测试通过")

    def test_stacking_routing(self):
        """测试 Stacking 方法路由"""
        args = SimpleNamespace(
            method='stacking',
            periods=100,
            count=1,
            compound=False
        )

        use_deep_learning = args.method in REQUIRES_DEEP_LEARNING
        assert use_deep_learning == True, "stacking 应该走深度学习路径（集成学习）"

        is_ensemble = args.method in ENSEMBLE_METHODS
        assert is_ensemble == True, "stacking 应该在集成学习方法中"

        print("✅ stacking 路由测试通过")

    def test_frequency_routing(self):
        """测试频率分析方法路由"""
        args = SimpleNamespace(
            method='frequency',
            periods=100,
            count=1,
            compound=False
        )

        use_deep_learning = args.method in REQUIRES_DEEP_LEARNING
        assert use_deep_learning == False, "frequency 不应该走深度学习路径"

        is_traditional = args.method in TRADITIONAL_METHODS
        assert is_traditional == True, "frequency 应该在传统方法中"

        print("✅ frequency 路由测试通过")


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("开始测试预测方法路由逻辑")
    print("=" * 60)

    test_categories = TestMethodCategories()
    test_routing = TestPredictionRouting()

    # 测试方法分类
    print("\n【测试1：方法分类常量】")
    test_categories.test_deep_learning_methods()
    test_categories.test_ensemble_methods()
    test_categories.test_ensemble_not_in_deep_learning()
    test_categories.test_requires_deep_learning()
    test_categories.test_is_deep_learning_method()
    test_categories.test_get_method_category()

    # 测试路由逻辑
    print("\n【测试2：预测路由逻辑】")
    test_routing.test_ensemble_routing()
    test_routing.test_lstm_routing()
    test_routing.test_stacking_routing()
    test_routing.test_frequency_routing()

    print("\n" + "=" * 60)
    print("✅ 所有测试通过！")
    print("=" * 60)


if __name__ == '__main__':
    run_all_tests()
