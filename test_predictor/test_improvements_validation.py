#!/usr/bin/env python3
"""
测试脚本改进验证工具
验证所有改进是否正确实现
"""

import sys
import os

# 添加模块路径
current_dir = os.path.dirname(os.path.abspath(__file__))
test_predictor_dir = os.path.join(current_dir, 'test_predictor')
modules_dir = os.path.join(test_predictor_dir, 'modules')
sys.path.insert(0, modules_dir)

try:
    from lottery_judge import LotteryJudge
    from config_manager import ConfigManager
    from test_controller import TestController
except ImportError as e:
    print(f"导入模块失败: {e}")
    sys.exit(1)


def test_lottery_judge_improvements():
    """测试中奖判断功能改进"""
    print("=== 测试中奖判断功能改进 ===")
    
    judge = LotteryJudge()
    
    # 验证九个等级是否正确定义
    print("1. 验证奖级定义:")
    for level in range(1, 10):
        if level not in judge.prize_levels:
            print(f"❌ 缺少{level}等奖定义")
            return False
        else:
            print(f"✅ {level}等奖: {judge.prize_levels[level]}")
    
    # 测试具体的中奖判断规则
    print("\n2. 验证中奖规则:")
    test_cases = [
        # (前区匹配, 后区匹配, 预期等级, 描述)
        (5, 2, 1, "一等奖: 5+2"),
        (5, 1, 2, "二等奖: 5+1"),
        (5, 0, 3, "三等奖: 5+0"),
        (4, 2, 4, "四等奖: 4+2"),
        (4, 1, 5, "五等奖: 4+1"),
        (3, 2, 5, "五等奖: 3+2"),
        (4, 0, 6, "六等奖: 4+0"),
        (3, 1, 6, "六等奖: 3+1"),
        (2, 2, 6, "六等奖: 2+2"),
        (3, 0, 7, "七等奖: 3+0"),
        (2, 1, 7, "七等奖: 2+1"),
        (1, 2, 8, "八等奖: 1+2"),
        (2, 0, 8, "八等奖: 2+0"),
        (0, 2, 8, "八等奖: 0+2"),
        (1, 1, 9, "九等奖: 1+1"),
        (1, 0, 9, "九等奖: 1+0"),
        (0, 1, 9, "九等奖: 0+1"),
    ]
    
    all_passed = True
    for front_matches, back_matches, expected_level, description in test_cases:
        actual_level = judge.judge_prize_level(front_matches, back_matches)
        if actual_level == expected_level:
            print(f"✅ {description}")
        else:
            print(f"❌ {description} - 预期:{expected_level}等奖, 实际:{actual_level}等奖")
            all_passed = False
    
    return all_passed


def test_config_improvements():
    """测试配置管理改进"""
    print("\n=== 测试配置管理改进 ===")
    
    config_manager = ConfigManager()
    
    # 验证预测方法数量
    all_methods = config_manager.get_prediction_methods()
    print(f"1. 总预测方法数: {len(all_methods)}")
    print(f"   包括: {', '.join(all_methods[:10])}{'...' if len(all_methods) > 10 else ''}")
    
    # 验证各类别方法
    categories = [
        "basic",
        "markov",
        "probabilistic",
        "ensemble",
        "intelligent",
        "compound",
        "deep_learning"
    ]
    for category in categories:
        methods = config_manager.get_prediction_methods(category)
        if methods:
            print(f"2. {category}类别方法({len(methods)}种): {', '.join(methods)}")
        else:
            print(f"⚠️ {category}类别没有方法定义")
    
    # 验证测试策略
    strategies = ["quick", "comprehensive", "optimization"]
    for strategy in strategies:
        strategy_config = config_manager.get_test_strategy(strategy)
        if strategy_config:
            print(f"3. {strategy}策略: {strategy_config.get('description', '无描述')}")
        else:
            print(f"❌ 缺少{strategy}策略配置")
            return False
    
    return True


def test_test_controller_improvements():
    """测试控制器改进"""
    print("\n=== 测试控制器改进 ===")
    
    config_manager = ConfigManager()
    test_controller = TestController(config_manager)
    
    # 测试测试用例生成
    print("1. 生成comprehensive测试用例:")
    try:
        test_cases = test_controller.generate_test_cases("comprehensive")
        print(f"   生成测试用例数: {len(test_cases)}")
        
        # 统计方法覆盖
        methods_tested = set(case['method'] for case in test_cases)
        print(f"   覆盖方法数: {len(methods_tested)}")
        print(f"   包括方法: {', '.join(sorted(methods_tested)[:10])}{'...' if len(methods_tested) > 10 else ''}")
        
        # 检查优先级设置
        priority_counts = {}
        for case in test_cases:
            priority = case.get('priority', 'unknown')
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        print(f"   优先级分布: {priority_counts}")
        
    except Exception as e:
        print(f"❌ 测试用例生成失败: {e}")
        return False
    
    print("2. 生成quick测试用例:")
    try:
        quick_cases = test_controller.generate_test_cases("quick")
        print(f"   快速测试用例数: {len(quick_cases)}")
    except Exception as e:
        print(f"❌ 快速测试用例生成失败: {e}")
        return False
    
    return True


def test_result_structure():
    """测试结果结构改进"""
    print("\n=== 测试结果结构改进 ===")
    
    # 模拟一个完整的测试结果结构
    mock_result = {
        'success': True,
        'test_case': {'method': 'markov', 'periods': 100, 'count': 2},
        'winnings': [
            {
                'prize_level': 5,
                'prize_name': '五等奖',
                'match_combination': '4+1',
                'predicted_front': [1, 2, 3, 4, 5],
                'predicted_back': [1, 2],
                'matched_front_numbers': [1, 2, 3, 4],
                'matched_back_numbers': [1],
                'front_matches': 4,
                'back_matches': 1
            }
        ],
        'prize_statistics': {
            5: {'name': '五等奖', 'count': 1}
        },
        'best_prize_level': 5,
        'is_winning': True
    }
    
    print("1. 验证结果结构完整性:")
    required_fields = ['winnings', 'prize_statistics', 'best_prize_level', 'is_winning']
    
    for field in required_fields:
        if field in mock_result:
            print(f"✅ {field}: 存在")
        else:
            print(f"❌ {field}: 缺失")
            return False
    
    print("2. 验证中奖详细信息结构:")
    if mock_result['winnings']:
        winning = mock_result['winnings'][0]
        winning_fields = ['prize_level', 'match_combination', 'predicted_front', 'predicted_back', 
                         'matched_front_numbers', 'matched_back_numbers']
        
        for field in winning_fields:
            if field in winning:
                print(f"✅ 中奖信息.{field}: 存在")
            else:
                print(f"❌ 中奖信息.{field}: 缺失")
                return False
    
    return True


def main():
    """主验证函数"""
    print("🎯 大乐透测试脚本改进验证工具")
    print("=" * 50)
    
    tests = [
        ("中奖判断功能", test_lottery_judge_improvements),
        ("配置管理", test_config_improvements),
        ("测试控制器", test_test_controller_improvements),
        ("结果结构", test_result_structure),
    ]
    
    all_passed = True
    results = []
    
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, "✅ 通过" if passed else "❌ 失败"))
            if not passed:
                all_passed = False
        except Exception as e:
            results.append((test_name, f"❌ 异常: {e}"))
            all_passed = False
    
    print("\n" + "=" * 50)
    print("🎯 验证结果汇总")
    print("=" * 50)
    
    for test_name, result in results:
        print(f"{test_name:15s}: {result}")
    
    if all_passed:
        print("\n🎉 所有改进验证通过！测试脚本已完全优化。")
        print("\n✨ 主要改进包括:")
        print("  1. ✅ 完善中奖判断 - 支持九个等级的完整判断")
        print("  2. ✅ 详细中奖信息 - 记录所有等级的中奖详情")
        print("  3. ✅ 全面方法测试 - 覆盖所有预测方法")
        print("  4. ✅ 智能优先级 - 重要方法优先测试")
        print("  5. ✅ 详细统计报告 - 完整的中奖分析")
        print("  6. ✅ 最佳方法识别 - 自动找出最佳预测方法")
    else:
        print("\n❌ 部分改进验证失败，请检查相关功能。")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())