#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试学习功能
"""

import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core_modules import data_manager, logger_manager
from adaptive_learning_modules import enhanced_adaptive_predictor

def test_learning():
    """测试学习功能"""
    print("🧪 测试学习功能...")
    
    # 获取数据状态
    df = data_manager.get_data()
    if df is None:
        print("❌ 数据未加载")
        return
    
    print(f"📊 数据状态: {len(df)} 期")
    print(f"📈 数据范围: {df.iloc[-1]['issue']} 到 {df.iloc[0]['issue']}")
    
    # 找到一个合适的起始期号
    start_issue = df.iloc[-100]['issue']  # 从倒数第100期开始
    print(f"🎯 选择起始期号: {start_issue}")
    
    # 进行小规模测试
    print("🔄 开始小规模学习测试...")
    try:
        results = enhanced_adaptive_predictor.enhanced_adaptive_learning(
            start_period=int(start_issue),
            test_periods=20
        )
        
        if results:
            print("✅ 学习测试成功!")
            print(f"  测试期数: {results.get('total_periods', 0)}")
            print(f"  中奖率: {results.get('win_rate', 0):.3f}")
            print(f"  平均得分: {results.get('average_score', 0):.2f}")
        else:
            print("❌ 学习测试失败")
    
    except Exception as e:
        print(f"❌ 学习测试出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_learning()
