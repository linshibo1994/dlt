#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试套件
运行所有单元测试
"""

import unittest
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def run_tests():
    """运行所有测试"""
    # 获取测试目录
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 发现所有测试
    test_suite = unittest.defaultTestLoader.discover(test_dir, pattern="test_*.py")
    
    # 创建测试运行器
    test_runner = unittest.TextTestRunner(verbosity=2)
    
    # 运行测试
    result = test_runner.run(test_suite)
    
    # 返回结果
    return result.wasSuccessful()


if __name__ == "__main__":
    print("🧪 运行所有测试...")
    
    # 运行测试
    success = run_tests()
    
    # 输出结果
    if success:
        print("✅ 所有测试通过")
        sys.exit(0)
    else:
        print("❌ 测试失败")
        sys.exit(1)