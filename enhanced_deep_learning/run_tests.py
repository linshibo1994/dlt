#!/usr/bin/env python3
"""
运行所有测试
"""

import unittest
import sys
import os
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def run_all_tests():
    """运行所有测试"""
    print("🧪 开始运行深度学习模块测试")
    print("=" * 60)
    print(f"📅 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 发现并运行测试
    test_dir = os.path.join(os.path.dirname(__file__), 'tests')
    
    if not os.path.exists(test_dir):
        print("❌ 测试目录不存在")
        return False
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern='test_*.py')
    
    # 运行测试
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        descriptions=True,
        failfast=False
    )
    
    print(f"🔍 在目录 {test_dir} 中发现测试")
    print("-" * 60)
    
    result = runner.run(suite)
    
    # 输出测试结果摘要
    print("\n" + "=" * 60)
    print("📊 测试结果摘要")
    print("=" * 60)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
    success = total_tests - failures - errors - skipped
    
    print(f"✅ 成功: {success}")
    print(f"❌ 失败: {failures}")
    print(f"💥 错误: {errors}")
    print(f"⏭️  跳过: {skipped}")
    print(f"📈 总计: {total_tests}")
    
    if failures > 0 or errors > 0:
        print(f"📉 成功率: {(success/total_tests)*100:.1f}%")
        
        if result.failures:
            print("\n❌ 失败的测试:")
            for test, traceback in result.failures:
                print(f"  - {test}")
        
        if result.errors:
            print("\n💥 错误的测试:")
            for test, traceback in result.errors:
                print(f"  - {test}")
        
        return False
    else:
        print("🎉 所有测试通过！")
        return True

def run_specific_test(test_name):
    """运行特定测试"""
    print(f"🧪 运行特定测试: {test_name}")
    print("=" * 60)
    
    test_dir = os.path.join(os.path.dirname(__file__), 'tests')
    test_file = os.path.join(test_dir, f'test_{test_name}.py')
    
    if not os.path.exists(test_file):
        print(f"❌ 测试文件不存在: {test_file}")
        return False
    
    # 加载特定测试
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(f'tests.test_{test_name}')
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return len(result.failures) == 0 and len(result.errors) == 0

def check_dependencies():
    """检查依赖项"""
    print("🔍 检查依赖项...")
    
    dependencies = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'scikit-learn': 'sklearn',
        'tensorflow': 'tensorflow',
        'psutil': 'psutil'
    }
    
    missing_deps = []
    
    for name, import_name in dependencies.items():
        try:
            __import__(import_name)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} (缺失)")
            missing_deps.append(name)
    
    if missing_deps:
        print(f"\n⚠️  缺失依赖项: {', '.join(missing_deps)}")
        print("请安装缺失的依赖项后再运行测试")
        return False
    
    print("✅ 所有依赖项已安装")
    return True

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='运行深度学习模块测试')
    parser.add_argument('--test', '-t', help='运行特定测试 (例如: lstm_predictor)')
    parser.add_argument('--check-deps', '-c', action='store_true', help='检查依赖项')
    parser.add_argument('--no-deps-check', action='store_true', help='跳过依赖项检查')
    
    args = parser.parse_args()
    
    # 检查依赖项
    if args.check_deps:
        check_dependencies()
        return
    
    if not args.no_deps_check:
        if not check_dependencies():
            return
    
    # 运行测试
    if args.test:
        success = run_specific_test(args.test)
    else:
        success = run_all_tests()
    
    # 退出码
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
