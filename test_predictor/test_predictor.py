#!/usr/bin/env python3
"""
大乐透预测方法测试脚本
主程序入口，集成所有模块功能
"""

import sys
import os
import argparse
import signal
import time
from typing import Optional

# 添加模块路径
current_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.join(current_dir, 'modules')
sys.path.insert(0, modules_dir)

try:
    from config_manager import ConfigManager
    from test_controller import TestController
    from result_recorder import ResultRecorder
    from lottery_data import LotteryData
    from lottery_judge import LotteryJudge
except ImportError as e:
    print(f"导入模块失败: {e}")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"脚本目录: {current_dir}")
    print(f"模块目录: {modules_dir}")
    sys.exit(1)


class PredictorTester:
    """预测器测试主程序"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.test_controller = TestController(self.config_manager)
        self.result_recorder = ResultRecorder()
        self.lottery_data = LotteryData()
        self.lottery_judge = LotteryJudge()
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # 设置回调函数
        self.test_controller.set_callbacks(
            progress_callback=self._progress_callback,
            result_callback=self._result_callback
        )

    def _signal_handler(self, signum, frame):
        """信号处理器"""
        print("\n\n收到中断信号，正在停止测试...")
        self.test_controller.stop()

    def _progress_callback(self, message: str, progress: Optional[float] = None):
        """进度回调"""
        timestamp = time.strftime("%H:%M:%S")
        if progress is not None:
            print(f"[{timestamp}] [{progress:6.1f}%] {message}")
        else:
            print(f"[{timestamp}] [INFO] {message}")

    def _result_callback(self, result):
        """结果回调 - 显示所有等级的中奖信息"""
        if result.get('is_winning'):
            method = result['test_case']['method']
            periods = result['test_case']['periods']
            count = result['test_case']['count']
            
            # 显示所有中奖信息，不只是最高奖级
            winnings = result.get('winnings', [])
            if winnings:
                for win_info in winnings:
                    prize_level = win_info.get('prize_level', 0)
                    prize_name = self.lottery_judge.prize_levels.get(prize_level, '未知奖级')
                    match_combo = win_info.get('match_combination', '')
                    print(f"🎯 中奖: {method}(p={periods},c={count}) -> {prize_name} [{match_combo}]")
                    
                    # 显示中奖号码详情
                    pred_front = win_info.get('predicted_front', [])
                    pred_back = win_info.get('predicted_back', [])
                    matched_front = win_info.get('matched_front_numbers', [])
                    matched_back = win_info.get('matched_back_numbers', [])
                    
                    print(f"   预测: 前区{pred_front} 后区{pred_back}")
                    print(f"   中奖: 前区{matched_front} 后区{matched_back}")
                    
                    # 如果是重大奖项，特别提醒
                    if self.lottery_judge.is_major_prize(prize_level):
                        print(f"🎉 重大奖项：{prize_name}！")
                        print(f"🎯 中奖方法详情: {method} 分析{periods}期数据 生成{count}注预测")
            else:
                # 向后兼容，处理旧格式
                prize_level = result.get('best_prize_level', 0)
                prize_name = self.lottery_judge.prize_levels.get(prize_level, '')
                print(f"🎯 中奖: {method}(p={periods},c={count}) -> {prize_name}")
                
                if self.lottery_judge.is_major_prize(prize_level):
                    print(f"🎉 重大奖项：{prize_name}！")
                    print(f"🎯 中奖方法详情: {method} 分析{periods}期数据 生成{count}注预测")
    def check_system(self) -> bool:
        """检查系统状态"""
        print("=== 系统检查 ===")
        
        # 检查配置
        print("1. 检查配置...")
        errors = self.config_manager.validate_config()
        if errors:
            print("❌ 配置错误:")
            for error in errors:
                print(f"   - {error}")
            return False
        print("✅ 配置正常")
        
        # 检查数据文件
        print("2. 检查数据文件...")
        try:
            stats = self.lottery_data.get_statistics()
            print(f"✅ 数据文件正常 (共{stats['total_count']}期)")
            print(f"   最新期号: {stats['latest_issue']} ({stats['latest_date']})")
        except Exception as e:
            print(f"❌ 数据文件错误: {e}")
            return False
        
        # 检查预测系统连接
        print("3. 检查预测系统...")
        if self.test_controller.predictor_caller.test_connection():
            print("✅ 预测系统连接正常")
        else:
            print("❌ 无法连接预测系统")
            return False
        
        # 检查中奖判断
        print("4. 检查中奖判断...")
        try:
            test_result = self.lottery_judge.check_winning(
                "01,02,03,04,05", "01,02",
                "01,02,03,04,05", "01,02"
            )
            if test_result.get('prize_level') == 1:
                print("✅ 中奖判断正常")
            else:
                print("❌ 中奖判断异常")
                return False
        except Exception as e:
            print(f"❌ 中奖判断错误: {e}")
            return False
        
        print("\n✅ 系统检查通过，可以开始测试\n")
        return True

    def run_quick_test(self):
        """运行快速测试"""
        print("=== 快速测试模式 ===")
        
        if not self.check_system():
            return
        
        try:
            summary = self.test_controller.run_tests("quick", max_parallel=2)
            self._save_results(summary, "quick")
            self._print_summary(summary)
            
        except KeyboardInterrupt:
            print("\n测试被用户中断")
        except Exception as e:
            print(f"测试异常: {e}")

    def run_comprehensive_test(self):
        """运行全面测试"""
        print("=== 全面测试模式 ===")
        
        if not self.check_system():
            return
        
        try:
            summary = self.test_controller.run_tests("comprehensive")
            self._save_results(summary, "comprehensive")
            self._print_summary(summary)
            
        except KeyboardInterrupt:
            print("\n测试被用户中断")
        except Exception as e:
            print(f"测试异常: {e}")

    def run_optimization_test(self):
        """运行优化测试"""
        print("=== 优化测试模式 ===")
        
        if not self.check_system():
            return
        
        try:
            summary = self.test_controller.run_tests("optimization")
            self._save_results(summary, "optimization")
            self._print_summary(summary)
            
        except KeyboardInterrupt:
            print("\n测试被用户中断")
        except Exception as e:
            print(f"测试异常: {e}")

    def run_custom_test(self, method: str, periods_start: int, periods_end: int, 
                       periods_step: int, count: int):
        """运行自定义测试"""
        print(f"=== 自定义测试模式: {method} ===")
        
        if not self.check_system():
            return
        
        if not self.test_controller.predictor_caller.validate_method(method):
            available = ', '.join(sorted(self.test_controller.predictor_caller.available_methods))
            print(f"错误: 不支持的预测方法: {method}")
            print(f"请在以下方法中选择: {available}")
            return

        # 生成自定义测试用例
        test_cases = []
        for periods in range(periods_start, periods_end + 1, periods_step):
            test_cases.append({
                'method': method,
                'periods': periods,
                'count': count,
                'strategy': 'custom'
            })
        
        print(f"生成 {len(test_cases)} 个测试用例")
        
        try:
            # 手动运行测试
            start_time = time.time()
            
            for i, test_case in enumerate(test_cases):
                progress = (i / len(test_cases)) * 100
                self._progress_callback(
                    f"执行测试 {i+1}/{len(test_cases)}: {method}(p={test_case['periods']})",
                    progress
                )
                
                result = self.test_controller.execute_single_test(test_case)
                self.test_controller._report_result(result)
                
                # 检查是否中重大奖项
                if result.get('is_major_prize'):
                    self._progress_callback(
                        f"🎉 中得{self.lottery_judge.prize_levels[result['best_prize_level']]}！停止测试"
                    )
                    break
            
            # 生成汇总
            execution_time = time.time() - start_time
            summary = self.test_controller._generate_summary(execution_time, 'custom')
            
            self._save_results(summary, "custom")
            self._print_summary(summary)
            
        except KeyboardInterrupt:
            print("\n测试被用户中断")
        except Exception as e:
            print(f"测试异常: {e}")

    def _save_results(self, summary, mode):
        """保存测试结果"""
        print("\n=== 保存结果 ===")
        
        results = self.test_controller.get_results()
        
        # 保存JSON结果
        json_file = self.result_recorder.save_test_results(
            results['test_results'], 
            f"{mode}_results_{self.result_recorder.timestamp}.json"
        )
        
        # 保存汇总报告
        summary_file = self.result_recorder.save_summary_report(
            summary,
            f"{mode}_summary_{self.result_recorder.timestamp}.json"
        )
        
        # 导出CSV
        csv_file = self.result_recorder.export_to_csv(
            results['test_results'],
            f"{mode}_results_{self.result_recorder.timestamp}.csv"
        )
        
        # 生成HTML报告
        html_file = self.result_recorder.generate_html_report(
            summary,
            results['test_results'],
            f"{mode}_report_{self.result_recorder.timestamp}.html"
        )
        
        # 如果有中奖结果，生成中奖者报告
        if results['winning_results']:
            winner_file = self.result_recorder.generate_winner_report(
                results['test_results'],
                f"{mode}_winners_{self.result_recorder.timestamp}.txt"
            )
        
        print(f"结果已保存到 {self.result_recorder.output_dir} 目录")

    def _print_summary(self, summary):
        """打印详细的测试摘要，重点显示中奖方法"""
        print("\n" + "="*60)
        print("🎯 测试完成摘要")
        print("="*60)
        
        print(f"测试策略: {summary.get('strategy', 'unknown')}")
        print(f"执行时间: {summary.get('execution_time', 0):.1f} 秒")
        print(f"总测试数: {summary.get('total_tests', 0)}")
        print(f"总预测数: {summary.get('total_predictions', 0)}")
        print(f"中奖测试: {summary.get('winning_tests', 0)}")
        print(f"中奖预测: {summary.get('winning_predictions', 0)}")
        print(f"重大奖项: {summary.get('major_prizes', 0)}")
        print(f"测试中奖率: {summary.get('test_winning_rate', 0)*100:.2f}%")
        print(f"预测中奖率: {summary.get('prediction_winning_rate', 0)*100:.2f}%")
        
        # 显示详细的奖级统计
        prize_stats = summary.get('prize_statistics', {})
        if prize_stats:
            print(f"\n🏆 中奖等级详细统计:")
            for level in sorted(prize_stats.keys()):
                stat = prize_stats[level]
                methods = stat.get('winning_methods', [])
                method_list = ', '.join(methods[:3])  # 只显示前3个方法
                if len(methods) > 3:
                    method_list += f"等{len(methods)}种方法"
                
                print(f"  {level}等奖 ({stat['name']}): {stat['count']} 次")
                print(f"    中奖方法: {method_list}")
        
        # 显示最佳中奖方法详情
        best_method = summary.get('best_method')
        if best_method:
            print(f"\n🎯 最佳预测方法: {best_method}")
            method_stats = summary.get('method_statistics', {}).get(best_method, {})
            
            print(f"  综合评分: {summary.get('best_method_score', 0)} 分")
            print(f"  测试次数: {method_stats.get('total_tests', 0)}")
            print(f"  中奖率: {method_stats.get('test_winning_rate', 0)*100:.1f}%")
            
            # 显示该方法的奖级分布
            prize_breakdown = method_stats.get('prize_breakdown', {})
            if prize_breakdown:
                print(f"  中奖分布:")
                for level in sorted(prize_breakdown.keys()):
                    count = prize_breakdown[level]
                    prize_name = self.lottery_judge.prize_levels.get(level, f"{level}等奖")
                    print(f"    {prize_name}: {count} 次")
            
            # 显示首次中一等奖的详情
            first_prize = method_stats.get('first_prize_test')
            if first_prize:
                print(f"  🎉 首次中一等奖详情:")
                print(f"    分析期数: {first_prize['periods']} 期")
                print(f"    预测注数: {first_prize['count']} 注")
                winning_details = first_prize.get('winning_details', {})
                if winning_details:
                    print(f"    中奖号码: 前区{winning_details.get('predicted_front', [])} 后区{winning_details.get('predicted_back', [])}")
        
        # 显示方法表现排行
        method_stats = summary.get('method_statistics', {})
        if method_stats:
            print(f"\n🔍 方法表现排行（按综合评分）:")
            
            # 按综合评分排序
            scored_methods = []
            for method, stats in method_stats.items():
                score = 0
                for level, count in stats.get('prize_breakdown', {}).items():
                    if level == 1:
                        score += count * 1000
                    elif level == 2:
                        score += count * 500
                    elif level == 3:
                        score += count * 200
                    else:
                        score += count * (10 - level) * 10
                scored_methods.append((method, stats, score))
            
            sorted_methods = sorted(scored_methods, key=lambda x: x[2], reverse=True)
            
            for i, (method, stats, score) in enumerate(sorted_methods[:10]):
                best_prize = stats.get('best_prize_level', 0)
                win_rate = stats.get('test_winning_rate', 0) * 100
                major_count = stats.get('major_prize_count', 0)
                
                if best_prize > 0:
                    prize_text = f"最高{best_prize}等奖"
                    if major_count > 0:
                        prize_text += f"(重大奖项{major_count}次)"
                else:
                    prize_text = "未中奖"
                
                print(f"  {i+1:2d}. {method:15s}: {prize_text:20s} 评分:{score:4d} 中奖率:{win_rate:5.1f}%")
        
        if summary.get('stopped_by_major_prize'):
            print(f"\n🎉 恭喜！测试因中得重大奖项而自动停止！")
            print(f"🎯 找到中一等奖的方法了！请查看上面的详细信息。")
    def show_config(self):
        """显示当前配置"""
        print("=== 当前配置 ===")
        
        print(f"超时时间: {self.config_manager.get('test_settings.timeout_seconds')} 秒")
        print(f"重试次数: {self.config_manager.get('test_settings.max_retries')}")
        print(f"并行数量: {self.config_manager.get('test_settings.parallel_workers')}")
        print(f"重大奖项停止: {self.config_manager.get('test_settings.stop_on_major_prize')}")
        
        print(f"\n可用预测方法:")
        for category, methods in self.config_manager.get('prediction_methods', {}).items():
            print(f"  {category}: {', '.join(methods)}")

    def create_config(self):
        """创建默认配置文件"""
        print("=== 创建配置文件 ===")
        self.config_manager.create_default_configs()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="大乐透预测方法测试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  %(prog)s check              # 检查系统状态
  %(prog)s quick              # 快速测试
  %(prog)s comprehensive      # 全面测试
  %(prog)s optimization       # 优化测试
  %(prog)s custom markov 10 2000 50 1  # 自定义测试
  %(prog)s config             # 显示配置
  %(prog)s create-config      # 创建配置文件
        """
    )
    
    parser.add_argument('command', 
                       choices=['check', 'quick', 'comprehensive', 'optimization', 
                               'custom', 'config', 'create-config'],
                       help='执行的命令')
    
    # 自定义测试参数
    parser.add_argument('method', nargs='?', help='预测方法 (用于custom命令)')
    parser.add_argument('periods_start', type=int, nargs='?', help='起始期数 (用于custom命令)')
    parser.add_argument('periods_end', type=int, nargs='?', help='结束期数 (用于custom命令)')
    parser.add_argument('periods_step', type=int, nargs='?', default=50, help='期数步长 (用于custom命令)')
    parser.add_argument('count', type=int, nargs='?', default=1, help='生成注数 (用于custom命令)')
    
    args = parser.parse_args()
    
    # 创建测试器实例
    tester = PredictorTester()
    
    try:
        if args.command == 'check':
            tester.check_system()
        
        elif args.command == 'quick':
            tester.run_quick_test()
        
        elif args.command == 'comprehensive':
            tester.run_comprehensive_test()
        
        elif args.command == 'optimization':
            tester.run_optimization_test()
        
        elif args.command == 'custom':
            if not all([args.method, args.periods_start, args.periods_end]):
                print("错误: custom命令需要指定 method periods_start periods_end 参数")
                parser.print_help()
                sys.exit(1)
            
            tester.run_custom_test(
                args.method, 
                args.periods_start, 
                args.periods_end,
                args.periods_step,
                args.count
            )
        
        elif args.command == 'config':
            tester.show_config()
        
        elif args.command == 'create-config':
            tester.create_config()
    
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"程序异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()