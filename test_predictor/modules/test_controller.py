#!/usr/bin/env python3
"""
测试控制器模块
负责控制整个测试流程
"""

import time
import random
import threading
from datetime import datetime
from typing import List, Dict, Optional, Callable, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from config_manager import ConfigManager
from predictor_caller import PredictorCaller
from lottery_data import LotteryData
from lottery_judge import LotteryJudge


class TestController:
    """测试控制器"""
    
    def __init__(self, config_manager: ConfigManager = None):
        self.config_manager = config_manager or ConfigManager()
        self.predictor_caller = PredictorCaller()
        self.lottery_data = LotteryData()
        self.lottery_judge = LotteryJudge()
        
        # 运行状态
        self.is_running = False
        self.is_paused = False
        self.should_stop = False
        self.current_test_count = 0
        self.total_tests = 0
        
        # 结果统计
        self.test_results = []
        self.winning_results = []
        self.major_prize_results = []
        
        # 线程锁
        self.lock = threading.Lock()
        
        # 回调函数
        self.progress_callback: Optional[Callable] = None
        self.result_callback: Optional[Callable] = None

    def set_callbacks(self, progress_callback: Callable = None, 
                     result_callback: Callable = None):
        """设置回调函数"""
        self.progress_callback = progress_callback
        self.result_callback = result_callback

    def _update_progress(self, message: str, progress: float = None):
        """更新进度"""
        if self.progress_callback:
            self.progress_callback(message, progress)
        else:
            if progress is not None:
                print(f"[{progress:.1f}%] {message}")
            else:
                print(f"[INFO] {message}")

    def _report_result(self, result: Dict):
        """报告结果"""
        with self.lock:
            self.test_results.append(result)
            
            if result.get('is_winning'):
                self.winning_results.append(result)
                
                prize_level = result.get('prize_level', 0)
                if self.lottery_judge.is_major_prize(prize_level):
                    self.major_prize_results.append(result)
        
        if self.result_callback:
            self.result_callback(result)

    def generate_test_cases(self, strategy: str = "comprehensive") -> List[Dict]:
        """生成测试用例，确保全面覆盖所有预测方法"""
        strategy_config = self.config_manager.get_test_strategy(strategy)
        if not strategy_config:
            raise ValueError(f"未找到测试策略: {strategy}")
        
        test_cases = []
        
        # 获取测试方法
        methods = strategy_config.get('methods', [])
        if methods == "all":
            methods = self.config_manager.get_prediction_methods()
        
        # 记录实际将要测试的方法
        print(f"将要测试的预测方法({len(methods)}种): {', '.join(methods[:10])}{'...' if len(methods) > 10 else ''}")
        
        # 生成测试用例
        if strategy == "quick":
            # 快速测试模式 - 重点测试几种方法
            periods_list = strategy_config.get('periods_list', [100, 500])
            count_list = strategy_config.get('count_list', [1, 2])
            max_tests = strategy_config.get('max_tests_per_method', 4)
            
            for method in methods:
                test_count = 0
                for periods in periods_list:
                    for count in count_list:
                        if test_count < max_tests:
                            test_cases.append({
                                'method': method,
                                'periods': periods,
                                'count': count,
                                'strategy': strategy,
                                'priority': 'normal'
                            })
                            test_count += 1
        
        elif strategy == "comprehensive":
            # 全面测试模式 - 确保测试所有方法
            periods_range = strategy_config.get('periods_range', [50, 1000])
            count_range = strategy_config.get('count_range', [1, 5])
            progressive = strategy_config.get('progressive_testing', True)
            priority_methods = strategy_config.get('priority_methods', [])
            
            # 分优先级测试
            high_priority_methods = [m for m in methods if m in priority_methods]
            normal_priority_methods = [m for m in methods if m not in priority_methods]
            
            print(f"高优先级方法({len(high_priority_methods)}种): {', '.join(high_priority_methods)}")
            print(f"普通优先级方法({len(normal_priority_methods)}种): {', '.join(normal_priority_methods[:5])}{'...' if len(normal_priority_methods) > 5 else ''}")
            
            # 先测试高优先级方法
            for method in high_priority_methods:
                if progressive:
                    # 累进式测试 - 对重要方法进行详细测试
                    periods_start, periods_end = periods_range
                    step = self.config_manager.get('parameter_ranges.periods.progressive_step', 50)
                    periods_list = list(range(periods_start, min(periods_end + 1, 1001), step))
                    
                    for periods in periods_list:
                        for count in range(count_range[0], min(count_range[1] + 1, 4)):  # 限制注数避免过多测试
                            test_cases.append({
                                'method': method,
                                'periods': periods,
                                'count': count,
                                'strategy': strategy,
                                'priority': 'high'
                            })
                else:
                    # 基础测试点
                    basic_periods = self.config_manager.get('parameter_ranges.periods.comprehensive_test', [50, 100, 200, 500, 1000])
                    basic_counts = self.config_manager.get('parameter_ranges.count.basic_test', [1, 2, 3])
                    
                    for periods in basic_periods:
                        for count in basic_counts:
                            test_cases.append({
                                'method': method,
                                'periods': periods,
                                'count': count,
                                'strategy': strategy,
                                'priority': 'high'
                            })
            
            # 再测试普通优先级方法 - 使用较少的测试点
            for method in normal_priority_methods:
                basic_periods = self.config_manager.get('parameter_ranges.periods.basic_test', [100, 500, 1000])
                basic_counts = [1, 2]  # 减少注数测试
                
                for periods in basic_periods:
                    for count in basic_counts:
                        test_cases.append({
                            'method': method,
                            'periods': periods,
                            'count': count,
                            'strategy': strategy,
                            'priority': 'normal'
                        })
        
        elif strategy == "optimization":
            # 优化测试模式 - 随机但重点关注有效方法
            max_tests = strategy_config.get('max_tests', 1000)
            periods_range = strategy_config.get('periods_range', [10, 2000])
            count_range = strategy_config.get('count_range', [1, 10])
            
            # 对重要方法给予更高的测试概率
            priority_methods = strategy_config.get('methods', methods)
            method_weights = []
            for method in methods:
                if method in priority_methods:
                    method_weights.extend([method] * 3)  # 高优先级方法权重x3
                else:
                    method_weights.append(method)
            
            test_count = 0
            while test_count < max_tests:
                method = random.choice(method_weights)
                periods = random.randint(periods_range[0], periods_range[1])
                count = random.randint(count_range[0], count_range[1])
                
                test_cases.append({
                    'method': method,
                    'periods': periods,
                    'count': count,
                    'strategy': strategy,
                    'priority': 'optimization'
                })
                test_count += 1
        
        # 按优先级和方法排序，确保重要方法优先测试
        priority_order = {'high': 0, 'normal': 1, 'optimization': 2}
        test_cases.sort(key=lambda x: (priority_order.get(x.get('priority', 'normal'), 1), x['method']))
        
        print(f"总共生成 {len(test_cases)} 个测试用例")
        method_counts = {}
        for case in test_cases:
            method = case['method']
            method_counts[method] = method_counts.get(method, 0) + 1
        
        print(f"各方法测试数量分布:")
        for method, count in sorted(method_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {method}: {count} 个测试")
        if len(method_counts) > 10:
            print(f"  ... 其他 {len(method_counts) - 10} 种方法")
        
        return test_cases

    def execute_single_test(self, test_case: Dict) -> Dict:
        """执行单个测试，详细记录所有中奖信息"""
        start_time = time.time()
        
        # 调用预测
        prediction_result = self.predictor_caller.predict(
            method=test_case['method'],
            periods=test_case['periods'],
            count=test_case['count']
        )
        
        # 获取最新开奖结果
        latest_result = self.lottery_data.get_latest_result()
        if not latest_result:
            return {
                'success': False,
                'error': '无法获取最新开奖结果',
                'test_case': test_case,
                'execution_time': time.time() - start_time
            }
        
        # 处理预测结果
        test_result = {
            'success': prediction_result['success'],
            'test_case': test_case,
            'prediction_result': prediction_result,
            'latest_draw': latest_result,
            'execution_time': time.time() - start_time,
            'timestamp': datetime.now().isoformat(),
            'predictions_analysis': [],
            'winnings': [],  # 新增：详细的中奖信息列表
            'prize_statistics': {}  # 新增：各奖级统计
        }
        
        if not prediction_result['success']:
            test_result['error'] = prediction_result.get('error', '预测失败')
            return test_result
        
        # 分析每个预测号码的中奖情况
        predictions = prediction_result.get('predictions', [])
        winning_count = 0
        best_prize_level = 0
        prize_count = {}  # 各奖级计数
        
        for i, prediction in enumerate(predictions):
            winning_result = self.lottery_judge.check_winning(
                prediction['front_balls'],
                prediction['back_balls'],
                latest_result['front_balls'],
                latest_result['back_balls']
            )
            
            prediction_analysis = {
                'prediction_index': i + 1,
                'prediction': prediction,
                'winning_result': winning_result
            }
            test_result['predictions_analysis'].append(prediction_analysis)
            
            if winning_result.get('is_winning'):
                winning_count += 1
                prize_level = winning_result.get('prize_level', 0)
                
                # 记录详细的中奖信息
                winning_info = {
                    'prediction_index': i + 1,
                    'prize_level': prize_level,
                    'prize_name': winning_result.get('prize_name', ''),
                    'match_combination': winning_result.get('match_combination', ''),
                    'predicted_front': winning_result.get('predicted_front', []),
                    'predicted_back': winning_result.get('predicted_back', []),
                    'matched_front_numbers': winning_result.get('matched_front_numbers', []),
                    'matched_back_numbers': winning_result.get('matched_back_numbers', []),
                    'front_matches': winning_result.get('front_matches', 0),
                    'back_matches': winning_result.get('back_matches', 0)
                }
                test_result['winnings'].append(winning_info)
                
                # 统计各奖级
                if prize_level > 0:
                    best_prize_level = max(best_prize_level, prize_level)
                    prize_count[prize_level] = prize_count.get(prize_level, 0) + 1
        
        # 生成奖级统计
        for level, count in prize_count.items():
            test_result['prize_statistics'][level] = {
                'name': self.lottery_judge.prize_levels.get(level, ''),
                'count': count
            }
        
        # 汇总结果
        test_result.update({
            'total_predictions': len(predictions),
            'winning_predictions': winning_count,
            'winning_rate': winning_count / len(predictions) if predictions else 0,
            'best_prize_level': best_prize_level,
            'is_winning': winning_count > 0,
            'is_major_prize': self.lottery_judge.is_major_prize(best_prize_level),
            'prize_level': best_prize_level,
            'total_prize_levels': len(prize_count),  # 中了几种不同奖级
        })
        
        return test_result

    def should_stop_testing(self) -> bool:
        """检查是否应该停止测试"""
        if self.should_stop:
            return True
        
        stop_on_major = self.config_manager.get('test_settings.stop_on_major_prize', True)
        if stop_on_major and self.major_prize_results:
            return True
        
        return False

    def run_tests(self, strategy: str = "comprehensive", 
                 max_parallel: int = None) -> Dict:
        """运行测试"""
        if self.is_running:
            raise RuntimeError("测试已在运行中")
        
        self.is_running = True
        self.should_stop = False
        self.current_test_count = 0
        
        # 清除之前的结果
        self.test_results.clear()
        self.winning_results.clear()
        self.major_prize_results.clear()
        
        start_time = time.time()
        
        try:
            # 生成测试用例
            self._update_progress("生成测试用例...")
            test_cases = self.generate_test_cases(strategy)
            self.total_tests = len(test_cases)
            
            self._update_progress(f"共生成 {self.total_tests} 个测试用例", 0)
            
            # 获取并行度配置
            if max_parallel is None:
                max_parallel = self.config_manager.get('test_settings.parallel_workers', 4)
            
            # 执行测试
            if max_parallel == 1:
                # 串行执行
                for i, test_case in enumerate(test_cases):
                    if self.should_stop_testing():
                        break
                    
                    self._update_progress(
                        f"执行测试 {i+1}/{self.total_tests}: {test_case['method']}",
                        (i / self.total_tests) * 100
                    )
                    
                    result = self.execute_single_test(test_case)
                    self._report_result(result)
                    self.current_test_count += 1
                    
                    # 检查重大奖项
                    if result.get('is_major_prize'):
                        self._update_progress(
                            f"🎉 中得{self.lottery_judge.prize_levels[result['prize_level']]}！停止测试"
                        )
                        break
            else:
                # 并行执行
                self._run_parallel_tests(test_cases, max_parallel)
            
            # 计算总结果
            execution_time = time.time() - start_time
            summary = self._generate_summary(execution_time, strategy)
            
            self._update_progress("测试完成", 100)
            return summary
            
        except Exception as e:
            self._update_progress(f"测试异常: {str(e)}")
            raise
        finally:
            self.is_running = False

    def _run_parallel_tests(self, test_cases: List[Dict], max_workers: int):
        """并行执行测试"""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_case = {
                executor.submit(self.execute_single_test, case): case 
                for case in test_cases
            }
            
            completed = 0
            
            # 处理完成的任务
            for future in as_completed(future_to_case):
                if self.should_stop_testing():
                    # 取消剩余任务
                    for f in future_to_case:
                        f.cancel()
                    break
                
                try:
                    result = future.result()
                    self._report_result(result)
                    
                    completed += 1
                    self.current_test_count = completed
                    
                    progress = (completed / len(test_cases)) * 100
                    test_case = future_to_case[future]
                    self._update_progress(
                        f"完成测试 {completed}/{len(test_cases)}: {test_case['method']}",
                        progress
                    )
                    
                    # 检查重大奖项
                    if result.get('is_major_prize'):
                        self._update_progress(
                            f"🎉 中得{self.lottery_judge.prize_levels[result['prize_level']]}！停止测试"
                        )
                        self.should_stop = True
                        
                except Exception as e:
                    self._update_progress(f"测试执行异常: {str(e)}")

    def _generate_summary(self, execution_time: float, strategy: str) -> Dict:
        """生成详细的测试摘要，包含所有奖级统计"""
        total_predictions = sum(r.get('total_predictions', 0) for r in self.test_results)
        total_winning_predictions = sum(r.get('winning_predictions', 0) for r in self.test_results)
        
        # 详细的按奖级统计 - 包含所有奖级的详细信息
        prize_stats = {}
        for i in range(1, 10):
            # 统计该奖级的总中奖次数（包括多注中同一奖级）
            total_count = 0
            winning_methods = set()
            
            for result in self.test_results:
                # 检查该测试中的所有中奖记录
                winnings = result.get('winnings', [])
                for winning in winnings:
                    if winning.get('prize_level') == i:
                        total_count += 1
                        winning_methods.add(result['test_case']['method'])
                
                # 向后兼容：检查旧格式
                if result.get('best_prize_level') == i:
                    if not winnings:  # 只有在新格式没有数据时才使用旧格式
                        total_count += 1
                        winning_methods.add(result['test_case']['method'])
            
            if total_count > 0:
                prize_stats[i] = {
                    'count': total_count,
                    'name': self.lottery_judge.prize_levels[i],
                    'winning_methods': list(winning_methods),
                    'method_count': len(winning_methods)
                }
        
        # 按方法统计 - 增加详细的奖级信息
        method_stats = {}
        for result in self.test_results:
            method = result['test_case']['method']
            if method not in method_stats:
                method_stats[method] = {
                    'total_tests': 0,
                    'winning_tests': 0,
                    'total_predictions': 0,
                    'winning_predictions': 0,
                    'best_prize_level': 0,
                    'prize_breakdown': {},  # 新增：每个奖级的中奖次数
                    'first_prize_test': None,  # 新增：首次中一等奖的测试信息
                    'major_prize_count': 0  # 新增：重大奖项次数
                }
            
            stats = method_stats[method]
            stats['total_tests'] += 1
            stats['total_predictions'] += result.get('total_predictions', 0)
            stats['winning_predictions'] += result.get('winning_predictions', 0)
            
            if result.get('is_winning'):
                stats['winning_tests'] += 1
                best_level = result.get('best_prize_level', 0)
                stats['best_prize_level'] = max(stats['best_prize_level'], best_level)
                
                # 统计各奖级详情
                winnings = result.get('winnings', [])
                for winning in winnings:
                    level = winning.get('prize_level', 0)
                    if level > 0:
                        stats['prize_breakdown'][level] = stats['prize_breakdown'].get(level, 0) + 1
                        
                        # 记录首次中一等奖的信息
                        if level == 1 and stats['first_prize_test'] is None:
                            stats['first_prize_test'] = {
                                'periods': result['test_case']['periods'],
                                'count': result['test_case']['count'],
                                'timestamp': result.get('timestamp'),
                                'winning_details': winning
                            }
                        
                        # 统计重大奖项
                        if self.lottery_judge.is_major_prize(level):
                            stats['major_prize_count'] += 1
        
        # 计算成功率
        for stats in method_stats.values():
            stats['test_winning_rate'] = stats['winning_tests'] / stats['total_tests'] if stats['total_tests'] > 0 else 0
            stats['prediction_winning_rate'] = stats['winning_predictions'] / stats['total_predictions'] if stats['total_predictions'] > 0 else 0
        
        # 找出最佳中奖方法
        best_method = None
        best_score = 0
        for method, stats in method_stats.items():
            # 综合评分：一等奖=1000分，二等奖=500分，其他按递减
            score = 0
            for level, count in stats['prize_breakdown'].items():
                if level == 1:
                    score += count * 1000
                elif level == 2:
                    score += count * 500
                elif level == 3:
                    score += count * 200
                else:
                    score += count * (10 - level) * 10
            
            if score > best_score:
                best_score = score
                best_method = method
        
        summary = {
            'strategy': strategy,
            'execution_time': execution_time,
            'total_tests': len(self.test_results),
            'total_predictions': total_predictions,
            'winning_tests': len(self.winning_results),
            'winning_predictions': total_winning_predictions,
            'major_prizes': len(self.major_prize_results),
            'test_winning_rate': len(self.winning_results) / len(self.test_results) if self.test_results else 0,
            'prediction_winning_rate': total_winning_predictions / total_predictions if total_predictions > 0 else 0,
            'prize_statistics': prize_stats,
            'method_statistics': method_stats,
            'best_method': best_method,
            'best_method_score': best_score,
            'completed_at': datetime.now().isoformat(),
            'stopped_by_major_prize': bool(self.major_prize_results)
        }
        
        return summary

    def stop(self):
        """停止测试"""
        self.should_stop = True

    def pause(self):
        """暂停测试"""
        self.is_paused = True

    def resume(self):
        """恢复测试"""
        self.is_paused = False

    def get_progress(self) -> Dict:
        """获取当前进度"""
        return {
            'is_running': self.is_running,
            'is_paused': self.is_paused,
            'current_test': self.current_test_count,
            'total_tests': self.total_tests,
            'progress_percent': (self.current_test_count / self.total_tests * 100) if self.total_tests > 0 else 0,
            'winning_count': len(self.winning_results),
            'major_prize_count': len(self.major_prize_results)
        }

    def get_results(self) -> Dict:
        """获取所有结果"""
        return {
            'test_results': self.test_results,
            'winning_results': self.winning_results,
            'major_prize_results': self.major_prize_results
        }


def test_controller():
    """测试函数"""
    config_manager = ConfigManager()
    controller = TestController(config_manager)
    
    print("=== 测试控制器测试 ===")
    
    # 设置回调函数
    def progress_callback(message, progress):
        if progress:
            print(f"[{progress:.1f}%] {message}")
        else:
            print(f"[INFO] {message}")
    
    def result_callback(result):
        if result.get('is_winning'):
            print(f"🎯 中奖: {result['test_case']['method']} - {result.get('best_prize_level', 0)}等奖")
    
    controller.set_callbacks(progress_callback, result_callback)
    
    # 生成测试用例
    print("\n1. 生成测试用例:")
    test_cases = controller.generate_test_cases("quick")
    print(f"生成了 {len(test_cases)} 个测试用例")
    for case in test_cases[:5]:
        print(f"  {case}")
    
    # 执行单个测试（如果连接正常）
    print("\n2. 单个测试:")
    if controller.predictor_caller.test_connection():
        result = controller.execute_single_test(test_cases[0])
        print(f"测试结果: {'成功' if result['success'] else '失败'}")
        if result['success']:
            print(f"  预测数量: {result.get('total_predictions', 0)}")
            print(f"  中奖数量: {result.get('winning_predictions', 0)}")
    else:
        print("无法连接到预测系统")


if __name__ == "__main__":
    test_controller()