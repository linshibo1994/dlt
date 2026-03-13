#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
大乐透预测系统 - 优化版主程序
支持延迟加载，避免启动时间过长
支持GPU加速功能
"""

import argparse
import sys
import os
import random
import json
from collections import Counter
from datetime import datetime
from typing import List, Dict, Tuple

# 设置模块搜索路径
# 添加 backend/app 及其子目录到 sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_core_dir = os.path.join(_current_dir, 'core')
_utils_dir = os.path.join(_current_dir, 'utils')
_predictors_dir = os.path.join(_current_dir, 'predictors')
_analyzers_dir = os.path.join(_current_dir, 'analyzers')
_learning_dir = os.path.join(_current_dir, 'learning')
_improvements_dir = os.path.join(_current_dir, 'improvements')

for _path in [_current_dir, _core_dir, _utils_dir, _predictors_dir, _analyzers_dir, _learning_dir, _improvements_dir]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

# 网络相关模块
try:
    import requests
except ImportError:
    requests = None

# 只导入核心模块
import core_modules as cm
from core.method_categories import REQUIRES_DEEP_LEARNING, DEEP_LEARNING_METHODS, ADVANCED_METHODS, MARKOV_METHODS
cache_manager = cm.cache_manager
logger_manager = cm.logger_manager
data_manager = cm.data_manager
task_manager = cm.task_manager


class OutputStatus:
    """统一状态输出格式"""
    OK = "[OK]"
    ERROR = "[ERROR]"
    WARNING = "[WARNING]"
    INFO = "[INFO]"
    SUCCESS = "[SUCCESS]"
    LOADING = "[...]"

# GPU加速模块
try:
    from gpu_accelerated_predictor import get_gpu_accelerator
    GPU_AVAILABLE = True
    print(f"{OutputStatus.OK} GPU加速模块已加载")
except ImportError:
    GPU_AVAILABLE = False
    print(f"{OutputStatus.WARNING} GPU加速模块不可用，使用CPU计算")

# 尝试加载增强功能集成模块
try:
    from enhanced_integration import enhanced_dlt_system, is_enhanced_available
    ENHANCED_INTEGRATION_AVAILABLE = True
    print(f"{OutputStatus.OK} 增强功能模块已启用")
except ImportError as e:
    ENHANCED_INTEGRATION_AVAILABLE = False
    enhanced_dlt_system = None
    print(f"{OutputStatus.INFO} 增强功能模块未找到: {e}")
except Exception as e:
    ENHANCED_INTEGRATION_AVAILABLE = False
    enhanced_dlt_system = None
    print(f"{OutputStatus.WARNING} 增强功能模块加载失败: {e}")

# 辅助函数：检查增强功能是否可用
# 注意：如果 enhanced_integration 模块成功导入，使用其中的 is_enhanced_available
# 否则使用本地定义的版本
if not ENHANCED_INTEGRATION_AVAILABLE:
    def is_enhanced_available():
        """检查增强功能是否可用（本地回退版本）"""
        return False


class DLTPredictorSystem:
    """大乐透预测系统主类"""

    def __init__(self):
        self.analyzers = {}
        self.predictors = {}
        self.adaptive_predictor = None

        # 延迟加载标志
        self._analyzers_loaded = False
        self._predictors_loaded = False
        self._adaptive_loaded = False

        # 配置验证
        valid, error_msg = self._validate_configuration()
        if not valid:
            logger_manager.error(f"配置验证失败: {error_msg}")
            print(f"{OutputStatus.ERROR} 配置验证失败: {error_msg}")
            raise RuntimeError(f"系统配置验证失败: {error_msg}")

        print(f"{OutputStatus.OK} 系统配置验证通过")

        # 初始化增强功能
        self.enhanced_available = ENHANCED_INTEGRATION_AVAILABLE and is_enhanced_available()
        if self.enhanced_available:
            self.enhanced_system = enhanced_dlt_system
            logger_manager.info("OK - 增强功能已集成到主系统")
        else:
            self.enhanced_system = None
            logger_manager.info("INFO - 使用基础功能模式")

    def _validate_configuration(self) -> Tuple[bool, str]:
        """
        验证系统配置

        Returns:
            Tuple[bool, str]: (验证结果, 错误信息)
        """
        try:
            # 导入路径配置管理器
            try:
                from path_config import get_path_manager
            except ImportError:
                # 如果在backend/app下导入失败，尝试从core导入
                try:
                    from core.path_config import get_path_manager
                except ImportError:
                    return False, "无法导入path_config模块，请检查模块路径"

            pm = get_path_manager()

            # 验证关键目录
            critical_dirs = {
                'cache': pm.cache_dir,
                'logs': pm.logs_dir,
                'models': pm.models_dir,
                'reports': pm.reports_dir
            }

            for dir_name, dir_path in critical_dirs.items():
                if not dir_path.exists():
                    # 尝试创建目录
                    try:
                        dir_path.mkdir(parents=True, exist_ok=True)
                        logger_manager.info(f"已创建{dir_name}目录: {dir_path}")
                    except Exception as e:
                        return False, f"无法创建{dir_name}目录 {dir_path}: {e}"

            # 验证数据文件
            if not pm.data_file.exists():
                return False, f"数据文件不存在: {pm.data_file}"

            # 验证数据文件可读性
            try:
                with open(pm.data_file, 'r', encoding='utf-8') as f:
                    # 尝试读取第一行
                    first_line = f.readline()
                    if not first_line:
                        return False, f"数据文件为空: {pm.data_file}"
            except Exception as e:
                return False, f"数据文件无法读取: {pm.data_file}, 错误: {e}"

            # 验证配置文件（可选验证）
            config_files = {
                'prediction': pm.prediction_config_file,
                'training': pm.training_config_file
            }

            for config_name, config_file in config_files.items():
                if config_file.exists():
                    logger_manager.info(f"{config_name}配置文件存在: {config_file}")
                else:
                    logger_manager.warning(f"{config_name}配置文件不存在: {config_file}")

            return True, "配置验证通过"

        except Exception as e:
            return False, f"配置验证过程出错: {e}"
    
    def _load_analyzers(self):
        """延迟加载分析器"""
        if not self._analyzers_loaded:
            print(" 加载分析器模块...")
            from analyzer_modules import basic_analyzer, advanced_analyzer, comprehensive_analyzer, visualization_analyzer
            self.analyzers = {
                'basic': basic_analyzer,
                'advanced': advanced_analyzer,
                'comprehensive': comprehensive_analyzer,
                'visualization': visualization_analyzer
            }
            self._analyzers_loaded = True
    
    def _load_predictors(self):
        """延迟加载预测器"""
        if not self._predictors_loaded:
            print(" 加载预测器模块...")
            from predictor_modules import get_traditional_predictor, get_advanced_predictor, get_super_predictor, CompoundPredictor
            self.predictors = {
                'traditional': get_traditional_predictor(),
                'advanced': get_advanced_predictor(),
                'super': get_super_predictor(),
                'compound': CompoundPredictor()
            }
            self._predictors_loaded = True
    
    def _load_adaptive_predictor(self):
        """延迟加载自适应预测器"""
        if not self._adaptive_loaded:
            print(" 加载自适应学习模块...")
            from adaptive_learning_modules import enhanced_adaptive_predictor
            self.adaptive_predictor = enhanced_adaptive_predictor
            self._adaptive_loaded = True
    
    def run_data_command(self, args):
        """处理数据管理命令"""
        if args.data_action == 'status':
            print(" 数据状态:")
            stats = data_manager.get_stats()
            print(f"  总期数: {stats.get('total_periods', 0)}")
            print(f"  数据范围: {stats.get('date_range', {}).get('start', 'N/A')} 到 {stats.get('date_range', {}).get('end', 'N/A')}")
            print(f"  最新期号: {stats.get('latest_issue', 'N/A')}")

            # 缓存信息
            cache_info = cache_manager.get_cache_info()
            print(f"\n 缓存状态:")
            print(f"  总文件数: {cache_info['total']['files']}")
            print(f"  总大小: {cache_info['total']['size_mb']:.2f} MB")

        elif args.data_action == 'latest':
            print(" 获取最新开奖结果...")
            try:
                # 获取本地最新数据
                df = data_manager.get_data()
                if df is not None and len(df) > 0:
                    latest_row = df.iloc[0]  # 第一行是最新数据
                    front_balls, back_balls = data_manager.parse_balls(latest_row)

                    print(f"{OutputStatus.OK} 最新开奖结果:")
                    print(f"  期号: {latest_row['issue']}")
                    print(f"  日期: {latest_row['date']}")
                    print(f"  开奖号码: {' '.join([str(b).zfill(2) for b in front_balls])} + {' '.join([str(b).zfill(2) for b in back_balls])}")

                    # 如果指定了比较选项
                    if hasattr(args, 'compare') and args.compare:
                        self._compare_with_latest(front_balls, back_balls)
                else:
                    print(f"{OutputStatus.ERROR} 没有找到开奖数据")
            except Exception as e:
                print(f"{OutputStatus.ERROR} 获取最新开奖结果失败: {e}")

        elif args.data_action == 'update':
            # 处理更新参数
            periods = getattr(args, 'periods', None)
            incremental = getattr(args, 'incremental', False)

            update_type = "增量更新" if incremental else "完整更新"
            print(f" {update_type} (数据源: {args.source}" + (f", 期数: {periods}" if periods else "") + ")...")

            try:
                from crawlers import ZhcwCrawler
                crawler = ZhcwCrawler()

                if incremental:
                    # 增量更新：只获取最新的几页数据
                    print(" 正在检查网络连接和数据源...")
                    count = crawler.crawl_recent_data(3)
                elif periods:
                    # 更新指定期数
                    print(f" 正在获取最近 {periods} 期数据...")
                    count = crawler.crawl_recent_data(periods)
                else:
                    # 更新所有数据
                    print(" 正在获取所有历史数据...")
                    count = crawler.crawl_all_data()

                # 清理缓存并重新加载数据
                cache_manager.clear_cache('data')
                data_manager._load_data()

                if count > 0:
                    print(f"OK 数据更新完成，新增 {count} 期数据")
                else:
                    print("INFO 没有新数据需要更新，当前数据已是最新")

            except ImportError:
                print("ERROR 爬虫模块未找到，请检查crawlers.py文件")
            except Exception as e:
                # 添加 requests 模块的 None 检查
                if requests is not None:
                    if isinstance(e, requests.exceptions.ConnectionError):
                        print("ERROR 网络连接失败，请检查网络连接")
                        print(" 提示：可以尝试使用离线模式或稍后重试")
                    elif isinstance(e, requests.exceptions.Timeout):
                        print("ERROR 网络请求超时，请稍后重试")
                    elif isinstance(e, requests.exceptions.HTTPError):
                        print(f"ERROR 服务器响应错误: {e}")
                        print(" 提示：数据源服务器可能暂时不可用，请稍后重试")
                    else:
                        print(f"ERROR 数据更新失败: {e}")
                        print(" 提示：系统严格要求使用真实开奖数据，不允许使用模拟数据")
                        print(" 建议：检查网络连接，或稍后重试多个真实数据源")
                else:
                    print(f"ERROR 数据更新失败: {e}")
                    print(" 提示：requests模块未安装，请安装后重试")

        elif args.data_action == 'check':
            print(" 开始数据完整性检查...")
            self._check_data_integrity(args)

    def _check_data_integrity(self, args):
        """检查数据完整性"""
        detailed = getattr(args, 'detailed', False)
        auto_fix = getattr(args, 'fix', False)

        issues = []

        try:
            # 1. 检查数据文件是否存在
            print("📁 检查数据文件...")
            df = data_manager.get_data()
            if df is None or len(df) == 0:
                issues.append("数据文件不存在或为空")
                print("ERROR 数据文件不存在或为空")
            else:
                print(f"OK 数据文件正常，共 {len(df)} 期数据")

            # 2. 检查数据格式
            if df is not None and len(df) > 0:
                print(" 检查数据格式...")
                # 数据使用合并后的格式：issue, date, front_balls, back_balls
                required_columns = ['issue', 'date', 'front_balls', 'back_balls']
                missing_columns = [col for col in required_columns if col not in df.columns]

                if missing_columns:
                    issues.append(f"缺少必要列: {missing_columns}")
                    print(f"ERROR 缺少必要列: {missing_columns}")
                else:
                    print("OK 数据格式正常")

                # 3. 检查数据范围
                print("检查数据范围...")
                # 数据使用合并格式：front_balls 和 back_balls 是逗号分隔的字符串
                range_issues_count = 0

                # 检查前区号码范围 (1-35)
                if 'front_balls' in df.columns:
                    for idx, row in df.iterrows():
                        try:
                            front_balls = [int(x.strip()) for x in str(row['front_balls']).split(',')]
                            for ball in front_balls:
                                if ball < 1 or ball > 35:
                                    range_issues_count += 1
                                    if detailed:
                                        print(f"ERROR 期号 {row.get('issue', idx)} 前区号码超出范围: {ball}")
                                    break
                        except (ValueError, AttributeError):
                            range_issues_count += 1
                            if detailed:
                                print(f"ERROR 期号 {row.get('issue', idx)} 前区号码格式错误")

                # 检查后区号码范围 (1-12)
                if 'back_balls' in df.columns:
                    for idx, row in df.iterrows():
                        try:
                            back_balls = [int(x.strip()) for x in str(row['back_balls']).split(',')]
                            for ball in back_balls:
                                if ball < 1 or ball > 12:
                                    range_issues_count += 1
                                    if detailed:
                                        print(f"ERROR 期号 {row.get('issue', idx)} 后区号码超出范围: {ball}")
                                    break
                        except (ValueError, AttributeError):
                            range_issues_count += 1
                            if detailed:
                                print(f"ERROR 期号 {row.get('issue', idx)} 后区号码格式错误")

                if range_issues_count > 0:
                    issues.append(f"号码范围检查发现 {range_issues_count} 个问题")
                else:
                    print("OK 数据范围正常")

                # 4. 检查重复期号
                print(" 检查重复期号...")
                if 'issue' in df.columns:
                    duplicates = df[df.duplicated(subset=['issue'], keep=False)]
                    if len(duplicates) > 0:
                        issues.append(f"发现重复期号: {len(duplicates)} 期")
                        if detailed:
                            print(f"ERROR 发现重复期号: {duplicates['issue'].unique()}")
                    else:
                        print("OK 无重复期号")

            # 5. 检查缓存状态
            print(" 检查缓存状态...")
            cache_info = cache_manager.get_cache_info()
            if cache_info['total']['files'] == 0:
                issues.append("缓存为空")
                print("WARNING 缓存为空")
            else:
                print(f"OK 缓存正常，{cache_info['total']['files']} 个文件")

            # 输出检查结果
            print("\n" + "="*50)
            if len(issues) == 0:
                print("SUCCESS 数据完整性检查通过，未发现问题")
            else:
                print(f"WARNING 发现 {len(issues)} 个问题:")
                for i, issue in enumerate(issues, 1):
                    print(f"  {i}. {issue}")

                if auto_fix:
                    print("\n 尝试自动修复...")
                    self._auto_fix_data_issues(issues)
                else:
                    print("\n 建议使用 --fix 参数自动修复问题")

        except Exception as e:
            print(f"ERROR 数据检查过程中出错: {e}")

    def _auto_fix_data_issues(self, issues):
        """自动修复数据问题"""
        fixed_count = 0

        for issue in issues:
            try:
                if "缓存为空" in issue:
                    print(" 清理并重建缓存...")
                    cache_manager.clear_cache()
                    data_manager._load_data()
                    fixed_count += 1
                    print("OK 缓存已重建")
                elif "数据文件不存在" in issue:
                    print(" 尝试重新加载数据...")
                    data_manager._load_data()
                    fixed_count += 1
                    print("OK 数据已重新加载")
                else:
                    print(f"WARNING 无法自动修复: {issue}")
            except Exception as e:
                print(f"ERROR 修复失败: {e}")

        if fixed_count > 0:
            print(f"\nSUCCESS 成功修复 {fixed_count} 个问题")
        else:
            print("\nWARNING 未能自动修复任何问题，请手动处理")

    def _compare_with_latest(self, actual_front: List[int], actual_back: List[int]):
        """与最新开奖结果比较"""
        print("\n 号码比较功能:")
        print("请输入您的号码进行比较")

        try:
            # 输入前区号码
            front_input = input("前区号码 (5个号码，用空格分隔): ").strip()
            front_numbers = [int(x) for x in front_input.split()]

            if len(front_numbers) != 5:
                print("ERROR 前区号码必须是5个")
                return

            # 输入后区号码
            back_input = input("后区号码 (2个号码，用空格分隔): ").strip()
            back_numbers = [int(x) for x in back_input.split()]

            if len(back_numbers) != 2:
                print("ERROR 后区号码必须是2个")
                return

            # 计算中奖情况
            from adaptive_learning_modules import AccuracyTracker
            tracker = AccuracyTracker()
            prize_level, front_hits, back_hits = tracker._calculate_prize_level(
                front_numbers, back_numbers, actual_front, actual_back
            )

            print(f"\n 比较结果:")
            print(f"  您的号码: {' '.join([str(b).zfill(2) for b in front_numbers])} + {' '.join([str(b).zfill(2) for b in back_numbers])}")
            print(f"  开奖号码: {' '.join([str(b).zfill(2) for b in actual_front])} + {' '.join([str(b).zfill(2) for b in actual_back])}")
            print(f"  前区命中: {front_hits} 个")
            print(f"  后区命中: {back_hits} 个")
            print(f"  中奖等级: {prize_level}")

        except ValueError:
            print("ERROR 请输入有效的数字")
        except KeyboardInterrupt:
            print("\nWARNING  操作被取消")
        except Exception as e:
            print(f"ERROR 比较失败: {e}")
    
    def run_analyze_command(self, args):
        """处理分析命令"""
        self._load_analyzers()
        
        print(f" 开始{args.type}分析 (期数: {args.periods})...")
        
        try:
            if args.type == 'basic':
                # 基础分析
                freq_result = self.analyzers['basic'].frequency_analysis(args.periods)
                hot_cold_result = self.analyzers['basic'].hot_cold_analysis(args.periods)
                
                print("OK 基础分析完成")
                print(f"  频率分析: {len(freq_result.get('front_frequency', {}))} 个前区号码")
                print(f"  冷热分析: 热号 {len(hot_cold_result.get('front_hot', []))} 个，冷号 {len(hot_cold_result.get('front_cold', []))} 个")
            
            elif args.type == 'advanced':
                # 高级分析
                markov_result = self.analyzers['advanced'].markov_analysis(args.periods)
                bayesian_result = self.analyzers['advanced'].bayesian_analysis(args.periods)
                
                print("OK 高级分析完成")
                print(f"  马尔可夫分析: {len(markov_result.get('front_transition_probs', {}))} 个转移概率")
                print(f"  贝叶斯分析: 后验概率计算完成")
            
            elif args.type == 'comprehensive':
                # 综合分析
                comp_result = self.analyzers['comprehensive'].comprehensive_analysis(args.periods)
                
                print("OK 综合分析完成")
                
                if args.report:
                    # 生成报告
                    report = self.analyzers['comprehensive'].generate_analysis_report(args.periods)
                    print("\n" + report)
                    
                    # 保存报告
                    if args.save:
                        # 确保输出目录存在
                        output_dir = "output/reports"
                        os.makedirs(output_dir, exist_ok=True)

                        if args.save.endswith('.txt'):
                            filename = os.path.join(output_dir, args.save)
                        else:
                            filename = os.path.join(output_dir, f"{args.save}.txt")

                        with open(filename, 'w', encoding='utf-8') as f:
                            f.write(report)
                        print(f" 报告已保存: {filename}")

            # 生成可视化图表
            if hasattr(args, 'visualize') and args.visualize:
                print(" 生成可视化图表...")
                viz_success = self.analyzers['visualization'].generate_all_charts("output", args.periods)
                if viz_success:
                    print("OK 可视化图表生成完成，保存在 output/ 目录")
                else:
                    print("ERROR 可视化图表生成失败")

        except Exception as e:
            logger_manager.error("分析失败", e)
            print(f"ERROR 分析失败: {e}")
    

    def _validate_predict_args(self, args) -> tuple:
        """
        验证预测参数
        
        Args:
            args: 命令行参数对象
            
        Returns:
            tuple: (is_valid, error_msg, acceleration_config)
                - is_valid: 参数是否有效
                - error_msg: 错误信息（如果无效）
                - acceleration_config: 加速配置字典
        """
        # 验证注数范围
        if args.count < 1 or args.count > 100:
            return (False, "注数必须在1-100之间", None)
        
        # 验证分析期数范围
        if args.periods < 50 or args.periods > 2748:
            return (False, "分析期数必须在50-2748之间", None)
        
        # 处理加速参数
        acceleration_config = self._process_acceleration_args(args)
        if acceleration_config:
            print(f"{OutputStatus.INFO} 加速配置: {acceleration_config['mode']}")
            if acceleration_config['mode'] == 'cpu_multi':
                print(f"{OutputStatus.INFO} CPU多线程: {acceleration_config['cpu_threads']} 线程")
            elif acceleration_config['mode'] in ['gpu', 'gpu_cuda']:
                print(f"{OutputStatus.INFO} GPU加速: 设备 {acceleration_config['gpu_device']}")
                if acceleration_config.get('gpu_memory_limit'):
                    print(f"{OutputStatus.INFO} GPU内存限制: {acceleration_config['gpu_memory_limit']} GB")
        
        return (True, "", acceleration_config)


    def _handle_enhanced_prediction(self, args, acceleration_config) -> tuple:
        """
        处理增强预测模式
        
        Args:
            args: 命令行参数对象
            acceleration_config: 加速配置
            
        Returns:
            tuple: (success, predictions)
                - success: 是否成功完成预测
                - predictions: 预测结果列表
        """
        print(f"{OutputStatus.INFO} 使用增强预测引擎...")
        try:
            if args.method == 'enhanced':
                # 使用增强系统的自动预测
                result = self.enhanced_system.enhanced_predict(
                    data=f"predict_{args.count}_numbers_periods_{args.periods}",
                    method="auto",
                    periods=args.periods,
                    count=args.count
                )
                if result.get('success'):
                    print(f"{OutputStatus.OK} 增强预测完成")
                    print(f"预测结果: {result['result']}")
                    print(f"使用方法: {result['method']}")
                    print(f"已缓存: {result['cached']}")
                    if getattr(args, 'json_output', False):
                        payload = {
                            'mode': 'enhanced',
                            'method': args.method,
                            'periods': args.periods,
                            'count': args.count,
                            'predictions': [],
                            'details': result,
                        }
                        print(json.dumps(payload, ensure_ascii=False))
                    return (True, [])
                else:
                    print(f"{OutputStatus.ERROR} 增强预测失败: {result.get('error')}")
                    print(f"{OutputStatus.INFO} 回退到传统预测方法...")
                    return (False, [])
            
            elif args.method in REQUIRES_DEEP_LEARNING:
                # 使用增强深度学习模型或集成方法
                print(f"{OutputStatus.INFO} 检测到深度学习方法: {args.method}")
                return self._handle_enhanced_deep_learning(args)
                
        except Exception as e:
            logger_manager.error(f"增强预测失败: {e}")
            print(f"{OutputStatus.ERROR} 增强预测失败: {e}")
            print(f"{OutputStatus.INFO} 回退到传统预测方法...")
        
        return (False, [])

    def _handle_enhanced_deep_learning(self, args) -> tuple:
        """
        处理增强模式下的深度学习预测
        
        Args:
            args: 命令行参数对象
            
        Returns:
            tuple: (success, predictions)
        """
        try:
            if args.method in DEEP_LEARNING_METHODS:
                # 深度学习模型
                print(f"{OutputStatus.LOADING} 导入深度学习模型注册表...")
                from enhanced_deep_learning.models import get_model_registry
                model_registry = get_model_registry()
                model = model_registry.get_model(args.method)
                print(f"{OutputStatus.INFO} 获取模型: {model}")
                
                if model:
                    print(f"{OutputStatus.INFO} 使用{args.method.upper()}深度学习模型...")
                    historical_data = data_manager.get_data()
                    print(f"{OutputStatus.INFO} 获取历史数据: {len(historical_data) if historical_data is not None else 0}期")
                    
                    if historical_data is not None and len(historical_data) > args.periods:
                        # 使用最新的periods期数据（数据按降序排列，最新在前，使用head）
                        historical_data = historical_data.head(args.periods)
                        print(f"{OutputStatus.INFO} 使用最新{args.periods}期数据进行{args.method.upper()}模型训练...")
                    
                    print(f"{OutputStatus.INFO} 开始{args.method.upper()}预测...")
                    # 修复方法调用一致性：使用与GUI相同的predict方法
                    predictions = []
                    for _ in range(args.count):
                        single_result = model.predict(historical_data)
                        if single_result:
                            predictions.extend(single_result)
                    
                    # 转换为命令行期望的格式
                    if predictions:
                        formatted_predictions = []
                        for front, back in predictions:
                            formatted_predictions.append({
                                'front_balls': front,
                                'back_balls': back,
                                'method': args.method,
                                'confidence': 0.85
                            })
                        predictions = formatted_predictions
                    
                    print(f"{OutputStatus.INFO} 预测结果: {len(predictions)}注")
                    
                    if predictions:
                        print(f"{OutputStatus.OK} {args.method.upper()}预测完成")
                        if getattr(args, 'json_output', False):
                            self._output_predictions_json(predictions, args, mode='deep_learning')
                        else:
                            self._display_enhanced_predictions(predictions, args.method)
                        return (True, predictions)
                    else:
                        print(f"{OutputStatus.ERROR} {args.method}深度学习模型预测失败，尝试集成方法...")
                else:
                    print(f"{OutputStatus.ERROR} {args.method}深度学习模型未找到，尝试集成方法...")
            
            # 如果深度学习模型失败或者是集成方法，使用improvements模块
            return self._handle_fallback_prediction(args)
            
        except Exception as e:
            print(f"{OutputStatus.ERROR} 增强预测失败: {e}")
            print(f"{OutputStatus.INFO} 回退到传统预测方法...")
        
        return (False, [])

    def _handle_fallback_prediction(self, args) -> tuple:
        """
        处理回退预测（当深度学习模型失败时）
        
        Args:
            args: 命令行参数对象
            
        Returns:
            tuple: (success, predictions)
        """
        try:
            from improvements.integration import get_integrator
            integrator = get_integrator()
            
            predictions = []
            if args.method == 'lstm':
                print(f"{OutputStatus.INFO} LSTM集成预测...")
                # 尝试使用advanced_lstm_predictor作为回退
                try:
                    from advanced_lstm_predictor import AdvancedLSTMPredictor
                    lstm_predictor = AdvancedLSTMPredictor()
                    results = lstm_predictor.lstm_predict(count=args.count, periods=args.periods)
                    predictions = [{'front_balls': r[0], 'back_balls': r[1], 'method': 'lstm', 'confidence': 0.85} for r in results]
                except Exception as e:
                    print(f"{OutputStatus.ERROR} LSTM预测失败: {e}")
                    predictions = []
            elif args.method == 'transformer':
                print(f"{OutputStatus.INFO} Transformer深度学习预测...")
                predictions = integrator.transformer_predict(args.count, args.periods)
            elif args.method == 'gan':
                print(f"{OutputStatus.INFO} GAN生成预测...")
                predictions = integrator.gan_predict(args.count, args.periods)
            elif args.method == 'stacking':
                print(f"{OutputStatus.INFO} Stacking集成预测...")
                predictions = integrator.stacking_predict(args.count, args.periods)
            elif args.method == 'adaptive_ensemble':
                print(f"{OutputStatus.INFO} 自适应集成预测...")
                predictions = integrator.adaptive_ensemble_predict(args.count, args.periods)
            elif args.method == 'ultimate_ensemble':
                print(f"{OutputStatus.INFO} 终极集成预测...")
                predictions = integrator.ultimate_ensemble_predict(args.count, args.periods)
            
            if predictions:
                print(f"{OutputStatus.OK} {args.method.upper()}预测完成")
                if getattr(args, 'json_output', False):
                    self._output_predictions_json(predictions, args, mode='deep_learning')
                else:
                    self._display_enhanced_predictions(predictions, args.method)
                return (True, predictions)
            else:
                print(f"{OutputStatus.ERROR} {args.method}预测失败，回退到传统方法...")
        
        except Exception as e:
            print(f"{OutputStatus.ERROR} 回退预测失败: {e}")
        
        return (False, [])


    def _handle_deep_learning_prediction(self, args, acceleration_config) -> tuple:
        """
        处理深度学习预测（独立于增强功能）
        
        Args:
            args: 命令行参数对象
            acceleration_config: 加速配置
            
        Returns:
            tuple: (success, predictions)
        """
        try:
            if args.method in DEEP_LEARNING_METHODS:
                # 深度学习模型
                from enhanced_deep_learning.models import get_model_registry
                model_registry = get_model_registry()
                model = model_registry.get_model(args.method)
                
                if model:
                    historical_data = data_manager.get_data()
                    
                    if historical_data is not None and len(historical_data) > args.periods:
                        # 使用最新的periods期数据（数据按降序排列，最新在前，使用head）
                        historical_data = historical_data.head(args.periods)
                    
                    # 修复方法调用一致性：使用与GUI相同的predict方法
                    predictions = model.predict(historical_data, count=args.count)
                    
                    # 转换为命令行期望的格式
                    if predictions:
                        formatted_predictions = []
                        for front, back in predictions:
                            formatted_predictions.append({
                                'front_balls': front,
                                'back_balls': back,
                                'method': args.method,
                                'confidence': 0.85
                            })
                        predictions = formatted_predictions
                    
                    if predictions:
                        if getattr(args, 'json_output', False):
                            self._output_predictions_json(predictions, args, mode='deep_learning')
                        else:
                            self._display_enhanced_predictions(predictions, args.method)
                        return (True, predictions)
                    else:
                        print(f"{OutputStatus.ERROR} {args.method}深度学习模型预测失败，尝试传统方法...")
                else:
                    print(f"{OutputStatus.ERROR} {args.method}深度学习模型未找到，尝试传统方法...")
            
            elif args.method in ['stacking', 'adaptive_ensemble', 'ultimate_ensemble']:
                # 集成学习方法
                print(f"{OutputStatus.INFO} 使用{args.method}集成学习方法...")
                predictions = self._handle_ensemble_prediction(args)
                
                if predictions:
                    print(f"{OutputStatus.OK} {args.method}预测完成")
                    if getattr(args, 'json_output', False):
                        self._output_predictions_json(predictions, args, mode='deep_learning')
                    else:
                        self._display_enhanced_predictions(predictions, args.method)
                    return (True, predictions)
                else:
                    print(f"{OutputStatus.ERROR} {args.method}预测失败，尝试传统方法...")
        
        except Exception as e:
            print(f"{OutputStatus.ERROR} 深度学习预测失败: {e}")
            print(f"{OutputStatus.INFO} 回退到传统预测方法...")
        
        return (False, [])

    def _handle_ensemble_prediction(self, args) -> list:
        """
        处理集成学习预测
        
        Args:
            args: 命令行参数对象
            
        Returns:
            list: 预测结果列表
        """
        predictions = []
        
        if args.method == 'stacking':
            # 使用简化的堆叠集成实现，避免深度学习模型初始化超时
            print(f"{OutputStatus.INFO} 使用堆叠集成预测...")
            predictions = self.predictors['advanced'].stacking_predict(count=args.count, periods=args.periods)
        
        elif args.method == 'adaptive_ensemble':
            from adaptive_learning_modules import EnhancedAdaptiveLearningPredictor
            learner = EnhancedAdaptiveLearningPredictor()
            predictions = learner.generate_enhanced_prediction(count=args.count, periods=args.periods)
        
        elif args.method == 'ultimate_ensemble':
            # 使用真正的终极集成实现
            print(f"{OutputStatus.INFO} 使用终极集成预测...")
            try:
                from improvements.integration import IntegratedPredictor
                integrator = IntegratedPredictor()
                predictions = integrator.ultimate_ensemble_predict(count=args.count, periods=args.periods)
                
                # 确保包含置信度信息
                if predictions and isinstance(predictions[0], dict):
                    # 已经是正确格式，检查置信度
                    for pred in predictions:
                        if 'confidence' not in pred or pred['confidence'] == 0.98:
                            # 重新计算置信度
                            pred['confidence'] = 0.85  # 设置合理的置信度
                else:
                    # 转换格式并添加置信度
                    predictions = [{'front_balls': r[0], 'back_balls': r[1], 'method': 'ultimate_ensemble', 'confidence': 0.85} for r in predictions]
            except Exception as e:
                print(f"{OutputStatus.ERROR} 终极集成预测失败: {e}")
                # 回退到基础集成方法
                predictions = self.predictors['advanced'].ensemble_predict(count=args.count, periods=args.periods)
                predictions = [{'front_balls': r[0], 'back_balls': r[1], 'method': 'ultimate_ensemble', 'confidence': 0.75} for r in predictions]
        
        return predictions


    def _handle_compound_prediction(self, args):
        """
        处理复式预测模式
        
        Args:
            args: 命令行参数对象
            
        Returns:
            compound_result: 复式预测结果对象，或None
        """
        print(f"{OutputStatus.INFO} 启用复式预测模式: {args.front_count}+{args.back_count}")
        
        try:
            try:
                from compound.compound_predictor import CompoundConfig
            except ImportError:
                from compound_modules.compound_predictor import CompoundConfig
            
            compound_config = CompoundConfig(
                front_count=args.front_count,
                back_count=args.back_count,
                periods=args.periods,
                max_cost=getattr(args, 'max_cost', 10000)
            )
            
            # 统一的复式预测处理
            compound_result = None
            
            if args.method in ['frequency', 'hot_cold', 'missing']:
                print(f"{OutputStatus.INFO} {args.method}复式预测 (分析{args.periods}期数据)...")
                from analyzer_modules import BasicAnalyzer
                analyzer = BasicAnalyzer()
                compound_result = analyzer.predict_compound(compound_config)
            
            elif args.method in MARKOV_METHODS + ['bayesian', 'ensemble']:
                print(f"{OutputStatus.INFO} {args.method}复式预测 (分析{args.periods}期数据)...")
                # 直接使用基础分析器进行复式预测
                from analyzer_modules import BasicAnalyzer
                analyzer = BasicAnalyzer()
                compound_result = analyzer.predict_compound(compound_config)
            
            elif args.method in DEEP_LEARNING_METHODS:
                print(f"{OutputStatus.INFO} {args.method}深度学习复式预测 (分析{args.periods}期数据)...")
                compound_result = self._handle_deep_learning_compound(args, compound_config)
            
            elif args.method in ['super', 'adaptive', 'enhanced', 'mixed_strategy', 'highly_integrated', 'advanced_integration', 'nine_models']:
                print(f"{OutputStatus.INFO} {args.method}智能复式预测 (分析{args.periods}期数据)...")
                # 使用超级预测器的复式预测功能
                if hasattr(self.predictors['super'], 'predict_compound'):
                    compound_result = self.predictors['super'].predict_compound(compound_config)
                else:
                    # 回退到基础分析器
                    from analyzer_modules import BasicAnalyzer
                    analyzer = BasicAnalyzer()
                    compound_result = analyzer.predict_compound(compound_config)
            
            elif args.method in ['stacking', 'adaptive_ensemble', 'ultimate_ensemble']:
                print(f"{OutputStatus.INFO} {args.method}集成复式预测 (分析{args.periods}期数据)...")
                # 使用集成预测器的复式预测功能
                if hasattr(self.predictors['advanced'], 'predict_compound'):
                    compound_result = self.predictors['advanced'].predict_compound(compound_config)
                else:
                    # 回退到基础分析器
                    from analyzer_modules import BasicAnalyzer
                    analyzer = BasicAnalyzer()
                    compound_result = analyzer.predict_compound(compound_config)
            
            else:
                print(f"{OutputStatus.INFO} {args.method}方法使用通用复式预测...")
                # 通用复式预测回退
                from analyzer_modules import BasicAnalyzer
                analyzer = BasicAnalyzer()
                compound_result = analyzer.predict_compound(compound_config)
            
            return compound_result
        
        except Exception as e:
            print(f"{OutputStatus.ERROR} 复式预测失败: {e}")
            print(f"{OutputStatus.INFO} 回退到单式预测...")
        
        return None

    def _handle_deep_learning_compound(self, args, compound_config):
        """
        处理深度学习复式预测
        
        Args:
            args: 命令行参数对象
            compound_config: 复式预测配置
            
        Returns:
            compound_result: 复式预测结果对象，或None
        """
        try:
            if args.method == 'lstm':
                from enhanced_deep_learning.models.lstm_predictor import LSTMPredictor
                predictor = LSTMPredictor()
                return predictor.predict_compound(compound_config)
            elif args.method == 'transformer':
                from enhanced_deep_learning.models.transformer_predictor import TransformerPredictor
                predictor = TransformerPredictor()
                return predictor.predict_compound(compound_config)
            elif args.method == 'gan':
                from enhanced_deep_learning.models.gan_predictor import GANPredictor
                predictor = GANPredictor()
                return predictor.predict_compound(compound_config)
        except Exception as e:
            print(f"{OutputStatus.WARNING} 深度学习复式预测失败: {e}")
            # 回退到基础分析器
            from analyzer_modules import BasicAnalyzer
            analyzer = BasicAnalyzer()
            return analyzer.predict_compound(compound_config)
        
        return None

    def _display_compound_result(self, compound_result):
        """
        显示复式预测结果
        
        Args:
            compound_result: 复式预测结果对象
        """
        print(f"{OutputStatus.OK} 复式预测完成!")
        print(f"{OutputStatus.INFO} 复式预测结果:")
        print(f"  前区号码 ({compound_result.front_count}个): {' '.join([str(x).zfill(2) for x in compound_result.front_balls])}")
        print(f"  后区号码 ({compound_result.back_count}个): {' '.join([str(x).zfill(2) for x in compound_result.back_balls])}")
        print(f"  总组合数: {compound_result.total_combinations:,}")
        print(f"  投注成本: {compound_result.total_cost:,} 元")
        print(f"  置信度: {compound_result.confidence:.3f}")
        print(f"  预测方法: {compound_result.method}")


    def _handle_traditional_prediction(self, args, acceleration_config) -> list:
        """
        处理传统预测方法
        
        Args:
            args: 命令行参数对象
            acceleration_config: 加速配置
            
        Returns:
            list: 预测结果列表
        """
        predictions = []

        # 传递遗漏预测模式到相关预测器（避免影响其他方法）
        missing_mode = getattr(args, 'missing_mode', 'auto')
        if 'traditional' in self.predictors and hasattr(self.predictors['traditional'], 'set_missing_mode_override'):
            self.predictors['traditional'].set_missing_mode_override(missing_mode)
        if 'advanced' in self.predictors and hasattr(self.predictors['advanced'], 'set_missing_mode_override'):
            self.predictors['advanced'].set_missing_mode_override(missing_mode)
        if 'super' in self.predictors and hasattr(self.predictors['super'], 'set_missing_mode_override'):
            self.predictors['super'].set_missing_mode_override(missing_mode)
        
        if args.method == 'consensus_halving':
            predictions = self._handle_consensus_halving(args)

        elif args.method in ['frequency', 'hot_cold', 'missing']:
            predictions = self._handle_basic_prediction(args, acceleration_config)
        
        elif args.method in ADVANCED_METHODS or args.method in ['markov', 'bayesian']:
            predictions = self._handle_advanced_prediction(args, acceleration_config)
        
        elif args.method == 'super':
            predictions = self._handle_super_prediction(args)
        
        elif args.method == 'adaptive':
            # 自适应预测 - 使用AdvancedPredictor的adaptive_predict方法
            results = self.predictors['advanced'].adaptive_predict(args.count, args.periods)
            predictions = [{'front_balls': r[0], 'back_balls': r[1], 'method': 'adaptive'} for r in results]
        
        elif args.method == 'compound':
            predictions = self._handle_compound_method(args)
        
        elif args.method == 'duplex':
            predictions = self._handle_duplex_method(args)
        
        elif args.method in ['transformer', 'gan', 'stacking', 'adaptive_ensemble', 'ultimate_ensemble']:
            predictions = self._handle_integration_methods(args)
        
        elif args.method == 'markov_custom':
            predictions = self._handle_markov_custom(args)
        
        elif args.method == 'mixed_strategy':
            predictions = self._handle_mixed_strategy(args)
        
        elif args.method == 'highly_integrated':
            predictions = self._handle_highly_integrated(args)
        
        elif args.method == 'advanced_integration':
            predictions = self._handle_advanced_integration(args)
        
        elif args.method == 'nine_models':
            results = self.predictors['advanced'].nine_models_predict(count=args.count, periods=args.periods)
            predictions = results
        
        elif args.method == 'nine_models_compound':
            predictions = self._handle_nine_models_compound(args)
        
        elif args.method == 'markov_compound':
            predictions = self._handle_markov_compound(args)
        
        elif args.method in ['markov_2nd', 'markov_3rd', 'adaptive_markov']:
            predictions = self._handle_enhanced_markov(args)
        
        return predictions

    def _handle_consensus_halving(self, args) -> list:
        """
        交集递减预测：
        每轮执行冷热号/一阶马尔可夫/频率/二阶马尔可夫，提取重复号，
        若不足则对半缩小期数继续，最后用全轮次高频号码补齐。
        """
        print(f"{OutputStatus.INFO} 交集递减预测 (初始分析{args.periods}期, 生成{args.count}注)...")

        predictions = []
        for _ in range(args.count):
            ticket = self._build_consensus_ticket(args.periods)
            predictions.append({
                'front_balls': ticket['front_balls'],
                'back_balls': ticket['back_balls'],
                'method': 'consensus_halving',
                'confidence': ticket['confidence'],
                'details': ticket['details']
            })

        return predictions

    def _build_consensus_ticket(self, initial_periods: int) -> dict:
        """构建单注交集递减号码。"""
        selected_front = []
        selected_back = []
        selected_front_set = set()
        selected_back_set = set()

        total_front_counter = Counter()
        total_back_counter = Counter()
        total_front_pool = []
        total_back_pool = []

        rounds = []
        periods = max(50, int(initial_periods))
        max_rounds = 8
        round_index = 0
        fallback_used = False

        while round_index < max_rounds:
            round_index += 1
            round_predictions = self._run_consensus_round(periods)

            round_front_counter = Counter()
            round_back_counter = Counter()

            for pred in round_predictions:
                front_balls = pred.get('front_balls', [])
                back_balls = pred.get('back_balls', [])
                round_front_counter.update(front_balls)
                round_back_counter.update(back_balls)
                total_front_counter.update(front_balls)
                total_back_counter.update(back_balls)
                total_front_pool.extend(front_balls)
                total_back_pool.extend(back_balls)

            repeated_front = sorted([num for num, cnt in round_front_counter.items() if cnt >= 2])
            repeated_back = sorted([num for num, cnt in round_back_counter.items() if cnt >= 2])

            for num in repeated_front:
                if num not in selected_front_set:
                    selected_front.append(num)
                    selected_front_set.add(num)

            for num in repeated_back:
                if num not in selected_back_set:
                    selected_back.append(num)
                    selected_back_set.add(num)

            rounds.append({
                'round': round_index,
                'periods': periods,
                'repeated_front': repeated_front,
                'repeated_back': repeated_back
            })

            if len(selected_front) >= 5 and len(selected_back) >= 2:
                break

            if periods <= 50:
                break

            next_periods = max(50, periods // 2)
            if next_periods == periods:
                break
            periods = next_periods

        if len(selected_front) < 5:
            fallback_used = True
            self._fill_from_counter(total_front_counter, selected_front, selected_front_set, 5, 1, 35)
        if len(selected_back) < 2:
            fallback_used = True
            self._fill_from_counter(total_back_counter, selected_back, selected_back_set, 2, 1, 12)

        # 二次兜底：若高频仍不足，则从所有已生成号码中随机补齐（去重）
        if len(selected_front) < 5 or len(selected_back) < 2:
            fallback_used = True
        self._fill_random_from_pool(total_front_pool, selected_front, selected_front_set, 5, 1, 35)
        self._fill_random_from_pool(total_back_pool, selected_back, selected_back_set, 2, 1, 12)

        # 最终兜底：若仍不足，使用随机合法号码补齐（去重）
        self._fill_with_random_range(selected_front, selected_front_set, 5, 1, 35)
        self._fill_with_random_range(selected_back, selected_back_set, 2, 1, 12)

        final_front = sorted(selected_front[:5])
        final_back = sorted(selected_back[:2])
        used_rounds = len(rounds)
        confidence = min(0.92, 0.55 + 0.04 * used_rounds)

        return {
            'front_balls': final_front,
            'back_balls': final_back,
            'confidence': round(confidence, 3),
            'details': {
                'strategy': 'consensus_halving',
                'rounds': rounds,
                'total_rounds': used_rounds,
                'fallback_used': fallback_used
            }
        }

    def _run_consensus_round(self, periods: int) -> list:
        """执行一轮四算法预测。"""
        methods = ['markov', 'hot_cold', 'frequency', 'markov_2nd']
        round_predictions = []

        for method in methods:
            prediction = self._predict_single_for_consensus(method, periods)
            if prediction:
                round_predictions.append(prediction)

        return round_predictions

    def _predict_single_for_consensus(self, method: str, periods: int) -> dict:
        """调用单个基础算法，返回标准化后的单注结果。"""
        try:
            results = []
            if method == 'markov':
                results = self.predictors['advanced'].markov_predict(1, periods)
            elif method == 'hot_cold':
                results = self.predictors['traditional'].hot_cold_predict(count=1, periods=periods)
            elif method == 'frequency':
                results = self.predictors['traditional'].frequency_predict(count=1, periods=periods)
            elif method == 'markov_2nd':
                from improvements.enhanced_markov import get_markov_predictor
                markov_predictor = get_markov_predictor()
                results = markov_predictor.multi_order_markov_predict(count=1, periods=periods, order=2)

            if not results:
                return {}

            first_result = results[0]
            if not isinstance(first_result, (list, tuple)) or len(first_result) < 2:
                return {}

            front_balls = self._normalize_ball_list(first_result[0], 1, 35)
            back_balls = self._normalize_ball_list(first_result[1], 1, 12)

            return {
                'method': method,
                'front_balls': front_balls,
                'back_balls': back_balls
            }

        except Exception as e:
            logger_manager.warning(f"交集递减方法调用{method}失败: {e}")
            return {}

    @staticmethod
    def _normalize_ball_list(ball_list, min_value: int, max_value: int) -> List[int]:
        """标准化号码列表，去重并过滤越界值。"""
        normalized = []
        seen = set()
        for ball in ball_list or []:
            try:
                num = int(ball)
            except (TypeError, ValueError):
                continue
            if min_value <= num <= max_value and num not in seen:
                normalized.append(num)
                seen.add(num)
        return normalized

    @staticmethod
    def _fill_from_counter(counter: Counter, selected: list, selected_set: set, target: int, min_value: int, max_value: int):
        """从全局频次中按高频优先补齐。"""
        for num, _ in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
            if num in selected_set:
                continue
            if not (min_value <= num <= max_value):
                continue
            selected.append(num)
            selected_set.add(num)
            if len(selected) >= target:
                break

    @staticmethod
    def _fill_random_from_pool(pool: list, selected: list, selected_set: set, target: int, min_value: int, max_value: int):
        """从已生成号码池中随机补齐（自动去重）。"""
        if len(selected) >= target:
            return
        candidates = [int(num) for num in pool if isinstance(num, int) or str(num).isdigit()]
        candidates = [num for num in candidates if min_value <= num <= max_value and num not in selected_set]
        random.shuffle(candidates)
        for num in candidates:
            if num in selected_set:
                continue
            selected.append(num)
            selected_set.add(num)
            if len(selected) >= target:
                break

    @staticmethod
    def _fill_with_random_range(selected: list, selected_set: set, target: int, min_value: int, max_value: int):
        """最终兜底：从合法范围随机补齐。"""
        if len(selected) >= target:
            return
        remaining_candidates = [num for num in range(min_value, max_value + 1) if num not in selected_set]
        random.shuffle(remaining_candidates)
        for num in remaining_candidates:
            if num in selected_set:
                continue
            selected.append(num)
            selected_set.add(num)
            if len(selected) >= target:
                break

    def _handle_basic_prediction(self, args, acceleration_config) -> list:
        """
        处理基础预测方法（频率分析、冷热号、遗漏值）
        
        Args:
            args: 命令行参数对象
            acceleration_config: 加速配置
            
        Returns:
            list: 预测结果列表
        """
        results = []
        
        if args.method == 'frequency':
            print(f"{OutputStatus.INFO} 频率分析预测 (分析{args.periods}期数据)...")
            results = self.predictors['traditional'].frequency_predict(count=args.count, periods=args.periods)
        
        elif args.method == 'hot_cold':
            print(f"{OutputStatus.INFO} 冷热号分析预测 (分析{args.periods}期数据)...")
            print(f"{OutputStatus.INFO} 分析冷热号分布...")
            
            # 获取冷热号分析结果
            from analyzer_modules import basic_analyzer
            hot_cold_analysis = basic_analyzer.hot_cold_analysis(args.periods)
            
            front_hot = hot_cold_analysis.get('front_hot', [])
            front_cold = hot_cold_analysis.get('front_cold', [])
            back_hot = hot_cold_analysis.get('back_hot', [])
            back_cold = hot_cold_analysis.get('back_cold', [])
            
            print(f"{OutputStatus.OK} 冷热号识别完成:")
            print(f"  前区热号 ({len(front_hot)}个): {sorted(front_hot)[:10]}{'...' if len(front_hot) > 10 else ''}")
            print(f"  前区冷号 ({len(front_cold)}个): {sorted(front_cold)[:10]}{'...' if len(front_cold) > 10 else ''}")
            print(f"  后区热号 ({len(back_hot)}个): {sorted(back_hot)}")
            print(f"  后区冷号 ({len(back_cold)}个): {sorted(back_cold)}")
            print(f"{OutputStatus.INFO} 基于冷热号分布进行智能预测...")
            
            results = self.predictors['traditional'].hot_cold_predict(count=args.count, periods=args.periods)
        
        elif args.method == 'missing':
            print(f"{OutputStatus.INFO} 遗漏值分析预测 (分析{args.periods}期数据)...")
            results = self.predictors['traditional'].missing_predict(
                count=args.count,
                periods=args.periods,
                mode=getattr(args, 'missing_mode', 'auto')
            )
        
        return [{'front_balls': r[0], 'back_balls': r[1], 'method': args.method} for r in results]

    def _handle_advanced_prediction(self, args, acceleration_config) -> list:
        """
        处理高级预测方法（马尔可夫、贝叶斯、集成、聚类）
        
        Args:
            args: 命令行参数对象
            acceleration_config: 加速配置
            
        Returns:
            list: 预测结果列表
        """
        results = []
        
        if args.method == 'markov':
            results = self.predictors['advanced'].markov_predict(args.count, args.periods)
        
        elif args.method == 'clustering':
            print(f"{OutputStatus.INFO} 聚类分析预测 (分析{args.periods}期数据)...")
            print(f"{OutputStatus.INFO} 构建特征向量...")
            print(f"{OutputStatus.INFO} 进行K-Means聚类...")
            
            # 应用加速配置
            accel_config = self._apply_acceleration_config('clustering', acceleration_config)
            
            results = self.predictors['advanced'].clustering_predict(count=args.count, periods=args.periods)
            print(f"{OutputStatus.OK} 聚类分析完成，生成{len(results)}注预测")
        
        elif args.method == 'bayesian':
            results = self._handle_bayesian_prediction(args, acceleration_config)
        
        elif args.method == 'ensemble':
            results = self.predictors['advanced'].ensemble_predict(args.count, args.periods)
        
        return [{'front_balls': r[0], 'back_balls': r[1], 'method': args.method} for r in results]

    def _handle_bayesian_prediction(self, args, acceleration_config) -> list:
        """
        处理贝叶斯预测
        
        Args:
            args: 命令行参数对象
            acceleration_config: 加速配置
            
        Returns:
            list: 原始预测结果
        """
        print(f"{OutputStatus.INFO} 贝叶斯分析预测 (分析{args.periods}期数据)...")
        print(f"{OutputStatus.INFO} 计算先验概率和似然函数...")
        
        # 应用加速配置
        accel_config = self._apply_acceleration_config('bayesian', acceleration_config)
        
        # 获取贝叶斯分析结果
        from analyzer_modules import advanced_analyzer, load_bayesian_config

        # 应用配置并解析模式
        bayes_cfg = load_bayesian_config()
        resolved_mode = getattr(args, 'bayes_mode', None) or bayes_cfg.get('mode') or 'legacy'
        print(f"{OutputStatus.INFO} 贝叶斯模式: {resolved_mode}")
        print(f"{OutputStatus.INFO} Bayes参数: mix={bayes_cfg.get('dirichlet_mix_weight', 'n/a')}, "
              f"conc={bayes_cfg.get('dirichlet_concentration', 'n/a')}, "
              f"decay={bayes_cfg.get('decay_enabled', 'n/a')}, "
              f"half_life={bayes_cfg.get('decay_half_life', 'n/a')}, "
              f"min_weight={bayes_cfg.get('decay_min_weight', 'n/a')}")
        if accel_config and 'n_jobs' in accel_config:
            print(f"{OutputStatus.INFO} 使用 {accel_config['n_jobs']} 个CPU线程并行计算")
            bayesian_analysis = advanced_analyzer.bayesian_analysis(args.periods, n_jobs=accel_config['n_jobs'])
        else:
            bayesian_analysis = advanced_analyzer.bayesian_analysis(args.periods)
        
        front_prior = bayesian_analysis.get('front_prior', {})
        back_prior = bayesian_analysis.get('back_prior', {})
        front_posterior = bayesian_analysis.get('front_posterior', {})
        back_posterior = bayesian_analysis.get('back_posterior', {})
        
        print(f"{OutputStatus.OK} 贝叶斯推理完成:")
        print(f"  前区先验概率计算: {len(front_prior)} 个号码")
        print(f"  前区后验概率计算: {len(front_posterior)} 个号码")
        print(f"  后区先验概率计算: {len(back_prior)} 个号码")
        print(f"  后区后验概率计算: {len(back_posterior)} 个号码")
        
        if front_posterior:
            top_front = sorted(front_posterior.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"  前区最高后验概率: {[f'{k}({v:.3f})' for k, v in top_front]}")
        
        if back_posterior:
            top_back = sorted(back_posterior.items(), key=lambda x: x[1], reverse=True)[:2]
            print(f"  后区最高后验概率: {[f'{k}({v:.3f})' for k, v in top_back]}")
        
        print(f"{OutputStatus.INFO} 基于贝叶斯推理进行概率预测...")
        
        # 应用加速配置到预测
        use_enhanced = resolved_mode == 'enhanced'
        if accel_config and 'n_jobs' in accel_config:
            return self.predictors['traditional'].bayesian_predict(
                count=args.count, periods=args.periods, n_jobs=accel_config['n_jobs'], use_enhanced=use_enhanced
            )
        else:
            return self.predictors['traditional'].bayesian_predict(
                count=args.count, periods=args.periods, use_enhanced=use_enhanced
            )

    def _handle_super_prediction(self, args) -> list:
        """
        处理超级预测
        
        Args:
            args: 命令行参数对象
            
        Returns:
            list: 预测结果列表
        """
        try:
            results = self.predictors['super'].predict_super(count=args.count, periods=args.periods)
            return results
        except Exception as e:
            print(f"{OutputStatus.WARNING} 超级预测失败: {e}")
            print(f"{OutputStatus.INFO} 回退到集成预测...")
            results = self.predictors['advanced'].ensemble_predict(args.count, args.periods)
            return [{'front_balls': r[0], 'back_balls': r[1], 'method': 'super_fallback'} for r in results]

    def _handle_compound_method(self, args) -> list:
        """
        处理复式投注预测
        
        Args:
            args: 命令行参数对象
            
        Returns:
            list: 预测结果列表
        """
        front_count = getattr(args, 'front_count', 8)
        back_count = getattr(args, 'back_count', 4)
        result = self.predictors['compound'].predict_compound(front_count, back_count, 'ensemble', args.periods)
        if result:
            return [result]
        return []

    def _handle_duplex_method(self, args) -> list:
        """
        处理胆拖投注预测
        
        Args:
            args: 命令行参数对象
            
        Returns:
            list: 预测结果列表
        """
        result = self.predictors['compound'].predict_duplex(
            periods=args.periods,
            front_dan_count=getattr(args, 'front_dan', 2),
            back_dan_count=getattr(args, 'back_dan', 1),
            front_tuo_count=getattr(args, 'front_tuo', 6),
            back_tuo_count=getattr(args, 'back_tuo', 4)
        )
        if result:
            return [result]
        return []

    def _handle_integration_methods(self, args) -> list:
        """
        处理增强预测方法
        
        Args:
            args: 命令行参数对象
            
        Returns:
            list: 预测结果列表
        """
        try:
            from improvements.integration import get_integrator
            integrator = get_integrator()
            
            if args.method == 'transformer':
                return integrator.transformer_predict(args.count, args.periods)
            elif args.method == 'gan':
                return integrator.gan_predict(args.count, args.periods)
            elif args.method == 'stacking':
                return integrator.stacking_predict(args.count, args.periods)
            elif args.method == 'adaptive_ensemble':
                return integrator.adaptive_ensemble_predict(args.count, args.periods)
            elif args.method == 'ultimate_ensemble':
                return integrator.ultimate_ensemble_predict(args.count, args.periods)
        except ImportError:
            print(f"{OutputStatus.ERROR} 增强预测模块未找到，请确保improvements目录存在且包含所需文件")
        except Exception as e:
            print(f"{OutputStatus.ERROR} 增强预测失败: {e}")
        
        return []

    def _handle_markov_custom(self, args) -> list:
        """
        处理马尔可夫自定义期数预测
        
        Args:
            args: 命令行参数对象
            
        Returns:
            list: 预测结果列表
        """
        analysis_periods = getattr(args, 'analysis_periods', 300)
        predict_periods = getattr(args, 'predict_periods', 1)
        return self.predictors['advanced'].markov_predict_custom(
            count=args.count,
            analysis_periods=analysis_periods,
            predict_periods=predict_periods
        )

    def _handle_mixed_strategy(self, args) -> list:
        """
        处理混合策略预测
        
        Args:
            args: 命令行参数对象
            
        Returns:
            list: 预测结果列表
        """
        strategy = getattr(args, 'strategy', 'balanced')
        return self.predictors['advanced'].mixed_strategy_predict(
            count=args.count,
            strategy=strategy,
            periods=args.periods
        )

    def _handle_highly_integrated(self, args) -> list:
        """
        处理高度集成复式预测
        
        Args:
            args: 命令行参数对象
            
        Returns:
            list: 预测结果列表
        """
        import platform
        
        front_count = getattr(args, 'front_count', 10)
        back_count = getattr(args, 'back_count', 5)
        integration_level = getattr(args, 'integration_level', 'ultimate')
        
        # Windows 不支持 signal.SIGALRM，使用不同的超时机制
        if platform.system() != 'Windows':
            return self._handle_highly_integrated_unix(args, front_count, back_count, integration_level)
        else:
            return self._handle_highly_integrated_windows(args, front_count, back_count, integration_level)

    def _handle_highly_integrated_unix(self, args, front_count, back_count, integration_level) -> list:
        """
        Unix系统下的高度集成预测（带超时）
        """
        try:
            import signal
            def timeout_handler(signum, frame):
                raise TimeoutError("高度集成预测超时")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(45)  # 45秒超时
            
            result = self.predictors['compound'].predict_highly_integrated_compound(
                front_count=front_count,
                back_count=back_count,
                integration_level=integration_level,
                periods=args.periods
            )
            
            signal.alarm(0)  # 取消超时
            
            if result:
                return [result]
            return []
        except (TimeoutError, Exception) as e:
            try:
                import signal
                signal.alarm(0)  # 确保取消超时
            except Exception:
                pass
            print(f"{OutputStatus.WARNING} 高度集成预测超时或失败: {e}")
            print(f"{OutputStatus.INFO} 回退到复式预测...")
            return self._fallback_to_compound(args)

    def _handle_highly_integrated_windows(self, args, front_count, back_count, integration_level) -> list:
        """
        Windows系统下的高度集成预测
        """
        try:
            result = self.predictors['compound'].predict_highly_integrated_compound(
                front_count=front_count,
                back_count=back_count,
                integration_level=integration_level,
                periods=args.periods
            )
            if result:
                return [result]
            return []
        except Exception as e:
            print(f"{OutputStatus.WARNING} 高度集成预测失败: {e}")
            print(f"{OutputStatus.INFO} 回退到复式预测...")
            return self._fallback_to_compound(args)

    def _fallback_to_compound(self, args) -> list:
        """
        回退到复式预测
        """
        result = self.predictors['compound'].predict_compound(
            front_count=8,
            back_count=4,
            method='ensemble',
            periods=args.periods
        )
        if result:
            return [result]
        return []

    def _handle_advanced_integration(self, args) -> list:
        """
        处理高级集成分析预测
        
        Args:
            args: 命令行参数对象
            
        Returns:
            list: 预测结果列表
        """
        integration_type = getattr(args, 'integration_type', 'comprehensive')
        return self.predictors['advanced'].advanced_integration_predict(
            count=args.count,
            integration_type=integration_type,
            periods=args.periods
        )

    def _handle_nine_models_compound(self, args) -> list:
        """
        处理9种数学模型复式预测
        
        Args:
            args: 命令行参数对象
            
        Returns:
            list: 预测结果列表
        """
        front_count = getattr(args, 'front_count', 8)
        back_count = getattr(args, 'back_count', 4)
        result = self.predictors['advanced'].nine_models_compound_predict(
            front_count=front_count,
            back_count=back_count,
            analysis_periods=args.periods
        )
        if result:
            return [result]
        return []

    def _handle_markov_compound(self, args) -> list:
        """
        处理马尔可夫链复式预测
        
        Args:
            args: 命令行参数对象
            
        Returns:
            list: 预测结果列表
        """
        front_count = getattr(args, 'front_count', 8)
        back_count = getattr(args, 'back_count', 4)
        markov_periods = args.periods  # 使用新的periods参数
        result = self.predictors['advanced'].markov_compound_predict(
            front_count=front_count,
            back_count=back_count,
            analysis_periods=markov_periods
        )
        if result:
            return [result]
        return []

    def _handle_enhanced_markov(self, args) -> list:
        """
        处理增强版马尔可夫链预测
        
        Args:
            args: 命令行参数对象
            
        Returns:
            list: 预测结果列表
        """
        try:
            from improvements.enhanced_markov import get_markov_predictor
            
            markov_periods = args.periods  # 使用新的periods参数
            
            if args.method == 'markov_2nd':
                return self._handle_markov_2nd(args, markov_periods)
            
            elif args.method == 'markov_3rd':
                return self._handle_markov_3rd(args, markov_periods)
            
            elif args.method == 'adaptive_markov':
                print(f"{OutputStatus.INFO} 自适应马尔可夫链预测...")
                markov_predictor = get_markov_predictor()
                return markov_predictor.adaptive_order_markov_predict(
                    count=args.count, 
                    periods=markov_periods
                )
        
        except ImportError:
            print(f"{OutputStatus.ERROR} 增强版马尔可夫链模块未找到，请确保improvements目录存在且包含所需文件")
        except Exception as e:
            print(f"{OutputStatus.ERROR} 增强版马尔可夫链预测失败: {e}")
        
        return []

    def _handle_markov_2nd(self, args, markov_periods) -> list:
        """
        处理二阶马尔可夫链预测
        """
        from improvements.enhanced_markov import get_markov_predictor
        
        print(f"{OutputStatus.INFO} 二阶马尔可夫链预测 (分析{markov_periods}期数据)...")
        print(f"{OutputStatus.INFO} 构建二阶状态转移矩阵...")
        print(f"{OutputStatus.INFO} 概率计算: 基于历史数据计算转移概率")
        print(f"{OutputStatus.INFO} 矩阵计算: 构建复合状态转移矩阵")
        
        markov_predictor = get_markov_predictor()
        
        # 获取二阶马尔可夫分析结果
        markov_analyzer = markov_predictor.analyzer
        analysis_result = markov_analyzer.multi_order_markov_analysis(markov_periods, max_order=2)
        
        if analysis_result and 'orders' in analysis_result and 2 in analysis_result['orders']:
            order_2_result = analysis_result['orders'][2]
            front_stats = order_2_result.get('front_stats', {})
            back_stats = order_2_result.get('back_stats', {})
            
            print(f"{OutputStatus.OK} 二阶状态转移矩阵构建完成:")
            print(f"   概率计算: 前区转移概率数 {front_stats.get('total_transitions', 0)}")
            print(f"   矩阵计算: 前区状态数 {front_stats.get('unique_states', 0)}")
            print(f"   概率计算: 后区转移概率数 {back_stats.get('total_transitions', 0)}")
            print(f"   矩阵计算: 后区状态数 {back_stats.get('unique_states', 0)}")
            print(f"   最大转移概率: 前区 {front_stats.get('max_probability', 0):.4f}, 后区 {back_stats.get('max_probability', 0):.4f}")
        
        results = markov_predictor.multi_order_markov_predict(
            count=args.count,
            periods=markov_periods,
            order=2
        )
        return [{'front_balls': r[0], 'back_balls': r[1], 'method': 'markov_2nd', 'confidence': 0.85, 'order': 2} for r in results]

    def _handle_markov_3rd(self, args, markov_periods) -> list:
        """
        处理三阶马尔可夫链预测
        """
        from improvements.enhanced_markov import get_markov_predictor
        
        print(f"{OutputStatus.INFO} 三阶马尔可夫链预测 (分析{markov_periods}期数据)...")
        print(f"{OutputStatus.INFO} 构建三阶状态转移矩阵...")
        print(f"{OutputStatus.INFO} 状态转移显示: 完整的状态转移矩阵构建和统计信息")
        print(f"{OutputStatus.INFO} 超高阶建模: 考虑前三期状态的复杂依赖关系")
        
        markov_predictor = get_markov_predictor()
        
        # 获取三阶马尔可夫分析结果
        markov_analyzer = markov_predictor.analyzer
        analysis_result = markov_analyzer.multi_order_markov_analysis(markov_periods, max_order=3)
        
        if analysis_result and 'orders' in analysis_result and 3 in analysis_result['orders']:
            order_3_result = analysis_result['orders'][3]
            front_stats = order_3_result.get('front_stats', {})
            back_stats = order_3_result.get('back_stats', {})
            
            print(f"{OutputStatus.OK} 三阶状态转移矩阵构建完成:")
            print(f"  前区状态数: {front_stats.get('unique_states', 0)}")
            print(f"  前区转移概率数: {front_stats.get('total_transitions', 0)}")
            print(f"  前区最大转移概率: {front_stats.get('max_probability', 0):.4f}")
            print(f"  后区状态数: {back_stats.get('unique_states', 0)}")
            print(f"  后区转移概率数: {back_stats.get('total_transitions', 0)}")
            print(f"  后区最大转移概率: {back_stats.get('max_probability', 0):.4f}")
        
        results = markov_predictor.multi_order_markov_predict(
            count=args.count,
            periods=markov_periods,
            order=3
        )
        return [{'front_balls': r[0], 'back_balls': r[1], 'method': 'markov_3rd', 'confidence': 0.9, 'order': 3} for r in results]


    def _display_prediction_results(self, predictions, args):
        """
        显示预测结果
        
        Args:
            predictions: 预测结果列表
            args: 命令行参数对象
        """
        print(f"{OutputStatus.OK} 预测完成!")
        print(f"\n{OutputStatus.INFO} 预测结果:")
        
        for i, pred in enumerate(predictions):
            self._display_single_prediction(i, pred, args)
        
        # 保存预测结果
        if hasattr(args, 'save') and args.save:
            self._save_predictions(predictions, args)

    def _output_predictions_json(self, predictions, args, mode='traditional'):
        """以稳定 JSON 协议输出预测结果。"""
        normalized = []
        for item in predictions or []:
            if isinstance(item, dict):
                front = item.get('front_balls', item.get('front', []))
                back = item.get('back_balls', item.get('back', []))
                method = item.get('method', getattr(args, 'method', 'unknown'))
                confidence = item.get('confidence')
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                front, back = item
                method = getattr(args, 'method', 'unknown')
                confidence = None
            else:
                continue

            try:
                front_norm = [int(x) for x in front]
                back_norm = [int(x) for x in back]
            except Exception:
                continue

            if len(front_norm) != 5 or len(back_norm) != 2:
                continue

            normalized.append(
                {
                    'front_balls': sorted(front_norm),
                    'back_balls': sorted(back_norm),
                    'method': method,
                    'confidence': confidence,
                }
            )

        payload = {
            'mode': mode,
            'method': getattr(args, 'method', 'unknown'),
            'periods': getattr(args, 'periods', None),
            'count': getattr(args, 'count', None),
            'predictions': normalized,
        }
        print(json.dumps(payload, ensure_ascii=False))

    def _output_compound_json(self, compound_result, args):
        """以 JSON 协议输出复式预测结果。"""
        payload = {
            'mode': 'compound',
            'method': getattr(args, 'method', 'compound'),
            'periods': getattr(args, 'periods', None),
            'count': getattr(args, 'count', None),
            'compound': {
                'front_balls': list(getattr(compound_result, 'front_balls', [])),
                'back_balls': list(getattr(compound_result, 'back_balls', [])),
                'front_count': getattr(compound_result, 'front_count', None),
                'back_count': getattr(compound_result, 'back_count', None),
                'total_combinations': getattr(compound_result, 'total_combinations', None),
                'total_cost': getattr(compound_result, 'total_cost', None),
                'confidence': getattr(compound_result, 'confidence', None),
                'method': getattr(compound_result, 'method', getattr(args, 'method', 'compound')),
            },
            'predictions': [],
        }
        print(json.dumps(payload, ensure_ascii=False))

    def _display_single_prediction(self, index, pred, args):
        """
        显示单个预测结果
        
        Args:
            index: 预测索引
            pred: 预测结果
            args: 命令行参数对象
        """
        # 处理不同格式的预测结果
        if isinstance(pred, tuple) and len(pred) == 2:
            # 标准元组格式: (前区号码, 后区号码)
            front_balls, back_balls = pred
            front_str = ' '.join([str(b).zfill(2) for b in front_balls])
            back_str = ' '.join([str(b).zfill(2) for b in back_balls])
            print(f"  第 {index+1} 注: {front_str} + {back_str} (方法: {args.method}, 置信度: 0.500)")
            return
        
        # 字典格式的预测结果
        if not isinstance(pred, dict):
            print(f"  第 {index+1} 注: 格式错误 - {type(pred)}")
            return
        
        if pred.get('front_dan'):
            self._display_duplex_prediction(index, pred)
        elif pred.get('front_count'):
            self._display_compound_prediction(index, pred)
        elif pred.get('overall_stability'):
            self._display_markov_custom_prediction(pred)
        elif pred.get('strategy'):
            self._display_strategy_prediction(pred)
        elif pred.get('integration_level'):
            self._display_integration_prediction(pred)
        elif pred.get('integration_type'):
            self._display_integration_type_prediction(pred)
        elif pred.get('method') == 'nine_mathematical_models':
            self._display_nine_models_prediction(pred)
        elif pred.get('method') == 'nine_models_compound':
            self._display_nine_models_compound_prediction(pred)
        elif pred.get('method') and 'compound' in pred['method']:
            self._display_generic_compound_prediction(pred)
        elif pred.get('method') == 'markov_compound':
            self._display_markov_compound_prediction(pred)
        else:
            self._display_simple_prediction(index, pred, args)

    def _display_duplex_prediction(self, index, pred):
        """显示胆拖投注结果"""
        front_dan_str = ' '.join([str(b).zfill(2) for b in pred['front_dan']])
        front_tuo_str = ' '.join([str(b).zfill(2) for b in pred['front_tuo']])
        back_dan_str = ' '.join([str(b).zfill(2) for b in pred['back_dan']])
        back_tuo_str = ' '.join([str(b).zfill(2) for b in pred['back_tuo']])
        
        print(f"  第 {index+1} 注胆拖:")
        print(f"    前区: {front_dan_str} + ({front_tuo_str})")
        print(f"    后区: {back_dan_str} + ({back_tuo_str})")
        print(f"    总组合数: {pred['total_combinations']} 注")
        print(f"    总投注额: {pred['total_cost']} 元")

    def _display_compound_prediction(self, index, pred):
        """显示复式投注结果"""
        front_str = ' '.join([str(b).zfill(2) for b in pred['front_balls']])
        back_str = ' '.join([str(b).zfill(2) for b in pred['back_balls']])
        
        method_name = pred.get('method', 'compound').replace('_', ' ').title()
        print(f"  第 {index+1} 注复式 ({method_name}): {front_str} + {back_str}")
        print(f"    前区: {pred['front_count']} 个号码")
        print(f"    后区: {pred['back_count']} 个号码")
        print(f"    总组合数: {pred['total_combinations']} 注")
        print(f"    总投注额: {pred['total_cost']} 元")
        print(f"    置信度: {pred.get('confidence', 0.5):.3f}")
        
        # 显示特定方法的详细信息
        if pred.get('method') == 'nine_models_compound':
            self._display_nine_models_details(pred)
        elif pred.get('method') == 'markov_compound':
            self._display_markov_details(pred)
        elif pred.get('integration_level'):
            print(f"    集成级别: {pred['integration_level']}")
            print(f"    使用算法: {len(pred.get('algorithms_used', []))} 种")

    def _display_nine_models_details(self, pred):
        """显示9种数学模型详细信息"""
        if 'models_used' in pred:
            print(f"    使用模型: {len(pred['models_used'])} 种")
        if 'model_details' in pred:
            details = pred['model_details']
            print(f"    统计学权重: {details.get('statistical_score', 0):.3f}")
            print(f"    概率论权重: {details.get('probability_score', 0):.3f}")
            print(f"    马尔可夫权重: {details.get('markov_score', 0):.3f}")
            print(f"    贝叶斯权重: {details.get('bayesian_score', 0):.3f}")

    def _display_markov_details(self, pred):
        """显示马尔可夫详细信息"""
        print(f"    分析期数: {pred.get('analysis_periods', 500)}")
        if 'markov_details' in pred:
            details = pred['markov_details']
            print(f"    转移矩阵规模: {details.get('transition_matrix_size', 0)}")
            print(f"    状态数量: {details.get('state_count', 0)}")
            print(f"    预测准确性: {details.get('prediction_accuracy', 0):.3f}")

    def _display_markov_custom_prediction(self, pred):
        """显示马尔可夫自定义预测结果"""
        front_str = ' '.join([str(b).zfill(2) for b in pred['front_balls']])
        back_str = ' '.join([str(b).zfill(2) for b in pred['back_balls']])
        
        print(f"  第 {pred['index']} 注 (期 {pred['period']}): {front_str} + {back_str}")
        print(f"    稳定性得分: {pred['overall_stability']:.3f}")
        print(f"    前区稳定性: {pred['front_stability']:.3f}")
        print(f"    后区稳定性: {pred['back_stability']:.3f}")
        print(f"    分析期数: {pred['analysis_periods']}")

    def _display_strategy_prediction(self, pred):
        """显示混合策略预测结果"""
        front_str = ' '.join([str(b).zfill(2) for b in pred['front_balls']])
        back_str = ' '.join([str(b).zfill(2) for b in pred['back_balls']])
        
        print(f"  第 {pred['index']} 注 ({pred['strategy']}策略): {front_str} + {back_str}")
        print(f"    风险等级: {pred['risk_level']}")
        print(f"    策略描述: {pred['description']}")
        print(f"    权重配置: {pred['weights']}")

    def _display_integration_prediction(self, pred):
        """显示高度集成复式预测结果"""
        front_str = ' '.join([str(b).zfill(2) for b in pred['front_balls']])
        back_str = ' '.join([str(b).zfill(2) for b in pred['back_balls']])
        
        print(f"  高度集成复式 ({pred['integration_level']}级): {front_str} + {back_str}")
        print(f"    前区: {pred['front_count']} 个号码")
        print(f"    后区: {pred['back_count']} 个号码")
        print(f"    总组合数: {pred['total_combinations']} 注")
        print(f"    总投注额: {pred['total_cost']} 元")
        print(f"    集成置信度: {pred['confidence']:.3f}")
        print(f"    使用算法: {len(pred['algorithms_used'])} 种")
        if 'candidate_scores' in pred:
            print(f"    前区热门: {list(pred['candidate_scores']['front_top10'].keys())[:5]}")
            print(f"    后区热门: {list(pred['candidate_scores']['back_top8'].keys())[:3]}")

    def _display_integration_type_prediction(self, pred):
        """显示高级集成分析预测结果"""
        front_str = ' '.join([str(b).zfill(2) for b in pred['front_balls']])
        back_str = ' '.join([str(b).zfill(2) for b in pred['back_balls']])
        
        print(f"  第 {pred['index']} 注 ({pred['integration_type']}集成): {front_str} + {back_str}")
        print(f"    集成类型: {pred['integration_type']}")
        print(f"    分析方法: {pred['method']}")
        print(f"    置信度: {pred['confidence']:.3f}")
        if 'analysis_source' in pred:
            print(f"    分析时间: {pred['analysis_source']}")

    def _display_nine_models_prediction(self, pred):
        """显示9种数学模型预测结果"""
        front_str = ' '.join([str(b).zfill(2) for b in pred['front_balls']])
        back_str = ' '.join([str(b).zfill(2) for b in pred['back_balls']])
        
        print(f"  第 {pred['index']} 注 (9种数学模型): {front_str} + {back_str}")
        print(f"    分析方法: {pred['method']}")
        print(f"    置信度: {pred['confidence']:.3f}")
        if 'models_used' in pred:
            print(f"    使用模型: {len(pred['models_used'])} 种")
        if 'model_consensus' in pred:
            print(f"    模型一致性: {pred['model_consensus']:.3f}")
        if 'analysis_timestamp' in pred:
            print(f"    分析时间: {pred['analysis_timestamp']}")

    def _display_nine_models_compound_prediction(self, pred):
        """显示9种数学模型复式预测结果"""
        front_str = ' '.join([str(b).zfill(2) for b in pred['front_balls']])
        back_str = ' '.join([str(b).zfill(2) for b in pred['back_balls']])
        
        print(f"  9种数学模型复式: {front_str} + {back_str}")
        print(f"    前区: {pred['front_count']} 个号码")
        print(f"    后区: {pred['back_count']} 个号码")
        print(f"    总组合数: {pred['total_combinations']} 注")
        print(f"    总投注额: {pred['total_cost']} 元")
        print(f"    置信度: {pred['confidence']:.3f}")
        self._display_nine_models_details(pred)

    def _display_generic_compound_prediction(self, pred):
        """显示通用复式预测结果"""
        front_str = ' '.join([str(b).zfill(2) for b in pred['front_balls']])
        back_str = ' '.join([str(b).zfill(2) for b in pred['back_balls']])
        
        method_name = pred['method'].replace('_', ' ').title()
        print(f"  {method_name}: {front_str} + {back_str}")
        print(f"    前区: {pred['front_count']} 个号码")
        print(f"    后区: {pred['back_count']} 个号码")
        print(f"    总组合数: {pred['total_combinations']} 注")
        print(f"    总投注额: {pred['total_cost']} 元")
        print(f"    置信度: {pred['confidence']:.3f}")

    def _display_markov_compound_prediction(self, pred):
        """显示马尔可夫链复式预测结果"""
        front_str = ' '.join([str(b).zfill(2) for b in pred['front_balls']])
        back_str = ' '.join([str(b).zfill(2) for b in pred['back_balls']])
        
        print(f"  马尔可夫链复式: {front_str} + {back_str}")
        print(f"    前区: {pred['front_count']} 个号码")
        print(f"    后区: {pred['back_count']} 个号码")
        print(f"    总组合数: {pred['total_combinations']} 注")
        print(f"    总投注额: {pred['total_cost']} 元")
        print(f"    置信度: {pred['confidence']:.3f}")
        self._display_markov_details(pred)

    def _display_simple_prediction(self, index, pred, args):
        """显示简单单式投注结果"""
        front_str = ' '.join([str(b).zfill(2) for b in pred['front_balls']])
        back_str = ' '.join([str(b).zfill(2) for b in pred['back_balls']])
        method = pred.get('method', args.method)
        confidence = pred.get('confidence', 0.5)
        
        print(f"  第 {index+1} 注: {front_str} + {back_str} (方法: {method}, 置信度: {confidence:.3f})")

    def _save_predictions(self, predictions, args):
        """
        保存预测结果到文件
        
        Args:
            predictions: 预测结果列表
            args: 命令行参数对象
        """
        import json
        
        # 确保输出目录存在
        output_dir = "output/predictions"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.save.endswith('.json'):
            filename = os.path.join(output_dir, args.save)
        else:
            filename = os.path.join(output_dir, f"predictions_{args.method}_{timestamp}.json")
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"{OutputStatus.INFO} 预测结果已保存: {filename}")

    def run_predict_command(self, args):
        """处理预测命令"""
        self._load_predictors()
        
        # 参数验证
        is_valid, error_msg, acceleration_config = self._validate_predict_args(args)
        if not is_valid:
            print(f"{OutputStatus.ERROR} {error_msg}")
            return
        
        print(f"{OutputStatus.INFO} 开始{args.method}预测 (分析期数: {args.periods}, 生成注数: {args.count})...")
        
        predictions = []
        
        # 检查是否可以使用增强功能或深度学习方法
        use_enhanced = self.enhanced_available and args.method == 'enhanced' and not (hasattr(args, 'compound') and args.compound)
        use_deep_learning = args.method in REQUIRES_DEEP_LEARNING
        
        # 增强预测模式
        if use_enhanced:
            success, predictions = self._handle_enhanced_prediction(args, acceleration_config)
            if success:
                return
        
        # 深度学习预测（独立于增强功能，但不在复式预测模式下）
        elif use_deep_learning and not (hasattr(args, 'compound') and args.compound):
            success, predictions = self._handle_deep_learning_prediction(args, acceleration_config)
            if success:
                return
        
        try:
            # 复式预测模式
            if hasattr(args, 'compound') and args.compound:
                compound_result = self._handle_compound_prediction(args)
                if compound_result:
                    if getattr(args, 'json_output', False):
                        self._output_compound_json(compound_result, args)
                    else:
                        self._display_compound_result(compound_result)
                    return
                # 如果复式预测失败，会打印回退消息并继续到单式预测
            
            # 传统预测方法
            predictions = self._handle_traditional_prediction(args, acceleration_config)
            
            # 显示预测结果
            if predictions:
                if getattr(args, 'json_output', False):
                    self._output_predictions_json(predictions, args, mode='traditional')
                else:
                    self._display_prediction_results(predictions, args)
            else:
                print(f"{OutputStatus.WARNING} 没有生成预测结果")
        
        except Exception as e:
            logger_manager.error(f"预测失败: {e}")
            print(f"{OutputStatus.ERROR} 预测失败: {e}")

    def _display_predictions(self, predictions, method):
        """显示预测结果"""
        if not predictions:
            print("ERROR 没有生成预测结果")
            return

        print(f"OK {method.upper()}预测完成")
        print("=" * 50)

        if isinstance(predictions, list):
            for i, pred in enumerate(predictions, 1):
                if isinstance(pred, tuple) and len(pred) == 2:
                    front, back = pred
                    print(f"第{i}注: 前区 {front} 后区 {back}")
                elif isinstance(pred, dict):
                    if 'front' in pred and 'back' in pred:
                        print(f"第{i}注: 前区 {pred['front']} 后区 {pred['back']}")
                    else:
                        print(f"第{i}注: {pred}")
                else:
                    print(f"第{i}注: {pred}")
        else:
            print(f"预测结果: {predictions}")

        print("=" * 50)

    def _display_enhanced_predictions(self, predictions, method):
        """显示增强预测结果"""
        if not predictions:
            print("ERROR 没有生成预测结果")
            return

        print(f"OK {method.upper()}深度学习预测完成")
        print("=" * 60)

        for i, pred in enumerate(predictions, 1):
            if isinstance(pred, dict):
                # 兼容多种字段名格式
                front = pred.get('front', pred.get('front_balls', []))
                back = pred.get('back', pred.get('back_balls', []))
                confidence = pred.get('confidence', 0.0)
                pred_method = pred.get('method', method)

                if front and back:
                    print(f"第{i}注 [{pred_method}]:")
                    print(f"  前区: {' '.join(f'{n:02d}' for n in front)}")
                    print(f"  后区: {' '.join(f'{n:02d}' for n in back)}")
                    print(f"  置信度: {confidence:.1%}")
                    print()
                else:
                    print(f"第{i}注: 数据格式异常 - {pred}")
            elif isinstance(pred, (list, tuple)) and len(pred) == 2:
                # 处理 (front_balls, back_balls) 格式
                front, back = pred
                print(f"第{i}注:")
                print(f"  前区: {' '.join(f'{n:02d}' for n in front)}")
                print(f"  后区: {' '.join(f'{n:02d}' for n in back)}")
                print()
            else:
                print(f"第{i}注: {pred}")

        print("=" * 60)
        print(f" 使用{method.upper()}深度学习算法生成 {len(predictions)} 注预测")
        print(" 深度学习模型已自动训练并优化参数")

    def run_learn_command(self, args):
        """处理学习命令"""
        self._load_adaptive_predictor()
        
        print(f" 开始自适应学习 (算法: {args.algorithm})...")
        print(f" 起始期数: {args.start}, 测试期数: {args.test}")
        
        try:
            # 设置多臂老虎机算法
            self.adaptive_predictor.bandit.algorithm = args.algorithm
            
            # 进行学习
            results = self.adaptive_predictor.enhanced_adaptive_learning(
                start_period=args.start,
                test_periods=args.test
            )
            
            if results:
                print("OK 自适应学习完成!")
                print(f" 中奖率: {results['win_rate']:.3f}")
                print(f" 平均得分: {results['average_score']:.2f}")
                print(f" 总测试期数: {results['total_periods']}")
                
                # 显示预测器性能
                print("\n 预测器性能排名:")
                bandit_values = results['bandit_final_values']
                predictor_names = self.adaptive_predictor.predictor_names
                
                performance_ranking = sorted(
                    zip(predictor_names, bandit_values), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                for i, (name, value) in enumerate(performance_ranking[:5]):
                    print(f"  {i+1}. {name}: {value:.3f}")
                
                # 保存学习结果
                output_dir = "output/learning"
                os.makedirs(output_dir, exist_ok=True)

                if args.save:
                    if args.save.endswith('.json'):
                        filename = os.path.join(output_dir, args.save)
                    else:
                        filename = os.path.join(output_dir, f"{args.save}.json")
                else:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(output_dir, f"learning_{args.algorithm}_{timestamp}.json")

                saved_file = self.adaptive_predictor.save_enhanced_results(filename)
                if saved_file:
                    print(f" 学习结果已保存: {saved_file}")
            else:
                print("ERROR 自适应学习失败")
        
        except Exception as e:
            logger_manager.error("学习失败", e)
            print(f"ERROR 学习失败: {e}")
    
    def run_smart_command(self, args):
        """处理智能预测命令"""
        self._load_adaptive_predictor()

        # 确定预测类型
        if args.compound:
            print(f" 智能复式预测 ({args.front_count}+{args.back_count})...")
        elif args.duplex:
            print(f" 智能胆拖预测 (前区{args.front_dan}胆{args.front_tuo}拖, 后区{args.back_dan}胆{args.back_tuo}拖)...")
        else:
            print(f" 智能预测 (注数: {args.count})...")

        try:
            # 加载学习结果
            if args.load:
                if self.adaptive_predictor.load_enhanced_results(args.load):
                    print(f"OK 已加载学习结果: {args.load}")
                else:
                    print(f"ERROR 加载学习结果失败: {args.load}")
                    return
            else:
                print("WARNING  未加载学习结果，使用默认配置")

            # 根据类型生成预测
            if args.compound:
                # 复式投注预测
                result = self.adaptive_predictor.smart_predict_compound(
                    front_count=args.front_count,
                    back_count=args.back_count,
                    periods=args.periods
                )

                if result:
                    print("OK 智能复式预测完成!")
                    print("\n 智能复式预测结果:")

                    front_str = ' '.join([str(b).zfill(2) for b in result['front_balls']])
                    back_str = ' '.join([str(b).zfill(2) for b in result['back_balls']])

                    print(f"  复式号码: {front_str} + {back_str}")
                    print(f"  前区: {result['front_count']} 个号码")
                    print(f"  后区: {result['back_count']} 个号码")
                    print(f"  总组合数: {result['total_combinations']} 注")
                    print(f"  总投注额: {result['total_cost']} 元")
                    print(f"  置信度: {result['confidence']:.3f}")
                    print(f"  使用预测器: {result['top_predictors']}")
                else:
                    print("ERROR 智能复式预测失败")

            elif args.duplex:
                # 胆拖投注预测
                result = self.adaptive_predictor.smart_predict_duplex(
                    front_dan_count=args.front_dan,
                    back_dan_count=args.back_dan,
                    front_tuo_count=args.front_tuo,
                    back_tuo_count=args.back_tuo,
                    periods=args.periods
                )

                if result:
                    print("OK 智能胆拖预测完成!")
                    print("\n 智能胆拖预测结果:")

                    front_dan_str = ' '.join([str(b).zfill(2) for b in result['front_dan']])
                    front_tuo_str = ' '.join([str(b).zfill(2) for b in result['front_tuo']])
                    back_dan_str = ' '.join([str(b).zfill(2) for b in result['back_dan']])
                    back_tuo_str = ' '.join([str(b).zfill(2) for b in result['back_tuo']])

                    print(f"  前区胆码: {front_dan_str}")
                    print(f"  前区拖码: {front_tuo_str}")
                    print(f"  后区胆码: {back_dan_str}")
                    print(f"  后区拖码: {back_tuo_str}")
                    print(f"  总组合数: {result['total_combinations']} 注")
                    print(f"  总投注额: {result['total_cost']} 元")
                    print(f"  置信度: {result['confidence']:.3f}")
                    print(f"  最优预测器: {result['best_predictor']}")
                else:
                    print("ERROR 智能胆拖预测失败")

            else:
                # 单式投注预测
                predictions = self.adaptive_predictor.generate_enhanced_prediction(args.count)

                if predictions:
                    print("OK 智能预测完成!")
                    print("\n 智能预测结果:")

                    for pred in predictions:
                        front_str = ' '.join([str(b).zfill(2) for b in pred['front_balls']])
                        back_str = ' '.join([str(b).zfill(2) for b in pred['back_balls']])
                        predictor = pred['predictor_used']
                        confidence = pred['confidence']
                        expected_reward = pred['expected_reward']

                        print(f"  第 {pred['index']} 注: {front_str} + {back_str}")
                        print(f"    预测器: {predictor}")
                        print(f"    置信度: {confidence:.3f}")
                        print(f"    期望奖励: {expected_reward:.3f}")
                else:
                    print("ERROR 智能预测失败")

        except Exception as e:
            logger_manager.error("智能预测失败", e)
            print(f"ERROR 智能预测失败: {e}")

    def run_optimize_command(self, args):
        """处理参数优化命令"""
        self._load_adaptive_predictor()

        print(f" 参数优化 (测试期数: {args.test_periods}, 优化轮数: {args.rounds})...")

        try:
            # 进行参数优化
            results = self.adaptive_predictor.parameter_optimization(
                test_periods=args.test_periods,
                optimization_rounds=args.rounds
            )

            if results:
                print("OK 参数优化完成!")
                print(f"\n 最佳参数:")
                for param, value in results['best_params'].items():
                    print(f"  {param}: {value}")

                print(f"\n 最佳得分: {results['best_score']:.3f}")

                print(f"\n 优化历史:")
                for history in results['optimization_history'][-5:]:  # 显示最后5轮
                    print(f"  轮次 {history['round']}: 得分 {history['score']:.3f}, 中奖率 {history['win_rate']:.3f}")

                # 保存优化结果
                if args.save:
                    import json

                    output_dir = "output/optimization"
                    os.makedirs(output_dir, exist_ok=True)

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    if args.save.endswith('.json'):
                        filename = os.path.join(output_dir, args.save)
                    else:
                        filename = os.path.join(output_dir, f"optimization_{timestamp}.json")

                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

                    print(f" 优化结果已保存: {filename}")
            else:
                print("ERROR 参数优化失败")

        except Exception as e:
            logger_manager.error("参数优化失败", e)
            print(f"ERROR 参数优化失败: {e}")

    def run_backtest_command(self, args):
        """处理回测命令"""
        self._load_predictors()

        print(f" 开始历史回测 (方法: {args.method})...")
        print(f" 起始期数: {args.start}, 测试期数: {args.test}")

        try:
            from adaptive_learning_modules import AccuracyTracker

            # 创建准确率跟踪器
            tracker = AccuracyTracker()

            # 获取数据
            df = data_manager.get_data()
            if df is None or len(df) < args.start + args.test:
                print("ERROR 数据不足")
                return

            total_predictions = 0
            total_wins = 0
            prize_stats = {}

            print(f" 开始回测...")

            for i in range(args.test):
                period_idx = args.start + i

                if period_idx >= len(df):
                    break

                # 获取当前期的真实开奖号码
                current_row = df.iloc[period_idx]
                actual_front, actual_back = data_manager.parse_balls(current_row)

                # 进行预测
                try:
                    if args.method in ['frequency', 'hot_cold', 'missing']:
                        if args.method == 'frequency':
                            result = self.predictors['traditional'].frequency_predict(1)
                        elif args.method == 'hot_cold':
                            result = self.predictors['traditional'].hot_cold_predict(1)
                        elif args.method == 'missing':
                            result = self.predictors['traditional'].missing_predict(
                                1,
                                mode=getattr(args, 'missing_mode', 'auto')
                            )

                        predicted_front, predicted_back = result[0]

                    elif args.method in ['markov', 'bayesian', 'ensemble']:  # 保持原样，此处是具体预测逻辑
                        if args.method == 'markov':
                            result = self.predictors['advanced'].markov_predict(1)
                        elif args.method == 'bayesian':
                            result = self.predictors['traditional'].bayesian_predict(1)
                        elif args.method == 'ensemble':
                            result = self.predictors['advanced'].ensemble_predict(1)

                        predicted_front, predicted_back = result[0]

                    else:
                        print(f"ERROR 不支持的回测方法: {args.method}")
                        return

                    # 计算中奖情况
                    prize_level, front_hits, back_hits = tracker._calculate_prize_level(
                        predicted_front, predicted_back, actual_front, actual_back
                    )

                    total_predictions += 1
                    if prize_level != "未中奖":
                        total_wins += 1

                    # 统计中奖等级
                    prize_stats[prize_level] = prize_stats.get(prize_level, 0) + 1

                    # 显示进度
                    if (i + 1) % 50 == 0:
                        win_rate = total_wins / total_predictions
                        print(f"  进度: {i+1}/{args.test}, 中奖率: {win_rate:.3f}")

                except Exception as e:
                    logger_manager.error(f"第 {i+1} 期回测失败", e)
                    continue

            # 显示回测结果
            print("OK 回测完成!")
            print(f"\n 回测结果统计:")
            print(f"  总预测期数: {total_predictions}")
            print(f"  中奖期数: {total_wins}")
            print(f"  中奖率: {total_wins/total_predictions:.3f}" if total_predictions > 0 else "  中奖率: 0.000")

            print(f"\n 中奖等级分布:")
            for prize, count in sorted(prize_stats.items()):
                rate = count / total_predictions if total_predictions > 0 else 0
                print(f"  {prize}: {count} 次 ({rate:.3f})")

        except Exception as e:
            logger_manager.error("回测失败", e)
            print(f"ERROR 回测失败: {e}")

    def run_system_command(self, args):
        """处理系统管理命令"""
        if args.system_action == 'cache':
            if args.action == 'info':
                print(" 缓存系统信息:")
                print("=" * 50)

                # 获取智能缓存状态
                try:
                    from analyzer_modules import get_analysis_cache_status
                    cache_status = get_analysis_cache_status()

                    print(" 智能缓存系统:")
                    smart_stats = cache_status.get('smart_cache', {})
                    memory_cache = smart_stats.get('memory_cache', {})
                    file_cache = smart_stats.get('file_cache', {})

                    print(f"  内存缓存: {memory_cache.get('size', 0)}/{memory_cache.get('max_size', 0)} 项")
                    print(f"  文件缓存: {file_cache.get('analysis_files', 0)} 个文件, {file_cache.get('total_size_mb', 0):.2f} MB")
                    print(f"  数据签名: {cache_status.get('data_signature', 'unknown')}")

                    print("\n 传统缓存系统:")
                    old_stats = cache_status.get('old_cache', {})
                    for cache_type in ['models', 'analysis', 'data']:
                        if cache_type in old_stats:
                            info = old_stats[cache_type]
                            print(f"  {cache_type}: {info.get('files', 0)} 个文件, {info.get('size_mb', 0):.2f} MB")

                except Exception as e:
                    print(f"ERROR 获取智能缓存状态失败: {e}")
                    # 回退到传统缓存信息
                    cache_info = cache_manager.get_cache_info()
                    for cache_type in ['models', 'analysis', 'data']:
                        info = cache_info[cache_type]
                        print(f"  {cache_type}: {info['files']} 个文件, {info['size_mb']:.2f} MB")

            elif args.action == 'clear':
                cache_type = getattr(args, 'type', 'all')
                print(f"  清理{cache_type}缓存...")

                try:
                    from analyzer_modules import clear_all_analysis_cache, force_refresh_cache

                    if cache_type == 'analysis' or cache_type == 'all':
                        # 使用智能缓存清理
                        cleared_count = clear_all_analysis_cache()
                        print(f"OK 已清理分析缓存 {cleared_count} 个文件")

                    if cache_type == 'all':
                        # 清理其他类型缓存
                        other_cleared = cache_manager.clear_cache('models') + cache_manager.clear_cache('data')
                        print(f"OK 已清理其他缓存 {other_cleared} 个文件")

                except Exception as e:
                    print(f"ERROR 智能缓存清理失败: {e}")
                    # 回退到传统缓存清理
                    cleared_count = cache_manager.clear_cache(cache_type)
                    print(f"OK 已清理 {cleared_count} 个缓存文件")

            elif args.action == 'refresh':
                print(" 强制刷新缓存...")
                try:
                    from analyzer_modules import force_refresh_cache
                    method_name = getattr(args, 'method', None)
                    cleared_count = force_refresh_cache(method_name)
                    if method_name:
                        print(f"OK 已强制刷新 {method_name} 缓存，删除 {cleared_count} 个缓存项")
                    else:
                        print(f"OK 已强制刷新所有缓存，删除 {cleared_count} 个缓存项")
                except Exception as e:
                    print(f"ERROR 强制刷新缓存失败: {e}")

            elif args.action == 'status':
                print(" 缓存系统状态:")
                print("=" * 50)
                try:
                    from analyzer_modules import get_analysis_cache_status
                    from smart_cache_system import smart_cache_manager

                    cache_status = get_analysis_cache_status()
                    smart_stats = cache_status.get('smart_cache', {})

                    print(f"智能缓存系统: {'OK 已启用' if smart_stats else 'ERROR 未启用'}")
                    print(f"数据版本控制: {'OK 已启用' if cache_status.get('data_signature') else 'ERROR 未启用'}")
                    print(f"内存缓存: {'OK 正常' if smart_stats.get('memory_cache') else 'ERROR 异常'}")
                    print(f"文件缓存: {'OK 正常' if smart_stats.get('file_cache') else 'ERROR 异常'}")

                except Exception as e:
                    print(f"ERROR 获取缓存状态失败: {e}")
                    print("缓存系统状态: ERROR 异常")
    
    def run_compare_command(self, args):
        """处理批量预测对比命令"""
        print(" 批量预测对比功能")
        print("="*60)
        
        # 导入批量对比模块
        try:
            from batch_comparison_module import BatchComparison, BatchComparisonConfig
        except ImportError as e:
            print(f"ERROR 批量对比模块导入失败: {e}")
            return
        
        # 创建配置
        config = BatchComparisonConfig(
            target_issue=args.issue,
            method=args.method,
            analysis_periods=args.periods,
            comparison_times=args.times,
            random_periods=args.random_periods,
            min_random_periods=args.min_periods,
            max_random_periods=args.max_periods,
            export_excel=args.export,
            show_progress=not args.no_progress
        )
        
        # 验证配置
        is_valid, error_msg = config.validate()
        if not is_valid:
            print(f"ERROR 配置验证失败: {error_msg}")
            return
        
        # 显示配置信息
        print(f" 配置信息:")
        print(f"  目标期号: {config.target_issue}")
        print(f"  预测方法: {config.method}")
        if config.random_periods:
            print(f"  分析期数: 随机 ({config.min_random_periods}-{config.max_random_periods}期)")
        else:
            print(f"  分析期数: 固定 {config.analysis_periods}期")
        print(f"  对比次数: {config.comparison_times}次")
        print(f"  导出Excel: {'是' if config.export_excel else '否'}")
        
        # 创建批量对比器并执行
        try:
            batch_comparison = BatchComparison()
            
            # 进度回调函数
            def progress_callback(current, total, message):
                if not args.no_progress:
                    progress = current / total * 100
                    print(f"\r⏳ {message} - {progress:.1f}%", end="", flush=True)
            
            print(f"\n 开始执行批量预测对比...")
            result = batch_comparison.execute(config, progress_callback)
            
            if not args.no_progress:
                print()  # 换行
            
            # 显示结果摘要
            result.print_summary()
            
            # 导出Excel
            if args.export:
                try:
                    filename = result.export_to_excel()
                    print(f"\n Excel文件已导出: {filename}")
                except Exception as e:
                    print(f"\nERROR Excel导出失败: {e}")
                    
        except Exception as e:
            print(f"\nERROR 批量对比执行失败: {e}")
            logger_manager.error(f"批量对比执行失败: {e}")
    
    def run_enhanced_command(self, args):
        """运行增强功能命令"""
        if not self.enhanced_available:
            print("ERROR 增强功能不可用")
            print("请确保已正确安装enhanced_deep_learning模块")
            return

        if args.enhanced_action == 'info':
            # 显示所有信息
            if hasattr(args, 'all') and args.all:
                args.gpu = True
                args.models = True
                args.performance = True

            # 如果没有指定任何参数，显示基本信息
            if not any([getattr(args, 'gpu', False), getattr(args, 'models', False), getattr(args, 'performance', False)]):
                print(" 增强系统信息")
                print("=" * 50)

                info = self.enhanced_system.get_system_info()
                print(f"系统类型: {info['system_type']}")

                if 'platform' in info:
                    platform = info['platform']
                    print(f"操作系统: {platform['os']} {platform['version']}")
                    print(f"架构: {platform['architecture']}")
                    print(f"Python版本: {platform['python_version']}")

                    hardware = info['hardware']
                    print(f"CPU核心: {hardware['cpu_count']}")
                    print(f"内存: {hardware['memory_total_gb']:.1f} GB")
                    print(f"GPU数量: {hardware['gpu_count']}")

                print("=" * 50)
                print(" 使用 --gpu, --models, --performance 或 --all 查看详细信息")
                return

            # 显示GPU信息
            if getattr(args, 'gpu', False):
                try:
                    from enhanced_deep_learning.cli.cli_commands import CommandDefinition
                    dl_commands = CommandDefinition()
                    dl_commands.info_command(args)
                except Exception as e:
                    print(f"ERROR 获取GPU信息失败: {e}")

            # 显示模型信息
            if getattr(args, 'models', False):
                print("\n 模型信息")
                print("=" * 50)
                info = self.enhanced_system.get_system_info()
                if 'models' in info:
                    models = info['models']
                    print(f"已注册模型: {len(models)}")
                    for model_name, model_info in models.items():
                        print(f"  {model_name}: {model_info['version']}")
                else:
                    print("没有可用的模型信息")
                print("=" * 50)

            # 显示性能信息
            if getattr(args, 'performance', False):
                print("\nPERFORMANCE 性能信息")
                print("=" * 50)
                info = self.enhanced_system.get_system_info()
                if 'performance' in info:
                    perf = info['performance']
                    print(f"CPU得分: {perf.get('cpu_score', 'N/A')}")
                    print(f"GPU得分: {perf.get('gpu_score', 'N/A')}")
                    print(f"推荐加速: {perf.get('recommended_acceleration', 'N/A')}")
                else:
                    print("没有可用的性能信息")
                print("=" * 50)

        elif args.enhanced_action == 'test':
            print(" 运行兼容性测试")
            print("=" * 50)

            result = self.enhanced_system.run_compatibility_test()
            if result.get('success'):
                for test in result['test_results']:
                    status_icon = 'OK' if test['status'] == 'passed' else 'ERROR'
                    print(f"{status_icon} {test['name']}: {test['message']} ({test['duration']:.2f}s)")
            else:
                print(f"ERROR 测试失败: {result.get('error', '未知错误')}")

        elif args.enhanced_action == 'predict':
            print(" 增强预测")
            print("=" * 50)

            if not args.data:
                print("ERROR 请提供预测数据 (-d 参数)")
                return

            result = self.enhanced_system.enhanced_predict(args.data, method=args.method)
            if result.get('success'):
                print(f"OK 预测成功")
                print(f"方法: {result['method']}")
                print(f"结果: {result['result']}")
                print(f"已缓存: {result['cached']}")
            else:
                print(f"ERROR 预测失败: {result.get('error', '未知错误')}")

        elif args.enhanced_action == 'visualize':
            print(" 增强可视化")
            print("=" * 50)

            if not args.data:
                print("ERROR 请提供可视化数据 (-d 参数)")
                return

            result = self.enhanced_system.enhanced_visualize(args.data, chart_type=args.type)
            if result.get('success'):
                print(f"OK 可视化成功")
                print(f"图表类型: {result['chart_type']}")
                print(f"结果: {result['result']}")
            else:
                print(f"ERROR 可视化失败: {result.get('error', '未知错误')}")

        else:
            print("ERROR 未知的增强功能操作")
            print("可用操作: info, test, predict, visualize")

    def show_version(self):
        """显示版本信息"""
        print(" 大乐透预测系统")
        print("版本: 2.0.0 Enhanced")
        print("作者: AI Assistant")
        print("更新时间: 2024-12-19")
        print("\n📦 功能模块:")
        print("  OK 数据爬取与管理")
        print("  OK 基础与高级分析")
        print("  OK 多种预测算法")
        print("  OK 自适应学习系统")
        print("  OK 智能预测与回测")
        print("  OK 缓存与日志管理")

        # 显示增强功能状态
        if self.enhanced_available:
            print("\n 增强功能模块:")
            print("  OK 企业级核心架构")
            print("  OK 高级数据处理")
            print("  OK 智能模型注册表")
            print("  OK 增强预测引擎")
            print("  OK 交互式可视化")
            print("  OK 工作流管理")
            print("  OK 跨平台兼容性")
            print("  OK 分布式计算")
            print("  OK 性能优化")
            print("  OK 智能缓存系统")
        else:
            print("\nWARNING 增强功能: 未启用")
            print("  提示: 运行 'python dlt_main.py enhanced info' 查看详情")

    def _process_acceleration_args(self, args):
        """处理加速参数"""
        acceleration_config = {}

        # 检查是否有加速参数
        if not hasattr(args, 'acceleration') or not args.acceleration:
            return None

        acceleration_mode = args.acceleration.lower()

        if acceleration_mode == 'auto':
            # 自动选择最优加速方式
            try:
                from enhanced_deep_learning.performance.enhanced_hardware_accelerator import EnhancedHardwareAccelerator
                accelerator = EnhancedHardwareAccelerator()
                hardware_info = accelerator.detect_hardware()

                if hardware_info.gpu_count > 0 and hardware_info.cuda_available:
                    acceleration_config = {
                        'mode': 'gpu_cuda',
                        'gpu_device': getattr(args, 'gpu_device', 0),
                        'gpu_memory_limit': getattr(args, 'gpu_memory_limit', None),
                        'mixed_precision': getattr(args, 'mixed_precision', False)
                    }
                elif hardware_info.cpu_count > 1:
                    acceleration_config = {
                        'mode': 'cpu_multi',
                        'cpu_threads': getattr(args, 'cpu_threads', hardware_info.cpu_count)
                    }
                else:
                    acceleration_config = {'mode': 'cpu'}

            except ImportError:
                logger_manager.warning("硬件加速器模块不可用，使用CPU单线程")
                acceleration_config = {'mode': 'cpu'}

        elif acceleration_mode == 'cpu':
            acceleration_config = {'mode': 'cpu'}

        elif acceleration_mode == 'cpu_multi':
            cpu_threads = getattr(args, 'cpu_threads', -1)
            if cpu_threads == -1:
                import multiprocessing
                cpu_threads = multiprocessing.cpu_count()
            acceleration_config = {
                'mode': 'cpu_multi',
                'cpu_threads': cpu_threads
            }

        elif acceleration_mode in ['gpu', 'gpu_cuda']:
            acceleration_config = {
                'mode': acceleration_mode,
                'gpu_device': getattr(args, 'gpu_device', 0),
                'gpu_memory_limit': getattr(args, 'gpu_memory_limit', None),
                'mixed_precision': getattr(args, 'mixed_precision', False)
            }

        else:
            logger_manager.warning(f"未知的加速模式: {acceleration_mode}")
            return None

        return acceleration_config

    def _apply_acceleration_config(self, method_name, acceleration_config):
        """应用加速配置到具体方法"""
        if not acceleration_config:
            return {}

        config = {}

        if acceleration_config['mode'] == 'cpu_multi':
            # 为支持并行的方法添加n_jobs参数
            if method_name in ['bayesian', 'clustering', 'markov']:
                config['n_jobs'] = acceleration_config['cpu_threads']

        elif acceleration_config['mode'] in ['gpu', 'gpu_cuda']:
            # 为深度学习方法添加GPU配置
            if method_name in ['lstm', 'transformer', 'gan']:
                config['use_gpu'] = True
                config['gpu_device'] = acceleration_config['gpu_device']
                if acceleration_config.get('gpu_memory_limit'):
                    config['gpu_memory_limit'] = acceleration_config['gpu_memory_limit']
                if acceleration_config.get('mixed_precision'):
                    config['mixed_precision'] = True

        return config


def main():
    """主函数"""
    # ==================== 快速配置检查 ====================
    print(f"{OutputStatus.INFO} 开始系统配置检查...")

    try:
        # 导入路径配置管理器
        try:
            from path_config import get_path_manager
        except ImportError:
            try:
                from core.path_config import get_path_manager
            except ImportError:
                print(f"{OutputStatus.ERROR} 无法导入path_config模块")
                print(f"{OutputStatus.INFO} 请确保path_config.py文件存在于backend/app/core目录")
                return

        pm = get_path_manager()

        # 快速验证关键路径
        critical_checks = [
            ('数据文件', pm.data_file, True),  # True表示必须存在
            ('缓存目录', pm.cache_dir, False),  # False表示可以创建
            ('日志目录', pm.logs_dir, False),
            ('模型目录', pm.models_dir, False),
        ]

        all_passed = True
        for name, path, must_exist in critical_checks:
            if path.exists():
                print(f"{OutputStatus.OK} {name}: {path}")
            elif not must_exist:
                # 尝试创建目录
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    print(f"{OutputStatus.OK} 已创建{name}: {path}")
                except Exception as e:
                    print(f"{OutputStatus.ERROR} 无法创建{name}: {path}")
                    print(f"{OutputStatus.ERROR} 错误: {e}")
                    all_passed = False
            else:
                print(f"{OutputStatus.ERROR} {name}不存在: {path}")
                all_passed = False

        if not all_passed:
            print(f"{OutputStatus.ERROR} 配置检查失败，请修复上述问题后重试")
            return

        print(f"{OutputStatus.OK} 配置检查通过")

    except Exception as e:
        print(f"{OutputStatus.ERROR} 配置检查过程出错: {e}")
        print(f"{OutputStatus.WARNING} 将继续运行，但可能会遇到问题")

    # ==================== 命令行参数解析 ====================
    parser = argparse.ArgumentParser(description="大乐透预测系统 - 优化版")
    
    # 添加子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # ==================== 数据管理命令 ====================
    data_parser = subparsers.add_parser('data', help='数据管理')
    data_subparsers = data_parser.add_subparsers(dest='data_action', help='数据操作')

    # 数据状态
    data_status_parser = data_subparsers.add_parser('status', help='查看数据状态')

    # 最新开奖结果
    data_latest_parser = data_subparsers.add_parser('latest', help='获取最新开奖结果')
    data_latest_parser.add_argument('--compare', action='store_true', help='与用户号码比较')

    # 数据更新
    data_update_parser = data_subparsers.add_parser('update', help='更新数据')
    data_update_parser.add_argument('--source', choices=['zhcw'], default='zhcw', help='数据源')
    data_update_parser.add_argument('--periods', type=int, help='更新指定期数')
    data_update_parser.add_argument('--incremental', action='store_true', help='增量更新（只获取最新数据）')

    # 数据检查
    data_check_parser = data_subparsers.add_parser('check', help='检查数据完整性')
    data_check_parser.add_argument('--detailed', action='store_true', help='显示详细检查信息')
    data_check_parser.add_argument('--fix', action='store_true', help='自动修复发现的问题')

    # ==================== 分析命令 ====================
    analyze_parser = subparsers.add_parser('analyze', help='数据分析')
    analyze_parser.add_argument('-t', '--type', choices=['basic', 'advanced', 'comprehensive'],
                               default='comprehensive', help='分析类型')
    analyze_parser.add_argument('-p', '--periods', type=int, default=500, help='分析期数')
    analyze_parser.add_argument('--report', action='store_true', help='生成分析报告')
    analyze_parser.add_argument('--visualize', action='store_true', help='生成可视化图表')
    analyze_parser.add_argument('--save', help='保存分析结果')

    # ==================== 预测命令 ====================
    predict_parser = subparsers.add_parser('predict', help='号码预测')
    predict_parser.add_argument('-m', '--method',
                               choices=['frequency', 'hot_cold', 'missing', 'markov', 'bayesian',
                                       'ensemble', 'super', 'adaptive', 'compound', 'duplex', 'markov_custom',
                                       'mixed_strategy', 'highly_integrated', 'advanced_integration',
                                       'nine_models', 'nine_models_compound', 'markov_compound',
                                       'lstm', 'transformer', 'gan', 'stacking', 'adaptive_ensemble', 'ultimate_ensemble',
                                       'markov_2nd', 'markov_3rd', 'adaptive_markov', 'enhanced', 'clustering',
                                       'consensus_halving'],
                               default='ensemble', help='预测方法')
    predict_parser.add_argument('--ensemble-method', choices=['stacking', 'weighted', 'adaptive'],
                               default='stacking', help='高级集成方法类型')
    predict_parser.add_argument('-c', '--count', type=int, default=1, help='生成注数 (1-100)')
    predict_parser.add_argument('-p', '--periods', type=int, default=500, help='分析期数 (50-2748，默认500期)')
    predict_parser.add_argument('--missing-mode', choices=['auto', 'legacy', 'enhanced'],
                               default='auto', help='遗漏预测模式 (auto/legacy/enhanced)')
    predict_parser.add_argument('--bayes-mode', choices=['legacy', 'enhanced'],
                               default=None, help='贝叶斯预测模式 (legacy/enhanced，默认读取配置)')
    predict_parser.add_argument('--front-count', type=int, default=8, help='复式投注前区号码数量')
    predict_parser.add_argument('--back-count', type=int, default=4, help='复式投注后区号码数量')
    predict_parser.add_argument('--strategy', choices=['conservative', 'aggressive', 'balanced'],
                               default='balanced', help='混合策略类型')
    predict_parser.add_argument('--integration-level', choices=['high', 'ultimate'],
                               default='ultimate', help='高度集成级别')
    predict_parser.add_argument('--integration-type', choices=['comprehensive', 'markov_bayesian', 'hot_cold_markov', 'multi_dimensional'],
                               default='comprehensive', help='高级集成分析类型')
    # 胆拖投注参数
    predict_parser.add_argument('--front-dan', type=int, default=2, help='前区胆码数量')
    predict_parser.add_argument('--back-dan', type=int, default=1, help='后区胆码数量')
    predict_parser.add_argument('--front-tuo', type=int, default=6, help='前区拖码数量')
    predict_parser.add_argument('--back-tuo', type=int, default=4, help='后区拖码数量')
    predict_parser.add_argument('--save', help='保存预测结果')
    predict_parser.add_argument('--json-output', action='store_true', help='以 JSON 格式输出预测结果（适用于外部系统解析）')

    # ==================== 加速功能参数 ====================
    predict_parser.add_argument('--acceleration', choices=['auto', 'cpu', 'cpu_multi', 'gpu', 'gpu_cuda'],
                               default='auto', help='加速方式选择')
    predict_parser.add_argument('--cpu-threads', type=int, default=-1, help='CPU线程数 (-1表示使用所有核心)')
    predict_parser.add_argument('--gpu-device', type=int, default=0, help='GPU设备ID')
    predict_parser.add_argument('--gpu-memory-limit', type=float, help='GPU内存限制 (GB)')
    predict_parser.add_argument('--mixed-precision', action='store_true', help='启用混合精度训练')
    predict_parser.add_argument('--batch-size-multiplier', type=float, default=1.0, help='批次大小倍数')
    predict_parser.add_argument('--benchmark-hardware', action='store_true', help='运行硬件基准测试')
    predict_parser.add_argument('--fallback-enabled', action='store_true', default=True, help='启用优雅降级')

    # ==================== 训练优化参数 ====================
    predict_parser.add_argument('--auto-epochs', action='store_true', help='启用智能训练轮数')
    predict_parser.add_argument('--min-epochs', type=int, default=10, help='最小训练轮数')
    predict_parser.add_argument('--max-epochs', type=int, default=1000, help='最大训练轮数')
    predict_parser.add_argument('--performance-mode', choices=['low', 'medium', 'high'], default='medium', help='性能模式')
    predict_parser.add_argument('--training-intensity', type=float, default=1.0, help='训练强度倍数')

    # ==================== 复式预测参数 ====================
    predict_parser.add_argument('--compound', action='store_true', help='启用复式预测')
    predict_parser.add_argument('--max-cost', type=int, default=10000, help='最大投注成本 (元)')
    predict_parser.add_argument('--min-confidence', type=float, default=0.5, help='最小置信度阈值')

    # ==================== 自适应学习命令 ====================
    learn_parser = subparsers.add_parser('learn', help='自适应学习')
    learn_parser.add_argument('-s', '--start', type=int, default=100, help='起始期数')
    learn_parser.add_argument('-t', '--test', type=int, default=1000, help='测试期数')
    learn_parser.add_argument('--algorithm', choices=['epsilon_greedy', 'ucb1', 'thompson_sampling'], 
                             default='ucb1', help='多臂老虎机算法')
    learn_parser.add_argument('--save', help='保存学习结果')
    
    # ==================== 智能预测命令 ====================
    smart_parser = subparsers.add_parser('smart', help='智能预测（基于学习结果）')
    smart_parser.add_argument('-c', '--count', type=int, default=1, help='生成注数')
    smart_parser.add_argument('-p', '--periods', type=int, default=500, help='分析期数')
    smart_parser.add_argument('--load', help='加载学习结果文件')
    smart_parser.add_argument('--compound', action='store_true', help='生成复式投注')
    smart_parser.add_argument('--front-count', type=int, default=8, help='复式前区号码数量')
    smart_parser.add_argument('--back-count', type=int, default=4, help='复式后区号码数量')
    smart_parser.add_argument('--duplex', action='store_true', help='生成胆拖投注')
    smart_parser.add_argument('--front-dan', type=int, default=2, help='前区胆码数量')
    smart_parser.add_argument('--back-dan', type=int, default=1, help='后区胆码数量')
    smart_parser.add_argument('--front-tuo', type=int, default=6, help='前区拖码数量')
    smart_parser.add_argument('--back-tuo', type=int, default=4, help='后区拖码数量')

    # ==================== 参数优化命令 ====================
    optimize_parser = subparsers.add_parser('optimize', help='参数优化')
    optimize_parser.add_argument('-t', '--test-periods', type=int, default=100, help='测试期数')
    optimize_parser.add_argument('-r', '--rounds', type=int, default=10, help='优化轮数')
    optimize_parser.add_argument('--save', help='保存优化结果')

    # ==================== 回测命令 ====================
    backtest_parser = subparsers.add_parser('backtest', help='历史回测')
    backtest_parser.add_argument('-s', '--start', type=int, default=100, help='起始期数')
    backtest_parser.add_argument('-t', '--test', type=int, default=500, help='测试期数')
    backtest_parser.add_argument('-m', '--method',
                                choices=['frequency', 'hot_cold', 'missing', 'markov', 'bayesian', 'ensemble'],
                                default='ensemble', help='预测方法')
    backtest_parser.add_argument('--missing-mode', choices=['auto', 'legacy', 'enhanced'],
                                default='auto', help='遗漏预测模式 (auto/legacy/enhanced)')

    # ==================== 系统管理命令 ====================
    system_parser = subparsers.add_parser('system', help='系统管理')
    system_subparsers = system_parser.add_subparsers(dest='system_action', help='系统操作')
    
    # 缓存管理
    cache_parser = system_subparsers.add_parser('cache', help='智能缓存管理')
    cache_parser.add_argument('action', choices=['info', 'clear', 'refresh', 'status'],
                             help='缓存操作: info(信息), clear(清理), refresh(强制刷新), status(状态)')
    cache_parser.add_argument('--type', choices=['all', 'models', 'analysis', 'data'],
                             default='all', help='缓存类型')
    cache_parser.add_argument('--method', type=str, help='指定要刷新的分析方法名称')
    
    # ==================== 增强功能命令 ====================
    enhanced_parser = subparsers.add_parser('enhanced', help='增强功能')
    enhanced_subparsers = enhanced_parser.add_subparsers(dest='enhanced_action', help='增强功能操作')

    # 系统信息
    info_parser = enhanced_subparsers.add_parser('info', help='显示增强系统信息')
    info_parser.add_argument('--gpu', action='store_true', help='显示GPU信息')
    info_parser.add_argument('--models', action='store_true', help='显示模型信息')
    info_parser.add_argument('--performance', action='store_true', help='显示性能信息')
    info_parser.add_argument('--all', action='store_true', help='显示所有信息')

    # 兼容性测试
    compat_parser = enhanced_subparsers.add_parser('test', help='运行兼容性测试')

    # 增强预测
    epredict_parser = enhanced_subparsers.add_parser('predict', help='增强预测')
    epredict_parser.add_argument('-d', '--data', help='预测数据')
    epredict_parser.add_argument('-m', '--method', default='auto', help='预测方法')

    # 增强可视化
    evisualize_parser = enhanced_subparsers.add_parser('visualize', help='增强可视化')
    evisualize_parser.add_argument('-d', '--data', help='可视化数据')
    evisualize_parser.add_argument('-t', '--type', default='auto', help='图表类型')

    # ==================== 批量对比功能 ====================
    compare_parser = subparsers.add_parser('compare', help='批量预测对比')
    compare_parser.add_argument('--issue', type=str, required=True, help='目标期号（如：25104）')
    compare_parser.add_argument('-m', '--method', type=str, default='markov',
                              choices=['frequency', 'hot_cold', 'missing', 'markov', 'markov_2nd', 'markov_3rd',
                                      'adaptive_markov', 'bayesian', 'ensemble', 'clustering', 'super', 'adaptive',
                                      'nine_models', 'advanced_integration', 'mixed_strategy', 'highly_integrated'],
                              help='预测方法（默认：markov）')
    compare_parser.add_argument('-p', '--periods', type=int, default=100, help='分析期数（默认：100）')
    compare_parser.add_argument('-t', '--times', type=int, default=50, help='对比次数（默认：50）')
    compare_parser.add_argument('--random-periods', action='store_true', help='使用随机期数分析')
    compare_parser.add_argument('--min-periods', type=int, default=20, help='随机期数最小值（默认：20）')
    compare_parser.add_argument('--max-periods', type=int, default=None, help='随机期数最大值（默认：最大可用期数）')
    compare_parser.add_argument('--export', action='store_true', help='导出Excel文件')
    compare_parser.add_argument('--no-progress', action='store_true', help='不显示进度信息')
    
    # ==================== 帮助和版本 ====================
    version_parser = subparsers.add_parser('version', help='显示版本信息')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # 创建系统实例
    system = DLTPredictorSystem()
    
    # 执行对应的命令
    try:
        if args.command == 'data':
            system.run_data_command(args)
        elif args.command == 'analyze':
            system.run_analyze_command(args)
        elif args.command == 'predict':
            system.run_predict_command(args)
        elif args.command == 'learn':
            system.run_learn_command(args)
        elif args.command == 'smart':
            system.run_smart_command(args)
        elif args.command == 'optimize':
            system.run_optimize_command(args)
        elif args.command == 'backtest':
            system.run_backtest_command(args)
        elif args.command == 'system':
            system.run_system_command(args)
        elif args.command == 'compare':
            system.run_compare_command(args)
        elif args.command == 'enhanced':
            system.run_enhanced_command(args)
        elif args.command == 'version':
            system.show_version()
    except KeyboardInterrupt:
        print("\nWARNING  操作被用户中断")
        task_manager.interrupt_current_task()
    except Exception as e:
        logger_manager.error("命令执行失败", e)
        print(f"ERROR 命令执行失败: {e}")


if __name__ == "__main__":
    main()
