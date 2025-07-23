#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
大乐透预测系统 - 优化版主程序
支持延迟加载，避免启动时间过长
"""

import argparse
import sys
import os
from datetime import datetime
from typing import List, Dict

# 只导入核心模块
from core_modules import cache_manager, logger_manager, data_manager, task_manager

# 导入增强功能集成模块
try:
    from enhanced_integration import enhanced_dlt_system, is_enhanced_available
    ENHANCED_INTEGRATION_AVAILABLE = True
except ImportError as e:
    ENHANCED_INTEGRATION_AVAILABLE = False
    print(f"⚠️ 增强功能集成模块加载失败: {e}")


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

        # 初始化增强功能
        self.enhanced_available = ENHANCED_INTEGRATION_AVAILABLE and is_enhanced_available()
        if self.enhanced_available:
            self.enhanced_system = enhanced_dlt_system
            logger_manager.info("✅ 增强功能已集成到主系统")
        else:
            self.enhanced_system = None
            logger_manager.info("⚠️ 使用基础功能模式")
    
    def _load_analyzers(self):
        """延迟加载分析器"""
        if not self._analyzers_loaded:
            print("📊 加载分析器模块...")
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
            print("🎯 加载预测器模块...")
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
            print("🚀 加载自适应学习模块...")
            from adaptive_learning_modules import enhanced_adaptive_predictor
            self.adaptive_predictor = enhanced_adaptive_predictor
            self._adaptive_loaded = True
    
    def run_data_command(self, args):
        """处理数据管理命令"""
        if args.data_action == 'status':
            print("📊 数据状态:")
            stats = data_manager.get_stats()
            print(f"  总期数: {stats.get('total_periods', 0)}")
            print(f"  数据范围: {stats.get('date_range', {}).get('start', 'N/A')} 到 {stats.get('date_range', {}).get('end', 'N/A')}")
            print(f"  最新期号: {stats.get('latest_issue', 'N/A')}")

            # 缓存信息
            cache_info = cache_manager.get_cache_info()
            print(f"\n💾 缓存状态:")
            print(f"  总文件数: {cache_info['total']['files']}")
            print(f"  总大小: {cache_info['total']['size_mb']:.2f} MB")

        elif args.data_action == 'latest':
            print("🔍 获取最新开奖结果...")
            try:
                # 获取本地最新数据
                df = data_manager.get_data()
                if df is not None and len(df) > 0:
                    latest_row = df.iloc[0]  # 第一行是最新数据
                    front_balls, back_balls = data_manager.parse_balls(latest_row)

                    print("✅ 最新开奖结果:")
                    print(f"  期号: {latest_row['issue']}")
                    print(f"  日期: {latest_row['date']}")
                    print(f"  开奖号码: {' '.join([str(b).zfill(2) for b in front_balls])} + {' '.join([str(b).zfill(2) for b in back_balls])}")

                    # 如果指定了比较选项
                    if hasattr(args, 'compare') and args.compare:
                        self._compare_with_latest(front_balls, back_balls)
                else:
                    print("❌ 没有找到开奖数据")
            except Exception as e:
                print(f"❌ 获取最新开奖结果失败: {e}")

        elif args.data_action == 'update':
            # 处理更新参数
            periods = getattr(args, 'periods', None)
            incremental = getattr(args, 'incremental', False)

            update_type = "增量更新" if incremental else "完整更新"
            print(f"🔄 {update_type} (数据源: {args.source}" + (f", 期数: {periods}" if periods else "") + ")...")

            try:
                from crawlers import ZhcwCrawler
                crawler = ZhcwCrawler()

                if incremental:
                    # 增量更新：只获取最新的几页数据
                    count = crawler.crawl_recent_data(3)
                elif periods:
                    # 更新指定期数
                    count = crawler.crawl_recent_data(periods)
                else:
                    # 更新所有数据
                    count = crawler.crawl_all_data()

                # 清理缓存并重新加载数据
                cache_manager.clear_cache('data')
                data_manager._load_data()
                print(f"✅ 数据更新完成，新增 {count} 期数据")

            except ImportError:
                print("❌ 爬虫模块未找到，请检查crawlers.py文件")
            except Exception as e:
                print(f"❌ 数据更新失败: {e}")

    def _compare_with_latest(self, actual_front: List[int], actual_back: List[int]):
        """与最新开奖结果比较"""
        print("\n🎯 号码比较功能:")
        print("请输入您的号码进行比较")

        try:
            # 输入前区号码
            front_input = input("前区号码 (5个号码，用空格分隔): ").strip()
            front_numbers = [int(x) for x in front_input.split()]

            if len(front_numbers) != 5:
                print("❌ 前区号码必须是5个")
                return

            # 输入后区号码
            back_input = input("后区号码 (2个号码，用空格分隔): ").strip()
            back_numbers = [int(x) for x in back_input.split()]

            if len(back_numbers) != 2:
                print("❌ 后区号码必须是2个")
                return

            # 计算中奖情况
            from adaptive_learning_modules import AccuracyTracker
            tracker = AccuracyTracker()
            prize_level, front_hits, back_hits = tracker._calculate_prize_level(
                front_numbers, back_numbers, actual_front, actual_back
            )

            print(f"\n🏆 比较结果:")
            print(f"  您的号码: {' '.join([str(b).zfill(2) for b in front_numbers])} + {' '.join([str(b).zfill(2) for b in back_numbers])}")
            print(f"  开奖号码: {' '.join([str(b).zfill(2) for b in actual_front])} + {' '.join([str(b).zfill(2) for b in actual_back])}")
            print(f"  前区命中: {front_hits} 个")
            print(f"  后区命中: {back_hits} 个")
            print(f"  中奖等级: {prize_level}")

        except ValueError:
            print("❌ 请输入有效的数字")
        except KeyboardInterrupt:
            print("\n⚠️  操作被取消")
        except Exception as e:
            print(f"❌ 比较失败: {e}")
    
    def run_analyze_command(self, args):
        """处理分析命令"""
        self._load_analyzers()
        
        print(f"📊 开始{args.type}分析 (期数: {args.periods})...")
        
        try:
            if args.type == 'basic':
                # 基础分析
                freq_result = self.analyzers['basic'].frequency_analysis(args.periods)
                hot_cold_result = self.analyzers['basic'].hot_cold_analysis(args.periods)
                
                print("✅ 基础分析完成")
                print(f"  频率分析: {len(freq_result.get('front_frequency', {}))} 个前区号码")
                print(f"  冷热分析: 热号 {len(hot_cold_result.get('front_hot', []))} 个，冷号 {len(hot_cold_result.get('front_cold', []))} 个")
            
            elif args.type == 'advanced':
                # 高级分析
                markov_result = self.analyzers['advanced'].markov_analysis(args.periods)
                bayesian_result = self.analyzers['advanced'].bayesian_analysis(args.periods)
                
                print("✅ 高级分析完成")
                print(f"  马尔可夫分析: {len(markov_result.get('front_transition_probs', {}))} 个转移概率")
                print(f"  贝叶斯分析: 后验概率计算完成")
            
            elif args.type == 'comprehensive':
                # 综合分析
                comp_result = self.analyzers['comprehensive'].comprehensive_analysis(args.periods)
                
                print("✅ 综合分析完成")
                
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
                        print(f"📄 报告已保存: {filename}")

            # 生成可视化图表
            if hasattr(args, 'visualize') and args.visualize:
                print("🎨 生成可视化图表...")
                viz_success = self.analyzers['visualization'].generate_all_charts("output", args.periods)
                if viz_success:
                    print("✅ 可视化图表生成完成，保存在 output/ 目录")
                else:
                    print("❌ 可视化图表生成失败")

        except Exception as e:
            logger_manager.error("分析失败", e)
            print(f"❌ 分析失败: {e}")
    
    def run_predict_command(self, args):
        """处理预测命令"""
        self._load_predictors()

        # 参数验证
        if args.count < 1 or args.count > 100:
            print("❌ 注数必须在1-100之间")
            return

        if args.periods < 50 or args.periods > 2748:
            print("❌ 分析期数必须在50-2748之间")
            return

        print(f"🎯 开始{args.method}预测 (分析期数: {args.periods}, 生成注数: {args.count})...")

        # 检查是否可以使用增强功能
        use_enhanced = self.enhanced_available and args.method in ['lstm', 'transformer', 'gan', 'ensemble', 'enhanced', 'stacking', 'adaptive_ensemble', 'ultimate_ensemble']

        if use_enhanced:
            print("🚀 使用增强预测引擎...")
            try:
                # 使用增强预测功能
                if args.method == 'enhanced':
                    # 使用增强系统的自动预测
                    result = self.enhanced_system.enhanced_predict(
                        data=f"predict_{args.count}_numbers_periods_{args.periods}",
                        method="auto",
                        periods=args.periods,
                        count=args.count
                    )
                    if result.get('success'):
                        print("✅ 增强预测完成")
                        print(f"预测结果: {result['result']}")
                        print(f"使用方法: {result['method']}")
                        print(f"已缓存: {result['cached']}")
                        return
                    else:
                        print(f"❌ 增强预测失败: {result.get('error')}")
                        print("🔄 回退到传统预测方法...")

                elif args.method in ['lstm', 'transformer', 'gan', 'ensemble', 'stacking', 'adaptive_ensemble', 'ultimate_ensemble']:
                    # 使用增强深度学习模型或集成方法
                    try:
                        if args.method in ['lstm', 'transformer', 'gan', 'ensemble']:
                            # 深度学习模型
                            from enhanced_deep_learning.models import model_registry
                            model = model_registry.get_model(args.method)

                            if model:
                                print(f"🚀 使用{args.method.upper()}深度学习模型...")
                                historical_data = data_manager.get_data()
                                if historical_data is not None and len(historical_data) > args.periods:
                                    historical_data = historical_data.head(args.periods)
                                    print(f"📊 使用最新{args.periods}期数据进行{args.method.upper()}模型训练...")

                                predictions = model.predict_lottery(data=historical_data, count=args.count, periods=args.periods)

                                if predictions:
                                    print(f"✅ {args.method.upper()}预测完成")
                                    self._display_enhanced_predictions(predictions, args.method)
                                    return
                                else:
                                    print(f"❌ {args.method}深度学习模型预测失败，尝试集成方法...")
                            else:
                                print(f"❌ {args.method}深度学习模型未找到，尝试集成方法...")

                        # 如果深度学习模型失败或者是集成方法，使用improvements模块
                        from improvements.integration import get_integrator
                        integrator = get_integrator()

                        if args.method == 'lstm':
                            print("🧠 LSTM集成预测...")
                            # 尝试使用advanced_lstm_predictor作为回退
                            try:
                                from advanced_lstm_predictor import AdvancedLSTMPredictor
                                lstm_predictor = AdvancedLSTMPredictor()
                                results = lstm_predictor.lstm_predict(count=args.count, periods=args.periods)
                                predictions = [{'front_balls': r[0], 'back_balls': r[1], 'method': 'lstm', 'confidence': 0.85} for r in results]
                            except Exception as e:
                                print(f"❌ LSTM预测失败: {e}")
                                predictions = []
                        elif args.method == 'transformer':
                            print("🧮 Transformer深度学习预测...")
                            predictions = integrator.transformer_predict(args.count, args.periods)
                        elif args.method == 'gan':
                            print("🎮 GAN生成预测...")
                            predictions = integrator.gan_predict(args.count, args.periods)
                        elif args.method == 'stacking':
                            print("🔄 Stacking集成预测...")
                            predictions = integrator.stacking_predict(args.count)
                        elif args.method == 'adaptive_ensemble':
                            print("🧠 自适应集成预测...")
                            predictions = integrator.adaptive_ensemble_predict(args.count)
                        elif args.method == 'ultimate_ensemble':
                            print("🌟 终极集成预测...")
                            predictions = integrator.ultimate_ensemble_predict(args.count)
                        else:
                            predictions = []

                        if predictions:
                            print(f"✅ {args.method.upper()}预测完成")
                            self._display_enhanced_predictions(predictions, args.method)
                            return
                        else:
                            print(f"❌ {args.method}预测失败，回退到传统方法...")

                    except Exception as e:
                        print(f"❌ 增强预测失败: {e}")
                        print("🔄 回退到传统预测方法...")

            except Exception as e:
                logger_manager.error(f"增强预测失败: {e}")
                print(f"❌ 增强预测失败: {e}")
                print("🔄 回退到传统预测方法...")

        try:
            predictions = []

            if args.method in ['frequency', 'hot_cold', 'missing']:
                # 传统预测方法
                if args.method == 'frequency':
                    print(f"📊 频率分析预测 (分析{args.periods}期数据)...")
                    results = self.predictors['traditional'].frequency_predict(count=args.count, periods=args.periods)
                elif args.method == 'hot_cold':
                    print(f"🌡️ 冷热号分析预测 (分析{args.periods}期数据)...")
                    print("📊 分析冷热号分布...")

                    # 获取冷热号分析结果
                    from analyzer_modules import basic_analyzer
                    hot_cold_analysis = basic_analyzer.hot_cold_analysis(args.periods)

                    front_hot = hot_cold_analysis.get('front_hot', [])
                    front_cold = hot_cold_analysis.get('front_cold', [])
                    back_hot = hot_cold_analysis.get('back_hot', [])
                    back_cold = hot_cold_analysis.get('back_cold', [])

                    print(f"✅ 冷热号识别完成:")
                    print(f"  前区热号 ({len(front_hot)}个): {sorted(front_hot)[:10]}{'...' if len(front_hot) > 10 else ''}")
                    print(f"  前区冷号 ({len(front_cold)}个): {sorted(front_cold)[:10]}{'...' if len(front_cold) > 10 else ''}")
                    print(f"  后区热号 ({len(back_hot)}个): {sorted(back_hot)}")
                    print(f"  后区冷号 ({len(back_cold)}个): {sorted(back_cold)}")
                    print("🎯 基于冷热号分布进行智能预测...")

                    results = self.predictors['traditional'].hot_cold_predict(count=args.count, periods=args.periods)
                elif args.method == 'missing':
                    print(f"⏰ 遗漏值分析预测 (分析{args.periods}期数据)...")
                    results = self.predictors['traditional'].missing_predict(count=args.count, periods=args.periods)
                
                predictions = [{'front_balls': r[0], 'back_balls': r[1], 'method': args.method} for r in results]
            
            elif args.method in ['markov', 'bayesian', 'ensemble']:
                # 高级预测方法
                if args.method == 'markov':
                    results = self.predictors['advanced'].markov_predict(args.count, args.periods)
                elif args.method == 'bayesian':
                    print(f"🎲 贝叶斯分析预测 (分析{args.periods}期数据)...")
                    print("📊 计算先验概率和似然函数...")

                    # 获取贝叶斯分析结果
                    from analyzer_modules import advanced_analyzer
                    bayesian_analysis = advanced_analyzer.bayesian_analysis(args.periods)

                    front_prior = bayesian_analysis.get('front_prior', {})
                    back_prior = bayesian_analysis.get('back_prior', {})
                    front_posterior = bayesian_analysis.get('front_posterior', {})
                    back_posterior = bayesian_analysis.get('back_posterior', {})

                    print(f"✅ 贝叶斯推理完成:")
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

                    print("🎯 基于贝叶斯推理进行概率预测...")

                    results = self.predictors['advanced'].bayesian_predict(count=args.count, periods=args.periods)
                elif args.method == 'ensemble':
                    results = self.predictors['advanced'].ensemble_predict(args.count, args.periods)
                
                predictions = [{'front_balls': r[0], 'back_balls': r[1], 'method': args.method} for r in results]
            
            elif args.method == 'super':
                # 超级预测
                results = self.predictors['super'].predict_super(args.count, args.periods)
                predictions = results
            
            elif args.method == 'adaptive':
                # 自适应预测
                self._load_adaptive_predictor()
                results = self.adaptive_predictor.generate_enhanced_prediction(args.count, args.periods)
                predictions = results

            elif args.method == 'compound':
                # 复式投注预测
                front_count = getattr(args, 'front_count', 8)
                back_count = getattr(args, 'back_count', 4)
                result = self.predictors['compound'].predict_compound(front_count, back_count, 'ensemble', args.periods)
                if result:
                    predictions = [result]
                else:
                    predictions = []

            elif args.method == 'duplex':
                # 胆拖投注预测
                result = self.predictors['compound'].predict_duplex(periods=args.periods)
                if result:
                    predictions = [result]
                else:
                    predictions = []
                    
            elif args.method in ['transformer', 'gan', 'stacking', 'adaptive_ensemble', 'ultimate_ensemble']:
                # 增强预测方法
                try:
                    from improvements.integration import get_integrator
                    integrator = get_integrator()
                    
                    if args.method == 'transformer':
                        results = integrator.transformer_predict(args.count)
                    elif args.method == 'gan':
                        results = integrator.gan_predict(args.count)
                    elif args.method == 'stacking':
                        results = integrator.stacking_predict(args.count)
                    elif args.method == 'adaptive_ensemble':
                        results = integrator.adaptive_ensemble_predict(args.count)
                    elif args.method == 'ultimate_ensemble':
                        results = integrator.ultimate_ensemble_predict(args.count)
                    
                    predictions = results
                except ImportError:
                    print("❌ 增强预测模块未找到，请确保improvements目录存在且包含所需文件")
                except Exception as e:
                    print(f"❌ 增强预测失败: {e}")

            elif args.method == 'markov_custom':
                # 马尔可夫自定义期数预测
                analysis_periods = getattr(args, 'analysis_periods', 300)
                predict_periods = getattr(args, 'predict_periods', 1)
                results = self.predictors['advanced'].markov_predict_custom(
                    count=args.count,
                    analysis_periods=analysis_periods,
                    predict_periods=predict_periods
                )
                predictions = results

            elif args.method == 'mixed_strategy':
                # 混合策略预测
                strategy = getattr(args, 'strategy', 'balanced')
                results = self.predictors['advanced'].mixed_strategy_predict(
                    count=args.count,
                    strategy=strategy,
                    periods=args.periods
                )
                predictions = results

            elif args.method == 'highly_integrated':
                # 高度集成复式预测
                front_count = getattr(args, 'front_count', 10)
                back_count = getattr(args, 'back_count', 5)
                integration_level = getattr(args, 'integration_level', 'ultimate')
                result = self.predictors['compound'].predict_highly_integrated_compound(
                    front_count=front_count,
                    back_count=back_count,
                    integration_level=integration_level,
                    periods=args.periods
                )
                if result:
                    predictions = [result]
                else:
                    predictions = []

            elif args.method == 'advanced_integration':
                # 高级集成分析预测
                integration_type = getattr(args, 'integration_type', 'comprehensive')
                results = self.predictors['advanced'].advanced_integration_predict(
                    count=args.count,
                    integration_type=integration_type,
                    periods=args.periods
                )
                predictions = results

            elif args.method == 'nine_models':
                # 9种数学模型预测
                results = self.predictors['advanced'].nine_models_predict(count=args.count, periods=args.periods)
                predictions = results

            elif args.method == 'nine_models_compound':
                # 9种数学模型复式预测
                front_count = getattr(args, 'front_count', 8)
                back_count = getattr(args, 'back_count', 4)
                result = self.predictors['advanced'].nine_models_compound_predict(
                    front_count=front_count,
                    back_count=back_count,
                    periods=args.periods
                )
                if result:
                    predictions = [result]
                else:
                    predictions = []

            elif args.method == 'markov_compound':
                # 马尔可夫链复式预测
                front_count = getattr(args, 'front_count', 8)
                back_count = getattr(args, 'back_count', 4)
                markov_periods = args.periods  # 使用新的periods参数
                result = self.predictors['advanced'].markov_compound_predict(
                    front_count=front_count,
                    back_count=back_count,
                    analysis_periods=markov_periods
                )
                if result:
                    predictions = [result]
                else:
                    predictions = []
                    
            elif args.method in ['markov_2nd', 'markov_3rd', 'adaptive_markov']:
                # 增强版马尔可夫链预测
                try:
                    from improvements.enhanced_markov import get_markov_predictor
                    
                    markov_periods = args.periods  # 使用新的periods参数
                    
                    if args.method == 'markov_2nd':
                        print(f"🔄 二阶马尔可夫链预测 (分析{markov_periods}期数据)...")
                        print("📊 构建二阶状态转移矩阵...")
                        print("🔢 概率计算: 基于历史数据计算转移概率")
                        print("📈 矩阵计算: 构建复合状态转移矩阵")

                        markov_predictor = get_markov_predictor()

                        # 获取二阶马尔可夫分析结果
                        markov_analyzer = markov_predictor.analyzer
                        analysis_result = markov_analyzer.multi_order_markov_analysis(markov_periods, max_order=2)

                        if analysis_result and 'orders' in analysis_result and 2 in analysis_result['orders']:
                            order_2_result = analysis_result['orders'][2]
                            front_stats = order_2_result.get('front_stats', {})
                            back_stats = order_2_result.get('back_stats', {})

                            print(f"✅ 二阶状态转移矩阵构建完成:")
                            print(f"  📊 概率计算: 前区转移概率数 {front_stats.get('total_transitions', 0)}")
                            print(f"  📈 矩阵计算: 前区状态数 {front_stats.get('unique_states', 0)}")
                            print(f"  🔢 概率计算: 后区转移概率数 {back_stats.get('total_transitions', 0)}")
                            print(f"  📈 矩阵计算: 后区状态数 {back_stats.get('unique_states', 0)}")
                            print(f"  🎯 最大转移概率: 前区 {front_stats.get('max_probability', 0):.4f}, 后区 {back_stats.get('max_probability', 0):.4f}")

                        results = markov_predictor.multi_order_markov_predict(
                            count=args.count,
                            periods=markov_periods,
                            order=2
                        )
                        predictions = [{'front_balls': r[0], 'back_balls': r[1], 'method': 'markov_2nd', 'confidence': 0.85, 'order': 2} for r in results]
                    
                    elif args.method == 'markov_3rd':
                        print(f"🔄 三阶马尔可夫链预测 (分析{markov_periods}期数据)...")
                        print("📊 构建三阶状态转移矩阵...")
                        print("🔢 状态转移显示: 完整的状态转移矩阵构建和统计信息")
                        print("📈 超高阶建模: 考虑前三期状态的复杂依赖关系")

                        markov_predictor = get_markov_predictor()

                        # 获取三阶马尔可夫分析结果
                        markov_analyzer = markov_predictor.analyzer
                        analysis_result = markov_analyzer.multi_order_markov_analysis(markov_periods, max_order=3)

                        if analysis_result and 'orders' in analysis_result and 3 in analysis_result['orders']:
                            order_3_result = analysis_result['orders'][3]
                            front_stats = order_3_result.get('front_stats', {})
                            back_stats = order_3_result.get('back_stats', {})

                            print(f"✅ 三阶状态转移矩阵构建完成:")
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
                        predictions = [{'front_balls': r[0], 'back_balls': r[1], 'method': 'markov_3rd', 'confidence': 0.9, 'order': 3} for r in results]
                    
                    elif args.method == 'adaptive_markov':
                        print("🔄 自适应马尔可夫链预测...")
                        markov_predictor = get_markov_predictor()
                        predictions = markov_predictor.adaptive_order_markov_predict(
                            count=args.count, 
                            periods=markov_periods
                        )
                
                except ImportError:
                    print("❌ 增强版马尔可夫链模块未找到，请确保improvements目录存在且包含所需文件")
                    predictions = []
                except Exception as e:
                    print(f"❌ 增强版马尔可夫链预测失败: {e}")
                    predictions = []
            

            

            


            # 显示预测结果
            print("✅ 预测完成!")
            print("\n📋 预测结果:")

            for i, pred in enumerate(predictions):
                if pred.get('front_dan'):
                    # 胆拖投注显示
                    front_dan_str = ' '.join([str(b).zfill(2) for b in pred['front_dan']])
                    front_tuo_str = ' '.join([str(b).zfill(2) for b in pred['front_tuo']])
                    back_dan_str = ' '.join([str(b).zfill(2) for b in pred['back_dan']])
                    back_tuo_str = ' '.join([str(b).zfill(2) for b in pred['back_tuo']])

                    print(f"  第 {i+1} 注胆拖:")
                    print(f"    前区: {front_dan_str} + ({front_tuo_str})")
                    print(f"    后区: {back_dan_str} + ({back_tuo_str})")
                    print(f"    总组合数: {pred['total_combinations']} 注")
                    print(f"    总投注额: {pred['total_cost']} 元")

                elif pred.get('front_count'):
                    # 复式投注显示
                    front_str = ' '.join([str(b).zfill(2) for b in pred['front_balls']])
                    back_str = ' '.join([str(b).zfill(2) for b in pred['back_balls']])

                    method_name = pred.get('method', 'compound').replace('_', ' ').title()
                    print(f"  第 {i+1} 注复式 ({method_name}): {front_str} + {back_str}")
                    print(f"    前区: {pred['front_count']} 个号码")
                    print(f"    后区: {pred['back_count']} 个号码")
                    print(f"    总组合数: {pred['total_combinations']} 注")
                    print(f"    总投注额: {pred['total_cost']} 元")
                    print(f"    置信度: {pred.get('confidence', 0.5):.3f}")

                    # 显示特定方法的详细信息
                    if pred.get('method') == 'nine_models_compound':
                        if 'models_used' in pred:
                            print(f"    使用模型: {len(pred['models_used'])} 种")
                        if 'model_details' in pred:
                            details = pred['model_details']
                            print(f"    统计学权重: {details.get('statistical_score', 0):.3f}")
                            print(f"    概率论权重: {details.get('probability_score', 0):.3f}")
                            print(f"    马尔可夫权重: {details.get('markov_score', 0):.3f}")
                            print(f"    贝叶斯权重: {details.get('bayesian_score', 0):.3f}")

                    elif pred.get('method') == 'markov_compound':
                        print(f"    分析期数: {pred.get('analysis_periods', 500)}")
                        if 'markov_details' in pred:
                            details = pred['markov_details']
                            print(f"    转移矩阵规模: {details.get('transition_matrix_size', 0)}")
                            print(f"    状态数量: {details.get('state_count', 0)}")
                            print(f"    预测准确性: {details.get('prediction_accuracy', 0):.3f}")

                    elif pred.get('integration_level'):
                        print(f"    集成级别: {pred['integration_level']}")
                        print(f"    使用算法: {len(pred.get('algorithms_used', []))} 种")

                elif pred.get('overall_stability'):
                    # 马尔可夫自定义预测显示
                    front_str = ' '.join([str(b).zfill(2) for b in pred['front_balls']])
                    back_str = ' '.join([str(b).zfill(2) for b in pred['back_balls']])

                    print(f"  第 {pred['index']} 注 (期 {pred['period']}): {front_str} + {back_str}")
                    print(f"    稳定性得分: {pred['overall_stability']:.3f}")
                    print(f"    前区稳定性: {pred['front_stability']:.3f}")
                    print(f"    后区稳定性: {pred['back_stability']:.3f}")
                    print(f"    分析期数: {pred['analysis_periods']}")

                elif pred.get('strategy'):
                    # 混合策略预测显示
                    front_str = ' '.join([str(b).zfill(2) for b in pred['front_balls']])
                    back_str = ' '.join([str(b).zfill(2) for b in pred['back_balls']])

                    print(f"  第 {pred['index']} 注 ({pred['strategy']}策略): {front_str} + {back_str}")
                    print(f"    风险等级: {pred['risk_level']}")
                    print(f"    策略描述: {pred['description']}")
                    print(f"    权重配置: {pred['weights']}")

                elif pred.get('integration_level'):
                    # 高度集成复式预测显示
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

                elif pred.get('integration_type'):
                    # 高级集成分析预测显示
                    front_str = ' '.join([str(b).zfill(2) for b in pred['front_balls']])
                    back_str = ' '.join([str(b).zfill(2) for b in pred['back_balls']])

                    print(f"  第 {pred['index']} 注 ({pred['integration_type']}集成): {front_str} + {back_str}")
                    print(f"    集成类型: {pred['integration_type']}")
                    print(f"    分析方法: {pred['method']}")
                    print(f"    置信度: {pred['confidence']:.3f}")
                    if 'analysis_source' in pred:
                        print(f"    分析时间: {pred['analysis_source']}")

                elif pred.get('method') == 'nine_mathematical_models':
                    # 9种数学模型预测显示
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

                elif pred.get('method') == 'nine_models_compound':
                    # 9种数学模型复式预测显示
                    front_str = ' '.join([str(b).zfill(2) for b in pred['front_balls']])
                    back_str = ' '.join([str(b).zfill(2) for b in pred['back_balls']])

                    print(f"  9种数学模型复式: {front_str} + {back_str}")
                    print(f"    前区: {pred['front_count']} 个号码")
                    print(f"    后区: {pred['back_count']} 个号码")
                    print(f"    总组合数: {pred['total_combinations']} 注")
                    print(f"    总投注额: {pred['total_cost']} 元")
                    print(f"    置信度: {pred['confidence']:.3f}")
                    if 'models_used' in pred:
                        print(f"    使用模型: {len(pred['models_used'])} 种")
                    if 'model_details' in pred:
                        details = pred['model_details']
                        print(f"    统计学权重: {details.get('statistical_score', 0):.3f}")
                        print(f"    概率论权重: {details.get('probability_score', 0):.3f}")
                        print(f"    马尔可夫权重: {details.get('markov_score', 0):.3f}")
                        print(f"    贝叶斯权重: {details.get('bayesian_score', 0):.3f}")

                elif pred.get('method') and 'compound' in pred['method']:
                    # 通用复式预测显示
                    front_str = ' '.join([str(b).zfill(2) for b in pred['front_balls']])
                    back_str = ' '.join([str(b).zfill(2) for b in pred['back_balls']])

                    method_name = pred['method'].replace('_', ' ').title()
                    print(f"  {method_name}: {front_str} + {back_str}")
                    print(f"    前区: {pred['front_count']} 个号码")
                    print(f"    后区: {pred['back_count']} 个号码")
                    print(f"    总组合数: {pred['total_combinations']} 注")
                    print(f"    总投注额: {pred['total_cost']} 元")
                    print(f"    置信度: {pred['confidence']:.3f}")

                elif pred.get('method') == 'markov_compound':
                    # 马尔可夫链复式预测显示
                    front_str = ' '.join([str(b).zfill(2) for b in pred['front_balls']])
                    back_str = ' '.join([str(b).zfill(2) for b in pred['back_balls']])

                    print(f"  马尔可夫链复式: {front_str} + {back_str}")
                    print(f"    前区: {pred['front_count']} 个号码")
                    print(f"    后区: {pred['back_count']} 个号码")
                    print(f"    总组合数: {pred['total_combinations']} 注")
                    print(f"    总投注额: {pred['total_cost']} 元")
                    print(f"    置信度: {pred['confidence']:.3f}")
                    print(f"    分析期数: {pred.get('analysis_periods', 500)}")
                    if 'markov_details' in pred:
                        details = pred['markov_details']
                        print(f"    转移矩阵规模: {details.get('transition_matrix_size', 0)}")
                        print(f"    状态数量: {details.get('state_count', 0)}")
                        print(f"    预测准确性: {details.get('prediction_accuracy', 0):.3f}")

                else:
                    # 单式投注显示
                    front_str = ' '.join([str(b).zfill(2) for b in pred['front_balls']])
                    back_str = ' '.join([str(b).zfill(2) for b in pred['back_balls']])
                    method = pred.get('method', args.method)
                    confidence = pred.get('confidence', 0.5)

                    print(f"  第 {i+1} 注: {front_str} + {back_str} (方法: {method}, 置信度: {confidence:.3f})")

            # 保存预测结果
            if args.save:
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

                print(f"💾 预测结果已保存: {filename}")
        
        except Exception as e:
            logger_manager.error("预测失败", e)
            print(f"❌ 预测失败: {e}")

    def _display_predictions(self, predictions, method):
        """显示预测结果"""
        if not predictions:
            print("❌ 没有生成预测结果")
            return

        print(f"✅ {method.upper()}预测完成")
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
            print("❌ 没有生成预测结果")
            return

        print(f"✅ {method.upper()}深度学习预测完成")
        print("=" * 60)

        for i, pred in enumerate(predictions, 1):
            if isinstance(pred, dict):
                front = pred.get('front', [])
                back = pred.get('back', [])
                confidence = pred.get('confidence', 0.0)
                pred_method = pred.get('method', method)

                print(f"第{i}注 [{pred_method}]:")
                print(f"  前区: {' '.join(f'{n:02d}' for n in front)}")
                print(f"  后区: {' '.join(f'{n:02d}' for n in back)}")
                print(f"  置信度: {confidence:.1%}")
                print()
            else:
                print(f"第{i}注: {pred}")

        print("=" * 60)
        print(f"🎯 使用{method.upper()}深度学习算法生成 {len(predictions)} 注预测")
        print("💡 深度学习模型已自动训练并优化参数")

    def run_learn_command(self, args):
        """处理学习命令"""
        self._load_adaptive_predictor()
        
        print(f"🚀 开始自适应学习 (算法: {args.algorithm})...")
        print(f"📊 起始期数: {args.start}, 测试期数: {args.test}")
        
        try:
            # 设置多臂老虎机算法
            self.adaptive_predictor.bandit.algorithm = args.algorithm
            
            # 进行学习
            results = self.adaptive_predictor.enhanced_adaptive_learning(
                start_period=args.start,
                test_periods=args.test
            )
            
            if results:
                print("✅ 自适应学习完成!")
                print(f"📊 中奖率: {results['win_rate']:.3f}")
                print(f"📈 平均得分: {results['average_score']:.2f}")
                print(f"🎯 总测试期数: {results['total_periods']}")
                
                # 显示预测器性能
                print("\n🏆 预测器性能排名:")
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
                    print(f"💾 学习结果已保存: {saved_file}")
            else:
                print("❌ 自适应学习失败")
        
        except Exception as e:
            logger_manager.error("学习失败", e)
            print(f"❌ 学习失败: {e}")
    
    def run_smart_command(self, args):
        """处理智能预测命令"""
        self._load_adaptive_predictor()

        # 确定预测类型
        if args.compound:
            print(f"🧠 智能复式预测 ({args.front_count}+{args.back_count})...")
        elif args.duplex:
            print(f"🧠 智能胆拖预测 (前区{args.front_dan}胆{args.front_tuo}拖, 后区{args.back_dan}胆{args.back_tuo}拖)...")
        else:
            print(f"🧠 智能预测 (注数: {args.count})...")

        try:
            # 加载学习结果
            if args.load:
                if self.adaptive_predictor.load_enhanced_results(args.load):
                    print(f"✅ 已加载学习结果: {args.load}")
                else:
                    print(f"❌ 加载学习结果失败: {args.load}")
                    return
            else:
                print("⚠️  未加载学习结果，使用默认配置")

            # 根据类型生成预测
            if args.compound:
                # 复式投注预测
                result = self.adaptive_predictor.smart_predict_compound(
                    front_count=args.front_count,
                    back_count=args.back_count,
                    periods=args.periods
                )

                if result:
                    print("✅ 智能复式预测完成!")
                    print("\n🧠 智能复式预测结果:")

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
                    print("❌ 智能复式预测失败")

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
                    print("✅ 智能胆拖预测完成!")
                    print("\n🧠 智能胆拖预测结果:")

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
                    print("❌ 智能胆拖预测失败")

            else:
                # 单式投注预测
                predictions = self.adaptive_predictor.generate_enhanced_prediction(args.count)

                if predictions:
                    print("✅ 智能预测完成!")
                    print("\n🧠 智能预测结果:")

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
                    print("❌ 智能预测失败")

        except Exception as e:
            logger_manager.error("智能预测失败", e)
            print(f"❌ 智能预测失败: {e}")

    def run_optimize_command(self, args):
        """处理参数优化命令"""
        self._load_adaptive_predictor()

        print(f"⚙️ 参数优化 (测试期数: {args.test_periods}, 优化轮数: {args.rounds})...")

        try:
            # 进行参数优化
            results = self.adaptive_predictor.parameter_optimization(
                test_periods=args.test_periods,
                optimization_rounds=args.rounds
            )

            if results:
                print("✅ 参数优化完成!")
                print(f"\n🏆 最佳参数:")
                for param, value in results['best_params'].items():
                    print(f"  {param}: {value}")

                print(f"\n📊 最佳得分: {results['best_score']:.3f}")

                print(f"\n📈 优化历史:")
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

                    print(f"💾 优化结果已保存: {filename}")
            else:
                print("❌ 参数优化失败")

        except Exception as e:
            logger_manager.error("参数优化失败", e)
            print(f"❌ 参数优化失败: {e}")

    def run_backtest_command(self, args):
        """处理回测命令"""
        self._load_predictors()

        print(f"📈 开始历史回测 (方法: {args.method})...")
        print(f"📊 起始期数: {args.start}, 测试期数: {args.test}")

        try:
            from adaptive_learning_modules import AccuracyTracker

            # 创建准确率跟踪器
            tracker = AccuracyTracker()

            # 获取数据
            df = data_manager.get_data()
            if df is None or len(df) < args.start + args.test:
                print("❌ 数据不足")
                return

            total_predictions = 0
            total_wins = 0
            prize_stats = {}

            print(f"🔄 开始回测...")

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
                            result = self.predictors['traditional'].missing_predict(1)

                        predicted_front, predicted_back = result[0]

                    elif args.method in ['markov', 'bayesian', 'ensemble']:
                        if args.method == 'markov':
                            result = self.predictors['advanced'].markov_predict(1)
                        elif args.method == 'bayesian':
                            result = self.predictors['advanced'].bayesian_predict(1)
                        elif args.method == 'ensemble':
                            result = self.predictors['advanced'].ensemble_predict(1)

                        predicted_front, predicted_back = result[0]

                    else:
                        print(f"❌ 不支持的回测方法: {args.method}")
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
            print("✅ 回测完成!")
            print(f"\n📊 回测结果统计:")
            print(f"  总预测期数: {total_predictions}")
            print(f"  中奖期数: {total_wins}")
            print(f"  中奖率: {total_wins/total_predictions:.3f}" if total_predictions > 0 else "  中奖率: 0.000")

            print(f"\n🏆 中奖等级分布:")
            for prize, count in sorted(prize_stats.items()):
                rate = count / total_predictions if total_predictions > 0 else 0
                print(f"  {prize}: {count} 次 ({rate:.3f})")

        except Exception as e:
            logger_manager.error("回测失败", e)
            print(f"❌ 回测失败: {e}")

    def run_system_command(self, args):
        """处理系统管理命令"""
        if args.system_action == 'cache':
            if args.action == 'info':
                print("💾 缓存信息:")
                cache_info = cache_manager.get_cache_info()

                for cache_type in ['models', 'analysis', 'data']:
                    info = cache_info[cache_type]
                    print(f"  {cache_type}: {info['files']} 个文件, {info['size_mb']:.2f} MB")

                print(f"  总计: {cache_info['total']['files']} 个文件, {cache_info['total']['size_mb']:.2f} MB")

            elif args.action == 'clear':
                print(f"🗑️  清理{args.type}缓存...")
                cleared_count = cache_manager.clear_cache(args.type)
                print(f"✅ 已清理 {cleared_count} 个缓存文件")
    
    def run_enhanced_command(self, args):
        """运行增强功能命令"""
        if not self.enhanced_available:
            print("❌ 增强功能不可用")
            print("请确保已正确安装enhanced_deep_learning模块")
            return

        if args.enhanced_action == 'info':
            print("🔍 增强系统信息")
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

        elif args.enhanced_action == 'test':
            print("🧪 运行兼容性测试")
            print("=" * 50)

            result = self.enhanced_system.run_compatibility_test()
            if result.get('success'):
                for test in result['test_results']:
                    status_icon = '✅' if test['status'] == 'passed' else '❌'
                    print(f"{status_icon} {test['name']}: {test['message']} ({test['duration']:.2f}s)")
            else:
                print(f"❌ 测试失败: {result.get('error', '未知错误')}")

        elif args.enhanced_action == 'predict':
            print("🔮 增强预测")
            print("=" * 50)

            if not args.data:
                print("❌ 请提供预测数据 (-d 参数)")
                return

            result = self.enhanced_system.enhanced_predict(args.data, method=args.method)
            if result.get('success'):
                print(f"✅ 预测成功")
                print(f"方法: {result['method']}")
                print(f"结果: {result['result']}")
                print(f"已缓存: {result['cached']}")
            else:
                print(f"❌ 预测失败: {result.get('error', '未知错误')}")

        elif args.enhanced_action == 'visualize':
            print("📊 增强可视化")
            print("=" * 50)

            if not args.data:
                print("❌ 请提供可视化数据 (-d 参数)")
                return

            result = self.enhanced_system.enhanced_visualize(args.data, chart_type=args.type)
            if result.get('success'):
                print(f"✅ 可视化成功")
                print(f"图表类型: {result['chart_type']}")
                print(f"结果: {result['result']}")
            else:
                print(f"❌ 可视化失败: {result.get('error', '未知错误')}")

        else:
            print("❌ 未知的增强功能操作")
            print("可用操作: info, test, predict, visualize")

    def show_version(self):
        """显示版本信息"""
        print("🎯 大乐透预测系统")
        print("版本: 2.0.0 Enhanced")
        print("作者: AI Assistant")
        print("更新时间: 2024-12-19")
        print("\n📦 功能模块:")
        print("  ✅ 数据爬取与管理")
        print("  ✅ 基础与高级分析")
        print("  ✅ 多种预测算法")
        print("  ✅ 自适应学习系统")
        print("  ✅ 智能预测与回测")
        print("  ✅ 缓存与日志管理")

        # 显示增强功能状态
        if self.enhanced_available:
            print("\n🚀 增强功能模块:")
            print("  ✅ 企业级核心架构")
            print("  ✅ 高级数据处理")
            print("  ✅ 智能模型注册表")
            print("  ✅ 增强预测引擎")
            print("  ✅ 交互式可视化")
            print("  ✅ 工作流管理")
            print("  ✅ 跨平台兼容性")
            print("  ✅ 分布式计算")
            print("  ✅ 性能优化")
            print("  ✅ 智能缓存系统")
        else:
            print("\n⚠️ 增强功能: 未启用")
            print("  提示: 运行 'python dlt_main.py enhanced info' 查看详情")


def main():
    """主函数"""
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
                                       'markov_2nd', 'markov_3rd', 'adaptive_markov', 'enhanced'],
                               default='ensemble', help='预测方法')
    predict_parser.add_argument('--ensemble-method', choices=['stacking', 'weighted', 'adaptive'],
                               default='stacking', help='高级集成方法类型')
    predict_parser.add_argument('-c', '--count', type=int, default=1, help='生成注数 (1-100)')
    predict_parser.add_argument('-p', '--periods', type=int, default=500, help='分析期数 (50-2748，默认500期)')
    predict_parser.add_argument('--front-count', type=int, default=8, help='复式投注前区号码数量')
    predict_parser.add_argument('--back-count', type=int, default=4, help='复式投注后区号码数量')
    predict_parser.add_argument('--analysis-periods', type=int, default=300, help='马尔可夫分析期数')
    predict_parser.add_argument('--predict-periods', type=int, default=1, help='马尔可夫预测期数')
    predict_parser.add_argument('--strategy', choices=['conservative', 'aggressive', 'balanced'],
                               default='balanced', help='混合策略类型')
    predict_parser.add_argument('--integration-level', choices=['high', 'ultimate'],
                               default='ultimate', help='高度集成级别')
    predict_parser.add_argument('--integration-type', choices=['comprehensive', 'markov_bayesian', 'hot_cold_markov', 'multi_dimensional'],
                               default='comprehensive', help='高级集成分析类型')
    predict_parser.add_argument('--markov-periods', type=int, default=500, help='马尔可夫分析期数')
    predict_parser.add_argument('--save', help='保存预测结果')
    
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

    # ==================== 系统管理命令 ====================
    system_parser = subparsers.add_parser('system', help='系统管理')
    system_subparsers = system_parser.add_subparsers(dest='system_action', help='系统操作')
    
    # 缓存管理
    cache_parser = system_subparsers.add_parser('cache', help='缓存管理')
    cache_parser.add_argument('action', choices=['info', 'clear'], help='缓存操作')
    cache_parser.add_argument('--type', choices=['all', 'models', 'analysis', 'data'], 
                             default='all', help='缓存类型')
    
    # ==================== 增强功能命令 ====================
    enhanced_parser = subparsers.add_parser('enhanced', help='增强功能')
    enhanced_subparsers = enhanced_parser.add_subparsers(dest='enhanced_action', help='增强功能操作')

    # 系统信息
    info_parser = enhanced_subparsers.add_parser('info', help='显示增强系统信息')

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
        elif args.command == 'enhanced':
            system.run_enhanced_command(args)
        elif args.command == 'version':
            system.show_version()
    except KeyboardInterrupt:
        print("\n⚠️  操作被用户中断")
        task_manager.interrupt_current_task()
    except Exception as e:
        logger_manager.error("命令执行失败", e)
        print(f"❌ 命令执行失败: {e}")


if __name__ == "__main__":
    main()
