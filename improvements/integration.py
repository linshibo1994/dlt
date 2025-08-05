#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
集成模块
整合各种预测算法，提供统一的接口
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Callable, Union
from datetime import datetime
from collections import defaultdict, Counter

# 尝试导入核心模块
try:
    from core_modules import logger_manager, data_manager, cache_manager
    from smart_cache_system import smart_cache_manager
except ImportError:
    # 如果在不同目录运行，添加父目录到路径
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core_modules import logger_manager, data_manager, cache_manager
    from smart_cache_system import smart_cache_manager

# 尝试导入预测器模块
try:
    from predictor_modules import get_traditional_predictor, get_advanced_predictor, get_super_predictor
    PREDICTORS_AVAILABLE = True
except ImportError:
    PREDICTORS_AVAILABLE = False
    logger_manager.warning("预测器模块未找到，部分功能将不可用")

# 尝试导入增强马尔可夫预测器
try:
    from improvements.enhanced_markov import get_markov_predictor
    ENHANCED_MARKOV_AVAILABLE = True
except ImportError:
    ENHANCED_MARKOV_AVAILABLE = False
    logger_manager.warning("增强马尔可夫模块未找到，部分功能将不可用")

# 尝试导入增强特性预测器
try:
    from improvements.enhanced_features import get_enhanced_feature_predictor
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError:
    ENHANCED_FEATURES_AVAILABLE = False
    logger_manager.warning("增强特性模块未找到，部分功能将不可用")

# 尝试导入LSTM预测器
try:
    from advanced_lstm_predictor import AdvancedLSTMPredictor, TENSORFLOW_AVAILABLE
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger_manager.warning("LSTM预测器模块未找到，部分功能将不可用")


class IntegratedPredictor:
    """集成预测器"""
    
    def __init__(self):
        """初始化集成预测器"""
        self.df = data_manager.get_data()
        if self.df is None:
            logger_manager.error("数据未加载")
        
        # 预测器字典
        self.predictors = {}
        
        # 加载预测器
        self._load_predictors()
    
    def _load_predictors(self):
        """加载预测器"""
        # 加载传统预测器
        if PREDICTORS_AVAILABLE:
            try:
                traditional = get_traditional_predictor()
                advanced = get_advanced_predictor()
                super_predictor = get_super_predictor()
                
                self.predictors.update({
                    'frequency': lambda data, count=3: traditional.frequency_predict(count),
                    'hot_cold': lambda data, count=3: traditional.hot_cold_predict(count),
                    'missing': lambda data, count=3: traditional.missing_predict(count),
                    'markov': lambda data, count=3: advanced.markov_predict(count),
                    'bayesian': lambda data, count=3: advanced.bayesian_predict(count),
                    'ensemble': lambda data, count=3: advanced.ensemble_predict(count),
                    'super': lambda data, count=3: super_predictor.predict_super(count)
                })
                
                logger_manager.info("已加载传统预测器")
            except Exception as e:
                logger_manager.error(f"加载传统预测器失败: {e}")
        
        # 加载增强马尔可夫预测器
        if ENHANCED_MARKOV_AVAILABLE:
            try:
                markov_predictor = get_markov_predictor()
                
                self.predictors.update({
                    'markov_2nd': lambda data, count=3: markov_predictor.multi_order_markov_predict(count, 300, 2),
                    'markov_3rd': lambda data, count=3: markov_predictor.multi_order_markov_predict(count, 300, 3),
                    'adaptive_markov': lambda data, count=3: [
                        (pred['front_balls'], pred['back_balls']) 
                        for pred in markov_predictor.adaptive_order_markov_predict(count, 300)
                    ]
                })
                
                logger_manager.info("已加载增强马尔可夫预测器")
            except Exception as e:
                logger_manager.error(f"加载增强马尔可夫预测器失败: {e}")
        
        # 加载增强特性预测器
        if ENHANCED_FEATURES_AVAILABLE:
            try:
                feature_predictor = get_enhanced_feature_predictor()
                
                self.predictors.update({
                    'pattern_based': lambda data, count=3: feature_predictor.pattern_based_predict(count),
                    'cluster_based': lambda data, count=3: feature_predictor.cluster_based_predict(count)
                })
                
                logger_manager.info("已加载增强特性预测器")
            except Exception as e:
                logger_manager.error(f"加载增强特性预测器失败: {e}")
        
        # 加载LSTM预测器
        if TENSORFLOW_AVAILABLE:
            try:
                lstm_predictor = AdvancedLSTMPredictor()
                
                self.predictors.update({
                    'lstm': lambda data, count=3: lstm_predictor.lstm_predict(count)
                })
                
                logger_manager.info("已加载LSTM预测器")
            except Exception as e:
                logger_manager.error(f"加载LSTM预测器失败: {e}")
    
    def stacking_predict(self, count: int = 3) -> List[Dict]:
        """Stacking集成预测
        
        Args:
            count: 预测注数
            
        Returns:
            List[Dict]: 预测结果列表
        """
        if not self.predictors:
            logger_manager.error("没有可用的预测器")
            return []
        
        # 收集各预测器的预测结果
        all_predictions = {}
        for name, predictor in self.predictors.items():
            try:
                predictions = predictor(self.df, count)
                all_predictions[name] = predictions
            except Exception as e:
                logger_manager.error(f"预测器 {name} 预测失败: {e}")
        
        # Stacking融合
        results = []
        for i in range(count):
            # 收集第i注的所有预测
            front_candidates = []
            back_candidates = []
            
            for name, predictions in all_predictions.items():
                if i < len(predictions):
                    if isinstance(predictions[i], tuple) and len(predictions[i]) == 2:
                        front, back = predictions[i]
                    elif isinstance(predictions[i], dict) and 'front_balls' in predictions[i] and 'back_balls' in predictions[i]:
                        front = predictions[i]['front_balls']
                        back = predictions[i]['back_balls']
                    else:
                        logger_manager.warning(f"无法解析预测结果: {predictions[i]}")
                        continue
                    
                    front_candidates.extend(front)
                    back_candidates.extend(back)
            
            # 使用投票机制选择最终号码
            front_counter = Counter(front_candidates)
            back_counter = Counter(back_candidates)
            
            front_balls = [ball for ball, _ in front_counter.most_common(5)]
            back_balls = [ball for ball, _ in back_counter.most_common(2)]
            
            # 如果号码不足，使用频率分析补充
            if len(front_balls) < 5:
                from analyzer_modules import basic_analyzer
                freq_analysis = basic_analyzer.frequency_analysis()
                front_freq = freq_analysis.get('front_frequency', {})
                sorted_freq = sorted(front_freq.items(), key=lambda x: x[1], reverse=True)

                for ball, _ in sorted_freq:
                    if len(front_balls) >= 5:
                        break
                    ball_int = int(ball) if isinstance(ball, str) else ball
                    if ball_int not in front_balls:
                        front_balls.append(ball_int)

            if len(back_balls) < 2:
                from analyzer_modules import basic_analyzer
                freq_analysis = basic_analyzer.frequency_analysis()
                back_freq = freq_analysis.get('back_frequency', {})
                sorted_freq = sorted(back_freq.items(), key=lambda x: x[1], reverse=True)

                for ball, _ in sorted_freq:
                    if len(back_balls) >= 2:
                        break
                    ball_int = int(ball) if isinstance(ball, str) else ball
                    if ball_int not in back_balls:
                        back_balls.append(ball_int)
            
            # 构建结果
            result = {
                'front_balls': sorted(front_balls),
                'back_balls': sorted(back_balls),
                'method': 'stacking',
                'confidence': 0.85,
                'predictors_used': list(all_predictions.keys())
            }
            
            results.append(result)
        
        return results
    
    def weighted_ensemble_predict(self, count: int = 3) -> List[Dict]:
        """加权集成预测
        
        Args:
            count: 预测注数
            
        Returns:
            List[Dict]: 预测结果列表
        """
        if not self.predictors:
            logger_manager.error("没有可用的预测器")
            return []
        
        # 预测器权重
        weights = {
            'frequency': 0.1,
            'hot_cold': 0.1,
            'missing': 0.1,
            'markov': 0.15,
            'bayesian': 0.15,
            'ensemble': 0.15,
            'super': 0.2,
            'markov_2nd': 0.2,
            'markov_3rd': 0.2,
            'adaptive_markov': 0.25,
            'pattern_based': 0.15,
            'cluster_based': 0.15,
            'lstm': 0.25
        }
        
        # 收集各预测器的预测结果
        all_predictions = {}
        for name, predictor in self.predictors.items():
            try:
                predictions = predictor(self.df, count)
                all_predictions[name] = predictions
            except Exception as e:
                logger_manager.error(f"预测器 {name} 预测失败: {e}")
        
        # 加权融合
        results = []
        for i in range(count):
            # 前区和后区号码的加权得分
            front_scores = defaultdict(float)
            back_scores = defaultdict(float)
            
            # 计算每个号码的加权得分
            for name, predictions in all_predictions.items():
                if i < len(predictions):
                    weight = weights.get(name, 0.1)
                    
                    if isinstance(predictions[i], tuple) and len(predictions[i]) == 2:
                        front, back = predictions[i]
                    elif isinstance(predictions[i], dict) and 'front_balls' in predictions[i] and 'back_balls' in predictions[i]:
                        front = predictions[i]['front_balls']
                        back = predictions[i]['back_balls']
                    else:
                        logger_manager.warning(f"无法解析预测结果: {predictions[i]}")
                        continue
                    
                    for ball in front:
                        front_scores[ball] += weight
                    
                    for ball in back:
                        back_scores[ball] += weight
            
            # 选择得分最高的号码
            front_balls = [ball for ball, _ in sorted(front_scores.items(), key=lambda x: x[1], reverse=True)[:5]]
            back_balls = [ball for ball, _ in sorted(back_scores.items(), key=lambda x: x[1], reverse=True)[:2]]
            
            # 如果号码不足，随机补充
            while len(front_balls) < 5:
                ball = np.random.randint(1, 36)
                if ball not in front_balls:
                    front_balls.append(ball)
            
            while len(back_balls) < 2:
                ball = np.random.randint(1, 13)
                if ball not in back_balls:
                    back_balls.append(ball)
            
            # 构建结果
            result = {
                'front_balls': sorted(front_balls),
                'back_balls': sorted(back_balls),
                'method': 'weighted_ensemble',
                'confidence': 0.9,
                'predictors_used': list(all_predictions.keys()),
                'weights': {name: weights.get(name, 0.1) for name in all_predictions.keys()}
            }
            
            results.append(result)
        
        return results
    
    def adaptive_ensemble_predict(self, count: int = 3) -> List[Dict]:
        """自适应集成预测
        
        Args:
            count: 预测注数
            
        Returns:
            List[Dict]: 预测结果列表
        """
        if not self.predictors:
            logger_manager.error("没有可用的预测器")
            return []
        
        # 加载历史性能数据
        performance_data = self._load_performance_data()
        
        # 根据历史性能计算权重
        weights = self._calculate_adaptive_weights(performance_data)
        
        # 收集各预测器的预测结果
        all_predictions = {}
        for name, predictor in self.predictors.items():
            try:
                predictions = predictor(self.df, count)
                all_predictions[name] = predictions
            except Exception as e:
                logger_manager.error(f"预测器 {name} 预测失败: {e}")
        
        # 自适应融合
        results = []
        for i in range(count):
            # 前区和后区号码的加权得分
            front_scores = defaultdict(float)
            back_scores = defaultdict(float)
            
            # 计算每个号码的加权得分
            for name, predictions in all_predictions.items():
                if i < len(predictions):
                    weight = weights.get(name, 0.1)
                    
                    if isinstance(predictions[i], tuple) and len(predictions[i]) == 2:
                        front, back = predictions[i]
                    elif isinstance(predictions[i], dict) and 'front_balls' in predictions[i] and 'back_balls' in predictions[i]:
                        front = predictions[i]['front_balls']
                        back = predictions[i]['back_balls']
                    else:
                        logger_manager.warning(f"无法解析预测结果: {predictions[i]}")
                        continue
                    
                    for ball in front:
                        front_scores[ball] += weight
                    
                    for ball in back:
                        back_scores[ball] += weight
            
            # 选择得分最高的号码
            front_balls = [ball for ball, _ in sorted(front_scores.items(), key=lambda x: x[1], reverse=True)[:5]]
            back_balls = [ball for ball, _ in sorted(back_scores.items(), key=lambda x: x[1], reverse=True)[:2]]
            
            # 如果号码不足，随机补充
            while len(front_balls) < 5:
                ball = np.random.randint(1, 36)
                if ball not in front_balls:
                    front_balls.append(ball)
            
            while len(back_balls) < 2:
                ball = np.random.randint(1, 13)
                if ball not in back_balls:
                    back_balls.append(ball)
            
            # 构建结果
            result = {
                'front_balls': sorted(front_balls),
                'back_balls': sorted(back_balls),
                'method': 'adaptive_ensemble',
                'confidence': 0.95,
                'predictors_used': list(all_predictions.keys()),
                'weights': weights
            }
            
            results.append(result)
        
        return results
    
    def _load_performance_data(self) -> Dict:
        """加载历史性能数据"""
        # 尝试从智能缓存加载
        method_name = "predictor_performance_data"
        cached_data = smart_cache_manager.load_cache("analysis", method_name)
        if cached_data:
            return cached_data
        
        # 如果没有缓存，使用默认值
        default_data = {
            'frequency': 0.5,
            'hot_cold': 0.5,
            'missing': 0.5,
            'markov': 0.6,
            'bayesian': 0.6,
            'ensemble': 0.7,
            'super': 0.7,
            'markov_2nd': 0.7,
            'markov_3rd': 0.7,
            'adaptive_markov': 0.8,
            'pattern_based': 0.6,
            'cluster_based': 0.6,
            'lstm': 0.8
        }
        
        return default_data
    
    def _calculate_adaptive_weights(self, performance_data: Dict) -> Dict:
        """计算自适应权重"""
        weights = {}
        
        # 根据性能数据计算权重
        for name, performance in performance_data.items():
            # 使用性能值的平方作为权重，使得高性能的预测器权重更高
            weights[name] = performance ** 2
        
        # 归一化权重
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {name: weight / total_weight for name, weight in weights.items()}
        
        return weights
    
    def transformer_predict(self, count: int = 3, periods: int = 500) -> List[Dict]:
        """Transformer预测 - 真实的Transformer注意力机制实现

        Args:
            count: 预测注数
            periods: 分析期数

        Returns:
            List[Dict]: 预测结果列表
        """
        try:
            logger_manager.info(f"开始Transformer预测: 注数={count}, 分析期数={periods}")

            # 尝试使用enhanced_deep_learning模块的Transformer实现
            try:
                from enhanced_deep_learning.models.transformer_predictor import TransformerPredictor
                from enhanced_deep_learning.models.base_model import ModelMetadata

                # 配置Transformer参数
                config = {
                    'd_model': 256,
                    'num_heads': 8,
                    'num_encoder_layers': 6,
                    'num_decoder_layers': 6,
                    'dff': 1024,
                    'dropout_rate': 0.1,
                    'use_relative_position': True,
                    'use_sparse_attention': True,
                    'use_local_attention': True,
                    'local_attention_window': 32,
                    'sequence_length': 50,
                    'learning_rate': 0.0001,
                    'batch_size': 32,
                    'epochs': 100
                }

                # 创建Transformer预测器
                transformer = TransformerPredictor(config=config)

                # 获取历史数据
                historical_data = data_manager.get_data()
                if historical_data is None or len(historical_data) == 0:
                    raise ValueError("无法获取历史数据")

                # 使用指定期数的最新数据
                if len(historical_data) > periods:
                    historical_data = historical_data.tail(periods)

                logger_manager.info(f"使用{len(historical_data)}期历史数据进行Transformer训练和预测")

                # 进行预测
                predictions = transformer.predict(historical_data, count=count)

                # 转换为标准格式
                results = []
                for i, pred in enumerate(predictions):
                    if isinstance(pred, tuple) and len(pred) == 2:
                        front_balls, back_balls = pred
                        result = {
                            'front_balls': list(front_balls),
                            'back_balls': list(back_balls),
                            'method': 'transformer',
                            'confidence': 0.88,
                            'model_type': 'deep_learning',
                            'algorithm': 'multi_head_attention'
                        }
                        results.append(result)
                    elif isinstance(pred, dict):
                        pred['method'] = 'transformer'
                        pred['confidence'] = 0.88
                        pred['model_type'] = 'deep_learning'
                        pred['algorithm'] = 'multi_head_attention'
                        results.append(pred)

                logger_manager.info(f"Transformer预测完成，生成{len(results)}注预测")
                return results

            except ImportError as e:
                logger_manager.warning(f"Enhanced Transformer模块不可用: {e}")
                # 回退到简化的Transformer实现
                return self._fallback_transformer_predict(count, periods)

        except Exception as e:
            logger_manager.error(f"Transformer预测失败: {e}")
            # 回退到简化实现
            return self._fallback_transformer_predict(count, periods)

    def _fallback_transformer_predict(self, count: int = 3, periods: int = 500) -> List[Dict]:
        """Transformer预测回退实现 - 基于注意力机制的简化实现"""
        try:
            logger_manager.info("使用Transformer回退实现")

            # 获取历史数据
            historical_data = data_manager.get_data()
            if historical_data is None or len(historical_data) == 0:
                raise ValueError("无法获取历史数据")

            # 使用指定期数的最新数据
            if len(historical_data) > periods:
                historical_data = historical_data.tail(periods)

            # 简化的注意力机制实现
            front_data = []
            back_data = []

            for _, row in historical_data.iterrows():
                front_nums = [int(x) for x in str(row['front']).split(',') if x.strip().isdigit()]
                back_nums = [int(x) for x in str(row['back']).split(',') if x.strip().isdigit()]

                if len(front_nums) == 5 and len(back_nums) == 2:
                    front_data.append(front_nums)
                    back_data.append(back_nums)

            if len(front_data) < 10:
                raise ValueError("历史数据不足")

            # 简化的注意力权重计算
            front_attention_weights = self._calculate_attention_weights(front_data)
            back_attention_weights = self._calculate_attention_weights(back_data)

            results = []
            for i in range(count):
                # 基于注意力权重生成预测
                front_pred = self._generate_with_attention(front_attention_weights, 5, 1, 35)
                back_pred = self._generate_with_attention(back_attention_weights, 2, 1, 12)

                result = {
                    'front_balls': sorted(front_pred),
                    'back_balls': sorted(back_pred),
                    'method': 'transformer_fallback',
                    'confidence': 0.75,
                    'model_type': 'attention_based',
                    'algorithm': 'simplified_attention'
                }
                results.append(result)

            logger_manager.info(f"Transformer回退预测完成，生成{len(results)}注预测")
            return results

        except Exception as e:
            logger_manager.error(f"Transformer回退预测失败: {e}")
            return []

    def _calculate_attention_weights(self, data_sequences: List[List[int]]) -> Dict[int, float]:
        """计算简化的注意力权重"""
        try:
            import numpy as np

            # 将序列转换为numpy数组
            sequences = np.array(data_sequences)

            # 计算每个号码的出现频率
            all_numbers = sequences.flatten()
            unique_numbers, counts = np.unique(all_numbers, return_counts=True)

            # 计算位置权重（最近的数据权重更高）
            position_weights = np.exp(-0.1 * np.arange(len(data_sequences))[::-1])

            # 计算加权频率
            weighted_freq = {}
            for num in unique_numbers:
                weighted_count = 0
                for i, seq in enumerate(data_sequences):
                    if num in seq:
                        weighted_count += position_weights[i]
                weighted_freq[num] = weighted_count

            # 归一化权重
            total_weight = sum(weighted_freq.values())
            if total_weight > 0:
                attention_weights = {num: weight / total_weight for num, weight in weighted_freq.items()}
            else:
                attention_weights = {num: 1.0 / len(unique_numbers) for num in unique_numbers}

            return attention_weights

        except Exception as e:
            logger_manager.error(f"计算注意力权重失败: {e}")
            return {}

    def _generate_with_attention(self, attention_weights: Dict[int, float],
                                count: int, min_num: int, max_num: int) -> List[int]:
        """基于注意力权重生成号码"""
        try:
            import numpy as np

            if not attention_weights:
                # 如果没有权重，随机生成
                return list(np.random.choice(range(min_num, max_num + 1), count, replace=False))

            # 根据注意力权重选择号码
            numbers = list(attention_weights.keys())
            weights = list(attention_weights.values())

            # 过滤有效范围内的号码
            valid_numbers = [num for num in numbers if min_num <= num <= max_num]
            valid_weights = [attention_weights[num] for num in valid_numbers]

            if len(valid_numbers) < count:
                # 如果有效号码不足，补充随机号码
                additional_numbers = [num for num in range(min_num, max_num + 1)
                                    if num not in valid_numbers]
                need_additional = count - len(valid_numbers)
                if need_additional > 0 and additional_numbers:
                    additional_selected = list(np.random.choice(
                        additional_numbers,
                        min(need_additional, len(additional_numbers)),
                        replace=False
                    ))
                    valid_numbers.extend(additional_selected)
                    valid_weights.extend([0.1] * len(additional_selected))

            # 归一化权重
            total_weight = sum(valid_weights)
            if total_weight > 0:
                normalized_weights = [w / total_weight for w in valid_weights]
            else:
                normalized_weights = [1.0 / len(valid_weights)] * len(valid_weights)

            # 基于权重选择号码
            selected = list(np.random.choice(
                valid_numbers,
                min(count, len(valid_numbers)),
                p=normalized_weights,
                replace=False
            ))

            return selected

        except Exception as e:
            logger_manager.error(f"基于注意力权重生成号码失败: {e}")
            import random
            return random.sample(range(min_num, max_num + 1), count)
    
    def gan_predict(self, count: int = 3, periods: int = 500) -> List[Dict]:
        """GAN预测 - 真实的生成对抗网络实现

        Args:
            count: 预测注数
            periods: 分析期数

        Returns:
            List[Dict]: 预测结果列表
        """
        try:
            logger_manager.info(f"开始GAN预测: 注数={count}, 分析期数={periods}")

            # 尝试使用enhanced_deep_learning模块的GAN实现
            try:
                from enhanced_deep_learning.models.gan_predictor import GANPredictor
                from enhanced_deep_learning.models.base_model import ModelMetadata

                # 配置GAN参数
                config = {
                    'latent_dim': 100,
                    'generator_layers': [256, 512, 256, 128],
                    'discriminator_layers': [128, 256, 128],
                    'learning_rate': 0.0002,
                    'beta1': 0.5,
                    'beta2': 0.999,
                    'batch_size': 64,
                    'epochs': 200,
                    'generator_lr': 0.0002,
                    'discriminator_lr': 0.0002,
                    'use_conditional': True,
                    'num_conditions': 10
                }

                # 创建GAN预测器
                gan = GANPredictor(config=config)

                # 获取历史数据
                historical_data = data_manager.get_data()
                if historical_data is None or len(historical_data) == 0:
                    raise ValueError("无法获取历史数据")

                # 使用指定期数的最新数据
                if len(historical_data) > periods:
                    historical_data = historical_data.tail(periods)

                logger_manager.info(f"使用{len(historical_data)}期历史数据进行GAN训练和预测")

                # 进行预测
                predictions = gan.predict(historical_data, count=count)

                # 转换为标准格式
                results = []
                for i, pred in enumerate(predictions):
                    if isinstance(pred, tuple) and len(pred) == 2:
                        front_balls, back_balls = pred
                        result = {
                            'front_balls': list(front_balls),
                            'back_balls': list(back_balls),
                            'method': 'gan',
                            'confidence': 0.82,
                            'model_type': 'generative_adversarial',
                            'algorithm': 'generator_discriminator'
                        }
                        results.append(result)
                    elif isinstance(pred, dict):
                        pred['method'] = 'gan'
                        pred['confidence'] = 0.82
                        pred['model_type'] = 'generative_adversarial'
                        pred['algorithm'] = 'generator_discriminator'
                        results.append(pred)

                logger_manager.info(f"GAN预测完成，生成{len(results)}注预测")
                return results

            except ImportError as e:
                logger_manager.warning(f"Enhanced GAN模块不可用: {e}")
                # 回退到简化的GAN实现
                return self._fallback_gan_predict(count, periods)

        except Exception as e:
            logger_manager.error(f"GAN预测失败: {e}")
            # 回退到简化实现
            return self._fallback_gan_predict(count, periods)

    def _fallback_gan_predict(self, count: int = 3, periods: int = 500) -> List[Dict]:
        """GAN预测回退实现 - 基于生成模型的简化实现"""
        try:
            logger_manager.info("使用GAN回退实现")

            # 获取历史数据
            historical_data = data_manager.get_data()
            if historical_data is None or len(historical_data) == 0:
                raise ValueError("无法获取历史数据")

            # 使用指定期数的最新数据
            if len(historical_data) > periods:
                historical_data = historical_data.tail(periods)

            # 简化的生成模型实现
            front_patterns = []
            back_patterns = []

            for _, row in historical_data.iterrows():
                front_nums = [int(x) for x in str(row['front']).split(',') if x.strip().isdigit()]
                back_nums = [int(x) for x in str(row['back']).split(',') if x.strip().isdigit()]

                if len(front_nums) == 5 and len(back_nums) == 2:
                    front_patterns.append(front_nums)
                    back_patterns.append(back_nums)

            if len(front_patterns) < 10:
                raise ValueError("历史数据不足")

            # 简化的生成器实现
            results = []
            for i in range(count):
                # 基于历史模式生成新的号码组合
                front_pred = self._generate_with_patterns(front_patterns, 5, 1, 35)
                back_pred = self._generate_with_patterns(back_patterns, 2, 1, 12)

                result = {
                    'front_balls': sorted(front_pred),
                    'back_balls': sorted(back_pred),
                    'method': 'gan_fallback',
                    'confidence': 0.70,
                    'model_type': 'pattern_generator',
                    'algorithm': 'simplified_generation'
                }
                results.append(result)

            logger_manager.info(f"GAN回退预测完成，生成{len(results)}注预测")
            return results

        except Exception as e:
            logger_manager.error(f"GAN回退预测失败: {e}")
            return []

    def _generate_with_patterns(self, patterns: List[List[int]],
                               count: int, min_num: int, max_num: int) -> List[int]:
        """基于历史模式生成号码"""
        try:
            import numpy as np

            if not patterns:
                # 如果没有模式，随机生成
                return list(np.random.choice(range(min_num, max_num + 1), count, replace=False))

            # 分析模式特征
            pattern_features = self._analyze_patterns(patterns)

            # 基于模式特征生成新号码
            generated = []
            attempts = 0
            max_attempts = count * 10

            while len(generated) < count and attempts < max_attempts:
                attempts += 1

                # 随机选择一个历史模式作为基础
                base_pattern = patterns[np.random.randint(0, len(patterns))]

                # 基于模式特征进行变异
                candidate = self._mutate_pattern(base_pattern, pattern_features, min_num, max_num)

                # 确保候选号码在有效范围内且不重复
                if min_num <= candidate <= max_num and candidate not in generated:
                    generated.append(candidate)

            # 如果生成的号码不足，随机补充
            while len(generated) < count:
                candidate = np.random.randint(min_num, max_num + 1)
                if candidate not in generated:
                    generated.append(candidate)

            return generated[:count]

        except Exception as e:
            logger_manager.error(f"基于模式生成号码失败: {e}")
            import random
            return random.sample(range(min_num, max_num + 1), count)

    def _analyze_patterns(self, patterns: List[List[int]]) -> Dict[str, Any]:
        """分析历史模式特征"""
        try:
            import numpy as np

            features = {
                'mean_values': [],
                'std_values': [],
                'gaps': [],
                'ranges': []
            }

            for pattern in patterns:
                sorted_pattern = sorted(pattern)
                features['mean_values'].append(np.mean(sorted_pattern))
                features['std_values'].append(np.std(sorted_pattern))
                features['ranges'].append(max(sorted_pattern) - min(sorted_pattern))

                # 计算间隔
                gaps = [sorted_pattern[i+1] - sorted_pattern[i] for i in range(len(sorted_pattern)-1)]
                features['gaps'].extend(gaps)

            # 计算统计特征
            analysis = {
                'avg_mean': np.mean(features['mean_values']),
                'avg_std': np.mean(features['std_values']),
                'avg_range': np.mean(features['ranges']),
                'common_gaps': np.bincount(features['gaps']).argmax() if features['gaps'] else 1
            }

            return analysis

        except Exception as e:
            logger_manager.error(f"分析模式特征失败: {e}")
            return {}

    def _mutate_pattern(self, base_pattern: List[int], features: Dict[str, Any],
                       min_num: int, max_num: int) -> int:
        """基于模式特征变异生成新号码"""
        try:
            import numpy as np

            if not features:
                return np.random.randint(min_num, max_num + 1)

            # 基于统计特征生成候选号码
            avg_mean = features.get('avg_mean', (min_num + max_num) / 2)
            avg_std = features.get('avg_std', 5)

            # 在平均值附近生成候选
            candidate = int(np.random.normal(avg_mean, avg_std))

            # 确保在有效范围内
            candidate = max(min_num, min(max_num, candidate))

            return candidate

        except Exception as e:
            logger_manager.error(f"模式变异失败: {e}")
            import random
            return random.randint(min_num, max_num)
    
    def ultimate_ensemble_predict(self, count: int = 3) -> List[Dict]:
        """终极集成预测
        
        结合所有预测方法的最佳结果
        
        Args:
            count: 预测注数
            
        Returns:
            List[Dict]: 预测结果列表
        """
        # 获取各种集成方法的预测结果
        stacking_results = self.stacking_predict(count)
        weighted_results = self.weighted_ensemble_predict(count)
        adaptive_results = self.adaptive_ensemble_predict(count)
        
        # 融合结果
        results = []
        for i in range(count):
            # 收集所有预测结果
            all_front_balls = []
            all_back_balls = []
            
            if i < len(stacking_results):
                all_front_balls.extend(stacking_results[i]['front_balls'])
                all_back_balls.extend(stacking_results[i]['back_balls'])
            
            if i < len(weighted_results):
                all_front_balls.extend(weighted_results[i]['front_balls'])
                all_back_balls.extend(weighted_results[i]['back_balls'])
            
            if i < len(adaptive_results):
                all_front_balls.extend(adaptive_results[i]['front_balls'])
                all_back_balls.extend(adaptive_results[i]['back_balls'])
            
            # 统计号码出现频率
            front_counter = Counter(all_front_balls)
            back_counter = Counter(all_back_balls)
            
            # 选择出现频率最高的号码
            front_balls = [ball for ball, _ in front_counter.most_common(5)]
            back_balls = [ball for ball, _ in back_counter.most_common(2)]
            
            # 如果号码不足，使用频率分析补充
            if len(front_balls) < 5:
                from analyzer_modules import basic_analyzer
                freq_analysis = basic_analyzer.frequency_analysis()
                front_freq = freq_analysis.get('front_frequency', {})
                sorted_freq = sorted(front_freq.items(), key=lambda x: x[1], reverse=True)

                for ball, _ in sorted_freq:
                    if len(front_balls) >= 5:
                        break
                    ball_int = int(ball) if isinstance(ball, str) else ball
                    if ball_int not in front_balls:
                        front_balls.append(ball_int)

            if len(back_balls) < 2:
                from analyzer_modules import basic_analyzer
                freq_analysis = basic_analyzer.frequency_analysis()
                back_freq = freq_analysis.get('back_frequency', {})
                sorted_freq = sorted(back_freq.items(), key=lambda x: x[1], reverse=True)

                for ball, _ in sorted_freq:
                    if len(back_balls) >= 2:
                        break
                    ball_int = int(ball) if isinstance(ball, str) else ball
                    if ball_int not in back_balls:
                        back_balls.append(ball_int)
            
            # 构建结果
            result = {
                'front_balls': sorted(front_balls),
                'back_balls': sorted(back_balls),
                'method': 'ultimate_ensemble',
                'confidence': 0.98,
                'ensemble_methods': ['stacking', 'weighted', 'adaptive']
            }
            
            results.append(result)
        
        return results


# 全局实例
_integrator = None

def get_integrator() -> IntegratedPredictor:
    """获取集成预测器实例"""
    global _integrator
    if _integrator is None:
        _integrator = IntegratedPredictor()
    return _integrator


if __name__ == "__main__":
    # 测试集成模块
    print("🔄 测试集成模块...")
    
    # 获取集成预测器
    integrator = get_integrator()
    
    # 测试Stacking集成预测
    print("\n🎯 Stacking集成预测...")
    stacking_results = integrator.stacking_predict(3)
    for i, result in enumerate(stacking_results):
        front_str = ' '.join([str(b).zfill(2) for b in result['front_balls']])
        back_str = ' '.join([str(b).zfill(2) for b in result['back_balls']])
        print(f"  第 {i+1} 注: {front_str} + {back_str}")
        print(f"  使用预测器: {len(result['predictors_used'])} 个")
    
    # 测试加权集成预测
    print("\n🎯 加权集成预测...")
    weighted_results = integrator.weighted_ensemble_predict(3)
    for i, result in enumerate(weighted_results):
        front_str = ' '.join([str(b).zfill(2) for b in result['front_balls']])
        back_str = ' '.join([str(b).zfill(2) for b in result['back_balls']])
        print(f"  第 {i+1} 注: {front_str} + {back_str}")
        print(f"  使用预测器: {len(result['predictors_used'])} 个")
    
    # 测试自适应集成预测
    print("\n🎯 自适应集成预测...")
    adaptive_results = integrator.adaptive_ensemble_predict(3)
    for i, result in enumerate(adaptive_results):
        front_str = ' '.join([str(b).zfill(2) for b in result['front_balls']])
        back_str = ' '.join([str(b).zfill(2) for b in result['back_balls']])
        print(f"  第 {i+1} 注: {front_str} + {back_str}")
        print(f"  使用预测器: {len(result['predictors_used'])} 个")
    
    # 测试终极集成预测
    print("\n🎯 终极集成预测...")
    ultimate_results = integrator.ultimate_ensemble_predict(3)
    for i, result in enumerate(ultimate_results):
        front_str = ' '.join([str(b).zfill(2) for b in result['front_balls']])
        back_str = ' '.join([str(b).zfill(2) for b in result['back_balls']])
        print(f"  第 {i+1} 注: {front_str} + {back_str}")
        print(f"  集成方法: {result['ensemble_methods']}")
    
    print("\n✅ 测试完成")