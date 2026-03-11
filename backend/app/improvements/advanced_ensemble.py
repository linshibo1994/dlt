#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
高级集成学习策略
基于Stacking、Voting、Blending等方法的改进
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.model_selection import KFold
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

try:
    from core_modules import logger_manager, data_manager
except ImportError:
    from core.core_modules import logger_manager, data_manager


class AdvancedEnsemblePredictor:
    """高级集成预测器"""
    
    def __init__(self):
        self.base_predictors = {}
        self.ensemble_weights = {}
        self.performance_history = {}
        
    def register_predictor(self, name: str, predictor, weight: float = 1.0):
        """注册基础预测器"""
        self.base_predictors[name] = predictor
        self.ensemble_weights[name] = weight
        self.performance_history[name] = []
    
    def stacking_predict(self, count: int = 1) -> List[Tuple[List[int], List[int]]]:
        """基于Stacking的集成预测"""
        logger_manager.info(f"Stacking集成预测，注数: {count}")
        
        # 收集各基础预测器的预测结果
        base_predictions = {}
        for name, predictor in self.base_predictors.items():
            try:
                if hasattr(predictor, 'predict'):
                    pred = predictor.predict(count)
                elif hasattr(predictor, 'frequency_predict'):
                    pred = predictor.frequency_predict(count)
                else:
                    continue
                base_predictions[name] = pred
            except Exception as e:
                logger_manager.warning(f"预测器 {name} 预测失败: {e}")
        
        # Stacking融合
        final_predictions = []
        for i in range(count):
            # 收集第i注的所有预测
            front_candidates = []
            back_candidates = []
            
            for name, preds in base_predictions.items():
                if i < len(preds):
                    front, back = preds[i]
                    front_candidates.extend(front)
                    back_candidates.extend(back)
            
            # 使用投票机制选择最终号码
            front_final = self._voting_selection(front_candidates, 5, 1, 35)
            back_final = self._voting_selection(back_candidates, 2, 1, 12)
            
            final_predictions.append((sorted(front_final), sorted(back_final)))

        return final_predictions

    def stacking_predict_with_cv(self, count: int = 1, n_folds: int = 5,
                                  use_out_of_fold: bool = True) -> List[Tuple[List[int], List[int]]]:
        """基于K-Fold交叉验证的Stacking集成预测

        使用K-Fold交叉验证生成out-of-fold预测，减少过拟合风险，
        提高集成预测的泛化能力。

        Args:
            count: 预测注数
            n_folds: K-Fold折数，默认5折
            use_out_of_fold: 是否使用out-of-fold预测，True为标准Stacking

        Returns:
            预测结果列表
        """
        logger_manager.info(f"K-Fold Stacking集成预测，注数: {count}, 折数: {n_folds}")

        if len(self.base_predictors) == 0:
            logger_manager.warning("没有注册基础预测器")
            return []

        # 收集各基础预测器的预测结果
        base_predictions = {}
        predictor_names = list(self.base_predictors.keys())

        for name, predictor in self.base_predictors.items():
            try:
                if hasattr(predictor, 'predict'):
                    pred = predictor.predict(count * n_folds)  # 获取更多预测用于交叉验证
                elif hasattr(predictor, 'frequency_predict'):
                    pred = predictor.frequency_predict(count * n_folds)
                else:
                    continue
                base_predictions[name] = pred
            except Exception as e:
                logger_manager.warning(f"预测器 {name} 预测失败: {e}")

        if not base_predictions:
            logger_manager.warning("没有有效的基础预测结果")
            return []

        # K-Fold交叉验证生成元特征
        if use_out_of_fold and len(base_predictions) >= 2:
            meta_features = self._generate_oof_predictions(base_predictions, n_folds, count)
        else:
            # 不使用交叉验证时，直接聚合
            meta_features = None

        # Stacking融合
        final_predictions = []
        for i in range(count):
            front_candidates = []
            back_candidates = []
            front_weights = []
            back_weights = []

            for name, preds in base_predictions.items():
                if i < len(preds):
                    front, back = preds[i]

                    # 如果有元特征，使用元特征权重
                    if meta_features is not None and name in meta_features:
                        weight = meta_features[name].get('weight', 1.0)
                    else:
                        weight = self.ensemble_weights.get(name, 1.0)

                    # 加权添加候选号码
                    for num in front:
                        front_candidates.append(num)
                        front_weights.append(weight)

                    for num in back:
                        back_candidates.append(num)
                        back_weights.append(weight)

            # 使用加权投票选择最终号码
            front_final = self._weighted_voting_selection(
                front_candidates, front_weights, 5, 1, 35
            )
            back_final = self._weighted_voting_selection(
                back_candidates, back_weights, 2, 1, 12
            )

            final_predictions.append((sorted(front_final), sorted(back_final)))

        logger_manager.info(f"K-Fold Stacking预测完成，生成 {len(final_predictions)} 注")
        return final_predictions

    def _generate_oof_predictions(self, base_predictions: Dict, n_folds: int,
                                   count: int) -> Dict:
        """生成Out-of-Fold预测元特征

        通过K-Fold交叉验证，计算每个预测器在验证集上的表现，
        作为元特征用于第二层聚合。

        Args:
            base_predictions: 基础预测器的预测结果
            n_folds: 折数
            count: 目标预测注数

        Returns:
            元特征字典，包含每个预测器的权重和可信度
        """
        meta_features = {}

        # 获取所有预测器的预测数量
        min_predictions = min(len(preds) for preds in base_predictions.values())
        if min_predictions < n_folds:
            logger_manager.warning(f"预测数量不足以进行{n_folds}折交叉验证")
            return {}

        # 创建K-Fold分割
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        indices = list(range(min_predictions))

        # 对每个预测器计算OOF性能
        for name, preds in base_predictions.items():
            oof_scores = []

            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices)):
                # 计算验证集上的一致性得分
                # 验证集预测与其他预测器的一致性作为性能指标
                val_predictions = [preds[i] for i in val_idx if i < len(preds)]

                if val_predictions:
                    # 计算与其他预测器的一致性
                    consistency_score = self._calculate_fold_consistency(
                        name, val_predictions, base_predictions, val_idx
                    )
                    oof_scores.append(consistency_score)

            # 计算平均OOF得分
            if oof_scores:
                avg_oof_score = np.mean(oof_scores)
                std_oof_score = np.std(oof_scores)

                # 稳定性调整：标准差小的预测器更可靠
                stability_factor = 1.0 / (1.0 + std_oof_score)

                # 最终权重 = 平均得分 * 稳定性因子
                final_weight = avg_oof_score * stability_factor

                meta_features[name] = {
                    'weight': max(0.1, final_weight),  # 最小权重0.1
                    'avg_score': avg_oof_score,
                    'std_score': std_oof_score,
                    'stability': stability_factor,
                    'n_folds': n_folds
                }

                logger_manager.debug(
                    f"预测器 {name} OOF元特征: 权重={final_weight:.4f}, "
                    f"得分={avg_oof_score:.4f}, 稳定性={stability_factor:.4f}"
                )

        return meta_features

    def _calculate_fold_consistency(self, predictor_name: str,
                                     predictions: List[Tuple[List[int], List[int]]],
                                     all_predictions: Dict, indices: List[int]) -> float:
        """计算预测器在某折验证集上与其他预测器的一致性

        Args:
            predictor_name: 当前预测器名称
            predictions: 当前预测器的预测结果
            all_predictions: 所有预测器的预测结果
            indices: 验证集索引

        Returns:
            一致性得分 (0-1)
        """
        if not predictions:
            return 0.0

        total_consistency = 0.0
        comparison_count = 0

        for idx, (front, back) in zip(indices, predictions):
            for other_name, other_preds in all_predictions.items():
                if other_name == predictor_name:
                    continue
                if idx >= len(other_preds):
                    continue

                other_front, other_back = other_preds[idx]

                # 计算前区重叠度
                front_overlap = len(set(front) & set(other_front)) / 5.0

                # 计算后区重叠度
                back_overlap = len(set(back) & set(other_back)) / 2.0

                # 综合一致性 (前区权重0.7, 后区权重0.3)
                consistency = front_overlap * 0.7 + back_overlap * 0.3
                total_consistency += consistency
                comparison_count += 1

        return total_consistency / comparison_count if comparison_count > 0 else 0.0

    def _weighted_voting_selection(self, candidates: List[int], weights: List[float],
                                    target_count: int, min_val: int, max_val: int) -> List[int]:
        """加权投票选择机制

        Args:
            candidates: 候选号码列表
            weights: 对应的权重列表
            target_count: 目标选择数量
            min_val: 最小有效值
            max_val: 最大有效值

        Returns:
            选中的号码列表
        """
        if not candidates:
            # 如果没有候选，随机生成
            return list(np.random.choice(range(min_val, max_val + 1),
                                         size=target_count, replace=False))

        # 加权投票统计
        weighted_votes = {}
        for num, weight in zip(candidates, weights):
            if min_val <= num <= max_val:
                weighted_votes[num] = weighted_votes.get(num, 0) + weight

        # 按加权票数排序
        sorted_candidates = sorted(weighted_votes.items(), key=lambda x: x[1], reverse=True)

        # 选择得分最高的号码
        selected = []
        for num, score in sorted_candidates:
            if len(selected) < target_count:
                selected.append(num)

        # 如果数量不足，随机补充
        while len(selected) < target_count:
            candidate = np.random.randint(min_val, max_val + 1)
            if candidate not in selected:
                selected.append(candidate)

        return selected[:target_count]
    
    def weighted_ensemble_predict(self, count: int = 1) -> List[Tuple[List[int], List[int]]]:
        """基于权重的集成预测"""
        logger_manager.info(f"权重集成预测，注数: {count}")
        
        # 收集预测结果和权重
        weighted_predictions = []
        total_weight = sum(self.ensemble_weights.values())
        
        for name, predictor in self.base_predictors.items():
            try:
                weight = self.ensemble_weights[name] / total_weight
                if hasattr(predictor, 'predict'):
                    pred = predictor.predict(count)
                elif hasattr(predictor, 'frequency_predict'):
                    pred = predictor.frequency_predict(count)
                else:
                    continue
                
                weighted_predictions.append((name, pred, weight))
            except Exception as e:
                logger_manager.warning(f"预测器 {name} 预测失败: {e}")
        
        # 权重融合
        final_predictions = []
        for i in range(count):
            front_scores = {}
            back_scores = {}
            
            # 计算每个号码的加权得分
            for name, preds, weight in weighted_predictions:
                if i < len(preds):
                    front, back = preds[i]
                    
                    for num in front:
                        front_scores[num] = front_scores.get(num, 0) + weight
                    
                    for num in back:
                        back_scores[num] = back_scores.get(num, 0) + weight
            
            # 选择得分最高的号码
            front_final = sorted(front_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            back_final = sorted(back_scores.items(), key=lambda x: x[1], reverse=True)[:2]
            
            front_numbers = [num for num, score in front_final]
            back_numbers = [num for num, score in back_final]
            
            final_predictions.append((sorted(front_numbers), sorted(back_numbers)))
        
        return final_predictions
    
    def adaptive_ensemble_predict(self, count: int = 1) -> List[Tuple[List[int], List[int]]]:
        """自适应集成预测（基于历史表现动态调整权重）"""
        logger_manager.info(f"自适应集成预测，注数: {count}")
        
        # 更新权重基于历史表现
        self._update_adaptive_weights()
        
        # 使用更新后的权重进行预测
        return self.weighted_ensemble_predict(count)

    def boosting_ensemble_predict(self, count: int = 1, n_iterations: int = 10,
                                   learning_rate: float = 0.1) -> List[Tuple[List[int], List[int]]]:
        """基于Boosting思想的集成预测

        借鉴AdaBoost的核心思想：
        1. 初始化等权重
        2. 迭代训练，关注"困难"样本（预测不一致的号码）
        3. 根据预测器表现调整权重
        4. 最终加权聚合

        Args:
            count: 预测注数
            n_iterations: Boosting迭代次数
            learning_rate: 学习率，控制权重更新幅度

        Returns:
            预测结果列表
        """
        logger_manager.info(f"Boosting集成预测，注数: {count}, 迭代: {n_iterations}")

        if len(self.base_predictors) == 0:
            logger_manager.warning("没有注册基础预测器")
            return []

        # 初始化预测器权重（等权重）
        n_predictors = len(self.base_predictors)
        predictor_weights = {name: 1.0 / n_predictors for name in self.base_predictors.keys()}

        # 初始化号码权重（用于关注"困难"号码）
        front_number_weights = {i: 1.0 for i in range(1, 36)}
        back_number_weights = {i: 1.0 for i in range(1, 13)}

        # 收集所有预测器的预测结果
        all_predictions = {}
        for name, predictor in self.base_predictors.items():
            try:
                if hasattr(predictor, 'predict'):
                    pred = predictor.predict(count)
                elif hasattr(predictor, 'frequency_predict'):
                    pred = predictor.frequency_predict(count)
                else:
                    continue
                all_predictions[name] = pred
            except Exception as e:
                logger_manager.warning(f"预测器 {name} 预测失败: {e}")

        if not all_predictions:
            logger_manager.warning("没有有效的基础预测结果")
            return []

        # Boosting迭代
        for iteration in range(n_iterations):
            # 计算每个预测器的"错误率"（与集成结果的不一致度）
            predictor_errors = self._calculate_boosting_errors(
                all_predictions, predictor_weights, front_number_weights, back_number_weights
            )

            # 更新预测器权重
            for name, error_rate in predictor_errors.items():
                if error_rate < 0.5:  # 只有错误率小于0.5的预测器才有正贡献
                    # AdaBoost权重更新公式
                    alpha = learning_rate * 0.5 * np.log((1 - error_rate + 1e-10) / (error_rate + 1e-10))
                    predictor_weights[name] = predictor_weights[name] * np.exp(alpha)
                else:
                    # 错误率高的预测器降权
                    predictor_weights[name] = predictor_weights[name] * 0.5

            # 归一化预测器权重
            total_weight = sum(predictor_weights.values())
            if total_weight > 0:
                predictor_weights = {k: v / total_weight for k, v in predictor_weights.items()}

            # 更新号码权重（关注预测不一致的号码）
            front_number_weights, back_number_weights = self._update_number_weights(
                all_predictions, predictor_weights, front_number_weights, back_number_weights,
                learning_rate
            )

            logger_manager.debug(f"Boosting迭代 {iteration + 1}: 权重更新完成")

        # 使用最终权重进行加权预测
        final_predictions = []
        for i in range(count):
            front_scores = {}
            back_scores = {}

            # 计算每个号码的加权得分
            for name, preds in all_predictions.items():
                if i < len(preds):
                    front, back = preds[i]
                    weight = predictor_weights.get(name, 0)

                    for num in front:
                        # 结合预测器权重和号码权重
                        combined_weight = weight * front_number_weights.get(num, 1.0)
                        front_scores[num] = front_scores.get(num, 0) + combined_weight

                    for num in back:
                        combined_weight = weight * back_number_weights.get(num, 1.0)
                        back_scores[num] = back_scores.get(num, 0) + combined_weight

            # 选择得分最高的号码
            front_final = sorted(front_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            back_final = sorted(back_scores.items(), key=lambda x: x[1], reverse=True)[:2]

            front_numbers = [num for num, score in front_final]
            back_numbers = [num for num, score in back_final]

            # 确保号码数量足够
            while len(front_numbers) < 5:
                candidate = np.random.randint(1, 36)
                if candidate not in front_numbers:
                    front_numbers.append(candidate)

            while len(back_numbers) < 2:
                candidate = np.random.randint(1, 13)
                if candidate not in back_numbers:
                    back_numbers.append(candidate)

            final_predictions.append((sorted(front_numbers), sorted(back_numbers)))

        logger_manager.info(f"Boosting预测完成，最终权重: {predictor_weights}")
        return final_predictions

    def _calculate_boosting_errors(self, all_predictions: Dict, predictor_weights: Dict,
                                    front_number_weights: Dict, back_number_weights: Dict) -> Dict:
        """计算每个预测器的Boosting错误率

        错误率定义为：预测器与加权集成结果的不一致程度

        Args:
            all_predictions: 所有预测器的预测结果
            predictor_weights: 预测器权重
            front_number_weights: 前区号码权重
            back_number_weights: 后区号码权重

        Returns:
            每个预测器的错误率字典
        """
        errors = {}

        # 首先计算当前加权集成结果
        ensemble_front_scores = {}
        ensemble_back_scores = {}

        for name, preds in all_predictions.items():
            weight = predictor_weights.get(name, 0)
            if preds and len(preds) > 0:
                front, back = preds[0]
                for num in front:
                    ensemble_front_scores[num] = ensemble_front_scores.get(num, 0) + weight
                for num in back:
                    ensemble_back_scores[num] = ensemble_back_scores.get(num, 0) + weight

        # 获取集成预测的号码
        ensemble_front = set(num for num, _ in sorted(
            ensemble_front_scores.items(), key=lambda x: x[1], reverse=True)[:5])
        ensemble_back = set(num for num, _ in sorted(
            ensemble_back_scores.items(), key=lambda x: x[1], reverse=True)[:2])

        # 计算每个预测器与集成结果的不一致度
        for name, preds in all_predictions.items():
            if preds and len(preds) > 0:
                front, back = preds[0]
                front_set = set(front)
                back_set = set(back)

                # 前区不一致度（考虑号码权重）
                front_error = 0.0
                for num in front_set:
                    if num not in ensemble_front:
                        front_error += front_number_weights.get(num, 1.0)

                # 后区不一致度
                back_error = 0.0
                for num in back_set:
                    if num not in ensemble_back:
                        back_error += back_number_weights.get(num, 1.0)

                # 综合错误率（归一化到0-1）
                max_front_error = sum(front_number_weights.get(num, 1.0) for num in front_set)
                max_back_error = sum(back_number_weights.get(num, 1.0) for num in back_set)

                front_error_rate = front_error / (max_front_error + 1e-10)
                back_error_rate = back_error / (max_back_error + 1e-10)

                # 综合错误率（前区权重0.7, 后区权重0.3）
                total_error = front_error_rate * 0.7 + back_error_rate * 0.3
                errors[name] = min(0.999, total_error)  # 限制最大错误率
            else:
                errors[name] = 0.5  # 默认错误率

        return errors

    def _update_number_weights(self, all_predictions: Dict, predictor_weights: Dict,
                                front_weights: Dict, back_weights: Dict,
                                learning_rate: float) -> Tuple[Dict, Dict]:
        """更新号码权重

        预测不一致的号码增加权重（更难预测的号码需要更多关注）

        Args:
            all_predictions: 所有预测器的预测结果
            predictor_weights: 预测器权重
            front_weights: 当前前区号码权重
            back_weights: 当前后区号码权重
            learning_rate: 学习率

        Returns:
            更新后的前区和后区号码权重
        """
        # 统计每个号码被预测的次数（加权）
        front_vote_scores = {}
        back_vote_scores = {}

        for name, preds in all_predictions.items():
            weight = predictor_weights.get(name, 0)
            if preds and len(preds) > 0:
                front, back = preds[0]
                for num in front:
                    front_vote_scores[num] = front_vote_scores.get(num, 0) + weight
                for num in back:
                    back_vote_scores[num] = back_vote_scores.get(num, 0) + weight

        # 计算投票得分的统计信息
        front_scores = list(front_vote_scores.values())
        back_scores = list(back_vote_scores.values())

        if front_scores:
            front_mean = np.mean(front_scores)
            front_std = np.std(front_scores) + 1e-10
        else:
            front_mean, front_std = 1.0, 1.0

        if back_scores:
            back_mean = np.mean(back_scores)
            back_std = np.std(back_scores) + 1e-10
        else:
            back_mean, back_std = 1.0, 1.0

        # 更新前区号码权重
        new_front_weights = {}
        for num in range(1, 36):
            vote_score = front_vote_scores.get(num, 0)
            # 投票得分低的号码（不一致）增加权重
            z_score = (vote_score - front_mean) / front_std
            # 使用sigmoid函数平滑权重更新
            weight_update = 1.0 / (1.0 + np.exp(z_score))
            new_front_weights[num] = front_weights[num] * (1 + learning_rate * (weight_update - 0.5))
            new_front_weights[num] = max(0.1, min(10.0, new_front_weights[num]))  # 限制权重范围

        # 更新后区号码权重
        new_back_weights = {}
        for num in range(1, 13):
            vote_score = back_vote_scores.get(num, 0)
            z_score = (vote_score - back_mean) / back_std
            weight_update = 1.0 / (1.0 + np.exp(z_score))
            new_back_weights[num] = back_weights[num] * (1 + learning_rate * (weight_update - 0.5))
            new_back_weights[num] = max(0.1, min(10.0, new_back_weights[num]))

        return new_front_weights, new_back_weights
    
    def _update_adaptive_weights(self):
        """基于历史表现更新权重（添加智能早停机制）"""
        # 尝试导入智能早停机制，失败时使用简单的早停逻辑
        try:
            from enhanced_deep_learning.utils.intelligent_early_stopping import GeneralIntelligentEarlyStopping
            early_stopping = GeneralIntelligentEarlyStopping(
                patience=20,  # 连续20次相同结果时停止
                min_delta=1e-6,
                verbose=0  # 静默模式
            )
            early_stopping.reset()
            use_intelligent_early_stopping = True
        except ImportError:
            # 回退到简单的早停逻辑
            use_intelligent_early_stopping = False
            patience_counter = 0
            max_patience = 20
            min_delta = 1e-6
            best_weight_diff = float('inf')

        # 记录权重变化
        previous_weights = self.ensemble_weights.copy()

        for iteration in range(100):  # 最大迭代次数
            weight_changed = False

            for name in self.base_predictors.keys():
                history = self.performance_history.get(name, [])
                if len(history) > 0:
                    # 计算最近表现的加权平均
                    recent_performance = np.mean(history[-10:])  # 最近10次表现

                    # 更新权重（表现好的预测器权重增加）
                    new_weight = max(0.1, recent_performance)
                    if abs(new_weight - self.ensemble_weights[name]) > 1e-6:
                        self.ensemble_weights[name] = new_weight
                        weight_changed = True

            # 计算权重变化的总和作为收敛指标
            weight_diff = sum(abs(self.ensemble_weights[name] - previous_weights.get(name, 0))
                            for name in self.ensemble_weights.keys())

            # 智能早停检查
            if use_intelligent_early_stopping:
                if early_stopping.update(weight_diff):
                    logger_manager.info(f"自适应权重更新智能早停，迭代次数: {iteration + 1}")
                    break
            else:
                # 简单早停逻辑
                if weight_diff < best_weight_diff - min_delta:
                    best_weight_diff = weight_diff
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= max_patience:
                    logger_manager.info(f"自适应权重更新早停（简单模式），迭代次数: {iteration + 1}")
                    break

            # 如果权重没有变化，也可以停止
            if not weight_changed:
                logger_manager.info(f"权重收敛，迭代次数: {iteration + 1}")
                break

            previous_weights = self.ensemble_weights.copy()
    
    def _voting_selection(self, candidates: List[int], target_count: int,
                         min_val: int, max_val: int) -> List[int]:
        """投票选择机制"""
        # 统计投票 (Counter 已在文件顶部导入)
        vote_counts = Counter(candidates)
        
        # 按票数排序
        sorted_candidates = sorted(vote_counts.items(), key=lambda x: x[1], reverse=True)
        
        # 选择票数最高的号码
        selected = []
        for num, count in sorted_candidates:
            if min_val <= num <= max_val and len(selected) < target_count:
                selected.append(num)
        
        # 如果数量不足，随机补充
        while len(selected) < target_count:
            candidate = np.random.randint(min_val, max_val + 1)
            if candidate not in selected:
                selected.append(candidate)
        
        return selected[:target_count]
    
    def evaluate_predictors(self, test_periods: int = 100):
        """评估各预测器的表现"""
        logger_manager.info(f"评估预测器表现，测试期数: {test_periods}")
        
        df = data_manager.get_data()
        if df is None or len(df) < test_periods:
            logger_manager.error("数据不足，无法进行评估")
            return {}
        
        evaluation_results = {}
        
        for name, predictor in self.base_predictors.items():
            logger_manager.info(f"评估预测器: {name}")
            
            correct_predictions = 0
            total_predictions = 0
            
            # 使用历史数据进行回测
            for i in range(test_periods):
                try:
                    # 获取测试期的实际结果
                    actual_row = df.iloc[i]
                    actual_front, actual_back = data_manager.parse_balls(actual_row)
                    
                    # 使用之前的数据进行预测
                    historical_data = df.iloc[i+1:i+501]  # 使用500期历史数据
                    
                    # 这里需要根据具体预测器的接口进行调整
                    # 简化处理：假设预测器可以基于历史数据预测
                    
                    total_predictions += 1
                    
                    # 简单的准确率计算（实际应该更复杂）
                    # 这里只是示例
                    
                except Exception as e:
                    logger_manager.warning(f"评估期 {i} 失败: {e}")
                    continue
            
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            evaluation_results[name] = {
                'accuracy': accuracy,
                'total_predictions': total_predictions,
                'correct_predictions': correct_predictions
            }
            
            # 更新历史表现
            self.performance_history[name].append(accuracy)

        return evaluation_results

    def calculate_diversity_metrics(self, count: int = 10) -> Dict:
        """计算预测器多样性指标

        评估集成中各预测器的多样性，多样性高的集成通常效果更好。
        包含三种主要指标：
        1. Q-statistic (Q统计量): 衡量成对预测器的一致性，值越低多样性越高
        2. Disagreement measure (不一致度): 预测器间不同预测的比例
        3. Correlation coefficient (相关系数): 预测器间的相关性

        Args:
            count: 用于评估的预测注数

        Returns:
            多样性指标字典
        """
        logger_manager.info(f"计算预测器多样性指标，评估注数: {count}")

        if len(self.base_predictors) < 2:
            logger_manager.warning("至少需要2个预测器才能计算多样性")
            return {'error': '预测器数量不足'}

        # 收集所有预测器的预测结果
        all_predictions = {}
        predictor_names = list(self.base_predictors.keys())

        for name, predictor in self.base_predictors.items():
            try:
                if hasattr(predictor, 'predict'):
                    pred = predictor.predict(count)
                elif hasattr(predictor, 'frequency_predict'):
                    pred = predictor.frequency_predict(count)
                else:
                    continue
                all_predictions[name] = pred
            except Exception as e:
                logger_manager.warning(f"预测器 {name} 预测失败: {e}")

        if len(all_predictions) < 2:
            logger_manager.warning("有效预测器数量不足")
            return {'error': '有效预测器数量不足'}

        # 计算多样性指标
        diversity_metrics = {
            'q_statistics': {},
            'disagreement_measures': {},
            'correlation_coefficients': {},
            'summary': {}
        }

        # 成对计算多样性指标
        names = list(all_predictions.keys())
        q_values = []
        disagreement_values = []
        correlation_values = []

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                name_i, name_j = names[i], names[j]
                pred_i = all_predictions[name_i]
                pred_j = all_predictions[name_j]

                # 计算Q统计量
                q_stat = self._calculate_q_statistic(pred_i, pred_j)
                pair_key = f"{name_i}_vs_{name_j}"
                diversity_metrics['q_statistics'][pair_key] = q_stat
                q_values.append(q_stat)

                # 计算不一致度
                disagreement = self._calculate_disagreement(pred_i, pred_j)
                diversity_metrics['disagreement_measures'][pair_key] = disagreement
                disagreement_values.append(disagreement)

                # 计算相关系数
                correlation = self._calculate_prediction_correlation(pred_i, pred_j)
                diversity_metrics['correlation_coefficients'][pair_key] = correlation
                correlation_values.append(correlation)

        # 汇总统计
        diversity_metrics['summary'] = {
            'avg_q_statistic': np.mean(q_values) if q_values else 0,
            'avg_disagreement': np.mean(disagreement_values) if disagreement_values else 0,
            'avg_correlation': np.mean(correlation_values) if correlation_values else 0,
            'diversity_score': self._calculate_overall_diversity_score(
                q_values, disagreement_values, correlation_values
            ),
            'num_predictors': len(all_predictions),
            'num_pairs': len(q_values),
            'recommendation': self._get_diversity_recommendation(
                np.mean(q_values) if q_values else 0,
                np.mean(disagreement_values) if disagreement_values else 0
            )
        }

        logger_manager.info(
            f"多样性评估完成: Q={diversity_metrics['summary']['avg_q_statistic']:.4f}, "
            f"不一致度={diversity_metrics['summary']['avg_disagreement']:.4f}, "
            f"综合得分={diversity_metrics['summary']['diversity_score']:.4f}"
        )

        return diversity_metrics

    def _calculate_q_statistic(self, pred_i: List[Tuple], pred_j: List[Tuple]) -> float:
        """计算Q统计量

        Q = (N11 * N00 - N01 * N10) / (N11 * N00 + N01 * N10)
        其中：
        - N11: 两个预测器都预测正确的次数
        - N00: 两个预测器都预测错误的次数
        - N01: 预测器i正确，j错误的次数
        - N10: 预测器i错误，j正确的次数

        Q值范围[-1, 1]，值越低表示多样性越高

        Args:
            pred_i: 预测器i的预测结果
            pred_j: 预测器j的预测结果

        Returns:
            Q统计量值
        """
        try:
            n11 = n00 = n01 = n10 = 0
            min_len = min(len(pred_i), len(pred_j))

            for k in range(min_len):
                front_i, back_i = pred_i[k] if k < len(pred_i) else ([], [])
                front_j, back_j = pred_j[k] if k < len(pred_j) else ([], [])

                # 将预测转换为集合进行比较
                set_i = set(front_i) | set([b + 100 for b in back_i])  # 后区加100避免冲突
                set_j = set(front_j) | set([b + 100 for b in back_j])

                # 计算重叠度作为"正确性"的代理
                overlap = len(set_i & set_j)
                total_unique = len(set_i | set_j)

                if total_unique == 0:
                    continue

                similarity = overlap / total_unique

                # 使用相似度阈值判断"一致性"
                i_agrees = similarity > 0.5
                j_agrees = similarity > 0.5

                if i_agrees and j_agrees:
                    n11 += 1
                elif not i_agrees and not j_agrees:
                    n00 += 1
                elif i_agrees and not j_agrees:
                    n01 += 1
                else:
                    n10 += 1

            # 计算Q统计量
            numerator = n11 * n00 - n01 * n10
            denominator = n11 * n00 + n01 * n10

            if denominator == 0:
                return 0.0

            return numerator / denominator

        except Exception as e:
            logger_manager.error(f"计算Q统计量失败: {e}")
            return 0.0

    def _calculate_disagreement(self, pred_i: List[Tuple], pred_j: List[Tuple]) -> float:
        """计算不一致度

        不一致度 = 预测不同的次数 / 总预测次数
        值越高表示多样性越高

        Args:
            pred_i: 预测器i的预测结果
            pred_j: 预测器j的预测结果

        Returns:
            不一致度值 (0-1)
        """
        try:
            disagreements = 0
            total = 0
            min_len = min(len(pred_i), len(pred_j))

            for k in range(min_len):
                front_i, back_i = pred_i[k] if k < len(pred_i) else ([], [])
                front_j, back_j = pred_j[k] if k < len(pred_j) else ([], [])

                # 前区不一致度
                front_overlap = len(set(front_i) & set(front_j))
                front_disagreement = (5 - front_overlap) / 5.0  # 前区5个号码

                # 后区不一致度
                back_overlap = len(set(back_i) & set(back_j))
                back_disagreement = (2 - back_overlap) / 2.0  # 后区2个号码

                # 综合不一致度（前区权重0.7，后区权重0.3）
                disagreements += front_disagreement * 0.7 + back_disagreement * 0.3
                total += 1

            return disagreements / total if total > 0 else 0.0

        except Exception as e:
            logger_manager.error(f"计算不一致度失败: {e}")
            return 0.0

    def _calculate_prediction_correlation(self, pred_i: List[Tuple], pred_j: List[Tuple]) -> float:
        """计算预测相关系数

        将预测结果转换为向量，计算Pearson相关系数
        值越低表示多样性越高

        Args:
            pred_i: 预测器i的预测结果
            pred_j: 预测器j的预测结果

        Returns:
            相关系数值 (-1到1)
        """
        try:
            # 将预测转换为向量（每个位置表示该号码是否被选中）
            vector_i = []
            vector_j = []
            min_len = min(len(pred_i), len(pred_j))

            for k in range(min_len):
                front_i, back_i = pred_i[k] if k < len(pred_i) else ([], [])
                front_j, back_j = pred_j[k] if k < len(pred_j) else ([], [])

                # 前区向量（35维）
                for num in range(1, 36):
                    vector_i.append(1 if num in front_i else 0)
                    vector_j.append(1 if num in front_j else 0)

                # 后区向量（12维）
                for num in range(1, 13):
                    vector_i.append(1 if num in back_i else 0)
                    vector_j.append(1 if num in back_j else 0)

            if len(vector_i) < 2:
                return 0.0

            # 计算Pearson相关系数
            vector_i = np.array(vector_i)
            vector_j = np.array(vector_j)

            # 避免常数向量
            if np.std(vector_i) == 0 or np.std(vector_j) == 0:
                return 0.0

            correlation = np.corrcoef(vector_i, vector_j)[0, 1]

            return correlation if not np.isnan(correlation) else 0.0

        except Exception as e:
            logger_manager.error(f"计算预测相关系数失败: {e}")
            return 0.0

    def _calculate_overall_diversity_score(self, q_values: List[float],
                                           disagreement_values: List[float],
                                           correlation_values: List[float]) -> float:
        """计算综合多样性得分

        综合考虑Q统计量、不一致度和相关系数，生成0-1的综合得分
        得分越高表示多样性越好

        Args:
            q_values: Q统计量列表
            disagreement_values: 不一致度列表
            correlation_values: 相关系数列表

        Returns:
            综合多样性得分 (0-1)
        """
        try:
            # Q统计量：越低越好，转换为正向指标
            avg_q = np.mean(q_values) if q_values else 0
            q_score = (1 - avg_q) / 2  # 映射到0-1，Q=-1时得分1，Q=1时得分0

            # 不一致度：越高越好，直接使用
            disagreement_score = np.mean(disagreement_values) if disagreement_values else 0

            # 相关系数：越低越好，转换为正向指标
            avg_corr = np.mean(correlation_values) if correlation_values else 0
            correlation_score = (1 - avg_corr) / 2  # 映射到0-1

            # 加权综合得分
            weights = {'q': 0.3, 'disagreement': 0.4, 'correlation': 0.3}
            overall_score = (
                weights['q'] * q_score +
                weights['disagreement'] * disagreement_score +
                weights['correlation'] * correlation_score
            )

            return max(0.0, min(1.0, overall_score))

        except Exception as e:
            logger_manager.error(f"计算综合多样性得分失败: {e}")
            return 0.5

    def _get_diversity_recommendation(self, avg_q: float, avg_disagreement: float) -> str:
        """获取多样性建议

        Args:
            avg_q: 平均Q统计量
            avg_disagreement: 平均不一致度

        Returns:
            多样性建议字符串
        """
        if avg_disagreement > 0.6:
            return "excellent - 预测器多样性非常高，集成效果预期良好"
        elif avg_disagreement > 0.4:
            return "good - 预测器多样性良好，建议保持当前配置"
        elif avg_disagreement > 0.2:
            return "moderate - 预测器多样性中等，可考虑添加不同类型的预测器"
        else:
            return "low - 预测器多样性较低，建议添加更多差异化的预测器以提升集成效果"


class MetaLearningPredictor:
    """元学习预测器"""
    
    def __init__(self):
        self.meta_model = None
        self.base_predictors = []
        
    def train_meta_model(self, base_predictions: List[Dict], actual_results: List[Tuple]):
        """训练元模型"""
        # 准备训练数据
        X = []  # 基础预测器的预测结果
        y = []  # 实际结果
        
        for i, actual in enumerate(actual_results):
            if i < len(base_predictions):
                # 将基础预测器的结果转换为特征向量
                feature_vector = self._predictions_to_features(base_predictions[i])
                X.append(feature_vector)
                y.append(self._result_to_target(actual))
        
        # 训练元模型
        if len(X) > 0:
            from sklearn.ensemble import RandomForestClassifier
            self.meta_model = RandomForestClassifier(n_estimators=100)
            self.meta_model.fit(X, y)
    
    def _predictions_to_features(self, predictions: Dict) -> List[float]:
        """将预测结果转换为特征向量"""
        features = []
        
        for predictor_name, pred_result in predictions.items():
            if isinstance(pred_result, list) and len(pred_result) > 0:
                front, back = pred_result[0]
                
                # 提取特征
                features.extend([
                    np.mean(front),  # 前区平均值
                    np.std(front),   # 前区标准差
                    sum(front),      # 前区和值
                    np.mean(back),   # 后区平均值
                    sum(back)        # 后区和值
                ])
        
        return features
    
    def _result_to_target(self, result: Tuple) -> int:
        """将实际结果转换为目标值"""
        front, back = result
        # 简化：使用和值作为目标
        return sum(front) + sum(back)


# 使用示例
"""
# 创建高级集成预测器
ensemble = AdvancedEnsemblePredictor()

# 注册基础预测器
ensemble.register_predictor('frequency', frequency_predictor, weight=0.3)
ensemble.register_predictor('markov', markov_predictor, weight=0.4)
ensemble.register_predictor('bayesian', bayesian_predictor, weight=0.3)

# 进行集成预测
predictions = ensemble.stacking_predict(5)
adaptive_predictions = ensemble.adaptive_ensemble_predict(5)

# 评估预测器表现
evaluation = ensemble.evaluate_predictors(100)
"""