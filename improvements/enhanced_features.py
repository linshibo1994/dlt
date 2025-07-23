#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
增强特性模块
提供高级预测和分析功能
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Callable, Union
from datetime import datetime
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns

# 尝试导入核心模块
try:
    from core_modules import logger_manager, data_manager, cache_manager
except ImportError:
    # 如果在不同目录运行，添加父目录到路径
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core_modules import logger_manager, data_manager, cache_manager

# 尝试导入scikit-learn
try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.mixture import GaussianMixture
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger_manager.warning("scikit-learn未安装，部分功能将不可用")


class EnhancedFeatureAnalyzer:
    """增强特性分析器"""
    
    def __init__(self):
        """初始化增强特性分析器"""
        self.df = data_manager.get_data()
        if self.df is None:
            logger_manager.error("数据未加载")
    
    def analyze_number_patterns(self, periods: int = 500) -> Dict:
        """分析号码模式
        
        Args:
            periods: 分析期数
            
        Returns:
            Dict: 分析结果
        """
        if self.df is None or len(self.df) < periods:
            logger_manager.error(f"数据不足，无法分析号码模式")
            return {}
        
        cache_key = f"number_patterns_{periods}"
        cached_result = cache_manager.load_cache("analysis", cache_key)
        if cached_result:
            return cached_result
        
        df_subset = self.df.head(periods)
        
        # 初始化结果
        result = {
            'consecutive_patterns': {},
            'sum_patterns': {},
            'odd_even_patterns': {},
            'big_small_patterns': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # 分析连号模式
        consecutive_counts = []
        for _, row in df_subset.iterrows():
            front_balls, _ = data_manager.parse_balls(row)
            front_balls = sorted(front_balls)
            consecutive_count = 0
            for i in range(len(front_balls) - 1):
                if front_balls[i + 1] - front_balls[i] == 1:
                    consecutive_count += 1
            consecutive_counts.append(consecutive_count)
        
        result['consecutive_patterns'] = {
            'counts': dict(Counter(consecutive_counts)),
            'avg': np.mean(consecutive_counts),
            'max': max(consecutive_counts),
            'min': min(consecutive_counts)
        }
        
        # 分析和值模式
        front_sums = []
        back_sums = []
        total_sums = []
        for _, row in df_subset.iterrows():
            front_balls, back_balls = data_manager.parse_balls(row)
            front_sum = sum(front_balls)
            back_sum = sum(back_balls)
            front_sums.append(front_sum)
            back_sums.append(back_sum)
            total_sums.append(front_sum + back_sum)
        
        result['sum_patterns'] = {
            'front_sum': {
                'avg': np.mean(front_sums),
                'std': np.std(front_sums),
                'max': max(front_sums),
                'min': min(front_sums),
                'distribution': dict(Counter(front_sums))
            },
            'back_sum': {
                'avg': np.mean(back_sums),
                'std': np.std(back_sums),
                'max': max(back_sums),
                'min': min(back_sums),
                'distribution': dict(Counter(back_sums))
            },
            'total_sum': {
                'avg': np.mean(total_sums),
                'std': np.std(total_sums),
                'max': max(total_sums),
                'min': min(total_sums),
                'distribution': dict(Counter(total_sums))
            }
        }
        
        # 缓存结果
        cache_manager.save_cache("analysis", cache_key, result)
        
        return result

    def cluster_analysis(self, periods: int = 500, n_clusters: int = 5) -> Dict:
        """聚类分析
        
        Args:
            periods: 分析期数
            n_clusters: 聚类数量
            
        Returns:
            Dict: 分析结果
        """
        if not SKLEARN_AVAILABLE:
            logger_manager.error("scikit-learn未安装，无法进行聚类分析")
            return {}
        
        if self.df is None or len(self.df) < periods:
            logger_manager.error(f"数据不足，无法进行聚类分析")
            return {}
        
        cache_key = f"cluster_analysis_{periods}_{n_clusters}"
        cached_result = cache_manager.load_cache("analysis", cache_key)
        if cached_result:
            return cached_result
        
        df_subset = self.df.head(periods)
        
        # 提取特征
        features = []
        for _, row in df_subset.iterrows():
            front_balls, back_balls = data_manager.parse_balls(row)
            
            # 基础特征
            front_sum = sum(front_balls)
            back_sum = sum(back_balls)
            front_mean = np.mean(front_balls)
            back_mean = np.mean(back_balls)
            front_std = np.std(front_balls)
            back_std = np.std(back_balls)
            
            # 模式特征
            front_balls = sorted(front_balls)
            consecutive_count = sum(1 for i in range(len(front_balls)-1) if front_balls[i+1] - front_balls[i] == 1)
            front_odd = sum(1 for ball in front_balls if ball % 2 == 1)
            back_odd = sum(1 for ball in back_balls if ball % 2 == 1)
            front_big = sum(1 for ball in front_balls if ball >= 18)
            back_big = sum(1 for ball in back_balls if ball >= 7)
            
            # 组合特征
            feature_vector = [
                front_sum, back_sum, front_mean, back_mean, front_std, back_std,
                consecutive_count, front_odd, back_odd, front_big, back_big
            ]
            features.append(feature_vector)
        
        # 标准化特征
        features = np.array(features)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # 降维（可选）
        pca = PCA(n_components=min(5, features.shape[1]))
        features_pca = pca.fit_transform(features_scaled)
        
        # K-Means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features_scaled)
        
        # 高斯混合模型
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        gmm_clusters = gmm.fit_predict(features_scaled)
        
        # DBSCAN聚类
        dbscan = DBSCAN(eps=1.0, min_samples=5)
        dbscan_clusters = dbscan.fit_predict(features_scaled)
        
        # 分析聚类结果
        cluster_stats = {}
        for i in range(n_clusters):
            cluster_indices = np.where(clusters == i)[0]
            if len(cluster_indices) > 0:
                cluster_features = features[cluster_indices]
                cluster_stats[i] = {
                    'count': len(cluster_indices),
                    'front_sum_avg': np.mean(cluster_features[:, 0]),
                    'back_sum_avg': np.mean(cluster_features[:, 1]),
                    'consecutive_avg': np.mean(cluster_features[:, 6]),
                    'front_odd_avg': np.mean(cluster_features[:, 7]),
                    'back_odd_avg': np.mean(cluster_features[:, 8]),
                    'front_big_avg': np.mean(cluster_features[:, 9]),
                    'back_big_avg': np.mean(cluster_features[:, 10])
                }
        
        # 准备结果
        result = {
            'kmeans': {
                'clusters': clusters.tolist(),
                'centers': kmeans.cluster_centers_.tolist(),
                'inertia': kmeans.inertia_,
                'stats': cluster_stats
            },
            'gmm': {
                'clusters': gmm_clusters.tolist(),
                'means': gmm.means_.tolist(),
                'weights': gmm.weights_.tolist()
            },
            'dbscan': {
                'clusters': dbscan_clusters.tolist(),
                'n_clusters': len(set(dbscan_clusters)) - (1 if -1 in dbscan_clusters else 0),
                'noise': np.sum(dbscan_clusters == -1)
            },
            'pca': {
                'components': pca.components_.tolist(),
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'features_pca': features_pca.tolist()
            },
            'feature_names': [
                'front_sum', 'back_sum', 'front_mean', 'back_mean', 'front_std', 'back_std',
                'consecutive_count', 'front_odd', 'back_odd', 'front_big', 'back_big'
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        # 缓存结果
        cache_manager.save_cache("analysis", cache_key, result)
        
        return result
    
    def predict_with_patterns(self, count: int = 5) -> List[Tuple[List[int], List[int]]]:
        """基于模式分析进行预测
        
        Args:
            count: 预测注数
            
        Returns:
            List[Tuple[List[int], List[int]]]: 预测结果列表
        """
        # 分析号码模式
        patterns = self.analyze_number_patterns(500)
        
        # 获取和值分布
        front_sum_dist = patterns.get('sum_patterns', {}).get('front_sum', {}).get('distribution', {})
        back_sum_dist = patterns.get('sum_patterns', {}).get('back_sum', {}).get('distribution', {})
        
        # 转换为概率分布
        front_sum_probs = {}
        total_front = sum(front_sum_dist.values())
        for sum_val, count in front_sum_dist.items():
            front_sum_probs[sum_val] = count / total_front
        
        back_sum_probs = {}
        total_back = sum(back_sum_dist.values())
        for sum_val, count in back_sum_dist.items():
            back_sum_probs[sum_val] = count / total_back
        
        # 获取连号分布
        consecutive_dist = patterns.get('consecutive_patterns', {}).get('counts', {})
        consecutive_probs = {}
        total_consecutive = sum(consecutive_dist.values())
        for cons_val, count in consecutive_dist.items():
            consecutive_probs[cons_val] = count / total_consecutive
        
        # 生成预测
        predictions = []
        for _ in range(count):
            # 选择目标和值
            front_sum_target = self._weighted_choice(front_sum_probs)
            back_sum_target = self._weighted_choice(back_sum_probs)
            consecutive_target = self._weighted_choice(consecutive_probs)
            
            # 生成符合目标和值和连号数的号码组合
            front_balls = self._generate_with_constraints(
                5, 1, 35, front_sum_target, consecutive_target
            )
            back_balls = self._generate_with_constraints(
                2, 1, 12, back_sum_target, 0
            )
            
            predictions.append((sorted(front_balls), sorted(back_balls)))
        
        return predictions
    
    def _weighted_choice(self, probs: Dict[int, float]) -> int:
        """加权随机选择
        
        Args:
            probs: 概率字典，键为值，值为概率
            
        Returns:
            int: 选择的值
        """
        items = list(probs.items())
        values = [item[0] for item in items]
        weights = [item[1] for item in items]
        return np.random.choice(values, p=weights)
    
    def _generate_with_constraints(self, count: int, min_val: int, max_val: int, 
                                  sum_target: int, consecutive_target: int) -> List[int]:
        """生成符合约束的号码组合
        
        Args:
            count: 号码数量
            min_val: 最小值
            max_val: 最大值
            sum_target: 目标和值
            consecutive_target: 目标连号数
            
        Returns:
            List[int]: 生成的号码列表
        """
        # 最大尝试次数
        max_attempts = 1000
        
        for _ in range(max_attempts):
            # 随机生成号码
            balls = sorted(np.random.choice(range(min_val, max_val + 1), count, replace=False))
            
            # 计算和值
            balls_sum = sum(balls)
            
            # 计算连号数
            consecutive_count = 0
            for i in range(len(balls) - 1):
                if balls[i + 1] - balls[i] == 1:
                    consecutive_count += 1
            
            # 检查约束
            sum_diff = abs(balls_sum - sum_target)
            consecutive_diff = abs(consecutive_count - consecutive_target)
            
            # 如果和值和连号数都接近目标，返回结果
            if sum_diff <= 5 and consecutive_diff <= 1:
                return balls
        
        # 如果达到最大尝试次数仍未找到符合条件的组合，使用基于历史数据的智能选择
        # 获取历史数据进行分析
        from core_modules import data_manager
        historical_data = data_manager.get_data()

        if historical_data is not None and len(historical_data) > 0:
            # 基于历史数据的频率分析选择
            from collections import Counter
            counter = Counter()

            for _, row in historical_data.head(100).iterrows():  # 使用最近100期数据
                if min_val == 1 and max_val == 35:  # 前区
                    balls_str = str(row.get('front_balls', ''))
                else:  # 后区
                    balls_str = str(row.get('back_balls', ''))

                balls = [int(x) for x in balls_str.split(',') if x.strip().isdigit()]
                for ball in balls:
                    if min_val <= ball <= max_val:
                        counter[ball] += 1

            # 选择频率最高的号码
            if counter:
                most_common = [ball for ball, freq in counter.most_common()]
                selected = []
                for ball in most_common:
                    if len(selected) >= count:
                        break
                    if ball not in selected:
                        selected.append(ball)

                # 如果不够，补充剩余号码
                while len(selected) < count:
                    for i in range(min_val, max_val + 1):
                        if i not in selected:
                            selected.append(i)
                            if len(selected) >= count:
                                break

                return sorted(selected[:count])

        # 如果没有历史数据，使用均匀分布选择
        selected = list(range(min_val, min(min_val + count, max_val + 1)))
        while len(selected) < count and len(selected) < (max_val - min_val + 1):
            for i in range(min_val, max_val + 1):
                if i not in selected:
                    selected.append(i)
                    if len(selected) >= count:
                        break

        return sorted(selected[:count])


class EnhancedFeaturePredictor:
    """增强特性预测器"""
    
    def __init__(self):
        """初始化增强特性预测器"""
        self.analyzer = EnhancedFeatureAnalyzer()
        self.df = data_manager.get_data()
        if self.df is None:
            logger_manager.error("数据未加载")
    
    def pattern_based_predict(self, count: int = 5) -> List[Tuple[List[int], List[int]]]:
        """基于模式的预测
        
        Args:
            count: 预测注数
            
        Returns:
            List[Tuple[List[int], List[int]]]: 预测结果列表
        """
        return self.analyzer.predict_with_patterns(count)
    
    def cluster_based_predict(self, count: int = 5) -> List[Tuple[List[int], List[int]]]:
        """基于聚类的预测
        
        Args:
            count: 预测注数
            
        Returns:
            List[Tuple[List[int], List[int]]]: 预测结果列表
        """
        if not SKLEARN_AVAILABLE:
            logger_manager.error("scikit-learn未安装，无法进行基于聚类的预测")
            return []
        
        # 进行聚类分析
        cluster_result = self.analyzer.cluster_analysis(500, 5)
        
        # 获取最大的聚类
        kmeans_stats = cluster_result.get('kmeans', {}).get('stats', {})
        largest_cluster = max(kmeans_stats.items(), key=lambda x: x[1]['count'])[0]
        
        # 获取该聚类的特征均值
        cluster_features = kmeans_stats[largest_cluster]
        
        # 生成预测
        predictions = []
        for _ in range(count):
            # 基于聚类特征生成号码
            front_balls = self._generate_with_features(
                5, 1, 35,
                cluster_features['front_sum_avg'],
                cluster_features['consecutive_avg'],
                cluster_features['front_odd_avg'],
                cluster_features['front_big_avg']
            )
            
            back_balls = self._generate_with_features(
                2, 1, 12,
                cluster_features['back_sum_avg'],
                0,
                cluster_features['back_odd_avg'],
                cluster_features['back_big_avg']
            )
            
            predictions.append((sorted(front_balls), sorted(back_balls)))
        
        return predictions
    
    def _generate_with_features(self, count: int, min_val: int, max_val: int,
                               sum_avg: float, consecutive_avg: float,
                               odd_avg: float, big_avg: float) -> List[int]:
        """基于特征生成号码
        
        Args:
            count: 号码数量
            min_val: 最小值
            max_val: 最大值
            sum_avg: 和值均值
            consecutive_avg: 连号均值
            odd_avg: 奇数均值
            big_avg: 大号均值
            
        Returns:
            List[int]: 生成的号码列表
        """
        # 最大尝试次数
        max_attempts = 1000
        
        # 确定目标值
        sum_target = int(round(sum_avg))
        consecutive_target = int(round(consecutive_avg))
        odd_target = int(round(odd_avg))
        big_target = int(round(big_avg))
        
        for _ in range(max_attempts):
            # 随机生成号码
            balls = sorted(np.random.choice(range(min_val, max_val + 1), count, replace=False))
            
            # 计算特征
            balls_sum = sum(balls)
            
            consecutive_count = 0
            for i in range(len(balls) - 1):
                if balls[i + 1] - balls[i] == 1:
                    consecutive_count += 1
            
            odd_count = sum(1 for ball in balls if ball % 2 == 1)
            
            big_threshold = 18 if max_val > 20 else max_val // 2
            big_count = sum(1 for ball in balls if ball >= big_threshold)
            
            # 计算特征差异
            sum_diff = abs(balls_sum - sum_target)
            consecutive_diff = abs(consecutive_count - consecutive_target)
            odd_diff = abs(odd_count - odd_target)
            big_diff = abs(big_count - big_target)
            
            # 计算总差异
            total_diff = sum_diff / count + consecutive_diff + odd_diff + big_diff
            
            # 如果总差异较小，返回结果
            if total_diff <= 3:
                return balls
        
        # 如果达到最大尝试次数仍未找到符合条件的组合，放宽条件
        return sorted(np.random.choice(range(min_val, max_val + 1), count, replace=False))


# 全局实例
_enhanced_feature_analyzer = None
_enhanced_feature_predictor = None

def get_enhanced_feature_analyzer() -> EnhancedFeatureAnalyzer:
    """获取增强特性分析器实例"""
    global _enhanced_feature_analyzer
    if _enhanced_feature_analyzer is None:
        _enhanced_feature_analyzer = EnhancedFeatureAnalyzer()
    return _enhanced_feature_analyzer

def get_enhanced_feature_predictor() -> EnhancedFeaturePredictor:
    """获取增强特性预测器实例"""
    global _enhanced_feature_predictor
    if _enhanced_feature_predictor is None:
        _enhanced_feature_predictor = EnhancedFeaturePredictor()
    return _enhanced_feature_predictor


if __name__ == "__main__":
    # 测试增强特性模块
    print("🔍 测试增强特性模块...")
    
    # 测试号码模式分析
    analyzer = get_enhanced_feature_analyzer()
    print("\n📊 分析号码模式...")
    patterns = analyzer.analyze_number_patterns(300)
    
    print(f"连号模式: 平均连号数 {patterns['consecutive_patterns']['avg']:.2f}")
    print(f"和值模式: 前区平均和值 {patterns['sum_patterns']['front_sum']['avg']:.2f}, 后区平均和值 {patterns['sum_patterns']['back_sum']['avg']:.2f}")
    
    # 测试聚类分析
    if SKLEARN_AVAILABLE:
        print("\n📊 聚类分析...")
        clusters = analyzer.cluster_analysis(300, 5)
        print(f"K-Means聚类: {len(clusters['kmeans']['stats'])} 个聚类")
        print(f"GMM聚类: {len(clusters['gmm']['weights'])} 个聚类")
        print(f"DBSCAN聚类: {clusters['dbscan']['n_clusters']} 个聚类, {clusters['dbscan']['noise']} 个噪声点")
    
    # 测试基于模式的预测
    predictor = get_enhanced_feature_predictor()
    print("\n🎯 基于模式的预测...")
    pattern_predictions = predictor.pattern_based_predict(3)
    for i, (front, back) in enumerate(pattern_predictions):
        print(f"  第 {i+1} 注: {front} + {back}")
    
    # 测试基于聚类的预测
    if SKLEARN_AVAILABLE:
        print("\n🎯 基于聚类的预测...")
        cluster_predictions = predictor.cluster_based_predict(3)
        for i, (front, back) in enumerate(cluster_predictions):
            print(f"  第 {i+1} 注: {front} + {back}")
    
    print("\n✅ 增强特性模块测试完成")