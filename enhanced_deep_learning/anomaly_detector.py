#!/usr/bin/env python3
"""
异常检测器
检测彩票数据中的异常模式和数据质量问题
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from core_modules import logger_manager
from interfaces import AnalyzerInterface


class AnomalyDetector(AnalyzerInterface):
    """异常检测器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化异常检测器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        
        # 异常检测参数
        self.contamination = self.config.get('contamination', 0.1)  # 异常比例
        self.z_threshold = self.config.get('z_threshold', 3.0)  # Z分数阈值
        self.iqr_factor = self.config.get('iqr_factor', 1.5)  # IQR因子
        
        # 模型
        self.isolation_forest = IsolationForest(
            contamination=self.contamination,
            random_state=42
        )
        self.scaler = StandardScaler()
        
        logger_manager.info("异常检测器初始化完成")
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        分析数据异常
        
        Args:
            data: 彩票数据
            
        Returns:
            异常分析结果
        """
        try:
            logger_manager.info("开始异常检测分析")
            
            results = {
                'data_quality': self._check_data_quality(data),
                'statistical_anomalies': self._detect_statistical_anomalies(data),
                'pattern_anomalies': self._detect_pattern_anomalies(data),
                'isolation_forest_anomalies': self._detect_isolation_forest_anomalies(data),
                'clustering_anomalies': self._detect_clustering_anomalies(data),
                'summary': {}
            }
            
            # 汇总异常信息
            total_anomalies = 0
            for key, value in results.items():
                if key != 'summary' and isinstance(value, dict) and 'anomaly_count' in value:
                    total_anomalies += value['anomaly_count']
            
            results['summary'] = {
                'total_records': len(data),
                'total_anomalies': total_anomalies,
                'anomaly_rate': total_anomalies / len(data) if len(data) > 0 else 0,
                'data_quality_score': self._calculate_quality_score(results)
            }
            
            logger_manager.info(f"异常检测完成，发现 {total_anomalies} 个异常")
            
            return results
            
        except Exception as e:
            logger_manager.error(f"异常检测分析失败: {e}")
            return {}
    
    def _check_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """检查数据质量"""
        try:
            quality_issues = []
            
            # 检查缺失值
            missing_count = data.isnull().sum().sum()
            if missing_count > 0:
                quality_issues.append(f"发现 {missing_count} 个缺失值")
            
            # 检查重复记录
            duplicate_count = data.duplicated().sum()
            if duplicate_count > 0:
                quality_issues.append(f"发现 {duplicate_count} 个重复记录")
            
            # 检查期号连续性
            if 'issue' in data.columns:
                issues = data['issue'].astype(int).sort_values()
                gaps = []
                for i in range(1, len(issues)):
                    if issues.iloc[i] - issues.iloc[i-1] > 1:
                        gaps.append((issues.iloc[i-1], issues.iloc[i]))
                
                if gaps:
                    quality_issues.append(f"发现 {len(gaps)} 个期号间隔")
            
            # 检查号码范围
            range_issues = self._check_number_ranges(data)
            quality_issues.extend(range_issues)
            
            return {
                'quality_issues': quality_issues,
                'anomaly_count': len(quality_issues),
                'missing_values': missing_count,
                'duplicate_records': duplicate_count
            }
            
        except Exception as e:
            logger_manager.error(f"数据质量检查失败: {e}")
            return {}
    
    def _check_number_ranges(self, data: pd.DataFrame) -> List[str]:
        """检查号码范围"""
        issues = []
        
        try:
            for _, row in data.iterrows():
                # 检查前区号码
                if 'front_balls' in row:
                    front_balls = [int(x) for x in str(row['front_balls']).split(',')]
                    
                    # 检查范围
                    if any(ball < 1 or ball > 35 for ball in front_balls):
                        issues.append(f"期号 {row.get('issue', 'unknown')} 前区号码超出范围")
                    
                    # 检查数量
                    if len(front_balls) != 5:
                        issues.append(f"期号 {row.get('issue', 'unknown')} 前区号码数量异常: {len(front_balls)}")
                    
                    # 检查重复
                    if len(set(front_balls)) != len(front_balls):
                        issues.append(f"期号 {row.get('issue', 'unknown')} 前区号码有重复")
                
                # 检查后区号码
                if 'back_balls' in row:
                    back_balls = [int(x) for x in str(row['back_balls']).split(',')]
                    
                    # 检查范围
                    if any(ball < 1 or ball > 12 for ball in back_balls):
                        issues.append(f"期号 {row.get('issue', 'unknown')} 后区号码超出范围")
                    
                    # 检查数量
                    if len(back_balls) != 2:
                        issues.append(f"期号 {row.get('issue', 'unknown')} 后区号码数量异常: {len(back_balls)}")
                    
                    # 检查重复
                    if len(set(back_balls)) != len(back_balls):
                        issues.append(f"期号 {row.get('issue', 'unknown')} 后区号码有重复")
        
        except Exception as e:
            logger_manager.error(f"号码范围检查失败: {e}")
        
        return issues
    
    def _detect_statistical_anomalies(self, data: pd.DataFrame) -> Dict[str, Any]:
        """检测统计异常"""
        try:
            anomalies = []
            
            # 提取数值特征
            features = self._extract_numerical_features(data)
            
            if features.empty:
                return {'anomalies': [], 'anomaly_count': 0}
            
            # Z分数异常检测
            z_scores = np.abs(stats.zscore(features))
            z_anomalies = np.where(z_scores > self.z_threshold)
            
            for i, j in zip(z_anomalies[0], z_anomalies[1]):
                anomalies.append({
                    'type': 'z_score',
                    'row': i,
                    'feature': features.columns[j],
                    'value': features.iloc[i, j],
                    'z_score': z_scores[i, j]
                })
            
            # IQR异常检测
            for col in features.columns:
                Q1 = features[col].quantile(0.25)
                Q3 = features[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - self.iqr_factor * IQR
                upper_bound = Q3 + self.iqr_factor * IQR
                
                outliers = features[(features[col] < lower_bound) | (features[col] > upper_bound)]
                
                for idx in outliers.index:
                    anomalies.append({
                        'type': 'iqr',
                        'row': idx,
                        'feature': col,
                        'value': features.loc[idx, col],
                        'bounds': (lower_bound, upper_bound)
                    })
            
            return {
                'anomalies': anomalies,
                'anomaly_count': len(anomalies)
            }
            
        except Exception as e:
            logger_manager.error(f"统计异常检测失败: {e}")
            return {}
    
    def _detect_pattern_anomalies(self, data: pd.DataFrame) -> Dict[str, Any]:
        """检测模式异常"""
        try:
            anomalies = []
            
            # 检测连续号码异常
            consecutive_anomalies = self._detect_consecutive_anomalies(data)
            anomalies.extend(consecutive_anomalies)
            
            # 检测重复模式异常
            repeat_anomalies = self._detect_repeat_pattern_anomalies(data)
            anomalies.extend(repeat_anomalies)
            
            # 检测和值异常
            sum_anomalies = self._detect_sum_anomalies(data)
            anomalies.extend(sum_anomalies)
            
            return {
                'anomalies': anomalies,
                'anomaly_count': len(anomalies)
            }
            
        except Exception as e:
            logger_manager.error(f"模式异常检测失败: {e}")
            return {}
    
    def _detect_consecutive_anomalies(self, data: pd.DataFrame) -> List[Dict]:
        """检测连续号码异常"""
        anomalies = []
        
        try:
            for idx, row in data.iterrows():
                if 'front_balls' in row:
                    front_balls = sorted([int(x) for x in str(row['front_balls']).split(',')])
                    
                    # 检查是否有过多连续号码
                    consecutive_count = 1
                    max_consecutive = 1
                    
                    for i in range(1, len(front_balls)):
                        if front_balls[i] == front_balls[i-1] + 1:
                            consecutive_count += 1
                            max_consecutive = max(max_consecutive, consecutive_count)
                        else:
                            consecutive_count = 1
                    
                    # 如果连续号码超过3个，标记为异常
                    if max_consecutive > 3:
                        anomalies.append({
                            'type': 'consecutive_numbers',
                            'row': idx,
                            'issue': row.get('issue', 'unknown'),
                            'consecutive_count': max_consecutive,
                            'numbers': front_balls
                        })
        
        except Exception as e:
            logger_manager.error(f"连续号码异常检测失败: {e}")
        
        return anomalies
    
    def _detect_repeat_pattern_anomalies(self, data: pd.DataFrame) -> List[Dict]:
        """检测重复模式异常"""
        anomalies = []
        
        try:
            # 检查完全相同的号码组合
            seen_combinations = {}
            
            for idx, row in data.iterrows():
                if 'front_balls' in row and 'back_balls' in row:
                    front_balls = tuple(sorted([int(x) for x in str(row['front_balls']).split(',')]))
                    back_balls = tuple(sorted([int(x) for x in str(row['back_balls']).split(',')]))
                    
                    combination = (front_balls, back_balls)
                    
                    if combination in seen_combinations:
                        anomalies.append({
                            'type': 'duplicate_combination',
                            'row': idx,
                            'issue': row.get('issue', 'unknown'),
                            'duplicate_of': seen_combinations[combination],
                            'combination': combination
                        })
                    else:
                        seen_combinations[combination] = row.get('issue', idx)
        
        except Exception as e:
            logger_manager.error(f"重复模式异常检测失败: {e}")
        
        return anomalies
    
    def _detect_sum_anomalies(self, data: pd.DataFrame) -> List[Dict]:
        """检测和值异常"""
        anomalies = []
        
        try:
            sums = []
            
            for idx, row in data.iterrows():
                if 'front_balls' in row:
                    front_balls = [int(x) for x in str(row['front_balls']).split(',')]
                    ball_sum = sum(front_balls)
                    sums.append((idx, ball_sum, row.get('issue', 'unknown')))
            
            if sums:
                sum_values = [s[1] for s in sums]
                mean_sum = np.mean(sum_values)
                std_sum = np.std(sum_values)
                
                # 检测和值异常（超过3个标准差）
                for idx, ball_sum, issue in sums:
                    z_score = abs(ball_sum - mean_sum) / std_sum if std_sum > 0 else 0
                    
                    if z_score > 3:
                        anomalies.append({
                            'type': 'sum_anomaly',
                            'row': idx,
                            'issue': issue,
                            'sum': ball_sum,
                            'z_score': z_score,
                            'mean_sum': mean_sum
                        })
        
        except Exception as e:
            logger_manager.error(f"和值异常检测失败: {e}")
        
        return anomalies
    
    def _detect_isolation_forest_anomalies(self, data: pd.DataFrame) -> Dict[str, Any]:
        """使用孤立森林检测异常"""
        try:
            features = self._extract_numerical_features(data)
            
            if features.empty:
                return {'anomalies': [], 'anomaly_count': 0}
            
            # 标准化特征
            features_scaled = self.scaler.fit_transform(features)
            
            # 训练孤立森林
            anomaly_labels = self.isolation_forest.fit_predict(features_scaled)
            anomaly_scores = self.isolation_forest.decision_function(features_scaled)
            
            # 提取异常点
            anomalies = []
            for i, (label, score) in enumerate(zip(anomaly_labels, anomaly_scores)):
                if label == -1:  # 异常点
                    anomalies.append({
                        'type': 'isolation_forest',
                        'row': i,
                        'issue': data.iloc[i].get('issue', 'unknown'),
                        'anomaly_score': score
                    })
            
            return {
                'anomalies': anomalies,
                'anomaly_count': len(anomalies)
            }
            
        except Exception as e:
            logger_manager.error(f"孤立森林异常检测失败: {e}")
            return {}
    
    def _detect_clustering_anomalies(self, data: pd.DataFrame) -> Dict[str, Any]:
        """使用聚类检测异常"""
        try:
            features = self._extract_numerical_features(data)
            
            if features.empty:
                return {'anomalies': [], 'anomaly_count': 0}
            
            # 标准化特征
            features_scaled = self.scaler.fit_transform(features)
            
            # DBSCAN聚类
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            cluster_labels = dbscan.fit_predict(features_scaled)
            
            # 噪声点（标签为-1）被认为是异常
            anomalies = []
            for i, label in enumerate(cluster_labels):
                if label == -1:  # 噪声点
                    anomalies.append({
                        'type': 'clustering_outlier',
                        'row': i,
                        'issue': data.iloc[i].get('issue', 'unknown')
                    })
            
            return {
                'anomalies': anomalies,
                'anomaly_count': len(anomalies),
                'n_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            }
            
        except Exception as e:
            logger_manager.error(f"聚类异常检测失败: {e}")
            return {}
    
    def _extract_numerical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """提取数值特征"""
        try:
            features = []
            
            for _, row in data.iterrows():
                feature_row = []
                
                if 'front_balls' in row and 'back_balls' in row:
                    front_balls = [int(x) for x in str(row['front_balls']).split(',')]
                    back_balls = [int(x) for x in str(row['back_balls']).split(',')]
                    
                    # 基本特征
                    feature_row.extend(front_balls)  # 前区5个号码
                    feature_row.extend(back_balls)   # 后区2个号码
                    
                    # 统计特征
                    feature_row.append(sum(front_balls))  # 前区和值
                    feature_row.append(sum(back_balls))   # 后区和值
                    feature_row.append(max(front_balls) - min(front_balls))  # 前区跨度
                    feature_row.append(max(back_balls) - min(back_balls))    # 后区跨度
                    
                    features.append(feature_row)
            
            if features:
                columns = [f'front_{i+1}' for i in range(5)] + [f'back_{i+1}' for i in range(2)] + \
                         ['front_sum', 'back_sum', 'front_span', 'back_span']
                
                return pd.DataFrame(features, columns=columns)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger_manager.error(f"特征提取失败: {e}")
            return pd.DataFrame()
    
    def _calculate_quality_score(self, results: Dict[str, Any]) -> float:
        """计算数据质量分数"""
        try:
            total_records = results.get('summary', {}).get('total_records', 1)
            total_anomalies = results.get('summary', {}).get('total_anomalies', 0)
            
            # 基础质量分数（100分制）
            base_score = 100.0
            
            # 根据异常率扣分
            anomaly_rate = total_anomalies / total_records if total_records > 0 else 0
            anomaly_penalty = min(anomaly_rate * 100, 50)  # 最多扣50分
            
            # 数据质量问题扣分
            quality_issues = results.get('data_quality', {}).get('anomaly_count', 0)
            quality_penalty = min(quality_issues * 5, 30)  # 每个问题扣5分，最多扣30分
            
            final_score = max(0, base_score - anomaly_penalty - quality_penalty)
            
            return round(final_score, 2)
            
        except Exception as e:
            logger_manager.error(f"质量分数计算失败: {e}")
            return 0.0
