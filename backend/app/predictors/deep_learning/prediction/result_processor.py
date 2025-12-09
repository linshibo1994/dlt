#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
结果处理器模块
Result Processor Module

提供预测结果的格式化、分析、转换和导出功能。
"""

import os
import json
import csv
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import numpy as np
import pandas as pd

from core_modules import logger_manager
from ..utils.exceptions import PredictionException
from .prediction_engine import PredictionResult, PredictionStatus


class ResultFormat(Enum):
    """结果格式枚举"""
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"
    XML = "xml"
    HTML = "html"
    NUMPY = "numpy"
    PANDAS = "pandas"


class AnalysisType(Enum):
    """分析类型枚举"""
    STATISTICAL = "statistical"
    DISTRIBUTION = "distribution"
    CORRELATION = "correlation"
    TREND = "trend"
    OUTLIER = "outlier"


@dataclass
class ProcessedResult:
    """处理后的结果"""
    original_result: PredictionResult
    formatted_data: Any
    format_type: ResultFormat
    analysis: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    processed_time: datetime = field(default_factory=datetime.now)


class ResultFormatter:
    """结果格式化器"""
    
    def __init__(self):
        """初始化结果格式化器"""
        logger_manager.info("结果格式化器初始化完成")
    
    def format_to_json(self, result: PredictionResult) -> Dict[str, Any]:
        """格式化为JSON"""
        try:
            formatted = {
                'request_id': result.request_id,
                'status': result.status.value,
                'model_info': result.model_info,
                'execution_time': result.execution_time,
                'completed_time': result.completed_time.isoformat() if result.completed_time else None,
                'predictions': result.predictions.tolist() if result.predictions is not None else None,
                'confidence_scores': result.confidence_scores.tolist() if result.confidence_scores is not None else None,
                'error_message': result.error_message,
                'metadata': result.metadata
            }
            
            logger_manager.debug(f"结果格式化为JSON: {result.request_id}")
            return formatted
            
        except Exception as e:
            logger_manager.error(f"JSON格式化失败: {e}")
            raise PredictionException(f"JSON格式化失败: {e}")
    
    def format_to_dataframe(self, result: PredictionResult) -> pd.DataFrame:
        """格式化为DataFrame"""
        try:
            if result.predictions is None:
                raise PredictionException("预测结果为空，无法转换为DataFrame")
            
            # 创建基础数据
            data = {
                'prediction': result.predictions.flatten() if result.predictions.ndim > 1 else result.predictions
            }
            
            # 添加置信度分数
            if result.confidence_scores is not None:
                if result.confidence_scores.ndim > 1:
                    for i in range(result.confidence_scores.shape[1]):
                        data[f'confidence_{i}'] = result.confidence_scores[:, i]
                else:
                    data['confidence'] = result.confidence_scores
            
            # 添加元数据
            data['request_id'] = result.request_id
            data['status'] = result.status.value
            data['execution_time'] = result.execution_time
            
            df = pd.DataFrame(data)
            
            logger_manager.debug(f"结果格式化为DataFrame: {result.request_id}, 形状: {df.shape}")
            return df
            
        except Exception as e:
            logger_manager.error(f"DataFrame格式化失败: {e}")
            raise PredictionException(f"DataFrame格式化失败: {e}")
    
    def format_to_csv(self, result: PredictionResult) -> str:
        """格式化为CSV字符串"""
        try:
            df = self.format_to_dataframe(result)
            csv_string = df.to_csv(index=False)
            
            logger_manager.debug(f"结果格式化为CSV: {result.request_id}")
            return csv_string
            
        except Exception as e:
            logger_manager.error(f"CSV格式化失败: {e}")
            raise PredictionException(f"CSV格式化失败: {e}")
    
    def format_to_html(self, result: PredictionResult) -> str:
        """格式化为HTML"""
        try:
            html = f"""
            <div class="prediction-result">
                <h3>预测结果 - {result.request_id}</h3>
                <div class="result-info">
                    <p><strong>状态:</strong> {result.status.value}</p>
                    <p><strong>执行时间:</strong> {result.execution_time:.4f}秒</p>
                    <p><strong>完成时间:</strong> {result.completed_time}</p>
                </div>
            """
            
            if result.predictions is not None:
                df = self.format_to_dataframe(result)
                html += f"""
                <div class="predictions">
                    <h4>预测数据</h4>
                    {df.to_html(classes='table table-striped', table_id='predictions-table')}
                </div>
                """
            
            if result.error_message:
                html += f"""
                <div class="error-message alert alert-danger">
                    <strong>错误信息:</strong> {result.error_message}
                </div>
                """
            
            html += "</div>"
            
            logger_manager.debug(f"结果格式化为HTML: {result.request_id}")
            return html
            
        except Exception as e:
            logger_manager.error(f"HTML格式化失败: {e}")
            raise PredictionException(f"HTML格式化失败: {e}")


class ResultAnalyzer:
    """结果分析器"""
    
    def __init__(self):
        """初始化结果分析器"""
        logger_manager.info("结果分析器初始化完成")
    
    def analyze_statistical(self, predictions: np.ndarray) -> Dict[str, Any]:
        """统计分析"""
        try:
            if predictions is None or len(predictions) == 0:
                return {}
            
            # 展平多维数组
            flat_predictions = predictions.flatten()
            
            analysis = {
                'count': len(flat_predictions),
                'mean': float(np.mean(flat_predictions)),
                'std': float(np.std(flat_predictions)),
                'min': float(np.min(flat_predictions)),
                'max': float(np.max(flat_predictions)),
                'median': float(np.median(flat_predictions)),
                'q25': float(np.percentile(flat_predictions, 25)),
                'q75': float(np.percentile(flat_predictions, 75)),
                'variance': float(np.var(flat_predictions)),
                'skewness': self._calculate_skewness(flat_predictions),
                'kurtosis': self._calculate_kurtosis(flat_predictions)
            }
            
            logger_manager.debug("统计分析完成")
            return analysis
            
        except Exception as e:
            logger_manager.error(f"统计分析失败: {e}")
            return {}
    
    def analyze_distribution(self, predictions: np.ndarray, bins: int = 20) -> Dict[str, Any]:
        """分布分析"""
        try:
            if predictions is None or len(predictions) == 0:
                return {}
            
            flat_predictions = predictions.flatten()
            
            # 计算直方图
            hist, bin_edges = np.histogram(flat_predictions, bins=bins)
            
            analysis = {
                'histogram': {
                    'counts': hist.tolist(),
                    'bin_edges': bin_edges.tolist(),
                    'bins': bins
                },
                'distribution_stats': {
                    'range': float(np.max(flat_predictions) - np.min(flat_predictions)),
                    'iqr': float(np.percentile(flat_predictions, 75) - np.percentile(flat_predictions, 25)),
                    'mode_bin': int(np.argmax(hist)),
                    'entropy': self._calculate_entropy(hist)
                }
            }
            
            logger_manager.debug("分布分析完成")
            return analysis
            
        except Exception as e:
            logger_manager.error(f"分布分析失败: {e}")
            return {}
    
    def analyze_confidence(self, confidence_scores: np.ndarray) -> Dict[str, Any]:
        """置信度分析"""
        try:
            if confidence_scores is None or len(confidence_scores) == 0:
                return {}
            
            flat_confidence = confidence_scores.flatten()
            
            analysis = {
                'mean_confidence': float(np.mean(flat_confidence)),
                'min_confidence': float(np.min(flat_confidence)),
                'max_confidence': float(np.max(flat_confidence)),
                'std_confidence': float(np.std(flat_confidence)),
                'high_confidence_ratio': float(np.sum(flat_confidence > 0.8) / len(flat_confidence)),
                'low_confidence_ratio': float(np.sum(flat_confidence < 0.5) / len(flat_confidence)),
                'confidence_distribution': {
                    'very_high': float(np.sum(flat_confidence > 0.9) / len(flat_confidence)),
                    'high': float(np.sum((flat_confidence > 0.7) & (flat_confidence <= 0.9)) / len(flat_confidence)),
                    'medium': float(np.sum((flat_confidence > 0.5) & (flat_confidence <= 0.7)) / len(flat_confidence)),
                    'low': float(np.sum(flat_confidence <= 0.5) / len(flat_confidence))
                }
            }
            
            logger_manager.debug("置信度分析完成")
            return analysis
            
        except Exception as e:
            logger_manager.error(f"置信度分析失败: {e}")
            return {}
    
    def detect_outliers(self, predictions: np.ndarray, method: str = "iqr") -> Dict[str, Any]:
        """异常值检测"""
        try:
            if predictions is None or len(predictions) == 0:
                return {}
            
            flat_predictions = predictions.flatten()
            
            if method == "iqr":
                q1 = np.percentile(flat_predictions, 25)
                q3 = np.percentile(flat_predictions, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = (flat_predictions < lower_bound) | (flat_predictions > upper_bound)
                
            elif method == "zscore":
                z_scores = np.abs((flat_predictions - np.mean(flat_predictions)) / np.std(flat_predictions))
                outliers = z_scores > 3
                
            else:
                raise PredictionException(f"不支持的异常值检测方法: {method}")
            
            outlier_indices = np.where(outliers)[0]
            outlier_values = flat_predictions[outliers]
            
            analysis = {
                'method': method,
                'outlier_count': len(outlier_indices),
                'outlier_ratio': float(len(outlier_indices) / len(flat_predictions)),
                'outlier_indices': outlier_indices.tolist(),
                'outlier_values': outlier_values.tolist(),
                'bounds': {
                    'lower': float(lower_bound) if method == "iqr" else None,
                    'upper': float(upper_bound) if method == "iqr" else None
                }
            }
            
            logger_manager.debug(f"异常值检测完成，方法: {method}")
            return analysis
            
        except Exception as e:
            logger_manager.error(f"异常值检测失败: {e}")
            return {}
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """计算偏度"""
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0.0
            
            skewness = np.mean(((data - mean) / std) ** 3)
            return float(skewness)
            
        except Exception:
            return 0.0
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """计算峰度"""
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0.0
            
            kurtosis = np.mean(((data - mean) / std) ** 4) - 3
            return float(kurtosis)
            
        except Exception:
            return 0.0
    
    def _calculate_entropy(self, hist: np.ndarray) -> float:
        """计算熵"""
        try:
            # 归一化直方图
            prob = hist / np.sum(hist)
            # 避免log(0)
            prob = prob[prob > 0]
            entropy = -np.sum(prob * np.log2(prob))
            return float(entropy)
            
        except Exception:
            return 0.0


class ResultProcessor:
    """结果处理器"""
    
    def __init__(self):
        """初始化结果处理器"""
        self.formatter = ResultFormatter()
        self.analyzer = ResultAnalyzer()
        
        logger_manager.info("结果处理器初始化完成")
    
    def process_result(self, result: PredictionResult, 
                      format_type: ResultFormat = ResultFormat.JSON,
                      include_analysis: bool = True,
                      analysis_types: List[AnalysisType] = None) -> ProcessedResult:
        """
        处理预测结果
        
        Args:
            result: 预测结果
            format_type: 格式类型
            include_analysis: 是否包含分析
            analysis_types: 分析类型列表
            
        Returns:
            处理后的结果
        """
        try:
            # 格式化数据
            if format_type == ResultFormat.JSON:
                formatted_data = self.formatter.format_to_json(result)
            elif format_type == ResultFormat.PANDAS:
                formatted_data = self.formatter.format_to_dataframe(result)
            elif format_type == ResultFormat.CSV:
                formatted_data = self.formatter.format_to_csv(result)
            elif format_type == ResultFormat.HTML:
                formatted_data = self.formatter.format_to_html(result)
            elif format_type == ResultFormat.NUMPY:
                formatted_data = result.predictions
            else:
                formatted_data = result
            
            # 创建处理结果
            processed = ProcessedResult(
                original_result=result,
                formatted_data=formatted_data,
                format_type=format_type
            )
            
            # 执行分析
            if include_analysis and result.predictions is not None:
                analysis_types = analysis_types or [AnalysisType.STATISTICAL]
                
                for analysis_type in analysis_types:
                    if analysis_type == AnalysisType.STATISTICAL:
                        processed.analysis['statistical'] = self.analyzer.analyze_statistical(result.predictions)
                    elif analysis_type == AnalysisType.DISTRIBUTION:
                        processed.analysis['distribution'] = self.analyzer.analyze_distribution(result.predictions)
                    elif analysis_type == AnalysisType.OUTLIER:
                        processed.analysis['outliers'] = self.analyzer.detect_outliers(result.predictions)
                
                # 置信度分析
                if result.confidence_scores is not None:
                    processed.analysis['confidence'] = self.analyzer.analyze_confidence(result.confidence_scores)
            
            logger_manager.info(f"结果处理完成: {result.request_id}")
            return processed
            
        except Exception as e:
            logger_manager.error(f"结果处理失败: {e}")
            raise PredictionException(f"结果处理失败: {e}")
    
    def batch_process_results(self, results: List[PredictionResult],
                            format_type: ResultFormat = ResultFormat.JSON,
                            **kwargs) -> List[ProcessedResult]:
        """批量处理结果"""
        try:
            processed_results = []
            
            for result in results:
                processed = self.process_result(result, format_type, **kwargs)
                processed_results.append(processed)
            
            logger_manager.info(f"批量结果处理完成: {len(processed_results)} 个")
            return processed_results
            
        except Exception as e:
            logger_manager.error(f"批量结果处理失败: {e}")
            raise PredictionException(f"批量结果处理失败: {e}")
    
    def export_result(self, processed_result: ProcessedResult, 
                     file_path: str) -> bool:
        """导出结果到文件"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            if processed_result.format_type == ResultFormat.JSON:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(processed_result.formatted_data, f, indent=2, ensure_ascii=False)
            
            elif processed_result.format_type == ResultFormat.CSV:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(processed_result.formatted_data)
            
            elif processed_result.format_type == ResultFormat.HTML:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(processed_result.formatted_data)
            
            elif processed_result.format_type == ResultFormat.PANDAS:
                if file_path.endswith('.csv'):
                    processed_result.formatted_data.to_csv(file_path, index=False)
                elif file_path.endswith('.xlsx'):
                    processed_result.formatted_data.to_excel(file_path, index=False)
                else:
                    processed_result.formatted_data.to_pickle(file_path)
            
            else:
                # 默认保存为pickle
                import pickle
                with open(file_path, 'wb') as f:
                    pickle.dump(processed_result.formatted_data, f)
            
            logger_manager.info(f"结果已导出: {file_path}")
            return True
            
        except Exception as e:
            logger_manager.error(f"导出结果失败: {e}")
            return False
    
    def create_summary_report(self, results: List[PredictionResult]) -> Dict[str, Any]:
        """创建汇总报告"""
        try:
            if not results:
                return {}
            
            # 基础统计
            total_results = len(results)
            successful_results = sum(1 for r in results if r.status == PredictionStatus.COMPLETED)
            failed_results = sum(1 for r in results if r.status == PredictionStatus.FAILED)
            
            # 执行时间统计
            execution_times = [r.execution_time for r in results if r.execution_time > 0]
            
            # 预测结果统计
            all_predictions = []
            all_confidences = []
            
            for result in results:
                if result.predictions is not None:
                    all_predictions.extend(result.predictions.flatten())
                if result.confidence_scores is not None:
                    all_confidences.extend(result.confidence_scores.flatten())
            
            summary = {
                'overview': {
                    'total_predictions': total_results,
                    'successful_predictions': successful_results,
                    'failed_predictions': failed_results,
                    'success_rate': successful_results / total_results if total_results > 0 else 0
                },
                'performance': {
                    'avg_execution_time': np.mean(execution_times) if execution_times else 0,
                    'min_execution_time': np.min(execution_times) if execution_times else 0,
                    'max_execution_time': np.max(execution_times) if execution_times else 0,
                    'total_execution_time': np.sum(execution_times) if execution_times else 0
                },
                'predictions_analysis': self.analyzer.analyze_statistical(np.array(all_predictions)) if all_predictions else {},
                'confidence_analysis': self.analyzer.analyze_confidence(np.array(all_confidences)) if all_confidences else {},
                'generated_time': datetime.now().isoformat()
            }
            
            logger_manager.info(f"汇总报告创建完成: {total_results} 个结果")
            return summary
            
        except Exception as e:
            logger_manager.error(f"创建汇总报告失败: {e}")
            return {}


# 全局结果处理器实例
result_processor = ResultProcessor()


if __name__ == "__main__":
    # 测试结果处理器功能
    print("📊 测试结果处理器功能...")
    
    try:
        from .prediction_engine import PredictionResult, PredictionStatus
        import numpy as np
        
        # 创建测试结果
        test_result = PredictionResult(
            request_id="test_request",
            status=PredictionStatus.COMPLETED,
            predictions=np.random.random((10, 2)),
            confidence_scores=np.random.random(10),
            execution_time=1.5
        )
        
        # 测试结果处理
        processor = ResultProcessor()
        
        # JSON格式处理
        json_result = processor.process_result(test_result, ResultFormat.JSON)
        print("✅ JSON格式处理成功")
        
        # DataFrame格式处理
        df_result = processor.process_result(test_result, ResultFormat.PANDAS)
        print(f"✅ DataFrame格式处理成功，形状: {df_result.formatted_data.shape}")
        
        # 包含分析的处理
        analyzed_result = processor.process_result(
            test_result, 
            ResultFormat.JSON,
            include_analysis=True,
            analysis_types=[AnalysisType.STATISTICAL, AnalysisType.DISTRIBUTION]
        )
        print(f"✅ 分析处理成功，分析项: {list(analyzed_result.analysis.keys())}")
        
        # 测试汇总报告
        summary = processor.create_summary_report([test_result])
        print(f"✅ 汇总报告创建成功，成功率: {summary['overview']['success_rate']}")
        
        print("✅ 结果处理器功能测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("结果处理器功能测试完成")
