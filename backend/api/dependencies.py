#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FastAPI 依赖及通用服务封装"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from backend.app.main import DLTPredictorSystem
from backend.app.core import core_modules as cm
from backend.app.core.method_categories import REQUIRES_DEEP_LEARNING
from backend.app.analyzers import analyzer_modules

# 批量对比模块导入（可选，由于循环依赖问题可能失败）
try:
    from backend.app.utils.batch_comparison_module import (
        BatchComparison,
        BatchComparisonConfig,
        ComparisonResult,
        PredictionRecord,
    )
    BATCH_COMPARISON_AVAILABLE = True
except ImportError as e:
    logging.warning(f"批量对比模块导入失败: {e}")
    BATCH_COMPARISON_AVAILABLE = False
    # 定义占位类
    class BatchComparison:
        def execute(self, config):
            raise NotImplementedError("批量对比模块不可用")
    class BatchComparisonConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        def validate(self):
            return False, "批量对比模块不可用"
    class ComparisonResult:
        pass
    class PredictionRecord:
        pass

logger = logging.getLogger(__name__)


# ==================== 单例获取函数 ====================

def get_data_manager():
    """返回核心数据管理器"""
    return cm.data_manager


def get_cache_manager():
    """返回缓存管理器"""
    return cm.cache_manager


def get_basic_analyzer():
    """返回基础分析器实例"""
    return analyzer_modules.basic_analyzer


def get_advanced_analyzer():
    """返回高级分析器实例"""
    return analyzer_modules.advanced_analyzer


def get_visualization_analyzer():
    """返回可视化分析器实例"""
    return analyzer_modules.visualization_analyzer


def get_predictor_system() -> DLTPredictorSystem:
    """惰性构建 DLTPredictorSystem"""
    global _SYSTEM_INSTANCE
    try:
        system = _SYSTEM_INSTANCE
    except NameError:
        system = DLTPredictorSystem()
        _SYSTEM_INSTANCE = system
    return system


def get_batch_comparison() -> BatchComparison:
    """每次返回新的批量对比器，内部已缓存预测器"""
    return BatchComparison()


# ==================== 算法元数据 ====================

ALGORITHM_DEFINITIONS: List[Dict[str, Any]] = [
    {"id": "frequency", "name": "频率分析", "category": "传统统计", "description": "根据期号频率分布生成组合", "support_compound": False},
    {"id": "hot_cold", "name": "冷热分析", "category": "传统统计", "description": "结合热号冷号趋势", "support_compound": False},
    {"id": "missing", "name": "遗漏分析", "category": "传统统计", "description": "评估号码遗漏周期", "support_compound": False},
    {"id": "markov", "name": "一阶马尔可夫", "category": "高级算法", "description": "基于状态转移矩阵预测", "support_compound": False},
    {"id": "markov_2nd", "name": "二阶马尔可夫", "category": "高级算法", "description": "使用更长历史刻画惯性", "support_compound": False},
    {"id": "consensus_halving", "name": "交集递减融合", "category": "高级算法", "description": "冷热号+马尔可夫+频率交集提取并按期数对半递减补齐", "support_compound": False},
    {"id": "markov_3rd", "name": "三阶马尔可夫", "category": "高级算法", "description": "多阶链综合判断趋势", "support_compound": False},
    {"id": "adaptive_markov", "name": "自适应马尔可夫", "category": "高级算法", "description": "动态调整阶数与权重", "support_compound": False},
    {"id": "bayesian", "name": "贝叶斯推理", "category": "高级算法", "description": "按照先验与似然生成概率模型", "support_compound": False},
    {"id": "ensemble", "name": "集成预测", "category": "高级算法", "description": "组合多模型加权", "support_compound": False},
    {"id": "clustering", "name": "聚类预测", "category": "高级算法", "description": "向量化特征后进行 K-Means", "support_compound": False},
    {"id": "super", "name": "超级预测", "category": "智能增强", "description": "调用超级预测器", "support_compound": True},
    {"id": "adaptive", "name": "自适应学习", "category": "智能增强", "description": "多臂老虎机动态选择预测器", "support_compound": False},
    {"id": "compound", "name": "复式选号", "category": "投注策略", "description": "生成高覆盖复式组合", "support_compound": True},
    {"id": "duplex", "name": "胆拖选号", "category": "投注策略", "description": "根据胆码拖码构建组合", "support_compound": True},
    {"id": "markov_custom", "name": "自定义马尔可夫", "category": "高级算法", "description": "自定义分析期与预测期", "support_compound": False},
    {"id": "mixed_strategy", "name": "混合策略", "category": "智能增强", "description": "多策略融合控制风险", "support_compound": False},
    {"id": "highly_integrated", "name": "高度集成", "category": "智能增强", "description": "多模型融合并加入评估", "support_compound": True},
    {"id": "advanced_integration", "name": "高级集成", "category": "智能增强", "description": "综合热冷、马尔可夫、贝叶斯等", "support_compound": True},
    {"id": "nine_models", "name": "九模型融合", "category": "智能增强", "description": "九种数学模型投票", "support_compound": False},
    {"id": "nine_models_compound", "name": "九模型复式", "category": "投注策略", "description": "九模型结果生成复式", "support_compound": True},
    {"id": "markov_compound", "name": "马尔可夫复式", "category": "投注策略", "description": "基于马尔可夫生成复式", "support_compound": True},
    {"id": "lstm", "name": "LSTM 深度学习", "category": "深度学习", "description": "调用 LSTM 序列模型", "support_compound": False},
    {"id": "transformer", "name": "Transformer", "category": "深度学习", "description": "Transformer 时序预测模型", "support_compound": False},
    {"id": "gan", "name": "GAN 生成式", "category": "深度学习", "description": "生成式对抗网络构造组合", "support_compound": False},
    {"id": "stacking", "name": "Stacking 集成", "category": "深度学习", "description": "深度模型堆叠融合", "support_compound": False},
    {"id": "adaptive_ensemble", "name": "自适应集成", "category": "深度学习", "description": "根据表现自适应调整", "support_compound": False},
    {"id": "ultimate_ensemble", "name": "终极集成", "category": "深度学习", "description": "多通道终极加权", "support_compound": True},
    {"id": "enhanced", "name": "增强引擎", "category": "系统增强", "description": "使用 enhanced_integration 提供的高级能力", "support_compound": True},
]


# ==================== 预测辅助类 ====================

class PredictionHistoryManager:
    """预测历史管理，存放在缓存目录中"""

    def __init__(self, cache_manager=None, max_records: int = 200):
        self.cache_manager = cache_manager or get_cache_manager()
        self.history_file = Path(self.cache_manager.get_cache_path('analysis', 'prediction_history'))
        self.max_records = max_records
        self._lock = Lock()
        self._records: List[Dict[str, Any]] = self._load()

    def _load(self) -> List[Dict[str, Any]]:
        if not self.history_file.exists():
            return []
        try:
            with self.history_file.open('r', encoding='utf-8') as fp:
                data = json.load(fp)
            if isinstance(data, list):
                return data
        except Exception as exc:
            logger.warning("预测历史加载失败: %s", exc)
        return []

    def _persist(self) -> None:
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        with self.history_file.open('w', encoding='utf-8') as fp:
            json.dump(self._records, fp, ensure_ascii=False, indent=2)

    def add_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            record.setdefault('timestamp', datetime.utcnow().isoformat())
            self._records.insert(0, record)
            if len(self._records) > self.max_records:
                self._records = self._records[: self.max_records]
            self._persist()
            return record

    def list_records(self, page: int, page_size: int) -> Dict[str, Any]:
        start = (page - 1) * page_size
        end = start + page_size
        total = len(self._records)
        return {
            'total': total,
            'page': page,
            'page_size': page_size,
            'records': self._records[start:end],
        }


_history_manager: Optional[PredictionHistoryManager] = None


def get_prediction_history_manager() -> PredictionHistoryManager:
    global _history_manager
    if _history_manager is None:
        _history_manager = PredictionHistoryManager()
    return _history_manager


class PredictorService:
    """封装 DLTPredictorSystem 的高层服务"""

    def __init__(self, system: Optional[DLTPredictorSystem] = None):
        self.system = system or get_predictor_system()

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        args = self._build_args(payload)
        self.system._load_predictors()
        is_valid, error_msg, acceleration_config = self.system._validate_predict_args(args)
        if not is_valid:
            raise ValueError(error_msg)

        use_enhanced = self.system.enhanced_available and args.method == 'enhanced' and not args.compound
        use_deep_learning = args.method in REQUIRES_DEEP_LEARNING

        if use_enhanced:
            success, predictions = self.system._handle_enhanced_prediction(args, acceleration_config)
            if success:
                return {
                    'mode': 'enhanced',
                    'predictions': self._normalize_predictions(predictions),
                }

        if use_deep_learning and not args.compound:
            success, predictions = self.system._handle_deep_learning_prediction(args, acceleration_config)
            if success and predictions:
                return {
                    'mode': 'deep_learning',
                    'predictions': self._normalize_predictions(predictions),
                }
            # 深度学习失败，回退到 ensemble 传统方法
            logger.warning(f"深度学习方法 {args.method} 失败，回退到 ensemble 方法")
            args.method = 'ensemble'

        if args.compound:
            compound_result = self.system._handle_compound_prediction(args)
            if compound_result:
                return {
                    'mode': 'compound',
                    'compound': self._serialize_compound(compound_result),
                    'predictions': [],
                }

        predictions = self.system._handle_traditional_prediction(args, acceleration_config)
        normalized = self._normalize_predictions(predictions)
        return {
            'mode': 'traditional',
            'predictions': normalized,
        }

    def _build_args(self, payload: Dict[str, Any]) -> SimpleNamespace:
        defaults = {
            'count': 1,
            'periods': 500,
            'method': 'ensemble',
            'missing_mode': 'auto',
            'front_count': 8,
            'back_count': 4,
            'front_dan': 2,
            'back_dan': 1,
            'front_tuo': 6,
            'back_tuo': 4,
            'analysis_periods': 300,
            'predict_periods': 1,
            'strategy': 'balanced',
            'integration_level': 'ultimate',
            'integration_type': 'comprehensive',
            'compound': payload.get('compound_mode', False),
            'max_cost': 10000,
            'min_confidence': 0.5,
            'acceleration': payload.get('acceleration', 'auto'),
            'cpu_threads': payload.get('cpu_threads', -1),
            'gpu_device': payload.get('gpu_device', 0),
            'gpu_memory_limit': payload.get('gpu_memory_limit'),
            'mixed_precision': payload.get('mixed_precision', False),
            'batch_size_multiplier': payload.get('batch_size_multiplier', 1.0),
            'benchmark_hardware': payload.get('benchmark_hardware', False),
            'fallback_enabled': payload.get('fallback_enabled', True),
            'auto_epochs': payload.get('auto_epochs', False),
            'min_epochs': payload.get('min_epochs', 10),
            'max_epochs': payload.get('max_epochs', 1000),
            'performance_mode': payload.get('performance_mode', 'medium'),
            'training_intensity': payload.get('training_intensity', 1.0),
        }
        merged = {**defaults, **payload}
        merged['compound'] = payload.get('compound_mode', merged.get('compound', False))
        return SimpleNamespace(**merged)

    def _normalize_predictions(self, predictions: Any) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        if not predictions:
            return normalized
        for pred in predictions:
            normalized.append(self._serialize_prediction(pred))
        return normalized

    def _serialize_prediction(self, pred: Any) -> Dict[str, Any]:
        if pred is None:
            return {}
        if isinstance(pred, dict):
            details = {k: _safe_value(v) for k, v in pred.items() if k not in {'front_balls', 'back_balls', 'method', 'confidence'}}
            return {
                'front_balls': _ensure_int_list(pred.get('front_balls')),
                'back_balls': _ensure_int_list(pred.get('back_balls')),
                'method': pred.get('method'),
                'confidence': pred.get('confidence'),
                'details': details or None,
            }
        if hasattr(pred, 'front_balls') and hasattr(pred, 'back_balls'):
            data = {
                'front_balls': _ensure_int_list(getattr(pred, 'front_balls')),
                'back_balls': _ensure_int_list(getattr(pred, 'back_balls')),
                'method': getattr(pred, 'method', None),
            }
            if hasattr(pred, 'confidence'):
                data['confidence'] = getattr(pred, 'confidence')
            extra = {}
            for attr in ['total_combinations', 'total_cost', 'analysis_periods', 'timestamp', 'details']:
                if hasattr(pred, attr):
                    extra[attr] = _safe_value(getattr(pred, attr))
            if extra:
                data['details'] = extra
            return data
        if isinstance(pred, (list, tuple)) and len(pred) == 2:
            return {
                'front_balls': _ensure_int_list(pred[0]),
                'back_balls': _ensure_int_list(pred[1]),
                'method': None,
            }
        return {'raw': pred}

    def _serialize_compound(self, compound_result: Any) -> Dict[str, Any]:
        if hasattr(compound_result, '__dict__'):
            base = asdict(compound_result) if hasattr(compound_result, '__dataclass_fields__') else dict(compound_result.__dict__)
        elif isinstance(compound_result, dict):
            base = compound_result
        else:
            base = {'raw': compound_result}
        if 'front_balls' in base:
            base['front_balls'] = _ensure_int_list(base['front_balls'])
        if 'back_balls' in base:
            base['back_balls'] = _ensure_int_list(base['back_balls'])
        if 'details' in base:
            base['details'] = _safe_value(base['details'])
        return base


def _ensure_int_list(values: Any) -> List[int]:
    if values is None:
        return []
    result = []
    for value in values:
        try:
            result.append(int(value))
        except (TypeError, ValueError):
            continue
    return result


def _safe_value(value: Any) -> Any:
    """将复杂对象转换为可序列化结构"""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {k: _safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_value(v) for v in value]
    if hasattr(value, 'tolist'):
        try:
            return value.tolist()
        except Exception:
            return str(value)
    return str(value)


__all__ = [
    'ALGORITHM_DEFINITIONS',
    'BatchComparison',
    'BatchComparisonConfig',
    'ComparisonResult',
    'PredictionHistoryManager',
    'PredictorService',
    'PredictionRecord',
    'get_data_manager',
    'get_cache_manager',
    'get_basic_analyzer',
    'get_advanced_analyzer',
    'get_visualization_analyzer',
    'get_predictor_system',
    'get_batch_comparison',
    'get_prediction_history_manager',
]
