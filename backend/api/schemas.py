#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FastAPI 请求与响应模型定义"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, validator

AlgorithmLiteral = Literal[
    'frequency', 'hot_cold', 'missing',
    'markov', 'markov_2nd', 'markov_3rd', 'adaptive_markov', 'markov_custom', 'markov_compound',
    'bayesian', 'ensemble', 'clustering', 'consensus_halving',
    'super', 'adaptive', 'mixed_strategy', 'highly_integrated', 'advanced_integration',
    'nine_models', 'nine_models_compound',
    'compound', 'duplex',
    'lstm', 'transformer', 'gan', 'stacking', 'adaptive_ensemble', 'ultimate_ensemble',
    'enhanced'
]


class ApiResponse(BaseModel):
    """统一响应格式"""

    success: bool = True
    data: Optional[Any] = None
    message: str = ''


class PaginationParams(BaseModel):
    """分页参数"""

    page: int = Field(1, ge=1)
    page_size: int = Field(50, ge=1, le=200)


class PredictionRequest(BaseModel):
    """预测请求体"""

    method: AlgorithmLiteral = Field('ensemble')
    count: int = Field(1, ge=1, le=100)
    periods: int = Field(500, ge=50, le=2748)
    missing_mode: Optional[Literal['auto', 'legacy', 'enhanced']] = Field('auto')
    front_count: int = Field(8, ge=6, le=15)
    back_count: int = Field(4, ge=3, le=12)
    front_dan: int = Field(2, ge=1, le=5)
    back_dan: int = Field(1, ge=1, le=2)
    front_tuo: int = Field(6, ge=3, le=30)
    back_tuo: int = Field(4, ge=2, le=20)
    analysis_periods: int = Field(300, ge=50, le=2748)
    predict_periods: int = Field(1, ge=1, le=30)
    strategy: str = Field('balanced')
    integration_level: str = Field('ultimate')
    integration_type: str = Field('comprehensive')
    compound_mode: bool = False
    max_cost: int = Field(10000, ge=100)
    min_confidence: float = Field(0.5, ge=0.0, le=1.0)
    acceleration: str = Field('auto')
    cpu_threads: int = Field(-1, ge=-1)
    gpu_device: int = Field(0, ge=0)
    gpu_memory_limit: Optional[float] = Field(None, gt=0)
    mixed_precision: bool = False
    batch_size_multiplier: float = Field(1.0, gt=0)
    benchmark_hardware: bool = False
    fallback_enabled: bool = True
    auto_epochs: bool = False
    min_epochs: int = Field(10, ge=1)
    max_epochs: int = Field(1000, ge=10)
    performance_mode: str = Field('medium')
    training_intensity: float = Field(1.0, gt=0)

    @validator('max_epochs')
    def validate_epochs(cls, value, values):
        min_epochs = values.get('min_epochs', 10)
        if value < min_epochs:
            raise ValueError('最大训练轮数必须大于最小训练轮数')
        return value


class PredictionResult(BaseModel):
    """预测结果项"""

    front_balls: List[int] = []
    back_balls: List[int] = []
    method: Optional[str] = None
    confidence: Optional[float] = None
    details: Optional[Dict[str, Any]] = None


class CompoundResultModel(BaseModel):
    """复式预测结果"""

    front_balls: List[int]
    back_balls: List[int]
    front_count: Optional[int] = None
    back_count: Optional[int] = None
    total_combinations: Optional[int] = None
    total_cost: Optional[int] = None
    confidence: Optional[float] = None
    method: Optional[str] = None
    analysis_periods: Optional[int] = None
    timestamp: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    raw: Optional[Any] = None


class PredictionResponseData(BaseModel):
    """预测响应数据"""

    mode: str
    predictions: List[PredictionResult] = []
    compound: Optional[CompoundResultModel] = None


class PredictionHistoryRecord(BaseModel):
    """预测历史记录"""

    timestamp: str
    request: Dict[str, Any]
    response: Dict[str, Any]


class PaginatedHistory(BaseModel):
    """预测历史分页结构"""

    total: int
    page: int
    page_size: int
    records: List[PredictionHistoryRecord]


class AnalysisRequest(BaseModel):
    """分析通用请求"""

    periods: Optional[int] = Field(None, ge=50, le=2748)


class CompareRequest(BaseModel):
    """批量对比请求"""

    target_issue: str = Field(..., min_length=5)
    method: AlgorithmLiteral = Field('markov')
    periods: int = Field(100, ge=50, le=2748)
    times: int = Field(50, ge=1, le=10000)
    random_periods: bool = False
    min_periods: int = Field(20, ge=10, le=2748)
    max_periods: Optional[int] = Field(None, ge=50, le=2748)
    export_excel: bool = False
    show_progress: bool = False


TestingStrategyLiteral = Literal['progressive', 'random']
TestingPrizeLiteral = Literal['一等奖', '二等奖', '三等奖', '四等奖', '五等奖', '六等奖', '七等奖', '八等奖', '九等奖']


class TestingRequest(BaseModel):
    """测试系统请求参数"""

    methods: List[AlgorithmLiteral] = Field(default_factory=list)
    strategy: TestingStrategyLiteral = Field('random')
    target_prize: TestingPrizeLiteral = Field('六等奖')
    periods_start: int = Field(50, ge=10, le=2748)
    periods_end: int = Field(500, ge=10, le=2748)
    count_start: int = Field(1, ge=1, le=100)
    count_end: int = Field(1, ge=1, le=100)
    max_tests: int = Field(20, ge=1, le=5000)
    parallel: bool = Field(False)
    workers: int = Field(4, ge=1, le=64)
    seed: Optional[int] = Field(None)
    target_issue: Optional[str] = Field(None, min_length=5)
    progressive_step: int = Field(50, ge=1, le=1000)
    timeout_seconds: int = Field(120, ge=10, le=3600)
    retries: int = Field(1, ge=0, le=5)

    @validator('methods')
    def validate_methods(cls, value):
        if not value:
            raise ValueError('methods 不能为空')
        return sorted(set(value))

    @validator('periods_end')
    def validate_periods_range(cls, value, values):
        start = values.get('periods_start', 10)
        if value < start:
            raise ValueError('periods_end 必须大于等于 periods_start')
        return value

    @validator('count_end')
    def validate_count_range(cls, value, values):
        start = values.get('count_start', 1)
        if value < start:
            raise ValueError('count_end 必须大于等于 count_start')
        return value


class TestingOptionsResponse(BaseModel):
    """测试系统可选项响应"""

    available_methods: List[str]
    target_prizes: List[str]

class CompareSummary(BaseModel):
    """批量对比摘要"""

    config: Dict[str, Any]
    actual_numbers: Dict[str, Any]
    statistics: Dict[str, Any]
    execution_info: Dict[str, Any]


class ComparisonRecord(BaseModel):
    """单次对比记录"""

    round_number: int
    analysis_periods: int
    predicted_front: List[int]
    predicted_back: List[int]
    actual_front: List[int]
    actual_back: List[int]
    front_hits: int
    back_hits: int
    prize_level: int
    execution_time: float
    timestamp: str


class CompareResponseData(BaseModel):
    """批量对比响应数据"""

    summary: CompareSummary
    records: List[ComparisonRecord]


class DataRecord(BaseModel):
    """单条开奖数据"""

    issue: str
    date: str
    front_balls: List[int]
    back_balls: List[int]
    prize_grades: Optional[List[Dict[str, Any]]] = None


class DataHistoryResponse(BaseModel):
    """历史数据分页响应"""

    total: int
    page: int
    page_size: int
    records: List[DataRecord]


class DataStatsResponse(BaseModel):
    """数据统计信息"""

    total_periods: int
    date_range: Dict[str, Any]
    latest_issue: Any
    cache: Optional[Dict[str, Any]] = None


class SystemInfoResponse(BaseModel):
    """系统信息"""

    enhanced_available: bool
    total_periods: int
    date_range: Dict[str, Any]
    cache_dir: str
    models_dir: str
    logs_dir: str


class HealthStatusResponse(BaseModel):
    """健康检查响应"""

    status: str
    components: Dict[str, bool]


class DataUpdateResponse(BaseModel):
    """数据更新响应"""

    updated_count: int
    total_periods: int
    latest_issue: Optional[str] = None
    latest_date: Optional[str] = None
    source: Optional[str] = None
    reason: Optional[str] = None
    updated_at: Optional[str] = None
    skipped: bool = False
    message: str = ''
