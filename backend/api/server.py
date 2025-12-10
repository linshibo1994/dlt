#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FastAPI 服务器 - 对外暴露预测与分析接口"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, List

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.core import core_modules as cm

from . import schemas
from .dependencies import (
    ALGORITHM_DEFINITIONS,
    BatchComparisonConfig,
    PredictorService,
    get_advanced_analyzer,
    get_basic_analyzer,
    get_batch_comparison,
    get_cache_manager,
    get_data_manager,
    get_prediction_history_manager,
    get_predictor_system,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="DLT Predictor API",
    description="大乐透智能预测系统 REST 接口",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS 配置 - 从环境变量读取允许的来源，生产环境应该配置具体域名
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:80,http://127.0.0.1:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

_predictor_service = PredictorService()


def get_predictor_service_instance() -> PredictorService:
    return _predictor_service


def build_response(data: Any = None, message: str = "") -> schemas.ApiResponse:
    return schemas.ApiResponse(success=True, data=data, message=message)


def raise_http_error(message: str, status_code: int = status.HTTP_400_BAD_REQUEST):
    raise HTTPException(status_code=status_code, detail=message)


# ==================== 数据接口 ====================

@app.get("/api/data/latest", response_model=schemas.ApiResponse)
def get_latest_data(data_manager=Depends(get_data_manager)):
    df = data_manager.get_data()
    if df is None or len(df) == 0:
        raise_http_error("暂未加载到开奖数据", status.HTTP_503_SERVICE_UNAVAILABLE)

    latest = df.iloc[0]
    front, back = data_manager.parse_balls(latest)
    record = schemas.DataRecord(
        issue=str(latest.get('issue')),
        date=str(latest.get('date')),
        front_balls=[int(x) for x in front],
        back_balls=[int(x) for x in back],
    )
    return build_response(record.dict(), "获取最新数据成功")


@app.get("/api/data/history", response_model=schemas.ApiResponse)
def get_data_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    data_manager=Depends(get_data_manager),
):
    df = data_manager.get_data()
    total = len(df) if df is not None else 0
    if total == 0:
        return build_response(schemas.DataHistoryResponse(total=0, page=page, page_size=page_size, records=[]).dict())

    start = (page - 1) * page_size
    end = start + page_size
    subset = df.iloc[start:end]

    records: List[schemas.DataRecord] = []
    for _, row in subset.iterrows():
        front, back = data_manager.parse_balls(row)
        records.append(
            schemas.DataRecord(
                issue=str(row.get('issue')),
                date=str(row.get('date')),
                front_balls=[int(x) for x in front],
                back_balls=[int(x) for x in back],
            )
        )

    response = schemas.DataHistoryResponse(
        total=total,
        page=page,
        page_size=page_size,
        records=records,
    )
    return build_response(response.dict())


@app.get("/api/data/stats", response_model=schemas.ApiResponse)
def get_data_stats(data_manager=Depends(get_data_manager), cache_manager=Depends(get_cache_manager)):
    stats = data_manager.get_stats() or {}
    cache_info = cache_manager.get_cache_info()
    response = schemas.DataStatsResponse(
        total_periods=stats.get('total_periods', 0),
        date_range=stats.get('date_range', {}),
        latest_issue=stats.get('latest_issue'),
        cache={
            'cache_dir': cache_info.get('cache_dir'),
            'total': cache_info.get('total'),
        },
    )
    return build_response(response.dict(), "统计信息获取成功")


# ==================== 预测接口 ====================

@app.get("/api/algorithms", response_model=schemas.ApiResponse)
def list_algorithms():
    return build_response(ALGORITHM_DEFINITIONS)


@app.post("/api/predict", response_model=schemas.ApiResponse)
def run_prediction(
    request: schemas.PredictionRequest,
    service: PredictorService = Depends(get_predictor_service_instance),
    history_manager=Depends(get_prediction_history_manager),
):
    try:
        result = service.predict(request.dict())
    except ValueError as exc:
        raise_http_error(str(exc))
    except Exception as exc:
        logger.exception("预测执行失败: %s", exc)
        raise_http_error("预测执行失败，请检查日志", status.HTTP_500_INTERNAL_SERVER_ERROR)

    response_data = schemas.PredictionResponseData(
        mode=result.get('mode', 'traditional'),
        predictions=[schemas.PredictionResult(**item) for item in result.get('predictions', [])],
        compound=schemas.CompoundResultModel(**result['compound']) if result.get('compound') else None,
    )

    history_manager.add_record(
        {
            'timestamp': datetime.utcnow().isoformat(),
            'request': request.dict(),
            'response': response_data.dict(),
        }
    )

    return build_response(response_data.dict(), "预测成功")


@app.get("/api/predict/history", response_model=schemas.ApiResponse)
def get_prediction_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    history_manager=Depends(get_prediction_history_manager),
):
    raw = history_manager.list_records(page, page_size)
    records = [schemas.PredictionHistoryRecord(**item) for item in raw.get('records', [])]
    response = schemas.PaginatedHistory(
        total=raw.get('total', 0),
        page=raw.get('page', page),
        page_size=raw.get('page_size', page_size),
        records=records,
    )
    return build_response(response.dict())


# ==================== 分析接口 ====================

@app.post("/api/analysis/frequency", response_model=schemas.ApiResponse)
def frequency_analysis(request: schemas.AnalysisRequest, analyzer=Depends(get_basic_analyzer)):
    result = analyzer.frequency_analysis(request.periods)
    return build_response(result)


@app.post("/api/analysis/hot-cold", response_model=schemas.ApiResponse)
def hot_cold_analysis(request: schemas.AnalysisRequest, analyzer=Depends(get_basic_analyzer)):
    result = analyzer.hot_cold_analysis(request.periods)
    return build_response(result)


@app.post("/api/analysis/trend", response_model=schemas.ApiResponse)
def trend_analysis(request: schemas.AnalysisRequest, analyzer=Depends(get_advanced_analyzer)):
    result = analyzer.trend_generation_analysis(request.periods or 500)
    return build_response(result)


@app.post("/api/analysis/missing", response_model=schemas.ApiResponse)
def missing_analysis(request: schemas.AnalysisRequest, analyzer=Depends(get_basic_analyzer)):
    result = analyzer.missing_analysis(request.periods)
    return build_response(result)


# ==================== 对比接口 ====================

@app.post("/api/compare", response_model=schemas.ApiResponse)
def run_comparison(request: schemas.CompareRequest, manager=Depends(get_batch_comparison)):
    config = BatchComparisonConfig(
        target_issue=request.target_issue,
        method=request.method,
        analysis_periods=request.periods,
        comparison_times=request.times,
        random_periods=request.random_periods,
        min_random_periods=request.min_periods,
        max_random_periods=request.max_periods,
        export_excel=request.export_excel,
        show_progress=request.show_progress,
    )

    valid, error_msg = config.validate()
    if not valid:
        raise_http_error(error_msg)

    try:
        result = manager.execute(config)
    except Exception as exc:
        logger.exception("批量对比执行失败: %s", exc)
        raise_http_error("批量对比执行失败，请检查日志", status.HTTP_500_INTERNAL_SERVER_ERROR)

    summary_data = result.get_summary()
    if request.export_excel:
        try:
            export_file = result.export_to_excel()
            summary_data.setdefault('execution_info', {})['export_file'] = export_file
        except Exception as exc:
            logger.warning("Excel 导出失败: %s", exc)

    summary = schemas.CompareSummary(**summary_data)
    records = [
        schemas.ComparisonRecord(
            round_number=record.round_number,
            analysis_periods=record.analysis_periods,
            predicted_front=[int(x) for x in record.predicted_front],
            predicted_back=[int(x) for x in record.predicted_back],
            actual_front=[int(x) for x in record.actual_front],
            actual_back=[int(x) for x in record.actual_back],
            front_hits=record.front_hits,
            back_hits=record.back_hits,
            prize_level=record.prize_level,
            execution_time=record.execution_time,
            timestamp=record.timestamp.isoformat() if hasattr(record, 'timestamp') else str(record.timestamp),
        )
        for record in result.predictions
    ]

    response = schemas.CompareResponseData(summary=summary, records=records)
    return build_response(response.dict(), "批量对比完成")


# ==================== 系统接口 ====================

@app.get("/api/system/info", response_model=schemas.ApiResponse)
def get_system_info():
    system = get_predictor_system()
    data_manager = get_data_manager()
    cache_manager = get_cache_manager()
    stats = data_manager.get_stats() or {}
    info = schemas.SystemInfoResponse(
        enhanced_available=bool(system.enhanced_available),
        total_periods=stats.get('total_periods', 0),
        date_range=stats.get('date_range', {}),
        cache_dir=str(cache_manager.cache_dir),
        models_dir=str(getattr(cm, 'MODELS_DIR', cache_manager.cache_dir)),
        logs_dir=str(getattr(cm, 'LOGS_DIR', cache_manager.cache_dir)),
    )
    return build_response(info.dict())


@app.get("/api/system/health", response_model=schemas.ApiResponse)
def health_check():
    data_manager = get_data_manager()
    cache_manager = get_cache_manager()
    system = get_predictor_system()

    dataset = data_manager.get_data()
    data_ok = dataset is not None and len(dataset) > 0
    cache_ok = cache_manager is not None
    predictor_ok = system is not None

    components = {
        'data_loaded': data_ok,
        'cache_ready': cache_ok,
        'predictor_ready': predictor_ok,
    }
    status_text = 'ok' if all(components.values()) else 'degraded'
    response = schemas.HealthStatusResponse(status=status_text, components=components)
    return build_response(response.dict())


@app.exception_handler(HTTPException)
async def http_exception_handler(_, exc: HTTPException):
    payload = schemas.ApiResponse(success=False, data=None, message=str(exc.detail))
    return JSONResponse(status_code=exc.status_code, content=payload.dict())


@app.exception_handler(Exception)
async def generic_exception_handler(_, exc: Exception):
    logger.exception("服务器异常: %s", exc)
    payload = schemas.ApiResponse(success=False, data=None, message="服务器内部错误")
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=payload.dict())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.api.server:app", host="0.0.0.0", port=8000, reload=False)
