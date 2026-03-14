#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FastAPI 服务器 - 对外暴露预测与分析接口"""

from __future__ import annotations

import logging
import os
import json
import queue
import threading
import requests
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from backend.app.core import core_modules as cm
from backend.app.utils.crawlers import update_data as crawler_update_data, incremental_update_data
from backend.testing import DltDataSource, DltRule, PredictionRunner, SessionConfig, TestEngine

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


def _build_testing_engine(request: schemas.TestingRequest, event_callback=None) -> tuple[TestEngine, SessionConfig]:
    runner = PredictionRunner(
        timeout_seconds=request.timeout_seconds,
        retries=request.retries,
        prefer_direct=True,
        fallback_subprocess=True,
    )
    engine = TestEngine(
        rule=DltRule(),
        runner=runner,
        data_source=DltDataSource(),
        results_dir="test_results",
        event_callback=event_callback,
        seed=request.seed,
    )
    cfg = SessionConfig(
        methods=request.methods,
        strategy=request.strategy,
        target_prize=request.target_prize,
        periods_start=request.periods_start,
        periods_end=request.periods_end,
        count_start=request.count_start,
        count_end=request.count_end,
        max_tests=request.max_tests,
        parallel=request.parallel,
        workers=request.workers,
        progressive_step=request.progressive_step,
        target_issue=request.target_issue,
    )
    return engine, cfg


# ==================== 数据接口 ====================

def fetch_prize_info(issue: str) -> Optional[List[Dict[str, Any]]]:
    """从 sporttery.cn 获取奖项信息"""
    try:
        url = (
            "https://webapi.sporttery.cn/gateway/lottery/getHistoryPageListV1.qry?"
            "gameNo=85&provinceId=0&pageSize=1&isVerify=1&pageNo=1"
        )
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()

        if data.get('value') and data['value'].get('list'):
            for item in data['value']['list']:
                if str(item.get('lotteryDrawNum')) == issue:
                    prize_list = item.get('prizeLevelList', [])
                    return [
                        {
                            'level': p.get('prizeLevelName'),
                            'count': p.get('prizeNum'),
                            'amount': p.get('prizeAmountStr'),
                        }
                        for p in prize_list
                    ]
        return None
    except Exception as exc:
        logger.warning(f"获取奖项信息失败 (期号: {issue}): {exc}")
        return None


@app.get("/api/data/latest", response_model=schemas.ApiResponse)
def get_latest_data(data_manager=Depends(get_data_manager)):
    df = data_manager.get_data()
    if df is None or len(df) == 0:
        raise_http_error("暂未加载到开奖数据", status.HTTP_503_SERVICE_UNAVAILABLE)

    latest = df.iloc[0]
    front, back = data_manager.parse_balls(latest)
    issue = str(latest.get('issue'))

    record = schemas.DataRecord(
        issue=issue,
        date=str(latest.get('date')),
        front_balls=[int(x) for x in front],
        back_balls=[int(x) for x in back],
        prize_grades=fetch_prize_info(issue),
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

    # 转换 numpy 类型为 Python 原生类型
    total_periods = stats.get('total_periods', 0)
    if hasattr(total_periods, 'item'):
        total_periods = total_periods.item()

    latest_issue = stats.get('latest_issue')
    if latest_issue is not None and hasattr(latest_issue, 'item'):
        latest_issue = latest_issue.item()

    date_range = stats.get('date_range', {})
    # 处理 date_range 中可能的 numpy 类型
    if isinstance(date_range, dict):
        date_range = {k: (v.item() if hasattr(v, 'item') else v) for k, v in date_range.items()}

    # 处理缓存信息（'total' 是一个字典结构）
    cache_total = cache_info.get('total', {})
    if isinstance(cache_total, dict):
        cache_files = cache_total.get('files', 0)
        cache_size_mb = cache_total.get('size_mb', 0.0)
    else:
        cache_files = 0
        cache_size_mb = 0.0

    response = schemas.DataStatsResponse(
        total_periods=int(total_periods) if total_periods else 0,
        date_range=date_range,
        latest_issue=str(latest_issue) if latest_issue else None,
        cache={
            'cache_dir': cache_info.get('cache_dir'),
            'total': int(cache_files) if cache_files else 0,
            'size_mb': float(cache_size_mb) if cache_size_mb else 0.0,
        },
    )
    return build_response(response.dict(), "统计信息获取成功")


@app.post("/api/data/update", response_model=schemas.ApiResponse)
def update_lottery_data(data_manager=Depends(get_data_manager)):
    """更新彩票数据（从官网爬取最新数据）"""
    try:
        # 调用增量更新（只爬取最新数据）
        updated_count = incremental_update_data(source="zhcw")

        # 重新加载数据
        data_manager.reload_data()

        # 获取更新后的统计信息
        df = data_manager.get_data()
        latest_issue = None
        latest_date = None
        total_periods = 0

        if df is not None and len(df) > 0:
            total_periods = len(df)
            latest = df.iloc[0]
            latest_issue = str(latest.get('issue', ''))
            latest_date = str(latest.get('date', ''))

        response = schemas.DataUpdateResponse(
            updated_count=updated_count,
            total_periods=total_periods,
            latest_issue=latest_issue,
            latest_date=latest_date,
        )

        message = f"数据更新成功，新增 {updated_count} 期数据" if updated_count > 0 else "数据已是最新，无需更新"
        return build_response(response.dict(), message)

    except Exception as exc:
        logger.exception("数据更新失败: %s", exc)
        raise_http_error(f"数据更新失败: {str(exc)}", status.HTTP_500_INTERNAL_SERVER_ERROR)


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


@app.post("/api/compare/stream")
def run_comparison_stream(request: schemas.CompareRequest, manager=Depends(get_batch_comparison)):
    """SSE 流式批量对比，实时推送进度"""
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
        def error_stream():
            payload = json.dumps({"type": "error", "message": error_msg}, ensure_ascii=False)
            yield f"data: {payload}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    def event_stream():
        q = queue.Queue()

        def progress_callback(current, total, message):
            event = {"type": "progress", "current": current, "total": total, "message": message}
            q.put(json.dumps(event, ensure_ascii=False))

        def run_comparison():
            try:
                result = manager.execute(config, progress_callback=progress_callback)
                # 构建最终结果
                summary_data = result.get_summary()
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
                resp = schemas.CompareResponseData(summary=summary, records=records)
                final = {"type": "complete", "data": resp.dict()}
                q.put(json.dumps(final, ensure_ascii=False))
            except Exception as exc:
                logger.exception("批量对比执行失败: %s", exc)
                q.put(json.dumps({"type": "error", "message": str(exc)}, ensure_ascii=False))
            finally:
                q.put(None)  # 结束信号

        thread = threading.Thread(target=run_comparison, daemon=True)
        thread.start()

        while True:
            try:
                event = q.get(timeout=120)
                if event is None:
                    break
                yield f"data: {event}\n\n"
            except queue.Empty:
                # 发送心跳保持连接
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ==================== 测试系统接口 ====================

@app.get("/api/testing/options", response_model=schemas.ApiResponse)
def get_testing_options():
    methods = [
        {"id": item["id"], "name": item["name"], "category": item["category"]}
        for item in ALGORITHM_DEFINITIONS
    ]
    target_prizes = DltRule().list_target_prizes()
    return build_response({
        "methods": methods,
        "target_prizes": target_prizes,
    })


@app.post("/api/testing/run", response_model=schemas.ApiResponse)
def run_testing(request: schemas.TestingRequest):
    try:
        engine, cfg = _build_testing_engine(request)
        summary = engine.run_session(cfg)
        summary["tested_methods"] = cfg.methods
        summary["successful_methods"] = [
            item["method"] for item in summary.get("method_outcomes", []) if item.get("hit_target")
        ]
        return build_response(summary, "测试执行完成")
    except ValueError as exc:
        raise_http_error(str(exc))
    except Exception as exc:
        logger.exception("测试系统执行失败: %s", exc)
        raise_http_error("测试系统执行失败，请检查日志", status.HTTP_500_INTERNAL_SERVER_ERROR)


@app.get("/api/testing/stream")
def run_testing_stream(
    methods: str = Query(..., description="逗号分隔的方法列表"),
    strategy: schemas.TestingStrategyLiteral = Query("random"),
    target_prize: schemas.TestingPrizeLiteral = Query("六等奖"),
    periods_start: int = Query(50, ge=10, le=2748),
    periods_end: int = Query(500, ge=10, le=2748),
    count_start: int = Query(1, ge=1, le=100),
    count_end: int = Query(1, ge=1, le=100),
    max_tests: int = Query(20, ge=1, le=5000),
    parallel: bool = Query(False),
    workers: int = Query(4, ge=1, le=64),
    seed: Optional[int] = Query(None),
    target_issue: Optional[str] = Query(None),
    progressive_step: int = Query(50, ge=1, le=1000),
    timeout_seconds: int = Query(120, ge=10, le=3600),
    retries: int = Query(1, ge=0, le=5),
):
    try:
        method_list = [item.strip() for item in methods.split(",") if item.strip()]
        request = schemas.TestingRequest(
            methods=method_list,
            strategy=strategy,
            target_prize=target_prize,
            periods_start=periods_start,
            periods_end=periods_end,
            count_start=count_start,
            count_end=count_end,
            max_tests=max_tests,
            parallel=parallel,
            workers=workers,
            seed=seed,
            target_issue=target_issue,
            progressive_step=progressive_step,
            timeout_seconds=timeout_seconds,
            retries=retries,
        )
    except Exception as exc:
        def error_stream():
            payload = json.dumps({"message": str(exc)}, ensure_ascii=False)
            yield f"event: error_event\ndata: {payload}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    def event_stream():
        q = queue.Queue()

        def callback(event_type: str, payload: Dict[str, Any]):
            if event_type == "complete":
                payload = dict(payload)
                payload["tested_methods"] = request.methods
                payload["successful_methods"] = [
                    item["method"] for item in payload.get("method_outcomes", []) if item.get("hit_target")
                ]
            q.put({"type": event_type, "data": payload})

        def run_task():
            try:
                engine, cfg = _build_testing_engine(request, event_callback=callback)
                engine.run_session(cfg)
            except Exception as exc:
                logger.exception("测试系统流式执行失败: %s", exc)
                q.put({"type": "error", "data": {"message": str(exc)}})
            finally:
                q.put(None)

        thread = threading.Thread(target=run_task, daemon=True)
        thread.start()

        while True:
            try:
                event = q.get(timeout=120)
                if event is None:
                    break
                event_type = event.get("type", "message")
                event_data = json.dumps(event.get("data", {}), ensure_ascii=False)
                yield f"event: {event_type}\ndata: {event_data}\n\n"
            except queue.Empty:
                yield f"event: log\ndata: {json.dumps({'message': 'heartbeat', 'level': 'debug'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


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

    uvicorn.run("backend.api.server:app", host="0.0.0.0", port=6000, reload=False)
