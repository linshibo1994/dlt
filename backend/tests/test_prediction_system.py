#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""大乐透测试系统命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.testing import DltDataSource, DltRule, PredictionRunner, SessionConfig, TestEngine
from backend.api import schemas as api_schemas


def parse_range(raw: str) -> tuple[int, int]:
    if ":" not in raw:
        raise argparse.ArgumentTypeError("范围参数必须是 start:end 格式")
    start, end = raw.split(":", 1)
    try:
        start_i = int(start)
        end_i = int(end)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("范围参数必须是整数") from exc
    if start_i > end_i:
        raise argparse.ArgumentTypeError("范围起点不能大于终点")
    return start_i, end_i


def resolve_methods(method_arg: str, methods_arg: str) -> List[str]:
    available = sorted(set(api_schemas.AlgorithmLiteral.__args__))

    if methods_arg:
        methods = [item.strip() for item in methods_arg.split(",") if item.strip()]
    elif method_arg == "all":
        methods = available
    else:
        methods = [item.strip() for item in method_arg.split(",") if item.strip()]

    invalid = [m for m in methods if m not in available]
    if invalid:
        raise ValueError(f"不支持的方法: {', '.join(invalid)}")

    return sorted(set(methods))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="大乐透预测方法测试系统")
    parser.add_argument("--method", default="markov", help="单方法/逗号分隔/all")
    parser.add_argument("--methods", default="", help="显式多方法（优先于 --method）")
    parser.add_argument("--strategy", choices=["progressive", "random"], default="random")
    parser.add_argument("--target-prize", default="六等奖", help="目标奖级，例如 六等奖")
    parser.add_argument("--periods-range", type=parse_range, default=(50, 500), help="分析期数范围 start:end")
    parser.add_argument("--count-range", type=parse_range, default=(1, 1), help="注数范围 start:end")
    parser.add_argument("--max-tests", type=int, default=20, help="每方法最大测试次数")
    parser.add_argument("--parallel", action="store_true", help="多方法并行")
    parser.add_argument("--workers", type=int, default=4, help="并行线程数")
    parser.add_argument("--seed", type=int, default=None, help="随机种子（保证参数抽样可复现）")
    parser.add_argument("--target-issue", default=None, help="指定评估目标期号，默认最新期")
    parser.add_argument("--timeout", type=int, default=120, help="单次预测超时秒数")
    parser.add_argument("--retries", type=int, default=1, help="预测失败重试次数")
    parser.add_argument("--results-dir", default="test_results", help="报告输出目录")
    parser.add_argument("--data-file", default=str(PROJECT_ROOT / "data" / "dlt_data_all.csv"), help="开奖数据 CSV 路径")
    parser.add_argument("--progressive-step", type=int, default=50, help="渐进策略期数步长")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        methods = resolve_methods(args.method, args.methods)
    except ValueError as exc:
        print(f"错误: {exc}")
        return 1

    rule = DltRule()
    if args.target_prize not in rule.list_target_prizes():
        print(f"错误: 不支持的目标奖级 {args.target_prize}")
        print(f"可选奖级: {', '.join(rule.list_target_prizes())}")
        return 1

    runner = PredictionRunner(timeout_seconds=args.timeout, retries=max(0, args.retries))
    data_source = DltDataSource(args.data_file)

    engine = TestEngine(
        rule=rule,
        runner=runner,
        data_source=data_source,
        results_dir=args.results_dir,
        seed=args.seed,
    )

    def on_event(event_type: str, payload: dict):
        if event_type == "progress":
            print(
                f"[PROGRESS] {payload.get('percent', 0):6.2f}% "
                f"({payload.get('current', 0)}/{payload.get('total', 0)}) 方法={payload.get('method', '')}"
            )
        elif event_type == "winning":
            print(
                f"[WIN] {payload.get('method')} p={payload.get('periods')} c={payload.get('count')} "
                f"=> {payload.get('prize_name')} [{payload.get('match_combination')}]"
            )
        elif event_type == "log":
            print(f"[LOG] {payload.get('message')}")
        elif event_type == "error":
            print(f"[ERROR] {payload.get('message')}")

    engine.set_event_callback(on_event)

    cfg = SessionConfig(
        methods=methods,
        strategy=args.strategy,
        target_prize=args.target_prize,
        periods_start=args.periods_range[0],
        periods_end=args.periods_range[1],
        count_start=args.count_range[0],
        count_end=args.count_range[1],
        max_tests=max(1, args.max_tests),
        parallel=args.parallel,
        workers=max(1, args.workers),
        progressive_step=max(1, args.progressive_step),
        target_issue=args.target_issue,
    )

    try:
        summary = engine.run_session(cfg)
    except Exception as exc:
        print(f"执行失败: {exc}")
        return 1

    print("\n=== 测试完成 ===")
    print(f"session_id: {summary['session_id']}")
    print(f"total_tests: {summary['total_tests']}")
    print(f"winning_tests: {summary['winning_tests']}")
    print(f"winning_rate: {summary['winning_rate']:.2%}")
    print(f"execution_time: {summary.get('execution_time', 0):.2f}s")
    print(f"best_methods: {summary.get('best_methods', [])}")
    print(f"report_files: {summary.get('report_files', {})}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
