"""概率基线预测与滚动评估的轻量终端入口。"""

from __future__ import annotations

import argparse
import json
import sys

from .prediction import BaselinePredictor, PredictionConfig
from .walk_forward import (
    SUPPORTED_METHODS,
    EvaluationConfig,
    WalkForwardEvaluator,
)


def _positive_int(value):
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("必须为正整数") from None
    if number <= 0:
        raise argparse.ArgumentTypeError("必须为正整数")
    return number


def _ticket_count(value):
    number = _positive_int(value)
    if number > 100:
        raise argparse.ArgumentTypeError("必须为 1..100 的正整数")
    return number


def _methods(value):
    methods = [method.strip() for method in value.split(",")]
    if not methods or any(not method for method in methods):
        raise argparse.ArgumentTypeError("methods 不能为空")

    invalid = [method for method in methods if method not in SUPPORTED_METHODS]
    if invalid:
        raise argparse.ArgumentTypeError(
            "methods 仅支持 uniform、dirichlet"
        )
    return tuple(dict.fromkeys(methods))


def _add_shared_arguments(parser):
    parser.add_argument("-p", "--periods", type=_positive_int, default=500, help="训练期数")
    parser.add_argument("-c", "--count", type=_ticket_count, default=5, help="每期票数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--alpha", type=float, default=1.0, help="狄利克雷平滑参数")
    parser.add_argument(
        "--json-output",
        action="store_true",
        help="输出单个 JSON 对象",
    )


def build_parser():
    """构建概率基线终端参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="evaluate",
        description="概率基线下一期预测与无泄漏滚动评估",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict_parser = subparsers.add_parser("predict", help="生成下一期基线号码")
    predict_parser.add_argument(
        "-m",
        "--method",
        choices=SUPPORTED_METHODS,
        default="dirichlet",
        help="概率基线方法",
    )
    _add_shared_arguments(predict_parser)

    walk_parser = subparsers.add_parser(
        "walk-forward",
        help="执行无泄漏滚动评估",
    )
    walk_parser.add_argument(
        "--methods",
        type=_methods,
        default=SUPPORTED_METHODS,
        help="逗号分隔的方法列表",
    )
    walk_parser.add_argument("--draws", type=_positive_int, default=30, help="评估期数")
    _add_shared_arguments(walk_parser)
    return parser


def _format_ticket(numbers):
    return " ".join(f"{number:02d}" for number in numbers)


def _print_prediction(result):
    data = result["data"]
    print("概率基线下一期预测")
    print(f"方法：{result['method']}")
    print(f"最新期号：{data['latest_issue']}")
    print(
        "训练窗口："
        f"{data['training_newest_issue']} 至 {data['training_oldest_issue']}"
        f"（{data['training_periods']} 期）"
    )
    for index, ticket in enumerate(result["tickets"], start=1):
        front = _format_ticket(ticket["front_balls"])
        back = _format_ticket(ticket["back_balls"])
        print(f"第 {index} 注：前区 {front} + 后区 {back}")
    print(f"免责声明：{result['disclaimer']}")


def _print_walk_forward(result):
    data = result["data"]
    print("概率基线无泄漏滚动评估")
    print(f"最新期号：{data['latest_issue']}")
    print(f"评估期数：{data['evaluated_draws']}")

    for method, summary in result["methods"].items():
        distribution = "，".join(
            f"{combination}={count}"
            for combination, count in summary["match_distribution"].items()
        )
        print(f"方法：{method}")
        print(
            f"平均前区命中：{summary['average_front_matches']:.6f}；"
            f"平均后区命中：{summary['average_back_matches']:.6f}"
        )
        print(f"命中分布：{distribution or '无'}")

    first_method = next(iter(result["methods"].values()))
    print("窗口审计：")
    for detail in first_method["draw_details"]:
        print(
            f"目标期 {detail['target_issue']}："
            f"训练 {detail['training_newest_issue']} 至 "
            f"{detail['training_oldest_issue']}（{detail['training_count']} 期）"
        )
    print(f"免责声明：{result['disclaimer']}")


def main(argv=None) -> int:
    """执行概率基线终端命令并返回进程退出码。"""
    args = build_parser().parse_args(argv)

    try:
        if args.command == "predict":
            result = BaselinePredictor().predict(
                PredictionConfig(
                    method=args.method,
                    periods=args.periods,
                    count=args.count,
                    seed=args.seed,
                    alpha=args.alpha,
                )
            )
        else:
            result = WalkForwardEvaluator().run(
                EvaluationConfig(
                    methods=args.methods,
                    draws=args.draws,
                    periods=args.periods,
                    count=args.count,
                    seed=args.seed,
                    alpha=args.alpha,
                )
            )
    except (FileNotFoundError, ValueError) as exc:
        print(f"评估失败: {exc}", file=sys.stderr)
        return 2

    if args.json_output:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    elif args.command == "predict":
        _print_prediction(result)
    else:
        _print_walk_forward(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
