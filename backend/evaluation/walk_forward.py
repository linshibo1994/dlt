"""独立、可复现的无泄漏滚动评估器。"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from random import Random
from typing import Dict, Tuple

from backend.testing import DltDataSource

from .baselines import DirichletBaseline, UniformBaseline, parse_numbers


SUPPORTED_METHODS = ("uniform", "dirichlet")


@dataclass(frozen=True)
class EvaluationConfig:
    """滚动评估配置。"""

    methods: Tuple[str, ...] = ("uniform", "dirichlet")
    draws: int = 30
    periods: int = 500
    count: int = 5
    seed: int = 42
    alpha: float = 1.0


@dataclass(frozen=True)
class EvaluationCase:
    """一个目标期及其严格位于目标期之前的训练窗口。"""

    target: Dict[str, str]
    training: Tuple[Dict[str, str], ...]


def derive_seed(seed, method, issue):
    """派生单个方法和目标期的独立随机种子。"""
    payload = f"{seed}:{method}:{issue}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def match_ticket(ticket, target):
    """计算单张票的前区和后区命中数。"""
    ticket_front = set(parse_numbers(ticket["front_balls"], 5, 1, 35))
    ticket_back = set(parse_numbers(ticket["back_balls"], 2, 1, 12))
    target_front = set(parse_numbers(target["front_balls"], 5, 1, 35))
    target_back = set(parse_numbers(target["back_balls"], 2, 1, 12))
    return len(ticket_front & target_front), len(ticket_back & target_back)


def _sorted_distribution(distribution):
    """按前区、后区命中数稳定排序命中组合。"""
    return {
        combination: distribution[combination]
        for combination in sorted(
            distribution,
            key=lambda value: tuple(int(item) for item in value.split("+")),
        )
    }


class WalkForwardEvaluator:
    """使用显式历史切片运行概率基线评估。"""

    def __init__(self, data_source=None):
        self.data_source = data_source or DltDataSource()

    @staticmethod
    def _validate_config(config: EvaluationConfig):
        methods = config.methods
        if isinstance(methods, (str, bytes)) or not isinstance(methods, Sequence):
            raise ValueError("methods 必须为非空的方法序列")
        if not methods:
            raise ValueError("methods 不能为空")

        invalid_methods = [method for method in methods if method not in SUPPORTED_METHODS]
        if invalid_methods:
            invalid_text = "、".join(str(method) for method in invalid_methods)
            raise ValueError(
                f"methods 仅支持 uniform、dirichlet，非法方法：{invalid_text}"
            )
        normalized_methods = tuple(dict.fromkeys(methods))

        for field_name in ("draws", "periods"):
            value = getattr(config, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} 必须为正整数")

        if (
            isinstance(config.count, bool)
            or not isinstance(config.count, int)
            or not 1 <= config.count <= 100
        ):
            raise ValueError("count 必须为 1..100 的正整数")
        if isinstance(config.seed, bool) or not isinstance(config.seed, int):
            raise ValueError("seed 必须为整数")

        normalized_alpha = DirichletBaseline(config.alpha).alpha
        return normalized_methods, normalized_alpha

    def build_cases(self, draws, config: EvaluationConfig):
        """按倒序开奖记录构造严格无泄漏的评估案例。"""
        self._validate_config(config)
        required = config.draws + config.periods
        if len(draws) < required:
            raise ValueError(
                f"数据不足：至少需要 {required} 期，当前只有 {len(draws)} 期"
            )
        return [
            EvaluationCase(
                target=draws[index],
                training=tuple(
                    draws[index + 1 : index + 1 + config.periods]
                ),
            )
            for index in range(config.draws)
        ]

    def run(self, config: EvaluationConfig):
        """运行滚动评估并返回可复现、可审计的结构化结果。"""
        normalized_methods, normalized_alpha = self._validate_config(config)
        draws = self.data_source.load_all()
        cases = self.build_cases(draws, config)
        method_summaries = {}
        raw_averages = {}

        for method in normalized_methods:
            baseline = (
                UniformBaseline()
                if method == "uniform"
                else DirichletBaseline(alpha=normalized_alpha)
            )
            front_matches_total = 0
            back_matches_total = 0
            jackpot_matches = 0
            method_distribution = Counter()
            draw_details = []

            for case in cases:
                target_issue = case.target["issue"]
                rng = Random(derive_seed(config.seed, method, target_issue))
                generated_tickets = baseline.generate(
                    case.training,
                    count=config.count,
                    rng=rng,
                )
                draw_distribution = Counter()
                ticket_details = []

                for ticket in generated_tickets:
                    front_balls = parse_numbers(ticket["front_balls"], 5, 1, 35)
                    back_balls = parse_numbers(ticket["back_balls"], 2, 1, 12)
                    front_hits, back_hits = match_ticket(ticket, case.target)
                    combination = f"{front_hits}+{back_hits}"

                    front_matches_total += front_hits
                    back_matches_total += back_hits
                    method_distribution[combination] += 1
                    draw_distribution[combination] += 1
                    if combination == "5+2":
                        jackpot_matches += 1

                    ticket_details.append(
                        {
                            "front_balls": front_balls,
                            "back_balls": back_balls,
                            "front_matches": front_hits,
                            "back_matches": back_hits,
                            "match_combination": combination,
                        }
                    )

                draw_details.append(
                    {
                        "target_issue": target_issue,
                        "target_date": case.target["date"],
                        "training_newest_issue": case.training[0]["issue"],
                        "training_oldest_issue": case.training[-1]["issue"],
                        "training_count": len(case.training),
                        "match_distribution": _sorted_distribution(
                            draw_distribution
                        ),
                        "tickets": ticket_details,
                    }
                )

            ticket_count = sum(method_distribution.values())
            average_front_matches = front_matches_total / ticket_count
            average_back_matches = back_matches_total / ticket_count
            raw_averages[method] = (
                average_front_matches,
                average_back_matches,
            )
            method_summaries[method] = {
                "evaluated_draws": len(cases),
                "ticket_count": ticket_count,
                "average_front_matches": round(average_front_matches, 6),
                "average_back_matches": round(average_back_matches, 6),
                "match_distribution": _sorted_distribution(method_distribution),
                "jackpot_matches": jackpot_matches,
                "draw_details": draw_details,
            }

        uniform_summary = method_summaries.get("uniform")
        if uniform_summary is not None:
            for method in normalized_methods:
                if method == "uniform":
                    continue
                summary = method_summaries[method]
                method_front_average, method_back_average = raw_averages[method]
                uniform_front_average, uniform_back_average = raw_averages["uniform"]
                summary["vs_uniform"] = {
                    "average_front_matches_delta": round(
                        method_front_average - uniform_front_average,
                        6,
                    ),
                    "average_back_matches_delta": round(
                        method_back_average - uniform_back_average,
                        6,
                    ),
                }

        return {
            "config": {
                "methods": list(normalized_methods),
                "draws": config.draws,
                "periods": config.periods,
                "count": config.count,
                "seed": config.seed,
                "alpha": normalized_alpha,
            },
            "data": {
                "latest_issue": draws[0]["issue"],
                "available_draws": len(draws),
                "evaluated_draws": len(cases),
            },
            "methods": method_summaries,
            "disclaimer": "仅用于历史比较，不代表未来中奖概率",
        }
