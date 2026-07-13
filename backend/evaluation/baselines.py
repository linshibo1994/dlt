"""可复现的大乐透概率基线算法。"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping
from numbers import Integral
from typing import Any, Callable, Dict, Iterable, List, Sequence


FRONT_POOL_SIZE = 35
FRONT_BALLS_PER_DRAW = 5
BACK_POOL_SIZE = 12
BACK_BALLS_PER_DRAW = 2


def _area_name(expected: int, minimum: int, maximum: int) -> str:
    if (expected, minimum, maximum) == (FRONT_BALLS_PER_DRAW, 1, FRONT_POOL_SIZE):
        return "前区"
    if (expected, minimum, maximum) == (BACK_BALLS_PER_DRAW, 1, BACK_POOL_SIZE):
        return "后区"
    return "历史"


def parse_numbers(
    value: Any,
    expected: int,
    minimum: int,
    maximum: int,
) -> List[int]:
    """解析并校验一组历史号码，返回升序整数列表。"""
    area = _area_name(expected, minimum, maximum)
    if isinstance(value, str):
        raw_numbers = [item for item in re.split(r"[\s,]+", value.strip()) if item]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        raw_numbers = list(value)
    else:
        raise ValueError(f"{area}号码必须包含 {expected} 个唯一整数")

    numbers: List[int] = []
    try:
        for item in raw_numbers:
            if isinstance(item, str):
                normalized_item = item.strip()
                if not re.fullmatch(r"[+-]?\d+", normalized_item):
                    raise ValueError
                numbers.append(int(normalized_item))
            elif isinstance(item, bool) or not isinstance(item, Integral):
                raise ValueError
            else:
                numbers.append(int(item))
    except (TypeError, ValueError):
        raise ValueError(f"{area}号码必须包含 {expected} 个唯一整数") from None

    if len(numbers) != expected or len(set(numbers)) != expected:
        raise ValueError(f"{area}号码必须包含 {expected} 个唯一整数")
    if any(number < minimum or number > maximum for number in numbers):
        raise ValueError(f"{area}号码范围必须为 {minimum}..{maximum}")
    return sorted(numbers)


def _validate_training_draws(training_draws: Iterable[Mapping[str, Any]]) -> List[Dict[str, List[int]]]:
    if training_draws is None:
        raise ValueError("训练数据不能为空")
    try:
        rows = list(training_draws)
    except TypeError:
        raise ValueError("训练数据必须为可迭代的历史开奖记录") from None
    if not rows:
        raise ValueError("训练数据不能为空")

    normalized: List[Dict[str, List[int]]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise ValueError(f"第 {index} 行训练数据必须为开奖记录")
        if "front_balls" not in row or "back_balls" not in row:
            raise ValueError(f"第 {index} 行训练数据缺少前区或后区号码")
        try:
            front = parse_numbers(row["front_balls"], FRONT_BALLS_PER_DRAW, 1, FRONT_POOL_SIZE)
            back = parse_numbers(row["back_balls"], BACK_BALLS_PER_DRAW, 1, BACK_POOL_SIZE)
        except ValueError as exc:
            raise ValueError(f"第 {index} 行训练数据非法：{exc}") from exc
        normalized.append({"front_balls": front, "back_balls": back})
    return normalized


def _validate_count(count: int) -> None:
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("count 必须为正整数")


def _generate_unique_tickets(
    count: int,
    candidate_factory: Callable[[], Dict[str, List[int]]],
) -> List[Dict[str, List[int]]]:
    _validate_count(count)
    tickets: List[Dict[str, List[int]]] = []
    seen = set()
    max_attempts = max(1000, count * 100)

    for _ in range(max_attempts):
        candidate = candidate_factory()
        key = (tuple(candidate["front_balls"]), tuple(candidate["back_balls"]))
        if key in seen:
            continue
        seen.add(key)
        tickets.append(candidate)
        if len(tickets) == count:
            return tickets

    raise ValueError(f"在 {max_attempts} 次尝试内无法生成 {count} 张互不重复的票")


def _smoothed_probabilities(
    rows: List[Dict[str, List[int]]],
    field: str,
    pool_size: int,
    balls_per_draw: int,
    alpha: float,
) -> Dict[int, float]:
    counts = Counter(number for row in rows for number in row[field])
    denominator = len(rows) * balls_per_draw + alpha * pool_size
    return {
        number: (counts[number] + alpha) / denominator
        for number in range(1, pool_size + 1)
    }


def _weighted_sample_without_replacement(
    probabilities: Mapping[int, float],
    count: int,
    rng: Any,
) -> List[int]:
    numbers = list(probabilities)
    weights = [probabilities[number] for number in numbers]
    selected: List[int] = []

    for _ in range(count):
        threshold = rng.random() * sum(weights)
        cumulative = 0.0
        selected_index = len(numbers) - 1
        for index, weight in enumerate(weights):
            cumulative += weight
            if threshold < cumulative:
                selected_index = index
                break
        selected.append(numbers.pop(selected_index))
        weights.pop(selected_index)

    return sorted(selected)


class UniformBaseline:
    """前后区均采用均匀随机不放回抽样。"""

    name = "uniform"

    def generate(self, training_draws, count, rng):
        _validate_training_draws(training_draws)
        _validate_count(count)
        return _generate_unique_tickets(
            count,
            lambda: {
                "front_balls": sorted(rng.sample(range(1, FRONT_POOL_SIZE + 1), FRONT_BALLS_PER_DRAW)),
                "back_balls": sorted(rng.sample(range(1, BACK_POOL_SIZE + 1), BACK_BALLS_PER_DRAW)),
            },
        )


class DirichletBaseline:
    """基于历史边际频率和 Dirichlet 平滑的概率基线。"""

    name = "dirichlet"

    def __init__(self, alpha=1.0):
        try:
            normalized_alpha = float(alpha)
        except (TypeError, ValueError):
            raise ValueError("alpha 必须大于 0") from None
        if not math.isfinite(normalized_alpha) or normalized_alpha <= 0:
            raise ValueError("alpha 必须大于 0")
        self.alpha = normalized_alpha

    def probabilities(self, training_draws):
        rows = _validate_training_draws(training_draws)
        return {
            "front": _smoothed_probabilities(
                rows,
                "front_balls",
                FRONT_POOL_SIZE,
                FRONT_BALLS_PER_DRAW,
                self.alpha,
            ),
            "back": _smoothed_probabilities(
                rows,
                "back_balls",
                BACK_POOL_SIZE,
                BACK_BALLS_PER_DRAW,
                self.alpha,
            ),
        }

    def generate(self, training_draws, count, rng):
        _validate_count(count)
        probabilities = self.probabilities(training_draws)
        return _generate_unique_tickets(
            count,
            lambda: {
                "front_balls": _weighted_sample_without_replacement(
                    probabilities["front"], FRONT_BALLS_PER_DRAW, rng
                ),
                "back_balls": _weighted_sample_without_replacement(
                    probabilities["back"], BACK_BALLS_PER_DRAW, rng
                ),
            },
        )
