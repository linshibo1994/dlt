"""概率基线算法测试。"""

import math
import re
from random import Random

import pytest

from backend.evaluation import DirichletBaseline, UniformBaseline, parse_numbers
from backend.evaluation.baselines import (
    _generate_unique_tickets,
    _weighted_sample_without_replacement,
)


TRAINING_DRAWS = [
    {
        "issue": "1003",
        "date": "2026-01-03",
        "front_balls": "01,02,03,04,05",
        "back_balls": "01,02",
    },
    {
        "issue": "1002",
        "date": "2026-01-02",
        "front_balls": "01,06,07,08,09",
        "back_balls": "01,03",
    },
]


def assert_valid_ticket(ticket):
    """断言单张票符合大乐透号码规则。"""
    front = ticket["front_balls"]
    back = ticket["back_balls"]

    assert len(front) == 5
    assert len(set(front)) == 5
    assert front == sorted(front)
    assert all(isinstance(value, int) and 1 <= value <= 35 for value in front)

    assert len(back) == 2
    assert len(set(back)) == 2
    assert back == sorted(back)
    assert all(isinstance(value, int) and 1 <= value <= 12 for value in back)


@pytest.mark.parametrize(
    "baseline",
    [UniformBaseline(), DirichletBaseline(alpha=1.0)],
    ids=["uniform", "dirichlet"],
)
def test_baseline_generates_unique_reproducible_valid_tickets(baseline):
    first = baseline.generate(TRAINING_DRAWS, count=5, rng=Random(42))
    second = baseline.generate(TRAINING_DRAWS, count=5, rng=Random(42))

    assert first == second
    assert len(first) == 5
    ticket_keys = {
        (tuple(ticket["front_balls"]), tuple(ticket["back_balls"]))
        for ticket in first
    }
    assert len(ticket_keys) == 5
    for ticket in first:
        assert_valid_ticket(ticket)


def test_baseline_names_and_parse_numbers_are_public():
    assert UniformBaseline.name == "uniform"
    assert DirichletBaseline.name == "dirichlet"
    assert parse_numbers("01, 02,03", expected=3, minimum=1, maximum=35) == [1, 2, 3]
    assert parse_numbers([1, 2, 3], expected=3, minimum=1, maximum=35) == [1, 2, 3]


@pytest.mark.parametrize(
    "invalid_front",
    [
        [1, 2, 3, 4, 35.9],
        [1, 2, 3, 4, 35.0],
        "01,02,03,04,35.0",
        [1, 2, 3, 4, True],
        [1, 2, 3, 4, None],
    ],
    ids=["fractional-float", "integral-float", "decimal-string", "bool", "none"],
)
def test_baseline_rejects_non_integer_historical_numbers(invalid_front):
    invalid_draw = dict(TRAINING_DRAWS[0], front_balls=invalid_front)

    with pytest.raises(ValueError, match="前区号码必须包含 5 个唯一整数"):
        UniformBaseline().generate([invalid_draw], count=1, rng=Random(1))


def test_unique_ticket_generation_raises_after_max_attempts():
    attempts = 0

    def constant_ticket_factory():
        nonlocal attempts
        attempts += 1
        return {"front_balls": [1, 2, 3, 4, 5], "back_balls": [1, 2]}

    with pytest.raises(ValueError, match="在 1000 次尝试内无法生成 2 张互不重复的票"):
        _generate_unique_tickets(count=2, candidate_factory=constant_ticket_factory)

    assert attempts == 1000


def test_dirichlet_probabilities_include_unseen_numbers_and_use_formula():
    probabilities = DirichletBaseline(alpha=1.0).probabilities(TRAINING_DRAWS)

    assert set(probabilities["front"]) == set(range(1, 36))
    assert set(probabilities["back"]) == set(range(1, 13))
    assert probabilities["front"][35] > 0
    assert probabilities["back"][12] > 0
    assert probabilities["front"][1] == pytest.approx(3 / 45)
    assert probabilities["front"][35] == pytest.approx(1 / 45)
    assert probabilities["back"][1] == pytest.approx(3 / 16)
    assert probabilities["back"][12] == pytest.approx(1 / 16)
    assert sum(probabilities["front"].values()) == pytest.approx(1.0)
    assert sum(probabilities["back"].values()) == pytest.approx(1.0)


def test_dirichlet_probabilities_remain_positive_and_finite_for_large_alpha():
    probabilities = DirichletBaseline(alpha=1e308).probabilities(TRAINING_DRAWS)

    for area in ("front", "back"):
        values = probabilities[area].values()
        assert all(math.isfinite(value) and value > 0 for value in values)
        assert sum(values) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "alpha",
    [0, -1, -0.5, float("inf"), float("nan"), 5e-324],
    ids=["zero", "negative-int", "negative-float", "inf", "nan", "underflow"],
)
def test_dirichlet_rejects_uncalculable_alpha(alpha):
    with pytest.raises(ValueError) as exc_info:
        DirichletBaseline(alpha=alpha)

    assert str(exc_info.value) == "alpha 必须为可计算的有限正数"


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ({1: -0.1, 2: 1.0}, "权重必须为有限非负数"),
        ({1: float("inf"), 2: 1.0}, "权重必须为有限非负数"),
        ({1: float("nan"), 2: 1.0}, "权重必须为有限非负数"),
        ({1: 0.0, 2: 0.0}, "权重总和必须为有限正数"),
        ({1: 1e308, 2: 1e308}, "权重总和必须为有限正数"),
    ],
    ids=["negative", "inf", "nan", "zero-total", "infinite-total"],
)
def test_weighted_sample_rejects_invalid_weights(weights, message):
    with pytest.raises(ValueError) as exc_info:
        _weighted_sample_without_replacement(weights, count=1, rng=Random(1))

    assert str(exc_info.value) == message


def test_dirichlet_generate_uses_weights_deterministically_without_replacement():
    biased_draws = [
        {"front_balls": [31, 32, 33, 34, 35], "back_balls": [11, 12]}
        for _ in range(100)
    ]

    class ScriptedRandom:
        def __init__(self):
            self.calls = 0

        def random(self):
            self.calls += 1
            return 0.23

    first_rng = ScriptedRandom()
    second_rng = ScriptedRandom()
    baseline = DirichletBaseline(alpha=1.0)

    first = baseline.generate(biased_draws, count=1, rng=first_rng)
    second = baseline.generate(biased_draws, count=1, rng=second_rng)

    assert first == second == [
        {"front_balls": [31, 32, 33, 34, 35], "back_balls": [11, 12]}
    ]
    assert len(set(first[0]["front_balls"])) == 5
    assert len(set(first[0]["back_balls"])) == 2
    assert first_rng.calls == second_rng.calls == 7


@pytest.mark.parametrize(
    "operation",
    [
        lambda: UniformBaseline().generate([], count=1, rng=Random(1)),
        lambda: DirichletBaseline().probabilities([]),
    ],
    ids=["uniform", "dirichlet"],
)
def test_baseline_rejects_empty_training_data(operation):
    with pytest.raises(ValueError, match="训练数据不能为空"):
        operation()


@pytest.mark.parametrize("baseline", [UniformBaseline(), DirichletBaseline()])
@pytest.mark.parametrize("count", [0, -1, 1.5, True])
def test_baseline_rejects_invalid_count(baseline, count):
    with pytest.raises(ValueError, match="count 必须为正整数"):
        baseline.generate(TRAINING_DRAWS, count=count, rng=Random(1))


@pytest.mark.parametrize("baseline", [UniformBaseline(), DirichletBaseline()])
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("front_balls", "01,02,03,04", "前区号码必须包含 5 个唯一整数"),
        ("front_balls", "01,02,03,04,36", "前区号码范围必须为 1..35"),
        ("back_balls", "01", "后区号码必须包含 2 个唯一整数"),
        ("back_balls", "01,13", "后区号码范围必须为 1..12"),
    ],
)
def test_baseline_rejects_invalid_historical_numbers(baseline, field, value, message):
    invalid_draw = dict(TRAINING_DRAWS[0], **{field: value})

    with pytest.raises(ValueError, match=re.escape(message)):
        baseline.generate([invalid_draw], count=1, rng=Random(1))
