"""概率基线算法测试。"""

from random import Random

import pytest

from backend.evaluation import DirichletBaseline, UniformBaseline, parse_numbers


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


@pytest.mark.parametrize("alpha", [0, -1, -0.5])
def test_dirichlet_rejects_non_positive_alpha(alpha):
    with pytest.raises(ValueError, match="alpha 必须大于 0"):
        DirichletBaseline(alpha=alpha)


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

    with pytest.raises(ValueError, match=message):
        baseline.generate([invalid_draw], count=1, rng=Random(1))
