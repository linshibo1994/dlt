"""下一期概率基线选号服务测试。"""

import csv
import json
import random
from dataclasses import FrozenInstanceError
from pathlib import Path
from random import Random

import pytest

from backend import evaluation
from backend.evaluation import DirichletBaseline, WalkForwardEvaluator, derive_seed
from backend.evaluation.prediction import BaselinePredictor, PredictionConfig
from backend.testing import DltDataSource


def make_draw(issue: int):
    """构造按期号可审计的合法开奖记录。"""
    offset = issue - 1000
    return {
        "issue": str(issue),
        "date": f"2026-01-{offset:02d}",
        "front_balls": ",".join(
            f"{number:02d}" for number in range(offset, offset + 5)
        ),
        "back_balls": f"{offset:02d},{offset + 1:02d}",
    }


def write_csv(tmp_path: Path, issues):
    """写入由真实数据源和严格评估读取器消费的临时 CSV。"""
    csv_file = tmp_path / "dlt_data_all.csv"
    with csv_file.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=("issue", "date", "front_balls", "back_balls"),
        )
        writer.writeheader()
        writer.writerows(make_draw(issue) for issue in issues)
    return DltDataSource(str(csv_file))


def collect_keys(value):
    """递归收集结构化结果中的字典键。"""
    if isinstance(value, dict):
        keys = set(value)
        for item in value.values():
            keys.update(collect_keys(item))
        return keys
    if isinstance(value, list):
        keys = set()
        for item in value:
            keys.update(collect_keys(item))
        return keys
    return set()


def assert_valid_unique_tickets(tickets, expected_count):
    """断言票数、号码规则和跨票唯一性。"""
    assert len(tickets) == expected_count
    keys = set()
    for ticket in tickets:
        assert set(ticket) == {"front_balls", "back_balls"}
        front = ticket["front_balls"]
        back = ticket["back_balls"]
        assert len(front) == len(set(front)) == 5
        assert len(back) == len(set(back)) == 2
        assert front == sorted(front)
        assert back == sorted(back)
        assert all(isinstance(number, int) and 1 <= number <= 35 for number in front)
        assert all(isinstance(number, int) and 1 <= number <= 12 for number in back)
        keys.add((tuple(front), tuple(back)))
    assert len(keys) == expected_count


def test_prediction_config_is_frozen_and_publicly_exported():
    config = evaluation.PredictionConfig()

    assert config == PredictionConfig(
        method="dirichlet",
        periods=500,
        count=5,
        seed=42,
        alpha=1.0,
    )
    with pytest.raises(FrozenInstanceError):
        config.count = 3

    assert evaluation.BaselinePredictor is BaselinePredictor
    assert "BaselinePredictor" in evaluation.__all__
    assert "PredictionConfig" in evaluation.__all__


def test_predict_is_reproducible_and_uses_latest_known_window(tmp_path):
    source = write_csv(tmp_path, issues=range(1010, 1000, -1))
    predictor = BaselinePredictor(data_source=source)
    config = PredictionConfig(
        method="dirichlet",
        periods=5,
        count=3,
        seed=42,
        alpha=1.0,
    )
    random.seed(20260713)
    global_state = random.getstate()

    first = predictor.predict(config)
    second = predictor.predict(config)

    assert first == second
    assert random.getstate() == global_state
    assert set(first) == {"config", "data", "method", "tickets", "disclaimer"}
    assert first["config"] == {
        "method": "dirichlet",
        "periods": 5,
        "count": 3,
        "seed": 42,
        "alpha": 1.0,
    }
    assert first["data"] == {
        "latest_issue": "1010",
        "training_newest_issue": "1010",
        "training_oldest_issue": "1006",
        "training_periods": 5,
        "available_draws": 10,
    }
    assert first["method"] == "dirichlet"
    assert first["disclaimer"] == "仅用于历史比较，不代表未来中奖概率"

    training = WalkForwardEvaluator(data_source=source).load_draws()[:5]
    expected_tickets = DirichletBaseline(alpha=1.0).generate(
        training,
        count=3,
        rng=Random(derive_seed(42, "dirichlet", "1010:next")),
    )
    assert first["tickets"] == expected_tickets
    assert json.loads(json.dumps(first, ensure_ascii=False)) == first

    result_keys = collect_keys(first)
    assert result_keys.isdisjoint(
        {
            "timestamp",
            "execution_time",
            "elapsed_time",
            "duration",
            "created_at",
            "generated_at",
        }
    )


@pytest.mark.parametrize("method", ["uniform", "dirichlet"])
def test_each_method_generates_requested_valid_unique_tickets(tmp_path, method):
    source = write_csv(tmp_path, issues=range(1010, 1000, -1))
    config = PredictionConfig(method=method, periods=5, count=5, seed=7, alpha=1.0)

    result = BaselinePredictor(data_source=source).predict(config)

    assert result["method"] == method
    assert_valid_unique_tickets(result["tickets"], expected_count=5)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("method", "markov", "method 仅支持 uniform、dirichlet"),
        ("method", None, "method 仅支持 uniform、dirichlet"),
        ("periods", 0, "periods 必须为正整数"),
        ("periods", -1, "periods 必须为正整数"),
        ("periods", 1.5, "periods 必须为正整数"),
        ("periods", True, "periods 必须为正整数"),
        ("count", 0, "count 必须为 1..100 的正整数"),
        ("count", 101, "count 必须为 1..100 的正整数"),
        ("count", 1.5, "count 必须为 1..100 的正整数"),
        ("count", True, "count 必须为 1..100 的正整数"),
        ("seed", "42", "seed 必须为整数"),
        ("seed", 1.5, "seed 必须为整数"),
        ("seed", True, "seed 必须为整数"),
        ("alpha", 0, "alpha 必须为可计算的有限正数"),
        ("alpha", -1, "alpha 必须为可计算的有限正数"),
        ("alpha", float("inf"), "alpha 必须为可计算的有限正数"),
        ("alpha", float("nan"), "alpha 必须为可计算的有限正数"),
        ("alpha", 5e-324, "alpha 必须为可计算的有限正数"),
        ("alpha", None, "alpha 必须为可计算的有限正数"),
    ],
)
def test_prediction_config_rejects_invalid_values(field, value, message):
    values = {
        "method": "dirichlet",
        "periods": 5,
        "count": 3,
        "seed": 42,
        "alpha": 1.0,
    }
    values[field] = value

    with pytest.raises(ValueError) as exc_info:
        PredictionConfig(**values)

    assert str(exc_info.value) == message


def test_predict_rejects_insufficient_history(tmp_path):
    source = write_csv(tmp_path, issues=range(1004, 1000, -1))
    config = PredictionConfig(method="uniform", periods=5, count=1)

    with pytest.raises(ValueError) as exc_info:
        BaselinePredictor(data_source=source).predict(config)

    assert str(exc_info.value) == "数据不足：至少需要 5 期，当前只有 4 期"
