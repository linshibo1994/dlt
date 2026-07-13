"""无泄漏滚动评估器测试。"""

import csv
import random
from dataclasses import FrozenInstanceError
from pathlib import Path
from random import Random

import pytest

from backend import evaluation
from backend.evaluation.walk_forward import (
    EvaluationCase,
    EvaluationConfig,
    WalkForwardEvaluator,
)
from backend.testing import DltDataSource


def make_draw(issue: int):
    """构造按期号可审计的合法开奖记录。"""
    return {
        "issue": str(issue),
        "date": f"2026-01-{issue - 1000:02d}",
        "front_balls": "01,02,03,04,05",
        "back_balls": "01,02",
    }


def write_csv(tmp_path: Path, issues):
    """写入真实数据源可读取的临时开奖记录。"""
    csv_file = tmp_path / "dlt_data_all.csv"
    with csv_file.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=("issue", "date", "front_balls", "back_balls"),
        )
        writer.writeheader()
        writer.writerows(make_draw(issue) for issue in issues)
    return DltDataSource(str(csv_file))


def write_raw_csv(tmp_path: Path, content: str):
    """写入用于严格读取测试的原始 CSV。"""
    csv_file = tmp_path / "strict_dlt_data.csv"
    csv_file.write_text(content, encoding="utf-8")
    return DltDataSource(str(csv_file))


def collect_keys(value):
    """递归收集结果中的所有字典键。"""
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


def test_build_cases_uses_only_draws_older_than_target():
    draws = [make_draw(issue) for issue in range(1008, 1000, -1)]
    config = EvaluationConfig(
        methods=("uniform",),
        draws=2,
        periods=3,
        count=1,
        seed=42,
        alpha=1.0,
    )

    cases = WalkForwardEvaluator().build_cases(draws, config)

    assert [case.target["issue"] for case in cases] == ["1008", "1007"]
    assert [row["issue"] for row in cases[0].training] == ["1007", "1006", "1005"]
    assert [row["issue"] for row in cases[1].training] == ["1006", "1005", "1004"]
    assert all(len(case.training) == 3 for case in cases)
    assert all(
        int(row["issue"]) < int(case.target["issue"])
        for case in cases
        for row in case.training
    )


def test_public_api_exports_frozen_config_with_documented_defaults():
    config = evaluation.EvaluationConfig()

    assert config == EvaluationConfig(
        methods=("uniform", "dirichlet"),
        draws=30,
        periods=500,
        count=5,
        seed=42,
        alpha=1.0,
    )
    with pytest.raises(FrozenInstanceError):
        config.draws = 10

    for name in (
        "EvaluationConfig",
        "EvaluationCase",
        "WalkForwardEvaluator",
        "derive_seed",
        "match_ticket",
    ):
        assert name in evaluation.__all__


def test_config_defensively_copies_list_methods_to_tuple():
    methods = ["uniform", "dirichlet"]

    config = EvaluationConfig(methods=methods)
    methods.append("uniform")

    assert config.methods == ("uniform", "dirichlet")


def test_evaluation_case_defensively_copies_target_and_training():
    target = make_draw(1003)
    training = [make_draw(1002), make_draw(1001)]

    case = EvaluationCase(target=target, training=tuple(training))
    target["issue"] = "9999"
    training[0]["issue"] = "9998"

    assert case.target["issue"] == "1003"
    assert [row["issue"] for row in case.training] == ["1002", "1001"]


def test_derive_seed_uses_stable_sha256_vector():
    assert evaluation.derive_seed(42, "uniform", "1008") == 14892183828196736113
    assert evaluation.derive_seed(42, "uniform", "1008") == evaluation.derive_seed(
        42, "uniform", "1008"
    )
    assert evaluation.derive_seed(42, "dirichlet", "1008") != evaluation.derive_seed(
        42, "uniform", "1008"
    )


@pytest.mark.parametrize(
    ("ticket", "expected"),
    [
        ({"front_balls": [1, 2, 3, 4, 5], "back_balls": [1, 2]}, (5, 2)),
        ({"front_balls": [1, 2, 6, 7, 8], "back_balls": [1, 3]}, (2, 1)),
    ],
)
def test_match_ticket_returns_exact_front_and_back_hits(ticket, expected):
    target = {
        "front_balls": "01,02,03,04,05",
        "back_balls": "01,02",
    }

    assert evaluation.match_ticket(ticket, target) == expected


def test_match_ticket_reuses_number_validation():
    target = {
        "front_balls": "01,02,03,04,05",
        "back_balls": "01,02",
    }
    invalid_ticket = {
        "front_balls": [1, 1, 2, 3, 4],
        "back_balls": [1, 2],
    }

    with pytest.raises(ValueError, match="前区号码必须包含 5 个唯一整数"):
        evaluation.match_ticket(invalid_ticket, target)


def test_build_cases_rejects_insufficient_data():
    draws = [make_draw(issue) for issue in range(1004, 1000, -1)]
    config = EvaluationConfig(methods=("uniform",), draws=2, periods=3, count=1)

    with pytest.raises(ValueError) as exc_info:
        WalkForwardEvaluator().build_cases(draws, config)

    assert str(exc_info.value) == "数据不足：至少需要 5 期，当前只有 4 期"


def test_build_cases_normalizes_unsorted_draws_from_newest_to_oldest():
    draws = [
        make_draw(1002),
        make_draw(1005),
        make_draw(1001),
        make_draw(1004),
        make_draw(1003),
    ]
    config = EvaluationConfig(methods=("uniform",), draws=2, periods=2, count=1)

    cases = WalkForwardEvaluator().build_cases(draws, config)

    assert [case.target["issue"] for case in cases] == ["1005", "1004"]
    assert [row["issue"] for row in cases[0].training] == ["1004", "1003"]
    assert [row["issue"] for row in cases[1].training] == ["1003", "1002"]


def test_build_cases_rejects_duplicate_issue():
    duplicate = dict(make_draw(1002), date="2026-01-01")
    draws = [make_draw(1003), make_draw(1002), duplicate, make_draw(1001)]
    config = EvaluationConfig(methods=("uniform",), draws=1, periods=2, count=1)

    with pytest.raises(ValueError, match="期号 1002 重复"):
        WalkForwardEvaluator().build_cases(draws, config)


@pytest.mark.parametrize(
    ("draws", "message"),
    [
        (
            [
                dict(make_draw(1002), date="2026-01-03"),
                dict(make_draw(1001), date="2026-01-03"),
            ],
            "训练日期必须早于目标日期",
        ),
        (
            [
                dict(make_draw(1001), date="2026-01-03"),
                dict(make_draw(1002), date="2026-01-02"),
            ],
            "训练期号必须早于目标期号",
        ),
    ],
)
def test_build_cases_explicitly_rejects_non_earlier_training_rows(draws, message):
    config = EvaluationConfig(methods=("uniform",), draws=1, periods=1, count=1)

    with pytest.raises(ValueError, match=message):
        WalkForwardEvaluator().build_cases(draws, config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("issue", "10A2", "期号 issue 必须为纯数字"),
        ("issue", " 1002", "期号 issue 必须为纯数字"),
        ("date", "2026-02-30", "date 必须为有效的 YYYY-MM-DD"),
        ("date", "2026-01-02 ", "date 必须为有效的 YYYY-MM-DD"),
        ("front_balls", "01,02,03,04,36", "前区号码范围必须为 1..35"),
        ("back_balls", "01,13", "后区号码范围必须为 1..12"),
    ],
)
def test_build_cases_strictly_validates_rows(field, value, message):
    invalid_target = dict(make_draw(1002), **{field: value})
    config = EvaluationConfig(methods=("uniform",), draws=1, periods=1, count=1)

    with pytest.raises(ValueError, match=message):
        WalkForwardEvaluator().build_cases([invalid_target, make_draw(1001)], config)


def test_build_cases_rejects_missing_required_field():
    invalid_target = make_draw(1002)
    del invalid_target["back_balls"]
    config = EvaluationConfig(methods=("uniform",), draws=1, periods=1, count=1)

    with pytest.raises(ValueError, match="缺少必需字段：back_balls"):
        WalkForwardEvaluator().build_cases([invalid_target, make_draw(1001)], config)


def test_build_cases_cases_do_not_change_when_input_draws_are_mutated():
    draws = [make_draw(issue) for issue in range(1004, 1000, -1)]
    config = EvaluationConfig(methods=("uniform",), draws=1, periods=2, count=1)

    cases = WalkForwardEvaluator().build_cases(draws, config)
    draws[0]["issue"] = "9999"
    draws[1]["issue"] = "9998"

    assert cases[0].target["issue"] == "1004"
    assert [row["issue"] for row in cases[0].training] == ["1003", "1002"]


@pytest.mark.parametrize(
    ("methods", "message"),
    [
        ((), "methods 不能为空"),
        (("markov",), "methods 仅支持 uniform、dirichlet"),
        (None, "methods 必须为非空的方法序列"),
    ],
)
def test_build_cases_rejects_invalid_methods(methods, message):
    config = EvaluationConfig(methods=methods, draws=1, periods=1, count=1)

    with pytest.raises(ValueError, match=message):
        WalkForwardEvaluator().build_cases([make_draw(1002), make_draw(1001)], config)


@pytest.mark.parametrize("field", ["draws", "periods"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_build_cases_rejects_invalid_positive_integer_config(field, value):
    values = {"methods": ("uniform",), "draws": 1, "periods": 1, "count": 1}
    values[field] = value
    config = EvaluationConfig(**values)

    with pytest.raises(ValueError, match=f"{field} 必须为正整数"):
        WalkForwardEvaluator().build_cases([make_draw(1002), make_draw(1001)], config)


@pytest.mark.parametrize("value", [0, -1, 101, 1.5, True])
def test_build_cases_rejects_invalid_count(value):
    config = EvaluationConfig(
        methods=("uniform",), draws=1, periods=1, count=value
    )

    with pytest.raises(ValueError, match="count 必须为 1..100 的正整数"):
        WalkForwardEvaluator().build_cases([make_draw(1002), make_draw(1001)], config)


@pytest.mark.parametrize("value", [1.5, "42", True])
def test_build_cases_rejects_invalid_seed(value):
    config = EvaluationConfig(
        methods=("uniform",), draws=1, periods=1, count=1, seed=value
    )

    with pytest.raises(ValueError, match="seed 必须为整数"):
        WalkForwardEvaluator().build_cases([make_draw(1002), make_draw(1001)], config)


@pytest.mark.parametrize(
    "value",
    [0, -1, float("inf"), float("nan"), 5e-324, None],
)
def test_build_cases_rejects_alpha_not_accepted_by_dirichlet(value):
    config = EvaluationConfig(
        methods=("uniform",), draws=1, periods=1, count=1, alpha=value
    )

    with pytest.raises(ValueError, match="alpha 必须为可计算的有限正数"):
        WalkForwardEvaluator().build_cases([make_draw(1002), make_draw(1001)], config)


def test_run_is_reproducible_and_reports_auditable_boundaries(tmp_path):
    source = write_csv(tmp_path, issues=range(1010, 1000, -1))
    config = EvaluationConfig(
        methods=("uniform", "dirichlet"),
        draws=2,
        periods=5,
        count=3,
        seed=7,
        alpha=1.0,
    )
    random.seed(20260713)
    global_state = random.getstate()

    first = WalkForwardEvaluator(data_source=source).run(config)
    second = WalkForwardEvaluator(data_source=source).run(config)

    assert first == second
    assert random.getstate() == global_state
    assert first["config"] == {
        "methods": ["uniform", "dirichlet"],
        "draws": 2,
        "periods": 5,
        "count": 3,
        "seed": 7,
        "alpha": 1.0,
    }
    assert first["data"] == {
        "latest_issue": "1010",
        "available_draws": 10,
        "evaluated_draws": 2,
    }
    assert first["disclaimer"] == "仅用于历史比较，不代表未来中奖概率"
    assert set(first["methods"]) == {"uniform", "dirichlet"}

    for method, summary in first["methods"].items():
        assert summary["evaluated_draws"] == 2
        assert summary["ticket_count"] == 6
        assert sum(summary["match_distribution"].values()) == 6
        assert len(summary["draw_details"]) == 2
        assert list(summary["match_distribution"]) == sorted(
            summary["match_distribution"],
            key=lambda combination: tuple(
                int(value) for value in combination.split("+")
            ),
        )

        expected_boundaries = [
            ("1010", "1009", "1005"),
            ("1009", "1008", "1004"),
        ]
        for detail, expected in zip(summary["draw_details"], expected_boundaries):
            target_issue, newest_issue, oldest_issue = expected
            assert detail["target_issue"] == target_issue
            assert detail["target_date"]
            assert detail["training_newest_issue"] == newest_issue
            assert detail["training_oldest_issue"] == oldest_issue
            assert int(detail["training_newest_issue"]) < int(detail["target_issue"])
            assert detail["training_count"] == 5
            assert sum(detail["match_distribution"].values()) == 3
            assert len(detail["tickets"]) == 3
            for ticket in detail["tickets"]:
                assert {
                    "front_balls",
                    "back_balls",
                    "front_matches",
                    "back_matches",
                    "match_combination",
                } <= set(ticket)

        if method == "uniform":
            assert "vs_uniform" not in summary
        else:
            assert set(summary["vs_uniform"]) == {
                "average_front_matches_delta",
                "average_back_matches_delta",
            }

    result_keys = collect_keys(first)
    assert "timestamp" not in result_keys
    assert "execution_time" not in result_keys


def test_run_uses_derived_rng_and_deduplicates_methods(tmp_path, monkeypatch):
    source = write_csv(tmp_path, issues=range(1003, 1000, -1))
    calls = []

    def record_generate(method):
        def generate(self, training_draws, count, rng):
            target_issue = str(int(training_draws[0]["issue"]) + 1)
            calls.append((method, target_issue, rng.random()))
            return [{"front_balls": [1, 2, 3, 4, 5], "back_balls": [1, 2]}]

        return generate

    monkeypatch.setattr(
        evaluation.UniformBaseline,
        "generate",
        record_generate("uniform"),
    )
    monkeypatch.setattr(
        evaluation.DirichletBaseline,
        "generate",
        record_generate("dirichlet"),
    )
    config = EvaluationConfig(
        methods=("uniform", "uniform", "dirichlet"),
        draws=2,
        periods=1,
        count=1,
        seed=99,
    )

    result = WalkForwardEvaluator(data_source=source).run(config)

    assert result["config"]["methods"] == ["uniform", "dirichlet"]
    assert [(method, issue) for method, issue, _ in calls] == [
        ("uniform", "1003"),
        ("uniform", "1002"),
        ("dirichlet", "1003"),
        ("dirichlet", "1002"),
    ]
    assert [value for _, _, value in calls] == [
        Random(evaluation.derive_seed(99, method, issue)).random()
        for method, issue, _ in calls
    ]


def test_run_calculates_exact_averages_and_match_distribution(
    tmp_path, monkeypatch
):
    source = write_csv(tmp_path, issues=range(1002, 1000, -1))

    def fixed_generate(self, training_draws, count, rng):
        assert [row["issue"] for row in training_draws] == ["1001"]
        assert count == 2
        return [
            {"front_balls": [1, 2, 3, 4, 5], "back_balls": [1, 2]},
            {"front_balls": [1, 2, 6, 7, 8], "back_balls": [1, 3]},
        ]

    monkeypatch.setattr(evaluation.UniformBaseline, "generate", fixed_generate)
    config = EvaluationConfig(
        methods=("uniform",), draws=1, periods=1, count=2, seed=42
    )

    result = WalkForwardEvaluator(data_source=source).run(config)
    summary = result["methods"]["uniform"]

    assert summary["evaluated_draws"] == 1
    assert summary["ticket_count"] == 2
    assert summary["average_front_matches"] == 3.5
    assert summary["average_back_matches"] == 1.5
    assert summary["match_distribution"] == {"2+1": 1, "5+2": 1}
    assert summary["jackpot_matches"] == 1
    detail = summary["draw_details"][0]
    assert detail["match_distribution"] == {"2+1": 1, "5+2": 1}
    assert [ticket["match_combination"] for ticket in detail["tickets"]] == [
        "5+2",
        "2+1",
    ]


def test_vs_uniform_delta_uses_unrounded_averages(tmp_path, monkeypatch):
    source = write_csv(tmp_path, issues=range(1002, 1000, -1))
    no_match_tickets = [
        {"front_balls": [6, 7, 8, 9, 10], "back_balls": [3, 4]},
        {"front_balls": [11, 12, 13, 14, 15], "back_balls": [5, 6]},
    ]

    def uniform_generate(self, training_draws, count, rng):
        return [
            {"front_balls": [1, 6, 7, 8, 9], "back_balls": [3, 4]},
            *no_match_tickets,
        ]

    def dirichlet_generate(self, training_draws, count, rng):
        return [
            {"front_balls": [1, 2, 6, 7, 8], "back_balls": [3, 4]},
            *no_match_tickets,
        ]

    monkeypatch.setattr(evaluation.UniformBaseline, "generate", uniform_generate)
    monkeypatch.setattr(
        evaluation.DirichletBaseline,
        "generate",
        dirichlet_generate,
    )
    config = EvaluationConfig(
        methods=("uniform", "dirichlet"),
        draws=1,
        periods=1,
        count=3,
    )

    result = WalkForwardEvaluator(data_source=source).run(config)

    assert result["methods"]["uniform"]["average_front_matches"] == 0.333333
    assert result["methods"]["dirichlet"]["average_front_matches"] == 0.666667
    assert result["methods"]["dirichlet"]["vs_uniform"] == {
        "average_front_matches_delta": 0.333333,
        "average_back_matches_delta": 0.0,
    }


def test_run_rejects_insufficient_csv_data(tmp_path):
    source = write_csv(tmp_path, issues=range(1005, 1000, -1))
    config = EvaluationConfig(
        methods=("uniform",), draws=2, periods=5, count=1
    )

    with pytest.raises(ValueError, match="数据不足"):
        WalkForwardEvaluator(data_source=source).run(config)


def test_run_strict_csv_rejects_missing_required_header(tmp_path):
    source = write_raw_csv(
        tmp_path,
        "issue,date,front_balls\n"
        '1002,2026-01-02,"01,02,03,04,05"\n'
        '1001,2026-01-01,"01,02,03,04,05"\n',
    )
    config = EvaluationConfig(methods=("uniform",), draws=1, periods=1, count=1)

    with pytest.raises(ValueError, match="CSV 缺少必需表头：back_balls"):
        WalkForwardEvaluator(data_source=source).run(config)


@pytest.mark.parametrize(
    ("invalid_row", "reason"),
    [
        (
            '10A2,2026-01-02,"01,02,03,04,05","01,02"',
            "期号 issue 必须为纯数字",
        ),
        (
            '1002,2026-02-30,"01,02,03,04,05","01,02"',
            "date 必须为有效的 YYYY-MM-DD",
        ),
        (
            '1002,2026-01-02,"01,02,03,04,36","01,02"',
            "前区号码范围必须为 1..35",
        ),
        (
            '1002,2026-01-02,"01,02,03,04,05","01,13"',
            "后区号码范围必须为 1..12",
        ),
        (
            '1002,2026-01-02,"01,02,03,04,05",',
            "后区号码必须包含 2 个唯一整数",
        ),
    ],
)
def test_run_strict_csv_rejects_bad_physical_row_with_line_number(
    tmp_path, invalid_row, reason
):
    source = write_raw_csv(
        tmp_path,
        "issue,date,front_balls,back_balls\n"
        '1003,2026-01-03,"01,02,03,04,05","01,02"\n'
        f"{invalid_row}\n"
        '1001,2026-01-01,"01,02,03,04,05","01,02"\n',
    )
    config = EvaluationConfig(methods=("uniform",), draws=1, periods=1, count=1)

    with pytest.raises(ValueError) as exc_info:
        WalkForwardEvaluator(data_source=source).run(config)

    assert "CSV 第 3 行" in str(exc_info.value)
    assert reason in str(exc_info.value)


def test_run_strict_csv_rejects_blank_physical_data_line(tmp_path):
    source = write_raw_csv(
        tmp_path,
        "issue,date,front_balls,back_balls\n"
        '1003,2026-01-03,"01,02,03,04,05","01,02"\n'
        "\n"
        '1001,2026-01-01,"01,02,03,04,05","01,02"\n',
    )
    config = EvaluationConfig(methods=("uniform",), draws=1, periods=1, count=1)

    with pytest.raises(ValueError, match="CSV 第 3 行：数据行不能为空"):
        WalkForwardEvaluator(data_source=source).run(config)


def test_run_strict_csv_rejects_duplicate_issue_with_line_number(tmp_path):
    source = write_raw_csv(
        tmp_path,
        "issue,date,front_balls,back_balls\n"
        '1003,2026-01-03,"01,02,03,04,05","01,02"\n'
        '1002,2026-01-02,"01,02,03,04,05","01,02"\n'
        '1002,2026-01-01,"01,02,03,04,05","01,02"\n',
    )
    config = EvaluationConfig(methods=("uniform",), draws=1, periods=1, count=1)

    with pytest.raises(ValueError, match="CSV 第 4 行：期号 1002 重复"):
        WalkForwardEvaluator(data_source=source).run(config)


def test_run_strictly_validates_injected_source_without_data_file():
    class InjectedSource:
        def load_all(self):
            return [
                dict(make_draw(1002), date="2026-02-30"),
                make_draw(1001),
            ]

    config = EvaluationConfig(methods=("uniform",), draws=1, periods=1, count=1)

    with pytest.raises(ValueError, match="date 必须为有效的 YYYY-MM-DD"):
        WalkForwardEvaluator(data_source=InjectedSource()).run(config)
