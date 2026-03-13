#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

from backend.testing import DltDataSource, DltRule, SessionConfig, TestEngine
from backend.testing.runner import PredictionRunner


class DummyRunner:
    """用于测试引擎参数透传的假执行器。"""

    def __init__(self):
        self.calls = []

    def run(self, method: str, periods: int, count: int, extra=None):
        self.calls.append({"method": method, "periods": periods, "count": count})
        # 每次返回一个稳定票据，便于判断中奖逻辑
        return {
            "success": True,
            "method": method,
            "periods": periods,
            "count": count,
            "predictions": [
                {
                    "front_balls": [2, 4, 8, 10, 21],
                    "back_balls": [9, 12],
                }
            ],
            "source": "dummy",
        }


def test_dlt_rule_prize_level():
    rule = DltRule()
    winning = {"front_balls": "01,02,03,04,05", "back_balls": "01,02"}

    first = rule.evaluate({"front_balls": [1, 2, 3, 4, 5], "back_balls": [1, 2]}, winning)
    second = rule.evaluate({"front_balls": [1, 2, 3, 4, 5], "back_balls": [1, 3]}, winning)
    ninth = rule.evaluate({"front_balls": [1, 2, 9, 10, 11], "back_balls": [1, 3]}, winning)

    assert first["prize_level"] == 1
    assert second["prize_level"] == 2
    assert ninth["prize_level"] == 9


def test_prediction_runner_parse_cli_output_inline_and_multiline():
    runner = PredictionRunner(prefer_direct=False, fallback_subprocess=False)

    inline = "第 1 注: 02 04 08 10 21 + 09 12 (方法: markov, 置信度: 0.5)"
    parsed_inline = runner.parse_cli_output(inline)
    assert len(parsed_inline) == 1
    assert parsed_inline[0]["front_balls"] == [2, 4, 8, 10, 21]
    assert parsed_inline[0]["back_balls"] == [9, 12]

    multiline = """
第1注 [markov]:
  前区: 02 04 08 10 21
  后区: 09 12
"""
    parsed_multiline = runner.parse_cli_output(multiline)
    assert len(parsed_multiline) == 1
    assert parsed_multiline[0]["front_balls"] == [2, 4, 8, 10, 21]
    assert parsed_multiline[0]["back_balls"] == [9, 12]


def test_test_engine_random_strategy_parameter_routing(tmp_path: Path):
    csv_file = tmp_path / "dlt_data_all.csv"
    csv_file.write_text(
        "issue,date,front_balls,back_balls\n"
        "26024,2026-03-09,\"02,04,08,10,21\",\"09,12\"\n"
        "26023,2026-03-07,\"09,25,26,27,28\",\"01,08\"\n",
        encoding="utf-8",
    )

    runner = DummyRunner()
    engine = TestEngine(
        rule=DltRule(),
        runner=runner,
        data_source=DltDataSource(str(csv_file)),
        results_dir=str(tmp_path / "results"),
        seed=42,
    )

    cfg = SessionConfig(
        methods=["markov", "frequency"],
        strategy="random",
        target_prize="六等奖",
        periods_start=20,
        periods_end=30,
        count_start=1,
        count_end=3,
        max_tests=3,
        parallel=False,
        workers=1,
    )

    summary = engine.run_session(cfg)

    assert summary["total_tests"] == 2  # 目标奖级很快达成，方法会提前停止
    assert "best_methods" in summary
    assert summary["report_files"]["json"].endswith(".json")

    for call in runner.calls:
        assert 20 <= call["periods"] <= 30
        assert 1 <= call["count"] <= 3


def test_test_engine_progressive_strategy_uses_step(tmp_path: Path):
    csv_file = tmp_path / "dlt_data_all.csv"
    csv_file.write_text(
        "issue,date,front_balls,back_balls\n"
        "26024,2026-03-09,\"02,04,08,10,21\",\"09,12\"\n",
        encoding="utf-8",
    )

    class NonWinningRunner(DummyRunner):
        def run(self, method: str, periods: int, count: int, extra=None):
            self.calls.append({"method": method, "periods": periods, "count": count})
            return {
                "success": True,
                "method": method,
                "periods": periods,
                "count": count,
                "predictions": [{"front_balls": [1, 6, 7, 8, 9], "back_balls": [1, 2]}],
                "source": "dummy",
            }

    runner = NonWinningRunner()
    engine = TestEngine(
        rule=DltRule(),
        runner=runner,
        data_source=DltDataSource(str(csv_file)),
        results_dir=str(tmp_path / "results"),
        seed=7,
    )

    cfg = SessionConfig(
        methods=["markov"],
        strategy="progressive",
        target_prize="一等奖",
        periods_start=10,
        periods_end=30,
        count_start=1,
        count_end=1,
        max_tests=3,
        parallel=False,
        workers=1,
        progressive_step=10,
    )

    summary = engine.run_session(cfg)

    assert summary["total_tests"] == 3
    assert [call["periods"] for call in runner.calls] == [10, 20, 30]
