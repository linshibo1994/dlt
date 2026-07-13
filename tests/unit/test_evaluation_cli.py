"""概率基线评估终端命令测试。"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from backend.evaluation import cli


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_root(*arguments):
    """通过项目根入口执行真实终端命令。"""
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "main.py"), *arguments],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_build_parser_exposes_documented_defaults_and_normalizes_methods():
    parser = cli.build_parser()

    predict = parser.parse_args(["predict"])
    walk_forward = parser.parse_args(
        ["walk-forward", "--methods", "uniform,dirichlet,uniform"]
    )

    assert vars(predict) == {
        "command": "predict",
        "method": "dirichlet",
        "periods": 500,
        "count": 5,
        "seed": 42,
        "alpha": 1.0,
        "json_output": False,
    }
    assert vars(walk_forward) == {
        "command": "walk-forward",
        "methods": ("uniform", "dirichlet"),
        "draws": 30,
        "periods": 500,
        "count": 5,
        "seed": 42,
        "alpha": 1.0,
        "json_output": False,
    }


def test_predict_parser_accepts_existing_cli_short_options():
    parsed = cli.build_parser().parse_args(
        ["predict", "-m", "dirichlet", "-p", "20", "-c", "1"]
    )

    assert parsed.method == "dirichlet"
    assert parsed.periods == 20
    assert parsed.count == 1


def test_predict_json_output_is_a_single_parseable_object(capsys):
    exit_code = cli.main(
        [
            "predict",
            "--method",
            "dirichlet",
            "--periods",
            "5",
            "--count",
            "3",
            "--seed",
            "7",
            "--json-output",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert payload["method"] == "dirichlet"
    assert payload["config"]["count"] == 3
    assert len(payload["tickets"]) == 3


def test_walk_forward_json_output_is_parseable_and_methods_are_deduplicated(capsys):
    exit_code = cli.main(
        [
            "walk-forward",
            "--methods",
            "uniform,dirichlet,uniform",
            "--draws",
            "2",
            "--periods",
            "5",
            "--count",
            "2",
            "--json-output",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert payload["config"]["methods"] == ["uniform", "dirichlet"]
    assert payload["data"]["evaluated_draws"] == 2
    assert set(payload["methods"]) == {"uniform", "dirichlet"}


@pytest.mark.parametrize(
    "arguments",
    [
        ("evaluate", "predict", "--periods", "5", "--count", "3", "--json-output"),
        (
            "evaluate",
            "walk-forward",
            "--draws",
            "2",
            "--periods",
            "5",
            "--count",
            "2",
            "--json-output",
        ),
    ],
    ids=["predict", "walk-forward"],
)
def test_root_evaluate_json_commands_are_clean(arguments):
    completed = run_root(*arguments)

    payload = json.loads(completed.stdout)

    assert completed.returncode == 0
    assert completed.stderr == ""
    assert isinstance(payload, dict)
    assert "gpu" not in (completed.stdout + completed.stderr).lower()


@pytest.mark.parametrize("method", ["uniform", "dirichlet"])
def test_root_predict_alias_uses_probability_baseline(method):
    completed = run_root(
        "predict",
        "-m",
        method,
        "-p",
        "20",
        "-c",
        "1",
        "--json-output",
    )

    payload = json.loads(completed.stdout)

    assert completed.returncode == 0
    assert completed.stderr == ""
    assert payload["method"] == method
    assert len(payload["tickets"]) == 1
    assert "gpu" not in completed.stdout.lower()


def test_baseline_predict_method_detection_preserves_existing_methods():
    from main import _baseline_predict_method

    assert _baseline_predict_method(["-m", "dirichlet", "-c", "1"]) == "dirichlet"
    assert _baseline_predict_method(["--method=uniform"]) == "uniform"
    assert _baseline_predict_method(["-m", "hot_cold", "-c", "1"]) is None


def test_root_help_lists_evaluate_command_and_examples():
    completed = run_root("--help")

    assert completed.returncode == 0
    assert completed.stderr == ""
    assert "evaluate" in completed.stdout
    assert "python main.py predict -m dirichlet" in completed.stdout
    assert "python main.py evaluate walk-forward" in completed.stdout


def test_evaluate_help_returns_zero():
    completed = run_root("evaluate", "--help")

    assert completed.returncode == 0
    assert completed.stderr == ""
    assert "predict" in completed.stdout
    assert "walk-forward" in completed.stdout


@pytest.mark.parametrize(
    "arguments",
    [
        ("evaluate", "predict", "--method", "markov"),
        ("evaluate", "predict", "--count", "0"),
        ("evaluate", "walk-forward", "--methods", "uniform,markov"),
    ],
    ids=["invalid-method", "invalid-count", "invalid-walk-method"],
)
def test_root_evaluate_rejects_invalid_arguments(arguments):
    completed = run_root(*arguments)

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "error:" in completed.stderr


def test_predict_text_output_lists_five_tickets_and_disclaimer(capsys):
    exit_code = cli.main(["predict", "--periods", "5"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert "方法：dirichlet" in captured.out
    assert "最新期号：" in captured.out
    assert "训练窗口：" in captured.out
    assert captured.out.count("前区") == 5
    assert captured.out.count(" + 后区") == 5
    assert "仅用于历史比较，不代表未来中奖概率" in captured.out


def test_walk_forward_text_output_includes_summary_and_window_audit(capsys):
    exit_code = cli.main(
        ["walk-forward", "--draws", "1", "--periods", "5", "--count", "1"]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert "平均前区命中" in captured.out
    assert "平均后区命中" in captured.out
    assert "命中分布" in captured.out
    assert "窗口审计" in captured.out
    assert "仅用于历史比较，不代表未来中奖概率" in captured.out
