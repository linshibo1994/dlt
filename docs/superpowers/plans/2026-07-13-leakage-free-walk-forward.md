# 无泄漏滚动评估与概率基线实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增可复现的均匀随机与 Dirichlet 概率基线，并通过严格只读目标期之前数据的滚动评估命令进行历史比较。

**Architecture:** 新功能放在轻量的 `backend/evaluation` 包中，不依赖现有全局预测器。根入口在加载旧后端系统前直接分发 `evaluate`，保证启动快速且 `--json-output` 不混入初始化日志。

**Tech Stack:** Python 3.9+、标准库 `argparse/csv/hashlib/json/random`、pytest、现有 `backend.testing.data_source.DltDataSource`

---

## 文件职责

- `backend/evaluation/__init__.py`：导出概率基线、配置与评估器公共接口。
- `backend/evaluation/baselines.py`：解析历史开奖、计算平滑概率并生成合法唯一票据。
- `backend/evaluation/walk_forward.py`：构造无泄漏训练窗口、派生随机种子并汇总命中指标。
- `backend/evaluation/cli.py`：参数解析、校验、文本/JSON 输出和退出码。
- `main.py`：在旧后端初始化前分发 `evaluate` 命令并更新根帮助。
- `tests/unit/test_probability_baselines.py`：基线算法的确定性、合法性和平滑测试。
- `tests/unit/test_walk_forward_evaluation.py`：时间边界、指标和错误处理测试。
- `tests/unit/test_evaluation_cli.py`：CLI 参数、纯 JSON 与根入口测试。
- `README.md`：原理、命令、参数、结果解释与限制。

### Task 1: 概率基线算法

**Files:**
- Create: `backend/evaluation/__init__.py`
- Create: `backend/evaluation/baselines.py`
- Create: `tests/unit/test_probability_baselines.py`

- [ ] **Step 1: 写票据合法性与确定性失败测试**

```python
from random import Random

import pytest

from backend.evaluation.baselines import DirichletBaseline, UniformBaseline


TRAINING_DRAWS = [
    {"issue": "1003", "date": "2026-01-03", "front_balls": "01,02,03,04,05", "back_balls": "01,02"},
    {"issue": "1002", "date": "2026-01-02", "front_balls": "01,06,07,08,09", "back_balls": "01,03"},
]


def assert_valid_ticket(ticket):
    assert len(ticket["front_balls"]) == 5
    assert len(set(ticket["front_balls"])) == 5
    assert ticket["front_balls"] == sorted(ticket["front_balls"])
    assert all(1 <= value <= 35 for value in ticket["front_balls"])
    assert len(ticket["back_balls"]) == 2
    assert len(set(ticket["back_balls"])) == 2
    assert ticket["back_balls"] == sorted(ticket["back_balls"])
    assert all(1 <= value <= 12 for value in ticket["back_balls"])


@pytest.mark.parametrize("baseline", [UniformBaseline(), DirichletBaseline(alpha=1.0)])
def test_baseline_generates_unique_reproducible_valid_tickets(baseline):
    first = baseline.generate(TRAINING_DRAWS, count=5, rng=Random(42))
    second = baseline.generate(TRAINING_DRAWS, count=5, rng=Random(42))
    assert first == second
    assert len(first) == 5
    assert len({(tuple(item["front_balls"]), tuple(item["back_balls"])) for item in first}) == 5
    for ticket in first:
        assert_valid_ticket(ticket)
```

- [ ] **Step 2: 运行测试并确认 RED**

Run: `python -m pytest tests/unit/test_probability_baselines.py -v`

Expected: FAIL，错误为 `ModuleNotFoundError: No module named 'backend.evaluation'`。

- [ ] **Step 3: 实现最小基线公共接口与生成逻辑**

```python
class UniformBaseline:
    name = "uniform"

    def generate(self, training_draws, count, rng):
        validate_training_draws(training_draws)
        return generate_unique_tickets(
            count,
            lambda: {
                "front_balls": sorted(rng.sample(range(1, 36), 5)),
                "back_balls": sorted(rng.sample(range(1, 13), 2)),
            },
        )


class DirichletBaseline:
    name = "dirichlet"

    def __init__(self, alpha=1.0):
        if alpha <= 0:
            raise ValueError("alpha 必须大于 0")
        self.alpha = float(alpha)

    def probabilities(self, training_draws):
        rows = validate_training_draws(training_draws)
        return {
            "front": smoothed_probabilities(rows, "front_balls", 35, 5, self.alpha),
            "back": smoothed_probabilities(rows, "back_balls", 12, 2, self.alpha),
        }
```

实现 `parse_numbers`、`validate_training_draws`、`smoothed_probabilities`、`weighted_sample_without_replacement` 和 `generate_unique_tickets`。所有错误使用中文 `ValueError`，最多尝试 `max(1000, count * 100)` 次生成唯一票据，失败时显式报错。

- [ ] **Step 4: 写平滑概率与非法输入失败测试**

```python
def test_dirichlet_probabilities_include_unseen_numbers():
    probabilities = DirichletBaseline(alpha=1.0).probabilities(TRAINING_DRAWS)
    assert set(probabilities["front"]) == set(range(1, 36))
    assert set(probabilities["back"]) == set(range(1, 13))
    assert probabilities["front"][35] > 0
    assert sum(probabilities["front"].values()) == pytest.approx(1.0)
    assert sum(probabilities["back"].values()) == pytest.approx(1.0)


@pytest.mark.parametrize("alpha", [0, -1])
def test_dirichlet_rejects_non_positive_alpha(alpha):
    with pytest.raises(ValueError, match="alpha 必须大于 0"):
        DirichletBaseline(alpha=alpha)


def test_baseline_rejects_empty_training_data():
    with pytest.raises(ValueError, match="训练数据不能为空"):
        UniformBaseline().generate([], count=1, rng=Random(1))
```

- [ ] **Step 5: 运行测试并确认 GREEN**

Run: `python -m pytest tests/unit/test_probability_baselines.py -v`

Expected: PASS，全部基线测试通过。

- [ ] **Step 6: 提交概率基线**

```bash
git add backend/evaluation tests/unit/test_probability_baselines.py
git commit -m "feat: 添加可复现概率基线算法"
```

### Task 2: 无泄漏滚动评估器

**Files:**
- Create: `backend/evaluation/walk_forward.py`
- Modify: `backend/evaluation/__init__.py`
- Create: `tests/unit/test_walk_forward_evaluation.py`

- [ ] **Step 1: 写训练窗口边界失败测试**

```python
from backend.evaluation.walk_forward import EvaluationConfig, WalkForwardEvaluator


def make_draw(issue):
    return {
        "issue": str(issue),
        "date": f"2026-01-{issue - 1000:02d}",
        "front_balls": "01,02,03,04,05",
        "back_balls": "01,02",
    }


def test_build_cases_uses_only_draws_older_than_target():
    draws = [make_draw(issue) for issue in range(1008, 1000, -1)]
    config = EvaluationConfig(methods=("uniform",), draws=2, periods=3, count=1, seed=42, alpha=1.0)
    cases = WalkForwardEvaluator().build_cases(draws, config)
    assert [case.target["issue"] for case in cases] == ["1008", "1007"]
    assert [row["issue"] for row in cases[0].training] == ["1007", "1006", "1005"]
    assert all(int(row["issue"]) < int(cases[0].target["issue"]) for row in cases[0].training)
```

- [ ] **Step 2: 运行边界测试并确认 RED**

Run: `python -m pytest tests/unit/test_walk_forward_evaluation.py::test_build_cases_uses_only_draws_older_than_target -v`

Expected: FAIL，错误为 `ModuleNotFoundError` 或缺少 `WalkForwardEvaluator`。

- [ ] **Step 3: 实现配置、评估案例与严格切片**

```python
@dataclass(frozen=True)
class EvaluationConfig:
    methods: Tuple[str, ...] = ("uniform", "dirichlet")
    draws: int = 30
    periods: int = 500
    count: int = 5
    seed: int = 42
    alpha: float = 1.0


@dataclass(frozen=True)
class EvaluationCase:
    target: Dict[str, str]
    training: Tuple[Dict[str, str], ...]


def build_cases(self, draws, config):
    self.validate_config(config)
    required = config.draws + config.periods
    if len(draws) < required:
        raise ValueError(f"数据不足：至少需要 {required} 期，当前只有 {len(draws)} 期")
    return [
        EvaluationCase(target=draws[index], training=tuple(draws[index + 1:index + 1 + config.periods]))
        for index in range(config.draws)
    ]
```

- [ ] **Step 4: 写确定性、指标与数据不足测试**

```python
def test_run_is_reproducible_and_reports_auditable_boundaries(tmp_path):
    source = write_csv(tmp_path, issues=range(1010, 1000, -1))
    config = EvaluationConfig(methods=("uniform", "dirichlet"), draws=2, periods=5, count=3, seed=7, alpha=1.0)
    first = WalkForwardEvaluator(data_source=source).run(config)
    second = WalkForwardEvaluator(data_source=source).run(config)
    assert first == second
    assert first["methods"]["uniform"]["ticket_count"] == 6
    detail = first["methods"]["uniform"]["draw_details"][0]
    assert int(detail["training_newest_issue"]) < int(detail["target_issue"])
    assert detail["training_count"] == 5
    assert sum(first["methods"]["uniform"]["match_distribution"].values()) == 6
    assert "vs_uniform" in first["methods"]["dirichlet"]


def test_run_rejects_insufficient_data(tmp_path):
    source = write_csv(tmp_path, issues=range(1005, 1000, -1))
    config = EvaluationConfig(methods=("uniform",), draws=2, periods=5, count=1)
    with pytest.raises(ValueError, match="数据不足"):
        WalkForwardEvaluator(data_source=source).run(config)
```

辅助函数 `write_csv` 使用 `DltDataSource` 指向临时 CSV；断言汇总字段为 `evaluated_draws`、`ticket_count`、`average_front_matches`、`average_back_matches`、`match_distribution`、`jackpot_matches` 和 `draw_details`。

- [ ] **Step 5: 实现评估运行、稳定种子和汇总**

```python
def derive_seed(seed, method, issue):
    payload = f"{seed}:{method}:{issue}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def match_ticket(ticket, target):
    target_front = set(parse_numbers(target["front_balls"], 5, 1, 35))
    target_back = set(parse_numbers(target["back_balls"], 2, 1, 12))
    front_hits = len(set(ticket["front_balls"]) & target_front)
    back_hits = len(set(ticket["back_balls"]) & target_back)
    return front_hits, back_hits
```

`run` 必须按方法和目标期生成独立 `Random`，累计整数命中总数后在最终阶段统一四舍五入到 6 位；正式结果不得包含时间戳或耗时。

- [ ] **Step 6: 运行评估器测试和已有测试**

Run: `python -m pytest tests/unit/test_walk_forward_evaluation.py tests/unit/test_probability_baselines.py -v`

Expected: PASS。

- [ ] **Step 7: 提交滚动评估器**

```bash
git add backend/evaluation tests/unit/test_walk_forward_evaluation.py
git commit -m "feat: 实现无泄漏滚动评估器"
```

### Task 3: 轻量 CLI 与根入口

**Files:**
- Create: `backend/evaluation/cli.py`
- Modify: `main.py`
- Create: `tests/unit/test_evaluation_cli.py`

- [ ] **Step 1: 写 CLI JSON 与根入口失败测试**

```python
import json
import subprocess
import sys


def test_evaluation_cli_json_output_is_parseable(capsys):
    exit_code = evaluation_main([
        "walk-forward", "--methods", "uniform,dirichlet", "--draws", "1",
        "--periods", "5", "--count", "1", "--seed", "42", "--json-output",
    ])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["config"]["methods"] == ["uniform", "dirichlet"]


def test_root_main_dispatches_evaluate_without_legacy_initialization():
    process = subprocess.run(
        [sys.executable, "main.py", "evaluate", "walk-forward", "--methods", "uniform", "--draws", "1", "--periods", "5", "--count", "1", "--json-output"],
        capture_output=True, text=True, check=False,
    )
    assert process.returncode == 0
    payload = json.loads(process.stdout)
    assert payload["methods"]["uniform"]["ticket_count"] == 1
    assert "GPU" not in process.stdout
```

- [ ] **Step 2: 运行 CLI 测试并确认 RED**

Run: `python -m pytest tests/unit/test_evaluation_cli.py -v`

Expected: FAIL，错误为缺少 `backend.evaluation.cli`。

- [ ] **Step 3: 实现 CLI 参数、输出与退出码**

```python
def parse_methods(value):
    methods = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = [method for method in methods if method not in {"uniform", "dirichlet"}]
    if not methods or unknown:
        raise argparse.ArgumentTypeError("methods 仅支持 uniform,dirichlet")
    return methods


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = EvaluationConfig(
            methods=args.methods,
            draws=args.draws,
            periods=args.periods,
            count=args.count,
            seed=args.seed,
            alpha=args.alpha,
        )
        result = WalkForwardEvaluator().run(config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"评估失败: {exc}", file=sys.stderr)
        return 2
    if args.json_output:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    else:
        print_text_report(result)
    return 0
```

`build_parser` 注册 `walk-forward`、`--methods`、`--draws`、`--periods`、`--count`、`--seed`、`--alpha` 和 `--json-output`。正整数与 1 至 100 的票数分别使用独立 argparse 类型校验器。

- [ ] **Step 4: 根入口在旧系统前分发 evaluate**

```python
def run_evaluation():
    from backend.evaluation.cli import main as evaluation_main
    sys.exit(evaluation_main(sys.argv[2:]))


# main() 中在 valid_commands 分支之前：
if first_arg == "evaluate":
    run_evaluation()
    return
```

同步根帮助的可用命令与示例；不要修改 `backend/app/main.py`，避免引入旧系统日志。

- [ ] **Step 5: 运行 CLI、评估器和根帮助测试**

Run: `python -m pytest tests/unit/test_evaluation_cli.py tests/unit/test_walk_forward_evaluation.py tests/unit/test_probability_baselines.py -v`

Expected: PASS。

Run: `python main.py evaluate walk-forward --help`

Expected: 返回码 0，帮助中包含 `--methods`、`--draws`、`--periods`、`--count`、`--seed`、`--alpha`。

- [ ] **Step 6: 提交 CLI 接入**

```bash
git add backend/evaluation main.py tests/unit/test_evaluation_cli.py
git commit -m "feat: 接入无泄漏滚动评估命令"
```

### Task 4: README 使用文档

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 在批量预测对比章节后增加独立功能说明**

文档必须包含：

```markdown
## 无泄漏滚动评估

滚动评估会把每个目标开奖期视为未知结果，只使用该期之前的数据生成票据。
它用于比较历史样本外表现，不代表未来中奖概率。

### 支持的概率基线

- `uniform`：前后区均匀随机不放回抽样，作为最低假设基线。
- `dirichlet`：对历史边际频率进行 Dirichlet/Laplace 平滑后加权抽样。

```bash
python main.py evaluate walk-forward --methods uniform,dirichlet --draws 30 --periods 500 --count 5 --seed 42 --alpha 1.0
python main.py evaluate walk-forward --draws 10 --periods 300 --count 3 --json-output
```
```

补充参数表和结果字段说明，明确“命中组合”示例 `2+1` 表示前区命中 2 个、后区命中 1 个。说明未输出奖级是因为仓库共享规则仍是 2019 九奖级，而当前数据跨越 2026 七奖级变更。

- [ ] **Step 2: 校验 README 命令与免责声明**

Run: `rg -n "evaluate walk-forward|uniform|dirichlet|不代表未来中奖概率|2019|2026" README.md`

Expected: 所有关键内容均可定位。

Run: `python main.py evaluate walk-forward --methods uniform,dirichlet --draws 2 --periods 20 --count 2 --seed 42`

Expected: 返回码 0，文本摘要包含两种方法、训练窗口与免责声明。

- [ ] **Step 3: 提交 README**

```bash
git add README.md
git commit -m "docs: 补充无泄漏滚动评估用法"
```

### Task 5: 真实流程验证、审查与最终提交

**Files:**
- Review: all files changed since `4f92c67`
- Update if needed: implementation, tests, README, design and plan documents

- [ ] **Step 1: 运行完整相关单元测试**

Run: `python -m pytest tests/unit -q`

Expected: 所有测试通过，失败数为 0。

- [ ] **Step 2: 运行真实历史滚动评估两次验证复现**

Run twice:

```bash
python main.py evaluate walk-forward --methods uniform,dirichlet --draws 10 --periods 500 --count 3 --seed 42 --alpha 1.0 --json-output
```

```bash
python main.py evaluate walk-forward --methods uniform,dirichlet --draws 10 --periods 500 --count 3 --seed 42 --alpha 1.0 --json-output > /tmp/dlt-evaluation-first.json
python main.py evaluate walk-forward --methods uniform,dirichlet --draws 10 --periods 500 --count 3 --seed 42 --alpha 1.0 --json-output > /tmp/dlt-evaluation-second.json
cmp /tmp/dlt-evaluation-first.json /tmp/dlt-evaluation-second.json
python -m json.tool /tmp/dlt-evaluation-first.json > /dev/null
```

临时文件不得加入 Git。

- [ ] **Step 3: 运行文本模式真实流程**

Run: `python main.py evaluate walk-forward --methods uniform,dirichlet --draws 10 --periods 500 --count 3 --seed 42 --alpha 1.0`

Expected: 返回码 0，显示方法摘要、命中组合、训练窗口审计信息与免责声明。

- [ ] **Step 4: 执行代码 review 自检**

Run:

```bash
git diff --stat 4f92c67..HEAD
git diff 4f92c67..HEAD
git diff --check 4f92c67..HEAD
git status --short
```

检查行为回归、未来数据泄漏、随机复现、参数错误、无关改动、敏感信息和 README 命令一致性。修复任何重要问题后重新运行 Task 5 的测试和真实流程。

- [ ] **Step 5: 请求独立代码审查并处理反馈**

以 `4f92c67` 为基线、当前 HEAD 为结果，要求审查者重点检查数据截止边界、概率抽样、确定性、CLI 退出码和测试覆盖。修复 Critical/Important 问题并重新验证。

- [ ] **Step 6: 提交审查修复（仅在有改动时）**

```bash
git add backend/evaluation main.py README.md tests/unit/test_probability_baselines.py tests/unit/test_walk_forward_evaluation.py tests/unit/test_evaluation_cli.py
git commit -m "fix: 修正滚动评估审查问题"
```

- [ ] **Step 7: 报告最终状态，不推送开发提交**

Run:

```bash
git log --oneline --decorate 4f92c67..HEAD
git status --short --branch
git rev-list --left-right --count origin/main...HEAD
```

报告当前分支、最终提交哈希、测试和真实流程结果、工作区状态。不得执行 `git push`。
