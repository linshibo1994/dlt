"""基于最新已知历史数据的下一期概率基线选号服务。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from random import Random

from .baselines import DirichletBaseline, UniformBaseline
from .walk_forward import SUPPORTED_METHODS, WalkForwardEvaluator, derive_seed


DISCLAIMER = "仅用于历史比较，不代表未来中奖概率"


@dataclass(frozen=True)
class PredictionConfig:
    """下一期概率基线选号配置。"""

    method: str = "dirichlet"
    periods: int = 500
    count: int = 5
    seed: int = 42
    alpha: float = 1.0

    def __post_init__(self):
        if self.method not in SUPPORTED_METHODS:
            raise ValueError("method 仅支持 uniform、dirichlet")
        if (
            isinstance(self.periods, bool)
            or not isinstance(self.periods, int)
            or self.periods <= 0
        ):
            raise ValueError("periods 必须为正整数")
        if (
            isinstance(self.count, bool)
            or not isinstance(self.count, int)
            or not 1 <= self.count <= 100
        ):
            raise ValueError("count 必须为 1..100 的正整数")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed 必须为整数")

        normalized_alpha = DirichletBaseline(self.alpha).alpha
        object.__setattr__(self, "alpha", normalized_alpha)


class BaselinePredictor:
    """使用概率基线为下一未知期生成可复现号码。"""

    def __init__(self, data_source=None):
        self.evaluator = WalkForwardEvaluator(data_source=data_source)

    def predict(self, config: PredictionConfig):
        """从最新已知开奖记录构造训练窗口并生成号码。"""
        draws = self.evaluator.load_draws()
        if len(draws) < config.periods:
            raise ValueError(
                f"数据不足：至少需要 {config.periods} 期，当前只有 {len(draws)} 期"
            )

        training = draws[: config.periods]
        latest_issue = draws[0]["issue"]
        baseline = (
            UniformBaseline()
            if config.method == "uniform"
            else DirichletBaseline(alpha=config.alpha)
        )
        rng = Random(
            derive_seed(config.seed, config.method, f"{latest_issue}:next")
        )
        tickets = baseline.generate(training, count=config.count, rng=rng)

        return {
            "config": asdict(config),
            "data": {
                "latest_issue": latest_issue,
                "training_newest_issue": training[0]["issue"],
                "training_oldest_issue": training[-1]["issue"],
                "training_periods": len(training),
                "available_draws": len(draws),
            },
            "method": config.method,
            "tickets": tickets,
            "disclaimer": DISCLAIMER,
        }
