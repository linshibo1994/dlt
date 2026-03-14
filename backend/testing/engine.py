#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试引擎：策略执行、事件推送、报告输出。"""

from __future__ import annotations

import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from random import Random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .data_source import DltDataSource
from .rules import DltRule
from .runner import PredictionRunner


EventCallback = Callable[[str, Dict[str, Any]], None]


@dataclass
class SessionConfig:
    methods: List[str]
    strategy: str
    target_prize: str
    periods_start: int
    periods_end: int
    count_start: int
    count_end: int
    max_tests: int
    parallel: bool
    workers: int
    progressive_step: int = 50
    target_issue: Optional[str] = None


class TestEngine:
    """通用测试引擎，支持大乐透规则插件。"""
    __test__ = False

    def __init__(
        self,
        rule: Optional[DltRule] = None,
        runner: Optional[PredictionRunner] = None,
        data_source: Optional[DltDataSource] = None,
        results_dir: Optional[str] = None,
        event_callback: Optional[EventCallback] = None,
        seed: Optional[int] = None,
    ):
        self.rule = rule or DltRule()
        self.runner = runner or PredictionRunner()
        self.data_source = data_source or DltDataSource()
        self.results_dir = Path(results_dir) if results_dir else Path("test_results")
        self.event_callback = event_callback

        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        self.rng = Random(seed)
        self.seed = seed

        self.lock = threading.Lock()
        self.test_results: List[Dict[str, Any]] = []
        self.winning_records: List[Dict[str, Any]] = []
        self.logs: List[str] = []

        self._progress_current = 0
        self._progress_total = 0

    def set_event_callback(self, callback: Optional[EventCallback]) -> None:
        self.event_callback = callback

    def _emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        payload = dict(payload)
        payload.setdefault("timestamp", datetime.now().isoformat())
        if self.event_callback:
            self.event_callback(event_type, payload)

    def _log(self, message: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
        with self.lock:
            self.logs.append(line)
        self._emit("log", {"message": message})

    def _tick_progress(self, method: str) -> None:
        with self.lock:
            self._progress_current += 1
            current = self._progress_current
            total = self._progress_total
        percent = (current / total * 100.0) if total > 0 else 0.0
        self._emit(
            "progress",
            {
                "current": current,
                "total": total,
                "percent": round(percent, 2),
                "method": method,
            },
        )

    def _get_target_draw(self, target_issue: Optional[str] = None) -> Dict[str, Any]:
        draw = self.data_source.get_draw_by_issue(target_issue) if target_issue else self.data_source.get_latest_draw()
        if not draw:
            raise ValueError("无法读取目标开奖期")
        return draw

    def test_single(
        self,
        method: str,
        periods: int,
        count: int,
        target_draw: Dict[str, Any],
    ) -> Dict[str, Any]:
        start = time.time()
        run_result = self.runner.run(method=method, periods=periods, count=count)

        result: Dict[str, Any] = {
            "session_id": self.session_id,
            "method": method,
            "periods": periods,
            "count": count,
            "success": run_result.get("success", False),
            "execution_time": round(time.time() - start, 4),
            "prediction_source": run_result.get("source", "unknown"),
            "target_issue": target_draw.get("issue"),
            "target_date": target_draw.get("date"),
            "target_front_balls": [int(x) for x in target_draw.get("front_balls", "").split(",") if x.strip()],
            "target_back_balls": [int(x) for x in target_draw.get("back_balls", "").split(",") if x.strip()],
            "predictions": run_result.get("predictions", []),
            "winnings": [],
            "best_prize_level": 0,
            "best_prize_name": "未中奖",
            "winning_count": 0,
            "error": run_result.get("error", ""),
            "created_at": datetime.now().isoformat(),
        }

        if result["success"]:
            prize_levels: List[int] = []
            for idx, ticket in enumerate(run_result.get("predictions", []), start=1):
                judge = self.rule.evaluate(ticket, target_draw)
                judge["prediction_index"] = idx
                if judge.get("is_winning"):
                    result["winnings"].append(judge)
                    prize_levels.append(judge.get("prize_level", 0))

            result["winning_count"] = len(result["winnings"])
            best_level = self.rule.select_best_prize(prize_levels)
            result["best_prize_level"] = best_level
            result["best_prize_name"] = self.rule.prize_levels.get(best_level, "未中奖")

        with self.lock:
            self.test_results.append(result)
            if result["winnings"]:
                self.winning_records.append(result)

        self._emit(
            "result",
            {
                "method": method,
                "periods": periods,
                "count": count,
                "success": result["success"],
                "best_prize_level": result["best_prize_level"],
                "best_prize_name": result["best_prize_name"],
                "winning_count": result["winning_count"],
                "execution_time": result["execution_time"],
            },
        )

        for winning in result["winnings"]:
            self._emit(
                "winning",
                {
                    "method": method,
                    "periods": periods,
                    "count": count,
                    "prize_level": winning.get("prize_name") or winning.get("prize_level"),
                    "prize_name": winning.get("prize_name"),
                    "match_combination": winning.get("match_combination"),
                    "predicted_fronts": winning.get("predicted_front", []),
                    "predicted_backs": winning.get("predicted_back", []),
                    "winning_fronts": result.get("target_front_balls", []),
                    "winning_backs": result.get("target_back_balls", []),
                    "matched_front_numbers": winning.get("matched_front_numbers", []),
                    "matched_back_numbers": winning.get("matched_back_numbers", []),
                    "issue": result.get("target_issue", ""),
                    "date": result.get("target_date", ""),
                },
            )

        self._tick_progress(method)
        return result

    def _run_method_random(self, method: str, cfg: SessionConfig, target_draw: Dict[str, Any]) -> Dict[str, Any]:
        self._log(f"开始随机策略测试: {method}")
        best_prize = 0
        tests_run = 0
        hit_target = False

        for _ in range(cfg.max_tests):
            periods = self.rng.randint(cfg.periods_start, cfg.periods_end)
            count = self.rng.randint(cfg.count_start, cfg.count_end)
            result = self.test_single(method, periods, count, target_draw)
            tests_run += 1

            level = result.get("best_prize_level", 0)
            if level > 0:
                best_prize = level if best_prize == 0 else min(best_prize, level)

            if self.rule.is_target_prize_hit(level, cfg.target_prize):
                hit_target = True
                self._log(f"{method} 达到目标奖级 {cfg.target_prize}，提前停止")
                break

        return {
            "method": method,
            "tests_run": tests_run,
            "hit_target": hit_target,
            "best_prize_level": best_prize,
            "best_prize_name": self.rule.prize_levels.get(best_prize, "未中奖"),
        }

    def _run_method_progressive(self, method: str, cfg: SessionConfig, target_draw: Dict[str, Any]) -> Dict[str, Any]:
        self._log(f"开始渐进策略测试: {method}")
        best_prize = 0
        tests_run = 0
        hit_target = False

        periods = cfg.periods_start
        while periods <= cfg.periods_end and tests_run < cfg.max_tests:
            count = cfg.count_start if cfg.count_start == cfg.count_end else self.rng.randint(cfg.count_start, cfg.count_end)
            result = self.test_single(method, periods, count, target_draw)
            tests_run += 1

            level = result.get("best_prize_level", 0)
            if level > 0:
                best_prize = level if best_prize == 0 else min(best_prize, level)

            if self.rule.is_target_prize_hit(level, cfg.target_prize):
                hit_target = True
                self._log(f"{method} 达到目标奖级 {cfg.target_prize}，提前停止")
                break

            periods += max(1, cfg.progressive_step)

        return {
            "method": method,
            "tests_run": tests_run,
            "hit_target": hit_target,
            "best_prize_level": best_prize,
            "best_prize_name": self.rule.prize_levels.get(best_prize, "未中奖"),
        }

    def _run_single_method(self, method: str, cfg: SessionConfig, target_draw: Dict[str, Any]) -> Dict[str, Any]:
        if cfg.strategy == "progressive":
            return self._run_method_progressive(method, cfg, target_draw)
        return self._run_method_random(method, cfg, target_draw)

    def run_session(self, cfg: SessionConfig) -> Dict[str, Any]:
        """执行一轮完整测试。"""
        start = time.time()
        target_draw = self._get_target_draw(cfg.target_issue)

        if cfg.strategy not in {"random", "progressive"}:
            raise ValueError("strategy 仅支持 random / progressive")
        if not cfg.methods:
            raise ValueError("methods 不能为空")

        self._log(f"测试会话开始: {self.session_id}")
        self._log(f"目标开奖期: {target_draw['issue']} ({target_draw['date']})")

        # 进度总数按最坏情况估算
        self._progress_current = 0
        self._progress_total = len(cfg.methods) * cfg.max_tests

        method_outcomes: List[Dict[str, Any]] = []
        if cfg.parallel and len(cfg.methods) > 1:
            workers = max(1, min(cfg.workers, len(cfg.methods)))
            self._log(f"并行执行方法数: {workers}")
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(self._run_single_method, method, cfg, target_draw): method
                    for method in cfg.methods
                }
                for future in as_completed(futures):
                    method = futures[future]
                    try:
                        method_outcomes.append(future.result())
                    except Exception as exc:
                        self._emit("error", {"method": method, "message": str(exc)})
                        method_outcomes.append(
                            {
                                "method": method,
                                "tests_run": 0,
                                "hit_target": False,
                                "best_prize_level": 0,
                                "best_prize_name": "未中奖",
                                "error": str(exc),
                            }
                        )
        else:
            for method in cfg.methods:
                method_outcomes.append(self._run_single_method(method, cfg, target_draw))

        summary = self.generate_report(method_outcomes=method_outcomes, cfg=cfg, target_draw=target_draw)
        summary["execution_time"] = round(time.time() - start, 4)
        self._emit("complete", summary)
        self._log("测试会话完成")
        return summary

    def generate_report(self, method_outcomes: List[Dict[str, Any]], cfg: SessionConfig, target_draw: Dict[str, Any]) -> Dict[str, Any]:
        """生成 JSON/TXT 报告，并计算 best_methods。"""
        method_stats: Dict[str, Dict[str, Any]] = {}
        prize_stats: Dict[str, int] = {}

        for result in self.test_results:
            method = result["method"]
            stats = method_stats.setdefault(
                method,
                {
                    "tests": 0,
                    "success_tests": 0,
                    "winning_tests": 0,
                    "winning_predictions": 0,
                    "best_prize_level": 0,
                    "avg_execution_time": 0.0,
                    "prize_breakdown": {},
                },
            )

            stats["tests"] += 1
            if result.get("success"):
                stats["success_tests"] += 1
            stats["avg_execution_time"] += float(result.get("execution_time") or 0)

            if result.get("winning_count", 0) > 0:
                stats["winning_tests"] += 1
                stats["winning_predictions"] += int(result.get("winning_count", 0))

            level = int(result.get("best_prize_level") or 0)
            if level > 0:
                if stats["best_prize_level"] == 0:
                    stats["best_prize_level"] = level
                else:
                    stats["best_prize_level"] = min(stats["best_prize_level"], level)

            for winning in result.get("winnings", []):
                w_level = int(winning.get("prize_level") or 0)
                if w_level <= 0:
                    continue
                name = self.rule.prize_levels.get(w_level, f"{w_level}等奖")
                prize_stats[name] = prize_stats.get(name, 0) + 1
                breakdown = stats["prize_breakdown"]
                breakdown[name] = breakdown.get(name, 0) + 1

        # 收尾统计
        for method, stats in method_stats.items():
            tests = max(1, stats["tests"])
            stats["avg_execution_time"] = round(stats["avg_execution_time"] / tests, 4)
            stats["winning_rate"] = round(stats["winning_tests"] / tests, 6)
            level = stats.get("best_prize_level", 0)
            stats["best_prize_name"] = self.rule.prize_levels.get(level, "未中奖")

        total_tests = len(self.test_results)
        winning_tests = len(self.winning_records)

        ranking = []
        for method, stats in method_stats.items():
            score = 0
            for prize_name, count in stats.get("prize_breakdown", {}).items():
                level = self.rule.prize_level_from_name(prize_name) or 0
                if level == 0:
                    continue
                if level == 1:
                    score += count * 1000
                elif level == 2:
                    score += count * 500
                elif level == 3:
                    score += count * 200
                else:
                    score += count * max(10, (10 - level) * 10)
            ranking.append({"method": method, "score": score, "best_prize": stats.get("best_prize_name", "未中奖")})

        ranking.sort(key=lambda item: item["score"], reverse=True)
        best_methods = ranking[:5]

        summary = {
            "session_id": self.session_id,
            "test_time": datetime.now().isoformat(),
            "seed": self.seed,
            "target_draw": target_draw,
            "config": {
                "methods": cfg.methods,
                "strategy": cfg.strategy,
                "target_prize": cfg.target_prize,
                "periods_range": [cfg.periods_start, cfg.periods_end],
                "count_range": [cfg.count_start, cfg.count_end],
                "max_tests": cfg.max_tests,
                "parallel": cfg.parallel,
                "workers": cfg.workers,
                "progressive_step": cfg.progressive_step,
            },
            "total_tests": total_tests,
            "winning_tests": winning_tests,
            "winning_rate": round((winning_tests / total_tests), 6) if total_tests else 0,
            "method_stats": method_stats,
            "prize_stats": prize_stats,
            "best_methods": best_methods,
            "method_outcomes": method_outcomes,
        }

        report_paths = self._write_report_files(summary)
        summary["report_files"] = report_paths
        return summary

    def _write_report_files(self, summary: Dict[str, Any]) -> Dict[str, str]:
        reports_dir = self.results_dir / "reports"
        logs_dir = self.results_dir / "logs"
        reports_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        json_path = reports_dir / f"test_report_{self.session_id}.json"
        txt_path = reports_dir / f"test_report_{self.session_id}.txt"
        log_path = logs_dir / f"test_{self.session_id}.log"

        with json_path.open("w", encoding="utf-8") as fp:
            json.dump(summary, fp, ensure_ascii=False, indent=2)

        with txt_path.open("w", encoding="utf-8") as fp:
            fp.write(self._build_text_report(summary))

        with log_path.open("w", encoding="utf-8") as fp:
            fp.write("\n".join(self.logs))

        return {
            "json": str(json_path),
            "txt": str(txt_path),
            "log": str(log_path),
        }

    def _build_text_report(self, summary: Dict[str, Any]) -> str:
        lines = [
            "大乐透测试系统报告",
            "=" * 60,
            f"session_id: {summary['session_id']}",
            f"test_time: {summary['test_time']}",
            f"total_tests: {summary['total_tests']}",
            f"winning_tests: {summary['winning_tests']}",
            f"winning_rate: {summary['winning_rate']:.2%}",
            "",
            "方法统计:",
        ]

        for method, stats in sorted(summary.get("method_stats", {}).items()):
            lines.append(
                f"- {method}: tests={stats['tests']}, winning_tests={stats['winning_tests']}, "
                f"best={stats['best_prize_name']}, avg_time={stats['avg_execution_time']:.2f}s"
            )

        lines.extend(["", "最佳方法:"])
        for item in summary.get("best_methods", []):
            lines.append(f"- {item['method']}: score={item['score']}, best={item['best_prize']}")

        lines.extend(["", "奖级统计:"])
        for prize_name, count in sorted(summary.get("prize_stats", {}).items()):
            lines.append(f"- {prize_name}: {count}")

        return "\n".join(lines)
