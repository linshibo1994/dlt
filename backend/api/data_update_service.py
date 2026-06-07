#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""开奖数据自动更新服务。"""

from __future__ import annotations

import argparse
import os
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, Optional

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows 兼容兜底
    fcntl = None

from backend.app.core import core_modules as cm
from backend.app.utils.crawlers import incremental_update_data


FALSE_VALUES = {"0", "false", "no", "off", "disabled"}
DEFAULT_SOURCE = "zhcw"
DEFAULT_UPDATE_TIME = "00:01"

_update_lock = threading.Lock()
_scheduler: Optional["DailyLotteryDataUpdateScheduler"] = None
_scheduler_guard = threading.Lock()


@dataclass
class LotteryDataUpdateResult:
    """开奖数据更新结果。"""

    updated_count: int
    total_periods: int
    latest_issue: Optional[str]
    latest_date: Optional[str]
    source: str
    reason: str
    updated_at: str
    skipped: bool = False
    message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _bool_from_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in FALSE_VALUES


def _parse_update_time(value: str) -> tuple[int, int]:
    try:
        hour_text, minute_text = value.split(":", 1)
        hour = int(hour_text)
        minute = int(minute_text)
    except ValueError as exc:
        raise ValueError(f"更新时间格式错误，应为 HH:MM，当前值: {value}") from exc

    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"更新时间超出范围，应为 00:00-23:59，当前值: {value}")
    return hour, minute


def _lock_file_path() -> Path:
    logs_dir = Path(getattr(cm, "LOGS_DIR", "logs"))
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir / "latest_data_update.lock"


@contextmanager
def _process_update_lock(blocking: bool = False) -> Iterator[bool]:
    """跨进程文件锁，避免多 worker 或外部脚本同时写入数据文件。"""

    lock_path = _lock_file_path()
    lock_fp = lock_path.open("w", encoding="utf-8")
    acquired = False

    try:
        if fcntl is None:
            acquired = True
            yield True
            return

        flags = fcntl.LOCK_EX
        if not blocking:
            flags |= fcntl.LOCK_NB

        try:
            fcntl.flock(lock_fp.fileno(), flags)
            acquired = True
            yield True
        except BlockingIOError:
            yield False
    finally:
        if acquired and fcntl is not None:
            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)
        lock_fp.close()


def _collect_current_data_state(data_manager=None) -> tuple[int, Optional[str], Optional[str]]:
    manager = data_manager or cm.data_manager
    df = manager.get_data()
    if df is None or len(df) == 0:
        return 0, None, None

    latest = df.iloc[0]
    return len(df), str(latest.get("issue", "")), str(latest.get("date", ""))


def update_latest_lottery_data(
    *,
    data_manager=None,
    source: str = DEFAULT_SOURCE,
    reason: str = "manual",
    blocking: bool = False,
) -> LotteryDataUpdateResult:
    """增量更新最新开奖结果并刷新内存缓存。"""

    acquired_thread_lock = _update_lock.acquire(blocking=blocking)
    if not acquired_thread_lock:
        total_periods, latest_issue, latest_date = _collect_current_data_state(data_manager)
        return LotteryDataUpdateResult(
            updated_count=0,
            total_periods=total_periods,
            latest_issue=latest_issue,
            latest_date=latest_date,
            source=source,
            reason=reason,
            updated_at=datetime.now().isoformat(timespec="seconds"),
            skipped=True,
            message="已有开奖数据更新任务正在执行",
        )

    try:
        with _process_update_lock(blocking=blocking) as acquired_process_lock:
            if not acquired_process_lock:
                total_periods, latest_issue, latest_date = _collect_current_data_state(data_manager)
                return LotteryDataUpdateResult(
                    updated_count=0,
                    total_periods=total_periods,
                    latest_issue=latest_issue,
                    latest_date=latest_date,
                    source=source,
                    reason=reason,
                    updated_at=datetime.now().isoformat(timespec="seconds"),
                    skipped=True,
                    message="已有其他进程正在更新开奖数据",
                )

            cm.logger_manager.info(f"开始更新最新开奖结果，来源: {source}，触发: {reason}")
            updated_count = incremental_update_data(source=source)

            manager = data_manager or cm.data_manager
            manager.reload_data()
            total_periods, latest_issue, latest_date = _collect_current_data_state(manager)

            if updated_count > 0:
                message = f"数据更新成功，新增 {updated_count} 期数据"
            else:
                message = "数据已是最新，无需更新"

            result = LotteryDataUpdateResult(
                updated_count=updated_count,
                total_periods=total_periods,
                latest_issue=latest_issue,
                latest_date=latest_date,
                source=source,
                reason=reason,
                updated_at=datetime.now().isoformat(timespec="seconds"),
                message=message,
            )
            cm.logger_manager.info(
                f"{message}，最新期号: {latest_issue or 'N/A'}，总期数: {total_periods}"
            )
            return result
    finally:
        _update_lock.release()


class DailyLotteryDataUpdateScheduler:
    """每天固定时间执行一次开奖数据增量更新。"""

    def __init__(self, update_time: str = DEFAULT_UPDATE_TIME, source: str = DEFAULT_SOURCE):
        self.update_time = update_time
        self.source = source
        self.hour, self.minute = _parse_update_time(update_time)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="dlt-latest-data-updater",
            daemon=True,
        )
        self._thread.start()
        cm.logger_manager.info(f"已启动开奖数据每日自动更新任务，执行时间: {self.update_time}")

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        cm.logger_manager.info("已停止开奖数据每日自动更新任务")

    def _seconds_until_next_run(self) -> float:
        now = datetime.now()
        target = now.replace(hour=self.hour, minute=self.minute, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        return max(0.0, (target - now).total_seconds())

    def _run(self) -> None:
        while not self._stop_event.is_set():
            wait_seconds = self._seconds_until_next_run()
            next_run = datetime.now() + timedelta(seconds=wait_seconds)
            cm.logger_manager.info(
                f"下一次开奖数据自动更新时间: {next_run.strftime('%Y-%m-%d %H:%M:%S')}"
            )

            if self._stop_event.wait(wait_seconds):
                break

            try:
                update_latest_lottery_data(
                    source=self.source,
                    reason="scheduled",
                    blocking=False,
                )
            except Exception as exc:
                cm.logger_manager.error("开奖数据每日自动更新失败", exc)


def start_daily_update_scheduler_from_env() -> Optional[DailyLotteryDataUpdateScheduler]:
    """按环境变量启动每日自动更新任务。"""

    global _scheduler
    if not _bool_from_env("DLT_AUTO_UPDATE_ENABLED", True):
        cm.logger_manager.info("开奖数据每日自动更新任务已通过环境变量关闭")
        return None

    update_time = os.getenv("DLT_AUTO_UPDATE_TIME", DEFAULT_UPDATE_TIME).strip() or DEFAULT_UPDATE_TIME
    source = os.getenv("DLT_AUTO_UPDATE_SOURCE", DEFAULT_SOURCE).strip() or DEFAULT_SOURCE

    with _scheduler_guard:
        if _scheduler and _scheduler.running:
            return _scheduler

        try:
            _scheduler = DailyLotteryDataUpdateScheduler(update_time=update_time, source=source)
            _scheduler.start()
            return _scheduler
        except Exception as exc:
            cm.logger_manager.error("启动开奖数据每日自动更新任务失败", exc)
            _scheduler = None
            return None


def stop_daily_update_scheduler() -> None:
    """停止每日自动更新任务。"""

    global _scheduler
    with _scheduler_guard:
        if _scheduler:
            _scheduler.stop()
            _scheduler = None


def run_cli() -> int:
    parser = argparse.ArgumentParser(description="自动更新大乐透最新开奖结果")
    parser.add_argument("--source", default=os.getenv("DLT_AUTO_UPDATE_SOURCE", DEFAULT_SOURCE), help="数据源")
    parser.add_argument("--time", default=os.getenv("DLT_AUTO_UPDATE_TIME", DEFAULT_UPDATE_TIME), help="每日执行时间，格式 HH:MM")
    parser.add_argument("--once", action="store_true", help="只执行一次更新后退出")
    parser.add_argument("--daemon", action="store_true", help="常驻进程，每天固定时间执行")
    args = parser.parse_args()

    if not args.once and not args.daemon:
        args.once = True

    if args.once:
        result = update_latest_lottery_data(source=args.source, reason="script", blocking=True)
        print(result.to_dict())
        return 0

    scheduler = DailyLotteryDataUpdateScheduler(update_time=args.time, source=args.source)
    scheduler.start()
    try:
        while True:
            threading.Event().wait(3600)
    except KeyboardInterrupt:
        scheduler.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
