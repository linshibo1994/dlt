#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""大乐透开奖数据读取模块。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional


class DltDataSource:
    """读取 data/dlt_data_all.csv，按期号倒序输出。"""

    def __init__(self, data_file: Optional[str] = None):
        root = Path(__file__).resolve().parents[2]
        self.data_file = Path(data_file) if data_file else (root / "data" / "dlt_data_all.csv")

    def _parse_row(self, row: Dict[str, str]) -> Optional[Dict]:
        issue = (row.get("issue") or "").strip()
        date = (row.get("date") or "").strip()
        front = (row.get("front_balls") or "").strip().strip('"').strip("'")
        back = (row.get("back_balls") or "").strip().strip('"').strip("'")
        if not issue or not issue.isdigit() or not date or not front or not back:
            return None
        return {
            "issue": issue,
            "date": date,
            "front_balls": front,
            "back_balls": back,
        }

    def load_all(self) -> List[Dict]:
        if not self.data_file.exists():
            raise FileNotFoundError(f"开奖数据文件不存在: {self.data_file}")

        draws: List[Dict] = []
        with self.data_file.open("r", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                parsed = self._parse_row(row)
                if parsed:
                    draws.append(parsed)

        draws.sort(key=lambda item: int(item["issue"]), reverse=True)
        return draws

    def get_latest_draw(self) -> Optional[Dict]:
        draws = self.load_all()
        return draws[0] if draws else None

    def get_draw_by_issue(self, issue: str) -> Optional[Dict]:
        target = str(issue)
        for draw in self.load_all():
            if draw["issue"] == target:
                return draw
        return None
