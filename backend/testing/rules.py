#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""大乐透规则插件。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


class LotteryRule(ABC):
    """彩种规则抽象。"""

    prize_levels: Dict[int, str]

    @abstractmethod
    def parse_prediction(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """把原始预测结果解析为标准票据列表。"""

    @abstractmethod
    def evaluate(self, ticket: Dict[str, Any], winning: Dict[str, Any]) -> Dict[str, Any]:
        """评估单注票据。"""

    @abstractmethod
    def validate_ticket(self, ticket: Dict[str, Any]) -> bool:
        """校验票据是否合法。"""

    @abstractmethod
    def format_ticket(self, ticket: Dict[str, Any]) -> str:
        """格式化票据。"""


class DltRule(LotteryRule):
    """大乐透 2019 新规则（9 奖级）。"""

    def __init__(self):
        self.prize_levels = {
            1: "一等奖",
            2: "二等奖",
            3: "三等奖",
            4: "四等奖",
            5: "五等奖",
            6: "六等奖",
            7: "七等奖",
            8: "八等奖",
            9: "九等奖",
        }
        # level -> [(前区命中数, 后区命中数)]
        self.prize_conditions = {
            1: [(5, 2)],
            2: [(5, 1)],
            3: [(5, 0)],
            4: [(4, 2)],
            5: [(4, 1)],
            6: [(4, 0), (3, 2)],
            7: [(3, 1)],
            8: [(3, 0), (2, 2)],
            9: [(2, 1), (1, 2), (0, 2)],
        }
        self.prize_to_level = {name: level for level, name in self.prize_levels.items()}

    def normalize_numbers(self, value: Any, expected_len: int, min_num: int, max_num: int) -> List[int]:
        """把号码统一为去重排序后的整型列表。"""
        nums: List[int] = []
        if value is None:
            return nums

        if isinstance(value, str):
            chunks = [x.strip() for x in value.replace("|", ",").replace(" ", ",").split(",") if x.strip()]
            for chunk in chunks:
                if chunk.isdigit():
                    nums.append(int(chunk))
        elif isinstance(value, Iterable):
            for item in value:
                try:
                    nums.append(int(item))
                except (TypeError, ValueError):
                    continue

        nums = sorted(set(nums))
        if len(nums) != expected_len:
            return []
        if any(num < min_num or num > max_num for num in nums):
            return []
        return nums

    def normalize_ticket(self, ticket: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """把不同字段名格式统一为标准票据。"""
        front = (
            ticket.get("front_balls")
            or ticket.get("front")
            or ticket.get("front_balls_list")
            or ticket.get("predicted_front")
        )
        back = (
            ticket.get("back_balls")
            or ticket.get("back")
            or ticket.get("back_balls_list")
            or ticket.get("predicted_back")
        )

        front_numbers = self.normalize_numbers(front, expected_len=5, min_num=1, max_num=35)
        back_numbers = self.normalize_numbers(back, expected_len=2, min_num=1, max_num=12)
        if not front_numbers or not back_numbers:
            return None

        return {
            "front_balls": front_numbers,
            "back_balls": back_numbers,
            "front_balls_text": ",".join(f"{n:02d}" for n in front_numbers),
            "back_balls_text": ",".join(f"{n:02d}" for n in back_numbers),
        }

    def parse_prediction(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """解析预测结果，优先结构化字段。"""
        predictions: List[Any] = []

        if isinstance(payload, dict):
            if isinstance(payload.get("predictions"), list):
                predictions = payload["predictions"]
            elif isinstance(payload.get("data"), dict) and isinstance(payload["data"].get("predictions"), list):
                predictions = payload["data"]["predictions"]
            elif isinstance(payload.get("tickets"), list):
                predictions = payload["tickets"]

        normalized: List[Dict[str, Any]] = []
        seen = set()
        for item in predictions:
            if not isinstance(item, dict):
                continue
            ticket = self.normalize_ticket(item)
            if not ticket:
                continue
            key = (tuple(ticket["front_balls"]), tuple(ticket["back_balls"]))
            if key in seen:
                continue
            seen.add(key)
            normalized.append(ticket)

        return normalized

    def validate_ticket(self, ticket: Dict[str, Any]) -> bool:
        return self.normalize_ticket(ticket) is not None

    def format_ticket(self, ticket: Dict[str, Any]) -> str:
        normalized = self.normalize_ticket(ticket)
        if not normalized:
            return "<非法票据>"
        front = " ".join(f"{n:02d}" for n in normalized["front_balls"])
        back = " ".join(f"{n:02d}" for n in normalized["back_balls"])
        return f"{front} + {back}"

    def resolve_prize_level(self, front_hits: int, back_hits: int) -> int:
        for level in sorted(self.prize_conditions.keys()):
            if (front_hits, back_hits) in self.prize_conditions[level]:
                return level
        return 0

    def evaluate(self, ticket: Dict[str, Any], winning: Dict[str, Any]) -> Dict[str, Any]:
        normalized_ticket = self.normalize_ticket(ticket)
        normalized_winning = self.normalize_ticket(winning)

        if not normalized_ticket or not normalized_winning:
            return {
                "is_winning": False,
                "prize_level": 0,
                "prize_name": "未中奖",
                "error": "票据或开奖号码格式不合法",
            }

        ticket_front = set(normalized_ticket["front_balls"])
        ticket_back = set(normalized_ticket["back_balls"])
        winning_front = set(normalized_winning["front_balls"])
        winning_back = set(normalized_winning["back_balls"])

        front_hits = len(ticket_front & winning_front)
        back_hits = len(ticket_back & winning_back)
        prize_level = self.resolve_prize_level(front_hits, back_hits)

        return {
            "is_winning": prize_level > 0,
            "prize_level": prize_level,
            "prize_name": self.prize_levels.get(prize_level, "未中奖"),
            "front_matches": front_hits,
            "back_matches": back_hits,
            "match_combination": f"{front_hits}+{back_hits}",
            "predicted_front": sorted(ticket_front),
            "predicted_back": sorted(ticket_back),
            "winning_front": sorted(winning_front),
            "winning_back": sorted(winning_back),
            "matched_front_numbers": sorted(ticket_front & winning_front),
            "matched_back_numbers": sorted(ticket_back & winning_back),
        }

    def prize_level_from_name(self, name: str) -> Optional[int]:
        return self.prize_to_level.get(name)

    def is_target_prize_hit(self, best_prize_level: int, target_prize_name: str) -> bool:
        if best_prize_level <= 0:
            return False
        target_level = self.prize_level_from_name(target_prize_name)
        if target_level is None:
            return False
        # 奖级数值越小表示奖级越高
        return best_prize_level <= target_level

    @staticmethod
    def select_best_prize(prize_levels: Sequence[int]) -> int:
        valid = [level for level in prize_levels if isinstance(level, int) and level > 0]
        return min(valid) if valid else 0

    def list_target_prizes(self) -> List[str]:
        return [self.prize_levels[level] for level in sorted(self.prize_levels.keys())]
