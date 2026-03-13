#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""预测执行器：优先结构化结果，失败回退子进程解析。"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


class PredictionRunner:
    """统一预测执行入口。"""
    COMPOUND_LIKE_METHODS = {
        "compound",
        "duplex",
        "markov_compound",
        "nine_models_compound",
        "highly_integrated",
    }

    def __init__(
        self,
        timeout_seconds: int = 120,
        retries: int = 1,
        prefer_direct: bool = True,
        fallback_subprocess: bool = True,
        predictor_service: Any = None,
        project_root: Optional[str] = None,
    ):
        root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
        self.project_root = root
        self.main_script = self.project_root / "main.py"
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.prefer_direct = prefer_direct
        self.fallback_subprocess = fallback_subprocess
        self._predictor_service = predictor_service
        self.max_expand_predictions = 5000

    @property
    def predictor_service(self):
        if self._predictor_service is None:
            from backend.api.dependencies import PredictorService

            self._predictor_service = PredictorService()
        return self._predictor_service

    def run(self, method: str, periods: int, count: int, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """执行预测并返回标准化结果。"""
        extra = extra or {}
        attempts = []
        if self.prefer_direct:
            attempts.append(self._run_direct)
        if self.fallback_subprocess:
            attempts.append(self._run_subprocess)

        errors: List[str] = []
        for attempt in attempts:
            for retry in range(self.retries + 1):
                result = attempt(method=method, periods=periods, count=count, extra=extra, retry=retry)
                if result.get("success") and result.get("predictions"):
                    return result
                error = result.get("error")
                if error:
                    errors.append(f"{attempt.__name__}[{retry}]: {error}")

        return {
            "success": False,
            "method": method,
            "periods": periods,
            "count": count,
            "predictions": [],
            "error": " | ".join(errors) if errors else "预测失败",
        }

    def _build_payload(self, method: str, periods: int, count: int, extra: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "method": method,
            "periods": int(periods),
            "count": int(count),
            "compound_mode": False,
            "acceleration": extra.get("acceleration", "cpu"),
            "cpu_threads": extra.get("cpu_threads", 1),
            "fallback_enabled": True,
        }
        for key in ("missing_mode", "strategy", "performance_mode", "training_intensity"):
            if key in extra:
                payload[key] = extra[key]
        if method in self.COMPOUND_LIKE_METHODS and "compound_mode" not in extra:
            payload["compound_mode"] = True
        return payload

    def _run_direct(self, method: str, periods: int, count: int, extra: Dict[str, Any], retry: int = 0) -> Dict[str, Any]:
        start = time.time()
        try:
            payload = self._build_payload(method, periods, count, extra)
            data = self.predictor_service.predict(payload)
            predictions = self._extract_predictions_from_payload(data, count_hint=count, method=method)
            return {
                "success": bool(predictions),
                "method": method,
                "periods": periods,
                "count": count,
                "mode": data.get("mode", "direct"),
                "predictions": predictions,
                "raw_output": data,
                "execution_time": time.time() - start,
                "source": "direct",
                "error": "无可用预测结果" if not predictions else "",
            }
        except Exception as exc:
            return {
                "success": False,
                "method": method,
                "periods": periods,
                "count": count,
                "predictions": [],
                "execution_time": time.time() - start,
                "source": "direct",
                "error": f"直接调用失败: {exc}",
            }

    def _run_subprocess(self, method: str, periods: int, count: int, extra: Dict[str, Any], retry: int = 0) -> Dict[str, Any]:
        start = time.time()
        cmd = [
            sys.executable,
            str(self.main_script),
            "predict",
            "-m",
            method,
            "-p",
            str(periods),
            "-c",
            str(count),
            "--json-output",
        ]

        # 重试时降级加速配置，避免 GPU/并行导致阻塞
        if retry > 0:
            cmd.extend(["--acceleration", "cpu", "--cpu-threads", "1"])
        elif extra.get("acceleration"):
            cmd.extend(["--acceleration", str(extra["acceleration"])])

        try:
            run_kwargs = {
                "cwd": str(self.project_root),
                "capture_output": True,
                "text": True,
                "encoding": "utf-8",
            }
            if self.timeout_seconds > 0:
                run_kwargs["timeout"] = self.timeout_seconds
            proc = subprocess.run(cmd, **run_kwargs)
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "method": method,
                "periods": periods,
                "count": count,
                "predictions": [],
                "execution_time": time.time() - start,
                "source": "subprocess",
                "error": f"命令执行超时(>{self.timeout_seconds}s)",
            }
        except Exception as exc:
            return {
                "success": False,
                "method": method,
                "periods": periods,
                "count": count,
                "predictions": [],
                "execution_time": time.time() - start,
                "source": "subprocess",
                "error": f"命令执行失败: {exc}",
            }

        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        predictions = self.parse_cli_output(stdout)
        success = proc.returncode == 0 and bool(predictions)

        error = ""
        if not success:
            if proc.returncode != 0:
                error = f"返回码{proc.returncode}: {stderr.strip() or '未知错误'}"
            else:
                error = "预测输出解析失败"

        return {
            "success": success,
            "method": method,
            "periods": periods,
            "count": count,
            "predictions": predictions,
            "raw_output": stdout,
            "stderr": stderr,
            "execution_time": time.time() - start,
            "source": "subprocess",
            "command": " ".join(cmd),
            "error": error,
        }

    def _normalize_predictions(self, predictions: List[Any]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        seen = set()

        for item in predictions:
            if not isinstance(item, dict):
                continue
            front = item.get("front_balls") or item.get("front")
            back = item.get("back_balls") or item.get("back")
            front_nums = self._parse_number_blob(front, expected=5, min_num=1, max_num=35)
            back_nums = self._parse_number_blob(back, expected=2, min_num=1, max_num=12)
            if not front_nums or not back_nums:
                continue
            key = (tuple(front_nums), tuple(back_nums))
            if key in seen:
                continue
            seen.add(key)
            normalized.append(
                {
                    "front_balls": front_nums,
                    "back_balls": back_nums,
                    "front_balls_text": ",".join(f"{n:02d}" for n in front_nums),
                    "back_balls_text": ",".join(f"{n:02d}" for n in back_nums),
                }
            )
        return normalized

    def parse_cli_output(self, stdout: str) -> List[Dict[str, Any]]:
        """优先 JSON 解析，再降级为文本正则解析。"""
        parsed = self._parse_json_from_output(stdout)
        if parsed:
            return parsed

        parsed = self._parse_text_patterns(stdout)
        return parsed

    def _parse_json_from_output(self, stdout: str) -> List[Dict[str, Any]]:
        # 1) 逐行查找 JSON 对象
        for line in stdout.splitlines():
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                normalized = self._extract_predictions_from_payload(data, count_hint=50)
                if normalized:
                    return normalized

        # 2) 在整段文本里寻找 "predictions" JSON 片段
        blocks = re.findall(r"\{[\s\S]*?\}", stdout)
        for block in blocks:
            try:
                data = json.loads(block)
            except Exception:
                continue
            if isinstance(data, dict):
                normalized = self._extract_predictions_from_payload(data, count_hint=50)
                if normalized:
                    return normalized

        return []

    def _extract_predictions_from_payload(
        self,
        payload: Any,
        count_hint: int = 1,
        method: str = "",
    ) -> List[Dict[str, Any]]:
        if not isinstance(payload, dict):
            return []

        predictions: List[Dict[str, Any]] = []
        limit = self._resolve_expand_limit(count_hint)

        candidates: List[Any] = []
        if isinstance(payload.get("predictions"), list):
            candidates.extend(payload["predictions"])
        if isinstance(payload.get("data"), dict) and isinstance(payload["data"].get("predictions"), list):
            candidates.extend(payload["data"]["predictions"])

        # 先处理标准预测项（含 details 中的复式/胆拖）
        for item in candidates:
            predictions.extend(self._extract_predictions_from_item(item, limit=limit, method=method))
            if len(predictions) >= limit:
                return self._deduplicate(predictions)[:limit]

        # 再处理 compound / duplex 顶层结构
        predictions.extend(self._expand_compound_payload(payload.get("compound"), limit=limit, method=method))
        if len(predictions) < limit:
            predictions.extend(self._expand_compound_payload(payload, limit=limit, method=method))
        if len(predictions) < limit:
            predictions.extend(self._expand_duplex_payload(payload, limit=limit, method=method))

        return self._deduplicate(predictions)[:limit]

    def _extract_predictions_from_item(self, item: Any, limit: int, method: str = "") -> List[Dict[str, Any]]:
        if not isinstance(item, dict):
            return []

        result: List[Dict[str, Any]] = []

        # 标准 5+2 单注
        ticket = self._build_ticket(
            front_blob=item.get("front_balls") or item.get("front"),
            back_blob=item.get("back_balls") or item.get("back"),
            method=item.get("method") or method,
            confidence=item.get("confidence"),
        )
        if ticket:
            result.append(ticket)
            return result

        # details 中可能携带复式/胆拖结构
        details = item.get("details")
        if isinstance(details, dict):
            result.extend(self._expand_compound_payload(details, limit=limit, method=item.get("method") or method))
            if len(result) < limit:
                result.extend(self._expand_duplex_payload(details, limit=limit, method=item.get("method") or method))

        # item 本身可能是复式/胆拖结构
        if len(result) < limit:
            result.extend(self._expand_compound_payload(item, limit=limit, method=item.get("method") or method))
        if len(result) < limit:
            result.extend(self._expand_duplex_payload(item, limit=limit, method=item.get("method") or method))

        return self._deduplicate(result)[:limit]

    def _expand_compound_payload(self, payload: Any, limit: int, method: str = "") -> List[Dict[str, Any]]:
        if not isinstance(payload, dict):
            return []

        result: List[Dict[str, Any]] = []
        payload_method = payload.get("method") or method

        # 情况1：直接携带 combinations 列表
        combos = payload.get("combinations")
        if isinstance(combos, list):
            for combo in combos:
                if not isinstance(combo, dict):
                    continue
                ticket = self._build_ticket(
                    front_blob=combo.get("front_balls") or combo.get("front"),
                    back_blob=combo.get("back_balls") or combo.get("back"),
                    method=payload_method,
                    confidence=payload.get("confidence"),
                )
                if ticket:
                    result.append(ticket)
                    if len(result) >= limit:
                        return result

        # 情况2：前后区大于 5+2 的复式集合，展开组合
        front_pool = self._parse_number_pool(payload.get("front_balls"), min_num=1, max_num=35)
        back_pool = self._parse_number_pool(payload.get("back_balls"), min_num=1, max_num=12)
        if len(front_pool) >= 5 and len(back_pool) >= 2:
            for front_ticket in combinations(front_pool, 5):
                for back_ticket in combinations(back_pool, 2):
                    ticket = self._build_ticket(
                        front_blob=list(front_ticket),
                        back_blob=list(back_ticket),
                        method=payload_method,
                        confidence=payload.get("confidence"),
                    )
                    if ticket:
                        result.append(ticket)
                        if len(result) >= limit:
                            return result

        return result

    def _expand_duplex_payload(self, payload: Any, limit: int, method: str = "") -> List[Dict[str, Any]]:
        if not isinstance(payload, dict):
            return []

        front_dan = self._parse_number_pool(payload.get("front_dan"), min_num=1, max_num=35)
        front_tuo = self._parse_number_pool(payload.get("front_tuo"), min_num=1, max_num=35)
        back_dan = self._parse_number_pool(payload.get("back_dan"), min_num=1, max_num=12)
        back_tuo = self._parse_number_pool(payload.get("back_tuo"), min_num=1, max_num=12)

        if not front_dan and not front_tuo:
            return []
        if not back_dan and not back_tuo:
            return []

        # 胆码与拖码去重（胆码优先）
        front_tuo = [num for num in front_tuo if num not in set(front_dan)]
        back_tuo = [num for num in back_tuo if num not in set(back_dan)]

        need_front = 5 - len(front_dan)
        need_back = 2 - len(back_dan)
        if need_front < 0 or need_back < 0:
            return []
        if need_front > len(front_tuo) or need_back > len(back_tuo):
            return []

        result: List[Dict[str, Any]] = []
        payload_method = payload.get("method") or method
        for front_extra in combinations(front_tuo, need_front):
            for back_extra in combinations(back_tuo, need_back):
                front_ticket = sorted(front_dan + list(front_extra))
                back_ticket = sorted(back_dan + list(back_extra))
                ticket = self._build_ticket(
                    front_blob=front_ticket,
                    back_blob=back_ticket,
                    method=payload_method,
                    confidence=payload.get("confidence"),
                )
                if ticket:
                    result.append(ticket)
                    if len(result) >= limit:
                        return result
        return result

    def _resolve_expand_limit(self, count_hint: int) -> int:
        safe_hint = max(1, int(count_hint or 1))
        # 单方法最多展开 count_hint*300 注，避免超大组合撑爆内存
        return min(self.max_expand_predictions, max(50, safe_hint * 300))

    def _build_ticket(self, front_blob: Any, back_blob: Any, method: str = "", confidence: Any = None) -> Optional[Dict[str, Any]]:
        front = self._parse_number_blob(front_blob, expected=5, min_num=1, max_num=35)
        back = self._parse_number_blob(back_blob, expected=2, min_num=1, max_num=12)
        if not front or not back:
            return None
        ticket = {
            "front_balls": front,
            "back_balls": back,
            "front_balls_text": ",".join(f"{n:02d}" for n in front),
            "back_balls_text": ",".join(f"{n:02d}" for n in back),
        }
        if method:
            ticket["method"] = method
        if confidence is not None:
            ticket["confidence"] = confidence
        return ticket

    @staticmethod
    def _parse_number_pool(blob: Any, min_num: int, max_num: int) -> List[int]:
        values: List[int] = []
        if blob is None:
            return values

        if isinstance(blob, str):
            values = [int(token) for token in re.findall(r"\d{1,2}", blob)]
        elif isinstance(blob, Iterable) and not isinstance(blob, (bytes, bytearray, dict)):
            for item in blob:
                try:
                    values.append(int(item))
                except (TypeError, ValueError):
                    continue
        else:
            return []

        return sorted({num for num in values if min_num <= num <= max_num})

    def _parse_text_patterns(self, stdout: str) -> List[Dict[str, Any]]:
        predictions: List[Dict[str, Any]] = []

        # 行内格式：第 1 注: 01 02 03 04 05 + 06 07
        inline_pattern = re.compile(
            r"第\s*\d+\s*注[^:：]*[:：]\s*(\d{1,2}(?:\s+\d{1,2}){4})\s*\+\s*(\d{1,2}(?:\s+\d{1,2}))"
        )
        # 行内格式：前区: 01,02,03,04,05 后区: 06,07
        side_pattern = re.compile(r"前区\s*[:：]\s*([0-9,\s]+)\s*后区\s*[:：]\s*([0-9,\s]+)")

        pending_front: Optional[str] = None
        for raw_line in stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            m = inline_pattern.search(line)
            if m:
                self._append_ticket(predictions, m.group(1), m.group(2))
                continue

            m = side_pattern.search(line)
            if m:
                self._append_ticket(predictions, m.group(1), m.group(2))
                continue

            # 多行格式：
            # 前区: xx xx xx xx xx
            # 后区: xx xx
            if "前区" in line and ":" in line:
                pending_front = line.split(":", 1)[1]
                continue
            if pending_front is not None and "后区" in line and ":" in line:
                back_part = line.split(":", 1)[1]
                self._append_ticket(predictions, pending_front, back_part)
                pending_front = None

        return self._deduplicate(predictions)

    def _append_ticket(self, container: List[Dict[str, Any]], front_blob: Any, back_blob: Any) -> None:
        front = self._parse_number_blob(front_blob, expected=5, min_num=1, max_num=35)
        back = self._parse_number_blob(back_blob, expected=2, min_num=1, max_num=12)
        if not front or not back:
            return
        container.append(
            {
                "front_balls": front,
                "back_balls": back,
                "front_balls_text": ",".join(f"{n:02d}" for n in front),
                "back_balls_text": ",".join(f"{n:02d}" for n in back),
            }
        )

    @staticmethod
    def _parse_number_blob(blob: Any, expected: int, min_num: int, max_num: int) -> List[int]:
        values: List[int] = []
        if blob is None:
            return []

        if isinstance(blob, str):
            tokens = re.findall(r"\d{1,2}", blob)
            values = [int(token) for token in tokens]
        elif isinstance(blob, list):
            for item in blob:
                try:
                    values.append(int(item))
                except (TypeError, ValueError):
                    continue
        else:
            return []

        values = sorted(set(values))
        if len(values) != expected:
            return []
        if any(value < min_num or value > max_num for value in values):
            return []
        return values

    @staticmethod
    def _deduplicate(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        unique: List[Dict[str, Any]] = []
        seen = set()
        for item in predictions:
            key = (tuple(item.get("front_balls", [])), tuple(item.get("back_balls", [])))
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        return unique
