#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SOI（南方涛动指数）工具：大气侧证据。

数据源：NOAA CPC `https://www.cpc.ncep.noaa.gov/data/indices/soi`
实测格式（2026-11-04）：**定宽表**，同一个文件里有两块数据。

    块 1（行 0-83）  ANOMALY            —— 塔希提-达尔文气压差的距平
    块 2（行 84-167）STANDARDIZED DATA  —— 标准化 SOI，业务上常用这个

每行：前 4 字符年份，之后 12 个宽度为 6 的字段（每月一个），缺失值是 -999.9。

为什么要用定宽切片而不是 split()：
`2026 ... -1.7-999.9-999.9` 这种情况下，负号和 -999.9 会粘在一起，split 会切错。
"""

from __future__ import annotations

from typing import Any

from weather_agent import evidence
from weather_agent.cache import DataSourceError, fetch_text
from weather_agent.errors import data_source_error, tool_error

SOI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/soi"
MISSING = -999.9
FIELD_WIDTH = 6
MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")


def _split_blocks(text: str) -> list[tuple[str, list[str]]]:
    """按表头把文件切成块，返回 [(表头文字, 数据行)]。"""
    blocks: list[tuple[str, list[str]]] = []
    header = ""
    rows: list[str] = []
    for line in text.splitlines():
        if line[:4].strip().isdigit():
            rows.append(line)
        elif "STAND" in line.upper() or "YEAR" in line.upper():
            if rows:
                blocks.append((header, rows))
            header = line.strip()
            rows = []
    if rows:
        blocks.append((header, rows))
    return blocks


def parse_soi(text: str, block: str = "standardized") -> list[dict[str, Any]]:
    """解析 SOI，默认取标准化块。

    block: "standardized"（默认）或 "anomaly"。
    """
    blocks = _split_blocks(text)
    if not blocks:
        raise ValueError("SOI file has no data blocks")

    if block == "standardized":
        chosen = next((item for item in blocks if "STANDARDIZED" in item[0].upper()), blocks[-1])
    else:
        chosen = next((item for item in blocks if "ANOMALY" in item[0].upper()), blocks[0])

    rows: list[dict[str, Any]] = []
    for line in chosen[1]:
        year_text = line[:4].strip()
        if not year_text.isdigit():
            continue
        year = int(year_text)
        for index in range(12):
            raw = line[4 + index * FIELD_WIDTH: 4 + (index + 1) * FIELD_WIDTH].strip()
            if not raw:
                continue
            try:
                value = float(raw)
            except ValueError:
                continue
            if value <= MISSING + 0.05:
                continue  # 未发布的月份
            rows.append(
                {
                    "yr": year,
                    "mon": index + 1,
                    "period": f"{year}-{index + 1:02d}",
                    "label": f"{year}-{index + 1:02d}",
                    "soi": value,
                }
            )
    if not rows:
        raise ValueError("SOI parsed to zero rows; upstream format may have changed")
    return rows


def _load_rows(force: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result = fetch_text(SOI_URL, force=force)
    rows = parse_soi(result["text"])
    return rows, {
        "source": SOI_URL,
        "updated_at": result.get("updated_at"),
        "from_cache": result.get("from_cache", False),
    }


def fetch_soi(months: int = 12, start_year: int | None = None, end_year: int | None = None) -> dict[str, Any]:
    """获取标准化 SOI 月值（负值偏 El Niño，正值偏 La Niña）。

    Args:
        months: 返回最近多少个月，默认 12，最大 60。
        start_year: 起始年份（含）。
        end_year: 结束年份（含）。
    """
    try:
        rows, meta = _load_rows()
        if start_year is not None or end_year is not None:
            lo = int(start_year) if start_year is not None else 1900
            hi = int(end_year) if end_year is not None else 2100
            picked = [row for row in rows if lo <= row["yr"] <= hi]
            if not picked:
                return (
                    f"[LA-TOOL] no SOI rows between {lo} and {hi}. "
                    f"Guidance: 数据从 {rows[0]['period']} 到 {rows[-1]['period']}。"
                )
        else:
            count = max(1, min(int(months), len(rows), 60))
            picked = rows[-count:]

        latest = picked[-1]
        streak = 0
        for row in reversed(rows):
            if row["soi"] < 0:
                streak += 1
            else:
                break
        positive_streak = 0
        for row in reversed(rows):
            if row["soi"] > 0:
                positive_streak += 1
            else:
                break

        payload = {
            "index": "SOI (standardized)",
            "unit": "standard deviation",
            "note": "SOI 持续为负通常对应 El Niño（沃克环流减弱），持续为正对应 La Niña；它是大气侧指标，可与海洋侧的 ONI 交叉验证",
            "data": [{"period": row["period"], "soi": row["soi"]} for row in picked],
            "latest": {"period": latest["period"], "soi": latest["soi"]},
            "negative_streak_months": streak,
            "positive_streak_months": positive_streak,
            "count": len(picked),
            "source": meta["source"],
            "updated_at": meta["updated_at"],
        }
        for row in picked:
            evidence.record("SOI/NOAA CPC", {"soi": row["soi"]}, object_label=row["period"])
        return payload
    except DataSourceError as exc:
        return data_source_error(exc, "NOAA CPC SOI")
    except Exception as exc:
        return tool_error(exc, "解析 SOI 数据")


fetch_soi.tool_info = {
    "tool_name": "fetch_soi",
    "tool_title": "SOI 南方涛动指数",
    "tool_description": "获取标准化 SOI 月值（塔希提-达尔文气压差），用于从大气侧交叉验证 ENSO 状态。负值偏 El Niño，正值偏 La Niña。数据从 1951 年至今。",
    "tool_params": [
        {"name": "months", "description": "返回最近多少个月，默认 12，最大 60", "type": "integer", "required": False},
        {"name": "start_year", "description": "起始年份（含）", "type": "integer", "required": False},
        {"name": "end_year", "description": "结束年份（含）", "type": "integer", "required": False},
    ],
}
