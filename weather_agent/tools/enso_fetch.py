#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ENSO 取数工具：ONI 指数。

数据源：NOAA CPC 的 oni.ascii.txt（纯文本，无鉴权）。
真实格式（2026-10-29 实测 920 行，1950 DJF → 2026 JAS）：

     SEAS  YR   TOTAL   ANOM
      DJF 1950  25.01  -1.32
      JFM 1950  25.36  -1.20
      ...
      JAS 2026  29.12   2.16

- SEAS 是三字母季节，代表三个月窗口，中心是中间那个月（JAS = 7-9 月，中心 8 月）
- TOTAL 是 Niño3.4 区绝对海温
- ANOM 就是 ONI（Niño3.4 区 3 个月滑动平均的距平）
- 文件本身没有时间戳，updated_at 只能取 HTTP Last-Modified

工具约定（三条，防幻觉的地基）：
1. 返回必带 source 和 updated_at；
2. 返回前把数值登记进 evidence 账本；
3. 失败不抛异常，返回 [LA- 错误串，让模型"查不到就说查不到"。
"""

from __future__ import annotations

from typing import Any

# 与 LightAgent 同级后不需要再往 sys.path 里塞路径：
# 从项目根运行（python run_weather.py / python -m weather_agent.cli）时，
# sys.path[0] 就是项目根，weather_agent 与 LightAgent 都可直接导入。
from weather_agent import evidence
from weather_agent.errors import data_source_error, tool_error
from weather_agent.cache import DataSourceError, fetch_text

ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"

# 12 个季节标签，顺序即中心月份 1..12
SEASONS = ("DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ")
_SEASON_TO_CENTER_MONTH = {season: index + 1 for index, season in enumerate(SEASONS)}

# 业务上的 ENSO 判据：连续 5 个重叠的 3 个月季节，ONI 达到 ±0.5°C
ENSO_THRESHOLD = 0.5
ENSO_CONSECUTIVE_SEASONS = 5

# 单次最多返回多少季。100 行 × 4 字段约 6400 字符，仍在 12000 的 inline 上限内。
# 再多就必须换更精确的工具（年份范围 / 极值查询），不能靠"多给点"。
MAX_ROWS_PER_CALL = 100


def parse_oni(text: str) -> list[dict[str, Any]]:
    """把 oni.ascii.txt 解析成结构化列表（按时间升序）。"""
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        parts = line.split()
        # 表头是 4 列但第一列不是合法季节标签，会被下面的判断过滤掉
        if len(parts) != 4:
            continue
        seas, year_text, total_text, anom_text = parts
        if seas not in _SEASON_TO_CENTER_MONTH:
            continue
        try:
            year = int(year_text)
            total = float(total_text)
            anom = float(anom_text)
        except ValueError:
            # 脏行跳过，不让一行坏数据毁掉整个文件
            continue
        month = _SEASON_TO_CENTER_MONTH[seas]
        rows.append(
            {
                "seas": seas,
                "yr": year,
                "month": month,
                "period": f"{year}-{month:02d}",
                "label": f"{seas} {year}",
                "total": total,
                "anom": anom,
            }
        )
    if not rows:
        raise ValueError("ONI file parsed to zero rows; upstream format may have changed")
    return rows


def _record_rows(rows: list[dict[str, Any]]) -> None:
    """按季节逐条登记观测值，并打上"属于哪个季节"的标签。

    为什么要逐条而不是整批：值-对象绑定需要知道"2.16 是 JAS 2026 的"，
    整批登记会丢掉这个归属关系，只能查"值存不存在"，查不出"年份配错"。
    """
    for row in rows:
        evidence.record(
            "ONI/NOAA CPC",
            {"total": row["total"], "anom": row["anom"]},
            object_label=row["label"],
        )


def _load_rows(force: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """下载 + 解析，返回 (行列表, 元信息)。"""
    result = fetch_text(ONI_URL, force=force)
    rows = parse_oni(result["text"])
    meta = {
        "source": ONI_URL,
        "updated_at": result.get("updated_at"),
        "from_cache": result.get("from_cache", False),
    }
    return rows, meta


def _slim(row: dict[str, Any]) -> dict[str, Any]:
    """输出给模型的行只保留必要字段。

    原因：一个季节 7 个字段约 110 字符，60 个季节就 6600 字符，
    超过 ContextCompactor 的 inline 上限后会被溢出成哈希，模型反而读不到数据
    （实测：传 920 个季节，模型只收到一条 "[spilled] sha256..." 引用）。
    """
    return {"seas": row["seas"], "yr": row["yr"], "total": row["total"], "anom": row["anom"]}


def fetch_oni(
        seasons: int = 12,
        start_year: int | None = None,
        end_year: int | None = None,
) -> dict[str, Any]:
    """获取 ONI（海洋尼诺指数）。

    Args:
        seasons: 返回最近多少个季节，默认 12；单次最多 100（MAX_ROWS_PER_CALL）。
        start_year: 起始年份（含），与 end_year 一起用于查历史区间。
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
                    f"[LA-TOOL] no ONI rows between {lo} and {hi}. "
                    f"Guidance: 数据从 {rows[0]['label']} 到 {rows[-1]['label']}，请调整年份范围。"
                )
        else:
            count = max(1, min(int(seasons), len(rows)))
            picked = rows[-count:]

        range_note = ""
        if start_year is not None or end_year is not None:
            first_year, last_year = rows[0]["yr"], rows[-1]["yr"]
            if lo < first_year or hi > last_year:
                range_note = (
                    f"请求范围 {lo}-{hi} 超出数据范围 {rows[0]['label']}-{rows[-1]['label']}，"
                    f"已按交集返回；更早或更晚的年份没有数据"
                )

        truncated = len(picked) > MAX_ROWS_PER_CALL
        if truncated:
            picked = picked[-MAX_ROWS_PER_CALL:]

        payload = {
            "index": "ONI",
            "unit": "°C",
            "note": "季节标签代表三个月窗口，中心是中间那个月（JAS=7-9月，中心8月）",
            "data": [_slim(row) for row in picked],
            "latest": _slim(picked[-1]),
            "count": len(picked),
            "truncated": truncated,
            "source": meta["source"],
            "updated_at": meta["updated_at"],
        }
        if range_note:
            payload["note"] += "；" + range_note
        if truncated:
            payload["note"] += f"；结果已截断为最近 {MAX_ROWS_PER_CALL} 季，查峰值请用 find_oni_extremes"
        _record_rows(picked)
        return payload
    except DataSourceError as exc:
        return data_source_error(exc, "NOAA CPC ONI")
    except Exception as exc:  # 解析或逻辑错误，属于我们要修的问题，不是数据源问题
        return tool_error(exc, "解析 ONI 文件")


def find_oni_extremes(
        start_year: int = 1950,
        end_year: int = 2100,
        top_n: int = 3,
) -> dict[str, Any]:
    """在指定年份区间里找 ONI 的最高/最低季节（查历史事件峰值用）。

    Args:
        start_year: 起始年份（含）。
        end_year: 结束年份（含）。
        top_n: 各返回几个极值，默认 3。
    """
    try:
        rows, meta = _load_rows()
        lo = int(start_year)
        hi = int(end_year)
        window = [row for row in rows if lo <= row["yr"] <= hi]
        if not window:
            return (
                f"[LA-TOOL] no ONI rows between {lo} and {hi}. "
                f"Guidance: 数据从 {rows[0]['label']} 到 {rows[-1]['label']}。"
            )
        n = max(1, min(int(top_n), 10))
        warmest = sorted(window, key=lambda row: row["anom"], reverse=True)[:n]
        coldest = sorted(window, key=lambda row: row["anom"])[:n]
        payload = {
            "index": "ONI",
            "unit": "°C",
            "window": f"{lo}-{hi}",
            "warmest": [_slim(row) for row in warmest],
            "coldest": [_slim(row) for row in coldest],
            "source": meta["source"],
            "updated_at": meta["updated_at"],
        }
        _record_rows(warmest)
        _record_rows(coldest)
        return payload
    except DataSourceError as exc:
        return data_source_error(exc, "NOAA CPC ONI")
    except Exception as exc:
        return tool_error(exc, "查询 ONI 极值")


def fetch_enso_index(index: str = "oni") -> dict[str, Any]:
    """获取最新的 ENSO 指数值（当前支持 oni），附带与上一季的变化量。

    Args:
        index: 指数名，目前只支持 "oni"（别名 "nino34"）。
    """
    key = (index or "oni").strip().lower()
    if key not in ("oni", "nino34", "nino3.4"):
        return (
            f"[LA-TOOL] index `{index}` is not supported. "
            f"Guidance: 当前只提供 ONI（Niño3.4 距平）。Details: supported=['oni']"
        )
    try:
        rows, meta = _load_rows()
        latest = rows[-1]
        previous = rows[-2] if len(rows) > 1 else None
        change = round(latest["anom"] - previous["anom"], 2) if previous else None
        payload = {
            "index": "ONI",
            "value": latest["anom"],
            "unit": "°C",
            "period": latest["period"],
            "label": latest["label"],
            "absolute_sst": latest["total"],
            "previous_period": previous["label"] if previous else None,
            "previous_value": round(previous["anom"], 2) if previous else None,
            "change_from_previous": change,
            "source": meta["source"],
            "updated_at": meta["updated_at"],
            "from_cache": meta["from_cache"],
        }
        # 分开记：当前值属于本期，上一期值属于上一期，环比是派生量没有对象标签。
        # 整包递归会把 previous_value 也打上本期标签，反而制造错配。
        evidence.record("ONI/NOAA CPC",
                        {"value": payload["value"], "absolute_sst": payload["absolute_sst"]},
                        object_label=payload["label"])
        if previous:
            evidence.record("ONI/NOAA CPC",
                            {"previous_value": payload["previous_value"]},
                            object_label=payload["previous_period"])
        if change is not None:
            evidence.record("ONI/NOAA CPC", {"change_from_previous": change})
        return payload
    except DataSourceError as exc:
        return data_source_error(exc, "NOAA CPC ONI")
    except Exception as exc:
        return tool_error(exc, "读取最新 ENSO 指数")


def describe_enso_state() -> dict[str, Any]:
    """判断当前 ENSO 状态（El Niño / La Niña / 中性），按官方 5 季节判据。"""
    try:
        rows, meta = _load_rows()
        tail = rows[-ENSO_CONSECUTIVE_SEASONS:]
        if len(tail) < ENSO_CONSECUTIVE_SEASONS:
            raise ValueError("not enough seasons to apply the 5-season rule")

        warm_streak = 0
        for row in reversed(rows):
            if row["anom"] >= ENSO_THRESHOLD:
                warm_streak += 1
            else:
                break
        cold_streak = 0
        for row in reversed(rows):
            if row["anom"] <= -ENSO_THRESHOLD:
                cold_streak += 1
            else:
                break

        if warm_streak >= ENSO_CONSECUTIVE_SEASONS:
            state, rule_hit = "El Niño", True
        elif cold_streak >= ENSO_CONSECUTIVE_SEASONS:
            state, rule_hit = "La Niña", True
        elif warm_streak or cold_streak:
            # 有连续但不足 5 季：属于"接近阈值但未达标"，不能直接叫厄尔尼诺
            direction = "El Niño" if warm_streak else "La Niña"
            state = f"中性（偏 {direction}，连续 {max(warm_streak, cold_streak)} 季达阈值，未满 5 季）"
            rule_hit = False
        else:
            state, rule_hit = "中性", False

        payload = {
            "state": state,
            "meets_5_season_rule": rule_hit,
            "rule": "连续 5 个重叠的 3 个月季节 ONI ≥ +0.5°C（El Niño）或 ≤ -0.5°C（La Niña）",
            "warm_streak_seasons": warm_streak,
            "cold_streak_seasons": cold_streak,
            "latest": rows[-1],
            "window": [row["label"] for row in tail],
            "source": meta["source"],
            "updated_at": meta["updated_at"],
        }
        _record_rows([payload["latest"]])
        evidence.record(
            "ONI/NOAA CPC",
            {"warm_streak": payload["warm_streak_seasons"], "cold_streak": payload["cold_streak_seasons"]},
        )
        return payload
    except DataSourceError as exc:
        return data_source_error(exc, "NOAA CPC ONI")
    except Exception as exc:
        return tool_error(exc, "判定 ENSO 状态")


# ---- 工具元信息：LightAgent 的注册接口要求手工挂这个字典（tools.py:42）----

fetch_oni.tool_info = {
    "tool_name": "fetch_oni",
    "tool_title": "ONI 指数查询",
    "tool_description": "获取 NOAA CPC 发布的 ONI（海洋尼诺指数，Niño3.4 区 3 个月滑动平均距平）。数据从 1950 年至今，单次最多返回 100 个季节；查指定年代用 start_year/end_year，查峰值用 find_oni_extremes。返回带数据源与更新时间。",
    "tool_params": [
        {
            "name": "seasons",
            "description": "返回最近多少个季节，默认 12，最大 100",
            "type": "integer",
            "required": False,
        },
        {
            "name": "start_year",
            "description": "起始年份（含），例如查 1997 年事件传 1995",
            "type": "integer",
            "required": False,
        },
        {
            "name": "end_year",
            "description": "结束年份（含），例如 2000",
            "type": "integer",
            "required": False,
        },
    ],
}

find_oni_extremes.tool_info = {
    "tool_name": "find_oni_extremes",
    "tool_title": "ONI 极值查询",
    "tool_description": "在指定年份区间内找出 ONI 最高和最低的几个季节，用于回答“某年厄尔尼诺峰值多少”这类问题。数据从 1950 年至今。",
    "tool_params": [
        {
            "name": "start_year",
            "description": "起始年份（含），默认 1950",
            "type": "integer",
            "required": False,
        },
        {
            "name": "end_year",
            "description": "结束年份（含），默认 2100",
            "type": "integer",
            "required": False,
        },
        {
            "name": "top_n",
            "description": "各返回几个极值，默认 3，最大 10",
            "type": "integer",
            "required": False,
        },
    ],
}

fetch_enso_index.tool_info = {
    "tool_name": "fetch_enso_index",
    "tool_title": "最新 ENSO 指数",
    "tool_description": "获取最新的 ONI 数值、所属季节、绝对海温以及与上一季的变化量，返回带数据源与更新时间。",
    "tool_params": [
        {
            "name": "index",
            "description": "指数名，目前只支持 oni",
            "type": "string",
            "required": False,
        },
    ],
}

describe_enso_state.tool_info = {
    "tool_name": "describe_enso_state",
    "tool_title": "ENSO 状态判定",
    "tool_description": "按官方判据（连续 5 个重叠季节 ONI 达到 ±0.5°C）判定当前处于 El Niño / La Niña / 中性，并给出连续季节数与判定窗口。",
    "tool_params": [],
}
