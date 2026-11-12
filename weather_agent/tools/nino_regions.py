#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Niño 分区数据、ENSO 分型、历史事件与绘图。

数据源：NOAA CPC ersst5.nino.mth.91-20.ascii（月值，1991-2020 为基准）
实测格式（2026-11-04，1950-01 至 2026-06）：

     YR   MON  NINO1+2  ANOM   NINO3    ANOM   NINO4    ANOM   NINO3.4  ANOM
    1950   1   23.01   -1.55   23.56   -2.10   26.94   -1.38   24.55   -1.99

为什么用它做分型：ONI 只有 Niño3.4 一个窗口，判不了"东部型还是中部型"。
东部型看 Niño3（东太平洋），中部型看 Niño4（中太平洋），必须分区数据。

⏸ 说明：真正的 SST 距平场图需要网格数据（NetCDF），当前环境没装 xarray/netCDF4，
   所以这里用"四分区距平柱状图"代替。要画场图，见 学习/气象Agent_改动前清单.md §2。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from weather_agent import evidence
from weather_agent.cache import DataSourceError, fetch_text
from weather_agent.errors import data_source_error, tool_error

NINO_URL = "https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii"

REGIONS = ("nino12", "nino3", "nino4", "nino34")
REGION_TITLES = {
    "nino12": "Niño1+2 (0-10S, 90W-80W)",
    "nino3": "Niño3 (5S-5N, 150W-90W)",
    "nino4": "Niño4 (5S-5N, 160E-150W)",
    "nino34": "Niño3.4 (5S-5N, 170W-120W)",
}

# 分型阈值：Niño3 与 Niño4 距平之差超过这个值才判型，否则算过渡态。
# 【推测】0.2 是经验值，CPC 没有公开发布的分型阈值，这里取一个保守值避免过度判定。
TYPE_MARGIN = 0.2

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def parse_nino_regions(text: str) -> list[dict[str, Any]]:
    """把 10 列的月值表解析成结构化列表（按时间升序）。"""
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 10:
            continue
        try:
            year = int(parts[0])
            month = int(parts[1])
            values = {region: float(parts[2 + index * 2]) for index, region in enumerate(REGIONS)}
            anomalies = {region: float(parts[3 + index * 2]) for index, region in enumerate(REGIONS)}
        except ValueError:
            # 表头和脏行都走这里：表头第一列是 "YR"，int() 失败
            continue
        if not 1 <= month <= 12:
            continue
        rows.append(
            {
                "yr": year,
                "mon": month,
                "period": f"{year}-{month:02d}",
                "label": f"{year}-{month:02d}",
                "abs": values,
                "anom": anomalies,
            }
        )
    if not rows:
        raise ValueError("nino regions file parsed to zero rows; upstream format may have changed")
    return rows


def _load_rows(force: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result = fetch_text(NINO_URL, force=force)
    rows = parse_nino_regions(result["text"])
    return rows, {
        "source": NINO_URL,
        "updated_at": result.get("updated_at"),
        "from_cache": result.get("from_cache", False),
    }


def _record(rows: list[dict[str, Any]]) -> None:
    """按月份登记四个分区的距平，打上 "2026-06" 这样的对象标签。"""
    for row in rows:
        evidence.record("NINO regions/NOAA CPC", row["anom"], object_label=row["label"])


def _slim(row: dict[str, Any]) -> dict[str, Any]:
    return {"period": row["period"], **row["anom"]}


def fetch_nino_regions(months: int = 12, start_year: int | None = None, end_year: int | None = None) -> dict[str, Any]:
    """获取 Niño1+2 / 3 / 4 / 3.4 四个分区的月值距平。

    Args:
        months: 返回最近多少个月，默认 12，最多 60。
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
                    f"[LA-TOOL] no Niño region rows between {lo} and {hi}. "
                    f"Guidance: 数据从 {rows[0]['label']} 到 {rows[-1]['label']}。"
                )
        else:
            count = max(1, min(int(months), len(rows), 60))
            picked = rows[-count:]

        payload = {
            "index": "Niño region SST anomalies",
            "unit": "°C",
            "baseline": "1991-2020",
            "note": "四列依次为 Niño1+2 / 3 / 4 / 3.4 的月度距平；分型看 nino3 与 nino4 的相对大小",
            "data": [_slim(row) for row in picked],
            "latest": _slim(picked[-1]),
            "count": len(picked),
            "source": meta["source"],
            "updated_at": meta["updated_at"],
        }
        _record(picked)
        return payload
    except DataSourceError as exc:
        return data_source_error(exc, "NOAA CPC Niño regions")
    except Exception as exc:
        return tool_error(exc, "解析 Niño 分区数据")


def _find_period(rows: list[dict[str, Any]], period: str | None) -> dict[str, Any]:
    """按 "2026-06" / "2026-6" / "JAS 2026" 找对应月份；不传则取最新。"""
    if not period:
        return rows[-1]
    text = str(period).strip()
    if "-" in text and text.split("-")[0].isdigit() and len(text.split("-")[0]) == 4:
        target = text.replace("/", "-")
        for row in reversed(rows):
            if row["period"] == target:
                return row
    # 季节标签：取该季节三个月里的中间月
    from weather_agent.tools.enso_fetch import SEASONS

    parts = text.upper().split()
    if len(parts) == 2 and parts[0] in SEASONS and parts[1].isdigit():
        month = SEASONS.index(parts[0]) + 1
        target = f"{parts[1]}-{month:02d}"
        for row in reversed(rows):
            if row["period"] == target:
                return row
    raise ValueError(f"period `{period}` not found; use YYYY-MM or a season label like JAS 2026")


def classify_enso_type(period: str | None = None) -> dict[str, Any]:
    """判断某个月属于东部型（EP）还是中部型（CP）El Niño。

    Args:
        period: 目标月份，格式 "2026-06" 或季节标签 "JAS 2026"；不传则用最新月份。
    """
    try:
        rows, meta = _load_rows()
        row = _find_period(rows, period)
        anom3 = row["anom"]["nino3"]
        anom4 = row["anom"]["nino4"]
        diff = round(anom3 - anom4, 2)

        if diff >= TYPE_MARGIN:
            kind = "东部型（EP）"
            basis = "Niño3 距平明显大于 Niño4，暖中心偏东太平洋"
        elif diff <= -TYPE_MARGIN:
            kind = "中部型（CP）"
            basis = "Niño4 距平不低于 Niño3，暖中心偏中太平洋"
        else:
            kind = "过渡/混合型"
            basis = f"Niño3 与 Niño4 距平相差不足 {TYPE_MARGIN}°C"

        payload = {
            "period": row["period"],
            "type": kind,
            "basis": basis,
            "nino3_anom": anom3,
            "nino4_anom": anom4,
            "nino3_minus_nino4": diff,
            "threshold": TYPE_MARGIN,
            "note": "分型阈值是经验值，CPC 未发布官方判据；结论仅用于描述暖中心位置",
            "source": meta["source"],
            "updated_at": meta["updated_at"],
        }
        evidence.record("NINO regions/NOAA CPC", {"nino3": anom3, "nino4": anom4, "diff": diff}, object_label=row["period"])
        return payload
    except DataSourceError as exc:
        return data_source_error(exc, "NOAA CPC Niño regions")
    except Exception as exc:
        return tool_error(exc, "判定 ENSO 分型")


def _aggregate_events(rows: list[dict[str, Any]], threshold: float, min_seasons: int) -> list[dict[str, Any]]:
    """把 ONI 序列按"连续超过阈值"聚合成事件。rows 需含 seas/yr/anom/label。"""
    events: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    need = max(1, int(min_seasons))

    def flush() -> None:
        nonlocal current
        if len(current) >= need:
            peak = max(current, key=lambda item: abs(item["anom"]))
            events.append(
                {
                    "start": current[0]["label"],
                    "end": current[-1]["label"],
                    "seasons": len(current),
                    "peak": peak["anom"],
                    "peak_label": peak["label"],
                    "kind": "El Niño" if peak["anom"] > 0 else "La Niña",
                }
            )
        current = []

    for row in rows:
        if abs(row["anom"]) >= float(threshold):
            current.append(row)
        else:
            flush()
    flush()
    return events


def analog_evolution(top_n: int = 3, direction: str = "warm", offsets: list[int] | None = None) -> dict[str, Any]:
    """找历史同类事件，看它们在峰值之后如何演变（历史类比用）。

    Args:
        top_n: 取最强的几个历史事件，默认 3。
        direction: "warm" 取 El Niño，"cold" 取 La Niña。
        offsets: 峰值之后第几个季节取值，默认 [3, 6, 9, 12]。
    """
    try:
        from statistics import mean, median

        from weather_agent.tools.enso_fetch import _load_rows as load_oni_rows

        raw_rows, oni_meta = load_oni_rows()
        rows = [
            {"seas": row["seas"], "yr": row["yr"], "anom": row["anom"], "label": row["label"]}
            for row in raw_rows
        ]
        wanted = [int(item) for item in (offsets or [3, 6, 9, 12])]
        events = _aggregate_events(rows, 0.5, 3)

        ongoing_label = rows[-1]["label"]
        pool = [item for item in events if (item["peak"] > 0) == (direction != "cold")]
        pool = sorted(pool, key=lambda item: abs(item["peak"]), reverse=True)

        # 先挑"已完成"的事件（否则进行中的事件会占掉一个名额，统计样本变少），
        # 再把进行中的那条单独附上，供模型参考"目前走到哪一步"。
        completed_pool = [item for item in pool if item["end"] != ongoing_label]
        ongoing_pool = [item for item in pool if item["end"] == ongoing_label]
        pool = completed_pool[: max(1, int(top_n))] + ongoing_pool[:1]

        index_by_label = {row["label"]: index for index, row in enumerate(rows)}
        trajectories: list[dict[str, Any]] = []
        samples: dict[int, list[float]] = {offset: [] for offset in wanted}

        for event in pool:
            start_index = index_by_label[event["peak_label"]]
            ongoing = event["end"] == ongoing_label
            points: dict[str, Any] = {}
            for offset in wanted:
                position = start_index + offset
                value = rows[position]["anom"] if position < len(rows) else None
                points[f"+{offset}"] = value
                # 进行中的事件没有"后续"，不能进统计，否则会把均值拉偏
                if value is not None and not ongoing:
                    samples[offset].append(value)
            trajectories.append({
                "peak_label": event["peak_label"],
                "peak": event["peak"],
                "end": event["end"],
                "ongoing": ongoing,
                "after_peak": points,
            })

        used = [item["peak_label"] for item in trajectories if not item["ongoing"]]
        mean_after = {f"+{offset}": round(mean(values), 2) for offset, values in samples.items() if values}
        median_after = {f"+{offset}": round(median(values), 2) for offset, values in samples.items() if values}
        payload = {
            "direction": "El Niño" if direction != "cold" else "La Niña",
            "events_used": used,
            "events_excluded_ongoing": [item["peak_label"] for item in trajectories if item["ongoing"]],
            "trajectories": trajectories,
            "mean_after_peak": mean_after,
            "median_after_peak": median_after,
            "offsets_seasons": wanted,
            "note": "off=峰值季；+3 表示峰值后 3 个季节（约 9 个月）。均值只统计已完成的事件，进行中的事件单独列出",
            "source": oni_meta["source"],
            "updated_at": oni_meta["updated_at"],
        }
        for offset, value in mean_after.items():
            evidence.record("ONI/NOAA CPC", {"mean_anom": value}, object_label=f"after{offset}")
        for offset, value in median_after.items():
            evidence.record("ONI/NOAA CPC", {"median_anom": value}, object_label=f"median{offset}")
        return payload
    except DataSourceError as exc:
        return data_source_error(exc, "NOAA CPC ONI")
    except Exception as exc:
        return tool_error(exc, "计算历史类比演变")


def list_enso_history(threshold: float = 0.5, min_seasons: int = 3, top_n: int = 5) -> dict[str, Any]:
    """列出历史上的 ENSO 事件（按 ONI 连续超阈值的季节段聚合）。

    Args:
        threshold: 判事件的阈值，默认 ±0.5°C。
        min_seasons: 至少连续多少季才算一次事件，默认 3。
        top_n: 返回最强的前几个事件，默认 5。
    """
    try:
        # 注意：不能调 fetch_oni(seasons=920) —— 它单次只返回 100 季（防上下文溢出），
        # 会把历史事件截断到最近十来年。这里直接读全序列，只在返回时做聚合。
        from weather_agent.tools.enso_fetch import _load_rows as load_oni_rows

        raw_rows, oni_meta = load_oni_rows()
        rows = [
            {"seas": row["seas"], "yr": row["yr"], "anom": row["anom"], "label": row["label"]}
            for row in raw_rows
        ]

        events = _aggregate_events(rows, threshold, min_seasons)

        strongest_warm = sorted([item for item in events if item["peak"] > 0], key=lambda item: item["peak"], reverse=True)
        strongest_cold = sorted([item for item in events if item["peak"] < 0], key=lambda item: item["peak"])

        n = max(1, min(int(top_n), 10))
        payload = {
            "window": f"{rows[0]['seas']} {rows[0]['yr']} - {rows[-1]['seas']} {rows[-1]['yr']}",
            "threshold": float(threshold),
            "min_seasons": int(min_seasons),
            "warm_events": strongest_warm[:n],
            "cold_events": strongest_cold[:n],
            "total_events": len(events),
            "source": oni_meta["source"],
            "updated_at": oni_meta["updated_at"],
        }
        for item in payload["warm_events"] + payload["cold_events"]:
            evidence.record(
                "ONI/NOAA CPC",
                {"peak": item["peak"], "seasons": item["seasons"]},
                object_label=item["peak_label"],
            )
        return payload
    except DataSourceError as exc:
        return data_source_error(exc, "NOAA CPC ONI")
    except Exception as exc:
        return tool_error(exc, "汇总历史 ENSO 事件")


def plot_enso_trend(seasons: int = 60) -> dict[str, Any]:
    """画最近若干季的 ONI 曲线图，返回图片路径（图片不会传给模型）。

    Args:
        seasons: 画最近多少个季节，默认 60（15 年）。
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # 无界面后端，避免在服务器/IDE 里弹窗
        import matplotlib.pyplot as plt

        from weather_agent.tools.enso_fetch import fetch_oni

        data = fetch_oni(seasons=seasons)
        if isinstance(data, str):
            return data
        rows = data["data"]
        labels = [f"{row['seas']} {row['yr']}" for row in rows]
        values = [row["anom"] for row in rows]

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / f"oni_trend_{date.today():%Y%m%d}.png"

        fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
        ax.axhline(0.5, color="#c0392b", linestyle="--", linewidth=0.8)
        ax.axhline(-0.5, color="#2471a3", linestyle="--", linewidth=0.8)
        ax.axhline(0, color="#7f8c8d", linewidth=0.6)
        ax.fill_between(range(len(values)), values, 0.5, where=[v >= 0.5 for v in values], color="#e74c3c", alpha=0.25)
        ax.fill_between(range(len(values)), values, -0.5, where=[v <= -0.5 for v in values], color="#3498db", alpha=0.25)
        ax.plot(range(len(values)), values, color="#2c3e50", linewidth=1.4)
        ax.set_xticks(range(0, len(values), max(1, len(values) // 10)))
        ax.set_xticklabels([labels[i] for i in range(0, len(values), max(1, len(values) // 10))], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("ONI anomaly (degC)")
        ax.set_title(f"ONI trend, last {len(values)} seasons (source: NOAA CPC, updated {data['updated_at']})", fontsize=9)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

        evidence.record("ONI/NOAA CPC", {"seasons_plotted": len(values)})
        return {
            "file_path": str(path),
            "seasons_plotted": len(values),
            "first": labels[0],
            "last": labels[-1],
            "source": data["source"],
            "updated_at": data["updated_at"],
            "note": "图片已保存到本地；回答里给出路径即可，不要把数值当作从图里读出来的",
        }
    except DataSourceError as exc:
        return data_source_error(exc, "NOAA CPC ONI")
    except Exception as exc:
        return tool_error(exc, "绘制 ONI 曲线")


def plot_enso_regions(period: str | None = None) -> dict[str, Any]:
    """画某个时间点四个 Niño 分区的距平柱状图，返回图片路径。

    Args:
        period: 目标月份 "2026-06" 或季节标签 "JAS 2026"；不传则用最新月份。
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows, meta = _load_rows()
        row = _find_period(rows, period)
        values = [row["anom"][region] for region in REGIONS]
        colors = ["#c0392b" if value >= 0 else "#2471a3" for value in values]

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / f"nino_regions_{row['period']}.png"

        fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
        bars = ax.bar([REGION_TITLES[region] for region in REGIONS], values, color=colors)
        ax.axhline(0, color="#7f8c8d", linewidth=0.6)
        ax.set_ylabel("SST anomaly (degC)")
        ax.set_title(f"Niño region SST anomalies, {row['period']}", fontsize=10)
        ax.tick_params(axis="x", labelsize=7, rotation=15)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:+.2f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=8)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

        _record([row])
        return {
            "file_path": str(path),
            "period": row["period"],
            "anomalies": row["anom"],
            "source": meta["source"],
            "updated_at": meta["updated_at"],
        }
    except DataSourceError as exc:
        return data_source_error(exc, "NOAA CPC Niño regions")
    except Exception as exc:
        return tool_error(exc, "绘制分区距平图")


# ---- 工具元信息 ----

fetch_nino_regions.tool_info = {
    "tool_name": "fetch_nino_regions",
    "tool_title": "Niño 分区距平",
    "tool_description": "获取 Niño1+2 / 3 / 4 / 3.4 四个分区的月度海温距平（1991-2020 基准）。判断东部型/中部型、比较暖中心位置时用。数据从 1950 年至今。",
    "tool_params": [
        {"name": "months", "description": "返回最近多少个月，默认 12，最大 60", "type": "integer", "required": False},
        {"name": "start_year", "description": "起始年份（含）", "type": "integer", "required": False},
        {"name": "end_year", "description": "结束年份（含）", "type": "integer", "required": False},
    ],
}

classify_enso_type.tool_info = {
    "tool_name": "classify_enso_type",
    "tool_title": "ENSO 分型判定",
    "tool_description": "根据 Niño3 与 Niño4 距平的相对大小，判断某个月属于东部型（EP）、中部型（CP）还是过渡型。不要自己下结论，必须调用本工具。",
    "tool_params": [
        {"name": "period", "description": "目标月份，格式 2026-06 或季节标签 JAS 2026；不传用最新月份", "type": "string", "required": False},
    ],
}

list_enso_history.tool_info = {
    "tool_name": "list_enso_history",
    "tool_title": "历史 ENSO 事件",
    "tool_description": "把 1950 年以来的 ONI 序列按“连续超过 ±0.5°C”聚合成事件，返回最强的暖事件和冷事件（起止时间、持续季数、峰值）。回答“历史上有哪些强厄尔尼诺/拉尼娜”时用。",
    "tool_params": [
        {"name": "threshold", "description": "事件阈值，默认 0.5", "type": "number", "required": False},
        {"name": "min_seasons", "description": "至少连续多少季才算事件，默认 3", "type": "integer", "required": False},
        {"name": "top_n", "description": "各返回几个，默认 5", "type": "integer", "required": False},
    ],
}

plot_enso_trend.tool_info = {
    "tool_name": "plot_enso_trend",
    "tool_title": "ONI 曲线图",
    "tool_description": "生成最近若干季的 ONI 曲线图并保存为 PNG，返回文件路径。回答里给出路径即可，不要描述图中数值。",
    "tool_params": [
        {"name": "seasons", "description": "画最近多少季，默认 60", "type": "integer", "required": False},
    ],
}

plot_enso_regions.tool_info = {
    "tool_name": "plot_enso_regions",
    "tool_title": "分区距平柱状图",
    "tool_description": "生成某个时间点四个 Niño 分区距平的柱状图并保存为 PNG，返回文件路径。用于直观展示暖中心偏东还是偏中。",
    "tool_params": [
        {"name": "period", "description": "目标月份 2026-06 或季节标签 JAS 2026；不传用最新月份", "type": "string", "required": False},
    ],
}


analog_evolution.tool_info = {
    "tool_name": "analog_evolution",
    "tool_title": "历史类比演变",
    "tool_description": "找历史上最强的同类 ENSO 事件，统计它们峰值之后第 3/6/9/12 个季节的 ONI 均值与中位数。回答“按历史规律后续会怎么走”时用；必须同时说明样本量。",
    "tool_params": [
        {"name": "top_n", "description": "取最强的几个历史事件，默认 3", "type": "integer", "required": False},
        {"name": "direction", "description": "warm 取 El Niño，cold 取 La Niña，默认 warm", "type": "string", "required": False},
    ],
}
