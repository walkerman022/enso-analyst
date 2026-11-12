#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""预测模型注册表：把 model_id 映射到权重、提前期、运行环境。

为什么需要注册表：
1. 权重文件路径长且易改名，硬编码在工具里迟早接错；
2. 每个权重的 lead、是否含风场分支都不同，必须随权重一起记；
3. 模型卡（训练区间、指标来源）要能和权重对得上。

配置文件是同目录的 models.json；路径是本机绝对路径，换机器只改这一份。
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


class PredictionError(RuntimeError):
    """预测链路的结构化错误：调用方据此给用户可行动提示。"""


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    path = Path(__file__).resolve().parent / "models.json"
    if not path.exists():
        raise PredictionError(f"缺少模型注册表 {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def list_models() -> list[dict[str, Any]]:
    config = load_config()
    return [
        {"model_id": model_id, "lead": item.get("lead"), "note": item.get("note", "")}
        for model_id, item in config.get("models", {}).items()
    ]


def get_model(model_id: str | None = None) -> dict[str, Any]:
    config = load_config()
    target = model_id or config.get("default_model")
    models = config.get("models", {})
    if target not in models:
        raise PredictionError(
            f"未知 model_id: {target}；可用：{', '.join(models.keys())}"
        )
    entry = dict(models[target])
    entry["model_id"] = target
    entry["config"] = config
    return entry


def month_to_index(month: str) -> int:
    """把 YYYY-MM 转成数据数组下标。数据集从 dataset_start 起逐月排列。"""
    config = load_config()
    start_year, start_month = int(config["dataset_start"][:4]), int(config["dataset_start"][5:7])
    try:
        year, mon = int(month[:4]), int(month[5:7])
    except (ValueError, IndexError) as exc:
        raise PredictionError(f"月份格式应为 YYYY-MM，收到 {month!r}") from exc
    if not 1 <= mon <= 12:
        raise PredictionError(f"月份越界：{month}")
    return (year - start_year) * 12 + (mon - start_month)


def index_to_month(index: int) -> str:
    config = load_config()
    start_year, start_month = int(config["dataset_start"][:4]), int(config["dataset_start"][5:7])
    total = start_year * 12 + (start_month - 1) + index
    return f"{total // 12:04d}-{total % 12 + 1:02d}"


def resolve_window(model_id: str | None, issued_month: str | None) -> dict[str, Any]:
    """算出输入窗口与起报月，并把越界挡在这里。"""
    model = get_model(model_id)
    config = model["config"]
    frames = int(config.get("input_frames", 12))
    data_through = config.get("data_through") or _dataset_last_month(model)
    latest_end = month_to_index(data_through)
    latest_start = latest_end - (frames - 1)
    if latest_start < 0:
        raise PredictionError("数据集长度不足一个输入窗口")

    if issued_month:
        start = month_to_index(issued_month) - (frames - 1)
        end = month_to_index(issued_month)
    else:
        start, end = latest_start, latest_end

    if start < 0:
        raise PredictionError(f"起报月过早：需要的输入窗口超出数据起点 {config['dataset_start']}")
    if end > latest_end:
        raise PredictionError(
            f"起报月超出数据范围：数据到 {data_through}，起报月最多到 {index_to_month(latest_end)}"
        )
    return {
        "start_idx": start,
        "issued_month": index_to_month(end),
        "input_window": [index_to_month(start), index_to_month(end)],
        "data_through": data_through,
        "frames": frames,
    }


def _dataset_last_month(model: dict[str, Any]) -> str:
    """读 nc 文件的最后一个月；读不到就退回配置里的值。"""
    config = model["config"]
    if config.get("data_through"):
        return str(config["data_through"])
    try:
        import xarray as xr

        path = Path(config["repo"]) / config["dataset"]
        with xr.open_dataset(path) as dataset:
            return str(dataset.time.values[-1])[:7]
    except Exception as exc:  # 读不到就明确报错，不猜
        raise PredictionError(f"无法确定数据截止月份：{type(exc).__name__}: {exc}") from exc
