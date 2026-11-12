#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""预测工具：调用训练好的 ENSO 模型，返回未来若干月的 Niño3.4 预测。

和取数工具的分界：
- fetch_* 返回**观测**，可用于判断"当前状态"；
- predict_enso 返回**模型预测**，必须标注 model_id / lead / 起报月 / 数据截止，
  不得写成"当前 ONI 是 X"。

指数口径要说清楚：模型输出是标准化距平（z-score），不是摄氏度。
"""

from __future__ import annotations

from typing import Any

from weather_agent import evidence
from weather_agent.prediction import PredictionError, run_prediction
from weather_agent.prediction.registry import list_models

FORECAST_NOTE = (
    "这是模型预测，不是观测：不要写成“当前 ONI 为 X”，"
    "引用时必须说明 model_id、起报月与数据截止月份；指数为 Niño3.4 z-score（非摄氏度）。"
)


def predict_enso(issued_month: str | None = None, model_id: str | None = None) -> dict[str, Any]:
    """用训练好的 ENSO 模型预测未来若干月的 Niño3.4 指数。

    Args:
        issued_month: 起报月（YYYY-MM）；不传则用数据支持的最新月份
        model_id: 模型标识；不传用默认模型（见 list_prediction_models）
    """
    try:
        payload = run_prediction(model_id=model_id, issued_month=issued_month)
    except PredictionError as exc:
        return f"[LA-TOOL] 预测失败：{exc}。Guidance: {FORECAST_NOTE}"
    except Exception as exc:  # 兜底，避免异常穿透到 Agent 循环
        return f"[LA-TOOL] 预测异常：{type(exc).__name__}: {exc}"

    # 预测值也记账：模型引用时必须能在本轮工具返回里找到出处。
    # 但 source 写成 forecast/...，和观测（ONI/NOAA CPC）区分开。
    source = f"forecast/{payload.get('model_id')}"
    for item in payload.get("forecast", []):
        evidence.record(source, item.get("value"), object_label=str(item.get("period")))
    if payload.get("rmse_z") is not None:
        evidence.record(source, payload["rmse_z"], object_label="hindcast_rmse")

    payload["kind"] = "forecast"
    payload["note"] = FORECAST_NOTE + payload.get("note", "")
    payload["supported_leads"] = [item["lead"] for item in list_models()]
    return payload


def list_prediction_models() -> dict[str, Any]:
    """列出可用的预测模型及其提前期。"""
    try:
        return {"models": list_models()}
    except Exception as exc:
        return f"[LA-TOOL] 读取模型注册表失败：{type(exc).__name__}: {exc}"


predict_enso.tool_info = {
    "tool_name": "predict_enso",
    "tool_title": "ENSO 预测（模型推理）",
    "tool_description": "调用训练好的物理约束 ENSO 模型，预测未来最多 12 个月的 Niño3.4 标准化距平。返回预测值、起报月、数据截止月份、模型版本与历史回测误差。注意：这是预测不是观测，不能用于回答“当前 ONI 是多少”。",
    "tool_params": [
        {"name": "issued_month", "description": "起报月 YYYY-MM，不传则用数据支持的最新月份", "type": "string", "required": False},
        {"name": "model_id", "description": "模型标识，不传用默认模型；可用 list_prediction_models 查询", "type": "string", "required": False},
    ],
}

list_prediction_models.tool_info = {
    "tool_name": "list_prediction_models",
    "tool_title": "列出预测模型",
    "tool_description": "列出可用的 ENSO 预测模型标识、支持的最大提前期与说明。",
    "tool_params": [],
}
