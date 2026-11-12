#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ENSO 预测：模型注册表 + 跨环境推理调用。"""

from weather_agent.prediction.inference import run_prediction
from weather_agent.prediction.registry import (
    PredictionError,
    get_model,
    index_to_month,
    list_models,
    load_config,
    month_to_index,
    resolve_window,
)

__all__ = [
    "PredictionError",
    "get_model",
    "index_to_month",
    "list_models",
    "load_config",
    "month_to_index",
    "resolve_window",
    "run_prediction",
]
