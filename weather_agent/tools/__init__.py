#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""气象工具的注册出口。"""

from weather_agent.tools.enso_fetch import (
    describe_enso_state,
    fetch_enso_index,
    fetch_oni,
    find_oni_extremes,
)
from weather_agent.tools.nino_regions import (
    analog_evolution,
    classify_enso_type,
    fetch_nino_regions,
    list_enso_history,
    plot_enso_regions,
    plot_enso_trend,
)
from weather_agent.tools.knowledge import search_knowledge
from weather_agent.tools.prediction import list_prediction_models, predict_enso
from weather_agent.tools.memory_tools import forget_preference, recall_preferences, save_preference
from weather_agent.tools.soi import fetch_soi

ALL_TOOLS = [
    fetch_oni,
    fetch_enso_index,
    describe_enso_state,
    find_oni_extremes,
    fetch_nino_regions,
    classify_enso_type,
    list_enso_history,
    plot_enso_trend,
    plot_enso_regions,
    analog_evolution,
    fetch_soi,
    save_preference,
    recall_preferences,
    forget_preference,
    search_knowledge,
    predict_enso,
    list_prediction_models,
]

__all__ = [
    "ALL_TOOLS",
    "fetch_oni",
    "fetch_enso_index",
    "describe_enso_state",
    "find_oni_extremes",
    "fetch_nino_regions",
    "classify_enso_type",
    "list_enso_history",
    "plot_enso_trend",
    "plot_enso_regions",
    "analog_evolution",
    "fetch_soi",
    "save_preference",
    "recall_preferences",
    "forget_preference",
    "search_knowledge",
    "predict_enso",
    "list_prediction_models",
]
