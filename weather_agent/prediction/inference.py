#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""推理调用层：用子进程在模型环境里跑 infer_once.py，把 JSON 拿回来。

为什么用子进程而不是直接 import：
- Agent 环境（llm）没有 torch，模型环境（dy_enso）有；
- 进程隔离：模型崩了不会把 Agent 进程带下去；
- 模型侧可独立命令行验证，排障不用经过 Agent。

约定：infer_once.py 用 --json-only，stdout 只输出一行 JSON，日志走 stderr。
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from weather_agent.prediction.registry import PredictionError, get_model, resolve_window


def run_prediction(
        model_id: str | None = None,
        issued_month: str | None = None,
        *,
        timeout: int = 300,
        cpu: bool = True,
) -> dict[str, Any]:
    model = get_model(model_id)
    config = model["config"]
    window = resolve_window(model["model_id"], issued_month)

    command = [
        config["python"],
        "infer_once.py",
        "--ckpt", model["checkpoint"],
        "--nc", config["dataset"],
        "--start-idx", str(window["start_idx"]),
        "--input-frames", str(window["frames"]),
        "--lead", str(int(model["lead"])),
        "--json-only",
    ]
    if cpu:
        command.append("--cpu")

    try:
        completed = subprocess.run(
            command,
            cwd=config["repo"],
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired as exc:
        raise PredictionError(f"推理超时（>{timeout}s）；模型环境可能被占用") from exc
    except FileNotFoundError as exc:
        raise PredictionError(f"找不到模型环境解释器：{config['python']}") from exc

    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()[-3:]
        raise PredictionError("推理失败：" + " | ".join(tail) if tail else "推理失败")

    payload = _parse_json(completed.stdout)
    payload["model_id"] = model["model_id"]
    payload["model_note"] = model.get("note", "")
    payload["input_window"] = window["input_window"]
    payload["data_through"] = window["data_through"]
    return payload


def _parse_json(stdout: str) -> dict[str, Any]:
    for line in reversed((stdout or "").strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    raise PredictionError("推理输出里没有可解析的 JSON；检查 infer_once.py 是否被改动")


def model_paths(model_id: str | None = None) -> dict[str, str]:
    model = get_model(model_id)
    config = model["config"]
    checkpoint = Path(config["repo"]) / model["checkpoint"]
    return {"checkpoint": str(checkpoint), "exists": str(checkpoint.exists()).lower()}
