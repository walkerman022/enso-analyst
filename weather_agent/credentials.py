#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""密钥读取。单独成模块，避免 agent.py 与 summarizer.py 互相 import。"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def _find_env_file() -> Path:
    """向上逐层找 LLM/.env，不写死层数（目录搬过家）。"""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "LLM" / ".env"
        if candidate.exists():
            return candidate
    return Path(__file__).resolve().parents[-1] / "LLM" / ".env"


DEFAULT_ENV_FILE = _find_env_file()


def load_credentials(env_file: Path | None = None) -> dict[str, str]:
    """读取 DeepSeek 的 key / base_url / 模型名。"""
    path = env_file or Path(os.getenv("WEATHER_ENV_FILE", DEFAULT_ENV_FILE))
    if path.exists():
        load_dotenv(path, override=False)
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            f"DEEPSEEK_API_KEY 未设置；应在 {path} 中配置，或通过环境变量传入。"
        )
    return {
        "api_key": api_key,
        "base_url": os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com",
        "model": os.getenv("WEATHER_MODEL") or "deepseek-chat",
    }
