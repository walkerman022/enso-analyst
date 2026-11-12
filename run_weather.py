#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PyCharm 运行入口：右键 → Run 'run_weather' 即可。

为什么需要这个文件：
    直接运行 weather_agent/cli.py 时，Python 把 sys.path[0] 设成 weather_agent/ 目录，
    `from weather_agent import evidence` 会因为找不到包而报 ModuleNotFoundError。
    放在项目根运行，sys.path[0] 就是项目根，`weather_agent` 与 `LightAgent` 都能导入。

三种用法：
    1. 右键直接 Run          → 跑下面 DEMO_QUERY 这一条
    2. 命令行带问题           → python run_weather.py "1997 年厄尔尼诺峰值多少"
    3. python run_weather.py -i → 交互模式（PyCharm 里需勾选 Emulate terminal in output console）
"""

from __future__ import annotations

import sys
from pathlib import Path

# 保证从任何工作目录启动都能找到包（PyCharm 的 Run 有时会用别的 cwd）
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from weather_agent.cli import main  # noqa: E402

# 想换问题直接改这一行，改完再点 Run
DEMO_QUERY = "你是从哪获取厄尔尼诺数据的？"


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(main(sys.argv[1:]))
    raise SystemExit(main([DEMO_QUERY]))
