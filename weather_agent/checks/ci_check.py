#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CI 门禁：跑代码级断言，并与基线对比。

为什么要有基线：
只跑“当前是否全过”发现不了退化——昨天 138 条、今天新增 3 条失败但总数也涨了，
只看 failed=0 会漏。基线记录“上次通过多少条”，少于它就算回归。

用法：
    python weather_agent/checks/ci_check.py            # 跑并对比
    python weather_agent/checks/ci_check.py --update   # 跑并更新基线

退出码：0 通过；1 回归或失败；2 执行异常（无法解析输出）。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
BASELINE = Path(__file__).resolve().parent / "baseline.json"
LATEST = Path(__file__).resolve().parent / "latest.json"
SUMMARY = re.compile(r"合计\s*(\d+)\s*项，通过\s*(\d+)，失败\s*(\d+)")


def run_checks(timeout: int = 600) -> tuple[int, int, int, str]:
    completed = subprocess.run(
        [sys.executable, "weather_agent/checks/check_stage1.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    matches = SUMMARY.findall(output)
    if not matches:
        print(output[-2000:])
        raise RuntimeError("未能从输出中解析出断言统计")
    total, passed, failed = (int(value) for value in matches[-1])
    return total, passed, failed, output


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="weather_agent CI 门禁")
    parser.add_argument("--update", action="store_true", help="把本次结果写为基线")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args(argv)

    try:
        total, passed, failed, _ = run_checks(args.timeout)
    except Exception as exc:
        print(f"[CI] 执行失败：{type(exc).__name__}: {exc}")
        return 2

    record = {
        "total": total,
        "passed": passed,
        "failed": failed,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    LATEST.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[CI] 断言 {total} 项，通过 {passed}，失败 {failed}")
    if args.update:
        BASELINE.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[CI] 基线已更新：{BASELINE}")
        return 0 if failed == 0 else 1

    baseline = read_json(BASELINE)
    if baseline is None:
        print("[CI] 没有基线，先执行 --update 生成一份")
        return 0 if failed == 0 else 1

    regressions: list[str] = []
    if failed > 0:
        regressions.append(f"本次有 {failed} 条失败")
    if passed < int(baseline.get("passed", 0)):
        regressions.append(f"通过数下降：{baseline.get('passed')} → {passed}")
    if total < int(baseline.get("total", 0)):
        regressions.append(f"断言总数下降：{baseline.get('total')} → {total}（可能误删了检查）")

    if regressions:
        print("[CI] 回归：")
        for item in regressions:
            print(f"  - {item}")
        return 1

    print(f"[CI] 与基线一致或更好（基线 {baseline.get('passed')}/{baseline.get('total')}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
