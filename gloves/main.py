"""PyCharm project entry point.

Open and Run the numbered scripts under ``scripts/`` for actual acquisition.
"""

import sys
from pathlib import Path


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except (AttributeError, ValueError):
                pass
    root = Path(__file__).resolve().parent
    print("Linker FFG 独立数据采集项目")
    print(f"项目目录: {root}")
    print()
    print("主要流程:")
    print("  scripts/01_check_ready.py       环境、连接和数据检查")
    print("  scripts/02_calibrate_ffg.py     FFG 标定")
    print("  scripts/03_collect_offline.py   离线采集并保存 H5")
    print("  scripts/04_review_data.py       自动生成质量结论")
    print()
    print("辅助入口:")
    print("  scripts/00_offline_self_test.py 无硬件自检")
    print("  scripts/05_realtime_capture.py  实时采集与最新帧接口")


if __name__ == "__main__":
    main()
