from __future__ import annotations

from _bootstrap import bootstrap

bootstrap()

from ffg_capture.calibration import run_calibration
from ffg_capture.config import load_config
from ffg_capture.console import announce_artifact, configure_logging, run_script
from ffg_capture.device import FFGSource


def main() -> int:
    logger = configure_logging("INFO", log_name="ffg_calibration")

    def action() -> int:
        config = load_config()
        source = FFGSource(config.device, logger)
        try:
            source.start()
            result = run_calibration(
                source,
                config.calibration,
                device_info=source.device.info,
            )
            print()
            print("标定完成")
            for name, anchor in result.anchors.items():
                print(
                    f"  {name:5s}: stable={anchor.stable}, "
                    f"mean_var={anchor.mean_variance:.5f}, "
                    f"samples={anchor.sample_count}"
                )
            for warning in result.warnings:
                logger.warning(warning)
            announce_artifact("标定文件", result.path)
            if not result.stable:
                logger.warning("部分姿势稳定性未达到阈值，请重新标定")
                return 2
            return 0
        finally:
            source.stop()

    return run_script("FFG 标定", action, logger=logger)


if __name__ == "__main__":
    raise SystemExit(main())
