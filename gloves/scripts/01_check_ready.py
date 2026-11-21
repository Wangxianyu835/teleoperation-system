from __future__ import annotations

import platform
import sys
import traceback
from importlib import util

from _bootstrap import bootstrap

bootstrap()

from ffg_capture.config import load_config, project_path
from ffg_capture.console import (
    announce_artifact,
    configure_logging,
    run_script,
    timestamp_label,
    write_json,
)
from ffg_capture.device import FFGSource, discover_serial_ports
from ffg_capture.diagnostics import probe_ffg


def main() -> int:
    logger = configure_logging("INFO", log_name="ffg_check_ready")

    def action() -> int:
        dependencies = {
            "numpy": "numpy",
            "pyserial": "serial",
            "h5py": "h5py",
            "matplotlib": "matplotlib",
            "PyYAML": "yaml",
        }
        status = {
            name: util.find_spec(module) is not None
            for name, module in dependencies.items()
        }
        print("1. 环境")
        for name, ready in status.items():
            print(f"   {name:10s}: {'OK' if ready else 'MISSING'}")
        if not all(status.values()):
            logger.error("依赖不完整，请运行: pip install -r requirements.txt")
            return 2

        config = load_config()
        ports = discover_serial_ports()
        print("2. 串口")
        for item in ports:
            print(f"   {item.device}: {item.description}")

        print("3. 设备")
        print("   请在手部检查期间张开并握拳数次。")
        source = FFGSource(config.device, logger)
        report_path = project_path(
            f"outputs/diagnostics/ready_{timestamp_label()}.json"
        )
        try:
            source.start()
            result = probe_ffg(
                source,
                frames=config.diagnostics.sample_frames,
                timeout_s=config.diagnostics.timeout_s,
                motion_check=config.diagnostics.motion_check,
            )
            report = {
                "success": result.success,
                "platform": platform.platform(),
                "python": sys.executable,
                "dependencies": status,
                "device": source.device.info,
                "ports": [item.as_dict() for item in ports],
                "probe": result.report,
            }
            saved = write_json(report_path, report)
            print("4. 数据")
            print(f"   识别手侧: {source.device.info.get('side')}")
            print(f"   帧率: {result.report.get('measured_rate_hz')}")
            print(f"   通道: {result.channel_count}")
            print(f"   动作变化: {'YES' if result.changed else 'NO'}")
            print(f"   结果: {'PASS' if result.success else 'FAIL'}")
            announce_artifact("检查报告", saved)
            return 0 if result.success else 2
        except Exception as exc:
            saved = write_json(
                report_path,
                {
                    "success": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "ports": [item.as_dict() for item in ports],
                },
            )
            print("4. 数据")
            print("   结果: FAIL")
            announce_artifact("失败报告", saved)
            raise
        finally:
            source.stop()

    return run_script("FFG 就绪检查", action, logger=logger)


if __name__ == "__main__":
    raise SystemExit(main())
