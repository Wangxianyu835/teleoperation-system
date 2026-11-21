from __future__ import annotations

import array
import struct

from _bootstrap import bootstrap

bootstrap()

from ffg_capture.config import project_path
from ffg_capture.console import (
    announce_artifact,
    configure_logging,
    run_script,
    timestamp_label,
    write_json,
)
from ffg_capture.device import SyntheticSource
from ffg_capture.quality import analyze_capture
from ffg_capture.recorder import CaptureRecorder, create_session_path
from ffg_capture.replay import load_capture
from ffg_capture.visualization import save_summary_plot, set_headless_backend


def main() -> int:
    logger = configure_logging("INFO", log_name="ffg_self_test")

    def action() -> int:
        set_headless_backend()
        checks: dict[str, bool] = {}
        details: dict[str, object] = {}
        try:
            from ffg_capture.transport.vendor.linkerforce import FrameHandler, HandType

            degrees = [float((index * 17) % 360) for index in range(21)]
            frame = FrameHandler.pack_data(0x03, struct.pack("<21f", *degrees))
            parsed = FrameHandler(HandType.right).handle_frame(array.array("B", frame))
            checks["protocol_parse"] = bool(parsed and len(parsed["poslist"]) == 21)
            details["parsed_radians"] = parsed["poslist"][:3] if parsed else None
        except Exception as exc:
            checks["protocol_parse"] = False
            details["protocol_error"] = str(exc)

        source = SyntheticSource(
            fps=50.0,
            side="right",
            duration_s=7.0,
            still_seconds=5.0,
        )
        session_path = create_session_path(
            project_path("outputs/self_test"),
            "ffg_right_synthetic",
        )
        source.start()
        try:
            with CaptureRecorder(
                session_path,
                stream_id="ffg_right",
                device_info={"model": "synthetic_ffg", "side": "right"},
                session_metadata={"self_test": True},
            ) as recorder:
                while (sample := source.read(timeout_s=0.2)) is not None:
                    recorder.append(sample)
        finally:
            source.stop()

        data = load_capture(session_path)
        analysis = analyze_capture(data, neutral_seconds=0.5)
        checks["record_replay"] = data.sample_count > 50
        checks["quality_metrics"] = analysis["channel_count"] == 21
        plot_path = save_summary_plot(
            data,
            project_path(f"outputs/self_test/{session_path.stem}.png"),
        )
        checks["plot"] = plot_path.is_file()
        report = {
            "success": all(checks.values()),
            "checks": checks,
            "details": details,
            "sample_count": data.sample_count,
            "data_file": str(session_path),
            "plot_file": str(plot_path),
        }
        report_path = write_json(
            project_path(f"outputs/self_test/self_test_{timestamp_label()}.json"),
            report,
        )
        print("离线自检")
        for name, passed in checks.items():
            print(f"  {name:20s}: {'PASS' if passed else 'FAIL'}")
        print(f"  总结果: {'PASS' if report['success'] else 'FAIL'}")
        announce_artifact("示例 H5", session_path)
        announce_artifact("简要曲线", plot_path)
        announce_artifact("自检报告", report_path)
        return 0 if report["success"] else 2

    return run_script("FFG 离线自检", action, logger=logger)


if __name__ == "__main__":
    raise SystemExit(main())
