from __future__ import annotations

import time
from datetime import datetime, timezone

from _bootstrap import bootstrap

bootstrap()

from ffg_capture.calibration import resolve_calibration
from ffg_capture.config import load_config, project_path
from ffg_capture.console import (
    announce_artifact,
    configure_logging,
    run_script,
)
from ffg_capture.device import FFGSource
from ffg_capture.quality import grade_capture
from ffg_capture.recorder import CaptureRecorder, create_session_path, write_summary
from ffg_capture.replay import load_capture
from ffg_capture.visualization import save_capture_plot, save_summary_plot


def main() -> int:
    logger = configure_logging("INFO", log_name="ffg_collect_offline")

    def action() -> int:
        config = load_config()
        source = FFGSource(config.device, logger)
        recorder: CaptureRecorder | None = None
        session_path = None
        sample_count = 0
        interrupted = False
        started_at = datetime.now(timezone.utc)

        try:
            source.start()
            device_info = source.device.info
            calibration_path, calibration = resolve_calibration(
                config.calibration,
                side=str(device_info.get("side", "right")),
            )
            if calibration_path:
                logger.info("使用标定文件: %s", calibration_path)
            else:
                logger.warning("没有找到 FFG 标定文件，本次会记录为未标定数据")

            session_path = create_session_path(
                config.recording.output_dir,
                source.stream_id,
            )
            recorder = CaptureRecorder(
                session_path,
                stream_id=source.stream_id,
                device_info=device_info,
                calibration=calibration,
                session_metadata={
                    "calibration_path": str(calibration_path)
                    if calibration_path
                    else None,
                    "config_path": str(config.source_path),
                    "recording_started_at_utc": started_at.isoformat(),
                },
            )
            print("开始离线采集。按 Ctrl+C 停止。")
            if config.recording.default_duration_s > 0:
                print(f"计划采集 {config.recording.default_duration_s:.1f} 秒")
            start_monotonic = time.monotonic()
            last_log = start_monotonic
            while True:
                if (
                    config.recording.default_duration_s > 0
                    and time.monotonic() - start_monotonic
                    >= config.recording.default_duration_s
                ):
                    break
                sample = source.read(timeout_s=0.2)
                if sample is None:
                    continue
                recorder.append(sample)
                sample_count += 1
                now = time.monotonic()
                if now - last_log >= 2.0:
                    rate = sample_count / max(1e-6, now - start_monotonic)
                    print(
                        f"\r采集进度: {sample_count} 帧 | "
                        f"平均 {rate:.1f} Hz | 最近序号 {sample.sequence}",
                        end="",
                        flush=True,
                    )
                    last_log = now
            print()
        except KeyboardInterrupt:
            interrupted = True
            print()
            logger.warning("收到 Ctrl+C，正在安全结束采集")
        finally:
            if recorder is not None:
                recorder.close()
            source.stop()

        if session_path is None or not session_path.exists():
            logger.error("没有生成数据文件")
            return 2

        data = load_capture(session_path)
        report = grade_capture(data, config.quality)
        analysis = report["analysis"]
        grade_summary = report["summary"]
        plot_path = None
        if config.recording.save_final_plot and data.sample_count:
            plot_function = (
                save_capture_plot
                if config.visualization.detailed
                else save_summary_plot
            )
            plot_path = plot_function(
                data,
                project_path(f"outputs/plots/{session_path.stem}.png"),
                dpi=config.visualization.dpi,
                max_points=config.visualization.max_points,
            )
        summary = {
            "status": "interrupted" if interrupted else "completed",
            "sample_count": data.sample_count,
            "recording_started_at_utc": started_at.isoformat(),
            "recording_finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "data_file": str(session_path.resolve()),
            "plot_file": str(plot_path) if plot_path else None,
            "grade": report["grade"],
            "grade_summary": grade_summary,
            "failures": report["failures"],
            "warnings": report["warnings"],
            "quality": analysis,
        }
        summary_path = (
            write_summary(session_path, summary)
            if config.recording.save_summary
            else None
        )
        print()
        print("采集结束")
        print(f"  总帧数: {data.sample_count}")
        print(f"  实测帧率: {grade_summary['rate_hz']}")
        print(f"  数据结论: {report['grade']}")
        announce_artifact("H5 数据", session_path)
        announce_artifact("摘要", summary_path)
        announce_artifact("曲线图", plot_path)
        if data.sample_count < 10:
            logger.error("有效数据帧过少")
            return 2
        if report["grade"] == "FAIL":
            return 2
        return 0

    return run_script("FFG 离线采集", action, logger=logger)


if __name__ == "__main__":
    raise SystemExit(main())
