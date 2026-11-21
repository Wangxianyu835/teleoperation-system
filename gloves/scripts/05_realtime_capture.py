from __future__ import annotations

import time
from datetime import datetime, timezone

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
from ffg_capture.device import FFGSource
from ffg_capture.realtime import RealtimeStats
from ffg_capture.recorder import CaptureRecorder, create_session_path
from ffg_capture.visualization import LivePlot


def main() -> int:
    logger = configure_logging("INFO", log_name="ffg_realtime")

    def action() -> int:
        config = load_config()
        source = FFGSource(config.device, logger)
        plot: LivePlot | None = None
        recorder: CaptureRecorder | None = None
        recording_path = None
        stats = RealtimeStats()
        started_at = datetime.now(timezone.utc)
        interrupted = False

        try:
            source.start()
            device_info = source.device.info
            side = str(device_info.get("side", "unknown"))
            if config.realtime.debug_recording:
                recording_path = create_session_path(
                    config.recording.output_dir,
                    source.stream_id,
                    suffix="realtime_debug",
                )
                recorder = CaptureRecorder(
                    recording_path,
                    stream_id=source.stream_id,
                    device_info=device_info,
                    session_metadata={
                        "mode": "realtime_debug",
                        "started_at_utc": started_at.isoformat(),
                        "query_interval_ms": config.device.serial_query_interval_ms,
                    },
                )
            if config.realtime.display_enabled:
                plot = LivePlot(
                    max_points=config.visualization.max_points,
                    refresh_hz=config.realtime.display_refresh_hz,
                    side=side,
                )
                plot.open()

            print(
                f"实时采集已启动: {source.stream_id}，"
                f"查询周期配置 {config.device.serial_query_interval_ms:.0f} ms"
            )
            print("按 Ctrl+C 停止。")
            started = time.monotonic()
            last_status = started
            while True:
                elapsed = time.monotonic() - started
                if (
                    config.realtime.duration_s > 0
                    and elapsed >= config.realtime.duration_s
                ):
                    break
                if plot is not None and not plot.window_open:
                    logger.info("可视化窗口已关闭，停止实时采集")
                    break
                sample = source.read(timeout_s=0.2)
                if sample is None:
                    continue
                stats.update(sample)
                if recorder is not None:
                    recorder.append(sample)
                if plot is not None:
                    plot.update(sample)
                now = time.monotonic()
                if now - last_status >= config.realtime.status_interval_s:
                    snapshot = stats.snapshot()
                    rate = snapshot["rate_hz"]
                    rate_text = "N/A" if rate is None else f"{rate:.1f}"
                    print(
                        f"实时: {snapshot['frames']} 帧 | "
                        f"{rate_text} Hz | "
                        f"无效 {snapshot['invalid_frames']} | "
                        f"最大间隔 {snapshot['max_gap_ms']:.1f} ms | "
                        f"序号 {snapshot['last_sequence']}"
                    )
                    last_status = now
        except KeyboardInterrupt:
            interrupted = True
            print()
            logger.warning("收到 Ctrl+C，正在停止实时采集")
        finally:
            if plot is not None:
                plot.close()
            if recorder is not None:
                recorder.close()
            source.stop()

        snapshot = stats.snapshot()
        report_path = project_path(
            f"outputs/reports/realtime_{source.stream_id}_{timestamp_label()}.json"
        )
        report = {
            "status": "interrupted" if interrupted else "completed",
            "stream_id": source.stream_id,
            "device": source.device.info,
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "query_interval_ms": config.device.serial_query_interval_ms,
            "stats": snapshot,
            "latest_sample_available": source.latest() is not None,
            "debug_recording": str(recording_path) if recording_path else None,
        }
        saved = write_json(report_path, report)
        print()
        print("实时采集结束")
        print(f"  总帧数: {snapshot['frames']}")
        print(f"  实测帧率: {snapshot['rate_hz']}")
        print(f"  无效帧: {snapshot['invalid_frames']}")
        print(f"  最大间隔: {snapshot['max_gap_ms']:.1f} ms")
        announce_artifact("实时报告", saved)
        announce_artifact("调试 H5", recording_path)
        if int(snapshot["frames"] or 0) < 10:
            logger.error("实时采集帧数过少")
            return 2
        return 0

    return run_script("FFG 实时采集", action, logger=logger)


if __name__ == "__main__":
    raise SystemExit(main())
