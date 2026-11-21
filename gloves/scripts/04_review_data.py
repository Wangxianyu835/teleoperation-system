from __future__ import annotations

from pathlib import Path

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
from ffg_capture.quality import grade_capture
from ffg_capture.recorder import write_summary
from ffg_capture.replay import find_latest_in_directories, load_capture
from ffg_capture.visualization import save_capture_plot, save_summary_plot


def _latest_file(output_dir: Path) -> Path:
    return find_latest_in_directories([output_dir, project_path("outputs/self_test")])


def main() -> int:
    logger = configure_logging("INFO", log_name="ffg_review")

    def action() -> int:
        config = load_config()
        path = _latest_file(config.recording.output_dir)
        data = load_capture(path)
        report = grade_capture(data, config.quality)
        summary = report["summary"]
        report_path = project_path(
            f"outputs/reports/review_{data.stream_id}_{timestamp_label()}.json"
        )
        plot_path = project_path(
            f"outputs/plots/review_{data.stream_id}_{timestamp_label()}.png"
        )
        plot_function = (
            save_capture_plot if config.visualization.detailed else save_summary_plot
        )
        saved_plot = plot_function(
            data,
            plot_path,
            dpi=config.visualization.dpi,
            max_points=config.visualization.max_points,
        )
        saved_report = write_json(
            report_path,
            {
                **report,
                "data_file": str(data.path),
                "plot_file": str(saved_plot),
            },
        )
        session_summary = data.path.with_suffix(".summary.json")
        if not session_summary.exists():
            session_summary = write_summary(
                data.path,
                {
                    "status": "recovered_by_review",
                    "sample_count": data.sample_count,
                    "data_file": str(data.path),
                    "plot_file": str(saved_plot),
                    "grade": report["grade"],
                    "grade_summary": summary,
                    "failures": report["failures"],
                    "warnings": report["warnings"],
                    "quality": report["analysis"],
                },
            )
        print(f"数据文件: {data.path}")
        print(f"结论: {report['grade']}")
        print(
            f"帧率: {summary['rate_hz']:.1f} Hz" if summary["rate_hz"] else "帧率: N/A"
        )
        print(
            f"无效帧: {summary['invalid_percent']:.2f}%"
            if summary["invalid_percent"] is not None
            else "无效帧: N/A"
        )
        print(
            f"最大帧间隔: {summary['max_gap_ms']:.1f} ms"
            if summary["max_gap_ms"] is not None
            else "最大帧间隔: N/A"
        )
        print(
            f"动作范围: {summary['motion_range_rad']:.3f} rad"
            if summary["motion_range_rad"] is not None
            else "动作范围: N/A"
        )
        print(
            f"静止抖动: {summary['static_jitter_rad']:.4f} rad"
            if summary["static_jitter_rad"] is not None
            else "静止抖动: N/A"
        )
        print(
            f"端点饱和: {summary['saturation_percent']:.1f}%"
            if summary["saturation_percent"] is not None
            else "端点饱和: N/A"
        )
        for warning in report["warnings"]:
            logger.warning(warning)
        for failure in report["failures"]:
            logger.error(failure)
        announce_artifact("质量报告", saved_report)
        announce_artifact("数据摘要", session_summary)
        announce_artifact("简要曲线", saved_plot)
        return 2 if report["grade"] == "FAIL" else 0

    return run_script("FFG 数据审查", action, logger=logger)


if __name__ == "__main__":
    raise SystemExit(main())
