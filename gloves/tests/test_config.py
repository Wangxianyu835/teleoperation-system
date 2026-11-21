from ffg_capture.config import load_config


def test_default_config_loads_and_creates_directories():
    config = load_config()
    assert config.device.model == "linker_ffg"
    assert config.device.side in {"left", "right", "auto"}
    assert config.device.baudrates
    assert config.device.serial_query_interval_ms == 6
    assert config.recording.output_dir.is_absolute()
    assert config.recording.output_dir.is_dir()
    assert config.calibration.output_dir.is_dir()
    assert config.visualization.detailed is False
    assert config.realtime.display_enabled is True
    assert config.realtime.debug_recording is False
