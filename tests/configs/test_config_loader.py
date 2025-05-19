from pathlib import Path

import yaml

from ai_baby_monitor.config import RoomConfig, load_room_config_file


def test_load_room_config_file_with_optional_fields(tmp_path: Path):
    """Test loading a config file with optional fields specified."""
    sample = {
        "name": "custom_room",
        "camera": {
            "uri": "rtsp://example.com/stream",
            "frame_width": 1280,
            "frame_height": 720,
            "subsampled_stream_maxlen": 100,
            "subsample_rate": 8,
        },
        "instructions": ["custom instruction"],
    }

    config_path = tmp_path / "custom_room.yaml"
    config_path.write_text(yaml.safe_dump(sample))

    config = load_room_config_file(config_path)

    # Check required fields
    assert config.name == "custom_room"
    assert config.camera_uri == "rtsp://example.com/stream"
    assert config.instructions == ["custom instruction"]

    # Check optional fields are loaded correctly
    assert config.frame_width == 1280
    assert config.frame_height == 720
    assert config.subsampled_stream_maxlen == 100
    assert config.subsample_rate == 8

    assert isinstance(config, RoomConfig)


def test_load_room_config_file_with_partial_optional_fields(tmp_path: Path):
    """Test loading a config file with only some optional fields specified."""
    sample = {
        "name": "partial_room",
        "camera": {
            "uri": "2",
            # Only specify frame_width, leaving other optional fields as defaults
            "frame_width": 800,
        },
    }

    config_path = tmp_path / "partial_room.yaml"
    config_path.write_text(yaml.safe_dump(sample))

    config = load_room_config_file(config_path)

    # Check the specified optional field
    assert config.frame_width == 800

    # Check that other optional fields keep their defaults
    assert config.frame_height == 360  # default
    assert config.subsampled_stream_maxlen == 64  # default
    assert config.subsample_rate == 4  # default
    assert config.instructions == []  # default
