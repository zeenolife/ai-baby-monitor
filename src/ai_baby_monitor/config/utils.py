from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(slots=True, kw_only=True)
class RoomConfig:
    name: str
    camera_uri: str
    llm_model_name: str
    instructions: list[str] = field(default_factory=list)
    subsampled_stream_maxlen: int = 64
    frame_width: int = 640
    frame_height: int = 360
    subsample_rate: int = 4

    def __str__(self):
        return self.name


def load_room_config_file(config_path: str | Path) -> RoomConfig:
    """Load a single room configuration from a specified YAML file"""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        data = yaml.safe_load(config_path.read_text())
        
        config_kwargs = {
            # Required fields
            "name": data["name"],
            "camera_uri": data["camera"]["uri"],
            "llm_model_name": data["llm"]["model_name"],
            "instructions": data.get("instructions", []),
            
            # Optional fields (only include if specified in YAML)
            **{k: data["camera"][k] for k in 
               ["subsampled_stream_maxlen", "frame_width", "frame_height", "subsample_rate"]
               if k in data["camera"]}
        }
        
        return RoomConfig(**config_kwargs)
    except KeyError as e:
        raise ValueError(f"Missing required field in config: {e}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")


def load_multiple_room_configs(config_files: list[str | Path]) -> dict[str, RoomConfig]:
    """Load multiple room configurations from a list of YAML files."""
    configs = {}
    for config_file in config_files:
        config = load_room_config_file(config_file)
        configs[config.name] = config
    return configs
