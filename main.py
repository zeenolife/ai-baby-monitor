import logging
import sys
import argparse
from pathlib import Path

import yaml
from vllm import LLM

from ai_baby_monitor.camera_stream import CameraStream
from ai_baby_monitor.instructions import Instruction
from ai_baby_monitor.watcher import Watcher

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing config file: {e}")
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="AI Baby Monitor")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/living_room.yaml"),
        help="Path to the configuration file (default: configs/living_room.yaml)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)

    # Initialize the LLM
    llm = LLM(
        model=config["llm"]["model_name"],
        gpu_memory_utilization=config["llm"]["gpu_memory_utilization"],
        enable_prefix_caching=config["llm"]["enable_prefix_caching"],
        max_seq_len_to_capture=8192,
    )

    # Initialize the camera stream
    camera_stream = CameraStream(stream_id=config["name"], uri=config["camera"]["uri"])
    camera_stream.start()
    import time
    import cv2
    print("started stream")
    time.sleep(14)
    print("first 14 seconds")
    frames = camera_stream.get_latest_n_seconds(n=10, fps=2)
    for frame in frames:
        print(frame.frame_idx, frame.timestamp)
        print("--------------------------------")

    time.sleep(10)
    print("next 10 frames")
    frames = camera_stream.get_latest_n_seconds(n=5, fps=2)
    for frame in frames:
        print(frame.frame_idx, frame.timestamp)
        cv2.imwrite(f"tmp/frame_{frame.timestamp}.jpg", frame.frame_data)
    print("--------------------------------")
    camera_stream.stop()

    # Create instructions from config
    instructions = [Instruction(instr) for instr in config["instructions"]]

    if not instructions:
        logger.error("No instructions found in config")
        return

    # Create and start the watcher
    nanny = Watcher(
        watcher_id=config["name"],
        camera_stream=camera_stream,
        instructions=instructions,
        llm=llm,
    )

    try:
        logger.info("Starting AI Baby Monitor...")
        nanny.watch()
    except KeyboardInterrupt:
        logger.info("Shutting down AI Baby Monitor...")
    finally:
        camera_stream.stop()


if __name__ == "__main__":
    main()
