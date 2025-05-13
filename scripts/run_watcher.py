import argparse
import time
from pathlib import Path

import structlog
from playsound import playsound

from ai_baby_monitor import RedisStreamHandler, Watcher
from ai_baby_monitor.config import load_room_config_file

logger = structlog.get_logger()


def run_watcher(
    redis_stream_key: str,
    redis_host: str,
    redis_port: int,
    instructions: list[str],
    vllm_host: str,
    vllm_port: int,
    model_name: str,
    num_frames_to_process: int,
):
    """
    Run the Watcher continuously to monitor frames from Redis stream.

    Args:
        redis_stream_key: Base Redis stream key (e.g., room name from config). Will use {key}:subsampled for video frames and {key}:logs for logs.
        redis_host: Redis server host.
        redis_port: Redis server port.
        instructions: List of monitoring instructions to check (from room config).
        vllm_host: vLLM server host.
        vllm_port: vLLM server port.
        model_name: Model name to use for inference (from room config).
        num_frames_to_process: Number of frames to analyze in each batch.
    """
    # Initialize Redis stream handler
    redis_handler = RedisStreamHandler(
        redis_host=redis_host,
        redis_port=redis_port,
    )

    # Initialize Watcher
    nanny_watcher = Watcher(
        instructions=instructions,
        vllm_host=vllm_host,
        vllm_port=vllm_port,
        model_name=model_name,
    )

    # Subsampled stream key
    subsampled_key = f"{redis_stream_key}:subsampled"
    logs_key = f"{redis_stream_key}:logs"
    logger.info(
        "Starting Watcher monitoring Redis",
        video_queue_key=subsampled_key,
        logs_queue_key=logs_key,
    )
    logger.info(
        "Using model", model_name=model_name, vllm_host=vllm_host, vllm_port=vllm_port
    )
    logger.info("Monitoring instructions", instructions=instructions)

    try:
        while True:
            # Get latest frames from Redis
            frames = redis_handler.get_latest_frames(
                subsampled_key, num_frames_to_process
            )

            if not frames:
                logger.warning(
                    "No frames available in stream", video_queue_key=subsampled_key
                )
                time.sleep(0.3)
                continue

            # Reverse frames to get chronological order (oldest first)
            frames.reverse()

            logger.info("Analyzing frames from stream", num_frames=len(frames))

            # Process frames with Watcher
            result = nanny_watcher.process_frames(frames)

            if result["success"]:
                # Log the result
                alert_status = (
                    "🚨 ALERT TRIGGERED"
                    if result["should_alert"]
                    else "✅ No alert needed"
                )
                awareness = result["recommended_awareness_level"]

                logger.info(
                    "Alert status and reasoning",
                    alert_status=alert_status,
                    awareness_level=awareness,
                    reasoning=result["reasoning"],
                )

                # Stream logs back to Redis
                log_data = {
                    "timestamp": time.time(),
                    "should_alert": int(result["should_alert"]),
                    "awareness_level": awareness,
                    "reasoning": result["reasoning"],
                }
                redis_handler.add_logs(logs_key, log_data)

                if result["should_alert"]:
                    playsound("assets/alert.wav")
            else:
                error_msg = result.get("error", "Unknown error")
                logger.error("Error processing frames", error=error_msg)

                # Stream error to Redis
                error_data = {
                    "timestamp": time.time(),
                    "error": error_msg,
                }
                redis_handler.add_logs(logs_key, error_data)

                time.sleep(0.3)

    except KeyboardInterrupt:
        logger.info("Watcher stopped by user")
    except Exception as e:
        logger.error("Error in Watcher", error=e)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Watcher to monitor Redis stream frames based on room configuration."
    )

    parser.add_argument(
        "--config-file", required=True, help="Path to room configuration YAML file"
    )
    parser.add_argument("--redis-host", default="localhost", help="Redis server host")
    parser.add_argument(
        "--redis-port", default=6379, type=int, help="Redis server port"
    )
    parser.add_argument("--vllm-host", default="localhost", help="vLLM server host")
    parser.add_argument("--vllm-port", default=8000, type=int, help="vLLM server port")
    parser.add_argument(
        "--num-frames-to-process",
        default=16,
        type=int,
        help="Number of frames to analyze in each batch",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    try:
        # Load room configuration from file
        room_config = load_room_config_file(args.config_file)
        config_file_path = Path(args.config_file)
        logger.info(
            f"Loaded configuration for room: {room_config.name}",
            config_file=str(config_file_path.resolve()),
        )

        # Extract parameters from config
        redis_stream_key = room_config.name
        instructions = room_config.instructions
        model_name = room_config.llm_model_name

        # Ensure instructions are provided, as RoomConfig defaults to an empty list if not in YAML.
        if not instructions:
            logger.error(
                "The 'instructions' list is empty or missing in the configuration file. At least one instruction is required for the watcher.",
                config_file=str(config_file_path.resolve()),
            )
            exit(1)

        run_watcher(
            redis_stream_key=redis_stream_key,
            redis_host=args.redis_host,
            redis_port=args.redis_port,
            instructions=instructions,
            vllm_host=args.vllm_host,
            vllm_port=args.vllm_port,
            model_name=model_name,
            num_frames_to_process=args.num_frames_to_process,
        )
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {args.config_file}")
        exit(1)
    except Exception as e:
        logger.error("Failed to start watcher", error=str(e))
        exit(1)
