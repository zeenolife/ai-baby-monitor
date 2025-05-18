import argparse
import time
from pathlib import Path

import structlog

from ai_baby_monitor.stream import CameraStream, RedisStreamHandler
from ai_baby_monitor.config import load_room_config_file

logger = structlog.get_logger()


def stream_to_redis(
    camera_uri: str,
    redis_stream_key: str,
    redis_host: str,
    redis_port: int,
    subsampled_stream_maxlen: int,
    save_stream_path: str | None,
    frame_width: int | None,
    frame_height: int | None,
    subsample_rate: int = 4,
):
    """
    Stream camera frames to Redis short realtime and long subsampled queues.
    
    This function is used by the main entry point and expects configuration
    parameters to be loaded from the room's YAML config file.
    """
    # Convert camera_uri to int if it's a digit (for webcam index)
    if camera_uri.isdigit():
        camera_uri = int(camera_uri)

    # Set frame shape if both dimensions are provided
    frame_shape = None
    if frame_width is not None and frame_height is not None:
        frame_shape = (frame_width, frame_height)

    logger.info("Initializing camera stream", camera_uri=camera_uri)
    logger.info(
        "Streaming to Redis",
        redis_stream_key=redis_stream_key,
        redis_host=redis_host,
        redis_port=redis_port,
    )
    logger.info(
        f"Using subsample rate of 1 out of {subsample_rate} for subsampled queue"
    )

    try:
        # Initialize camera stream
        camera = CameraStream(
            uri=camera_uri,
            save_stream_path=save_stream_path,
            frame_shape=frame_shape,
        )

        # Initialize Redis stream handler
        redis_handler = RedisStreamHandler(
            redis_host=redis_host,
            redis_port=redis_port,
        )

        logger.info("Starting to stream to redis.")

        # Main streaming loop
        while True:
            # Only capture when a new frame is available
            frame = camera.capture_new_frame()
            if frame:
                # Always add frame to realtime queue
                redis_handler.add_frame(
                    frame, f"{redis_stream_key}:realtime", 3, approximate=False
                )

                # Add to subsampled queue every nth frame
                if frame.frame_idx % subsample_rate == 0:
                    redis_handler.add_frame(
                        frame, f"{redis_stream_key}:subsampled", subsampled_stream_maxlen
                    )
                    logger.info(
                        "Added frame to subsampled queue",
                        frame_idx=frame.frame_idx,
                        timestamp=frame.timestamp,
                    )
            else:
                logger.warning("Failed to capture frame, retrying...")
                time.sleep(0.01)

    except KeyboardInterrupt:
        logger.info("Stream interrupted by user")
    except Exception as e:
        logger.error("Error in streaming", error=e)
    finally:
        if "camera" in locals():
            camera.close()
        logger.info("Stream closed")


def parse_args():
    """Parse command line arguments for room-based configuration."""
    parser = argparse.ArgumentParser(
        description="Stream camera frames to Redis based on room configuration."
    )

    parser.add_argument(
        "--config-file", 
        required=True, 
        help="Path to room configuration YAML file"
    )
    parser.add_argument(
        "--redis-host", default="localhost", help="Redis server host"
    )
    parser.add_argument(
        "--redis-port", default=6379, type=int, help="Redis server port"
    )
    parser.add_argument(
        "--save-stream-path",
        default=None,
        help="Path to save the video stream (optional)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Override camera URI with demo footage if available",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Load room configuration from file
    try:
        room_config = load_room_config_file(args.config_file)
        config_file_path = Path(args.config_file)
        logger.info(f"Loaded configuration for room: {room_config.name}", 
                   config_file=str(config_file_path.resolve()))
        
        # Extract camera settings from config
        camera_uri = room_config.camera_uri
        
        # Override with demo if requested
        if args.demo:
            camera_uri = "assets/demo/demo.mp4"
            logger.info("Using demo video source", camera_uri=camera_uri)
                
        # Use the room name as the redis stream key
        redis_stream_key = room_config.name
        
        stream_to_redis(
            camera_uri=camera_uri,
            redis_stream_key=redis_stream_key,
            redis_host=args.redis_host,
            redis_port=args.redis_port,
            subsampled_stream_maxlen=room_config.subsampled_stream_maxlen,
            save_stream_path=args.save_stream_path,
            frame_width=room_config.frame_width,
            frame_height=room_config.frame_height,
            subsample_rate=room_config.subsample_rate,
        )
    except Exception as e:
        logger.error("Failed to start streaming", error=e)
        exit(1)
