import logging
import time
import argparse
from ai_baby_monitor import CameraStream, RedisStreamHandler

logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s] %(asctime)s - %(message)s"
)
logger = logging.getLogger(__name__)


def stream_to_redis(
    camera_uri: str,
    redis_stream_key: str,
    redis_host: str,
    redis_port: int,
    max_frames: int,
    save_stream_path: str | None,
    frame_width: int | None,
    frame_height: int | None,
):
    """Stream camera frames to Redis."""
    # Convert camera_uri to int if it's a digit (for webcam index)
    if camera_uri.isdigit():
        camera_uri = int(camera_uri)

    # Set frame shape if both dimensions are provided
    frame_shape = None
    if frame_width is not None and frame_height is not None:
        frame_shape = (frame_width, frame_height)

    logger.info(f"Initializing camera stream from: {camera_uri}")
    logger.info(
        f"Streaming to Redis key: {redis_stream_key} at {redis_host}:{redis_port}"
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
            stream_key=redis_stream_key,
            redis_host=redis_host,
            redis_port=redis_port,
            max_num_frames=max_frames,
        )

        logger.info("Starting stream to redis.")

        # Main streaming loop
        while True:
            # Only capture when a new frame is available
            frame = camera.capture_new_frame()

            if frame:
                # Add frame to Redis
                redis_handler.add_frame(frame)

                # Log progress every 100 frames
                logger.info(
                    f"Streamed {frame.frame_idx + 1} frames. Frame timestamp: {frame.timestamp}"
                )
            else:
                logger.warning("Failed to capture frame, retrying...")
                time.sleep(0.01)

    except KeyboardInterrupt:
        logger.info("Stream interrupted by user")
    except Exception as e:
        logger.error(f"Error in streaming: {e}")
    finally:
        if "camera" in locals():
            camera.close()
        logger.info("Stream closed")


def parse_args():
    parser = argparse.ArgumentParser(description="Stream camera frames to Redis.")

    parser.add_argument(
        "--camera-uri",
        required=True,
        help="Camera URI or device index (use 0 for default webcam)",
    )
    parser.add_argument(
        "--redis-stream-key", required=True, help="Redis stream key to use"
    )
    parser.add_argument("--redis-host", default="localhost", help="Redis server host")
    parser.add_argument(
        "--redis-port", default=6379, type=int, help="Redis server port"
    )
    parser.add_argument(
        "--max-frames",
        default=150,
        type=int,
        help="Maximum number of frames to keep in Redis",
    )
    parser.add_argument(
        "--save-stream-path",
        default=None,
        help="Path to save the video stream (optional)",
    )
    parser.add_argument(
        "--frame-width",
        default=640,
        type=int,
        help="Frame width to resize to (optional)",
    )
    parser.add_argument(
        "--frame-height",
        default=360,
        type=int,
        help="Frame height to resize to (optional)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    stream_to_redis(
        camera_uri=args.camera_uri,
        redis_stream_key=args.redis_stream_key,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        max_frames=args.max_frames,
        save_stream_path=args.save_stream_path,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
    )
