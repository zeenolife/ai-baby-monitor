import logging

import msgpack
import numpy as np
import redis

from ai_baby_monitor.stream.camera_stream import Frame

logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s] %(asctime)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RedisStreamHandler:
    def __init__(
        self,
        stream_key: str,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        max_num_frames: int = 120,
    ):
        """
        Handler for streaming camera frames to Redis.

        Args:
            stream_key: Redis stream key to use

            redis_host: Redis server host
            redis_port: Redis server port
            max_num_frames: Maximum number of frames to keep in the stream
        """
        self.redis_client = redis.Redis(host=redis_host, port=redis_port)
        self.stream_key = stream_key
        self.max_num_frames = max_num_frames
        logger.info(f"Initialized Redis stream handler with key: {stream_key}")

    @staticmethod
    def serialize_frame(frame: Frame) -> dict:
        """Serialize a Frame object to a dictionary for Redis storage."""
        # Convert numpy array to bytes using msgpack
        frame_bytes = msgpack.packb(
            {
                "shape": frame.frame_data.shape,
                "dtype": str(frame.frame_data.dtype),
                "data": frame.frame_data.tobytes(),
            }
        )

        # Create the data dictionary
        data = {
            "frame_bytes": frame_bytes,
            "timestamp": frame.timestamp.isoformat(),
            "frame_idx": frame.frame_idx,
        }
        return data

    @staticmethod
    def deserialize_frame(frame_data: dict) -> Frame | None:
        """Deserialize a frame from Redis data."""
        try:
            # Unpack the frame data
            frame_info = msgpack.unpackb(frame_data["frame_bytes"])

            # Reconstruct the numpy array
            frame_array = np.frombuffer(
                frame_info["data"], dtype=np.dtype(frame_info["dtype"])
            ).reshape(frame_info["shape"])

            # Create and return a Frame object
            return Frame(
                frame_data=frame_array,
                timestamp=frame_data["timestamp"],
                frame_idx=frame_data["frame_idx"],
            )
        except Exception as e:
            logger.error(f"Error deserializing frame: {e}")
            return None

    def add_frame(self, frame: Frame) -> str:
        """Add a frame to the Redis stream and trim if necessary."""
        # Serialize the frame
        data = self.serialize_frame(frame)

        # Add to Redis stream
        entry_id = self.redis_client.xadd(
            name=self.stream_key,
            fields=data,
            maxlen=self.max_num_frames,
            approximate=True,
        )

        return entry_id

    def get_latest_frames(self, count: int = 1) -> list[Frame]:
        """Get the latest frames from the Redis stream."""
        # Get the latest entries from the stream
        entries = self.redis_client.xrevrange(name=self.stream_key, count=count)

        # Deserialize the frames
        frames = []
        for entry_id, data in entries:
            frame = self.deserialize_frame(data)
            if frame:
                frames.append(frame)

        return frames
