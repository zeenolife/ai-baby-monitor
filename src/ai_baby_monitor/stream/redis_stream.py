import logging
import datetime as dt
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
        redis_host: str = "localhost",
        redis_port: int = 6379,
    ):
        """
        Handler for streaming camera frames to Redis.

        Args:
            redis_host: Redis server host
            redis_port: Redis server port
        """
        self.redis_client = redis.Redis(host=redis_host, port=redis_port)
        logger.info(f"Initialized Redis stream handler on {redis_host}:{redis_port}")

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
            # Unpack the frame data - Redis returns bytes for binary data
            frame_bytes = frame_data[b"frame_bytes"]
            frame_info = msgpack.unpackb(frame_bytes)

            # Reconstruct the numpy array
            frame_array = np.frombuffer(
                frame_info["data"], dtype=np.dtype(frame_info["dtype"])
            ).reshape(frame_info["shape"])

            # Convert timestamp string back to datetime
            timestamp = dt.datetime.fromisoformat(
                frame_data[b"timestamp"].decode("utf-8")
            )

            # Create and return a Frame object
            return Frame(
                frame_data=frame_array,
                timestamp=timestamp,
                frame_idx=int(frame_data[b"frame_idx"]),
            )
        except Exception as e:
            logger.error(f"Error deserializing frame: {e}")
            return None

    def add_frame(
        self, frame: Frame, key: str, maxlen: int, approximate: bool = True
    ) -> str:
        """Add a frame to the Redis stream and trim if necessary."""
        # Serialize the frame
        data = self.serialize_frame(frame)

        # Add to Redis stream
        entry_id = self.redis_client.xadd(
            name=key,
            fields=data,
            maxlen=maxlen,
            approximate=approximate,
        )

        return entry_id

    def get_latest_frames(self, key: str, count: int = 1) -> list[Frame]:
        """Get the latest frames from the Redis stream."""
        # Get the latest entries from the stream
        entries = self.redis_client.xrevrange(name=key, count=count)

        # Deserialize the frames
        frames = []
        for entry_id, data in entries:
            frame = self.deserialize_frame(data)
            if frame:
                frames.append(frame)

        return frames
