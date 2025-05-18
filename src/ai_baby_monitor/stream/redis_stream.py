import datetime as dt

import numpy as np
import redis
import structlog

from ai_baby_monitor.stream import Frame

logger = structlog.get_logger()


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
        logger.info(
            "Initialized Redis stream handler",
            redis_host=redis_host,
            redis_port=redis_port,
        )

    @staticmethod
    def serialize_frame(frame: Frame) -> dict:
        """Serialize a Frame object to a dictionary for Redis storage."""
        frame_bytes = frame.frame_data.tobytes()

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
            # Get the raw JPEG bytes
            frame_bytes = frame_data[b"frame_bytes"]

            # Convert bytes back to numpy array (still JPEG encoded)
            frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)

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
            logger.error("Error deserializing frame", error=e)
            return None

    @staticmethod
    def deserialize_log(log_data: dict) -> dict:
        """Deserialize a log from Redis data."""
        new_log_data = {}
        for key, value in log_data.items():
            if isinstance(key, bytes):
                key = key.decode("utf-8")
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            new_log_data[key] = value
        return new_log_data

    def add_frame(
        self, frame: Frame, key: str, maxlen: int, approximate: bool = True
    ) -> str:
        """Add a frame to the Redis stream and trim if necessary.

        Args:
            frame: Frame object to add to the stream
            key: Redis stream key
            maxlen: Maximum length of the stream
            approximate: Whether to use approximate length

        Returns:
            entry_id: ID of the added entry
        """
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

    def add_logs(
        self,
        key: str,
        log_data: dict,
        maxlen: int = 3600 * 6,
        approximate: bool = True,
    ) -> str:
        """Add logs to the Redis stream."""
        entry_id = self.redis_client.xadd(
            name=key, fields=log_data, maxlen=maxlen, approximate=approximate
        )
        return entry_id

    def get_latest_entries(
        self, key: str, count: int = 1, last_id: str | None = None
    ) -> list[tuple[bytes, dict]]:
        """Get the latest entries from a Redis stream.
        This bit is kinda tricky. To get the latest entries, you need to use xrevrange, 
        but then reverse it back to get chronological order. If you use xrange, you get
        earliest entries first.
        """
        min_id = last_id if last_id else "-"
        return self.redis_client.xrevrange(name=key, max="+", min=min_id, count=count)[
            ::-1
        ]

    def get_latest_frames(self, key: str, count: int = 1) -> list[Frame]:
        """Get the latest frames from the Redis stream."""
        entries = self.get_latest_entries(key=key, count=count)

        # Deserialize the frames
        frames = []
        for entry_id, data in entries:
            frame = self.deserialize_frame(data)
            if frame:
                frames.append(frame)

        return frames

    def get_latest_logs(
        self, key: str, count: int = 1, last_log_id: str | None = None
    ) -> list[dict]:
        """Get the latest logs from the Redis stream."""
        return self.get_latest_entries(key=key, count=count, last_id=last_log_id)
