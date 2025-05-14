import datetime as dt
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import redis  # Imported for spec in MagicMock

from ai_baby_monitor.stream.camera_stream import Frame
from ai_baby_monitor.stream.redis_stream import RedisStreamHandler


@pytest.fixture
def mock_redis_client():
    """Fixture for a mock Redis client."""
    client = MagicMock(spec=redis.Redis)
    client.xadd = MagicMock(return_value=b"mock_entry_id")
    client.xrevrange = MagicMock(return_value=[])
    return client


@pytest.fixture
def redis_handler(mock_redis_client: MagicMock):
    """Fixture for RedisStreamHandler initialized with a mock Redis client."""
    with patch("redis.Redis", return_value=mock_redis_client) as mock_redis_constructor:
        handler = RedisStreamHandler(redis_host="mock_host", redis_port=1234)
        mock_redis_constructor.assert_called_once_with(host="mock_host", port=1234)
    return handler


@pytest.fixture
def sample_frame_data() -> np.ndarray:
    """Fixture for sample raw frame data (e.g., JPEG bytes as numpy array)."""
    return np.array([10, 20, 30, 40, 50], dtype=np.uint8)


@pytest.fixture
def sample_frame(sample_frame_data: np.ndarray) -> Frame:
    """Fixture for a sample Frame object."""
    return Frame(
        frame_data=sample_frame_data,
        timestamp=dt.datetime(2023, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc),
        frame_idx=100,
    )


def test_serialize_deserialize_frame(sample_frame: Frame):
    """Test serialization and deserialization of a Frame object."""
    serialized_data = RedisStreamHandler.serialize_frame(sample_frame)

    assert isinstance(serialized_data, dict)
    assert "frame_bytes" in serialized_data
    assert "timestamp" in serialized_data
    assert "frame_idx" in serialized_data
    assert serialized_data["timestamp"] == sample_frame.timestamp.isoformat()
    assert serialized_data["frame_idx"] == sample_frame.frame_idx
    assert serialized_data["frame_bytes"] == sample_frame.frame_data.tobytes()

    # deserialize_frame expects byte keys and byte values (for strings) from Redis
    redis_formatted_data = {
        b"frame_bytes": sample_frame.frame_data.tobytes(),
        b"timestamp": sample_frame.timestamp.isoformat().encode("utf-8"),
        b"frame_idx": str(sample_frame.frame_idx).encode("utf-8"),
    }

    deserialized_frame = RedisStreamHandler.deserialize_frame(redis_formatted_data)

    assert deserialized_frame is not None
    assert np.array_equal(deserialized_frame.frame_data, sample_frame.frame_data)
    assert deserialized_frame.timestamp == sample_frame.timestamp
    assert deserialized_frame.frame_idx == sample_frame.frame_idx


def test_add_frame(
    redis_handler: RedisStreamHandler, mock_redis_client: MagicMock, sample_frame: Frame
):
    """Test adding a frame to the Redis stream."""
    key = "test_stream"
    maxlen = 100
    approximate = True

    entry_id = redis_handler.add_frame(sample_frame, key, maxlen, approximate)

    assert entry_id == b"mock_entry_id"
    mock_redis_client.xadd.assert_called_once()
    call_args = mock_redis_client.xadd.call_args[1]  # kwargs

    assert call_args["name"] == key
    assert call_args["maxlen"] == maxlen
    assert call_args["approximate"] == approximate

    expected_fields = RedisStreamHandler.serialize_frame(sample_frame)
    assert call_args["fields"]["frame_bytes"] == expected_fields["frame_bytes"]
    assert call_args["fields"]["timestamp"] == expected_fields["timestamp"]
    assert call_args["fields"]["frame_idx"] == expected_fields["frame_idx"]


def test_get_latest_frames(
    redis_handler: RedisStreamHandler,
    mock_redis_client: MagicMock,
    sample_frame: Frame,
    sample_frame_data: np.ndarray,
):
    """Test getting the latest frames from the Redis stream."""
    key = "test_stream"
    count = 2

    # Prepare mock data returned by xrevrange
    # Timestamps need to be strings for fromisoformat
    frame_1_raw_data = {
        b"frame_bytes": sample_frame_data.tobytes(),
        b"timestamp": sample_frame.timestamp.isoformat().encode("utf-8"),
        b"frame_idx": str(sample_frame.frame_idx).encode("utf-8"),
    }
    frame_2_raw_data = {
        b"frame_bytes": np.array([1, 2, 3], dtype=np.uint8).tobytes(),
        b"timestamp": dt.datetime(2023, 1, 1, 12, 0, 1, tzinfo=dt.timezone.utc)
        .isoformat()
        .encode("utf-8"),
        b"frame_idx": b"101",
    }
    mock_redis_client.xrevrange.return_value = [
        (b"entry_id_1", frame_1_raw_data),
        (b"entry_id_2", frame_2_raw_data),
    ]

    frames = redis_handler.get_latest_frames(key, count)

    mock_redis_client.xrevrange.assert_called_once_with(name=key, count=count)
    assert len(frames) == 2

    # Check first frame (corresponds to sample_frame)
    assert np.array_equal(frames[0].frame_data, sample_frame.frame_data)
    assert frames[0].timestamp == sample_frame.timestamp
    assert frames[0].frame_idx == sample_frame.frame_idx

    # Check second frame
    assert np.array_equal(frames[1].frame_data, np.array([1, 2, 3], dtype=np.uint8))
    assert frames[1].timestamp == dt.datetime(
        2023, 1, 1, 12, 0, 1, tzinfo=dt.timezone.utc
    )
    assert frames[1].frame_idx == 101


def test_add_logs(redis_handler: RedisStreamHandler, mock_redis_client: MagicMock):
    """Test adding logs to the Redis stream."""
    key = "log_stream"
    log_data = {"message": "Test log", "level": "INFO"}

    entry_id = redis_handler.add_logs(key, log_data)

    assert entry_id == b"mock_entry_id"
    mock_redis_client.xadd.assert_called_once_with(name=key, fields=log_data)


def test_get_latest_entries(
    redis_handler: RedisStreamHandler, mock_redis_client: MagicMock
):
    """Test getting the latest generic entries from Redis."""
    key = "generic_stream"
    count = 5
    mock_return_data = [
        (b"id1", {b"field1": b"value1"}),
        (b"id2", {b"field2": b"value2"}),
    ]
    mock_redis_client.xrevrange.return_value = mock_return_data

    entries = redis_handler.get_latest_entries(key, count)

    mock_redis_client.xrevrange.assert_called_once_with(name=key, count=count)
    assert entries == mock_return_data


def test_get_latest_logs(
    redis_handler: RedisStreamHandler, mock_redis_client: MagicMock
):
    """Test getting the latest logs, which uses get_latest_entries."""
    key = "log_stream_for_get"
    count = 3
    mock_log_data = [
        (b"log_id_1", {b"timestamp": b"ts1", b"message": b"Log 1"}),
        (b"log_id_2", {b"timestamp": b"ts2", b"message": b"Log 2"}),
    ]
    # Configure the mock for the specific call by get_latest_entries
    mock_redis_client.xrevrange.return_value = mock_log_data

    logs = redis_handler.get_latest_logs(key, count)

    # get_latest_logs calls get_latest_entries, which calls xrevrange
    mock_redis_client.xrevrange.assert_called_once_with(name=key, count=count)
    assert logs == mock_log_data
