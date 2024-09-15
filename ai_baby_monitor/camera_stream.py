import datetime as dt
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Frame:
    frame_data: np.ndarray
    timestamp: dt.datetime
    frame_idx: int | None = None


class CameraStream:
    def __init__(
        self,
        stream_id: str,
        uri: str | int,
        buffer_duration: int = 15,
        capture_fps: int = 2,
    ):
        """
        Initialize the camera stream with background frame capture.

        Args:
            stream_id: Unique identifier for the stream
            uri: Camera URI or device index
            buffer_duration: Buffer duration in seconds
            capture_fps: Target capture frame rate
        """
        # Stream
        self.stream_id = stream_id
        self.cap = cv2.VideoCapture(uri)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")

        # Buffer settings
        buffer_size = int(buffer_duration * capture_fps * 1.2)  # 20% safety margin
        self.frame_buffer = deque(maxlen=buffer_size)
        self.capture_fps = capture_fps

        # Threading
        self.running = False
        self.lock = threading.Lock()
        self._capture_thread = None
        self.frame_idx = 0

    def start(self):
        """Start the background frame capture thread."""
        if self.running:
            return

        self.running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

    def stop(self):
        """Stop the background frame capture thread."""
        self.running = False
        if self._capture_thread:
            self._capture_thread.join()
        self.cap.release()

    def _capture_loop(self):
        """Background thread function to continuously capture frames."""
        last_capture = 0

        while self.running:
            # Control capture rate
            current_time = time.time()
            elapsed = current_time - last_capture
            sleep_time = 1.0 / self.capture_fps - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            last_capture = time.time()

            ret, frame = self.cap.read()
            if not ret:
                logger.warning(f"Failed to capture frame from {self.stream_id}")
                continue

            frame = Frame(
                frame_data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                timestamp=dt.datetime.now(),
                frame_idx=self.frame_idx,
            )
            self.frame_idx += 1
            self.frame_buffer.append(frame)

    def _validate_fps(self, fps: int):
        """Validate the fps for the buffer."""
        if fps > self.capture_fps:
            raise ValueError(
                f"Requested fps {fps} is greater than the capture fps {self.capture_fps}"
            )
        if self.capture_fps % fps != 0:
            raise ValueError(
                f"Requested fps {fps} is not a multiple of the capture fps {self.capture_fps}"
            )

    def get_frame_buffer(self) -> list[Frame]:
        """Get a copy of the current frame buffer."""
        with self.lock:
            return list(self.frame_buffer)

    def get_latest_frame(self) -> Frame | None:
        """Get the most recent frame from the buffer."""
        with self.lock:
            return self.frame_buffer[-1] if self.frame_buffer else None

    def get_latest_n_frames(self, n: int, fps: int | None = None) -> list[Frame]:
        """Get the latest and available n frames from the buffer at specified fps."""
        if fps is None:
            fps = self.capture_fps
        self._validate_fps(fps)
        with self.lock:
            step = int(self.capture_fps / fps)
            all_frames = list(self.frame_buffer)
            selected_frames = all_frames[::-step][:n]
            return selected_frames[::-1]  # reverse to chronological order

    def get_latest_n_seconds(self, n: float, fps: int | None = None) -> list[Frame]:
        """Get the latest and available n seconds of frames from the buffer at specified fps."""
        if fps is None:
            fps = self.capture_fps
        self._validate_fps(fps)
        threshold = dt.datetime.now() - dt.timedelta(seconds=n)
        with self.lock:
            step = int(self.capture_fps / fps)
            all_frames = list(self.frame_buffer)
            selected_frames = all_frames[::-step]
            selected_frames = [
                frame for frame in selected_frames if frame.timestamp >= threshold
            ]
            return selected_frames[::-1]  # reverse to chronological order

    def is_healthy(self) -> bool:
        """Check if the stream is healthy by verifying recent frame captures."""
        with self.lock:
            if not self.frame_buffer:
                return False
            latest_frame = self.frame_buffer[-1]
            time_since_last_frame = (
                dt.datetime.now() - latest_frame.timestamp
            ).total_seconds()
            return time_since_last_frame < (3.0 / self.capture_fps)

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
