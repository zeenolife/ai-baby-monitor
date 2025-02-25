import datetime as dt
import logging
import time
from dataclasses import dataclass

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s] %(asctime)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class Frame:
    frame_data: np.ndarray
    timestamp: dt.datetime
    frame_idx: int


class CameraStream:
    def __init__(
        self,
        uri: str | int,
        target_fps: int = 8,
        save_stream_path: str | None = None,
        frame_shape: tuple[int, int] | None = None,
    ):
        """
        Simple camera stream handler.

        Args:
            uri: Camera URI or device index
            target_fps: Target FPS to capture stream at
            save_stream_path: Path to save the stream
            frame_shape: Frame shape to capture. Would be resized to this shape if provided.
        """
        self.frame_shape = frame_shape
        self.target_fps = target_fps
        self.capture = self._init_capture(uri)
        self.stream_writer = self._init_stream_writer(save_stream_path)

        self.last_capture = dt.datetime.now()
        self.frame_idx = 0

    def _init_capture(self, uri: str | int, max_retries: int = 3) -> cv2.VideoCapture:
        """Initialize and return the camera capture."""
        retries = 0
        while retries < max_retries:
            try:
                capture = cv2.VideoCapture(uri)
                if capture.isOpened():
                    logger.info(f"Successfully connected to RTSP stream: {uri}")
                    return capture
                raise ConnectionError("Failed to open RTSP stream")
            except Exception as e:
                retries += 1
                logger.error(
                    f"Reconnect Attempt to {uri} {retries}/{max_retries}. Error: {str(e)}"
                )
                time.sleep(2)
        raise ConnectionError(f"Failed to connect to camera stream: {uri}")

    def _init_stream_writer(
        self, save_stream_path: str | None
    ) -> cv2.VideoWriter | None:
        """Initialize and return the stream writer if a path is provided."""
        if save_stream_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                save_stream_path, fourcc, self.target_fps, self.frame_shape
            )
            return writer
        else:
            return None

    def capture_frame(self, at_target_fps: bool = True) -> Frame | None:
        """Capture a frame from the camera at target FPS with a sleep."""

        # Sleep if needed
        current_time = dt.datetime.now()
        elapsed = current_time - self.last_capture
        sleep_time = 1.0 / self.target_fps - elapsed
        if sleep_time > 0 and at_target_fps:
            time.sleep(sleep_time)

        self.last_capture = dt.datetime.now()

        # Capture frame
        ret, frame = self.capture.read()
        if not ret:
            logger.warning(
                "Failed to capture frame from camera", datetime=self.last_capture
            )
            return None

        # Resize frame if needed
        if self.frame_shape and frame.shape != self.frame_shape:
            frame = cv2.resize(frame, self.frame_shape)

        # Write frame to stream writer if it exists
        if self.stream_writer:
            self.stream_writer.write(frame)

        # Create frame object
        frame_obj = Frame(
            frame_data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            timestamp=dt.datetime.now(),
            frame_idx=self.frame_idx,
        )
        self.frame_idx += 1
        return frame_obj

    def close(self):
        """Close the camera capture and stream writer."""
        self.capture.release()
        if self.stream_writer:
            self.stream_writer.release()
