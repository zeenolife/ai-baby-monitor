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
        save_stream_path: str | None = None,
        frame_shape: tuple[int, int] | None = None,
    ):
        """
        Simple camera stream handler.

        Args:
            uri: Camera URI or device index
            save_stream_path: Path to save the stream
            frame_shape: Frame shape to capture. Would be resized to this shape if provided.
        """
        self.frame_shape = frame_shape
        self.capture = self._init_capture(uri)
        self.stream_writer = self._init_stream_writer(save_stream_path)
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
                save_stream_path, fourcc, self.capture.get(cv2.CAP_PROP_FPS), self.frame_shape
            )
            return writer
        else:
            return None

    def capture_new_frame(self) -> Frame | None:
        """Capture a new frame from the camera, only when available. Encode into jpeg."""
        # Check if a new frame is available
        if not self.capture.grab():
            logger.warning("No frame available to grab")
            return None
            
        # Retrieve the grabbed frame
        ret, frame = self.capture.retrieve()
        if not ret:
            logger.warning("Failed to retrieve grabbed frame")
            return None
        
        timestamp = dt.datetime.now()
            
        # Resize frame if needed
        if self.frame_shape and (frame.shape[1], frame.shape[0]) != self.frame_shape:
            frame = cv2.resize(frame, self.frame_shape)

        # Write frame to stream writer if it exists
        if self.stream_writer:
            self.stream_writer.write(frame)
            
        # Convert into jpeg
        ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        if not ret:
            logger.warning("Failed to encode frame as JPEG")
            return None

        # Create frame object
        frame_obj = Frame(
            frame_data=jpeg,
            timestamp=timestamp,
            frame_idx=self.frame_idx,
        )
        
        self.frame_idx += 1
        return frame_obj

    def close(self):
        """Close the camera capture and stream writer."""
        self.capture.release()
        if self.stream_writer:
            self.stream_writer.release()
