import cv2
from PIL import Image


class CameraStream:
    def __init__(self, uri: str):
        self.cap = cv2.VideoCapture(uri)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")
        
    def capture_image(self):
        """Capture a frame from the camera."""
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture image")
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return img_pil
    