import json
import logging
import time
from enum import Enum

import numpy as np
from vllm import LLM

from ai_baby_monitor.stream.camera_stream import CameraStream, Frame
from ai_baby_monitor.instructions import Instruction

logger = logging.getLogger(__name__)


class AwarenessLevel(Enum):
    LOW = 1  # Relaxed monitoring, lower FPS
    MEDIUM = 2  # Regular monitoring
    HIGH = 3  # High alert, maximum FPS


class Watcher:
    FPS_CONFIG = {
        AwarenessLevel.LOW: 0.5,
        AwarenessLevel.MEDIUM: 1,
        AwarenessLevel.HIGH: 2,
    }

    def __init__(
        self,
        watcher_id: str,
        camera_stream: CameraStream,
        instructions: list[Instruction],
        llm: LLM,
        fps_config: dict[AwarenessLevel, int] = None,
    ):
        """
        Initialize a camera stream watcher.

        Args:
        """
        self.watcher_id = watcher_id
        self.camera_stream = camera_stream
        self.instructions = instructions
        self.llm = llm
        self.current_awareness = AwarenessLevel.MEDIUM
        self.fps_config = fps_config or self.FPS_CONFIG

    def set_awareness_level(self, level: AwarenessLevel):
        """Update the current awareness level."""
        if level != self.current_awareness:
            logger.info(
                f"Changing awareness level from {self.current_awareness} to {level}"
            )
            self.current_awareness = level

    def process_frames(self, frames: list[Frame]) -> list[dict]:
        """
        Get llm input for video frames.
        """
        return np.stack([_.frame_data for _ in frames])

    def try_parse_response(self, response: str) -> dict:
        """
        Try to parse the response from the LLM.
        """
        try:
            parsed = json.loads(response)
            result = {
                "should_alert": parsed.get("should_alert", False),
                "reasoning": parsed.get("reasoning", ""),
                "recommended_awareness_level": parsed.get(
                    "recommended_awareness_level", AwarenessLevel.MEDIUM
                ),
            }
            return result
        except json.JSONDecodeError:
            return {}

    def watch(self):
        """
        Main monitoring loop.
        Gets frames based on current awareness level and processes them.
        """
        while True:
            current_fps = self.fps_config[self.current_awareness]
            frames = self.camera_stream.get_latest_n_seconds(n=6, fps=current_fps)
            if not frames:
                logger.warning(f"No frame available for watcher {self.watcher_id}")
                continue

            start_time = time.time()
            responses = []
            for instruction in self.instructions:
                prompt = instruction.get_instruction_prompt()
                data = self.process_frames(frames)
                inputs = {
                    "prompt": prompt,
                    "multi_modal_data": {"video": data},
                }

                for _ in range(2):  # Try 2 times to get a valid response
                    response = self.llm.generate(inputs)
                    for o in response:
                        generated_text = o.outputs[0].text
                    parsed = self.try_parse_response(generated_text)
                    if parsed:
                        responses.append(parsed)
                        break
            alerts = [_["reasoning"] for _ in responses if _["should_alert"]]
            if alerts:
                logger.info(f"Alerts: {alerts}")

            # Adjust awareness level based on responses
            awareness_level = max(
                [_["recommended_awareness_level"] for _ in responses]
                + [self.current_awareness]
            )
            self.set_awareness_level(awareness_level)

            # Sleep to maintain desired FPS
            elapsed_time = time.time() - start_time
            time.sleep(max(0, 1.0 / current_fps - elapsed_time))
