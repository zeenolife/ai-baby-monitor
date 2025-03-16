import base64
import json
import logging
from enum import Enum
from pydantic import BaseModel, ValidationError

from openai import OpenAI

from ai_baby_monitor.stream.camera_stream import Frame

logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s] %(asctime)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AwarenessLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class WatcherResponse(BaseModel):
    should_alert: bool
    reasoning: str
    recommended_awareness_level: AwarenessLevel


class Watcher:
    def __init__(
        self,
        instructions: list[str],
        vllm_host: str = "localhost",
        vllm_port: int = 8000,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
    ):
        """
        Initialize the Watcher with instructions and vLLM server details.

        Args:
            instructions: List of monitoring instructions to check against frames
            vllm_host: Hostname of the vLLM server
            vllm_port: Port of the vLLM server
            model_name: Name of the model to use for inference
        """
        self.instructions = instructions
        self.vllm_host = vllm_host
        self.vllm_port = vllm_port
        self.model_name = model_name

        # Initialize OpenAI client for vLLM server
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=f"http://{vllm_host}:{vllm_port}/v1",
        )

        logger.info(f"Initialized Watcher with model {model_name}")
        logger.info(f"Connected to vLLM server at {vllm_host}:{vllm_port}")
        logger.info(f"Monitoring instructions: {', '.join(instructions)}")

    def _frames_to_base64(self, frames: list[Frame]) -> list[str]:
        """Convert JPEG-encoded frame data to base64 encoded strings."""
        base64_frames = []
        for frame in frames:
            jpeg_bytes = bytes(frame.frame_data)
            base64_str = base64.b64encode(jpeg_bytes).decode("utf-8")
            base64_frames.append(base64_str)

        return base64_frames

    def process_frames(self, frames: list[Frame], fps: int = 2) -> dict[str, str | bool]:
        """
        Process a list of frames through the vLLM model to check for instruction violations.

        Args:
            frames: List of Frame objects to process

        Returns:
            dict: Results of the inference including alerts and reasoning
        """
        if not frames:
            logger.warning("No frames provided for processing")
            return {
                "success": False,
                "error": "No frames provided",
                "should_alert": False,
                "reasoning": "No data to analyze",
                "recommended_awareness_level": "MEDIUM",
                "raw_response": "",
            }

        try:
            # Convert frames to base64
            base64_frames = self._frames_to_base64(frames)

            # Create video URL with proper format for vLLM
            video_url = f"data:video/jpeg;base64,{','.join(base64_frames)}"

            # Create instruction text
            instruction_text = (
                "You are given the following instructions: "
                f"{', '.join(self.instructions)}.\n"
                "If the instructions are violated, you should alert the user.\n"
                "You should also recommend the awareness level based on the image.\n"
                "Please respond with a JSON containing should_alert (boolean), reasoning (string), "
                "and recommended_awareness_level (one of: LOW, MEDIUM, HIGH).\n"
                "Always respond in English, regardless of the content in the images."
            )

            # Get JSON schema from Pydantic model
            json_schema = WatcherResponse.model_json_schema()

            # Create message with instructions
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "video_url", "video_url": {"url": video_url}},
                        {"type": "text", "text": instruction_text},
                    ],
                },
            ]

            # Send to vLLM server with proper mm_processor_kwargs and guided_json
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,  
                max_tokens=512,
                extra_body={
                    "mm_processor_kwargs": {"fps": [fps]},
                    "guided_json": json_schema
                },
            )

            try:
                parsed_response = WatcherResponse.model_validate_json(response.choices[0].message.content)
            except ValidationError as e:
                logger.error(f"Failed to parse response as JSON: {e}")
                logger.error(f"Response text: {response.choices[0].message.content}")
                return {
                    "success": False,
                    "error": str(e),
                    "should_alert": False,
                    "reasoning": "Error in parsing response",
                    "recommended_awareness_level": "MEDIUM",
                }

            # Add success flag and timestamps
            result["success"] = True
            result["timestamps"] = [frame.timestamp.isoformat() for frame in frames]
            result["frame_indices"] = [frame.frame_idx for frame in frames]

            return result

        except Exception as e:
            logger.error(f"Error processing frames: {e}")
            return {
                "success": False,
                "error": str(e),
                "should_alert": False,
                "reasoning": "Error in processing",
                "recommended_awareness_level": "HIGH",  # Default to HIGH on error
            }
