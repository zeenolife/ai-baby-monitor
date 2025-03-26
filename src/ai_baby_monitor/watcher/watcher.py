import base64
import logging
from enum import Enum
from pydantic import BaseModel, ValidationError

from openai import OpenAI

from ai_baby_monitor.stream.camera_stream import Frame
from ai_baby_monitor.watcher.base_prompt import get_instructions_prompt
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
        self.instructions_prompt = get_instructions_prompt(instructions)
        self.json_schema = WatcherResponse.model_json_schema()
        
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

    def _calculate_fps(self, frames: list[Frame], default_fps: int = 2) -> int:
        """Calculate FPS from frame timestamps, defaulting to 2 if calculation fails."""
        if len(frames) < 2:
            logger.warning("Too few frames to calculate FPS, using default of 2")
            return default_fps
            
        try:
            time_diff = (frames[-1].timestamp - frames[0].timestamp).total_seconds()
            if time_diff <= 0:
                return default_fps
                
            fps = round((len(frames) - 1) / time_diff)
            
            # Return default if calculated FPS is unreasonable
            return default_fps if fps < 0.02 or fps > 60 else fps
            
        except Exception as e:
            logger.warning(f"FPS calculation error: {e}")
            return default_fps

    def process_frames(self, frames: list[Frame], fps: int | None = None) -> dict[str, str | bool]:
        """Process frames to detect instruction violations.

        Returns:
            dict containing:
                - success (bool): Whether processing completed successfully
                - should_alert (bool): If an alert should be triggered
                - reasoning (str): Explanation for the decision
                - recommended_awareness_level (AwarenessLevel): Suggested monitoring level
                - raw_response (str): Original model response
                - error (str, optional): Error message if processing failed
        """
        if not frames:
            logger.warning("No frames provided for processing")
            return {
                "success": False,
                "error": "No frames provided",
            }

        try:
            # Convert frames to base64
            base64_frames = self._frames_to_base64(frames)

            # Create video URL with proper format for vLLM
            encoded_video = f"data:video/jpeg;base64,{','.join(base64_frames)}"

            # Create message with instructions
            messages = [
                {"role": "system", "content": "You are a helpful assistant and baby sitter."},
                {
                    "role": "user",
                    "content": [
                        {"type": "video_url", "video_url": {"url": encoded_video}},
                        {"type": "text", "text": self.instructions_prompt},
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
                    "mm_processor_kwargs": {"fps": fps or [self._calculate_fps(frames)]},
                    "guided_json": self.json_schema
                },
            )

            try:
                parsed_response = WatcherResponse.model_validate_json(response.choices[0].message.content)
            except ValidationError as e:
                logger.error(f"Failed to parse response as JSON: {e}")
                logger.error(f"Response text: {response.choices[0].message.content}")
                return {
                    "success": False,
                    "error": str(e) + "\nRaw response: " + response.choices[0].message.content,
                }

            # Create result dictionary from parsed response
            return {
                "success": True,
                "should_alert": parsed_response.should_alert,
                "reasoning": parsed_response.reasoning,
                "recommended_awareness_level": parsed_response.recommended_awareness_level,
                "raw_response": response.choices[0].message.content
            }

        except Exception as e:
            logger.error(f"Error processing frames: {e}")
            return {
                "success": False,
                "error": str(e),
            }
