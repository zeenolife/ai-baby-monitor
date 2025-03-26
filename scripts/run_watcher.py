import argparse
import logging
import time
from typing import List

from ai_baby_monitor import RedisStreamHandler, Watcher, AwarenessLevel

logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s] %(asctime)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_watcher(
    redis_frames_stream_key: str,
    redis_logs_stream_key: str,
    redis_host: str,
    redis_port: int,
    instructions: List[str],
    vllm_host: str,
    vllm_port: int,
    model_name: str,
    num_frames_to_process: int,
):
    """
    Run the Watcher continuously to monitor frames from Redis stream.
    
    Args:
        redis_frames_stream_key: Base Redis stream key (will use {key}:subsampled)
        redis_logs_stream_key: Redis stream key for logging output
        redis_host: Redis server host
        redis_port: Redis server port
        instructions: List of monitoring instructions to check
        vllm_host: vLLM server host
        vllm_port: vLLM server port
        model_name: Model name to use for inference
        num_frames_to_process: Number of frames to analyze in each batch
    """
    # Initialize Redis stream handler
    redis_handler = RedisStreamHandler(
        redis_host=redis_host,
        redis_port=redis_port,
    )
    
    # Initialize Watcher
    nanny_watcher = Watcher(
        instructions=instructions,
        vllm_host=vllm_host,
        vllm_port=vllm_port,
        model_name=model_name,
    )
    
    # Subsampled stream key
    subsampled_key = f"{redis_frames_stream_key}:subsampled"
    
    logger.info(f"Starting Watcher monitoring Redis stream: {subsampled_key}")
    logger.info(f"Using model: {model_name} on {vllm_host}:{vllm_port}")
    logger.info(f"Monitoring instructions: {'\n* '.join(instructions)}")
    
    try:
        while True:
            # Get latest frames from Redis
            frames = redis_handler.get_latest_frames(subsampled_key, num_frames_to_process)
            
            if not frames:
                logger.warning(f"No frames available in stream {subsampled_key}")
                time.sleep(0.3)
                continue
                
            # Reverse frames to get chronological order (oldest first)
            frames.reverse()
            
            logger.info(f"Analyzing {len(frames)} frames from stream")
            
            # Process frames with Watcher
            result = nanny_watcher.process_frames(frames)
            
            if result["success"]:
                # Log the result
                alert_status = "🚨 ALERT TRIGGERED" if result["should_alert"] else "✅ No alert needed"
                awareness = result["recommended_awareness_level"]
                
                logger.info(f"{alert_status} - Awareness Level: {awareness}")
                logger.info(f"Reasoning: {result['reasoning']}")
                
                # Stream logs back to Redis
                log_data = {
                    "timestamp": time.time(),
                    "should_alert": int(result["should_alert"]),
                    "awareness_level": awareness.value if isinstance(awareness, AwarenessLevel) else awareness,
                    "reasoning": result["reasoning"],
                }
                redis_handler.add_to_stream(redis_logs_stream_key, log_data)
                
                if result["should_alert"]:
                    # TODO: Implement alert notification system
                    pass
            else:
                error_msg = f"Error processing frames: {result.get('error', 'Unknown error')}"
                logger.error(error_msg)
                
                # Stream error to Redis
                error_data = {
                    "timestamp": time.time(),
                    "error": error_msg,
                }
                redis_handler.add_to_stream(redis_logs_stream_key, error_data)
                
                time.sleep(0.3)
            
    except KeyboardInterrupt:
        logger.info("Watcher stopped by user")
    except Exception as e:
        error_msg = f"Error in Watcher: {e}"
        logger.error(error_msg)
        
        


def parse_args():
    parser = argparse.ArgumentParser(description="Run Watcher to monitor Redis stream frames.")
    
    parser.add_argument(
        "--redis-frames-stream-key", required=True, help="Base Redis stream key for frames"
    )
    parser.add_argument(
        "--redis-logs-stream-key", required=True, help="Redis stream key for logging output"
    )
    parser.add_argument(
        "--redis-host", default="localhost", help="Redis server host"
    )
    parser.add_argument(
        "--redis-port", default=6379, type=int, help="Redis server port"
    )
    parser.add_argument(
        "--instructions", 
        nargs="+", 
        required=True,
        help="List of monitoring instructions to check"
    )
    parser.add_argument(
        "--vllm-host", default="localhost", help="vLLM server host"
    )
    parser.add_argument(
        "--vllm-port", default=8000, type=int, help="vLLM server port"
    )
    parser.add_argument(
        "--model-name", 
        default="Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
        help="Model name to use for inference"
    )
    parser.add_argument(
        "--num-frames-to-process", 
        default=16, 
        type=int,
        help="Number of frames to analyze in each batch"
    )    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    run_watcher(
        redis_frames_stream_key=args.redis_frames_stream_key,
        redis_logs_stream_key=args.redis_logs_stream_key,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        instructions=args.instructions,
        vllm_host=args.vllm_host,
        vllm_port=args.vllm_port,
        model_name=args.model_name,
        num_frames_to_process=args.num_frames_to_process,
    ) 