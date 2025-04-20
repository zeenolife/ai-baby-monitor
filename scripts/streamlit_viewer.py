import argparse
import io
import time

import streamlit as st
import structlog
from PIL import Image

from ai_baby_monitor import RedisStreamHandler

logger = structlog.get_logger()
# Set page config
st.set_page_config(
    page_title="Baby Monitor Stream Viewer",
    page_icon="👶",
    layout="wide",
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Baby Monitor Stream Viewer")
    parser.add_argument("--redis-stream-key", required=True, help="Redis stream key")
    parser.add_argument("--redis-host", default="localhost", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")

    return parser.parse_known_args()[0]


def main():
    # Parse command line arguments
    args = parse_args()

    # Get query parameters
    query_params = st.query_params

    # App title and description
    st.title("Baby Monitor Stream Viewer")
    st.markdown("View the live camera stream from your baby monitor")

    # Sidebar for configuration
    with st.sidebar:
        st.header("Stream Configuration")

        redis_stream_key = st.text_input(
            "Room Name",
            value=query_params.get("redis_stream_key", args.redis_stream_key),
        )

    redis_handler = RedisStreamHandler(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
    )

    # Create placeholder for the video stream
    stream_placeholder = st.empty()
    info_placeholder = st.empty()

    # Main display loop
    while True:
        # Get the latest frame
        start_time = time.time()
        frames = redis_handler.get_latest_frames(
            f"{redis_stream_key}:realtime", count=1
        )
        end_time = time.time()
        logger.info(
            "Time taken to get frames",
            time_taken=end_time - start_time,
        )

        if not frames:
            with stream_placeholder.container():
                st.warning("No frames available in the stream")
            time.sleep(0.01)
            continue

        frame = frames[0]

        # Display the frame - decode the JPEG data first
        try:
            # Convert the numpy array containing JPEG data to bytes
            jpeg_bytes = bytes(frame.frame_data)

            # Open the JPEG bytes as an image
            image = Image.open(io.BytesIO(jpeg_bytes))

            with stream_placeholder.container():
                st.image(image, use_container_width=True)

            # Display frame information
            info_text = f"Timestamp: {frame.timestamp}\n"
            info_text += f"Frame Index: {frame.frame_idx}\n"

            with info_placeholder.container():
                st.text(info_text)

        except Exception as e:
            logger.error("Error displaying frame", error=e)
            with stream_placeholder.container():
                st.error(f"Error displaying frame: {e}")
            time.sleep(0.01)


if __name__ == "__main__":
    main()
