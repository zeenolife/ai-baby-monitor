import argparse
import io
import time

import streamlit as st
import structlog
from PIL import Image

from ai_baby_monitor import RedisStreamHandler
from ai_baby_monitor.config import load_multiple_room_configs

logger = structlog.get_logger()
# Set page config
st.set_page_config(
    page_title="Baby Monitor Stream Viewer",
    page_icon="👶",
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Baby Monitor Stream Viewer")
    parser.add_argument(
        "--config-files",
        nargs="+",  # Accept one or more config files
        required=True,
        help="Paths to room configuration YAML files",
    )
    parser.add_argument("--redis-host", default="localhost", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")

    return parser.parse_known_args()[0]

args = parse_args()

redis_handler = RedisStreamHandler(
    redis_host=args.redis_host,
    redis_port=args.redis_port,
)

room_configs = load_multiple_room_configs(args.config_files)
if not room_configs:
    st.error("No valid room configurations found. Please check the config files.")



def main():

    room_names = list(room_configs.keys())
    default_room = room_names[0]

    # App title and description
    st.title("Baby Monitor Stream Viewer")

    # Sidebar for configuration
    with st.sidebar:
        st.header("Room Configuration")

        selected_room_name = st.selectbox(
            "Select Room",
            options=room_names,
            index=room_names.index(default_room),
        )

        # Display config summary
        if selected_room_name and selected_room_name in room_configs:
            selected_config = room_configs[selected_room_name]
            st.caption("Instructions:")
            for instruction in selected_config.instructions:
                st.markdown(f"- {instruction}")
            st.divider()
            with st.expander("Camera details"):
                st.caption(f"Camera URI:  {selected_config.camera_uri}")
                st.caption(f"Frame width: {selected_config.frame_width}")
                st.caption(f"Frame height: {selected_config.frame_height}")
                st.caption(f"Subsample rate: {selected_config.subsample_rate}")
            st.divider()
            with st.expander("LLM Model"):
                st.caption(f"LLM Model: {selected_config.llm_model_name}")

    # Create placeholder for the video stream
    stream_placeholder = st.empty()
    info_placeholder = st.empty()

    # Main display loop
    while True:
        # Get the latest frame
        frames = redis_handler.get_latest_frames(
            f"{selected_room_name}:realtime", count=1
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


            with info_placeholder.container():
                st.text(info_text)

        except Exception as e:
            logger.error("Error displaying frame", error=e)
            with stream_placeholder.container():
                st.error(f"Error displaying frame: {e}")
            time.sleep(0.01)


main()
