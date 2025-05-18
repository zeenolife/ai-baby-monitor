import datetime as dt
import io
import os

import streamlit as st
from PIL import Image

from ai_baby_monitor.stream import RedisStreamHandler


def display_sidebar(
    room_configs_key: str = "room_configs",
    config_key: str = "selected_config",
    mode_key: str = "selected_mode",
):
    with st.sidebar:
        st.header("Room Configuration")

        selected_config = st.selectbox(
            "Select Room",
            options=list(st.session_state[room_configs_key].values()),
            index=0,
            key=config_key,
        )

        st.radio(
            "Select Mode",
            options=["Real-time stream", "Historic logs"],
            index=0,
            key=mode_key,
        )

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
            st.caption(f"LLM Model: {os.getenv('LLM_MODEL_NAME')}")


@st.cache_resource
def get_cached_redis_handler(redis_host: str, redis_port: int):
    return RedisStreamHandler(
        redis_host=redis_host,
        redis_port=redis_port,
    )


def get_last_image_with_timestamp(
    redis_handler: RedisStreamHandler, room_name: str
) -> tuple[Image.Image | None, dt.datetime | None]:
    frames = redis_handler.get_latest_frames(f"{room_name}:realtime", count=1)

    if not frames:
        return None

    jpeg_bytes = bytes(frames[0].frame_data)
    image = Image.open(io.BytesIO(jpeg_bytes))
    return image, frames[0].timestamp


def fetch_logs(
    redis_handler: RedisStreamHandler,
    room_name: str,
    num_logs: int = 3,
) -> list[dict]:
    logs = redis_handler.get_latest_logs(f"{room_name}:logs", count=num_logs)

    new_logs = []
    for log_id, log_data in logs:
        log_data = redis_handler.deserialize_log(log_data)
        log_data["timestamp"] = dt.datetime.fromtimestamp(float(log_data["timestamp"]))
        new_logs.append(log_data)

    return new_logs


def render_logs(logs: list[dict]):
    
    for log in reversed(logs):
        timestamp = log["timestamp"]
        alert_status = "🚨" if log["should_alert"] == "1" else "🟢"
        awareness_level = log["awareness_level"]
        if awareness_level == "LOW":
            awareness_level = f":green[{awareness_level}]"
        elif awareness_level == "MEDIUM":
            awareness_level = f":yellow[{awareness_level}]"
        else:
            awareness_level = f":red[{awareness_level}]"
        reasoning = log["reasoning"]
        with st.chat_message("ai"):
            st.write(f"Timestamp: {timestamp}")
            st.write(f"Alert status: {alert_status}")
            st.write(f"Awareness level: {awareness_level}")
            st.write(f"Reasoning: {reasoning}")
            st.divider()
