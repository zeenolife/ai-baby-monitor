import pytest
from ai_baby_monitor.watcher.base_prompt import get_instructions_prompt
from ai_baby_monitor.watcher import WatcherResponse, AwarenessLevel
from pydantic import ValidationError
from unittest.mock import patch
from ai_baby_monitor.watcher import Watcher
from ai_baby_monitor.stream import Frame
import datetime

def test_get_instructions_prompt_success():
    instructions = ["Instruction 1", "Instruction 2"]
    expected_prompt = """
You are given the following instructions: 
* Instruction 1
* Instruction 2

If the instructions are violated, you should alert the user.
You should also recommend the awareness level based on the image.
Please generate a structured response in raw JSON format:
- should_alert (boolean)
- reasoning (string)
- recommended_awareness_level (Enum AwarenessLevel; one of: LOW, MEDIUM, HIGH)
Always respond in English, regardless of the content in the images.
        """
    assert get_instructions_prompt(instructions) == expected_prompt

def test_get_instructions_prompt_empty_list():
    with pytest.raises(ValueError, match="Instructions must be a non-empty list"):
        get_instructions_prompt([])

def test_get_instructions_prompt_single_instruction():
    instructions = ["Single instruction"]
    expected_prompt = """
You are given the following instructions: 
* Single instruction

If the instructions are violated, you should alert the user.
You should also recommend the awareness level based on the image.
Please generate a structured response in raw JSON format:
- should_alert (boolean)
- reasoning (string)
- recommended_awareness_level (Enum AwarenessLevel; one of: LOW, MEDIUM, HIGH)
Always respond in English, regardless of the content in the images.
        """
    assert get_instructions_prompt(instructions) == expected_prompt 

def test_watcher_response_valid():
    data = {
        "should_alert": True,
        "reasoning": "Baby is crying",
        "recommended_awareness_level": "HIGH"
    }
    response = WatcherResponse(**data)
    assert response.should_alert is True
    assert response.reasoning == "Baby is crying"
    assert response.recommended_awareness_level == AwarenessLevel.HIGH

def test_watcher_response_invalid_awareness_level():
    data = {
        "should_alert": False,
        "reasoning": "Baby is sleeping",
        "recommended_awareness_level": "VERY_LOW"  # Invalid level
    }
    with pytest.raises(ValidationError):
        WatcherResponse(**data)

def test_watcher_response_missing_field():
    data = {
        "should_alert": True,
        "reasoning": "Baby is awake"
        # missing recommended_awareness_level
    }
    with pytest.raises(ValidationError):
        WatcherResponse(**data) 

# Tests for Watcher class
@patch('ai_baby_monitor.watcher.watcher.get_instructions_prompt')
def test_watcher_init(mock_get_instructions_prompt):
    mock_get_instructions_prompt.return_value = "Mocked Instructions Prompt"
    instructions = ["Don't cry"]
    
    watcher = Watcher(instructions=instructions, vllm_host="test_host", vllm_port=1234, model_name="test_model")

    mock_get_instructions_prompt.assert_called_once_with(instructions)
    assert watcher.instructions_prompt == "Mocked Instructions Prompt"
    assert watcher.json_schema == WatcherResponse.model_json_schema()
    assert watcher.vllm_host == "test_host"
    assert watcher.vllm_port == 1234
    assert watcher.model_name == "test_model"
    assert str(watcher.client.base_url) == "http://test_host:1234/v1/"

# Test for Watcher._calculate_fps
def test_watcher_calculate_fps_valid():
    watcher = Watcher(instructions=["Test"])
    now = datetime.datetime.now()
    frames = [
        Frame(frame_data=[], timestamp=now, frame_idx=0),
        Frame(frame_data=[], timestamp=now + datetime.timedelta(seconds=1), frame_idx=1),
        Frame(frame_data=[], timestamp=now + datetime.timedelta(seconds=2), frame_idx=2), # 3 frames, 2 seconds diff -> 1 FPS
    ]
    assert watcher._calculate_fps(frames) == 1

def test_watcher_calculate_fps_default_fps_param():
    watcher = Watcher(instructions=["Test"])
    now = datetime.datetime.now()
    frames = [
        Frame(frame_data=[], timestamp=now, frame_idx=0),
        Frame(frame_data=[], timestamp=now + datetime.timedelta(seconds=1), frame_idx=1),
        Frame(frame_data=[], timestamp=now + datetime.timedelta(seconds=2), frame_idx=2),
    ]
    assert watcher._calculate_fps(frames, default_fps=5) == 1 # Should still calculate correctly


def test_watcher_calculate_fps_too_few_frames():
    watcher = Watcher(instructions=["Test"])
    frames = [Frame(frame_data=[], timestamp=datetime.datetime.now(), frame_idx=0)]
    assert watcher._calculate_fps(frames, default_fps=3) == 3

def test_watcher_calculate_fps_zero_time_diff():
    watcher = Watcher(instructions=["Test"])
    now = datetime.datetime.now()
    frames = [
        Frame(frame_data=[], timestamp=now, frame_idx=0),
        Frame(frame_data=[], timestamp=now, frame_idx=1)
    ]
    assert watcher._calculate_fps(frames, default_fps=4) == 4
    
def test_watcher_calculate_fps_negative_time_diff(): # Should also use default
    watcher = Watcher(instructions=["Test"])
    now = datetime.datetime.now()
    frames = [
        Frame(frame_data=[], timestamp=now + datetime.timedelta(seconds=1), frame_idx=0),
        Frame(frame_data=[], timestamp=now, frame_idx=1)
    ]
    assert watcher._calculate_fps(frames, default_fps=4) == 4


def test_watcher_calculate_fps_unreasonable_low_fps():
    watcher = Watcher(instructions=["Test"])
    now = datetime.datetime.now()
    frames = [
        Frame(frame_data=[], timestamp=now, frame_idx=0),
        Frame(frame_data=[], timestamp=now + datetime.timedelta(seconds=1000), frame_idx=1), # (2-1)/1000 = 0.001 FPS
    ]
    assert watcher._calculate_fps(frames, default_fps=2) == 2

def test_watcher_calculate_fps_unreasonable_high_fps():
    watcher = Watcher(instructions=["Test"])
    now = datetime.datetime.now()
    frames = [
        Frame(frame_data=[], timestamp=now, frame_idx=0),
        Frame(frame_data=[], timestamp=now + datetime.timedelta(microseconds=100), frame_idx=1), # (2-1)/0.0001 = 10000 FPS
    ]
    assert watcher._calculate_fps(frames, default_fps=2) == 2
    
@patch('ai_baby_monitor.watcher.watcher.logger')
def test_watcher_calculate_fps_exception(mock_logger):
    watcher = Watcher(instructions=["Test"])
    frames = [
        Frame(frame_data=[], timestamp="not a datetime", frame_idx=0), # This will cause an error
        Frame(frame_data=[], timestamp="also not a datetime", frame_idx=1)
    ]
    assert watcher._calculate_fps(frames, default_fps=5) == 5
    called_with_correct_message = False
    for call_args in mock_logger.warning.call_args_list:
        args, kwargs = call_args
        if args[0] == "FPS calculation error" and isinstance(kwargs.get('error'), TypeError):
            called_with_correct_message = True
            break
    assert called_with_correct_message, "logger.warning not called with expected FPS calculation error"
