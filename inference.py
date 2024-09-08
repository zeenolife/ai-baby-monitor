import argparse
import time
from vllm import LLM
from camera_stream import CameraStream


def parse_args():
    parser = argparse.ArgumentParser(
        description="Monitor camera stream"
    )
    parser.add_argument(
        "stream_uri", help="URI for the camera stream"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    llm = LLM(model="llava-hf/llava-1.5-7b-hf")
    stream = CameraStream(args.stream_uri)
    prompt = "USER: <image>\nIs baby in the photo doing anything dangerous?\nASSISTANT:"

    for _ in range(20):
        start_time = time.time()
        image = stream.capture_image()
        outputs = llm.generate(
            {
                "prompt": prompt,
                "multi_modal_data": {"image": image},
            }
        )
        for o in outputs:
            generated_text = o.outputs[0].text
            print(generated_text)

        print(f"Processing time: {time.time() - start_time}")


if __name__ == "__main__":
    main()