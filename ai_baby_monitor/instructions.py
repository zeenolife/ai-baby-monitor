from dataclasses import dataclass


@dataclass
class Instruction:
    instruction: str

    def get_instruction_prompt(self) -> str:
        base_prompt: str = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>"
            f"You are given the following instruction: {self.instruction}\n"
            "If the instructions are violated, you should alert the user.\n"
            "You should also recommend the awareness level based on the image.\n"
            "Please, respond with following python dict format:\n"
            '{"should_alert": boolean, "reasoning": string, "recommended_awareness_level": "LOW" | "MEDIUM" | "HIGH"}\n'
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        return base_prompt
