def get_instructions_prompt(instructions: list[str]) -> str:
    """Inject the instructions into base prompt"""
    if len(instructions) == 0:
        raise ValueError("Instructions must be a non-empty list")

    instructions_str = "\n".join(["* " + _ for _ in instructions])

    return f"""
You are given the following instructions: 
{instructions_str}\n
If the instructions are violated, you should alert the user.
You should also recommend the awareness level based on the image.
Please generate a structured response in raw JSON format:
- should_alert (boolean)
- reasoning (string)
- recommended_awareness_level (Enum AwarenessLevel; one of: LOW, MEDIUM, HIGH)
Always respond in English, regardless of the content in the images.
        """