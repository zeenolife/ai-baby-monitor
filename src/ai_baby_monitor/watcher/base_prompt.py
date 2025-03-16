def get_instructions_prompt(instructions: list[str]) -> str:
    """Inject the instructions into base prompt"""
    if len(instructions) == 0:
        raise ValueError("Instructions must be a non-empty list")
    
    return (
        "You are given the following instructions: "
        f"{'\n * '.join(instructions)}.\n"
        "If the instructions are violated, you should alert the user.\n"
        "You should also recommend the awareness level based on the image.\n"
        "Please respond with a JSON containing should_alert (boolean), reasoning (string), "
        "and recommended_awareness_level (one of: LOW, MEDIUM, HIGH).\n"
        "Always respond in English, regardless of the content in the images."
    )
