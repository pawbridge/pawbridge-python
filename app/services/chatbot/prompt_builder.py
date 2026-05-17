def build_prompt(animal_context, recent_messages, question: str) -> str:
    context_lines = [
        f"{key}: {value}"
        for key, value in animal_context.model_dump().items()
        if value is not None and str(value).strip()
    ]
    message_lines = [
        f"{message.role}: {message.content}"
        for message in recent_messages
    ]

    sections = [
        "You are PawBridge's adoption-check assistant.",
        "Answer in Korean.",
        "Use the animal notice context only as reference information.",
        "Do not decide adoption eligibility or speak as the shelter.",
        "Do not provide medical diagnosis or treatment decisions.",
        "Recommend checking important details with the shelter or a veterinarian.",
        "Keep the answer practical and concise, about 3 to 6 sentences.",
        "Animal context:",
        "\n".join(context_lines) if context_lines else "No animal context provided.",
    ]
    if message_lines:
        sections.extend(["Recent messages:", "\n".join(message_lines)])
    sections.extend(["User question:", question.strip()])
    return "\n\n".join(sections)
