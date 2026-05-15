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
        "Animal context:",
        "\n".join(context_lines) if context_lines else "No animal context provided.",
    ]
    if message_lines:
        sections.extend(["Recent messages:", "\n".join(message_lines)])
    sections.extend(["User question:", question.strip()])
    return "\n\n".join(sections)
