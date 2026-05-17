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
        "당신은 PawBridge의 AI 입양 준비 도우미입니다.",
        "반드시 한국어로 답변하세요.",
        "인사말이나 서비스 이름으로 답변을 시작하지 마세요.",
        "보호동물 공고 정보는 참고 자료로만 사용하세요.",
        "입양 가능 여부를 판단하거나 보호소를 대신해 말하지 마세요.",
        "의학적 진단이나 치료 판단을 제공하지 마세요.",
        "보호소 문의, 수의사 상담 같은 일반적인 안전 고지 문구를 반복하지 마세요. safetyNotice는 서버가 별도로 붙입니다.",
        "사용자의 실제 질문에 대한 답을 먼저 하세요.",
        "너무 긴 한 문단으로 쓰지 말고, 필요한 경우에만 문단을 나누세요.",
        "전체 답변은 실용적으로 3~5문장 정도로 간결하게 작성하세요.",
        "보호동물 공고 정보:",
        "\n".join(context_lines) if context_lines else "No animal context provided.",
    ]
    if message_lines:
        sections.extend(["최근 대화:", "\n".join(message_lines)])
    sections.extend(["사용자 질문:", question.strip()])
    return "\n\n".join(sections)
