UNKNOWN_VALUES = {"", "UNKNOWN", "NONE", "NULL", "N/A", "미상", "모름", "알수없음", "알 수 없음"}

SPECIES_LABELS = {
    "DOG": "강아지",
    "CAT": "고양이",
}

GENDER_LABELS = {
    "MALE": "수컷",
    "FEMALE": "암컷",
}

NEUTERED_LABELS = {
    "YES": "중성화 완료",
    "NO": "중성화 미완료",
}

PROCESS_STATE_LABELS = {
    "PROTECT": "보호 중",
    "ADOPTED": "입양 완료",
    "RETURN": "반환",
    "EUTHANASIA": "안락사",
}


def _clean_value(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() in UNKNOWN_VALUES:
        return None
    return text


def _label(mapping: dict[str, str], value: str | None) -> str | None:
    cleaned = _clean_value(value)
    if cleaned is None:
        return None
    return mapping.get(cleaned.upper(), cleaned)


def _format_animal_summary(animal_data: dict) -> str:
    species = _label(SPECIES_LABELS, animal_data.get("species"))
    breed = _clean_value(animal_data.get("breed"))
    age = _clean_value(animal_data.get("age"))
    weight = _clean_value(animal_data.get("weight"))
    color = _clean_value(animal_data.get("color"))
    gender = _label(GENDER_LABELS, animal_data.get("gender"))
    neutered = _label(NEUTERED_LABELS, animal_data.get("neutered"))
    special_mark = _clean_value(animal_data.get("specialMark"))
    process_state = _label(PROCESS_STATE_LABELS, animal_data.get("processState"))

    identity_parts = [part for part in [age, breed or species] if part]
    summary_parts = []
    if identity_parts:
        summary_parts.append(f"이 보호동물은 {' '.join(identity_parts)}입니다.")
    elif species:
        summary_parts.append(f"이 보호동물은 {species}입니다.")

    details = []
    if weight:
        details.append(f"체중은 {weight}")
    if color:
        details.append(f"색상은 {color}")
    if gender:
        details.append(f"성별은 {gender}")
    if neutered:
        details.append(f"중성화 정보는 {neutered}")
    if details:
        summary_parts.append(", ".join(details) + "입니다.")

    if special_mark:
        summary_parts.append(f"특징은 {special_mark}입니다.")
    if process_state:
        summary_parts.append(f"현재 공고 상태는 {process_state}입니다.")

    return " ".join(summary_parts) if summary_parts else "제공된 보호동물 상세 정보가 없습니다."


def build_prompt(animal_context, recent_messages, question: str) -> str:
    animal_data = animal_context.model_dump()
    context_lines = [
        f"{key}: {value}"
        for key, value in animal_data.items()
        if _clean_value(value) is not None
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
        "입양 가능 여부를 판단하거나 보호소를 대신해 확정적으로 말하지 마세요.",
        "의학적 진단이나 치료 판단을 제공하지 마세요.",
        "답변은 실용적이고, 사용자의 실제 질문에 대한 답을 먼저 제시하세요.",
        (
            "safetyNotice는 서버가 별도로 붙입니다. 보호소 문의, 수의사 상담 같은 "
            "일반적인 안전 고지 문구를 반복하지 마세요. 다만 사용자의 질문이 알 수 없는 사실, "
            "의학적 판단, 입양 절차, 최종 확인을 요구할 때만 짧게 확인 필요성을 언급할 수 있습니다."
        ),
        (
            "질문 유형별 답변 가이드:\n"
            "- 돌봄 방법/생활 질문: 생활환경, 적응 기간, 식사, 배변, 산책/놀이, 관찰 신호를 중심으로 답하세요.\n"
            "- 입양 전 확인 질문: 보호소에 확인할 질문 목록을 구체적으로 제안하되, 건강, 성격, 사회성, 기존 생활 습관, 산책/목줄 적응, 중성화 여부를 포함하세요.\n"
            "- 건강 질문: 진단하지 말고 공고 정보에서 확인 가능한 내용과 관찰할 신호, 병원 확인이 필요한 상황을 구분하세요.\n"
            "- 비용/준비물 질문: 초기 준비물과 반복 관리 항목을 우선순위로 정리하세요.\n"
            "- 성격/적응 질문: 새 환경 적응, 가족 구성원, 기존 반려동물, 분리불안 관찰을 중심으로 답하세요."
        ),
        (
            "너무 긴 한 문단으로 쓰지 말고, 답변이 길어질 때만 의미 단위로 문단을 나누세요. "
            "문장마다 줄바꿈하지 마세요."
        ),
        "전체 답변은 실용적으로 3~5문장 정도로 간결하게 작성하세요.",
        "보호동물 공고 요약:",
        _format_animal_summary(animal_data),
        "보호동물 공고 원문 정보:",
        "\n".join(context_lines) if context_lines else "제공된 보호동물 공고 원문 정보가 없습니다.",
    ]
    if message_lines:
        sections.extend(["최근 대화:", "\n".join(message_lines)])
    sections.extend(["사용자 질문:", question.strip()])
    return "\n\n".join(sections)
