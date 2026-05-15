class StubChatbotProvider:
    name = "stub"

    async def generate_answer(self, prompt: str) -> str:
        return (
            "현재 공고 정보를 기준으로 보호소에 건강 상태, 예방접종 여부, "
            "중성화 여부, 성격과 생활 습관을 확인해 보세요."
        )
