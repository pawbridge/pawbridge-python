import os

from fastapi import HTTPException, status

from app.services.chatbot.prompt_builder import build_prompt
from app.services.chatbot.provider import get_chatbot_provider

SAFETY_NOTICE = (
    "이 답변은 일반적인 참고 정보이며, 정확한 건강 상태나 치료 판단은 "
    "동물병원 또는 수의사에게 확인해 주세요."
)


class ChatbotService:
    @staticmethod
    def verify_internal_api_key(api_key: str | None) -> None:
        configured_key = os.getenv("INTERNAL_API_KEY")
        if configured_key is None or not configured_key.strip():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="INTERNAL_API_KEY is not configured",
            )
        if api_key != configured_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid internal API key",
            )

    async def create_message(self, req):
        provider = get_chatbot_provider()
        prompt = build_prompt(req.animalContext, req.recentMessages, req.question)
        answer = await provider.generate_answer(prompt)
        return {
            "answer": answer,
            "safetyNotice": SAFETY_NOTICE,
            "provider": provider.name,
        }
