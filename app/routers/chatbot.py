from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from app.services.chatbot.service import ChatbotService

router = APIRouter()


class AnimalContext(BaseModel):
    species: str | None = None
    breed: str | None = None
    age: str | None = None
    weight: str | None = None
    color: str | None = None
    gender: str | None = None
    neutered: str | None = None
    specialMark: str | None = None
    processState: str | None = None


class RecentMessage(BaseModel):
    role: str
    content: str

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        if value not in {"user", "assistant"}:
            raise ValueError("role must be 'user' or 'assistant'")
        return value

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("content must not be blank")
        return value


class ChatbotMessageRequest(BaseModel):
    animalContext: AnimalContext
    recentMessages: list[RecentMessage] = Field(default_factory=list, max_length=6)
    question: str

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("question must not be blank")
        return value


class ChatbotMessageResponse(BaseModel):
    answer: str
    safetyNotice: str
    provider: str


def verify_internal_api_key(
    x_internal_api_key: str | None = Header(default=None, alias="X-Internal-Api-Key"),
) -> None:
    ChatbotService.verify_internal_api_key(x_internal_api_key)


@router.post(
    "/messages",
    response_model=ChatbotMessageResponse,
    dependencies=[Depends(verify_internal_api_key)],
)
async def create_chatbot_message(req: ChatbotMessageRequest):
    try:
        return await ChatbotService().create_message(req)
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        ) from exc
