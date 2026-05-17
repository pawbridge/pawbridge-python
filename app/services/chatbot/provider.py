import os
from abc import ABC, abstractmethod


class ChatbotProvider(ABC):
    name: str

    @abstractmethod
    async def generate_answer(self, prompt: str) -> str:
        raise NotImplementedError


class ChatbotProviderConfigurationError(RuntimeError):
    pass


class ChatbotProviderUpstreamError(RuntimeError):
    pass


class NotImplementedChatbotProvider(ChatbotProvider):
    def __init__(self, name: str):
        self.name = name

    async def generate_answer(self, prompt: str) -> str:
        raise NotImplementedError(f"LLM provider '{self.name}' is not implemented in Step 1")


def get_chatbot_provider() -> ChatbotProvider:
    provider_name = os.getenv("LLM_PROVIDER", "stub").strip().lower() or "stub"
    if provider_name == "stub":
        from app.services.chatbot.stub_provider import StubChatbotProvider

        return StubChatbotProvider()
    if provider_name == "gemini":
        from app.services.chatbot.gemini_provider import GeminiChatbotProvider

        return GeminiChatbotProvider()
    if provider_name == "openai":
        return NotImplementedChatbotProvider(provider_name)
    raise RuntimeError(f"Unsupported LLM_PROVIDER: {provider_name}")
