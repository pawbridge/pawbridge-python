import os

import httpx

from app.services.chatbot.provider import (
    ChatbotProvider,
    ChatbotProviderConfigurationError,
    ChatbotProviderUpstreamError,
)


class GeminiChatbotProvider(ChatbotProvider):
    name = "gemini"

    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
        self.timeout_seconds = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "15"))

    async def generate_answer(self, prompt: str) -> str:
        if not self.api_key:
            raise ChatbotProviderConfigurationError("GEMINI_API_KEY is not configured")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": 0.4,
                "maxOutputTokens": 512,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    url,
                    headers={
                        "x-goog-api-key": self.api_key,
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ChatbotProviderUpstreamError("Gemini request timed out") from exc
        except httpx.HTTPStatusError as exc:
            raise ChatbotProviderUpstreamError(
                f"Gemini request failed with status {exc.response.status_code}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ChatbotProviderUpstreamError("Gemini request failed") from exc

        answer = self._extract_answer(response.json())
        if not answer:
            raise ChatbotProviderUpstreamError("Gemini response did not include answer text")
        return answer

    @staticmethod
    def _extract_answer(response_body: dict) -> str:
        candidates = response_body.get("candidates") or []
        if not candidates:
            return ""

        parts = candidates[0].get("content", {}).get("parts", [])
        texts = [
            part.get("text", "")
            for part in parts
            if isinstance(part, dict) and part.get("text")
        ]
        return "\n".join(text.strip() for text in texts if text.strip()).strip()
