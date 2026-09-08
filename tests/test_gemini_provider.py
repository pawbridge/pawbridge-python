import json
import os
import unittest
from unittest.mock import patch

import httpx

from app.services.chatbot.gemini_provider import GeminiChatbotProvider


class GeminiProviderTest(unittest.IsolatedAsyncioTestCase):
    def test_missing_or_blank_model_uses_stable_default(self):
        for environment in ({}, {"GEMINI_MODEL": "  "}):
            with self.subTest(environment=environment), patch.dict(os.environ, environment, clear=True):
                self.assertEqual(GeminiChatbotProvider().model, "gemini-3.5-flash-lite")

    async def test_request_uses_selected_model_without_legacy_thinking_budget(self):
        for override in (None, " custom-model "):
            environment = {"GEMINI_API_KEY": "test-only-key"}
            if override is not None:
                environment["GEMINI_MODEL"] = override
            expected_model = override.strip() if override else "gemini-3.5-flash-lite"
            requests = []

            def respond(request):
                requests.append(request)
                return httpx.Response(200, json={
                    "candidates": [{
                        "finishReason": "STOP",
                        "content": {"parts": [{"text": "test answer"}]},
                    }],
                })

            client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
            with (
                self.subTest(model=expected_model),
                patch.dict(os.environ, environment, clear=True),
                patch("app.services.chatbot.gemini_provider.httpx.AsyncClient", return_value=client),
            ):
                answer = await GeminiChatbotProvider().generate_answer("test prompt")

                self.assertEqual(answer, "test answer")
                self.assertEqual(len(requests), 1)
                request = requests[0]
                self.assertEqual(request.method, "POST")
                self.assertEqual(str(request.url),
                    f"https://generativelanguage.googleapis.com/v1beta/models/{expected_model}:generateContent")
                self.assertEqual(request.headers["x-goog-api-key"], "test-only-key")
                payload = json.loads(request.content)
                self.assertEqual(payload["contents"], [
                    {"role": "user", "parts": [{"text": "test prompt"}]},
                ])
                self.assertEqual(payload["generationConfig"], {
                    "temperature": 0.4,
                    "maxOutputTokens": 1024,
                })
