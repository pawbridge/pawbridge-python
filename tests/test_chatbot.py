import os
import sys
import types
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

embedding_module = types.ModuleType("app.services.embedding")
embedding_module.extract_embedding_from_url = None
sys.modules["app.services.embedding"] = embedding_module

es_client_module = types.ModuleType("app.es.client")
es_client_module.get_animal_vector = None
es_client_module.save_animal_vector = None
es_client_module.knn_search = None
es_client_module.get_animals_without_vector = None
sys.modules["app.es.client"] = es_client_module

from app.main import app
from app.services.chatbot.gemini_provider import GeminiChatbotProvider
from app.services.chatbot.prompt_builder import build_prompt
from app.services.chatbot.provider import ChatbotProviderUpstreamError


class ChatbotApiTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.payload = {
            "animalContext": {
                "species": "DOG",
                "breed": "믹스견",
                "age": "2023년생",
                "weight": "12kg",
                "color": "갈색",
                "gender": "MALE",
                "neutered": "UNKNOWN",
                "specialMark": "순한 편",
                "processState": "PROTECT",
            },
            "recentMessages": [],
            "question": "입양 전에 뭘 확인해야 하나요?",
        }

    def post_message(self, headers=None, payload=None):
        return self.client.post(
            "/internal/chatbot/messages",
            headers=headers or {},
            json=payload or self.payload,
        )

    def test_chatbot_message_returns_200_with_valid_internal_api_key(self):
        with patch.dict(os.environ, {"INTERNAL_API_KEY": "test-key", "LLM_PROVIDER": "stub"}, clear=False):
            response = self.post_message(headers={"X-Internal-Api-Key": "test-key"})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["provider"], "stub")
        self.assertTrue(body["answer"])
        self.assertTrue(body["safetyNotice"])

    def test_chatbot_message_returns_401_without_internal_api_key_header(self):
        with patch.dict(os.environ, {"INTERNAL_API_KEY": "test-key", "LLM_PROVIDER": "stub"}, clear=False):
            response = self.post_message()

        self.assertEqual(response.status_code, 401)

    def test_chatbot_message_returns_401_with_invalid_internal_api_key(self):
        with patch.dict(os.environ, {"INTERNAL_API_KEY": "test-key", "LLM_PROVIDER": "stub"}, clear=False):
            response = self.post_message(headers={"X-Internal-Api-Key": "wrong-key"})

        self.assertEqual(response.status_code, 401)

    def test_chatbot_message_returns_500_when_internal_api_key_is_not_configured(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "stub"}, clear=True):
            response = self.post_message(headers={"X-Internal-Api-Key": "test-key"})

        self.assertEqual(response.status_code, 500)

    def test_chatbot_message_returns_422_for_blank_question(self):
        payload = {**self.payload, "question": "   "}
        with patch.dict(os.environ, {"INTERNAL_API_KEY": "test-key", "LLM_PROVIDER": "stub"}, clear=False):
            response = self.post_message(headers={"X-Internal-Api-Key": "test-key"}, payload=payload)

        self.assertEqual(response.status_code, 422)

    def test_chatbot_message_returns_422_when_recent_messages_exceed_limit(self):
        payload = {
            **self.payload,
            "recentMessages": [{"role": "user", "content": f"message {idx}"} for idx in range(7)],
        }
        with patch.dict(os.environ, {"INTERNAL_API_KEY": "test-key", "LLM_PROVIDER": "stub"}, clear=False):
            response = self.post_message(headers={"X-Internal-Api-Key": "test-key"}, payload=payload)

        self.assertEqual(response.status_code, 422)

    def test_chatbot_message_returns_500_for_gemini_provider_without_api_key(self):
        with patch.dict(os.environ, {"INTERNAL_API_KEY": "test-key", "LLM_PROVIDER": "gemini"}, clear=True):
            response = self.post_message(headers={"X-Internal-Api-Key": "test-key"})

        self.assertEqual(response.status_code, 500)

    def test_chatbot_message_returns_200_for_gemini_provider_with_mocked_call(self):
        with (
            patch.dict(
                os.environ,
                {
                    "INTERNAL_API_KEY": "test-key",
                    "LLM_PROVIDER": "gemini",
                    "GEMINI_API_KEY": "gemini-key",
                },
                clear=False,
            ),
            patch(
                "app.services.chatbot.gemini_provider.GeminiChatbotProvider.generate_answer",
                new=AsyncMock(return_value="Gemini answer"),
            ),
        ):
            response = self.post_message(headers={"X-Internal-Api-Key": "test-key"})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["provider"], "gemini")
        self.assertEqual(body["answer"], "Gemini answer")

    def test_gemini_provider_rejects_truncated_response(self):
        response_body = {"candidates": [{"finishReason": "MAX_TOKENS"}]}

        with self.assertRaises(ChatbotProviderUpstreamError):
            GeminiChatbotProvider._raise_if_response_truncated(response_body)

    def test_prompt_contains_readable_animal_summary(self):
        class AnimalContext:
            def model_dump(self):
                return {
                    "species": "DOG",
                    "breed": "믹스견",
                    "age": "2025년생",
                    "weight": "14.4kg",
                    "color": "갈색",
                    "gender": "FEMALE",
                    "neutered": "UNKNOWN",
                    "specialMark": "흰색 목줄",
                    "processState": "PROTECT",
                }

        prompt = build_prompt(AnimalContext(), [], "어떤 준비가 필요해요?")

        self.assertIn("보호동물 공고 요약:", prompt)
        self.assertIn("이 보호동물은 2025년생 믹스견입니다.", prompt)
        self.assertIn("체중은 14.4kg", prompt)
        self.assertIn("색상은 갈색", prompt)
        self.assertIn("성별은 암컷", prompt)
        self.assertIn("특징은 흰색 목줄입니다.", prompt)
        self.assertNotIn("중성화 정보는 UNKNOWN", prompt)

    def test_prompt_contains_question_guidance(self):
        class AnimalContext:
            def model_dump(self):
                return {"species": "CAT", "age": "2025년생"}

        prompt = build_prompt(AnimalContext(), [], "입양 전에 뭘 물어봐야 해요?")

        self.assertIn("질문 유형별 답변 가이드:", prompt)
        self.assertIn("돌봄 방법/생활 질문", prompt)
        self.assertIn("입양 전 확인 질문", prompt)
        self.assertIn("건강 질문", prompt)
        self.assertIn("비용/준비물 질문", prompt)
        self.assertIn("성격/적응 질문", prompt)

    def test_prompt_preserves_safety_notice_boundary(self):
        class AnimalContext:
            def model_dump(self):
                return {"species": "CAT", "age": "2025년생"}

        prompt = build_prompt(AnimalContext(), [], "이 고양이 어떻게 키워요?")

        self.assertIn("safetyNotice는 서버가 별도로 붙입니다.", prompt)
        self.assertIn("일반적인 안전 고지 문구를 반복하지 마세요", prompt)
        self.assertIn("사용자의 실제 질문에 대한 답을 먼저 제시하세요", prompt)
        self.assertIn("확인 필요성을 언급할 수 있습니다", prompt)
        self.assertNotIn("보호소나 수의사 확인 권고를 매번", prompt)

    def test_prompt_tells_model_not_to_line_break_every_sentence(self):
        class AnimalContext:
            def model_dump(self):
                return {"species": "CAT", "age": "2025년생"}

        prompt = build_prompt(AnimalContext(), [], "긴 답변은 어떻게 써요?")

        self.assertIn("의미 단위로 문단을 나누세요", prompt)
        self.assertIn("문장마다 줄바꿈하지 마세요", prompt)

    def test_prompt_contains_normal_utf8_korean_contract_strings(self):
        class AnimalContext:
            def model_dump(self):
                return {"species": "DOG", "breed": "믹스견", "age": "2025년생"}

        prompt = build_prompt(AnimalContext(), [], "질문")

        self.assertTrue("보호동물 공고 요약:" in prompt)
        self.assertTrue("질문 유형별 답변 가이드:" in prompt)
        self.assertTrue("문장마다 줄바꿈하지 마세요" in prompt)

    def test_chatbot_message_returns_501_for_openai_provider_in_step_1(self):
        with patch.dict(os.environ, {"INTERNAL_API_KEY": "test-key", "LLM_PROVIDER": "openai"}, clear=False):
            response = self.post_message(headers={"X-Internal-Api-Key": "test-key"})

        self.assertEqual(response.status_code, 501)

    def test_health_endpoint_still_returns_200(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)

    def test_similarity_routes_are_still_registered(self):
        paths = {route.path for route in app.routes}

        self.assertIn("/api/v1/animals/similar", paths)
        self.assertIn("/api/v1/animals/batch/embeddings", paths)


if __name__ == "__main__":
    unittest.main()
