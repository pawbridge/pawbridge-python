import os
import sys
import types
import unittest
from unittest.mock import patch

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
                "specialMark": "온순함",
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

    def test_chatbot_message_returns_501_for_gemini_provider_in_step_1(self):
        with patch.dict(os.environ, {"INTERNAL_API_KEY": "test-key", "LLM_PROVIDER": "gemini"}, clear=False):
            response = self.post_message(headers={"X-Internal-Api-Key": "test-key"})

        self.assertEqual(response.status_code, 501)

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
