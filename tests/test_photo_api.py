import asyncio
import os
import threading
import unittest
from io import BytesIO
from unittest.mock import patch

import httpx
from PIL import Image

from app.photo_main import create_app
from app.photo_optimizer import optimize_photo

PATH = "/internal/photos/optimize"
AUTH = {"X-Internal-API-Key": "test-key"}


class PhotoApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.env = patch.dict(os.environ, {"INTERNAL_API_KEY": "test-key"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app()), base_url="http://test"
        )
        self.addAsyncCleanup(self.client.aclose)
        output = BytesIO()
        Image.new("RGB", (180, 120), "brown").save(output, format="PNG", compress_level=0)
        self.photo = output.getvalue()

    async def test_authorized_binary_request_returns_sniffed_type_and_hashes(self):
        response = await self.client.post(PATH, content=self.photo, headers={**AUTH, "Content-Type": "text/plain"})
        expected = optimize_photo(self.photo)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, expected.data)
        self.assertEqual(response.headers["content-type"], "image/webp")
        self.assertEqual(response.headers["x-source-sha256"], expected.source_sha256)
        self.assertEqual(response.headers["x-stored-sha256"], expected.stored_sha256)
        self.assertEqual(response.headers["x-photo-width"], "180")
        self.assertEqual(response.headers["x-photo-height"], "120")
        self.assertEqual(response.headers["cache-control"], "no-store")

    async def test_authentication_fails_before_reading_body(self):
        async def body():
            raise AssertionError("Unauthorized body must not be read")
            yield b""
        for key in (None, "wrong-key"):
            response = await self.client.post(PATH, content=body(), headers={} if key is None else {"X-Internal-API-Key": key})
            self.assertEqual(response.status_code, 401)

    async def test_missing_server_key_is_unavailable(self):
        with patch.dict(os.environ, {"INTERNAL_API_KEY": ""}):
            response = await self.client.post(PATH, content=self.photo, headers=AUTH)
        self.assertEqual(response.status_code, 503)

    async def test_chunked_body_is_limited_even_without_content_length(self):
        async def body():
            yield b"a" * 8
            yield b"b" * 8
            raise AssertionError("Oversized body must stop reading")
        with patch("app.photo_main.MAX_INPUT_BYTES", 10):
            response = await self.client.post(PATH, content=body(), headers=AUTH)
        self.assertEqual(response.status_code, 413)
        self.assertEqual((await self.client.post(PATH, content=self.photo, headers=AUTH)).status_code, 200)

    async def test_slow_upload_times_out_and_releases_capacity(self):
        async def body():
            await asyncio.sleep(10)
            yield b""
        with patch("app.photo_main.BODY_TIMEOUT_SECONDS", 0.01):
            response = await self.client.post(PATH, content=body(), headers=AUTH)
        self.assertEqual(response.status_code, 408)
        self.assertEqual((await self.client.post(PATH, content=self.photo, headers=AUTH)).status_code, 200)

    async def test_invalid_photo_and_content_encoding_are_explicit_client_errors(self):
        response = await self.client.post(PATH, content=b"invalid", headers=AUTH)
        self.assertEqual(response.status_code, 422)
        response = await self.client.post(PATH, content=self.photo, headers={**AUTH, "Content-Encoding": "gzip"})
        self.assertEqual(response.status_code, 415)

    async def test_busy_worker_rejects_another_body_without_queueing(self):
        started = threading.Event()
        finish = threading.Event()
        expected = optimize_photo(self.photo)

        def blocking_optimizer(data):
            started.set()
            if not finish.wait(3):
                raise TimeoutError("test did not release worker")
            return expected

        async def unread_body():
            raise AssertionError("Busy request must not be buffered")
            yield b""

        with patch("app.photo_main.optimize_photo", side_effect=blocking_optimizer):
            first = asyncio.create_task(self.client.post(PATH, content=self.photo, headers=AUTH))
            try:
                self.assertTrue(await asyncio.to_thread(started.wait, 2))
                response = await self.client.post(PATH, content=unread_body(), headers=AUTH)
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.headers["retry-after"], "3")
                self.assertEqual((await self.client.get("/health")).status_code, 200)
            finally:
                finish.set()
                self.assertEqual((await first).status_code, 200)

    async def test_encoder_failure_is_retryable_and_releases_capacity(self):
        with patch("app.photo_main.optimize_photo", side_effect=OSError("private internal failure")):
            response = await self.client.post(PATH, content=self.photo, headers=AUTH)
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private internal failure", response.text)
        self.assertEqual((await self.client.post(PATH, content=self.photo, headers=AUTH)).status_code, 200)

    async def test_cancelled_request_keeps_capacity_until_cpu_work_finishes(self):
        started = threading.Event()
        finish = threading.Event()
        expected = optimize_photo(self.photo)

        def blocking_optimizer(data):
            started.set()
            if not finish.wait(3):
                raise TimeoutError("test did not release worker")
            return expected

        with patch("app.photo_main.optimize_photo", side_effect=blocking_optimizer):
            first = asyncio.create_task(self.client.post(PATH, content=self.photo, headers=AUTH))
            try:
                self.assertTrue(await asyncio.to_thread(started.wait, 2))
                first.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await first
                response = await self.client.post(PATH, content=self.photo, headers=AUTH)
                self.assertEqual(response.status_code, 503)
            finally:
                finish.set()
            async def capacity_recovers():
                while True:
                    response = await self.client.post(PATH, content=self.photo, headers=AUTH)
                    if response.status_code != 503:
                        return response
                    await asyncio.sleep(0.01)
            self.assertEqual((await asyncio.wait_for(capacity_recovers(), 2)).status_code, 200)


if __name__ == "__main__":
    unittest.main()
