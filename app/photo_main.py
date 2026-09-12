"""Separate CPU process: uvicorn app.photo_main:app --workers 1."""

import asyncio
import hmac
import os

import anyio
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response

from app.photo_optimizer import InvalidPhoto, MAX_INPUT_BYTES, PhotoTooLarge, optimize_photo

BODY_TIMEOUT_SECONDS = 15


def authenticate(x_internal_api_key: str | None = Header(None)):
    expected = os.getenv("INTERNAL_API_KEY", "")
    if not expected:
        raise HTTPException(503, "Photo service is not configured")
    if not x_internal_api_key or not hmac.compare_digest(
        x_internal_api_key.encode(), expected.encode()
    ):
        raise HTTPException(401, "Authentication required")


def create_app() -> FastAPI:
    app = FastAPI(title="PawBridge Photo Optimizer", docs_url=None, redoc_url=None, openapi_url=None)
    gate = anyio.Semaphore(1)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/internal/photos/optimize", dependencies=[Depends(authenticate)])
    async def optimize(request: Request):
        try:
            gate.acquire_nowait()
        except anyio.WouldBlock as exc:
            raise HTTPException(503, "Photo service is busy", headers={"Retry-After": "3"}) from exc
        release_here = True
        try:
            if request.headers.get("content-encoding", "identity") != "identity":
                raise HTTPException(415, "Encoded request bodies are not supported")
            body = bytearray()
            try:
                with anyio.fail_after(BODY_TIMEOUT_SECONDS):
                    async for chunk in request.stream():
                        if len(body) + len(chunk) > MAX_INPUT_BYTES:
                            raise HTTPException(413, "Photo exceeds 10 MiB")
                        body.extend(chunk)
            except TimeoutError as exc:
                raise HTTPException(408, "Photo upload timed out") from exc
            # A cancelled HTTP task must not free capacity while its CPU thread runs.
            # The shielded worker owns the permit after scheduling, even on disconnect.
            worker = asyncio.create_task(anyio.to_thread.run_sync(optimize_photo, bytes(body)))

            def finished(task):
                gate.release()
                if not task.cancelled():
                    task.exception()  # Retrieve failures even when the client has left.

            worker.add_done_callback(finished)
            release_here = False
            result = await asyncio.shield(worker)
            return Response(result.data, media_type=result.content_type, headers={
                "Cache-Control": "no-store",
                "X-Source-Sha256": result.source_sha256,
                "X-Stored-Sha256": result.stored_sha256,
                "X-Photo-Width": str(result.width),
                "X-Photo-Height": str(result.height),
                "X-Photo-Recipe": result.recipe,
            })
        except PhotoTooLarge as exc:
            raise HTTPException(413, str(exc)) from exc
        except InvalidPhoto as exc:
            raise HTTPException(422, str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(503, "Photo optimization failed", headers={"Retry-After": "3"}) from exc
        finally:
            if release_here:
                gate.release()

    return app


app = create_app()
