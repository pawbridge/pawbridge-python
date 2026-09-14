import asyncio
import hmac
import os
from datetime import date
from typing import Literal

import anyio
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field, ValidationError, field_validator
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.services.lost_search import MAX_IMAGE_BYTES, InvalidPhoto, search_photo

router = APIRouter()
_gate = asyncio.Lock()
MAX_REQUEST_BYTES = MAX_IMAGE_BYTES + 64 * 1024


class SearchConditions(BaseModel):
    species: Literal["DOG", "CAT"]
    lostDate: date | None = None
    region: str | None = Field(default=None, max_length=100)
    description: str | None = Field(default=None, max_length=500)

    @field_validator("region", "description", mode="before")
    @classmethod
    def normalize_optional_text(cls, value):
        return (value.strip() or None) if isinstance(value, str) else value


def authenticate(x_internal_api_key: str | None = Header(None)):
    expected = os.getenv("INTERNAL_API_KEY", "")
    if not expected:
        raise HTTPException(503, "내부 검색 설정을 확인해 주세요")
    if not x_internal_api_key or not hmac.compare_digest(x_internal_api_key.encode(), expected.encode()):
        raise HTTPException(401, "인증이 필요합니다")


@router.post("/lost-candidates", dependencies=[Depends(authenticate)])
async def lost_candidates(request: Request):
    refresh = getattr(request.app.state, "gallery_refresh", None)
    if refresh is not None and not refresh.ready:
        raise HTTPException(503, "검색 자료를 준비하고 있습니다. 잠시 후 다시 시도해 주세요")
    if _gate.locked():
        raise HTTPException(503, "검색 중입니다. 잠시 후 다시 시도해 주세요", headers={"Retry-After": "3"})
    async with _gate:
        # Limit total body before multipart parsing/spooling, even without Content-Length.
        body = bytearray()
        try:
            with anyio.fail_after(15):
                async for chunk in request.stream():
                    if len(body) + len(chunk) > MAX_REQUEST_BYTES:
                        raise HTTPException(413, "사진은 5MiB 이하이어야 합니다")
                    body.extend(chunk)
        except TimeoutError as exc:
            raise HTTPException(408, "사진 업로드 시간이 초과됐습니다") from exc
        async def receive():
            return {"type": "http.request", "body": bytes(body), "more_body": False}
        parsed = Request(request.scope, receive)
        try:
            async with parsed.form(max_files=1, max_fields=4) as form:
                if set(form.keys()) - {"image", "species", "lostDate", "region", "description"}:
                    raise HTTPException(422, "지원하지 않는 입력입니다")
                if len(form.multi_items()) != len(form):
                    raise HTTPException(422, "중복된 입력입니다")
                photo = form.get("image")
                if not isinstance(photo, UploadFile):
                    raise HTTPException(422, "사진을 선택해 주세요")
                conditions = SearchConditions(**{k: (v or None) for k, v in form.items() if k != "image"})
                data = await photo.read(MAX_IMAGE_BYTES + 1)
                if len(data) > MAX_IMAGE_BYTES:
                    raise HTTPException(413, "사진은 5MiB 이하이어야 합니다")
            # Do not abandon a running thread on cancellation: keep the gate until it exits.
            return await anyio.to_thread.run_sync(
                search_photo, data, conditions.species, conditions.lostDate,
                conditions.region, conditions.description)
        except ValidationError as exc:
            raise HTTPException(422, "종류·날짜·지역·특징 입력을 확인해 주세요") from exc
        except InvalidPhoto as exc:
            raise HTTPException(422, str(exc)) from exc
        except StarletteHTTPException:
            raise
        except Exception as exc:
            raise HTTPException(503, "검색을 완료하지 못했습니다. 다시 시도해 주세요") from exc
