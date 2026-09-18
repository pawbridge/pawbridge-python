"""WSL GPU entry point; excludes the existing DINOv2 and chatbot startup path."""
from contextlib import asynccontextmanager, nullcontext
import os
import anyio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from app.routers.lost_search import router
from app.services.dinov3 import get_encoder, gallery_index, visual_profile, DIMENSIONS


class ColorGalleryRefreshRequired(RuntimeError):
    """The active alias is safe to read but must be rebuilt before ranking."""


def validate_focus_gallery(encoder):
    from app.es.client import es
    index = gallery_index()
    client = es.options(request_timeout=10, max_retries=0)
    mappings = client.indices.get_mapping(index=index)
    if len(mappings) != 1:
        raise RuntimeError("Animal focus gallery must resolve to exactly one index")
    mapping = next(iter(mappings.values()))["mappings"]
    properties = mapping.get("properties", {})
    if (mapping.get("_meta", {}).get("model_version") != encoder.model_version
            or any(properties.get(field, {}).get("type") != "dense_vector"
                   or properties[field].get("dims") != DIMENSIONS
                   for field in ("image_vector", "animal_vector"))):
        raise RuntimeError("Animal focus gallery mapping/version mismatch")
    from app.services.coat_color import VERSION as COLOR_VERSION, ranking_weight
    if ranking_weight() and mapping.get("_meta", {}).get("coat_color_version") != COLOR_VERSION:
        raise ColorGalleryRefreshRequired("Coat-color ranking requires a completed color gallery")
    count = client.count(index=index, query={"bool": {"filter": [
        {"term": {"model_version": encoder.model_version}},
        {"exists": {"field": "image_vector"}}, {"exists": {"field": "id"}}]}})["count"]
    if count == 0:
        raise RuntimeError("Animal focus gallery has no searchable records")


@asynccontextmanager
async def lifespan(app):
    if not os.getenv("INTERNAL_API_KEY"):
        raise RuntimeError("Internal authentication must be configured")
    gallery_index()
    from app.services.coat_color import ranking_weight
    if ranking_weight() and visual_profile() != "sam3-animal-focus":
        raise RuntimeError("Coat-color ranking requires the SAM 3 gallery")
    enabled = os.getenv("LOST_GALLERY_SYNC_ENABLED", "false").lower() == "true"
    state_dir = os.getenv("LOST_GALLERY_STATE_DIR")
    if enabled and (visual_profile() != "sam3-animal-focus" or not state_dir):
        raise RuntimeError("Automatic gallery refresh requires SAM 3 and a persistent state directory")
    from app.services.gallery_runtime import runtime_owner, GalleryRefresh
    from app.services.gallery_source import GallerySource
    from app.services.gallery_pages import PagedGallerySource
    protocol = os.getenv('LOST_GALLERY_PROTOCOL', 'v1')
    if protocol not in {'v1', 'v2'}:
        raise RuntimeError('Gallery protocol must be v1 or v2')
    source_class = PagedGallerySource if protocol == 'v2' else GallerySource
    # Acquire ownership before loading weights; CLI and API must use the same state directory.
    with runtime_owner(state_dir) if state_dir else nullcontext():
        encoder = await anyio.to_thread.run_sync(get_encoder)
        ready = True
        if visual_profile() in {"animal-focus", "sam3-animal-focus"}:
            from elasticsearch import NotFoundError
            try:
                await anyio.to_thread.run_sync(validate_focus_gallery, encoder)
            except (NotFoundError, ColorGalleryRefreshRequired):
                if not enabled:
                    raise
                ready = False
        refresh = None
        if enabled:
            from app.es.client import es
            source = source_class(os.environ["LOST_GALLERY_SOURCE_URL"],
                                   os.environ["LOST_GALLERY_SOURCE_KEY"],
                                   os.environ["LOST_GALLERY_R2_HOST"], state_dir,
                                   os.environ["LOST_GALLERY_PHOTO_ROOT"])
            try:
                refresh = GalleryRefresh(es, encoder, source, gallery_index(), state_dir,
                                         interval=int(os.getenv("LOST_GALLERY_INTERVAL_SECONDS", "900")), ready=ready)
                app.state.gallery_refresh = refresh
                refresh.start()
            except BaseException:
                source.close()
                raise
        try:
            yield
        finally:
            if refresh:
                await anyio.to_thread.run_sync(refresh.close)


app = FastAPI(title="PawBridge Lost Animal Search", lifespan=lifespan,
              docs_url=None, redoc_url=None, openapi_url=None)
app.include_router(router, prefix="/internal/animals")


@app.get("/health")
def health(request: Request):
    refresh = getattr(request.app.state, "gallery_refresh", None)
    ready = refresh is None or refresh.ready
    return JSONResponse({"status": "ok" if ready else "initializing",
                         "modelVersion": get_encoder().model_version,
                         "galleryRefresh": refresh.status if refresh else {"state": "disabled"}},
                        status_code=200 if ready else 503)
