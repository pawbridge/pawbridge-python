"""Photo-first retrieval. Auxiliary evidence changes rank, never eligibility."""
import io
import logging
import math
import re
import warnings
from datetime import date
from PIL import Image, ImageOps, UnidentifiedImageError
from app.services.coat_color import mismatch, ranking_weight

MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_PIXELS = 16_000_000
CANDIDATE_POOL = 200
MAX_RESULTS = 20
AUXILIARY_BOOST = 0.01
# Provisional evaluation weight, not a calibrated identity probability.
ANIMAL_REGION_WEIGHT = 0.7
ACTIVE_STATUSES = ("NOTICE", "PROTECT")
RESOLVED_STATUSES = ("ADOPTED", "RETURNED")


class InvalidPhoto(ValueError):
    pass


def decode_photo(data: bytes) -> Image.Image:
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise InvalidPhoto("사진은 5MiB 이하이어야 합니다")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if (image.format not in {"JPEG", "PNG", "MPO"}
                        or (getattr(image, "n_frames", 1) != 1 and image.format != "MPO")):
                    raise InvalidPhoto("JPG 또는 PNG 정지 사진을 선택해 주세요")
                if image.width * image.height > MAX_PIXELS:
                    raise InvalidPhoto("사진 해상도는 1600만 픽셀 이하이어야 합니다")
                image.verify()
            with Image.open(io.BytesIO(data)) as image:
                # Match the gallery: MPO is a primary JPEG plus auxiliary still images.
                image.seek(0)
                oriented = ImageOps.exif_transpose(image)
                try:
                    rgb = oriented.convert("RGB")
                    rgb.info.clear()
                    return rgb
                finally:
                    if oriented is not image:
                        oriented.close()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise InvalidPhoto("사진을 읽을 수 없습니다") from exc


def auxiliary_evidence(source, lost_date=None, region=None, description=None):
    evidence = []
    if lost_date:
        try:
            found = date.fromisoformat(str(source.get("happen_date", ""))[:10])
            if found >= lost_date:
                evidence.append("FOUND_ON_OR_AFTER_LOST_DATE")
        except ValueError:
            pass
    # Only the reported discovery place, never the shelter address.
    region = (region or "").strip()
    if region and region in str(source.get("happen_place") or ""):
        evidence.append("DISCOVERY_PLACE_TEXT_MATCH")
    tokens = set(re.findall(r"[가-힣A-Za-z0-9]+", (description or "").casefold()))
    registered = " ".join(str(source.get(k) or "") for k in ("color", "special_mark", "description")).casefold()
    if any(len(t) >= 2 and t in registered for t in tokens):
        evidence.append("REGISTERED_DESCRIPTION_TEXT_MATCH")
    return evidence


def rank_candidates(hits, lost_date=None, region=None, description=None, *, coat_color=None, color_weight=0.):
    candidates = []
    penalties = {}
    for hit in hits:
        src = hit["_source"]
        score = float(hit["_score"])
        if not math.isfinite(score):
            continue
        distance = mismatch(coat_color, src.get("coat_color")) if color_weight else None
        penalties[src["id"]] = color_weight * distance if distance is not None else 0.
        evidence = auxiliary_evidence(src, lost_date, region, description)
        candidates.append({"animalId": src["id"], "imageScore": score - 1.0,
                           "matchedEvidence": evidence})
    # Provisional bounded boost: <= .03, so metadata cannot overturn large visual gaps.
    candidates.sort(key=lambda c: (-(c["imageScore"] + AUXILIARY_BOOST * len(c["matchedEvidence"]) - penalties[c["animalId"]]),
                                   -c["imageScore"], c["animalId"]))
    return candidates[:MAX_RESULTS]


def search_photo(data, species, lost_date=None, region=None, description=None, include_adopted_or_returned=False):
    from app.services.dinov3 import get_encoder, gallery_index
    from app.es.client import es
    color_weight = ranking_weight()
    with decode_photo(data) as image:
        embedding = get_encoder().encode_with_metadata(image, species)
    logging.getLogger(__name__).info("Lost-search image processing: %s", embedding.focus_status)
    script = {"source": "cosineSimilarity(params.vector, 'image_vector') + 1.0",
              "params": {"vector": embedding.vector}}
    if embedding.animal_vector is not None:
        script = {
            "source": "double original = cosineSimilarity(params.vector, 'image_vector'); "
                      "if (doc.containsKey('animal_vector') && doc['animal_vector'].size() != 0) { "
                      "double animal = cosineSimilarity(params.animal, 'animal_vector'); "
                      "return 1.0 + (1.0 - params.weight) * original + params.weight * animal; } "
                      "return 1.0 + original;",
            "params": {"vector": embedding.vector, "animal": embedding.animal_vector,
                       "weight": ANIMAL_REGION_WEIGHT}}
    allowed_statuses = ACTIVE_STATUSES + (RESOLVED_STATUSES if include_adopted_or_returned else ())
    # Missing status keeps a legacy gallery searchable during a staged rollout;
    # Animal Service always rechecks the current MySQL status before responding.
    status_filter = {"bool": {"should": [
        {"terms": {"status": allowed_statuses}},
        {"bool": {"must_not": {"exists": {"field": "status"}}}}
    ], "minimum_should_match": 1}}
    response = es.options(request_timeout=15, max_retries=0).search(
        index=gallery_index(), size=CANDIDATE_POOL, timeout="10s",
        query={"script_score": {
            "query": {"bool": {"filter": [
                {"term": {"species": species}}, {"term": {"model_version": embedding.model_version}},
                status_filter,
                {"exists": {"field": "image_vector"}},
                {"exists": {"field": "id"}}]}},
            "script": script}},
        sort=[{"_score": "desc"}, {"id": "asc"}],
        source=["id", "happen_date", "happen_place", "color", "special_mark", "description", "coat_color"])
    if response.get("timed_out") or response.get("_shards", {}).get("failed", 0):
        raise RuntimeError("Incomplete search response")
    return {"candidates": rank_candidates(response["hits"]["hits"], lost_date, region, description,
                                          coat_color=embedding.coat_color, color_weight=color_weight)}
