import logging
from typing import Literal
from fastapi import APIRouter, Depends, HTTPException, Path, Request
from app.routers.lost_search import authenticate
from app.services.recommendation import recommend_animals
from app.services.pg_gallery_store import RecommendationGalleryUnavailable, RecommendationSourceUnavailable

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get('/{animal_id}/similar', response_model=list[int], dependencies=[Depends(authenticate)])
def similar_animals(request: Request, species: Literal['DOG', 'CAT'], animal_id: int = Path(gt=0)):
    refresh = getattr(request.app.state, 'gallery_refresh', None)
    if refresh is not None and not refresh.ready:
        logger.warning('Animal recommendation unavailable: animal_id=%s reason=GALLERY_NOT_READY', animal_id)
        raise HTTPException(503, '추천 자료를 준비하고 있습니다')
    try:
        return recommend_animals(animal_id, species)
    except Exception as exc:
        reason = ('SOURCE_FEATURES_NOT_READY' if isinstance(exc, RecommendationSourceUnavailable)
                  else 'GALLERY_CONTRACT_UNAVAILABLE' if isinstance(exc, RecommendationGalleryUnavailable)
                  else 'QUERY_FAILED')
        # Exception messages may contain DSNs or signed URLs; never log them or a traceback.
        logger.warning('Animal recommendation unavailable: animal_id=%s reason=%s exception_type=%s',
                       animal_id, reason, type(exc).__name__)
        raise HTTPException(503, '추천 자료를 확인하지 못했습니다. 잠시 후 다시 시도해 주세요') from exc
