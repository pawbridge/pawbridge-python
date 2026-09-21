from typing import Literal
from fastapi import APIRouter, Depends, HTTPException, Path, Request
from app.routers.lost_search import authenticate
from app.services.recommendation import recommend_animals

router = APIRouter()


@router.get('/{animal_id}/similar', response_model=list[int], dependencies=[Depends(authenticate)])
def similar_animals(request: Request, species: Literal['DOG', 'CAT'], animal_id: int = Path(gt=0)):
    refresh = getattr(request.app.state, 'gallery_refresh', None)
    if refresh is not None and not refresh.ready:
        raise HTTPException(503, '추천 자료를 준비하고 있습니다')
    try:
        return recommend_animals(animal_id, species)
    except Exception as exc:
        raise HTTPException(503, '추천 자료를 확인하지 못했습니다. 잠시 후 다시 시도해 주세요') from exc
