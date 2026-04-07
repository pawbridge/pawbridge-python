from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.embedding import extract_embedding_from_url
from app.es.client import get_animal_vector, save_animal_vector, knn_search, get_animals_without_vector

router = APIRouter()


class SimilarRequest(BaseModel):
    animal_id: int
    image_url: str


class BatchEmbeddingResponse(BaseModel):
    processed: int
    failed: int


@router.post("/similar", response_model=list[int])
async def get_similar_animals(req: SimilarRequest):
    """
    특정 동물과 유사한 동물 ID 목록 반환
    - ES에 벡터가 있으면 바로 kNN 검색
    - 없으면 imageUrl로 임베딩 추출 후 저장하고 검색
    """
    vector = get_animal_vector(req.animal_id)

    if vector is None:
        vector = await extract_embedding_from_url(req.image_url)
        if vector is None:
            raise HTTPException(status_code=422, detail="이미지 임베딩 추출 실패")
        save_animal_vector(req.animal_id, vector)

    return knn_search(vector, exclude_id=req.animal_id)


@router.post("/batch/embeddings", response_model=BatchEmbeddingResponse)
async def generate_embeddings():
    """
    image_vector가 없는 동물들의 임베딩을 일괄 생성하여 ES에 저장
    - Spring Batch 완료 후 호출하거나 수동으로 호출
    """
    animals = get_animals_without_vector(size=200)
    processed = 0
    failed = 0

    for animal in animals:
        animal_id = animal.get("id")
        image_url = animal.get("image_url")

        if not image_url:
            failed += 1
            continue

        vector = await extract_embedding_from_url(image_url)
        if vector is None:
            failed += 1
            continue

        save_animal_vector(animal_id, vector)
        processed += 1

    return BatchEmbeddingResponse(processed=processed, failed=failed)
