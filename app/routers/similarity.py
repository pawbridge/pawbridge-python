from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.embedding import extract_embedding_from_url
from app.es.client import get_animal_vector, save_animal_vector, knn_search, get_animals_without_vector

router = APIRouter()


class SimilarRequest(BaseModel):
    animal_id: int
    image_url: str
    species: str | None = None


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
        if not save_animal_vector(req.animal_id, vector):
            raise HTTPException(status_code=500, detail="벡터 저장 실패: ES 문서를 찾을 수 없습니다")

    return knn_search(vector, exclude_id=req.animal_id, species=req.species)


@router.post("/batch/embeddings", response_model=BatchEmbeddingResponse)
async def generate_embeddings():
    """
    image_vector가 없는 동물들의 임베딩을 일괄 생성하여 ES에 저장
    - 처리할 동물이 없거나 모두 실패할 때까지 반복
    """
    total_processed = 0
    failed_ids: set = set()

    while True:
        animals = get_animals_without_vector(size=200)
        if not animals:
            break

        batch_processed = 0

        for animal in animals:
            animal_id = animal.get("id")
            image_url = animal.get("image_url")

            if not animal_id:
                continue

            if not image_url:
                failed_ids.add(animal_id)
                continue

            vector = await extract_embedding_from_url(image_url)
            if vector is None:
                failed_ids.add(animal_id)
                continue

            if save_animal_vector(animal_id, vector):
                total_processed += 1
                batch_processed += 1
                failed_ids.discard(animal_id)
            else:
                failed_ids.add(animal_id)

        if batch_processed == 0:
            break

    return BatchEmbeddingResponse(processed=total_processed, failed=len(failed_ids))
