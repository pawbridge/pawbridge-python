from elasticsearch import Elasticsearch, NotFoundError
import os

ES_URL = os.getenv("ES_URL", "http://localhost:9200")

es = Elasticsearch(ES_URL)

INDEX_NAME = "animals"


def get_animal_vector(animal_id: int) -> list[float] | None:
    """ES에서 특정 동물의 image_vector 조회 (MySQL PK = ES _id)"""
    try:
        res = es.get(index=INDEX_NAME, id=str(animal_id))
        return res["_source"].get("image_vector")
    except NotFoundError:
        return None


def save_animal_vector(animal_id: int, vector: list[float]) -> bool:
    """ES에 동물의 image_vector 저장 (MySQL PK = ES _id). 성공 시 True 반환.
    문서가 없으면 False 반환 (Spring 배치 미실행 또는 타이밍 이슈 → 다음 배치에서 재처리).
    """
    try:
        es.update(
            index=INDEX_NAME,
            id=str(animal_id),
            body={"doc": {"image_vector": vector}}
        )
        return True
    except NotFoundError:
        return False


def knn_search(vector: list[float], exclude_id: int, species: str | None = None, k: int = 6, min_score: float = 1.6) -> list[int]:
    """image_vector 기준 코사인 유사도 검색으로 유사 동물 ID 반환 (ES 7.x script_score)
    min_score=1.6: 코사인 유사도 0.6 이상인 동물만 반환 (유사하지 않으면 빈 리스트)
    species: DOG/CAT/ETC 필터 — 종이 다른 동물이 유사 결과에 포함되는 것을 방지
    """
    filters = [
        {"terms": {"status": ["NOTICE", "PROTECT"]}},
        {"exists": {"field": "image_vector"}}
    ]
    if species:
        filters.append({"term": {"species": species}})

    res = es.search(
        index=INDEX_NAME,
        body={
            "size": k + 1,
            "min_score": min_score,
            "query": {
                "script_score": {
                    "query": {
                        "bool": {
                            "filter": filters
                        }
                    },
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'image_vector') + 1.0",
                        "params": {"query_vector": vector}
                    }
                }
            },
            "_source": ["id"]
        }
    )
    hits = res["hits"]["hits"]
    return [
        hit["_source"]["id"]
        for hit in hits
        if hit["_source"].get("id") != exclude_id
    ][:k]


def get_animals_without_vector(size: int = 100, exclude_ids: list[int] | None = None) -> list[dict]:
    """image_vector가 없는 동물 목록 조회 (배치용)
    exclude_ids: 임베딩 추출 실패한 ID 제외 — 동일 배치 반복 방지
    """
    must_not = [{"exists": {"field": "image_vector"}}]
    if exclude_ids:
        must_not.append({"terms": {"id": exclude_ids}})

    res = es.search(
        index=INDEX_NAME,
        body={
            "query": {
                "bool": {
                    "must_not": must_not,
                    "filter": {"exists": {"field": "image_url"}}
                }
            },
            "_source": ["id", "image_url"],
            "size": size
        }
    )
    return [{"_id": hit["_id"], **hit["_source"]} for hit in res["hits"]["hits"]]
