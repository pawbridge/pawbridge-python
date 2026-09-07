import os
from urllib.parse import urlparse

from elasticsearch import Elasticsearch, NotFoundError


def create_elasticsearch_client() -> Elasticsearch:
    url = os.getenv("ES_URL", "http://localhost:9200")
    username = os.getenv("ES_USERNAME")
    password = os.getenv("ES_PASSWORD")
    ca_cert_path = os.getenv("ES_CA_CERT_PATH")

    scheme = urlparse(url).scheme.lower()
    if scheme not in {"http", "https"}:
        raise RuntimeError("ES_URL must use http or https")

    if bool(username) != bool(password):
        raise RuntimeError("ES_USERNAME and ES_PASSWORD must be configured together")

    client_options = {}
    if username and password:
        client_options["basic_auth"] = (username, password)

    if scheme == "https":
        if not username or not password:
            raise RuntimeError("HTTPS Elasticsearch requires ES_USERNAME and ES_PASSWORD")
        if not ca_cert_path:
            raise RuntimeError("HTTPS Elasticsearch requires ES_CA_CERT_PATH")
        client_options["ca_certs"] = ca_cert_path

    return Elasticsearch(url, **client_options)


es = create_elasticsearch_client()

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
            doc={"image_vector": vector}
        )
        return True
    except NotFoundError:
        return False


def knn_search(vector: list[float], exclude_id: int, species: str | None = None, k: int = 6, min_score: float = 1.6) -> list[int]:
    """image_vector 기준 script_score 코사인 유사도 검색으로 유사 동물 ID 반환
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
        size=k + 1,
        min_score=min_score,
        query={
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
        source=["id"]
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
        query={
            "bool": {
                "must_not": must_not,
                "filter": [
                    {"exists": {"field": "image_url"}},
                    {"exists": {"field": "id"}}
                ]
            }
        },
        source=["id", "image_url"],
        size=size
    )
    return [{"_id": hit["_id"], **hit["_source"]} for hit in res["hits"]["hits"]]
