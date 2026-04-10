# Pawbridge Python AI Service

유기동물 이미지 기반 **유사 동물 검색** 기능을 담당하는 AI 마이크로서비스입니다.
Pawbridge MSA의 일부로, Java animal-service로부터 요청을 받아 이미지 벡터 유사도 검색 결과를 반환합니다.

---

## 주요 기능

- **이미지 임베딩 추출**: DINOv2 ViT-S/14 모델로 동물 이미지 → 384차원 벡터 변환
- **유사 동물 검색**: Elasticsearch kNN 검색 (코사인 유사도 기반)
- **임베딩 배치 처리**: image_vector 없는 동물을 순차 처리하여 ES에 저장

---

## 기술 스택

| | |
|--|--|
| **Language** | Python 3.11 |
| **Framework** | FastAPI |
| **ML** | PyTorch 2.2 (CPU), TorchVision, DINOv2 ViT-S/14 |
| **Search** | Elasticsearch 7.x (dense_vector, script_score) |
| **HTTP** | httpx (비동기) |
| **Image** | Pillow |
| **Deploy** | Docker, Kubernetes (Helm) |

---

## 핵심 기술 결정

### MobileNetV3 → DINOv2 ViT-S/14 교체

초기에 MobileNetV3 Large(1280차원)를 사용했으나 유사도 검색 품질 문제로 교체했습니다.

| | MobileNetV3 Large | DINOv2 ViT-S/14 |
|--|--|--|
| 학습 방식 | 분류(classification) 지도학습 | 자기지도학습(self-supervised) |
| 학습 목표 | 이미지 분류 경계 최적화 | 외형 유사도 직접 최적화 |
| 출력 차원 | 1280 | 384 |
| CPU 추론 | 가능 | 가능 (torch+cpu) |

분류 모델은 "이게 강아지냐 고양이냐"를 판단하는 데 최적화되어 있어, 외형이 비슷해도 종이 다르면 벡터 거리가 크게 나왔습니다.
DINOv2는 같은 이미지의 다른 뷰는 가깝게, 다른 이미지는 멀게 학습(DINO)하므로 외형 유사도 검색에 적합합니다.

### species 필터 + min_score 임계값

```python
def knn_search(vector, exclude_id, species=None, k=6, min_score=1.6):
    filters = [
        {"terms": {"status": ["NOTICE", "PROTECT"]}},
        {"exists": {"field": "image_vector"}}
    ]
    if species:
        filters.append({"term": {"species": species}})
```

- `min_score=1.6`: 코사인 유사도 0.6 미만 결과 제외 → 관련 없는 동물을 억지로 채우지 않음
- `species` 필터: min_score만으로는 외형이 비슷한 이종 동물 혼재 방지 불충분 (고양이 검색 시 강아지 포함 문제 해결)

### L2 정규화

```python
vector = F.normalize(vector, dim=-1)
```

DINOv2 공식 권장(retrieval 태스크 기준). L2 정규화된 벡터끼리의 내적 = 코사인 유사도로, 이미지 밝기·크기 차이로 인한 벡터 크기 편차를 제거합니다.

### Docker 모델 사전 캐싱

```dockerfile
RUN python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)"
```

컨테이너 기동 시 모델 다운로드(~80MB)를 없애 즉시 기동을 보장합니다. 네트워크 제한 k8s 환경 대응 목적입니다.

---

## 트러블슈팅

### ES dims 불일치 (mapper_parsing_exception)

MobileNetV3(1280차원) → DINOv2(384차원) 교체 후 임베딩 저장 실패.
배치가 증분 업데이트 방식으로 변경되면서 기존 인덱스(dims:1280)가 유지된 것이 원인.
기존 인덱스 삭제 → ES 인덱스 템플릿(dims:384) 기반 재생성 → 배치 재동기화 → 임베딩 재추출로 해결.

### 임베딩 배치 중단 (failed:200 반복)

APMS 만료 URL(404) 동물 200개가 매번 첫 배치에 포함되어 전부 실패 → 루프 탈출 → 나머지 9,000개 미처리.
`get_animals_without_vector()`에 `exclude_ids` 파라미터를 추가해 실패 ID를 다음 배치에서 제외하도록 해결.

### 고양이 검색에 강아지 포함

털 색상·자세가 비슷하면 종이 달라도 벡터 유사도가 min_score를 넘는 케이스 발생.
`knn_search()`에 `species` term filter를 추가해 같은 축종 내에서만 검색하도록 해결.

---

## 프로젝트 구조

```
app/
├── main.py                  # FastAPI 앱 진입점
├── routers/
│   └── similarity.py        # API 엔드포인트 (유사 검색, 배치 임베딩)
├── services/
│   └── embedding.py         # DINOv2 이미지 임베딩 추출
└── es/
    └── client.py            # Elasticsearch 클라이언트 (벡터 저장/검색)
```

---

## 관련 레포지토리

- [pawbridge-backend-k8s](https://github.com/pawbridge/pawbridge-backend-k8s) — Java MSA 백엔드 (animal-service 등)
- [pawbridge-infra-k8s](https://github.com/pawbridge/pawbridge-infra-k8s) — Kubernetes 인프라 (Helm Charts, Vagrant)
