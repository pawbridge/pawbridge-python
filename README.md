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

## 실종동물 검색 GPU 프로필

기존 CPU 서비스와 분리된 진입점은 `app.lost_main:app`이다. GPU 프로필은 기본값을 자동으로 바꾸지 않는다.

| `LOST_SEARCH_VISUAL_PROFILE` | 이미지 처리 | `LOST_SEARCH_INDEX` 접두사 |
|---|---|---|
| `original` (기본값) | DINOv3 중앙 자르기 | `animals-lost-dinov3-` (아래 전용 접두사 제외) |
| `animal-focus` | Mask R-CNN + DINOv3 | `animals-lost-dinov3-focus-` |
| `sam3-animal-focus` | SAM 3 + DINOv3 | `animals-lost-dinov3-sam3-` |

SAM 3는 Python 3.12의 독립 CUDA 환경에서 `requirements-lost-sam3.txt`로 설치한다. 이 파일은 검증한 공식 SAM 3 소스 커밋을 고정한다. 체크포인트는 별도로 확보하고, 라이선스 조건을 확인한 뒤 읽기 전용 로컬 파일로 제공한다. 실행 중 가중치를 다운로드하지 않는다.

```bash
export LOST_SEARCH_VISUAL_PROFILE=sam3-animal-focus
export LOST_SEARCH_INDEX=animals-lost-dinov3-sam3-v1
export DINOV3_CHECKPOINT=/path/to/dinov3/model.safetensors
export SAM3_CHECKPOINT=/path/to/sam3/sam3.pt
export ES_URL=http://127.0.0.1:9200
# INTERNAL_API_KEY 및 필요한 ES 인증값은 환경/비밀 저장소에서 주입한다.
python -m uvicorn app.lost_main:app --host 127.0.0.1 --port 8001 --workers 1
```

운영 네트워크 연결은 별도 배포 설정으로 구성한다. GPU당 프로세스 하나로 실행하고 `--reload`나 다중 worker를 사용하지 않는다. 프로세스 내부의 순차 처리 잠금은 다른 프로세스의 GPU 사용까지 제어하지 않는다.

SAM 3 프로필은 BF16 연산을 지원하는 CUDA GPU를 요구하며 PyTorch 할당 한도를 7GiB로 설정한다. 가중치는 FP32를 유지하고 SAM 추론만 BF16 autocast를 사용한다. 메모리 매핑과 `assign=True`로 CPU 중복 가중치 복사를 줄인다. 한도는 GPU 드라이버·다른 앱을 포함한 전체 사용량이나 호스트 RAM 한도가 아니다. 호스트의 메모리 제한·재시작 정책은 별도 서비스 설정이 필요하다.

갤러리 재생성은 다운로드 전용 스레드 하나로 다음에 필요한 사진 한 장을 미리 받는다. GPU 추론은 기존 gate로 직렬화하며 사용자 검색 우선권을 유지한다. 현재 처리 사진과 다음 사진만 임시 파일로 보관하고, 실패·취소 시에도 정리한다. SAM 3는 고정된 `dog`·`cat` 텍스트 특징만 모델 프로세스 안에서 재사용하며 이미지 특징이나 마스크를 사진 사이에 재사용하지 않는다.

DINOv3 및 SAM 3 체크포인트 SHA-256이 코드의 고정값과 다르면 시작하지 않는다. SAM 3 모델 버전은 `dinov3-large-sam39999e234-dual-pad256-v1`이다. 갤러리는 동일한 `DinoV3Encoder.encode_with_metadata` 경로로 생성해야 하며, 다음 계약을 갖춰야 시작한다.

- 매핑 `_meta.model_version`이 해당 프로필의 모델 버전과 일치한다.
- `image_vector`와 `animal_vector`는 각각 1024차원 `dense_vector`이다.
- 동일 모델 버전의 `id`와 `image_vector`를 가진 검색 가능한 문서가 하나 이상 존재한다.
- 검색은 `species`와 `model_version`으로 필터링한다. 모델 버전이 다른 벡터를 기존 인덱스에 덮어쓰지 않는다.

전체 사진 벡터를 항상 보존하고, 단일 동물 영역이 확인된 경우에만 별도 동물 벡터를 생성한다. 양쪽에 동물 벡터가 있으면 전체 사진 0.3 + 동물 영역 0.7을 사용하고, 없으면 전체 사진으로 비교한다. 복수 동물·미검출은 정상적인 전체 사진 대체 처리다. 잘못된 모델 출력·추론 오류·메모리 부족은 빈 성공 결과로 숨기지 않고 API 503으로 반환한다. 메모리 부족 시 실패한 연산을 해제한 뒤 캐시를 정리하고 다음 요청을 허용한다.

이 점수는 동일 개체일 확률이 아니다. 원본 사진과 가려진 부위는 생성하거나 덮어쓰지 않는다. 갤러리 생성·교체 및 Spring/Gateway/화면 연결은 별도 배포 단계다.

### 갤러리 최초 생성과 갱신

`app.gallery_main`은 일회성 보정 스크립트가 아니라 같은 모델로 갤러리를 반복 생성하는 실행 진입점이다. APMS/R2/DB 자격 증명을 가져오거나 원본을 수정하지 않는다. 입력은 별도 수집 경로에서 준비한 완전한 스냅샷과 로컬 사진 캐시다.

```json
{"complete":true,"records":[{"id":1,"species":"DOG","source_sha256":"<64자리 소문자 SHA-256>","happen_date":"2026-09-13","happen_place":"<발견 장소>","color":"<등록된 털색>"}]}
```

사진은 `--photo-root` 아래 `<source_sha256>.image` 이름으로 저장한다. 기록의 `complete` 값은 수집자가 원본과 건수·누락을 대조한 결과여야 한다. 이 명령이 VM 원본의 완전성까지 대신 확인하지는 않는다. 날짜·장소·털색·특징·설명은 검색 보조 정보이며 상태의 최종 응답은 Spring의 DB 재조회가 담당한다.

```bash
export LOST_SEARCH_VISUAL_PROFILE=sam3-animal-focus
export LOST_SEARCH_INDEX=animals-lost-dinov3-sam3-gallery-v1
# 가중치 및 ES 인증 환경은 위 실행 계약대로 설정한다.
python -m app.gallery_main --manifest /path/to/snapshot.json \
  --photo-root /path/to/photos --state-dir /path/to/gallery-state
# 검증 결과를 확인한 뒤 실제 작성·교체할 때만 --apply를 추가한다.
```

- 같은 사진 해시·종류·모델의 벡터는 재사용한다. 메타데이터만 바뀌면 재추론하지 않는다.
- 새 스냅샷은 별도 물리 인덱스에 작성한다. 전체 건수 확인 후 단일 alias를 원자적으로 전환한다. 입력/추론/bulk 실패 시 기존 alias를 변경하지 않으며, 같은 스냅샷으로 재실행하면 완료된 벡터를 재사용한다.
- 일부 APMS JPEG는 MPO 컨테이너다. 기존 평가와 같이 첫 대표 프레임만 사용한다. 애니메이션·1600만 픽셀 초과·10MiB 초과·해시 불일치는 실패한다.
- 지원 범위는 GPU 호스트 하나다. 모든 실행은 같은 `--state-dir`을 사용한다. 모델 로딩 전에 파일 잠금을 획득해 병렬 실행을 거부한다. 다중 호스트 게시자나 임의의 외부 alias 수정은 지원하지 않는다.
- CLI와 검색 서버를 별도 프로세스로 동시에 실행하면 모델 메모리가 중복되므로 그렇게 기동하지 않는다. 이 함수는 같은 프로세스의 공유 encoder로도 실행할 수 있지만, 상시 서버 내 스케줄러는 아직 연결하지 않았다.
- 과거 인덱스는 복구를 위해 자동 삭제하지 않는다. 자동 주기 실행과 과거 세대 정리 정책은 별도 운영 단계이며, 무제한으로 새 스냅샷을 생성해 두면 저장 공간이 계속 증가한다.


### 갤러리 자동 갱신과 재시작

SAM3 실종 후보 검색의 내부 스냅샷 공급, 단일 GPU 공유, 원자적 갤러리 교체와 재시작 설정은 [실행 런북](deploy/README.md)을 따른다. 기능은 기본 비활성화이며 실제 feed 배포 및 연결 확인 후 켠다.
