# 실종동물 검색 품질 평가

운영 API·DB·갤러리를 수정하지 않고 고정된 사진으로 특징 추출과 순위를 비교한다.
이 도구는 `app`에서 import하지 않는다. 운영 적용 여부는 별도 결정한다.

## 실행 환경

저장소 루트에서 기존 Python 환경으로 실행한다. 자료 검증·보고서·CPU 계약 테스트에는
NumPy와 Pillow가 필요하다. 실제 추출은 저장소의 `requirements-lost-gpu.txt`,
`requirements-lost-sam3.txt`와 기존 승인된 로컬 가중치가 있는 CUDA 환경을 사용한다.
가중치를 자동 다운로드하지 않는다. CPU 테스트 통과는 모델 정확도 검증이 아니다.

```bash
python -m unittest discover -s tests -p test_lost_quality_evaluation.py -v
python -m evaluation.lost_search validate --manifest /path/to/dataset/manifest.json
python -m evaluation.lost_search preview --manifest /path/to/dataset/manifest.json --output /path/to/new-preview
```

사진, 가중치, 벡터, 보고서와 개인 경로 목록은 저장소 밖에 둔다. 결과 디렉터리를
덮어쓰지 않는다. 이미지 입력은 원본을 보존하며 보고서 미리보기는 EXIF를 제거한다.

## 자료 계약

질의 1~30장, 후보 1~300장, 파일당 5 MiB와 1,600만 픽셀 이하의 JPG/PNG/MPO다.
MPO는 운영과 같은 기본 사진을 읽는다. 파일 경로는 manifest 하위의 상대 경로다.
SHA256, 중복 바이트, 평가 집합 간 개체 누출과 같은 촬영의 재사용을 검사한다.
해시는 재압축·연속 프레임까지 판별하지 못하므로 수집자가 촬영 독립성을 확인해야 한다.

```json
{
  "version": 1,
  "name": "pilot",
  "photos": [
    {
      "id": "query-1", "file": "images/query-1.jpg", "sha256": "REPLACE_WITH_SHA256",
      "role": "query", "species": "DOG", "partition": "case",
      "identity": null, "capture": null, "truth": "unknown",
      "source": "사진 제공 경위와 이용 권한 기록",
      "metadata": {"lost_date": null, "region": "", "query_description": "",
                   "include_adopted_or_returned": false}
    },
    {
      "id": "gallery-1", "file": "images/gallery-1.jpg", "sha256": "REPLACE_WITH_SHA256",
      "role": "gallery", "species": "DOG", "partition": "case",
      "identity": null, "capture": null, "truth": "unknown",
      "source": "공개 공고 출처와 취득 시각·이용 조건 기록",
      "metadata": {"animal_id": 1, "status": "PROTECT", "happen_date": "2026-09-11",
                   "happen_place": "용인", "color": "흰색", "special_mark": "귀 끝 갈색"}
    }
  ]
}
```

- `truth=present`: 같은 개체의 다른 촬영 사진이 후보에 있음을 검증한 질의.
- `truth=absent`: 고정 후보 집합에 정답이 없음을 확인한 질의. ID가 없다는 이유만으로 판정하지 않는다.
- `truth=unknown`: 정답 미확인. 사례 비교에는 포함하지만 정확도 분모에서 제외한다.
- `tune`과 `holdout`: 조정용/최종 평가용. 같은 개체가 두 집합을 오갈 수 없다.
  각 질의는 동일 partition의 후보만 검색한다. 두 집합의 오답 후보도 따로 준비한다.
- `case`: 정답 미확인 사례. `truth=unknown`만 허용한다.
- 점수화하는 질의는 `identity`, `capture`, `verification` 근거 문자열이 필요하다.
  정답 후보의 identity는 같고 capture는 달라야 한다. 잘라내기·재압축본은 새 촬영이 아니다.
- 후보마다 양의 정수 `animal_id`가 필요하며 한 공고의 대표 사진 하나만 사용한다.

## 비교 방식

| 추출 설정 | 정밀도 | 집계 | 동물 입력 |
|---|---|---|---|
| baseline-fp16 | FP16 | 현행 평균 | 현행 256 |
| precision-fp32 | FP32 | 평균 | 현행 256 |
| cls-fp32 | FP32 | CLS | 현행 256 |
| foreground-fp32 | FP32 | 마스크 가중 패치 평균 | 현행 256 |
| native-256-fp32 | FP32 | 마스크 가중 패치 평균 | 원본에서 256 |
| native-384-fp32 | FP32 | 마스크 가중 패치 평균 | 원본에서 384 |
| native-512-fp32 | FP32 | 마스크 가중 패치 평균 | 원본에서 512 |

기본 실행은 앞의 네 방식이다. 해상도 실험은 뒤의 세 방식을 함께 지정해 crop 출처와
해상도 효과를 구분한다. 256으로 줄인 이미지를 다시 확대한 실험이 아니다.
전체 사진 경로는 항상 256이다. CLS 설정에서는 양쪽 경로 모두 CLS를 사용한다.
마스크 집계 설정은 마스크가 있는 동물 경로에만 가중 평균을 적용한다.
모든 방식에서 SAM3 BF16·가중치·마스크는 고정한다. DINO TF32는 모두 끈다.
평균 FP16 기준선은 현행 `forward_head`의 dtype/집계 순서를 유지한다.
이름이 baseline이어도 실제 운영과 수치 동등성 확인 전에는 동등성이 입증된 것으로 보고하지 않는다.

추출 설정별로 다음 순위를 비교한다.

- `current`: 전체 0.3 + 동물 0.7, 색상 감점 최대 0.1, 기존 부가정보 가산.
- `animal-only`: 최초 후보 선택부터 동물 벡터 가중치 1.0. 마스크 실패 시 전체 사진 사용.
- `patch-rerank`: 최초 200개에 대해 패치 상호 최근접 대응 점수 0.2를 혼합한다.
  양쪽 대응 비율 25% 이상·4개 이상일 때만 적용한다. 임시 실험값이며 확정 정책이 아니다.
- `date-policy`: 입력한 정확한 실종일 이전 접수 공고를 최초 후보 선택 전에 제외한다.
  당일·이후·날짜 미상은 포함한다. 날짜 미입력은 제한 없음이다. 지역은 소폭 가산이다.

패치는 마스크 점유율 50% 이상인 위치 중 최대 256개를 결정적으로 선택한다.
CLS와 4개 register는 패치 비교에서 제외한다. 대응 선은 학습 특징 대응 가설이며
귀·얼굴 같은 신체 부위를 판정한 결과가 아니다. 전처리 입력 위의 위치를 표시한다.

## 실제 GPU 실행과 복구

운영 AI와 별도 모델을 같은 GPU에 동시에 띄우지 않는다. 아래 명령은 **별도 운영 중단
승인 후** 실행한다. 이 도구는 서비스를 중지하거나 복구하지 않으며, 활성 상태가
정확히 `inactive`가 아니면 거부한다. 외부 watchdog이 실패 시에도 운영을 복구해야 한다.

```bash
python -m evaluation.lost_search extract \
  --manifest /path/to/dataset/manifest.json --output /path/to/new-features \
  --dino-checkpoint /path/to/existing/model.safetensors \
  --sam-checkpoint /path/to/existing/sam3.pt \
  --timeout-seconds 600 --confirm-exclusive-gpu APPROVED_OFFLINE_GPU_WINDOW
```

Linux systemd 환경에서는 사전에 검토한 transient unit으로 다음을 보장한다.
`ExecStartPre`에서 운영 AI 중지, `RuntimeMaxSec`로 전체 상한, `KillMode=control-group`으로
실험 자식 종료, `ExecStopPost`에서 원래 AI 시작. 명령·정확한 대상·중단 영향·복구 확인을
승인안에 넣는다. 단순 shell trap만으로 앱 강제 종료 시 복구를 보장하지 않는다.
갤러리 갱신 상태를 중단 전후 확인하고, 복구 후 active 상태와 `/health` 응답을 확인한다.

SAM과 DINO는 각각 별도 자식 프로세스로 **차례로** 실행한다. SAM이 완전히 종료한 뒤
DINO를 시작하여 CPU/GPU 모델 메모리 중첩을 피한다. 사진은 한 장씩 처리한다.
CPU 스레드 2, PyTorch GPU 할당 상한 7 GiB, 자식 RSS 감시 상한 8 GiB,
추출 단계 총 시간 최대 600초다. RSS는 0.2초 간격의 종료 기준으로 순간적인 hard limit은 아니다.
CUDA context 등 PyTorch 밖 메모리는 GPU 할당 상한에 포함되지 않는다.
시작 전 디스크 여유 5 GiB 이상을 요구한다. 실패 결과는 `failed`로 남기며 비교를 거부한다.
원인을 확인한 후 새 출력 폴더로 재실행하고 불필요한 실패 자료만 명시적으로 정리한다.

운영 AI 복구 후 비교·HTML 생성은 GPU 없이 가능하다.

```bash
python -m evaluation.lost_search compare \
  --manifest /path/to/dataset/manifest.json --features /path/to/completed-features \
  --output /path/to/new-report
```

결과의 코드·자료·가중치 해시와 설정을 검증한다. 추출 후 코드를 수정했다면 별도 결과로
재추출해야 한다. 해시 검증은 로컬 혼합 방지용이며 외부 서명이나 독립 검증은 아니다.

## 결과 해석

`index.html`은 원본 미리보기, 실제 모델 입력, 후보 순위와 점수 분해, 패치 대응과 자원을
보여 준다. `results.json`에 평가 분모와 모든 순위·조건을 보존한다.
확인된 정답만 Top-1/5/10/20·최초 후보 포함률에 집계한다. 정답 없는 표본은 최고 점수만
기록하며, 판정 임계값을 정하지 않은 상태에서 오탐률이라고 부르지 않는다.
최초 후보 상한이 적용되지 않은 작은 집합의 포함률을 운영 전체 검색 포함률로 일반화하지 않는다.

SAM·DINO 기동, 방식별 첫 장, 이후 장/초·장/분·중앙값·p95와 프로세스 RAM·CUDA 피크를
분리한다. SAM 시간은 이미지 준비 이후 분리와 후처리, DINO 시간은 사진 읽기·전처리·추론·
입력 미리보기 저장까지이며 특징 파일 저장 시간은 제외한다. 작은 표본 p95는 참고치다.
이 단계들을 따로 실행하므로 운영 API 지연·동시 요청 처리량으로 표기하지 않는다.
모의 특징 테스트, 실제 모델 품질, 운영 통합, Mac/MPS 검증은 각각 별도 근거다.
