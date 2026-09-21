# 실종동물 갤러리 자동 갱신 런북

이 디렉터리는 `app.lost_main`, `gallery_runtime`, animal-service의 `/internal/animals/lost-gallery` 계약에 연결된 실행 설정이다. 파일 추가만으로 운영 배포나 호스트 부팅 자동 실행이 설정되지는 않는다.

## 동작과 권한

- animal-service가 READY 상태의 대표 사진(slot 1), 현재 APMS 이미지와 일치하는 보관 사진, DOG/CAT의 **완전한 목록**을 읽는다. 실종 후보 검색이므로 종료 상태의 동물도 포함한다. APMS를 직접 재수집하거나 DB를 변경하지 않는다.
- 내부 키가 일치해야 조회하며 동시에 한 요청만 처리한다. 사진 보관 테이블이 배포되어 있어야 한다. SQL/전송 한도를 넘거나 빈 목록이면 503을 반환해 기존 갤러리를 유지한다.
- 내용의 ETag가 같으면 304를 반환한다. 전체 URL 서명은 최초 응답에 시간이 걸릴 수 있어 목록 응답 대기는 90초, 연결은 5초로 제한한다. 사진 다운로드는 별도의 20초 제한을 유지한다. 변경되면 animal-service가 1시간짜리 R2 GET URL을 생성한다. R2 비밀 키는 VM에만 남는다. 로그/상태 파일에는 서명 URL을 쓰지 않는다.
- GPU 서비스는 시작 즉시 새 목록을 확인하고 이후 15분 간격으로 확인한다. **15분은 장수 제한이 아니다.** 새 사진 전체를 순차 처리하며 실패하면 30초부터 간격을 늘려 재시도한다. ES에서 모델·사진 SHA·동물종이 같은 유효 벡터를 먼저 재사용한다. 추론이 필요한 사진만 R2에서 내려받고 성공·실패 모두 임시 파일을 제거한다.
- SAM3와 DINOv3 모델은 프로세스당 한 벌이다. 현재 실행 중인 이미지 추론이 끝나면 대기 중인 사용자 검색을 다음 배경 추론보다 먼저 실행한다. 실행 중 CUDA 커널을 선점하지는 않는다.
- 완전한 인덱스를 확인한 뒤 별칭을 원자적으로 교체한다. 실패/중단은 기존 별칭과 성공 ETag를 유지하며 다음 시도는 저장된 부분 인덱스를 재사용한다.
- 보존 일지는 현재·직전 성공 인덱스를 유지한다. 일지에 기록된 이전/실패 인덱스 중 어떤 별칭도 쓰지 않고 계약/모델이 일치하는 것만 정리한다. 다른 별칭이 붙은 인덱스는 삭제하지 않는다. 이런 참조가 쌓이면 무한히 인덱스를 만들지 않고 갱신을 중단한다. 여러 호스트의 동시 발행은 지원하지 않는다.
- 런타임은 사진을 영구 캐시하지 않는다. 주소 발급 후 45분이 지나면 검증된 목록으로 주소만 갱신한다. 403은 최근 60초 내 주소 갱신이 없을 때 한 번만 갱신·재시도한다. 작업 중인 레코드·ETag는 유지하며 동일 SHA·크기·MIME의 주소만 교체한다. 이전 버전이 남긴 사진 파일은 이 작업으로 삭제하지 않는다.

## 적용 전 준비

1. 기존 검증한 CUDA 가상환경을 `~/pawbridge-ai/venv`, 승인된 소스 경로를 `~/pawbridge-ai/current`에 배치한다. 이 런북은 패키지를 설치하지 않는다.
2. `lost-search.env.example`을 `~/.config/pawbridge/lost-search.env`에 복사하고 0600 권한으로 실제 경로/내부 키를 넣는다. 상태와 사진 경로는 `/tmp`가 아닌 영속 경로를 사용한다. API와 CLI가 **동일한 상태 경로**를 사용해야 GPU 중복 로딩을 막을 수 있다.
3. animal-service에 해당 코드와 기존 사진 보관 스키마가 배포되었는지 확인한다. `animal-gallery-feed.env.example`의 값을 기존 VM 시크릿 관리로 주입한다. 키를 명령 인자/저장소에 쓰지 않는다.
4. 새 feed는 공개 API Gateway에 노출하지 않는다. HTTPS 또는 인증된 로컬 SSH 터널로만 GPU 호스트가 접근하게 한다. 위 예시의 18082 포트는 실제 터널을 별도로 준비해야 한다. 검증용 서버를 연결하면 운영 자동 동기화가 아니다.
5. 실제 feed의 인증 실패 401, 정상 목록 200, 변경 없음 304를 확인한 뒤 양쪽 기능을 활성화한다. GPU API의 127.0.0.1 바인딩을 공개하지 말고 기존 승인된 내부 연결을 사용한다.

## 서비스 등록 및 확인

승인된 배포 단계에서 unit을 `~/.config/systemd/user/pawbridge-lost-search.service`에 복사한 후 실행한다.

```sh
systemctl --user daemon-reload
systemctl --user enable --now pawbridge-lost-search.service
systemctl --user status pawbridge-lost-search.service
journalctl --user -u pawbridge-lost-search.service -n 50
```

한 개의 uvicorn worker만 사용한다. `/health`는 사용 가능한 기존 갤러리가 있으면 200이고, 최초 구축 중이면 503이다. `galleryRefresh.state=failed`이면 검색 200이어도 갤러리가 오래될 수 있으므로 `lastSuccess`, `errorType`을 함께 확인한다. `refresh-status.json`에도 같은 상태를 기록한다. 중단 시 현재 이미지/요청을 마무리하고 다음 사진 전에 멈춘다. 강제 종료되면 부분 인덱스부터 재개한다.

user unit은 WSL 자체나 Windows를 부팅시키지 않는다. 사용자 로그인 없이 실행하는 설정, WSL 기동, SSH 터널/Elasticsearch 기동 순서는 별도 호스트 운영 설정이며 아직 이 파일로 해결되지 않는다.

## 되돌리기

- 자동 갱신만 중지: `LOST_GALLERY_SYNC_ENABLED=false`로 변경하고 승인된 재시작을 실행한다. 마지막 검색 별칭은 그대로 남는다.
- 완전 중지: `systemctl --user stop pawbridge-lost-search.service`. 상태/사진/인덱스를 삭제하지 않는다.
- 직전 인덱스로 수동 복구할 경우 먼저 갱신기를 멈추고 일지의 실제 두 인덱스와 모델/문서 수를 확인한 뒤 승인된 별칭 교체를 수행한다. 일지는 비밀 키를 포함하지 않는다.


## 운영 VM → WSL GPU 연결

`connect_vm.py`와 `pawbridge-lost-search-tunnel.service`는 현재 WSL의 `shyu` system service 배치를 버전 관리한다. 앞의 범용 user unit과 설치 위치가 다르다. 다른 호스트에서는 unit의 User/Group/Home 경로를 먼저 맞춘다. 기본 VM 주소는 `192.168.57.11`이며 `PAWBRIDGE_VM_ADDRESS`로 사설 IPv4를 지정할 수 있다.

연결은 기존 `~/pawbridge-ai/ssh/vm-key`, `known_hosts`와 VM의 `vagrant` 계정을 사용한다. 키/known_hosts를 덮어쓰거나 host key 검증을 끄지 않는다. VM은 Python 3, 비대화형 `sudo kubectl`, UID 1000, SSH streamlocal forwarding이 필요하다. `GatewayPorts`를 바꾸지 않는다.

- WSL `127.0.0.1:18082` → animal-service:8081: 운영 갤러리 목록 조회.
- WSL `127.0.0.1:13306` → MySQL:3306: 기존 로컬 미리보기 연결 보존. 기본 `PAWBRIDGE_DATABASE_BACKEND=mysql` 경로다.
- 선택형 `PAWBRIDGE_DATABASE_BACKEND=postgresql`: WSL `127.0.0.1:15432` → `databases/pawbridge-postgresql`:5432. 이 모드에서는 MySQL 서비스를 조회하거나 연결하지 않는다. GPU API/갤러리의 `LOST_PG_DSN`도 이 loopback 포트를 사용한다. DSN 계정은 제한된 벡터 역할이며 운영 비밀은 비공개 환경 파일에만 둔다.
- VM `127.0.0.1:18091` → WSL `127.0.0.1:18090`: 기존 GPU loopback 연결 보존.
- VM `/home/vagrant/.local/run/pawbridge-gpu/search.sock` → WSL `127.0.0.1:18090`: 운영 Pod 연결. 디렉터리 0700, SSH 소켓 0600, 프록시 UID 1000을 함께 맞춘다.

터널 backend는 터널 systemd unit의 환경(예: 승인된 drop-in)에 설정한다. GPU 서비스의 `lost-search.env`만 수정해도 터널이 그 파일을 자동으로 읽는 것은 아니다. 전환 시 터널 선택과 `LOST_STORAGE_BACKEND=postgresql`/`LOST_PG_DSN`, 실제 PostgreSQL 서비스 이름·5432 포트를 함께 확인한다. 선택한 DB가 없으면 기존 소켓을 건드리기 전에 실패하며 다른 DB로 자동 대체하지 않는다. 코드/환경을 복구할 때는 기본 MySQL 경로와 기존 loopback13306도 함께 검증한다. 이 경로는 로컬 테스트만으로 운영 설치/접속이 완료되지 않는다.

VM 재접속 시 서비스 IP를 다시 조회한다. 사용 중 소켓, 다른 파일·소유자·심볼릭 링크는 거절한다. 연결 거부를 확인한 동일 소켓만 정리하고 다시 연다. 설정/연결 실패는 30초 후 systemd가 재시도한다. 프로세스 관리자가 하나여야 하며 수동 중복 터널을 띄우지 않는다.

적용 전 기존 연결 스크립트·unit·GPU 환경 파일을 별도 0700 복구 디렉터리에 보관하고 GPU 갤러리 완료 상태를 확인한다. 승인된 배포에서만 새 `connect_vm.py`를 `/home/shyu/pawbridge-ai/bin/connect_vm.py`, unit을 `/etc/systemd/system/pawbridge-lost-search-tunnel.service`에 설치하고 daemon-reload 후 **터널 unit만** 재시작한다. 이 변경은 모델 설치나 GPU 프로세스 재시작을 요구하지 않는다.

GPU 인증키는 운영 animal-service의 `animal-python-internal-auth:INTERNAL_API_KEY`와 같아야 한다. 다른 경우 비노출 절차로 기존 GPU env를 백업하고 해당 필드만 갱신한다. 키 반영을 위한 GPU 재시작은 진행 중인 갤러리가 완료된 후 별도로 승인받아 수행한다. 로컬 미리보기의 기존 키도 확인한다. 실제 값은 명령 인자·로그·Git에 넣지 않는다.

인프라 저장소의 `charts/lost-search-gpu-proxy`를 먼저 배포하고 프록시 health, 내부 인증 실패 401, 인증된 후보 검색을 검증한 다음 animal-service의 `LOST_SEARCH_PYTHON_URL`을 전환한다. 터널 시작만으로 Pod 연결이나 공개 화면 배포를 완료했다고 판단하지 않는다.

복구 시 이전 연결 스크립트/unit으로 되돌리고 터널만 재시작한다. GPU 키를 바꿨다면 해당 env 백업과 미리보기 키도 함께 복원한다. 운영 animal-service URL은 인프라 런북의 순서로 복구한다. R2·ES·갤러리 상태 파일은 삭제하지 않는다.

소켓 복구 및 포워딩 계약 테스트:

```sh
python3 -m unittest discover -s tests -p test_vm_bridge.py -v
```

테스트는 임시 Unix 소켓만 사용한다. 소켓 bind를 막는 샌드박스에서는 실행이 차단될 수 있으며, 그 실패를 실제 코드 결함이나 통과로 기록하지 않는다.


## 동물 영역 색 특징 보정과 순위 활성화

`foreground-lab32-v1`은 SAM이 선택한 동물 영역에서 밝기·색 분포를 추출한다. 배경과 마스크 경계 한 픽셀을 제외하고 최대 8,192픽셀을 사용한다. 동물 마스크가 털만 정확히 분리한다는 보장은 없다. 색 특징은 ES의 `coat_color`, 처리 버전은 `coat_color_version`에 저장한다. 사진·원래 벡터는 이 필드에 저장하지 않는다.

- 기본 `LOST_SEARCH_COAT_COLOR_WEIGHT=0`은 기존 순위를 유지한다. 0보다 크면 후보 200건을 색 불일치 감점으로 재정렬한 뒤 최대 20건을 반환한다. 상한은 0.2이며 입력값은 시작 시 검증한다. 값과 허용 오차는 동일 개체 판정 확률이 아니며 실제 평가 후 선택한다.
- 색 특징이 없거나 버전이 다르면 색 감점을 적용하지 않는다. 밝기/색조의 작은 변화에는 허용 범위를 두지만 강한 조명·화이트밸런스, 젖거나 오염된 털, 부정확한 마스크에 따른 오판 가능성은 남는다. 후보군 밖 동물을 새로 찾는 변경은 아니다.
- 먼저 감점 0으로 승인된 새 런타임을 배포하고 자동 갤러리가 새 색 계약으로 완성되도록 한다. 기존 벡터가 유효하면 DINOv3는 재실행하지 않고 필요한 사진의 SAM 처리만 실행한다. 이미 색 처리된 문서는 다운로드하지 않는다. 이전에 마스크를 얻지 못했던 사진은 색 특징 없음으로 기록하며 같은 모델의 실패를 반복 실행하지 않는다.
- 소스 ETag가 같아도 새 계약은 별도의 물리 인덱스를 사용한다. 시작 시 소스를 다시 읽고, 완성된 색 계약 확인 전에는 성공 갱신으로 생략하지 않는다. `result/progress.color_processed`는 이번 빌드의 색 처리 횟수, `color_available`은 처리된 문서 중 유효 특징 수다. 기존 완성 인덱스의 빠른 재사용 결과에는 이 카운터가 생략될 수 있다.
- `galleryRefresh.state=idle`, 새 `lastSuccess`, 별칭 대상의 `_meta.coat_color_version`, 문서 수와 유효 색 특징 비율을 확인한다. 전체 수와 색 특징 유효 수가 같은 것은 필수가 아니다. 마스크 실패는 명시적 무특징 상태로 남는다.
- 보정 완료 후 별도의 승인된 활성화에서 감점 값을 지정하고 재시작한다. 기존 갤러리에 색 계약이 없으면 감점 활성 상태의 시작은 실패한다. 실제 사진 전후 순위·응답 지연을 확인한다. 사용자 사진을 원격 공개 API로 재전송하거나 보관하는 검증은 별도의 허용 범위 안에서만 수행한다.
- 순위만 복구하려면 감점 값을 0으로 되돌리고 승인된 재시작을 실행한다. 데이터는 삭제하지 않는다. 코드 자체 복구가 필요하면 기존 릴리스 경로로 되돌린다. 새 갤러리는 DINOv3 벡터 계약과 기존 응답 모양을 유지하므로 이전 코드도 읽을 수 있지만, 구버전 갱신기는 구버전 물리 인덱스를 다시 만들 수 있다. 예기치 않은 재처리를 막으려면 코드 복구 시 자동 갱신을 일시 중지하고 기존 완성 별칭을 유지한다.


## PostgreSQL 저장소 전환 검증 경로

기본 `LOST_STORAGE_BACKEND=elasticsearch`는 기존 ES 검색/갤러리를 사용한다. 선택형 `postgresql`은 SAM 3 프로필만 지원하며, 별도 `requirements-lost-postgresql.txt`와 Animal 저장소의 PostgreSQL V1–V5가 필요하다. 스키마나 확장을 Python 시작 시 자동 생성하지 않는다. DB 역할/사설 연결/백업·복원/기존 ES 벡터 복사 검증 후 별도의 승인된 배포에서만 전환한다. 이 옵션의 구현이나 로컬 테스트가 운영 전환을 의미하지 않는다.

- `LOST_PG_DSN`은 비밀 환경 파일에만 둔다. 기본 최대 연결2개, 허용1–4개, 대기열4개/획득3초, 연결3초/SQL10초/잠금2초로 제한한다. GPU 프로세스와 Spring Hikari 풀은 별도다. 프로세스 수를 곱한 전체 연결 예산은 별도로 확정해야 한다.
- 조회·저장 함수 안에서만 연결을 빌린다. 검색은 읽기 전용 REPEATABLE READ로 공개본 검증과 후보 조회가 같은 DB 시점을 읽게 한다. 사진 다운로드/CPU 준비/SAM/DINO 추론은 연결 반납 후 수행한다. 최대100개 문서를 읽고 저장한 뒤 트랜잭션이 커밋된 경우에만 피드 커서를 진행한다.
- `lost_gallery_builds/documents/heads`가 ES 물리 인덱스/문서/별칭 역할을 담당한다. 문서 수·전체 피드 검증 뒤 head를 트랜잭션 안에서 교체한다. 기존 head가 바뀌었으면 공개를 거부하며, 공개된 빌드는 불변이다. 이전 공개본/진행 중 빌드는 기존 보존 정책을 따르고 PostgreSQL 보존 일지는 별도 이름을 사용한다. 이 스냅샷은 현재 animals 행을 대신하지 않으며 최종 상태 확인은 Animal Service 책임이다.
- 전체/동물 영역1024차원과 기존 모델·색 특징을 유지한다. 정확 코사인 + 기존0.7 동물 영역 가중치로 후보200개를 조회하고 Python 색/부가정보 재정렬로20개를 반환한다. ANN이나 새로운 점수 정책을 도입하지 않는다. DB/언어의 부동소수점 차이에 따른 경계 순위는 실제 벡터 대조가 필요하다.
- 기존 ES 벡터 자동 복사 기능은 이 실행 경로에 없다. 복사 검증 없이 빈 PostgreSQL에서 자동 갱신을 켜면 사진을 다시 추론한다. 전환 전 별도 이관이 필요하며 DINOv2 384차원은 복사하지 않는다.
- 실제 WSL system unit에 별도 `check_es.py` 시작 훅이 있는 경우, PostgreSQL 전환 시 해당 훅과 터널/사설 주소를 함께 검토해야 한다. 저장소의 범용 user unit만 바꿔서는 운영 unit이 바뀌지 않는다.
- 되돌리기는 기존 ES 별칭·릴리스·환경 복원 후 승인된 재기동으로 수행한다. 실제 운영 DB/ES 삭제나 live head 교체는 검증 명령에 포함하지 않는다.

새 PostgreSQL 통합 테스트는 GPU 없이 실제 DB에서 저장 원자성·기존 점수·연결 반납을 검사한다. 폐기 가능한 DB의 `migration_test_guard.guard = animal-pg-disposable` 표식과 스키마가 필요하며, 테스트 시작 시 갤러리 세 테이블을 비운다. **운영에 실행하지 않는다.**

```sh
ANIMAL_PG_MIGRATION_TEST_PORT=25437 python -m unittest discover -s tests/integration -p test_pg_gallery.py -v
```


## PostgreSQL 유사동물 추천 연결

`app.lost_main`은 내부 인증을 요구하는 `GET /internal/animals/{animal_id}/similar?species=DOG|CAT`을 제공한다. 사진이나 벡터를 요청 본문으로 받지 않고, 공개된 PostgreSQL 갤러리의 해당 동물1024차원 벡터를 재사용한다. 모델·색 처리 버전이 맞지 않거나 기준 동물의 벡터가 없으면503이다. 자동 DINOv2 대체·요청 중 다운로드/재추론은 없다.

- Animal Service의 `postgresql` 프로필은 `pawbridge.recommendation.backend=postgresql` 및 OSIV 비활성화를 선택한다. `lost-search.python-url`은 위 GPU 진입점, `python-ai-service.url`은 기존 챗봇 진입점으로 각각 유지한다. 내부 키는 기존 `python-ai-service.internal-api-key` 계약을 사용하며 Feign 로그/재시도 정책도 기존 내부 검색 설정을 따른다. 기본 프로필의 ES 추천은 유지한다.
- Python GPU 프로세스는 `LOST_STORAGE_BACKEND=postgresql`, `sam3-animal-focus`와 대응하는 갤러리 이름이 필요하다. **기존 `app.main` 프로세스에도 전환 시 `LOST_STORAGE_BACKEND=postgresql`을 지정**해야 이전384차원 추천/배치 API 등록과 임베딩 모듈 로딩을 중단한다. 챗봇 API는 유지한다. 이것은 운영에 이미 적용된 설정이 아니다.
- 비교 기준 동물은 입양 완료여도 된다. 후보는 같은 종이며 현재 `animals.status`가 NOTICE/PROTECT인 동물로 한정한다. SQL에서 상태를 먼저 거른 후 점수 상위200건을 선정하므로 종료 동물이 후보 자리를 소모하지 않는다. Animal Service가 응답 직전에 현재 상태를 다시 확인한다. 공고 종료일만으로 제외하지 않는다.
- 전체/동물 영역 벡터의 기존0.7 조합과 같은 색 보정 함수를 재사용한다. 기존 추천의 최소 시각 점수0.6은 재정렬 전에 적용하며 최대6개 ID를 반환한다.0.6은 DINOv3 정확도를 실사진으로 보정한 기준이나 동일 개체 확률이 아니다. 색 가중치는 `LOST_SEARCH_COAT_COLOR_WEIGHT`를 공유한다.
- DB 조회는 같은 공개본의 짧은 읽기 트랜잭션에서 끝내고 색 재정렬 전에 연결을 반납한다. GPU DB 역할에는 갤러리 테이블 권한 외에 `animals`의 `id`, `species`, `status` 컬럼 SELECT 권한이 필요하다. 계정·GRANT는 승인된 이관 단계에서 구성한다.
- 운영 전환 게이트: 현재 `animals` 이관, 완성된 모델/색 계약의 갤러리 이관, 위 최소 조회 권한, 내부 중계의 새 GET 경로/키 전달, 실제 사용자 사진/추천 순위·응답 시간 비교를 확인한다. 로컬 단위/격리 DB 검증만으로 운영 공개를 판단하지 않는다.
- 롤백은 승인된 이전 코드·프로필·ES 환경과 갤러리를 복원한다. PostgreSQL 벡터를384차원으로 변환하지 않는다. 빈 PG에서 무조건 자동 갤러리를 시작하면 재추론이 발생하므로 이관 검증 전에 활성화하지 않는다.

### PostgreSQL 시작 검사와 system unit 후보 (운영 미설치)

`check_storage.py`는 모델을 로딩하기 전에 선택한 저장소만 검사한다. PG에서는
3초 연결/쿼리 제한과 읽기 전용 연결로 DB 이름, 제한된 역할, 1024차원 컬럼,
갤러리 및 현재 상태 읽기 권한을 확인하고 연결을 닫는다. 실제 공개 갤러리의
완성도와 모델 버전은 기존 애플리케이션 검증이 담당한다. 오류에는 DSN/비밀을
출력하지 않는다. ES 기본값은 유지한다.

`postgresql-cutover/`의 두 파일은 현재 확인한 `/home/shyu` system unit 전용
후보이며 아직 설치되지 않았다. 실제 unit의 모든 ExecStartPre를 먼저 확인하고,
승인된 설치에서 기존 `check_es.py` 하나를 `check_storage.py`로 교체한다.
`ExecStartPre=`는 기존 훅을 전부 지우므로 새 훅이 추가되어 있으면 보존해야 한다.
터널의 PG 선택과 GPU의 `LOST_STORAGE_BACKEND=postgresql`, 비공개 `LOST_PG_DSN`
(loopback15432, 제한된 vector 역할)을 함께 전환한다. 기존 환경 파일의 내부 API
키·모델 경로·갤러리 키 등은 유지한다. 갤러리 프로세스에도 같은 저장소 설정이
필요하다. 파일 복사나 systemd reload/restart는 이 코드 변경에 포함하지 않는다.

롤백은 승인된 변경 전 unit/env/실행 파일 사본으로 복구한다. 단, PostgreSQL에
새 쓰기가 발생한 이후에는 설정만 MySQL/ES로 되돌리는 것이 데이터 롤백이 아니다.
새 쓰기의 역반영과 CDC 정합성을 확인하기 전에는 이전 writer를 열지 않는다.
