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
- WSL `127.0.0.1:13306` → MySQL:3306: 기존 로컬 미리보기 연결 보존.
- VM `127.0.0.1:18091` → WSL `127.0.0.1:18090`: 기존 GPU loopback 연결 보존.
- VM `/home/vagrant/.local/run/pawbridge-gpu/search.sock` → WSL `127.0.0.1:18090`: 운영 Pod 연결. 디렉터리 0700, SSH 소켓 0600, 프록시 UID 1000을 함께 맞춘다.

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
