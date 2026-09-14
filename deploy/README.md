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
