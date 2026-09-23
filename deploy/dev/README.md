# GPU dev 실행 경계

운영 `pawbridge-lost-search.service`를 재시작하거나 설정을 재사용하지 않는다. 별도 dev 서비스·포트(18091)·DB role·갤러리·state 경로가 필요하다.

`lost-search.env.example`을 **Git 밖**의 dev runtime 설정으로 복사한다. 해당 환경변수를 주입한 서비스의 ExecStartPre에서 `python scripts/check_dev_runtime.py`를 실행한다. 검사 이후에만 `python -m uvicorn app.lost_main:app --host 127.0.0.1 --port 18091 --workers 1`로 실행한다. 이 문서는 설치나 기동 명령을 자동 실행하지 않는다.

dev PostgreSQL은 로컬 Compose의 loopback 포트 15433을 사용하고, `pawbridge_dev_vector` 계정은 dev DB에만 생성한다. 검사기는 접속하지 않으므로 실제 DB 식별·권한 확인을 대신하지 않는다. 자동 갤러리 수집은 초기에는 끄며 승인된 표본 갤러리를 준비해야 startup 검증을 통과한다.

모델 가중치 파일을 읽기 전용으로 공유할 수 있어도 GPU 메모리는 프로세스마다 필요하다. 운영 모델과 동시 실행할 메모리가 검증되지 않은 상태에서는 `LOST_DEV_GPU_ENABLED=false`를 유지한다. 운영 중단·순차 GPU 사용은 별도 승인 대상이다. mock으로 통과한 E2E를 실제 모델 품질 검증이라고 표시하지 않는다.
