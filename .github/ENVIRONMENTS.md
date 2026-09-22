# 개발 검증과 운영 승격

- 기본 PR 대상은 dev이며 main 대상 PR도 검증한다. 기본 브랜치를 변경할 필요는 없다.
- dev 이미지 CI는 immutable digest를 만든 뒤 인프라 dev의 `environments/dev/isolated-values`만 변경한다. 인프라 환경 계약이 아직 없으면 PR 생성을 중단한다. 현재 운영이 읽는 `environments/dev/values`를 변경하지 않는다.
- main 병합으로 이미지를 재빌드해서 자동 운영 배포하지 않는다. dev에서 E2E 검증한 이미지 digest를 인프라 main의 `environments/prod/values`로 승격한다.
- 코드의 main 통합 결과가 검증한 source와 다르면 새 결과를 dev에서 검증한다. 운영에 미포함인 후속 dev 코드의 이미지를 가져오면 안 된다.
- 인프라 승격 도구는 테스트 근거와 정확한 dev revision/digest를 요구한다. 운영 설정·비밀값을 dev에서 복사하지 않는다.
- main 기준선 통합과 Argo 참조 전환은 별도의 승인 대상이다. 이번 CI 변경은 기존 운영 연결을 바꾸지 않는다.
- GPU WSL 서비스는 컨테이너 CI와 별도 실행 경로다. Python 이미지 빌드 통과를 GPU 서비스 배포로 기록하지 않는다.

## 로컬 dev 실행

개발 환경은 운영 VM이 아니라 인프라 저장소의 `scripts/environments/local_dev.py`로 로컬 Compose에 기동한다. `isolated-values`는 이미지 메타데이터이며 namespace 설정이 아니다. dev CI는 PC에 직접 배포하지 않는다. 로컬 DB는 127.0.0.1:15433, Kafka는 19092, Gateway는 28080이다. 전체 절차는 인프라 `environments/dev/README.md`를 따른다.
