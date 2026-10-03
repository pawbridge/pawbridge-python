"""Local HTML report with no external scripts, tracking, or inferred labels."""
from html import escape
import json
from pathlib import Path
import shutil


def text(value):
    return escape(str(value), quote=True)


def number(value):
    return '미측정' if value is None else f'{value:.4f}'


def write_report(output, dataset, evaluations, provenance, *, feature_root=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    assets = output / 'assets'
    assets.mkdir()
    for photo in dataset.photos:
        # Re-encode bounded decoded input to discard EXIF. This is an evaluation
        # artifact, never a replacement for the source image or its hash.
        from app.services.lost_search import decode_photo
        with decode_photo(photo.path.read_bytes()) as decoded:
            # At most 330 previews of 768x768. Do not replicate full-size source
            # photos in each ranking report or retain them all in RAM.
            decoded.thumbnail((768, 768))
            decoded.save(assets / f'{photo.id}.png')
    if feature_root is not None:
        for variant in sorted({item['variant']['name'] for item in evaluations}):
            for photo in dataset.photos:
                source = Path(feature_root) / 'inputs' / variant / f'{photo.id}.jpg'
                if source.is_file():
                    shutil.copyfile(source, assets / f'{variant}-{photo.id}.jpg')
    payload = {'provenance': provenance, 'evaluations': evaluations}
    (output / 'results.json').write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2))
    chunks = ['<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
              '<title>실종동물 검색 품질 비교</title><style>body{font:16px/1.6 system-ui;background:#f5f5f2;color:#202322;margin:0;padding:24px}main{max-width:1500px;margin:auto}h1{line-height:1.2}.notice{background:#fff4ce;padding:16px;border-radius:8px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:16px}.card{background:white;padding:16px;border-radius:10px}img{width:100%;height:280px;object-fit:contain}table{border-collapse:collapse;width:100%;background:white}td,th{padding:8px;border:1px solid #ddd;text-align:left}code{word-break:break-all}summary{cursor:pointer;font-weight:700}details{margin:24px 0}svg{width:100%;max-height:260px}.small{font-size:13px;word-break:break-word}pre{white-space:pre-wrap}</style><main>',
              '<h1>실종동물 검색 품질 비교</h1>',
              f'<p class="notice">실행 구분: <strong>{text(provenance.get("execution", "unknown"))}</strong>. '
              '고정된 소규모 후보 집합의 실험이며 운영 전체 검색 정확도가 아닙니다. '
              '정답 미확인 사진은 정확도에 포함하지 않습니다. 점수는 동일 개체 확률이 아닙니다.</p>',
              f'<p>자료: {text(dataset.name)} · 질의 {len(dataset.queries)}장 · 후보 {len(dataset.gallery)}장</p>',
              f'<p class="small">자료 SHA256: <code>{text(dataset.sha256)}</code></p>']
    for stage in ('sam', 'dino'):
        measurements = provenance.get(stage)
        if not measurements:
            continue
        chunks.append(f'<h2>{text(stage.upper())} 자원·속도</h2><p>기동 {number(measurements["startup_seconds"])}초 · 프로세스 RAM 피크 {number(measurements["peak_process_rss_bytes"]/1024**3)} GiB · PyTorch GPU 할당 피크 {number(measurements["peak_cuda_allocated_bytes"]/1024**3)} GiB</p><p>단계별 직렬 실험입니다. 아래 처리량은 운영 전체 API 지연이나 동시 요청 처리량이 아닙니다. 각 방식 첫 장을 제외했습니다.</p><table><tr><th>방식</th><th>측정 장수</th><th>장/초</th><th>장/분</th><th>중앙값 초</th><th>p95 초</th></tr>')
        for name, rate in measurements['rates'].items():
            chunks.append(f'<tr><td>{text(name)}</td><td>{rate["warm_images"]}</td><td>{number(rate["images_per_second"])}</td><td>{number(rate["images_per_minute"])}</td><td>{number(rate["median_seconds"])}</td><td>{number(rate["p95_seconds"])}</td></tr>')
        chunks.append('</table>')
    if not evaluations:
        chunks.append('<h2>입력 사진 확인 · AI 비교 아직 미실행</h2><p>아래는 비교에 사용할 자료입니다. 기존 검색 후보의 나열 순서는 이번 실험 순위가 아닙니다. 아직 개선된 순위나 정확도 결과는 없습니다.</p><div class="grid">')
        for photo in dataset.photos:
            label = ('입력 사진 · '+photo.id if photo.role == 'query'
                     else '기존 검색 후보 · 공고 ID '+str(photo.metadata['animal_id']))
            chunks.append(f'<article class="card"><strong>{text(label)}</strong><p>동일 개체 정답: {text(photo.truth)} · 새 실험 순위 없음</p><img src="assets/{text(photo.id)}.png" loading="lazy" alt="평가 입력"></article>')
        chunks.append('</div>')
    for evaluation in evaluations:
        variant, ranking = evaluation['variant']['name'], evaluation['ranking']['name']
        chunks.append(f'<h2>{text(variant)} / {text(ranking)}</h2><table><tr><th>집합</th><th>정답 확인 질의</th><th>개체 수</th><th>Top-1</th><th>Top-5</th><th>Top-20</th><th>최초 후보 포함률</th></tr>')
        for split, metric in evaluation['metrics'].items():
            chunks.append(f'<tr><td>{text(split)}</td><td>{metric["verified_present_queries"]}</td><td>{metric["verified_identities"]}</td><td>{number(metric["recall_at"]["1"])}</td><td>{number(metric["recall_at"]["5"])}</td><td>{number(metric["recall_at"]["20"])}</td><td>{number(metric["first_stage_recall"])}</td></tr>')
        chunks.append('</table>')
        for result in evaluation['queries']:
            qid = result['query_id']
            chunks.append(f'<details open><summary>{text(qid)} · 정답 {text(result["truth"])} · 마스크 {text(result["query_focus_status"])}</summary><p>대상 {result["eligible_count"]}개 → 최초 후보 {result["candidate_count"]}개. 후보 상한 적용: {result["pool_truncated"]}</p><div class="grid"><article class="card"><strong>질의 원본</strong><img src="assets/{text(qid)}.png" alt="질의 원본"></article>')
            input_name = f'{variant}-{qid}.jpg'
            if (assets / input_name).is_file():
                chunks.append(f'<article class="card"><strong>실제 동물 경로 입력</strong><img src="assets/{text(input_name)}" alt="모델 입력"></article>')
            for rank, candidate in enumerate(result['top20'], 1):
                cid = candidate['image_id']
                chunks.append(f'<article class="card"><strong>{rank}위 · {text(cid)}</strong><img src="assets/{text(cid)}.png" loading="lazy" alt="후보 원본"><p class="small">전체 {number(candidate["full_score"])} / 동물 {number(candidate["animal_score"])}<br>색상 감점 {number(candidate["color_penalty"])} / 부가 가산 {number(candidate["metadata_boost"])}<br>최종 {number(candidate["final_score"])}<br>마스크 {text(candidate["focus_status"])} / 전체 사진 대체 {candidate["animal_fallback"]}</p>')
                patch = candidate['patch']
                if patch:
                    chunks.append(f'<p class="small">패치 점수 {number(patch["score"])} · 상호 대응 {patch["matches"]}개<br>대응 비율 질의 {number(patch["query_coverage"])} / 후보 {number(patch["candidate_coverage"])}</p>')
                    qimage, cimage = f'{variant}-{qid}.jpg', f'{variant}-{cid}.jpg'
                    if (assets / qimage).is_file() and (assets / cimage).is_file():
                        chunks.append(f'<svg viewBox="0 0 520 256" role="img" aria-label="학습 특징 대응 위치. 해부학적 부위 판정 아님"><image href="assets/{text(qimage)}" width="256" height="256"/><image href="assets/{text(cimage)}" x="264" width="256" height="256"/>')
                        for index, pair in enumerate(patch['pairs']):
                            qx, qy = pair['query_xy']; cx, cy = pair['candidate_xy']
                            color = f'hsl({index*37%360} 85% 45%)'
                            chunks.append(f'<line x1="{qx*256:.1f}" y1="{qy*256:.1f}" x2="{264+cx*256:.1f}" y2="{cy*256:.1f}" stroke="{color}" stroke-width="2"/>')
                        chunks.append('</svg><p class="small">선은 패치 대응 가설이며 귀·얼굴 식별 결과가 아닙니다.</p>')
                chunks.append('</article>')
            chunks.append('</div></details>')
    chunks.append('<details><summary>실행 조건과 자원 기록</summary><pre>'+text(json.dumps(provenance, ensure_ascii=False, indent=2))+'</pre></details></main></html>')
    (output / 'index.html').write_text('\n'.join(chunks))
