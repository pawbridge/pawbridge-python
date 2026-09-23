"""CPU contracts only: these tests do not measure actual model accuracy."""
from dataclasses import replace
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch
import numpy as np
from PIL import Image
import unittest
import tempfile
from app.services.lost_search import rank_candidates
from evaluation.lost_search.dataset import Dataset, Photo, audit, digest, load_dataset
from evaluation.lost_search.features import VARIANTS, load_features, save_features, unit
from evaluation.lost_search.extract import native_view, pool_tokens, require_idle_service
from evaluation.lost_search.ranking import Ranking, eligible, evaluate_query, metrics, patch_match
from evaluation.lost_search.report import write_report
from evaluation.lost_search.__main__ import compare, exclusive_lock, source_fingerprint, supervise

def build_manifest(tmp_path):
    photos = []
    for (index, role) in enumerate(('query', 'gallery')):
        image = tmp_path / f'{index}.png'
        with Image.new('RGB', (80, 80), (index * 100, 50, 10)) as value:
            value.save(image)
        photos.append({'id': f'p{index}', 'file': image.name, 'sha256': digest(image), 'role': role, 'species': 'DOG', 'partition': 'holdout', 'identity': 'dog1', 'capture': f'session{index}', 'truth': 'present' if role == 'query' else 'unknown', 'verification': 'Independent captures verified for synthetic fixture only', 'source': 'Synthetic CPU test', 'metadata': {} if role == 'query' else {'animal_id': 1, 'status': 'PROTECT'}})
    path = tmp_path / 'manifest.json'
    data = {'version': 1, 'name': 'synthetic', 'photos': photos}
    path.write_text(json.dumps(data))
    return (path, data)

def changed(manifest, change):
    (path, data) = manifest
    data = deepcopy(data)
    change(data)
    path.write_text(json.dumps(data))
    return path

def vector(first, second=0):
    value = np.zeros(1024, np.float32)
    value[:2] = (first, second)
    return unit(value)

def feature(full=None, animal=None, patches=None):
    patches = np.zeros((0, 1024), np.float32) if patches is None else patches
    return {'full': vector(1) if full is None else full, 'animal': np.array([], np.float32) if animal is None else animal, 'patches': patches, 'coordinates': np.full((len(patches), 2), 0.5, np.float32)}

class LostQualityEvaluationTest(unittest.TestCase):

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest = build_manifest(self.root)

    def test_gpu_job_lock_rejects_overlap_and_releases_after_failure(self):
        with self.assertRaisesRegex(ValueError, 'intentional'):
            with exclusive_lock('unit-test'):
                with self.assertRaisesRegex(RuntimeError, 'Another'):
                    with exclusive_lock('unit-test'):
                        self.fail('overlap permitted')
                raise ValueError('intentional')
        with exclusive_lock('unit-test'):
            pass

    def test_complete_synthetic_features_produce_traceable_comparison_report(self):
        from dataclasses import asdict
        from types import SimpleNamespace
        from app.services.dinov3 import WEIGHTS_SHA256
        dataset = load_dataset(self.manifest[0])
        root = self.root / 'features'
        variant = VARIANTS[0]
        for photo in dataset.photos:
            save_features(root / 'features' / variant.name / photo.id, photo, variant,
                          **feature(), metadata={'focus_status': 'original', 'weights_sha256': WEIGHTS_SHA256})
        (root / 'run.json').write_text(json.dumps({'status': 'completed', 'execution': 'synthetic_cpu_contract_only',
            'dataset': audit(dataset), 'source_sha256': source_fingerprint(), 'variants': [asdict(variant)]}))
        output = self.root / 'result'
        compare(SimpleNamespace(features=root, output=output), dataset, [variant])
        result = json.loads((output / 'results.json').read_text())
        assert len(result['evaluations']) == 4
        assert result['provenance']['execution'] == 'synthetic_cpu_contract_only'
        assert result['evaluations'][0]['metrics']['holdout']['recall_at']['1'] == 1.
        assert 'synthetic_cpu_contract_only' in (output / 'index.html').read_text()

    def test_ground_truth_audit_preserves_independent_captures(self):
        manifest = self.manifest
        dataset = load_dataset(manifest[0])
        assert audit(dataset)['present_queries'] == 1
        assert dataset.queries[0].capture != dataset.gallery[0].capture

    def test_invalid_or_leaked_benchmark_is_rejected(self):
        manifest = self.manifest
        for change in [lambda d: d['photos'][0].update(file='../escape.png'), lambda d: d['photos'][0].update(sha256='0' * 64), lambda d: d['photos'][1].update(file='0.png', sha256=d['photos'][0]['sha256']), lambda d: d['photos'][1].update(partition='tune'), lambda d: d['photos'][1].update(capture='session0'), lambda d: d['photos'][0].update(truth='absent'), lambda d: d['photos'][0].update(verification=None), lambda d: d['photos'][0]['metadata'].update(lost_date='2026-02-30')]:
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    load_dataset(changed(manifest, change))

    def test_duplicate_identity_is_not_scored_as_new_individual(self):
        manifest = self.manifest
        dataset = load_dataset(manifest[0])
        (q, candidate) = (dataset.queries[0], dataset.gallery[0])
        get = lambda p: (feature(), {'focus_status': 'animal_mask'})
        result = evaluate_query(q, [candidate], get, Ranking('current'))
        values = metrics([result, dict(result, query_id='other-photo')])['holdout']
        assert values['verified_present_queries'] == 2
        assert values['verified_identities'] == 1

    def test_unknown_identity_is_not_reported_as_accuracy_or_absence(self):
        manifest = self.manifest
        dataset = load_dataset(manifest[0])
        q = replace(dataset.queries[0], truth='unknown', identity=None)
        result = evaluate_query(q, dataset.gallery, lambda p: (feature(), {'focus_status': 'original'}), Ranking('current'))
        values = metrics([result])['holdout']
        assert values['unverified_queries'] == 1
        assert values['verified_absent_queries'] == 0
        assert values['recall_at']['20'] is None

    def test_date_filter_runs_before_candidate_cap_and_region_is_soft(self):
        manifest = self.manifest
        dataset = load_dataset(manifest[0])
        q = replace(dataset.queries[0], metadata={'lost_date': '2026-09-11', 'region': '용인'})
        base = dataset.gallery[0]
        old = replace(base, id='old', identity='other', metadata={'animal_id': 1, 'status': 'PROTECT', 'happen_date': '2026-09-10'})
        same = replace(base, id='positive', metadata={'animal_id': 2, 'status': 'PROTECT', 'happen_date': '2026-09-11', 'happen_place': '서울'})
        unknown = replace(base, id='unknown', identity=None, metadata={'animal_id': 3, 'status': 'PROTECT'})
        get = lambda p: (feature(full=vector(1, 0 if p.id in {q.id, 'old'} else 1)), {'focus_status': 'original'})
        current = evaluate_query(q, [old, same], get, Ranking('current', pool=1))
        proposed = evaluate_query(q, [old, same], get, Ranking('filter', pool=1, date_policy='filter'))
        assert current['final_positive_rank'] is None
        assert proposed['top20'][0]['image_id'] == 'positive'
        assert eligible(q, unknown, Ranking('filter', date_policy='filter'))
        assert eligible(replace(q, metadata={}), old, Ranking('filter', date_policy='filter'))

    def test_status_selection_does_not_mix_species_or_splits(self):
        manifest = self.manifest
        dataset = load_dataset(manifest[0])
        (q, g) = (dataset.queries[0], dataset.gallery[0])
        settings = Ranking('current')
        adopted = replace(g, metadata=dict(g.metadata, status='ADOPTED'))
        assert not eligible(q, adopted, settings)
        assert eligible(replace(q, metadata={'include_adopted_or_returned': True}), adopted, settings)
        for other in (replace(g, species='CAT'), replace(g, partition='tune'), replace(g, metadata=dict(g.metadata, status='EUTHANIZED'))):
            assert not eligible(q, other, settings)

    def test_current_ranking_matches_production_auxiliary_and_tie_order(self):
        manifest = self.manifest
        dataset = load_dataset(manifest[0])
        (q, g) = (dataset.queries[0], dataset.gallery[0])
        q = replace(q, metadata={'lost_date': '2026-09-11', 'region': '용인', 'query_description': '갈색'})
        candidates = [replace(g, id=f'g{i}', metadata={'animal_id': i, 'status': 'PROTECT', 'happen_date': '2026-09-12', 'happen_place': '용인' if i == 2 else '서울', 'color': '갈색'}) for i in (3, 2, 1)]
        result = evaluate_query(q, candidates, lambda p: (feature(), {'focus_status': 'original'}), Ranking('current'))
        from datetime import date
        expected = rank_candidates([{'_score': 2.0, '_source': dict(p.metadata, id=p.metadata['animal_id'])} for p in candidates], date(2026, 9, 11), '용인', '갈색')
        assert [r['animal_id'] for r in result['top20']] == [r['animalId'] for r in expected] == [2, 1, 3]
        assert result['top20'][0]['metadata_boost'] == 0.03

    def test_missing_animal_vector_falls_back_without_invented_similarity(self):
        manifest = self.manifest
        dataset = load_dataset(manifest[0])
        (q, g) = (dataset.queries[0], dataset.gallery[0])
        result = evaluate_query(q, [g], lambda p: (feature(animal=vector(0, 1) if p.role == 'query' else None), {'focus_status': 'original'}), Ranking('animal-only', animal_weight=1.0))
        assert result['top20'][0]['animal_score'] is None
        assert result['top20'][0]['final_score'] == 1.0

    def test_patch_matching_requires_bilateral_coverage_and_is_position_independent(self):
        patches = np.eye(1024, dtype=np.float32)[:8]
        result = patch_match(feature(patches=patches), feature(patches=patches[::-1]))
        assert result['score'] == 1.0
        assert result['matches'] == 8
        repeated = np.repeat(patches[:1], 8, axis=0)
        assert patch_match(feature(patches=repeated), feature(patches=repeated))['score'] is None
        many = np.eye(1024, dtype=np.float32)[:64]
        assert patch_match(feature(patches=patches), feature(patches=many))['score'] is None

    def test_pooling_keeps_cls_registers_and_background_separate(self):
        tokens = np.zeros((9, 1024), np.float32)
        tokens[:5, 0] = 1
        tokens[5:, 1] = 1
        tokens[5, 2] = 2
        mask = np.zeros((32, 32), np.uint8)
        mask[:16, :16] = 255
        (cls, _, _) = pool_tokens(tokens, 5, mask, 'cls')
        (foreground, patches, coordinates) = pool_tokens(tokens, 5, mask, 'foreground')
        (average, _, _) = pool_tokens(tokens, 5, mask, 'avg')
        assert cls[0] == 1 and average[0] == 0
        assert foreground[2] > average[2]
        assert patches.shape == (1, 1024)
        np.testing.assert_allclose(coordinates, [[0.25, 0.25]])

    def test_high_resolution_patch_selection_is_bounded(self):
        tokens = np.ones((5 + 1024, 1024), np.float32)
        (_, patches, coordinates) = pool_tokens(tokens, 5, np.full((512, 512), 255, np.uint8), 'foreground')
        assert patches.shape == (256, 1024)
        assert np.unique(coordinates, axis=0).shape == (256, 2)

    def test_native_crop_uses_original_pixels_and_preserves_source(self):
        with Image.new('RGB', (1000, 600), 'red') as original, Image.new('L', (100, 60), 255) as mask:
            before = original.tobytes()
            (view, alpha) = native_view(original, mask, [0, 0, 100, 60], 384)
            with view, alpha:
                assert view.size == (384, 384)
                assert view.getpixel((192, 192)) == (255, 0, 0)
                assert alpha.getpixel((192, 0)) == 0
                assert alpha.getpixel((192, 192)) == 255
            assert original.tobytes() == before

    def test_feature_records_reject_stale_variant_and_changed_bytes(self):
        manifest = self.manifest
        tmp_path = self.root
        photo = load_dataset(manifest[0]).queries[0]
        folder = tmp_path / 'features'
        save_features(folder, photo, VARIANTS[0], **feature(), metadata={'focus_status': 'original'})
        (values, _) = load_features(folder, photo, VARIANTS[0])
        assert values['full'].shape == (1024,)
        with self.assertRaisesRegex(ValueError, 'provenance'):
            load_features(folder, photo, VARIANTS[1])
        with (folder / 'features.npz').open('ab') as output:
            output.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            load_features(folder, photo, VARIANTS[0])

    def test_preview_is_explicitly_unmeasured_and_strips_image_metadata(self):
        manifest = self.manifest
        tmp_path = self.root
        dataset = load_dataset(manifest[0])
        output = tmp_path / 'report'
        write_report(output, dataset, [], {'execution': 'input_inventory_only'})
        assert '아직 개선된 순위나 정확도 결과는 없습니다' in (output / 'index.html').read_text()
        assert json.loads((output / 'results.json').read_text())['evaluations'] == []
        with Image.open(output / 'assets/p0.png') as preview:
            assert not preview.getexif()
            assert max(preview.size) <= 768
        with self.assertRaises(FileExistsError):
            write_report(output, dataset, [], {})

    def test_gpu_worker_fails_closed_while_service_is_not_confirmed_inactive(self):
        for state in ['active', 'activating', 'failed', '', 'reloading']:
            with self.subTest(state=state):
                with patch('evaluation.lost_search.extract.subprocess.run', return_value=subprocess.CompletedProcess([], 0, state)):
                    with self.assertRaises(RuntimeError):
                        require_idle_service()

    def test_supervisor_terminates_worker_on_deadline_and_reports_worker_failure(self):
        started = time.monotonic()
        with self.assertRaises(TimeoutError):
            supervise([sys.executable, '-c', 'import time; time.sleep(30)'], deadline=started + 0.3)
        assert time.monotonic() - started < 5
        with self.assertRaisesRegex(RuntimeError, 'status 7'):
            supervise([sys.executable, '-c', 'raise SystemExit(7)'], deadline=time.monotonic() + 5)

    def test_compare_rejects_partial_extraction_before_reading_vectors(self):
        manifest = self.manifest
        tmp_path = self.root
        from types import SimpleNamespace
        dataset = load_dataset(manifest[0])
        root = tmp_path / 'run'
        root.mkdir()
        (root / 'run.json').write_text(json.dumps({'status': 'failed', 'dataset': audit(dataset), 'source_sha256': source_fingerprint()}))
        args = SimpleNamespace(features=root, output=tmp_path / 'report')
        with self.assertRaisesRegex(ValueError, 'Only complete'):
            compare(args, dataset, VARIANTS[:1])
if __name__ == '__main__':
    unittest.main()
