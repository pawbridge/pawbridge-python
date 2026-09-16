import hashlib
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
from PIL import Image
from app.services.inference_gate import InferenceGate
from app.services.gallery_source import GallerySource
from app.services.gallery_runtime import GalleryRefresh, runtime_owner

HOST = 'test.r2.cloudflarestorage.com'


def sample():
    data = io.BytesIO()
    with Image.new('RGB', (8, 8), 'red') as image:
        image.save(data, format='PNG')
    data = data.getvalue()
    sha = hashlib.sha256(data).hexdigest()
    record = {'id': 1, 'species': 'DOG', 'source_sha256': sha, 'status': 'PROTECT'}
    photo = {'sha256': sha, 'bytes': len(data), 'mime': 'image/png',
             'url': f'https://{HOST}/pawbridge-animal-originals/apms/photos/{sha}.png?X-Amz-Signature=test'}
    return data, {'complete': True, 'count': 1, 'records': [record], 'photos': [photo]}


class GalleryRuntimeTest(unittest.TestCase):
    def test_waiting_search_runs_before_next_gallery_image_and_lock_is_released_after_failure(self):
        gate = InferenceGate()
        order = []
        with gate.acquire(background=True):
            def background():
                with gate.acquire(background=True): order.append('gallery')
            def search():
                with gate.acquire(): order.append('search')
            low = threading.Thread(target=background); high = threading.Thread(target=search)
            low.start(); high.start()
            with gate.condition:
                self.assertTrue(gate.condition.wait_for(lambda: gate.searches == 1, timeout=2))
        high.join(2); low.join(2)
        self.assertEqual(order, ['search', 'gallery'])
        with self.assertRaises(ValueError), gate.acquire(): raise ValueError('inference failed')
        with gate.acquire(background=True): self.assertFalse(gate.searches)

    def test_feed_fetches_metadata_only_and_inference_photo_is_private_and_temporary(self):
        data, payload = sample(); calls = []
        def respond(request):
            calls.append(request)
            if request.url.host == '127.0.0.1':
                if request.headers.get('if-none-match'):
                    return httpx.Response(304)
                return httpx.Response(200, json=payload, headers={'etag': '"'+'a'*64+'"'})
            return httpx.Response(200, content=data, headers={'content-type': 'image/png'})
        with tempfile.TemporaryDirectory() as directory:
            source = GallerySource('http://127.0.0.1/feed', 'private-key', HOST, directory, Path(directory)/'photos',
                                   httpx.Client(transport=httpx.MockTransport(respond)))
            first = source.fetch(None, lambda: False)
            self.assertEqual(source.downloaded, 0)
            self.assertFalse(any(r.url.host == HOST for r in calls))
            with source.photo(payload['records'][0], lambda: False) as path:
                self.assertEqual(path.read_bytes(), data)
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertFalse(path.exists())
            self.assertEqual(list(source.photo_root.iterdir()), [])
            self.assertEqual(source.downloaded, 1)
            self.assertIsNone(source.fetch(first['etag'], lambda: False))
            remote = [r for r in calls if r.url.host == HOST]
            self.assertEqual(len(remote), 1)
            self.assertNotIn('x-internal-api-key', remote[0].headers)
            self.assertNotIn('X-Amz-Signature', (Path(directory)/'incoming.json').read_text())
            source.close()

    def test_incomplete_foreign_origin_or_corrupt_photo_is_rejected_without_retained_files(self):
        data, good = sample()
        for case in ['incomplete', 'foreign', 'corrupt']:
            payload = json.loads(json.dumps(good))
            if case == 'incomplete': payload['count'] = 2
            if case == 'foreign': payload['photos'][0]['url'] = 'http://169.254.169.254/credentials'
            downloads = []
            def respond(request):
                if request.url.host == '127.0.0.1':
                    return httpx.Response(200, json=payload, headers={'etag':'"'+'a'*64+'"'})
                downloads.append(request)
                return httpx.Response(200, content=b'bad', headers={'content-type':'image/png'})
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                source = GallerySource('http://127.0.0.1/feed','private-key',HOST,directory,Path(directory)/'photos',
                                       httpx.Client(transport=httpx.MockTransport(respond)))
                with self.assertRaises(ValueError):
                    source.fetch(None, lambda: False)
                    with source.photo(payload['records'][0], lambda: False): pass
                self.assertEqual(list((Path(directory)/'photos').iterdir()), [])
                if case != 'corrupt': self.assertEqual(downloads, [])
                source.close()

    def test_failed_build_retries_old_etag_and_unchanged_snapshot_recovers_missing_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Mock(photo_root=Path(directory), downloaded=1); es = Mock()
            refresh = GalleryRefresh(es, SimpleNamespace(model_version='test'), source, 'animals-lost-dinov3-sam3-runtime-test', directory)
            refresh.retention = Mock()
            refresh.etag = 'old'; refresh.status['lastSuccess'] = 'previous-success'
            snapshot = {'manifest_path':'manifest','etag':'new','downloaded':1,'reused_photos':0,'snapshot_sha256':'a'*64}
            source.fetch.return_value = snapshot
            with patch('app.services.gallery_runtime.build_gallery', side_effect=RuntimeError('bulk failed')):
                with self.assertRaises(RuntimeError): refresh.cycle()
            self.assertEqual(refresh.etag,'old')
            self.assertEqual(refresh.status['lastSuccess'],'previous-success')
            self.assertTrue(refresh.ready)
            source.fetch.side_effect = [None, snapshot]
            es.options.return_value.indices.get_alias.side_effect = RuntimeError('alias unavailable')
            result = {'index':'new-index','records':1,'snapshot_sha256':'a'*64}
            with patch('app.services.gallery_runtime.build_gallery',return_value=result): refresh.cycle()
            self.assertEqual(source.fetch.call_args_list[-2].args[0],'old')
            self.assertIsNone(source.fetch.call_args_list[-1].args[0])
            self.assertEqual(refresh.etag,'new'); self.assertTrue(refresh.ready)
            self.assertEqual(refresh.status['result'],result)
            with runtime_owner(directory):
                with self.assertRaises(BlockingIOError), runtime_owner(directory): pass

    def test_retention_only_deletes_recorded_unaliased_owned_builds_and_survives_restart(self):
        from app.services.gallery_retention import GalleryRetention
        from app.services.lost_gallery import gallery_target, gallery_mapping
        with tempfile.TemporaryDirectory() as directory:
            es = Mock(); es.options.return_value = es
            alias = 'animals-lost-dinov3-sam3-retention-test'
            keeper = GalleryRetention(es, directory, alias)
            targets = [gallery_target(c * 64) for c in 'abc']
            es.indices.exists.return_value = True
            es.indices.get_alias.side_effect = lambda index: {index: {'aliases': {'other-gallery': {}}}}
            es.indices.get_mapping.side_effect = lambda index: {index: {'mappings': gallery_mapping('a' * 64)}}
            for digest, target in zip('abc', targets):
                keeper.prepare(digest * 64); keeper.published(target)
            es.indices.delete.assert_not_called()
            resumed = GalleryRetention(es, directory, alias)
            self.assertEqual(resumed.journal['published'], targets[1:])
            es.indices.get_alias.side_effect = lambda index: {index: {'aliases': {}}}
            es.indices.get_mapping.side_effect = lambda index: {index: {'mappings': {'_meta': {'contract': 'foreign'}}}}
            with self.assertRaises(RuntimeError): resumed.prune(set(targets[1:]))
            es.indices.delete.assert_not_called()
            es.indices.get_mapping.side_effect = lambda index: {index: {'mappings': gallery_mapping('a' * 64)}}
            resumed.prune(set(targets[1:]))
            es.indices.delete.assert_called_once_with(index=targets[0])
            self.assertEqual(resumed.journal['known'], targets[1:])

    def test_url_renewal_preserves_build_manifest_and_cleans_up_on_inference_failure(self):
        data, original = sample()
        changed = json.loads(json.dumps(original))
        changed['records'][0]['color'] = 'new metadata'
        changed['photos'][0]['url'] += '-renewed'
        feeds = []; downloads = []
        def respond(request):
            if request.url.host == '127.0.0.1':
                feeds.append(request)
                payload = original if len(feeds) == 1 else changed
                return httpx.Response(200, json=payload, headers={'etag': '"' + ('a' if len(feeds) == 1 else 'b') * 64 + '"'})
            downloads.append(request)
            if 'renewed' not in str(request.url): return httpx.Response(403)
            return httpx.Response(200, content=data, headers={'content-type': 'image/png'})
        with tempfile.TemporaryDirectory() as directory:
            source = GallerySource('http://127.0.0.1/feed', 'private', HOST, directory,
                                   Path(directory)/'photos', httpx.Client(transport=httpx.MockTransport(respond)))
            snapshot = source.fetch(None, lambda: False)
            manifest = snapshot['manifest_path'].read_bytes()
            source.urls_refreshed_at -= 120
            with self.assertRaisesRegex(RuntimeError, 'inference failed'):
                with source.photo(original['records'][0], lambda: False) as path:
                    self.assertEqual(path.read_bytes(), data)
                    raise RuntimeError('inference failed')
            self.assertEqual(len(feeds), 2)
            self.assertEqual(len(downloads), 2)
            self.assertEqual(snapshot['manifest_path'].read_bytes(), manifest)
            self.assertEqual(list(source.photo_root.iterdir()), [])
            self.assertFalse(list(Path(directory).glob('url-refresh-*')))
            self.assertTrue(all('x-internal-api-key' not in r.headers for r in downloads))
            source.close()

    def test_unusable_renewed_url_is_retried_once_and_never_keeps_a_photo(self):
        _, payload = sample(); calls = []
        def respond(request):
            calls.append(request)
            if request.url.host == '127.0.0.1':
                return httpx.Response(200, json=payload, headers={'etag': '"'+'a'*64+'"'})
            return httpx.Response(403)
        with tempfile.TemporaryDirectory() as directory:
            source = GallerySource('http://127.0.0.1/feed', 'private', HOST, directory,
                                   Path(directory)/'photos', httpx.Client(transport=httpx.MockTransport(respond)))
            source.fetch(None, lambda: False)
            source.urls_refreshed_at -= 120
            with self.assertRaises(httpx.HTTPStatusError):
                with source.photo(payload['records'][0], lambda: False): pass
            self.assertEqual(len(calls), 4)
            self.assertEqual(list(source.photo_root.iterdir()), [])
            source.close()

    def test_unchanged_feed_rebuilds_when_published_color_contract_is_stale(self):
        from app.services.lost_gallery import gallery_mapping
        with tempfile.TemporaryDirectory() as directory:
            source = Mock(photo_root=Path(directory), downloaded=0)
            es = Mock(); es.options.return_value = es
            refresh = GalleryRefresh(es, SimpleNamespace(model_version="test"), source,
                                     "animals-lost-dinov3-sam3-runtime-test", directory)
            refresh.retention = Mock()
            result = {"index": "old", "records": 1, "snapshot_sha256": "a"*64}
            refresh.last_result = result; refresh.etag = "unchanged"
            mapping = gallery_mapping("a"*64); mapping["_meta"].pop("coat_color_version")
            es.indices.get_alias.return_value = {"old": {}}
            es.indices.get_mapping.return_value = {"old": {"mappings": mapping}}
            es.count.return_value = {"count": 1}
            snapshot = {"manifest_path": "manifest", "etag": "unchanged", "snapshot_sha256": "a"*64}
            source.fetch.side_effect = [None, snapshot]
            with patch("app.services.gallery_runtime.build_gallery", return_value=dict(result, index="new")) as build:
                refresh.cycle()
            self.assertEqual([call.args[0] for call in source.fetch.call_args_list], ["unchanged", None])
            build.assert_called_once()
            self.assertEqual(refresh.last_result["index"], "new")
