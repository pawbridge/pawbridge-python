from contextlib import contextmanager
import hashlib
import io
from types import SimpleNamespace
from unittest.mock import Mock
from PIL import Image
import json
import tempfile
import unittest
from pathlib import Path
from app.services.lost_gallery import read_manifest, reusable_document, build_gallery, gallery_mapping
from app.services.sam3_focus import FOCUS_VERSION


class LostGalleryTest(unittest.TestCase):
    def test_incomplete_empty_duplicate_or_invalid_snapshot_is_rejected(self):
        row = {"id": 1, "species": "DOG", "source_sha256": "a" * 64}
        invalid = [{"complete": False, "records": [row]}, {"complete": True, "records": []},
                   {"complete": True, "records": [row, row]},
                   {"complete": True, "records": [dict(row, species="ETC")]},
                   {"complete": True, "records": [dict(row, source_sha256="../bad")]}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            for payload in invalid:
                with self.subTest(payload=payload):
                    path.write_text(json.dumps(payload))
                    with self.assertRaises(ValueError):
                        read_manifest(path)

    def test_snapshot_hash_is_order_independent_but_includes_metadata_changes(self):
        rows = [{"id": i, "species": "DOG", "source_sha256": "a" * 64} for i in [1, 2]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            hashes = []
            for records in [rows, list(reversed(rows)), [dict(rows[0], color="white"), rows[1]]]:
                path.write_text(json.dumps({"complete": True, "records": records}))
                hashes.append(read_manifest(path)[1])
            self.assertEqual(hashes[0], hashes[1])
            self.assertNotEqual(hashes[0], hashes[2])

    def test_only_same_model_species_and_photo_can_reuse_a_valid_vector(self):
        row = {"id": 1, "species": "DOG", "source_sha256": "a" * 64}
        cached = dict(row, model_version=FOCUS_VERSION, image_vector=[1.] + [0.] * 1023,
                      focus_status="original_no_confident_animal")
        self.assertTrue(reusable_document(cached, dict(row, color="updated metadata")))
        for changed in [dict(row, species="CAT"), dict(row, source_sha256="b" * 64)]:
            self.assertFalse(reusable_document(cached, changed))
        for changed in [dict(cached, model_version="old"), dict(cached, image_vector=[0.] * 1024),
                        dict(cached, animal_vector=[float("nan")] * 1024)]:
            self.assertFalse(reusable_document(changed, row))

    def test_builder_fetches_only_stale_vectors_and_preserves_alias_on_failures(self):
        data = io.BytesIO()
        Image.new('RGB', (8, 8), 'red').save(data, format='PNG')
        image_bytes = data.getvalue()
        sha = hashlib.sha256(image_bytes).hexdigest()
        vector = [1.] + [0.] * 1023
        for failure in [None, 'inference', 'bulk']:
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                rows = [dict(id=i, species='DOG', source_sha256=sha, color='updated') for i in [1, 2]]
                manifest = root / 'manifest.json'
                manifest.write_text(json.dumps(dict(complete=True, records=rows)))
                old = 'animals-lost-dinov3-sam3-build-old'
                alias = 'animals-lost-dinov3-sam3-test'
                active = [old]
                es = Mock(); es.options.return_value = es
                es.indices.exists_alias.return_value = True
                es.indices.exists.return_value = False
                es.indices.get_alias.side_effect = lambda **kw: {active[0]: {}}
                es.indices.get_mapping.return_value = {old: {'mappings': gallery_mapping('a'*64)}}
                cached = dict(rows[0], color='old metadata', model_version=FOCUS_VERSION,
                              image_vector=vector, focus_status='original_no_confident_animal')
                stale = dict(rows[1], model_version='old', image_vector=vector)
                es.mget.side_effect = lambda index, ids: {'docs': [dict(found=True, _source=r) for r in [cached, stale]]} if index == old else {'docs': []}
                es.bulk.return_value = {'errors': failure == 'bulk'}
                es.count.return_value = {'count': 2}
                def publish(actions):
                    active[0] = actions[-1]['add']['index']
                    return {'acknowledged': True}
                es.indices.update_aliases.side_effect = publish
                encoder = Mock(model_version=FOCUS_VERSION)
                encoder.encode_with_metadata.return_value = SimpleNamespace(model_version=FOCUS_VERSION,
                    vector=vector, animal_vector=None, focus_status='original_no_confident_animal')
                if failure == 'inference': encoder.encode_with_metadata.side_effect = RuntimeError('inference failed')
                requested = []; paths = []
                @contextmanager
                def photo(row):
                    requested.append(row['id'])
                    with tempfile.TemporaryDirectory(dir=root) as temporary:
                        path = Path(temporary)/'photo.png'; path.write_bytes(image_bytes); paths.append(path)
                        yield path
                def build():
                    return build_gallery(es, lambda: encoder, manifest, root, alias, root, photo_provider=photo)
                if failure:
                    with self.assertRaises(RuntimeError): build()
                    self.assertEqual(active, [old])
                    es.indices.update_aliases.assert_not_called()
                else:
                    result = build()
                    self.assertEqual((result['reused'], result['encoded']), (1, 1))
                    self.assertEqual(active, [result['index']])
                    self.assertEqual(es.bulk.call_args.kwargs['operations'][1]['color'], 'updated')
                self.assertEqual(requested, [2])
                encoder.encode_with_metadata.assert_called_once()
                self.assertTrue(paths and all(not path.exists() for path in paths))
