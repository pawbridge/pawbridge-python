from contextlib import contextmanager
from pathlib import Path
import tempfile
from threading import Event, Lock
import unittest

from app.services.gallery_prefetch import PhotoPrefetch


class PhotoPrefetchTest(unittest.TestCase):
    def test_next_download_overlaps_current_use_and_only_one_photo_is_ahead(self):
        rows = [{'id': i} for i in (1, 2, 3)]
        started = {i: Event() for i in (1, 2, 3)}
        active, closed = set(), set()
        lock = Lock()
        with tempfile.TemporaryDirectory() as directory:
            @contextmanager
            def provider(row):
                i = row['id']
                path = Path(directory) / str(i)
                path.write_bytes(b'photo')
                with lock:
                    active.add(i)
                    self.assertLessEqual(len(active), 2)
                started[i].set()
                try:
                    yield path
                finally:
                    path.unlink()
                    with lock:
                        active.remove(i)
                        closed.add(i)

            with PhotoPrefetch(rows, provider) as photos:
                with photos.photo(rows[0]) as first:
                    self.assertTrue(started[2].wait(5), 'Next download did not overlap current inference')
                    self.assertTrue(first.exists())
                    self.assertFalse(started[3].is_set(), 'More than one photo was fetched ahead')
                for row in rows[1:]:
                    with photos.photo(row) as path:
                        self.assertTrue(path.exists())
            self.assertEqual(closed, {1, 2, 3})
            self.assertFalse(active)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_download_failure_is_propagated_and_current_photo_is_closed(self):
        closed = []
        @contextmanager
        def provider(row):
            if row['id'] == 2:
                raise RuntimeError('download failed')
            try:
                yield 'photo'
            finally:
                closed.append(row['id'])
        rows = [{'id': 1}, {'id': 2}]
        with self.assertRaisesRegex(RuntimeError, 'download failed'):
            with PhotoPrefetch(rows, provider) as photos:
                with photos.photo(rows[0]):
                    pass
                with photos.photo(rows[1]):
                    self.fail('Failed download must not be consumed')
        self.assertEqual(closed, [1])

    def test_cancelled_inference_closes_the_already_downloaded_next_photo(self):
        ready, closed = Event(), []
        @contextmanager
        def provider(row):
            if row['id'] == 2:
                ready.set()
            try:
                yield 'photo'
            finally:
                closed.append(row['id'])
        rows = [{'id': 1}, {'id': 2}]
        with self.assertRaisesRegex(RuntimeError, 'inference cancelled'):
            with PhotoPrefetch(rows, provider) as photos:
                with photos.photo(rows[0]):
                    self.assertTrue(ready.wait(5))
                    raise RuntimeError('inference cancelled')
        self.assertCountEqual(closed, [1, 2])
