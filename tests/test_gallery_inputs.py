from contextlib import contextmanager, nullcontext
import hashlib
import io
from pathlib import Path
import tempfile
import threading
import unittest

from PIL import Image, ImageOps
from app.services.gallery_inputs import GalleryInputPrefetch


class GalleryInputsTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        buffer = io.BytesIO()
        with Image.new('RGB', (32, 24), 'red') as image:
            image.save(buffer, format='PNG')
        self.data = buffer.getvalue()
        self.rows = [dict(id=i, source_sha256=hashlib.sha256(self.data).hexdigest()) for i in range(6)]
        self.closed = set()

    @contextmanager
    def provider(self, row):
        path = self.root / str(row['id'])
        path.write_bytes(self.data)
        try:
            yield path
        finally:
            path.unlink()
            self.closed.add(row['id'])

    def test_prepares_ahead_but_blocks_before_exceeding_count_or_byte_limit(self):
        for limit_by_bytes in (False, True):
            with self.subTest(limit_by_bytes=limit_by_bytes):
                count = 2 if limit_by_bytes else 3
                inputs = GalleryInputPrefetch(self.rows, self.provider, self.root,
                    max_bytes=2 * 32 * 24 * 8 if limit_by_bytes else 128 * 1024 * 1024)
                attempted = threading.Event()
                reserve = inputs._reserve
                calls = []
                def observe(size):
                    calls.append(size)
                    if len(calls) == count + 1:
                        attempted.set()
                    return reserve(size)
                inputs._reserve = observe
                with inputs:
                    with inputs.photo(self.rows[0]) as first:
                        self.assertTrue(attempted.wait(5), 'Preparation did not overlap consumption')
                        self.assertEqual(inputs.images, count)
                        self.assertEqual(inputs.bytes, count * 32 * 24 * 8)
                        self.assertEqual(inputs.queue.qsize(), count - 1)
                        self.assertEqual(first.original.getpixel((0, 0)), (255, 0, 0))
                    for row in self.rows[1:]:
                        with inputs.photo(row) as image:
                            self.assertEqual(image.original.size, (32, 24))
                self.assertFalse(inputs.thread.is_alive())
                self.assertEqual((inputs.images, inputs.bytes), (0, 0))
                self.assertEqual(list(self.root.iterdir()), [])
                with self.assertRaises(ValueError):
                    first.original.getpixel((0, 0))

    def test_cancellation_unblocks_a_full_queue_and_closes_images_and_downloads(self):
        cancelled = threading.Event()
        def check():
            if cancelled.is_set():
                raise RuntimeError('build cancelled')
        inputs = GalleryInputPrefetch(self.rows, self.provider, self.root, check, max_images=1)
        waiting = threading.Event()
        reserve = inputs._reserve
        calls = []
        def observe(size):
            calls.append(size)
            if len(calls) == 2:
                waiting.set()
            return reserve(size)
        inputs._reserve = observe
        with self.assertRaisesRegex(RuntimeError, 'build cancelled'), inputs:
            with inputs.photo(self.rows[0]) as first:
                self.assertTrue(waiting.wait(5))
                cancelled.set()
                check()
        self.assertFalse(inputs.thread.is_alive())
        self.assertEqual((inputs.images, inputs.bytes), (0, 0))
        self.assertEqual(list(self.root.iterdir()), [])
        with self.assertRaises(ValueError):
            first.focus.getpixel((0, 0))

    def test_inference_failure_and_provider_cleanup_failure_release_all_storage(self):
        for failure in ('inference', 'provider cleanup'):
            with self.subTest(failure=failure):
                @contextmanager
                def provider(row):
                    with self.provider(row) as path:
                        yield path
                    if failure == 'provider cleanup':
                        raise RuntimeError(failure)
                inputs = GalleryInputPrefetch(self.rows, provider, self.root)
                with self.assertRaisesRegex(RuntimeError, failure), inputs:
                    with inputs.photo(self.rows[0]):
                        raise RuntimeError('inference')
                self.assertFalse(inputs.thread.is_alive())
                self.assertEqual((inputs.images, inputs.bytes), (0, 0))
                self.assertEqual(list(self.root.iterdir()), [])

    def test_invalid_hash_is_rejected_without_leaving_a_preparer_or_photo(self):
        rows = [dict(self.rows[0], source_sha256='0' * 64)]
        inputs = GalleryInputPrefetch(rows, self.provider, self.root)
        with self.assertRaisesRegex(ValueError, 'hash mismatch'), inputs:
            with inputs.photo(rows[0]):
                self.fail('Invalid photo reached inference')
        self.assertFalse(inputs.thread.is_alive())
        self.assertEqual((inputs.images, inputs.bytes), (0, 0))
        self.assertEqual(list(self.root.iterdir()), [])

    def test_exif_orientation_and_thumbnail_match_existing_input_transforms(self):
        buffer = io.BytesIO()
        with Image.new('RGB', (1200, 800), 'blue') as image:
            image.paste('red', (0, 0, 600, 800))
            exif = Image.Exif(); exif[274] = 6
            image.save(buffer, format='JPEG', exif=exif)
        self.data = buffer.getvalue()
        row = dict(id=0, source_sha256=hashlib.sha256(self.data).hexdigest())
        with Image.open(io.BytesIO(self.data)) as source:
            original = ImageOps.exif_transpose(source).convert('RGB')
        self.addCleanup(original.close)
        focus = original.copy(); self.addCleanup(focus.close)
        focus.thumbnail((1024, 1024), Image.Resampling.BICUBIC)
        with GalleryInputPrefetch([row], self.provider, self.root) as inputs:
            with inputs.photo(row) as image:
                self.assertEqual(image.original.size, (800, 1200))
                self.assertEqual(image.original.tobytes(), original.tobytes())
                self.assertEqual(image.focus.tobytes(), focus.tobytes())
                self.assertEqual(list(self.root.iterdir()), [])

    def test_default_budget_still_accepts_the_existing_sixteen_megapixel_limit(self):
        buffer = io.BytesIO()
        with Image.new('RGB', (4000, 4000), 'red') as image:
            image.save(buffer, format='JPEG')
        self.data = buffer.getvalue()
        row = dict(id=0, source_sha256=hashlib.sha256(self.data).hexdigest())
        with GalleryInputPrefetch([row], self.provider, self.root) as inputs:
            with inputs.photo(row) as image:
                self.assertEqual(image.original.size, (4000, 4000))
                self.assertEqual(image.focus.size, (1024, 1024))
        self.assertEqual((inputs.images, inputs.bytes), (0, 0))

    def test_file_boundary_size_pixel_and_animation_rejections_survive_prefetch(self):
        allowed = self.root / "allowed"
        allowed.mkdir()
        for case, message in (("outside", "escapes"), ("size", "10MiB"),
                              ("pixels", "primary still"), ("animation", "primary still")):
            with self.subTest(case=case):
                path = (self.root if case == "outside" else allowed) / "image"
                if case in {"outside", "size"}:
                    path.write_bytes(self.data)
                    if case == "size":
                        with path.open("ab") as file:
                            file.truncate(10 * 1024 * 1024 + 1)
                elif case == "pixels":
                    with Image.new("RGB", (4001, 4000)) as image:
                        image.save(path, format="PNG")
                else:
                    with Image.new("RGB", (8, 8), "red") as first, Image.new("RGB", (8, 8), "blue") as second:
                        first.save(path, format="GIF", save_all=True, append_images=[second])
                row = dict(id=1, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                inputs = GalleryInputPrefetch([row], lambda row: nullcontext(path), allowed)
                with self.assertRaisesRegex(ValueError, message), inputs:
                    with inputs.photo(row):
                        self.fail("Rejected photo reached inference")
                self.assertFalse(inputs.thread.is_alive())
                self.assertEqual((inputs.images, inputs.bytes), (0, 0))
                path.unlink()
