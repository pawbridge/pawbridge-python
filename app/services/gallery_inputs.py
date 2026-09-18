"""Bounded CPU-only gallery preparation; no GPU tensors or retained photo cache."""
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import hashlib
from pathlib import Path
from queue import Empty, Queue
import threading

from PIL import Image, ImageOps
from app.services.gallery_prefetch import PhotoPrefetch

MAX_PREPARED_IMAGES = 3  # Current image plus at most two upcoming images.
MAX_PREPARED_BYTES = 128 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000


@dataclass(frozen=True)
class GalleryImage:
    original: Image.Image
    focus: Image.Image


class _PreparationStopped(Exception):
    pass


class GalleryInputPrefetch:
    def __init__(self, rows, provider, root, check_cancelled=lambda: None, *,
                 max_images=MAX_PREPARED_IMAGES, max_bytes=MAX_PREPARED_BYTES):
        if type(max_images) is not int or max_images < 1 or type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("Gallery preparation limits must be positive integers")
        self.rows, self.provider, self.root = rows, provider, Path(root).resolve(strict=True)
        self.check_cancelled = check_cancelled
        self.max_images, self.max_bytes = max_images, max_bytes
        self.condition = threading.Condition()
        self.images = self.bytes = 0
        self.stop = threading.Event()
        self.queue = Queue()  # Payload count/bytes are reserved BEFORE decoding.
        self.failure = None
        self.thread = threading.Thread(target=self._run, name="gallery-input")

    def _check(self):
        self.check_cancelled()
        if self.stop.is_set():
            raise _PreparationStopped()

    @contextmanager
    def _reserve(self, size):
        if size > self.max_bytes:
            raise ValueError("Gallery image exceeds the preparation memory budget")
        with self.condition:
            while True:
                self._check()
                if self.images < self.max_images and self.bytes + size <= self.max_bytes:
                    self.images += 1
                    self.bytes += size
                    break
                self.condition.wait(.1)
        try:
            yield
        finally:
            with self.condition:
                self.images -= 1
                self.bytes -= size
                self.condition.notify_all()

    @contextmanager
    def _prepare(self, row, supplied_path):
        path = Path(supplied_path).resolve(strict=True)
        if not path.is_relative_to(self.root):
            raise ValueError("Gallery photo escapes the configured root")
        if path.stat().st_size > 10 * 1024 * 1024:
            raise ValueError("Gallery photo exceeds 10MiB")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                self._check()
                digest.update(chunk)
        if digest.hexdigest() != row["source_sha256"]:
            raise ValueError("Gallery photo hash mismatch")
        # Read the header before reserving decoded RGB storage. Orientation may
        # swap width/height but cannot increase the number of pixels.
        with Image.open(path) as source:
            source.seek(0)
            pixels = source.width * source.height
            if (pixels > MAX_IMAGE_PIXELS
                    or (getattr(source, "n_frames", 1) != 1 and source.format != "MPO")):
                raise ValueError("Gallery photo must have a bounded primary still image")
        # PIL stores RGB pixels in four-byte slots. Reserve the original and
        # full-sized focus copy (before thumbnail), not compressed file bytes.
        # 128MiB admits every existing <=16MP input. Codec/EXIF temporaries,
        # model memory and the inference working set are outside this budget.
        with self._reserve(2 * pixels * 4), ExitStack() as images:
            with Image.open(path) as source:
                source.seek(0)
                oriented = ImageOps.exif_transpose(source)
                try:
                    original = oriented.convert("RGB")
                    images.callback(original.close)
                finally:
                    oriented.close()
            del source, oriented
            focus = original.copy()
            images.callback(focus.close)
            focus.thumbnail((1024, 1024), Image.Resampling.BICUBIC)
            self._check()
            yield GalleryImage(original, focus)

    def _run(self):
        try:
            with PhotoPrefetch(self.rows, self.provider) as photos:
                for row in self.rows:
                    self._check()
                    context = None
                    try:
                        with photos.photo(row) as path:
                            context = self._prepare(row, path)
                            image = context.__enter__()
                        self.queue.put((row["id"], context, image))
                        context = None  # Ownership passes to the consumer.
                    finally:
                        if context is not None:
                            context.__exit__(None, None, None)
        except _PreparationStopped:
            pass
        except BaseException as error:
            self.failure = error
            self.queue.put(error)
        finally:
            self.queue.put(None)

    def __enter__(self):
        self.thread.start()
        return self

    @contextmanager
    def photo(self, row):
        while True:
            self._check()
            try:
                value = self.queue.get(timeout=.1)
                break
            except Empty:
                continue
        if value is None:
            raise RuntimeError("Prepared gallery input ended before the requested photo")
        if isinstance(value, BaseException):
            raise value
        expected_id, context, image = value
        try:
            if expected_id != row["id"]:
                raise RuntimeError("Prepared gallery input order differs from gallery")
            yield image
        finally:
            context.__exit__(None, None, None)

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop.set()
        with self.condition:
            self.condition.notify_all()
        # Do not leave a downloader/preparer alive after the page is abandoned.
        # The existing photo provider owns bounded network deadlines/cancellation.
        self.thread.join()
        while not self.queue.empty():
            value = self.queue.get_nowait()
            if isinstance(value, tuple):
                value[1].__exit__(None, None, None)
        if exc_type is None and self.failure is not None:
            raise self.failure
        return False
