"""Overlap one upcoming photo download with inference, without a photo cache."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager


class PhotoPrefetch:
    def __init__(self, rows, provider):
        self.rows = iter(rows)
        self.provider = provider
        self.executor = None
        self.future = None
        self.pending_id = None

    def _open(self, row):
        context = self.provider(row)
        return context, context.__enter__()

    def _submit_next(self):
        row = next(self.rows, None)
        if row is not None:
            self.pending_id = row['id']
            self.future = self.executor.submit(self._open, row)

    def __enter__(self):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='gallery-photo')
        try:
            self._submit_next()
        except BaseException:
            self.executor.shutdown(wait=True, cancel_futures=True)
            raise
        return self

    @contextmanager
    def photo(self, row):
        if self.future is None or self.pending_id != row['id']:
            raise RuntimeError('Prefetched photo order differs from gallery')
        future, self.future = self.future, None
        context, path = future.result()
        with ExitStack() as current:
            current.push(context)
            # Only current + next photo can exist; no queue of the entire batch.
            self._submit_next()
            yield path

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if self.future is not None and not self.future.cancel():
                try:
                    context, _ = self.future.result()
                    context.__exit__(None, None, None)
                except BaseException:
                    if exc_type is None:
                        raise
        finally:
            self.future = None
            self.executor.shutdown(wait=True, cancel_futures=True)
        return False
