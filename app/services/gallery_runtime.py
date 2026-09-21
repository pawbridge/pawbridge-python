"""One restartable refresh thread shares the API encoder; failed cycles keep the alias."""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import fcntl
import logging
import threading

from app.services.gallery_source import atomic_json
from app.services.lost_gallery import build_gallery, GalleryBuildCancelled, gallery_mapping
from app.services.gallery_retention import GalleryRetention


@contextmanager
def runtime_owner(state_dir):
    state = Path(state_dir)
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (state / 'gpu-runtime.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


class BackgroundEncoder:
    def __init__(self, encoder):
        self.encoder = encoder
        self.model_version = encoder.model_version

    def encode_with_metadata(self, image, species, *, prepared_focus_image=None):
        return self.encoder.encode_with_metadata(image, species, background=True,
                                                 prepared_focus_image=prepared_focus_image)

    def describe_coat_color(self, image, species, *, prepared_focus_image=None):
        return self.encoder.describe_coat_color(image, species, background=True,
                                                prepared_focus_image=prepared_focus_image)


class GalleryRefresh:
    def __init__(self, es, encoder, source, alias, state_dir, interval=900, ready=True, store=None):
        if not 30 <= interval <= 86400:
            raise ValueError('Gallery interval must be between 30 seconds and one day')
        self.es, self.encoder, self.source, self.alias = es, BackgroundEncoder(encoder), source, alias
        self.state_dir, self.interval = Path(state_dir), interval
        self.store = store
        self.stop = threading.Event()
        self.ready = ready
        self.status = {'state': 'waiting', 'lastSuccess': None}
        self.retention = GalleryRetention(es, self.state_dir, alias, store=store)
        self.last_result = None
        self.etag = None  # Fetch afresh after every restart; never trust a stale local success flag.
        self.thread = threading.Thread(target=self.run, name='gallery-refresh', daemon=False)

    def published(self):
        if not self.last_result:
            return False
        if self.store is not None:
            try:
                return self.store.published(self.last_result)
            except Exception:
                return False
        client = self.es.options(request_timeout=10, max_retries=0)
        try:
            index = self.last_result['index']
            return (set(client.indices.get_alias(name=self.alias)) == {index}
                    and client.indices.get_mapping(index=index)[index]['mappings'].get('_meta')
                    == gallery_mapping(self.last_result['snapshot_sha256'])['_meta']
                    and client.count(index=self.alias)['count'] == self.last_result['records'])
        except Exception:
            return False

    def save_status(self, **changes):
        self.status = {**self.status, **changes}
        try:
            atomic_json(self.state_dir / 'refresh-status.json', self.status)
        except OSError:
            logging.getLogger(__name__).warning('Cannot persist gallery refresh status')

    def cycle(self):
        self.save_status(state='fetching', errorType=None)
        snapshot = self.source.fetch(self.etag, self.stop.is_set)
        if snapshot is None:
            if self.published():
                self.save_status(state='idle', lastChecked=datetime.now(timezone.utc).isoformat())
                return
            self.ready = False
            self.etag = None
            snapshot = self.source.fetch(None, self.stop.is_set)
        if snapshot is None:
            raise ValueError('Source did not provide a complete snapshot')
        self.retention.prepare(snapshot['snapshot_sha256'])
        self.save_status(state='building')
        arguments = {"store": self.store} if self.store is not None else {}
        if snapshot.get('paged') is True:
            arguments['stream'] = snapshot['stream']
        result = build_gallery(self.es, lambda: self.encoder, snapshot.get('manifest_path'),
                               self.source.photo_root, self.alias, self.state_dir,
                               progress=lambda row: self.save_status(state='building', progress=row),
                               cancelled=self.stop.is_set,
                               photo_provider=lambda row: self.source.photo(row, self.stop.is_set), **arguments)
        self.retention.published(result['index'])
        self.last_result = result
        self.ready = True
        # Mark success only after verified atomic publication. Do not persist the signed URLs/key.
        self.save_status(state='idle', lastSuccess=datetime.now(timezone.utc).isoformat(),
                         result=result, photos={'downloaded': self.source.downloaded, 'retained': 0}, progress=None)
        self.etag = snapshot['etag']
        if snapshot.get('paged') is True:
            # Publication has already succeeded; a cleanup failure is retried before another snapshot.
            self.source.finish(self.stop.is_set)

    def run(self):
        failures = 0
        while not self.stop.is_set():
            try:
                self.cycle()
                failures = 0
                delay = self.interval
            except GalleryBuildCancelled:
                break
            except Exception as error:
                failures += 1
                # URLs may contain signatures. Never record exception strings or HTTP tracebacks.
                logging.getLogger(__name__).warning('Gallery refresh failed: %s', type(error).__name__)
                self.save_status(state='failed', errorType=type(error).__name__)
                delay = min(self.interval, 30 * 2 ** min(failures - 1, 6))
            if self.stop.wait(delay):
                break
        self.save_status(state='stopped')

    def start(self):
        self.thread.start()

    def close(self):
        self.stop.set()
        self.thread.join()  # Never release GPU ownership while a CUDA worker is still running.
        self.source.close()
