"""Bounded authenticated snapshot download; R2 credentials never reach this worker."""
from contextlib import contextmanager
import hashlib
import tempfile
import json
import logging
import re
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from app.services.lost_gallery import read_manifest, GalleryBuildCancelled

MAX_FEED_BYTES = 128 * 1024 * 1024
MAX_PHOTO_BYTES = 10 * 1024 * 1024


def atomic_json(path, value):
    path = Path(path)
    import os
    import tempfile
    descriptor, name = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as target:
            json.dump(value, target, ensure_ascii=False)
            target.flush()
            os.fsync(target.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_bounded(response, limit, stopped, deadline=30):
    end = time.monotonic() + deadline
    payload = bytearray()
    for part in response.iter_bytes(chunk_size=65536):
        if stopped():
            raise GalleryBuildCancelled('Snapshot download stopped')
        if time.monotonic() > end or len(payload) + len(part) > limit:
            raise ValueError('Snapshot download exceeds its limit')
        payload.extend(part)
    return bytes(payload)


class GallerySource:
    def __init__(self, url, key, r2_host, state_dir, photo_root, client=None):
        address = urlsplit(url)
        if (address.username or address.password or address.fragment or address.query
                or not address.hostname
                or (address.scheme != 'https' and not
                    (address.scheme == 'http' and address.hostname in {'127.0.0.1', '::1'}))):
            raise ValueError('Gallery source needs HTTPS or a loopback tunnel')
        if not key or not re.fullmatch(r'[a-z0-9-]+\.r2\.cloudflarestorage\.com', r2_host):
            raise ValueError('Gallery source key and exact R2 host are required')
        # httpx INFO logs include full signed URLs; this dedicated worker must not emit them.
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        self.inventory = {}
        self.downloaded = 0
        self.urls_refreshed_at = 0
        self.url, self.key, self.r2_host = url, key, r2_host
        self.state_dir, self.photo_root = Path(state_dir), Path(photo_root)
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.photo_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.client = client or httpx.Client(timeout=httpx.Timeout(20, connect=5), follow_redirects=False,
                                             trust_env=False)

    def close(self):
        self.client.close()

    def fetch(self, etag, stopped):
        headers = {'X-Internal-Api-Key': self.key}
        if etag:
            headers['If-None-Match'] = etag
        # Signing the complete inventory is CPU work on the VM; measured 12k photos at ~31s cold.
        with self.client.stream('GET', self.url, headers=headers,
                                timeout=httpx.Timeout(90, connect=5)) as response:
            if response.status_code == 304:
                if not etag:
                    raise ValueError('Unexpected unchanged response')
                return None
            response.raise_for_status()
            payload = json.loads(read_bounded(response, MAX_FEED_BYTES, stopped))
            tag = response.headers.get('etag')
            if not tag or not re.fullmatch(r'"[a-f0-9]{64}"', tag):
                raise ValueError('A content fingerprint is required')
        if payload.get('complete') is not True or payload.get('count') != len(payload.get('records', [])):
            raise ValueError('Incomplete source snapshot')
        photos = payload.get('photos', [])
        if not isinstance(photos, list) or len(photos) > 100_000:
            raise ValueError('Invalid source photo inventory')
        manifest = {'complete': True, 'records': payload['records']}
        # Private attempt file may remain after failure. It contains no signed URLs.
        manifest_path = self.state_dir / 'incoming.json'
        atomic_json(manifest_path, manifest)
        records, digest = read_manifest(manifest_path)
        inventory = {}
        for photo in photos:
            sha = photo.get('sha256', '')
            address = urlsplit(photo.get('url', ''))
            if (not re.fullmatch(r'[a-f0-9]{64}', sha) or sha in inventory
                    or type(photo.get('bytes')) is not int or not 0 < photo['bytes'] <= MAX_PHOTO_BYTES
                    or photo.get('mime') not in {'image/jpeg', 'image/png', 'image/webp'}
                    or address.scheme != 'https' or address.hostname != self.r2_host
                    or address.port not in {None, 443} or address.username or address.password or address.fragment
                    or not re.fullmatch(r'/pawbridge-animal-originals/apms/photos/[a-f0-9]{64}\.(jpg|png|webp)', address.path)):
                raise ValueError('Invalid photo origin or integrity metadata')
            inventory[sha] = photo
        if set(inventory) != {row['source_sha256'] for row in records}:
            raise ValueError('Photo inventory differs from snapshot')
        # Only metadata is fetched here. The builder first checks existing ES vectors.
        self.inventory = inventory
        self.urls_refreshed_at = time.monotonic()
        self.downloaded = 0
        return {'etag': tag, 'manifest_path': manifest_path, 'snapshot_sha256': digest}

    def renew_urls(self, stopped):
        # A long inference run can outlive the one-hour presigned URLs. Refresh
        # only addresses for immutable photos in this build, never its records/ETag.
        with tempfile.TemporaryDirectory(prefix='url-refresh-', dir=self.state_dir) as directory:
            refreshed = GallerySource(self.url, self.key, self.r2_host, directory,
                                      self.photo_root, client=self.client)
            refreshed.fetch(None, stopped)
            for sha, old in self.inventory.items():
                new = refreshed.inventory.get(sha)
                if new is not None and (new['bytes'], new['mime']) == (old['bytes'], old['mime']):
                    self.inventory[sha] = new
        self.urls_refreshed_at = time.monotonic()

    @contextmanager
    def photo(self, row, stopped):
        """Fetch one image only for inference, then remove it even when inference fails."""
        if stopped():
            raise GalleryBuildCancelled('Snapshot download stopped')
        if time.monotonic() - self.urls_refreshed_at >= 45 * 60:
            self.renew_urls(stopped)
        sha = row['source_sha256']
        photo = self.inventory.get(sha)
        if photo is None:
            raise ValueError('Photo is not in the verified snapshot')
        # No persistent photo cache; signed URLs and the feed key never reach disk.
        with tempfile.TemporaryDirectory(prefix='inference-', dir=self.photo_root) as directory:
            path = Path(directory) / (sha + '.image')
            for attempt in range(2):
                with self.client.stream('GET', photo['url']) as response:
                    if (response.status_code == 403 and attempt == 0
                            and time.monotonic() - self.urls_refreshed_at >= 60):
                        renew = True
                    else:
                        response.raise_for_status()
                        image = read_bounded(response, photo['bytes'], stopped)
                        mime = response.headers.get('content-type', '').split(';')[0].strip()
                        renew = False
                if not renew:
                    break
                self.renew_urls(stopped)
                photo = self.inventory[sha]
            if (len(image) != photo['bytes'] or hashlib.sha256(image).hexdigest() != sha
                    or mime != photo['mime']):
                raise ValueError('Photo integrity check failed')
            with path.open('xb') as target:
                path.chmod(0o600)
                target.write(image)
            del image
            self.downloaded += 1
            if stopped():
                raise GalleryBuildCancelled('Snapshot download stopped')
            yield path
