"""Pull one bounded page at a time; persist only acknowledged cursor/hash state."""
import hashlib
import json
import re
import struct
import time
import uuid
from urllib.parse import urlsplit

import httpx
from app.services.gallery_source import GallerySource, atomic_json, read_bounded, MAX_PHOTO_BYTES
from app.services.lost_gallery import GalleryBuildCancelled, METADATA

PROTOCOL = 'pawbridge-gallery-pages-v2'
PAGE_BYTES = 2 * 1024 * 1024
PAGE_SIZE = 100
INITIAL_CHAIN = hashlib.sha256((PROTOCOL + '\n').encode()).hexdigest()


def advance(chain, row, photo):
    values = [row['id'], row['species'], row['source_sha256']]
    values += [row.get(field) for field in METADATA]
    values += [photo['key'], photo['bytes'], photo['mime']]
    digest = hashlib.sha256()
    for value in values:
        data = None if value is None else str(value).encode('utf-8')
        digest.update(struct.pack('>i', -1 if data is None else len(data)))
        if data is not None:
            digest.update(data)
    return hashlib.sha256(bytes.fromhex(chain) + digest.digest()).hexdigest()


class PagedGallerySource(GallerySource):
    def __init__(self, url, key, r2_host, state_dir, photo_root, client=None):
        super().__init__(url.rstrip('/') + '/snapshots', key, r2_host, state_dir, photo_root, client)
        self.checkpoint_path = self.state_dir / 'paged-gallery-checkpoint.json'
        self.creation_path = self.state_dir / 'paged-gallery-creation.json'
        self.checkpoint = None
        self.pending = None
        self.current_page = None
        self.current_cursor = None

    def request(self, method, suffix, stopped, *, params=None, etag=None, limit=PAGE_BYTES, request_id=None):
        for attempt in range(3):
            if stopped():
                raise GalleryBuildCancelled('Gallery page request stopped')
            delay = attempt + 1
            try:
                headers = {'X-Internal-Api-Key': self.key}
                if request_id:
                    headers['X-Gallery-Request-Id'] = request_id
                if etag:
                    headers['If-None-Match'] = etag
                with self.client.stream(method, self.url + suffix, headers=headers, params=params,
                                        timeout=httpx.Timeout(130 if method == 'POST' else 30, connect=5)) as response:
                    if response.status_code in (204, 304):
                        return None
                    if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                        try: delay = max(1, min(30, int(response.headers.get('Retry-After', delay))))
                        except ValueError: pass
                    else:
                        response.raise_for_status()
                        return json.loads(read_bounded(response, limit, stopped))
            except httpx.TransportError:
                if attempt == 2:
                    raise
            end = time.monotonic() + delay
            while time.monotonic() < end:
                if stopped():
                    raise GalleryBuildCancelled('Gallery page retry stopped')
                time.sleep(min(0.1, max(0, end-time.monotonic())))
        raise RuntimeError('Gallery page retry exhausted')

    def descriptor(self, value):
        if (not isinstance(value, dict) or value.get('protocol') != PROTOCOL
                or not re.fullmatch(r'[a-f0-9]{32}', value.get('snapshotId', ''))
                or not re.fullmatch(r'[a-f0-9]{64}', value.get('snapshotSha256', ''))
                or type(value.get('count')) is not int or not 1 <= value['count'] <= 100_000
                or not isinstance(value.get('cursor'), str) or not 1 <= len(value['cursor']) <= 256
                or type(value.get('expiresAt')) is not int):
            raise ValueError('Invalid paged snapshot descriptor')
        return value

    def load_checkpoint(self):
        if not self.checkpoint_path.exists():
            return None
        if self.checkpoint_path.is_symlink() or self.checkpoint_path.stat().st_size > 16384:
            raise ValueError('Invalid gallery checkpoint file')
        value = json.loads(self.checkpoint_path.read_text())
        descriptor = self.descriptor(value['descriptor'])
        if (type(value.get('processed')) is not int or not 0 <= value['processed'] <= descriptor['count']
                or type(value.get('last_id')) is not int or value['last_id'] < 0
                or (value['processed'] > 0 and value['last_id'] == 0)
                or (value['processed'] == 0 and (value['last_id'] != 0 or value.get('chain') != INITIAL_CHAIN))
                or not re.fullmatch(r'[a-f0-9]{64}', value.get('chain', ''))
                or type(value.get('complete')) is not bool
                or (value['processed'] < descriptor['count'] and
                    (not isinstance(value.get('cursor'), str) or not 1 <= len(value['cursor']) <= 256))):
            raise ValueError('Invalid gallery checkpoint')
        return value

    def fetch(self, etag, stopped):
        saved = self.load_checkpoint()
        if saved:
            self.checkpoint = saved
            if saved['complete']:
                self.finish(stopped)
            else:
                try:
                    descriptor = self.descriptor(self.request('GET', '/' + saved['descriptor']['snapshotId'], stopped, limit=16384))
                    if any(descriptor[k] != saved['descriptor'][k] for k in ('protocol', 'snapshotId', 'snapshotSha256', 'count', 'cursor')):
                        raise ValueError('Resumed snapshot descriptor changed')
                    self.checkpoint['descriptor'] = descriptor
                    self.creation_path.unlink(missing_ok=True)
                    return self.snapshot()
                except httpx.HTTPStatusError as error:
                    if error.response.status_code != 410:
                        raise
                    self.checkpoint_path.unlink(missing_ok=True)
                    self.checkpoint = None
        # Reuse a persisted idempotency key if the POST succeeded but its response was lost.
        if self.creation_path.exists():
            if self.creation_path.is_symlink() or self.creation_path.stat().st_size > 1024:
                raise ValueError('Invalid snapshot creation file')
            request_id = json.loads(self.creation_path.read_text()).get('requestId')
            if not isinstance(request_id, str) or not re.fullmatch(r'[a-f0-9]{32}', request_id):
                raise ValueError('Invalid snapshot creation ID')
        else:
            request_id = uuid.uuid4().hex
            atomic_json(self.creation_path, {'requestId': request_id})
        descriptor = self.request('POST', '', stopped, etag=etag, limit=16384, request_id=request_id)
        if descriptor is None:
            self.creation_path.unlink(missing_ok=True)
            if not etag:
                raise ValueError('Unexpected unchanged snapshot')
            return None
        self.checkpoint = {'descriptor': self.descriptor(descriptor)}
        self.reset_resume()
        self.creation_path.unlink(missing_ok=True)
        return self.snapshot()

    def snapshot(self):
        self.downloaded = 0
        self.inventory = {}
        self.pending = self.current_page = self.current_cursor = None
        descriptor = self.checkpoint['descriptor']
        return {'paged': True, 'stream': self, 'snapshot_sha256': descriptor['snapshotSha256'],
                'etag': '"' + descriptor['snapshotSha256'] + '"'}

    @property
    def total(self):
        return self.checkpoint['descriptor']['count']

    @property
    def processed(self):
        return self.checkpoint['processed']

    def reset_resume(self):
        descriptor = self.checkpoint['descriptor']
        self.checkpoint = {'descriptor': descriptor, 'processed': 0, 'last_id': 0,
                           'chain': INITIAL_CHAIN, 'cursor': descriptor['cursor'], 'complete': False}
        self.pending = None
        atomic_json(self.checkpoint_path, self.checkpoint)

    def validate_item(self, item):
        if not isinstance(item, dict):
            raise ValueError('Invalid gallery page item')
        row, photo = item.get('record'), item.get('photo')
        if not isinstance(row, dict) or not isinstance(photo, dict):
            raise ValueError('Missing gallery record or photo')
        if (type(row.get('id')) is not int or row['id'] <= 0 or row.get('species') not in ('DOG', 'CAT')
                or not re.fullmatch(r'[a-f0-9]{64}', row.get('source_sha256', ''))
                or row['source_sha256'] != photo.get('sha256')
                or type(photo.get('bytes')) is not int or not 0 < photo['bytes'] <= MAX_PHOTO_BYTES
                or photo.get('mime') not in ('image/jpeg', 'image/png', 'image/webp')
                or not re.fullmatch(r'apms/photos/[a-f0-9]{64}\.(jpg|png|webp)', photo.get('key', ''))):
            raise ValueError('Invalid gallery record or photo integrity metadata')
        for field in METADATA:
            value = row.get(field)
            if value is not None and (not isinstance(value, str) or len(value) > 10000):
                raise ValueError('Invalid gallery metadata')
        address = urlsplit(photo.get('url', ''))
        if (address.scheme != 'https' or address.hostname != self.r2_host or address.port not in (None, 443)
                or address.username or address.password or address.fragment
                or address.path != '/pawbridge-animal-originals/' + photo['key']):
            raise ValueError('Invalid gallery photo origin')
        return row, photo

    def get_page(self, cursor, stopped):
        descriptor = self.checkpoint['descriptor']
        return self.request('GET', '/' + descriptor['snapshotId'] + '/pages', stopped,
                            params={'cursor': cursor, 'limit': PAGE_SIZE})

    def pages(self, stopped):
        while self.processed < self.total:
            if self.pending is not None:
                raise RuntimeError('Previous page must be acknowledged after its ES write')
            descriptor = self.checkpoint['descriptor']
            cursor = self.checkpoint['cursor']
            page = self.get_page(cursor, stopped)
            if (not isinstance(page, dict) or page.get('protocol') != PROTOCOL
                    or page.get('snapshotId') != descriptor['snapshotId']
                    or page.get('snapshotSha256') != descriptor['snapshotSha256']
                    or type(page.get('total')) is not int or page['total'] != self.total
                    or type(page.get('start')) is not int or page['start'] != self.processed
                    or not isinstance(page.get('items'), list) or not 1 <= len(page['items']) <= PAGE_SIZE):
                raise ValueError('Gallery page identity, order or size differs')
            next_count = self.processed + len(page['items'])
            done = next_count == self.total
            if (next_count > self.total or type(page.get('complete')) is not bool or page['complete'] != done
                    or (done and page.get('nextCursor') is not None)
                    or (not done and (not isinstance(page.get('nextCursor'), str)
                        or not 1 <= len(page['nextCursor']) <= 256 or page['nextCursor'] == cursor))):
                raise ValueError('Invalid terminal page or cursor')
            rows, inventory = [], {}
            last_id, chain = self.checkpoint['last_id'], self.checkpoint['chain']
            for item in page['items']:
                row, photo = self.validate_item(item)
                if row['id'] <= last_id:
                    raise ValueError('Duplicate or out-of-order gallery ID')
                if photo['sha256'] in inventory and any(photo[k] != inventory[photo['sha256']][k] for k in ('bytes', 'mime')):
                    raise ValueError('Conflicting photo metadata within page')
                last_id, chain = row['id'], advance(chain, row, photo)
                rows.append(row); inventory[photo['sha256']] = photo
            if done and chain != descriptor['snapshotSha256']:
                raise ValueError('Complete snapshot fingerprint differs')
            self.inventory, self.current_page, self.current_cursor = inventory, page, cursor
            self.urls_refreshed_at = time.monotonic()
            self.pending = {'descriptor': descriptor, 'processed': next_count, 'last_id': last_id,
                            'chain': chain, 'cursor': page['nextCursor'], 'complete': False}
            yield rows
            if self.pending is not None:
                raise RuntimeError('Page was not acknowledged after ES write')
            self.inventory = {}
            self.current_page = self.current_cursor = None
            del page, rows, inventory
        self.verify_complete()

    def acknowledge(self, processed):
        if self.pending is None or processed != self.pending['processed']:
            raise RuntimeError('Page acknowledgment does not match written records')
        atomic_json(self.checkpoint_path, self.pending)
        self.checkpoint, self.pending = self.pending, None

    def verify_complete(self):
        if self.processed != self.total or self.checkpoint['chain'] != self.checkpoint['descriptor']['snapshotSha256']:
            raise ValueError('Snapshot has not been completely verified')

    def renew_urls(self, stopped):
        if self.current_page is None:
            raise RuntimeError('Photo renewal requires the current bounded page')
        page = self.get_page(self.current_cursor, stopped)
        expected = self.current_page
        if not isinstance(page, dict) or any(page.get(k) != expected.get(k) for k in
                ('protocol', 'snapshotId', 'snapshotSha256', 'total', 'start', 'nextCursor', 'complete')):
            raise ValueError('Renewed photo page identity differs')
        if len(page.get('items', [])) != len(expected['items']):
            raise ValueError('Renewed photo page size differs')
        replacement = {}
        for old, item in zip(expected['items'], page['items']):
            row, photo = self.validate_item(item)
            if row != old['record'] or any(photo[k] != old['photo'][k] for k in ('sha256', 'key', 'bytes', 'mime')):
                raise ValueError('Renewed photo page content differs')
            replacement[photo['sha256']] = photo
        self.inventory = replacement
        self.urls_refreshed_at = time.monotonic()

    def finish(self, stopped):
        self.checkpoint['complete'] = True
        atomic_json(self.checkpoint_path, self.checkpoint)
        try:
            self.request('DELETE', '/' + self.checkpoint['descriptor']['snapshotId'], stopped, limit=16384)
        except httpx.HTTPStatusError as error:
            if error.response.status_code != 410:
                raise
        self.checkpoint_path.unlink(missing_ok=True)
        self.checkpoint = None
        self.inventory = {}
