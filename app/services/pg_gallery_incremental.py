"""Stage changed rows, then atomically refresh one exclusively owned gallery."""
import hashlib
import json
import re
from copy import deepcopy

from app.services.lost_gallery import PREFIX, gallery_mapping


class IncrementalGallerySession:
    def __init__(self, base, alias, build_key, metadata, expected_count, snapshot_hash, total):
        base.names(alias, build_key)
        if (not isinstance(metadata, dict)
                or not isinstance(snapshot_hash, str)
                or not re.fullmatch(r'[a-f0-9]{64}', snapshot_hash)
                or type(expected_count) is not int or not 1 <= expected_count <= 100_000
                or type(total) is not int or not 1 <= total <= 100_000):
            raise ValueError('Invalid incremental gallery contract')
        previous_hash = metadata.get('snapshot_sha256')
        if (not isinstance(previous_hash, str)
                or not re.fullmatch(r'[a-f0-9]{64}', previous_hash)
                or metadata != gallery_mapping(previous_hash)['_meta']):
            raise ValueError('Incremental gallery requires current feature versions')
        self.base = base
        self.alias = alias
        self.build_key = build_key
        self.metadata = deepcopy(metadata)
        self.expected_count = expected_count
        self.snapshot_hash = snapshot_hash
        self.total = total
        self.new_metadata = gallery_mapping(snapshot_hash)['_meta']
        self.delta_target = PREFIX + 'build-' + hashlib.sha256(
            ('incremental:' + build_key + previous_hash + snapshot_hash).encode()).hexdigest()[:24]
        if self.delta_target == build_key:
            raise ValueError('Incremental staging identity collides with its base')
        self.delta_metadata = {**self.new_metadata, 'incremental_base': build_key,
                               'incremental_base_snapshot': previous_hash}
        self.already_current = False
        self.resume_required = True
        self.publication_stats = {}
        self._target = None
        self._seen_ids = set()
        self._page_ids = None
        self._active_page = {}
        self._delta_page = {}

    def _identity(self, alias, target, total=None, old_index=None):
        self.base.names(alias, target)
        if (alias != self.alias or (self._target is not None and target != self._target)
                or target != self.delta_target
                or (total is not None and (type(total) is not int or total != self.total))
                or (old_index is not None and old_index != self.build_key)):
            raise ValueError('Incremental gallery identity changed')

    def _ready(self, target):
        self._identity(self.alias, target)
        if self._target is None:
            raise RuntimeError('Incremental gallery session has not begun')

    @staticmethod
    def _ids(ids):
        if not 1 <= len(ids) <= 100:
            raise ValueError('Gallery reads must be bounded to 100 animals')
        parsed = []
        for value in ids:
            if type(value) is str and re.fullmatch(r'[1-9][0-9]{0,18}', value):
                value = int(value)
            if type(value) is not int or not 0 < value <= 9223372036854775807:
                raise ValueError('Invalid gallery animal identity')
            parsed.append(value)
        if len(set(parsed)) != len(parsed):
            raise ValueError('Duplicate gallery animal identity')
        return parsed

    def _base_contract(self, conn, *, locked=False):
        suffix = ' FOR UPDATE' if locked else ''
        head = conn.execute('SELECT build_key FROM lost_gallery_heads WHERE alias=%s' + suffix,
                            (self.alias,)).fetchone()
        if head != (self.build_key,):
            raise RuntimeError('Gallery head changed during the incremental build')
        build = conn.execute('SELECT metadata,completed,expected_count FROM lost_gallery_builds '
                             'WHERE build_key=%s' + suffix, (self.build_key,)).fetchone()
        if build != (self.metadata, True, self.expected_count):
            raise RuntimeError('Active gallery changed during the incremental build')
        if conn.execute('SELECT 1 FROM lost_gallery_heads WHERE build_key=%s AND alias<>%s LIMIT 1',
                        (self.build_key, self.alias)).fetchone():
            raise RuntimeError('Incremental gallery must have an exclusive alias')
        # Compare all version fields, including future ones, rather than selected versions.
        if ({k: v for k, v in build[0].items() if k != 'snapshot_sha256'} !=
                {k: v for k, v in self.new_metadata.items() if k != 'snapshot_sha256'}):
            raise RuntimeError('Gallery feature versions require a full rebuild')
        count = conn.execute('SELECT count(*) FROM lost_gallery_documents WHERE build_key=%s',
                             (self.build_key,)).fetchone()[0]
        if count != self.expected_count:
            raise RuntimeError('Active gallery count differs from its complete snapshot')
        return count

    def begin(self, alias, target, snapshot_hash, total):
        self._identity(alias, target, total)
        if self._target is not None or snapshot_hash != self.snapshot_hash:
            raise ValueError('Incremental gallery session cannot begin twice or change snapshot')
        with self.base.connection() as conn:
            count = self._base_contract(conn)
            already_current = self.metadata == self.new_metadata and count == self.total
            if not already_current:
                conn.execute('INSERT INTO lost_gallery_builds(build_key,metadata,expected_count) '
                             'VALUES (%s,%s::jsonb,%s) ON CONFLICT DO NOTHING',
                             (self.delta_target, json.dumps(self.delta_metadata), total))
                stored = conn.execute('SELECT metadata,completed,expected_count FROM lost_gallery_builds '
                                      'WHERE build_key=%s', (self.delta_target,)).fetchone()
                if stored != (self.delta_metadata, False, total):
                    raise RuntimeError('Existing incremental staging gallery has a different contract')
                if conn.execute('SELECT 1 FROM lost_gallery_heads WHERE build_key=%s LIMIT 1',
                                (self.delta_target,)).fetchone():
                    raise RuntimeError('Incremental staging gallery is already referenced')
        self._target = target
        self.already_current = already_current
        self.resume_required = not already_current
        return self.build_key

    def completed(self, target, total):
        self._ready(target)
        self._identity(self.alias, target, total)
        return self.already_current

    def count(self, target):
        self._ready(target)
        return 0 if self.already_current else self.base.count(self.delta_target)

    def documents(self, old_index, target, ids):
        self._ready(target)
        self._identity(self.alias, target, old_index=old_index)
        if old_index != self.build_key or self._page_ids is not None:
            raise ValueError('Incremental gallery requires one acknowledged page at a time')
        ids = self._ids(ids)
        active, delta = {}, {}
        with self.base.connection(read_only=True) as conn:
            rows = conn.execute('SELECT build_key,animal_id,document,image_vector::text,animal_vector::text '
                                'FROM lost_gallery_documents WHERE build_key=ANY(%s) AND animal_id=ANY(%s)',
                                ([self.build_key, self.delta_target], ids)).fetchall()
            for key, animal_id, metadata, vector, animal in rows:
                document = dict(metadata)
                if document.get('id') != animal_id:
                    raise RuntimeError('Gallery row and document identities differ')
                document['image_vector'] = json.loads(vector)
                if animal is not None:
                    document['animal_vector'] = json.loads(animal)
                (active if key == self.build_key else delta)[animal_id] = document
        self._page_ids = set(ids)
        self._active_page, self._delta_page = active, delta
        return {**active, **delta}

    def write_page(self, target, documents):
        self._ready(target)
        if self.already_current or not 1 <= len(documents) <= 100:
            raise ValueError('Invalid incremental gallery write')
        if any(not isinstance(document, dict) for document in documents):
            raise ValueError('Invalid gallery document')
        ids = self._ids([document.get('id') for document in documents])
        ids = set(ids)
        if (ids != self._page_ids or ids & self._seen_ids
                or len(self._seen_ids) + len(ids) > self.total):
            raise ValueError('Incremental gallery source has duplicate or unexpected records')
        changed = [document for document in documents
                   if document != self._active_page.get(document['id'])]
        stale = [document['id'] for document in documents
                 if document == self._active_page.get(document['id'])
                 and document['id'] in self._delta_page]
        if changed:
            self.base.write_page(self.delta_target, changed)
        if stale:
            # A replay that now matches active content must not retain an older delta.
            with self.base.connection() as conn:
                build = conn.execute('SELECT metadata,completed,expected_count FROM lost_gallery_builds '
                                     'WHERE build_key=%s FOR UPDATE', (self.delta_target,)).fetchone()
                if build != (self.delta_metadata, False, self.total):
                    raise RuntimeError('Incremental staging gallery changed during the build')
                conn.execute('DELETE FROM lost_gallery_documents WHERE build_key=%s AND animal_id=ANY(%s)',
                             (self.delta_target, stale))
        self._seen_ids.update(ids)
        self._page_ids = None
        self._active_page = self._delta_page = {}

    def result_index(self, target):
        self._ready(target)
        return self.build_key

    def publish(self, alias, target, old_index, total, check_cancelled):
        self._ready(target)
        self._identity(alias, target, total, old_index)
        if old_index != self.build_key:
            raise ValueError('Incremental gallery base identity changed')
        if not self.already_current and (len(self._seen_ids) != total or self._page_ids is not None):
            raise RuntimeError('Incremental gallery requires every complete source record')
        seen = sorted(self._seen_ids)
        stats = None
        with self.base.connection() as conn:
            # EXCLUSIVE also blocks legacy FOR UPDATE readers before locking the
            # head row; SHARE ROW EXCLUSIVE could deadlock their later UPDATE.
            # Ordinary gallery searches retain their ACCESS SHARE table access.
            conn.execute('LOCK TABLE lost_gallery_heads IN EXCLUSIVE MODE')
            self._base_contract(conn, locked=True)
            if self.already_current:
                if self.metadata != self.new_metadata or self.expected_count != total:
                    raise RuntimeError('Active gallery does not match the requested snapshot')
                check_cancelled()
                stats = {'mode': 'incremental', 'staged': 0, 'inserted': 0,
                         'updated': 0, 'removed': 0, 'unchanged': total}
            else:
                delta = conn.execute('SELECT metadata,completed,expected_count FROM lost_gallery_builds '
                                     'WHERE build_key=%s FOR UPDATE', (self.delta_target,)).fetchone()
                if delta != (self.delta_metadata, False, total):
                    raise RuntimeError('Incremental staging gallery changed during the build')
                if conn.execute('SELECT 1 FROM lost_gallery_heads WHERE build_key=%s LIMIT 1',
                                (self.delta_target,)).fetchone():
                    raise RuntimeError('Incremental staging gallery is already referenced')
                outside_source = conn.execute(
                    'SELECT 1 FROM lost_gallery_documents d WHERE d.build_key=%s AND NOT EXISTS '
                    '(SELECT 1 FROM unnest(%s::bigint[]) AS seen(animal_id) '
                    'WHERE seen.animal_id=d.animal_id) LIMIT 1', (self.delta_target, seen)).fetchone()
                if outside_source:
                    raise RuntimeError('Incremental staging contains records outside the complete source')
                staged, inserted = conn.execute(
                    'SELECT count(*),count(*) FILTER (WHERE a.animal_id IS NULL) '
                    'FROM lost_gallery_documents d LEFT JOIN lost_gallery_documents a '
                    'ON a.build_key=%s AND a.animal_id=d.animal_id WHERE d.build_key=%s',
                    (self.build_key, self.delta_target)).fetchone()
                removed = conn.execute(
                    'SELECT count(*) FROM lost_gallery_documents d WHERE d.build_key=%s AND NOT EXISTS '
                    '(SELECT 1 FROM unnest(%s::bigint[]) AS seen(animal_id) '
                    'WHERE seen.animal_id=d.animal_id)', (self.build_key, seen)).fetchone()[0]
                check_cancelled()
                conn.execute('INSERT INTO lost_gallery_documents '
                    '(build_key,animal_id,species,status,model_version,image_vector,animal_vector,document) '
                    'SELECT %s,animal_id,species,status,model_version,image_vector,animal_vector,document '
                    'FROM lost_gallery_documents WHERE build_key=%s '
                    'ON CONFLICT(build_key,animal_id) DO UPDATE SET species=excluded.species,'
                    'status=excluded.status,model_version=excluded.model_version,'
                    'image_vector=excluded.image_vector,animal_vector=excluded.animal_vector,'
                    'document=excluded.document', (self.build_key, self.delta_target))
                # Hash anti-join avoids probing a 100k-element array once per row.
                conn.execute('DELETE FROM lost_gallery_documents d WHERE d.build_key=%s AND NOT EXISTS '
                    '(SELECT 1 FROM unnest(%s::bigint[]) AS seen(animal_id) '
                    'WHERE seen.animal_id=d.animal_id)', (self.build_key, seen))
                count = conn.execute('SELECT count(*) FROM lost_gallery_documents WHERE build_key=%s',
                                     (self.build_key,)).fetchone()[0]
                if count != total:
                    raise RuntimeError('Incremental gallery count differs from the complete snapshot')
                conn.execute('UPDATE lost_gallery_builds SET metadata=%s::jsonb,expected_count=%s '
                             'WHERE build_key=%s', (json.dumps(self.new_metadata), total, self.build_key))
                conn.execute('DELETE FROM lost_gallery_builds WHERE build_key=%s', (self.delta_target,))
                conn.execute('UPDATE lost_gallery_heads SET updated_at=now() WHERE alias=%s', (alias,))
                stats = {'mode': 'incremental', 'staged': staged, 'inserted': inserted,
                         'updated': staged - inserted, 'removed': removed, 'unchanged': total - staged}
                check_cancelled()
        # Expose success only after the single transaction has committed.
        self.publication_stats = stats
        return total
