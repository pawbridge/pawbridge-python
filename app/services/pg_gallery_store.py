"""Bounded PostgreSQL gallery I/O. Connections are scoped to each short DB operation."""
import json
import re
from contextlib import contextmanager
from app.services.dinov3 import validate_vector
from app.services.lost_gallery import PREFIX, CONTRACT, gallery_mapping


class GalleryUnavailable(RuntimeError):
    pass


class PostgresqlGalleryStore:
    def __init__(self, pool):
        self.pool = pool

    @contextmanager
    def connection(self, *, read_only=False):
        with self.pool.connection(timeout=3) as conn:
            with conn.transaction():
                if read_only:
                    conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
                conn.execute("SET LOCAL search_path TO pawbridge_animal, public")
                conn.execute("SET LOCAL statement_timeout = '10s'")
                conn.execute("SET LOCAL lock_timeout = '2s'")
                conn.execute("SET LOCAL idle_in_transaction_session_timeout = '15s'")
                yield conn

    @staticmethod
    def names(alias, target):
        if (not re.fullmatch(PREFIX + r'[a-z0-9][a-z0-9-]{0,60}', alias) or '-build-' in alias
                or not re.fullmatch(PREFIX + r'build-[a-f0-9]{24}', target)):
            raise ValueError('Invalid gallery identity')

    def begin(self, alias, target, snapshot_hash, total):
        self.names(alias, target)
        meta = gallery_mapping(snapshot_hash)['_meta']
        with self.connection() as conn:
            conn.execute('INSERT INTO lost_gallery_heads(alias) VALUES (%s) ON CONFLICT DO NOTHING', (alias,))
            old = conn.execute('SELECT build_key FROM lost_gallery_heads WHERE alias=%s', (alias,)).fetchone()[0]
            conn.execute('INSERT INTO lost_gallery_builds(build_key,metadata,expected_count) VALUES (%s,%s::jsonb,%s) ON CONFLICT DO NOTHING',
                         (target, json.dumps(meta), total))
            stored = conn.execute('SELECT metadata,expected_count FROM lost_gallery_builds WHERE build_key=%s', (target,)).fetchone()
            if stored != (meta, total):
                raise RuntimeError('Existing staging gallery has a different contract')
            if old:
                previous = conn.execute('SELECT metadata,completed FROM lost_gallery_builds WHERE build_key=%s', (old,)).fetchone()
                if not previous or previous[0].get('contract') != CONTRACT or not previous[1]:
                    raise RuntimeError('Refusing to replace an unowned or incomplete gallery')
            return old

    def completed(self, target, total):
        with self.connection() as conn:
            row = conn.execute('SELECT completed,expected_count FROM lost_gallery_builds WHERE build_key=%s', (target,)).fetchone()
            return row == (True, total)

    def count(self, target):
        with self.connection() as conn:
            return conn.execute('SELECT count(*) FROM lost_gallery_documents WHERE build_key=%s', (target,)).fetchone()[0]

    def documents(self, old_index, target, ids):
        if not 1 <= len(ids) <= 100:
            raise ValueError('Gallery reads must be bounded to 100 animals')
        cache = {}
        with self.connection() as conn:
            for source in dict.fromkeys(x for x in (old_index, target) if x):
                rows = conn.execute('SELECT document,image_vector::text,animal_vector::text FROM lost_gallery_documents '
                                    'WHERE build_key=%s AND animal_id=ANY(%s)', (source, [int(i) for i in ids])).fetchall()
                for document, vector, animal in rows:
                    document['image_vector'] = json.loads(vector)
                    if animal is not None:
                        document['animal_vector'] = json.loads(animal)
                    cache[document['id']] = document
        return cache

    def write_page(self, target, documents):
        if not 1 <= len(documents) <= 100:
            raise ValueError('Gallery writes must be bounded to 100 animals')
        from app.services.coat_color import valid as valid_color
        from app.services.lost_gallery import STATUSES
        rows = []
        ids = set()
        for document in documents:
            validate_vector(document['image_vector'])
            if document.get('animal_vector') is not None:
                validate_vector(document['animal_vector'])
            if (type(document.get('id')) is not int or document['id'] <= 0 or document['id'] in ids
                    or document.get('species') not in {'DOG', 'CAT'}
                    or document.get('status') not in STATUSES
                    or not re.fullmatch(r'[a-f0-9]{64}', document.get('source_sha256', ''))
                    or not isinstance(document.get('focus_status'), str)
                    or not document['focus_status']
                    or (document.get('coat_color') is not None and not valid_color(document['coat_color']))):
                raise ValueError('Invalid gallery document')
            ids.add(document['id'])
            metadata = {key: value for key, value in document.items() if key not in {'image_vector', 'animal_vector'}}
            if len(json.dumps(metadata)) > 100_000:
                raise ValueError('Gallery metadata is too large')
            rows.append((target, document['id'], document['species'], document.get('status'), document['model_version'],
                         json.dumps(document['image_vector']),
                         json.dumps(document['animal_vector']) if document.get('animal_vector') is not None else None,
                         json.dumps(metadata)))
        with self.connection() as conn:
            build = conn.execute('SELECT metadata,completed FROM lost_gallery_builds WHERE build_key=%s FOR UPDATE', (target,)).fetchone()
            if not build or build[1]:
                raise RuntimeError('Published galleries are immutable')
            if any(d['model_version'] != build[0]['model_version'] or
                   d.get('coat_color_version') != build[0]['coat_color_version'] for d in documents):
                raise ValueError('Gallery feature versions differ from the build')
            with conn.cursor() as cursor:
                cursor.executemany('INSERT INTO lost_gallery_documents '
                    '(build_key,animal_id,species,status,model_version,image_vector,animal_vector,document) '
                    'VALUES (%s,%s,%s,%s,%s,%s::public.vector,%s::public.vector,%s::jsonb) '
                    'ON CONFLICT(build_key,animal_id) DO UPDATE SET species=excluded.species,status=excluded.status,'
                    'model_version=excluded.model_version,image_vector=excluded.image_vector,'
                    'animal_vector=excluded.animal_vector,document=excluded.document', rows)
        # The connection context commits before the caller acknowledges its page cursor.

    def publish(self, alias, target, old_index, total, check_cancelled):
        self.names(alias, target)
        with self.connection() as conn:
            head = conn.execute('SELECT build_key FROM lost_gallery_heads WHERE alias=%s FOR UPDATE', (alias,)).fetchone()
            if head != (old_index,):
                raise RuntimeError('Gallery head changed during the build')
            build = conn.execute('SELECT expected_count FROM lost_gallery_builds WHERE build_key=%s FOR UPDATE', (target,)).fetchone()
            count = conn.execute('SELECT count(*) FROM lost_gallery_documents WHERE build_key=%s', (target,)).fetchone()[0]
            if build != (total,) or count != total:
                raise RuntimeError('Staging gallery count differs from the complete snapshot')
            check_cancelled()
            conn.execute('UPDATE lost_gallery_builds SET completed=true WHERE build_key=%s', (target,))
            conn.execute('UPDATE lost_gallery_heads SET build_key=%s,updated_at=now() WHERE alias=%s', (target, alias))
        return count

    def published(self, result):
        with self.connection() as conn:
            row = conn.execute('SELECT h.build_key,b.metadata,b.expected_count,b.completed,'
                '(SELECT count(*) FROM lost_gallery_documents d WHERE d.build_key=b.build_key) '
                'FROM lost_gallery_heads h JOIN lost_gallery_builds b ON b.build_key=h.build_key WHERE h.alias=%s',
                (result['alias'],)).fetchone()
            return row == (result['index'], gallery_mapping(result['snapshot_sha256'])['_meta'],
                           result['records'], True, result['records'])

    def validate(self, alias, model_version, color_version=None):
        with self.connection() as conn:
            row = conn.execute('SELECT b.metadata,b.completed,b.expected_count,'
                '(SELECT count(*) FROM lost_gallery_documents d WHERE d.build_key=b.build_key AND d.model_version=%s) '
                'FROM lost_gallery_heads h JOIN lost_gallery_builds b ON b.build_key=h.build_key WHERE h.alias=%s',
                (model_version, alias)).fetchone()
            if not row:
                raise GalleryUnavailable('No published PostgreSQL gallery')
            meta, completed, expected, count = row
            if meta.get('contract') != CONTRACT or meta.get('model_version') != model_version or not completed or count != expected:
                raise RuntimeError('PostgreSQL gallery contract mismatch')
            if color_version and meta.get('coat_color_version') != color_version:
                from app.lost_main import ColorGalleryRefreshRequired
                raise ColorGalleryRefreshRequired('Completed color gallery required')

    def search(self, alias, embedding, species, statuses, limit=200):
        validate_vector(embedding.vector)
        if embedding.animal_vector is not None:
            validate_vector(embedding.animal_vector)
        if limit != 200 or species not in {'DOG', 'CAT'}:
            raise ValueError('Invalid candidate contract')
        from app.services.lost_search import ANIMAL_REGION_WEIGHT
        # Exact cosine on the eligible snapshot. No ANN truncation or score/feature change.
        with self.connection(read_only=True) as conn:
            head = conn.execute('SELECT b.metadata,b.completed FROM lost_gallery_heads h '
                'JOIN lost_gallery_builds b ON b.build_key=h.build_key WHERE h.alias=%s', (alias,)).fetchone()
            if not head or not head[1] or head[0].get('model_version') != embedding.model_version:
                raise GalleryUnavailable('Matching published PostgreSQL gallery required')
            rows = conn.execute('WITH query AS (SELECT %s::public.vector AS whole,%s::public.vector AS animal) '
                'SELECT d.document,(CASE WHEN q.animal IS NOT NULL AND d.animal_vector IS NOT NULL '
                'THEN 1.0 + (1.0-%s)*(1.0-(d.image_vector <=> q.whole)) + %s*(1.0-(d.animal_vector <=> q.animal)) '
                'ELSE 2.0-(d.image_vector <=> q.whole) END)::real AS score '
                'FROM lost_gallery_heads h JOIN lost_gallery_builds b ON b.build_key=h.build_key '
                'JOIN lost_gallery_documents d ON d.build_key=b.build_key CROSS JOIN query q '
                'WHERE h.alias=%s AND b.completed AND d.model_version=%s AND d.species=%s '
                'AND (d.status=ANY(%s) OR d.status IS NULL) ORDER BY score DESC,d.animal_id ASC LIMIT %s',
                (json.dumps(embedding.vector), json.dumps(embedding.animal_vector) if embedding.animal_vector is not None else None,
                 ANIMAL_REGION_WEIGHT, ANIMAL_REGION_WEIGHT, alias, embedding.model_version, species, list(statuses), limit)).fetchall()
        return [{'_source': row[0], '_score': row[1]} for row in rows]

    def recommendation_candidates(self, alias, animal_id, species, model_version, color_version=None):
        """Reuse one published 1024-dim snapshot; filter live availability before top-200."""
        if type(animal_id) is not int or animal_id <= 0 or species not in {'DOG', 'CAT'}:
            raise ValueError('Invalid recommendation source')
        from app.services.lost_search import ANIMAL_REGION_WEIGHT
        with self.connection(read_only=True) as conn:
            head = conn.execute('SELECT h.build_key,b.metadata,b.completed FROM lost_gallery_heads h '
                'JOIN lost_gallery_builds b ON b.build_key=h.build_key WHERE h.alias=%s', (alias,)).fetchone()
            if (not head or not head[2] or head[1].get('contract') != CONTRACT
                    or head[1].get('model_version') != model_version
                    or color_version and head[1].get('coat_color_version') != color_version):
                raise GalleryUnavailable('Matching published recommendation gallery required')
            build = head[0]
            source = conn.execute('SELECT document FROM lost_gallery_documents '
                'WHERE build_key=%s AND animal_id=%s AND species=%s AND model_version=%s',
                (build, animal_id, species, model_version)).fetchone()
            if not source:
                raise GalleryUnavailable('Source image features are not ready')
            # Source may already be adopted. Candidates must still be available now.
            # q and d refer to the same immutable build throughout this transaction.
            rows = conn.execute(
                'SELECT d.document,(CASE WHEN q.animal_vector IS NOT NULL AND d.animal_vector IS NOT NULL '
                'THEN 1.0+(1.0-%s)*(1.0-(d.image_vector <=> q.image_vector)) '
                '+%s*(1.0-(d.animal_vector <=> q.animal_vector)) '
                'ELSE 2.0-(d.image_vector <=> q.image_vector) END)::real AS score '
                'FROM lost_gallery_documents q JOIN lost_gallery_documents d ON d.build_key=q.build_key '
                'JOIN animals a ON a.id=d.animal_id '
                "WHERE q.build_key=%s AND q.animal_id=%s AND d.animal_id<>q.animal_id "
                "AND d.model_version=%s AND d.species=%s AND a.species=%s "
                "AND a.status IN ('NOTICE','PROTECT') ORDER BY score DESC,d.animal_id ASC LIMIT 200",
                (ANIMAL_REGION_WEIGHT, ANIMAL_REGION_WEIGHT, build, animal_id, model_version, species, species)).fetchall()
        return source[0], [{'_source': row[0], '_score': row[1]} for row in rows]

    def delete_unpublished(self, target):
        # Retention calls only for journal-owned generations. Lock matches publication order.
        with self.connection() as conn:
            conn.execute('LOCK TABLE lost_gallery_heads IN SHARE ROW EXCLUSIVE MODE')
            row = conn.execute('SELECT metadata FROM lost_gallery_builds WHERE build_key=%s FOR UPDATE', (target,)).fetchone()
            if not row:
                return True
            from app.services.sam3_focus import FOCUS_VERSION
            if row[0].get('contract') != CONTRACT or row[0].get('model_version') != FOCUS_VERSION:
                raise RuntimeError('Refusing to delete an unowned gallery')
            if conn.execute('SELECT 1 FROM lost_gallery_heads WHERE build_key=%s', (target,)).fetchone():
                return False
            conn.execute('DELETE FROM lost_gallery_builds WHERE build_key=%s', (target,))
            return True
