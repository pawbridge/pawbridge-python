"""Opt-in: only a guarded, disposable loopback PostgreSQL. No GPU/production access."""
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from PIL import Image
from app.services.pg_gallery_store import PostgresqlGalleryStore, GalleryUnavailable
from app.services.lost_gallery import build_gallery, gallery_target, gallery_mapping, GalleryBuildCancelled
from app.services.sam3_focus import FOCUS_VERSION
from app.services.coat_color import VERSION as COLOR_VERSION

ALIAS = 'animals-lost-dinov3-sam3-pg-test'
VECTOR = [1.] + [0.] * 1023


def document(id, **changes):
    return dict(id=id, species='DOG', status='PROTECT', source_sha256='a'*64,
                model_version=FOCUS_VERSION, focus_status='original_no_confident_animal',
                image_vector=VECTOR, coat_color_version=COLOR_VERSION, coat_color=None,
                happen_date='2026-09-19', **changes)


@unittest.skipUnless(
    "ANIMAL_PG_MIGRATION_TEST_PORT" in os.environ,
    "Opt-in PostgreSQL test: set ANIMAL_PG_MIGRATION_TEST_PORT for a guarded disposable DB",
)
class PostgresqlGalleryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        port = os.getenv('ANIMAL_PG_MIGRATION_TEST_PORT', '')
        if not port.isdigit() or not 1 <= int(port) <= 65535:
            raise RuntimeError('Explicit disposable loopback port required')
        from psycopg_pool import ConnectionPool
        cls.pool = ConnectionPool(f'host=127.0.0.1 port={port} dbname=pawbridge user=postgres password=local_pg_test_only',
                                 min_size=1, max_size=1, timeout=3, max_waiting=2,
                                 kwargs={'autocommit': True}, open=True)
        cls.pool.wait(timeout=5)
        with cls.pool.connection() as conn:
            if conn.execute('SELECT marker FROM migration_test_guard.guard').fetchall() != [('animal-pg-disposable',)]:
                cls.pool.close()
                raise RuntimeError('Refusing an unguarded database')
        cls.store = PostgresqlGalleryStore(cls.pool)

    @classmethod
    def tearDownClass(cls):
        cls.pool.close()

    def setUp(self):
        with self.store.connection() as conn:
            conn.execute('TRUNCATE lost_gallery_heads,lost_gallery_documents,lost_gallery_builds')

    def begin(self, char, total=1):
        digest = char*64
        target = gallery_target(digest)
        old = self.store.begin(ALIAS, target, digest, total)
        return target, old

    def publish(self, char, documents):
        target, old = self.begin(char, len(documents))
        for offset in range(0,len(documents),100):
            self.store.write_page(target, documents[offset:offset+100])
        self.store.publish(ALIAS, target, old, len(documents), lambda: None)
        return target

    def search(self, animal=None, statuses=('NOTICE','PROTECT')):
        embedding = SimpleNamespace(vector=VECTOR, animal_vector=animal, model_version=FOCUS_VERSION)
        return self.store.search(ALIAS, embedding, 'DOG', statuses)

    def test_partial_failed_and_cancelled_publication_preserve_previous_gallery(self):
        original = self.publish('a', [document(1)])
        target, old = self.begin('b', 2)
        self.store.write_page(target,[document(2)])
        self.assertEqual([h['_source']['id'] for h in self.search()], [1])
        with self.assertRaisesRegex(RuntimeError, 'count'):
            self.store.publish(ALIAS,target,old,2,lambda: None)
        self.store.write_page(target,[document(3)])
        def cancel(): raise GalleryBuildCancelled('cancelled')
        with self.assertRaises(GalleryBuildCancelled):
            self.store.publish(ALIAS,target,old,2,cancel)
        self.assertEqual([h['_source']['id'] for h in self.search()], [1])
        self.store.publish(ALIAS,target,old,2,lambda: None)
        self.assertEqual([h['_source']['id'] for h in self.search()], [2,3])
        with self.assertRaisesRegex(RuntimeError, 'immutable'):
            self.store.write_page(target,[document(4)])
        self.assertEqual(self.store.count(original),1)

    def test_concurrent_publisher_and_wrong_snapshot_contract_are_rejected(self):
        self.publish('a',[document(1)])
        first, old = self.begin('b'); other, other_old = self.begin('c')
        self.store.write_page(first,[document(2)]); self.store.write_page(other,[document(3)])
        self.store.publish(ALIAS,first,old,1,lambda:None)
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            self.store.publish(ALIAS,other,other_old,1,lambda:None)
        with self.assertRaisesRegex(RuntimeError, 'different contract'):
            self.store.begin(ALIAS,first,'b'*64,2)
        self.assertEqual([h['_source']['id'] for h in self.search()], [2])

    def test_a_completed_previous_snapshot_can_be_republished_without_rewriting_vectors(self):
        first=self.publish('a',[document(1)])
        self.publish('b',[document(2)])
        target,old=self.begin('a')
        self.assertEqual(target,first);self.assertTrue(self.store.completed(target,1))
        self.store.publish(ALIAS,target,old,1,lambda:None)
        self.assertEqual([h['_source']['id'] for h in self.search()],[1])

    def test_sql_failure_rolls_back_entire_page_and_returns_usable_connection(self):
        target, _ = self.begin('a',2)
        with self.assertRaises(Exception):
            self.store.write_page(target,[document(1),document(2**80)])
        self.assertEqual(self.store.count(target),0)
        self.store.write_page(target,[document(2)])
        self.assertEqual(self.store.count(target),1)
        self.assertEqual(self.pool.get_stats()['pool_available'],1)

    def test_weighted_cosine_status_species_and_tie_break_match_existing_contract(self):
        rows=[document(1), document(2,animal_vector=VECTOR), document(3), document(4), document(5)]
        rows[1]['image_vector']=[.8,.6]+[0.]*1022
        rows[2]['status']='ADOPTED'; rows[3]['status']='EUTHANIZED'; rows[4]['species']='CAT'
        self.publish('a',rows)
        hits=self.search(VECTOR)
        self.assertEqual([h['_source']['id'] for h in hits],[1,2])
        self.assertAlmostEqual(hits[0]['_score'],2.0,places=6)
        self.assertAlmostEqual(hits[1]['_score'],1.94,places=6)
        self.assertAlmostEqual(self.search()[1]['_score'],1.8,places=6)
        self.assertEqual([h['_source']['id'] for h in self.search(VECTOR,('PROTECT','ADOPTED','RETURNED'))],[1,3,2])

    def test_search_bounds_candidates_and_rejects_missing_or_wrong_model_gallery(self):
        with self.assertRaises(GalleryUnavailable): self.search()
        self.publish('a',[document(i) for i in range(1,206)])
        self.assertEqual([h['_source']['id'] for h in self.search()], list(range(1,201)))
        self.store.validate(ALIAS,FOCUS_VERSION,COLOR_VERSION)
        with self.assertRaisesRegex(RuntimeError,'mismatch'): self.store.validate(ALIAS,'old-model')
        with self.assertRaises(ValueError): self.store.documents(None,gallery_target('a'*64),list(range(101)))

    def test_runtime_pool_caps_connections_times_out_and_closes(self):
        import time
        from unittest.mock import patch
        from psycopg_pool import PoolTimeout
        from app.services.lost_storage import storage_session
        port=os.environ['ANIMAL_PG_MIGRATION_TEST_PORT']
        env={'LOST_STORAGE_BACKEND':'postgresql','LOST_PG_POOL_MAX_SIZE':'1',
             'LOST_PG_DSN':f'host=127.0.0.1 port={port} dbname=pawbridge user=postgres password=local_pg_test_only'}
        with patch.dict(os.environ,env), storage_session() as store:
            with store.connection() as first:
                self.assertEqual(first.execute('SELECT 1').fetchone(),(1,))
                start=time.monotonic()
                with self.assertRaises(PoolTimeout):
                    with store.connection(): self.fail('Cannot exceed the configured connection limit')
                elapsed=time.monotonic()-start
                self.assertGreaterEqual(elapsed,2.5);self.assertLess(elapsed,6)
                self.assertEqual(store.pool.get_stats()['pool_size'],1)
            with store.connection() as returned:
                self.assertEqual(returned.execute('SELECT 2').fetchone(),(2,))
        self.assertTrue(store.pool.closed)

    def test_search_keeps_one_published_snapshot_during_concurrent_switch(self):
        import psycopg
        from unittest.mock import patch
        old=self.publish('a',[document(1)])
        new=self.publish('b',[document(2)])
        self.store.publish(ALIAS,old,new,1,lambda:None)
        original=self.store.connection
        switched=[]
        port=os.environ['ANIMAL_PG_MIGRATION_TEST_PORT']
        @contextmanager
        def intercept(**kwargs):
            with original(**kwargs) as connection:
                class Proxy:
                    def execute(inner,sql,*args):
                        result=connection.execute(sql,*args)
                        if sql.startswith('SELECT b.metadata,b.completed FROM lost_gallery_heads'):
                            with psycopg.connect(f'host=127.0.0.1 port={port} dbname=pawbridge user=postgres password=local_pg_test_only') as writer:
                                writer.execute('UPDATE pawbridge_animal.lost_gallery_heads SET build_key=%s WHERE alias=%s',(new,ALIAS))
                            switched.append(True)
                        return result
                yield Proxy()
        with patch.object(self.store,'connection',intercept):
            hits=self.search()
        self.assertEqual(switched,[True])
        self.assertEqual([h['_source']['id'] for h in hits],[1])
        self.assertEqual([h['_source']['id'] for h in self.search()],[2])

    def test_retention_never_deletes_an_active_gallery(self):
        first=self.publish('a',[document(1)])
        self.assertFalse(self.store.delete_unpublished(first))
        second=self.publish('b',[document(2)])
        self.assertTrue(self.store.delete_unpublished(first))
        self.assertEqual(self.store.count(first),0)
        self.assertFalse(self.store.delete_unpublished(second))

    def test_retention_bulk_cleanup_uses_bounded_committed_transactions(self):
        retired=self.publish('a',[document(i) for i in range(1,1202)])
        active=self.publish('b',[document(2001)])
        with self.store.connection() as conn:
            conn.execute('CREATE TEMP TABLE retention_deleted_rows (transaction_id bigint)')
            conn.execute('CREATE FUNCTION pg_temp.audit_retention_delete() RETURNS trigger LANGUAGE plpgsql AS $$ '
                         'BEGIN INSERT INTO retention_deleted_rows VALUES (txid_current()); RETURN OLD; END $$')
            conn.execute('CREATE TRIGGER retention_audit AFTER DELETE ON lost_gallery_documents '
                         'FOR EACH ROW EXECUTE FUNCTION pg_temp.audit_retention_delete()')
        try:
            self.assertTrue(self.store.delete_unpublished(retired))
            with self.store.connection() as conn:
                sizes=[r[0] for r in conn.execute('SELECT count(*) FROM retention_deleted_rows GROUP BY transaction_id').fetchall()]
            self.assertEqual(sum(sizes),1201)
            self.assertGreater(len(sizes),1)
            self.assertLessEqual(max(sizes),500)
            self.assertEqual(self.store.count(active),1)
            self.assertEqual([h['_source']['id'] for h in self.search()],[2001])
        finally:
            with self.store.connection() as conn:
                conn.execute('DROP TRIGGER retention_audit ON lost_gallery_documents')
                conn.execute('DROP TABLE retention_deleted_rows')

    def test_retention_sql_failure_keeps_committed_progress_and_can_resume(self):
        retired=self.publish('a',[document(i) for i in range(1,1102)])
        self.publish('b',[document(2001)])
        with self.store.connection() as conn:
            conn.execute("CREATE FUNCTION pg_temp.fail_retention_row() RETURNS trigger LANGUAGE plpgsql AS $$ "
                         "BEGIN IF OLD.animal_id=501 THEN RAISE EXCEPTION 'controlled cleanup failure'; END IF; RETURN OLD; END $$")
            conn.execute('CREATE TRIGGER retention_failure BEFORE DELETE ON lost_gallery_documents '
                         'FOR EACH ROW EXECUTE FUNCTION pg_temp.fail_retention_row()')
        try:
            with self.assertRaisesRegex(Exception,'controlled cleanup failure'):
                self.store.delete_unpublished(retired)
            self.assertEqual(self.store.count(retired),601)
            self.assertEqual([h['_source']['id'] for h in self.search()],[2001])
            self.assertFalse(self.store.completed(retired,1101))
        finally:
            with self.store.connection() as conn:
                conn.execute('DROP TRIGGER retention_failure ON lost_gallery_documents')
        self.assertTrue(self.store.delete_unpublished(retired))
        self.assertEqual(self.store.count(retired),0)
        self.assertTrue(self.store.delete_unpublished(retired))

    def test_retention_cancellation_between_pages_preserves_search_and_resumes(self):
        retired=self.publish('a',[document(i) for i in range(1,1102)])
        self.publish('b',[document(2001)])
        checks=[]
        def cancel_after_first_page():
            checks.append(True)
            if len(checks)>1: raise GalleryBuildCancelled('controlled stop')
        with self.assertRaises(GalleryBuildCancelled):
            self.store.delete_unpublished(retired,check_cancelled=cancel_after_first_page)
        self.assertEqual(self.store.count(retired),601)
        self.assertEqual([h['_source']['id'] for h in self.search()],[2001])
        self.assertTrue(self.store.delete_unpublished(retired))

    def test_retention_rechecks_new_alias_references_between_pages(self):
        retired=self.publish('a',[document(i) for i in range(1,1102)])
        self.publish('b',[document(2001)])
        calls=[]
        def attach_protected_reference():
            calls.append(True)
            if len(calls)==2:
                with self.store.connection() as conn:
                    conn.execute('INSERT INTO lost_gallery_heads(alias,build_key) VALUES (%s,%s)',
                                 (ALIAS+'-reserved',retired))
        self.assertFalse(self.store.delete_unpublished(retired,check_cancelled=attach_protected_reference))
        self.assertEqual(self.store.count(retired),601)
        self.assertEqual([h['_source']['id'] for h in self.search()],[2001])

    def test_retention_preserves_rollback_alias_and_rejects_foreign_metadata(self):
        retired=self.publish('a',[document(i) for i in range(1,602)])
        self.publish('b',[document(2001)])
        other,_=self.begin('c');self.store.write_page(other,[document(3001)])
        with self.store.connection() as conn:
            conn.execute('INSERT INTO lost_gallery_heads(alias,build_key) VALUES (%s,%s)',(ALIAS+'-rollback',retired))
            conn.execute("UPDATE lost_gallery_builds SET metadata=metadata||'{\"contract\":\"foreign\"}'::jsonb WHERE build_key=%s",(other,))
        self.assertFalse(self.store.delete_unpublished(retired))
        self.assertEqual(self.store.count(retired),601)
        with self.assertRaisesRegex(RuntimeError,'unowned'):
            self.store.delete_unpublished(other)
        self.assertEqual(self.store.count(other),1)

    def test_retention_slow_delete_respects_each_statement_timeout(self):
        from unittest.mock import patch
        retired=self.publish('a',[document(i) for i in range(1,1102)])
        self.publish('b',[document(2001)])
        with self.store.connection() as conn:
            conn.execute('CREATE FUNCTION pg_temp.slow_retention_row() RETURNS trigger LANGUAGE plpgsql AS $$ '
                         'BEGIN PERFORM pg_sleep(0.002); RETURN OLD; END $$')
            conn.execute('CREATE TRIGGER retention_slow BEFORE DELETE ON lost_gallery_documents '
                         'FOR EACH ROW EXECUTE FUNCTION pg_temp.slow_retention_row()')
        original=self.store.connection
        @contextmanager
        def short_timeout(**kwargs):
            with original(**kwargs) as conn:
                conn.execute("SET LOCAL statement_timeout = '1800ms'")
                yield conn
        try:
            # Controlled row delay makes one cascade exceed this test-only budget.
            # Production keeps the existing 10-second statement timeout.
            with patch.object(self.store,'connection',short_timeout):
                self.assertTrue(self.store.delete_unpublished(retired))
            self.assertEqual(self.store.count(retired),0)
            self.assertEqual([h['_source']['id'] for h in self.search()],[2001])
        finally:
            with self.store.connection() as conn:
                conn.execute('DROP TRIGGER retention_slow ON lost_gallery_documents')


    def test_builder_releases_pool_during_download_inference_and_reuses_features(self):
        image=io.BytesIO()
        Image.new('RGB',(8,8),'white').save(image,format='PNG')
        payload=image.getvalue(); digest=hashlib.sha256(payload).hexdigest()
        def unborrowed(): self.assertEqual(self.pool.get_stats()['pool_available'],1)
        class Encoder:
            model_version=FOCUS_VERSION
            calls=0
            def encode_with_metadata(inner,*args,**kwargs):
                unborrowed(); inner.calls+=1
                return SimpleNamespace(model_version=FOCUS_VERSION,vector=VECTOR,animal_vector=None,
                                       focus_status='original_no_confident_animal',coat_color=None)
        encoder=Encoder()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); path=root/'photo.png'; path.write_bytes(payload)
            @contextmanager
            def provider(row):
                unborrowed(); yield path
            rows=[dict(id=i,species='DOG',status='PROTECT',source_sha256=digest) for i in (1,2)]
            manifest=root/'manifest.json';manifest.write_text(json.dumps({'complete':True,'records':rows}))
            result=build_gallery(None,lambda:encoder,manifest,root,ALIAS,root,photo_provider=provider,store=self.store)
            self.assertEqual(result['encoded'],2);self.assertTrue(self.store.published(result))
            rows[0]['color']='metadata changed'
            manifest.write_text(json.dumps({'complete':True,'records':rows}))
            result=build_gallery(None,lambda:encoder,manifest,root,ALIAS,root,photo_provider=provider,store=self.store)
            self.assertEqual(result['encoded'],0);self.assertEqual(result['reused'],2)
            self.assertEqual(encoder.calls,2);self.assertTrue(self.store.published(result))

    def test_incremental_builder_adds_one_animal_without_copying_unchanged_rows(self):
        from app.services.gallery_retention import GalleryRetention
        payload=io.BytesIO()
        with Image.new('RGB',(8,8),'white') as image: image.save(payload,format='PNG')
        data=payload.getvalue(); digest=hashlib.sha256(data).hexdigest()
        calls=[]
        class Encoder:
            model_version=FOCUS_VERSION
            def encode_with_metadata(inner,*args,**kwargs):
                self.assertEqual(self.pool.get_stats()['pool_available'],1)
                calls.append('encode')
                return SimpleNamespace(model_version=FOCUS_VERSION,vector=VECTOR,animal_vector=None,
                                       focus_status='original_no_confident_animal',coat_color=None)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); photo=root/'photo.png'; photo.write_bytes(data)
            manifest=root/'manifest.json'
            rows=[dict(id=i,species='DOG',status='PROTECT',source_sha256=digest) for i in (1,2)]
            keeper=GalleryRetention(None,root,ALIAS,store=self.store)
            @contextmanager
            def provider(row): yield photo
            def run():
                manifest.write_text(json.dumps({'complete':True,'records':rows}))
                return build_gallery(None,Encoder,manifest,root,ALIAS,root,photo_provider=provider,
                                     store=self.store,incremental=True,prepare=keeper.prepare_target)
            first=run(); keeper.published(first['index'])
            with self.store.connection() as conn:
                before=conn.execute('SELECT animal_id,xmin::text,ctid::text FROM lost_gallery_documents '
                                    'WHERE build_key=%s ORDER BY animal_id',(first['index'],)).fetchall()
            rows.append(dict(id=3,species='DOG',status='PROTECT',source_sha256=digest))
            second=run(); keeper.published(second['index'])
            self.assertEqual(second['index'],first['index'])
            self.assertEqual((second['encoded'],second['staged'],second['inserted'],second['updated']),(1,1,1,0))
            self.assertTrue(self.store.published(second))
            with self.store.connection() as conn:
                after=conn.execute('SELECT animal_id,xmin::text,ctid::text FROM lost_gallery_documents '
                                   'WHERE build_key=%s AND animal_id IN (1,2) ORDER BY animal_id',(first['index'],)).fetchall()
                self.assertEqual(conn.execute('SELECT count(*) FROM lost_gallery_builds').fetchone(),(1,))
            self.assertEqual(before,after)
            self.assertEqual(keeper.journal,{'known':[first['index']],'published':[first['index']]})
            rows.pop()  # Returning to an earlier source hash still uses the stable active key safely.
            third=run(); keeper.published(third['index'])
            self.assertEqual((third['index'],third['encoded'],third['removed']),(first['index'],0,1))
            self.assertTrue(self.store.published(third)); self.assertEqual(len(calls),3)

    def test_incremental_restart_replays_metadata_without_repeating_staged_photo_inference(self):
        payload=io.BytesIO()
        with Image.new('RGB',(8,8),'white') as image: image.save(payload,format='PNG')
        data=payload.getvalue(); digest=hashlib.sha256(data).hexdigest()
        rows=[dict(id=i,species='DOG',status='PROTECT',source_sha256=digest) for i in range(1,5)]
        calls=[]; downloads=[]
        class Encoder:
            model_version=FOCUS_VERSION
            fail=False
            def encode_with_metadata(inner,*args,**kwargs):
                self.assertEqual(self.pool.get_stats()['pool_available'],1)
                calls.append('encode')
                if inner.fail and len(calls)==4: raise RuntimeError('Last new photo failed')
                return SimpleNamespace(model_version=FOCUS_VERSION,vector=VECTOR,animal_vector=None,
                                       focus_status='original_no_confident_animal',coat_color=None)
        encoder=Encoder()
        class Stream:
            total=4; processed=0; resets=0; verified=False
            checkpoint={'descriptor':{'snapshotSha256':'d'*64}}
            def pages(inner,cancelled):
                for row in rows[inner.processed:]: yield [row]
            def acknowledge(inner,count): inner.processed=count
            def reset_resume(inner): inner.processed=0; inner.resets+=1
            def verify_complete(inner): inner.verified=True
        stream=Stream()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); photo=root/'photo.png'; photo.write_bytes(data)
            manifest=root/'manifest.json';manifest.write_text(json.dumps({'complete':True,'records':rows[:2]}))
            @contextmanager
            def provider(row): downloads.append(row['id']); yield photo
            original=build_gallery(None,lambda:encoder,manifest,root,ALIAS,root,store=self.store,photo_provider=provider)
            downloads.clear(); encoder.fail=True
            def run():
                return build_gallery(None,lambda:encoder,None,root,ALIAS,root,store=self.store,
                                     stream=stream,incremental=True,photo_provider=provider)
            with self.assertRaisesRegex(RuntimeError,'Last new photo failed'): run()
            self.assertEqual(stream.processed,3)
            self.assertTrue(self.store.published(original))
            self.assertEqual([hit['_source']['id'] for hit in self.search()],[1,2])
            result=run()
            self.assertEqual(stream.resets,1); self.assertTrue(stream.verified)
            self.assertEqual(downloads,[3,4,4]); self.assertEqual(len(calls),5)
            self.assertEqual(result['encoded'],1); self.assertEqual(result['inserted'],2)
            self.assertTrue(self.store.published(result))

    def test_incremental_final_fingerprint_failure_keeps_published_search_unchanged(self):
        original=self.publish('a',[document(1)])
        rows=[document(1),document(2)]
        # Pre-populate the second animal's reusable features in a delta to avoid GPU fixtures.
        session,target=self.store.refresh_store(ALIAS,gallery_target('b'*64),'b'*64,2)
        session.begin(ALIAS,target,'b'*64,2)
        self.store.write_page(target,[document(2)])
        class Stream:
            total=2; processed=0
            checkpoint={'descriptor':{'snapshotSha256':'b'*64}}
            def pages(inner,cancelled): yield rows
            def acknowledge(inner,count): inner.processed=count
            def verify_complete(inner): raise ValueError('Complete snapshot fingerprint differs')
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError,'fingerprint'):
            build_gallery(None,lambda:SimpleNamespace(model_version=FOCUS_VERSION),None,directory,ALIAS,directory,
                          store=self.store,incremental=True,stream=Stream())
        self.assertEqual([hit['_source']['id'] for hit in self.search()],[1])
        self.assertEqual(self.store.count(original),1)

    def test_page_cursor_is_not_advanced_when_commit_acknowledgment_fails(self):
        row=document(1)
        target, _=self.begin('a')
        self.store.write_page(target,[row])  # Resumed staging; requires no GPU download.
        class Stream:
            total=1;processed=0
            checkpoint={'descriptor':{'snapshotSha256':'a'*64}}
            acknowledged=[]
            verified=False
            def pages(inner,cancelled): yield [row]
            def acknowledge(inner,count): inner.acknowledged.append(count)
            def verify_complete(inner): inner.verified=True
        stream=Stream()
        original=self.store.write_page
        def uncertain(*args):
            original(*args)
            raise RuntimeError('Commit acknowledgment uncertain')
        with tempfile.TemporaryDirectory() as directory:
            from unittest.mock import patch
            with patch.object(self.store,'write_page',side_effect=uncertain), self.assertRaises(RuntimeError):
                build_gallery(None,lambda:SimpleNamespace(model_version=FOCUS_VERSION),None,directory,ALIAS,directory,
                              stream=stream,store=self.store)
            self.assertEqual(stream.acknowledged,[]);self.assertFalse(stream.verified)
            with self.assertRaises(GalleryUnavailable): self.search()
            result=build_gallery(None,lambda:SimpleNamespace(model_version=FOCUS_VERSION),None,directory,ALIAS,directory,
                                 stream=stream,store=self.store)
            self.assertEqual(stream.acknowledged,[1]);self.assertTrue(stream.verified)
            self.assertTrue(self.store.published(result));self.assertEqual(result['reused'],1)

    def test_color_v3_refresh_reuses_vectors_resumes_and_can_restore_v2(self):
        from unittest.mock import patch
        import numpy as np
        from app.services.coat_color import describe, valid
        from app.lost_main import ColorGalleryRefreshRequired

        old_version = 'foreground-lab32-joint8-v2'
        snapshot = 'd' * 64
        with Image.new('RGB', (32, 32), (128, 129, 128)) as image:
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            payload = buffer.getvalue()
            corrected = describe(image, np.ones((32, 32), dtype=bool))
        digest = hashlib.sha256(payload).hexdigest()
        rows = [dict(id=i, species='DOG', status='PROTECT', source_sha256=digest) for i in (1, 2, 3)]
        legacy = [dict(document(i, animal_vector=VECTOR), source_sha256=digest,
                       focus_status='animal_mask', coat_color_version=old_version,
                       coat_color=dict(corrected, version=old_version)) for i in (1, 2, 3)]
        # A previous no-mask result remains null and requires no extra inference.
        legacy[-1].update(focus_status='original_no_confident_animal', coat_color=None)
        legacy[-1].pop('animal_vector')
        with patch('app.services.coat_color.VERSION', old_version), \
                patch('app.services.lost_gallery.COLOR_VERSION', old_version):
            old_target = self.publish('d', legacy)
        old_hits = self.search(VECTOR)
        self.store.validate(ALIAS, FOCUS_VERSION)  # Color weight zero can serve the old generation.
        with self.assertRaises(ColorGalleryRefreshRequired):
            self.store.validate(ALIAS, FOCUS_VERSION, COLOR_VERSION)

        class Stream:
            total = 3
            processed = 0
            checkpoint = {'descriptor': {'snapshotSha256': snapshot}}
            verified = False
            def pages(inner, cancelled):
                for row in rows[inner.processed:]:
                    yield [row]
            def acknowledge(inner, count):
                inner.processed = count
            def verify_complete(inner):
                inner.verified = True

        calls = []
        class Encoder:
            model_version = FOCUS_VERSION
            def encode_with_metadata(inner, *args, **kwargs):
                raise AssertionError('Unchanged DINO vectors must not be recomputed')
            def describe_coat_color(inner, image, species, **kwargs):
                self.assertEqual(self.pool.get_stats()['pool_available'], 1)
                calls.append(species)
                if len(calls) == 2:
                    raise RuntimeError('Injected second-photo failure')
                return describe(image, np.ones((image.height, image.width), dtype=bool))

        stream = Stream()
        downloads = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            photo = root / 'photo.png'
            photo.write_bytes(payload)
            @contextmanager
            def provider(row):
                downloads.append(row['id'])
                yield photo
            def run():
                return build_gallery(None, Encoder, None, root, ALIAS, root,
                                     stream=stream, photo_provider=provider, store=self.store)
            with self.assertRaisesRegex(RuntimeError, 'Injected second-photo failure'):
                run()
            target = gallery_target(snapshot)
            self.assertNotEqual(target, old_target)
            self.assertEqual(stream.processed, 1)
            self.assertEqual(self.store.count(target), 1)
            self.assertFalse(stream.verified)
            self.assertFalse(self.store.completed(target, 3))
            self.assertEqual(self.search(VECTOR), old_hits)
            self.store.validate(ALIAS, FOCUS_VERSION, old_version)
            result = run()

        self.assertTrue(stream.verified)
        self.assertEqual(result['encoded'], 0)
        self.assertEqual(result['reused'], 3)
        self.assertEqual(downloads, [1, 2, 2])  # Completed row 1 is not fetched again; row 3 has no mask.
        self.assertTrue(self.store.published(result))
        self.store.validate(ALIAS, FOCUS_VERSION, COLOR_VERSION)
        with self.store.connection() as conn:
            vectors = conn.execute(
                'SELECT animal_id,image_vector::text,animal_vector::text '
                'FROM lost_gallery_documents WHERE build_key=%s ORDER BY animal_id', (old_target,)).fetchall()
            self.assertEqual(vectors, conn.execute(
                'SELECT animal_id,image_vector::text,animal_vector::text '
                'FROM lost_gallery_documents WHERE build_key=%s ORDER BY animal_id', (target,)).fetchall())
        new_hits = self.search(VECTOR)
        self.assertEqual([(h['_source']['id'], h['_score']) for h in new_hits],
                         [(h['_source']['id'], h['_score']) for h in old_hits])
        for hit in new_hits:
            self.assertEqual(hit['_source']['coat_color_version'], COLOR_VERSION)
            color = hit['_source']['coat_color']
            self.assertEqual(color is None, hit['_source']['id'] == 3)
            if color is not None:
                self.assertTrue(valid(color))
                self.assertEqual(color, corrected)

        # Rollback switches only the head; neither generation is deleted or rewritten.
        self.store.publish(ALIAS, old_target, target, 3, lambda: None)
        self.store.validate(ALIAS, FOCUS_VERSION, old_version)
        self.assertEqual(self.search(VECTOR), old_hits)
        self.assertEqual(self.store.count(target), 3)
        with self.assertRaises(ColorGalleryRefreshRequired):
            self.store.validate(ALIAS, FOCUS_VERSION, COLOR_VERSION)

    def test_color_rollback_reference_survives_two_new_publications_and_retention(self):
        from unittest.mock import patch
        from app.services.gallery_retention import GalleryRetention
        old_version = 'foreground-lab32-joint8-v2'
        rollback_alias = 'animals-lost-dinov3-sam3-color-v2-rollback'
        with patch('app.services.lost_gallery.COLOR_VERSION', old_version):
            old = self.publish('a', [dict(document(1), coat_color_version=old_version)])
        with self.store.connection() as conn:
            conn.execute('INSERT INTO lost_gallery_heads(alias,build_key) VALUES (%s,%s)', (rollback_alias, old))
        with tempfile.TemporaryDirectory() as directory:
            with patch('app.services.lost_gallery.COLOR_VERSION', old_version):
                retention = GalleryRetention(None, directory, ALIAS, store=self.store)
                retention.prepare('a' * 64)
                retention.published(old)
            for char, animal_id in [('b', 2), ('c', 3)]:
                retention.prepare(char * 64)
                current = self.publish(char, [document(animal_id)])
                retention.published(current)
            restarted = GalleryRetention(None, directory, ALIAS, store=self.store)
            restarted.prune(set(restarted.journal['published']))
        self.assertNotIn(old, restarted.journal['published'])
        self.assertFalse(self.store.delete_unpublished(old))
        self.assertEqual(self.store.count(old), 1)
        self.store.validate(rollback_alias, FOCUS_VERSION, old_version)
        self.assertEqual([h['_source']['id'] for h in self.search()], [3])
        self.store.publish(ALIAS, old, current, 1, lambda: None)
        self.assertEqual([h['_source']['id'] for h in self.search()], [1])

    def seed_recommendation_animals(self, statuses):
        with self.store.connection() as conn:
            conn.execute("INSERT INTO shelters(id,created_at,care_reg_no,name) VALUES (900001,now(),'recommendation-fixture','test') ON CONFLICT(id) DO NOTHING")
            for animal_id, status, species in statuses:
                conn.execute("INSERT INTO animals(id,created_at,api_source,apms_notice_no,favorite_count,gender,neuter_status,notice_end_date,notice_start_date,species,status,shelter_id) "
                    "VALUES (%s,now(),'MANUAL',%s,0,'UNKNOWN','UNKNOWN',DATE '2020-01-10',DATE '2020-01-01',%s,%s,900001) "
                    "ON CONFLICT(id) DO UPDATE SET status=excluded.status,species=excluded.species",
                    (animal_id, 'recommendation-fixture-'+str(animal_id), species, status))

    def test_recommendations_filter_live_availability_before_200_and_allow_adopted_source(self):
        rows = [document(i) for i in range(900001, 900209)]
        rows[0]['status'] = 'ADOPTED'
        rows[-1]['status'] = 'ADOPTED'  # Snapshot is old; current DB says PROTECT.
        rows[-1]['image_vector'] = [.8, .6] + [0.] * 1022
        self.seed_recommendation_animals([(i, 'ADOPTED', 'DOG') for i in range(900001, 900209)])
        self.seed_recommendation_animals([(900208, 'PROTECT', 'DOG'), (900207, 'PROTECT', 'CAT')])
        self.publish('a', rows)
        source, hits = self.store.recommendation_candidates(ALIAS, 900001, 'DOG', FOCUS_VERSION, COLOR_VERSION)
        self.assertEqual(source['id'], 900001)
        self.assertEqual([hit['_source']['id'] for hit in hits], [900208])
        self.assertAlmostEqual(hits[0]['_score'], 1.8, places=6)
        self.assertEqual(self.pool.get_stats()['pool_available'], 1)
        with self.assertRaises(GalleryUnavailable):
            self.store.recommendation_candidates(ALIAS, 900001, 'DOG', 'old-model')
        with self.assertRaises(GalleryUnavailable):
            self.store.recommendation_candidates(ALIAS, 900999, 'DOG', FOCUS_VERSION)
        with self.assertRaises(GalleryUnavailable):
            self.store.recommendation_candidates(ALIAS, 900001, 'DOG', FOCUS_VERSION, 'old-color')

    def test_recommendations_keep_dual_vector_score_and_reject_384_dimensions(self):
        rows = [document(900001, animal_vector=VECTOR), document(900002, animal_vector=VECTOR)]
        rows[1]['image_vector'] = [.8, .6] + [0.] * 1022
        self.seed_recommendation_animals([(900001, 'ADOPTED', 'DOG'), (900002, 'NOTICE', 'DOG')])
        self.publish('a', rows)
        source, hits = self.store.recommendation_candidates(ALIAS, 900001, 'DOG', FOCUS_VERSION)
        self.assertEqual([hit['_source']['id'] for hit in hits], [900002])
        self.assertAlmostEqual(hits[0]['_score'], 1.94, places=6)
        target, _ = self.begin('b')
        invalid = document(900003)
        invalid['image_vector'] = [1.] + [0.] * 383
        with self.assertRaisesRegex(RuntimeError, 'Invalid DINOv3'):
            self.store.write_page(target, [invalid])
        self.assertEqual(self.store.count(target), 0)

    def test_recommendation_source_and_candidates_stay_on_one_snapshot_during_publication(self):
        import psycopg
        from unittest.mock import patch
        self.seed_recommendation_animals([(900001, 'ADOPTED', 'DOG'), (900002, 'PROTECT', 'DOG'), (900003, 'PROTECT', 'DOG')])
        old = self.publish('a', [document(900001), document(900002)])
        new = self.publish('b', [document(900001), document(900003)])
        self.store.publish(ALIAS, old, new, 2, lambda: None)
        original = self.store.connection
        port = os.environ['ANIMAL_PG_MIGRATION_TEST_PORT']
        switched = []
        @contextmanager
        def intercept(**kwargs):
            with original(**kwargs) as connection:
                class Proxy:
                    def execute(inner, sql, *args):
                        result = connection.execute(sql, *args)
                        if sql.startswith('SELECT h.build_key,b.metadata,b.completed FROM lost_gallery_heads'):
                            with psycopg.connect(f'host=127.0.0.1 port={port} dbname=pawbridge user=postgres password=local_pg_test_only') as writer:
                                writer.execute('UPDATE pawbridge_animal.lost_gallery_heads SET build_key=%s WHERE alias=%s', (new, ALIAS))
                            switched.append(True)
                        return result
                yield Proxy()
        with patch.object(self.store, 'connection', intercept):
            _, hits = self.store.recommendation_candidates(ALIAS, 900001, 'DOG', FOCUS_VERSION)
        self.assertEqual(switched, [True])
        self.assertEqual([hit['_source']['id'] for hit in hits], [900002])
        _, hits = self.store.recommendation_candidates(ALIAS, 900001, 'DOG', FOCUS_VERSION)
        self.assertEqual([hit['_source']['id'] for hit in hits], [900003])

    def test_recommendation_http_queries_postgresql_and_returns_connection_before_reranking(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from unittest.mock import patch
        from app.routers.recommendation import router
        from app.services.lost_search import rank_candidates
        self.seed_recommendation_animals([(900001, 'ADOPTED', 'DOG'), (900002, 'NOTICE', 'DOG'), (900003, 'ADOPTED', 'DOG')])
        self.publish('a', [document(900001), document(900002), document(900003)])
        app = FastAPI()
        app.include_router(router, prefix='/internal/animals')
        def rank(*args, **kwargs):
            self.assertEqual(self.pool.get_stats()['pool_available'], 1)
            return rank_candidates(*args, **kwargs)
        env = {'INTERNAL_API_KEY': 'test-key', 'LOST_STORAGE_BACKEND': 'postgresql',
               'LOST_SEARCH_VISUAL_PROFILE': 'sam3-animal-focus', 'LOST_SEARCH_INDEX': ALIAS,
               'LOST_SEARCH_COAT_COLOR_WEIGHT': '0.1'}
        with patch.dict(os.environ, env), patch('app.services.recommendation.get_postgresql_store', return_value=self.store), \
                patch('app.services.recommendation.rank_candidates', side_effect=rank), TestClient(app) as client:
            response = client.get('/internal/animals/900001/similar?species=DOG', headers={'X-Internal-Api-Key': 'test-key'})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), [900002])
