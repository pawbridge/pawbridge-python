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
