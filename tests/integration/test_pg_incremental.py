"""Sparse refresh contracts in the existing guarded disposable PostgreSQL fixture."""
from contextlib import contextmanager
import os
import unittest

from app.services.lost_gallery import GalleryBuildCancelled, gallery_mapping, gallery_target
import test_pg_gallery as gallery_fixture
from test_pg_gallery import ALIAS, VECTOR, document


@unittest.skipUnless(
    "ANIMAL_PG_MIGRATION_TEST_PORT" in os.environ,
    "Opt-in PostgreSQL test: set ANIMAL_PG_MIGRATION_TEST_PORT for a guarded disposable DB",
)
class PostgresqlIncrementalGalleryTest(unittest.TestCase):
    # Share only the guarded lifecycle and helpers; inheriting its TestCase would
    # collect all full-generation tests a second time.
    setUpClass = classmethod(gallery_fixture.PostgresqlGalleryTest.setUpClass.__func__)
    tearDownClass = classmethod(gallery_fixture.PostgresqlGalleryTest.tearDownClass.__func__)
    setUp = gallery_fixture.PostgresqlGalleryTest.setUp
    begin = gallery_fixture.PostgresqlGalleryTest.begin
    publish = gallery_fixture.PostgresqlGalleryTest.publish
    search = gallery_fixture.PostgresqlGalleryTest.search

    @staticmethod
    def row(animal_id, **changes):
        result = document(animal_id)
        result.update(changes)
        return result

    def refresh(self, char, total):
        digest = char * 64
        session, target = self.store.refresh_store(ALIAS, gallery_target(digest), digest, total)
        old = session.begin(ALIAS, target, digest, total)
        return session, target, old

    def stage(self, session, target, old, rows):
        session.documents(old, target, [row['id'] for row in rows])
        session.write_page(target, rows)

    def fingerprints(self, target):
        with self.store.connection() as conn:
            return conn.execute(
                'SELECT animal_id,xmin::text,ctid::text,document,image_vector::text,animal_vector::text '
                'FROM lost_gallery_documents WHERE build_key=%s ORDER BY animal_id',
                (target,),
            ).fetchall()

    def build_state(self, target):
        with self.store.connection() as conn:
            return conn.execute(
                'SELECT metadata,expected_count,completed FROM lost_gallery_builds WHERE build_key=%s',
                (target,),
            ).fetchone()

    def head(self, alias=ALIAS):
        with self.store.connection() as conn:
            return conn.execute('SELECT build_key FROM lost_gallery_heads WHERE alias=%s', (alias,)).fetchone()[0]

    @contextmanager
    def audited_document_writes(self):
        with self.store.connection() as conn:
            conn.execute('CREATE TEMP TABLE incremental_writes (operation text,build_key text,animal_id bigint)')
            conn.execute(
                'CREATE OR REPLACE FUNCTION pg_temp.audit_incremental_write() RETURNS trigger LANGUAGE plpgsql AS $$ '
                'BEGIN IF TG_OP=\'DELETE\' THEN '
                'INSERT INTO incremental_writes VALUES (TG_OP,OLD.build_key,OLD.animal_id); RETURN OLD; '
                'ELSE INSERT INTO incremental_writes VALUES (TG_OP,NEW.build_key,NEW.animal_id); RETURN NEW; '
                'END IF; END $$'
            )
            conn.execute(
                'CREATE TRIGGER incremental_write_audit AFTER INSERT OR UPDATE OR DELETE ON lost_gallery_documents '
                'FOR EACH ROW EXECUTE FUNCTION pg_temp.audit_incremental_write()'
            )
        try:
            yield
        finally:
            with self.store.connection() as conn:
                conn.execute('DROP TRIGGER incremental_write_audit ON lost_gallery_documents')
                conn.execute('DROP TABLE incremental_writes')

    def writes(self):
        with self.store.connection() as conn:
            return conn.execute(
                'SELECT operation,build_key,animal_id FROM incremental_writes ORDER BY build_key,animal_id,operation'
            ).fetchall()

    def assert_published(self, base, char, total):
        self.assertEqual(self.head(), base)
        self.assertEqual(self.build_state(base), (gallery_mapping(char * 64)['_meta'], total, True))
        self.assertEqual(self.store.count(base), total)

    def test_unchanged_pages_stage_no_document_writes_and_preserve_all_base_tuples(self):
        rows = [document(i) for i in range(1, 106)]
        base = self.publish('a', rows)
        before = self.fingerprints(base)
        with self.audited_document_writes():
            session, target, old = self.refresh('b', len(rows))
            self.assertIsNot(session, self.store)
            self.assertEqual(old, base)
            for offset in range(0, len(rows), 100):
                self.stage(session, target, old, rows[offset:offset + 100])
            self.assertEqual(session.count(target), 0)
            self.assertEqual(self.writes(), [])
            session.publish(ALIAS, target, old, len(rows), lambda: None)
            self.assertEqual(self.writes(), [])
        self.assertEqual(self.fingerprints(base), before)
        self.assert_published(base, 'b', len(rows))
        self.assertEqual(session.result_index(target), base)
        self.assertIsNone(self.build_state(target))

    def test_one_added_animal_stages_and_merges_only_that_row(self):
        base = self.publish('a', [document(1)])
        before = self.fingerprints(base)
        with self.audited_document_writes():
            session, target, old = self.refresh('b', 2)
            self.stage(session, target, old, [document(1), document(2)])
            self.assertEqual(session.count(target), 1)
            self.assertEqual(self.writes(), [('INSERT', target, 2)])
            self.assertEqual(self.fingerprints(base), before)
            session.publish(ALIAS, target, old, 2, lambda: None)
            self.assertCountEqual(self.writes(), [('INSERT', target, 2), ('INSERT', base, 2), ('DELETE', target, 2)])
        self.assertEqual(self.fingerprints(base)[0], before[0])
        self.assertEqual([hit['_source']['id'] for hit in self.search()], [1, 2])
        self.assert_published(base, 'b', 2)
        self.assertIsNone(self.build_state(target))

    def test_metadata_only_change_preserves_both_feature_vectors(self):
        source = self.row(1, animal_vector=VECTOR)
        base = self.publish('a', [source, document(2)])
        before = self.fingerprints(base)
        session, target, old = self.refresh('b', 2)
        cached = session.documents(old, target, [1, 2])
        changed = dict(cached[1], status='ADOPTED', description='updated shelter metadata')
        session.write_page(target, [changed, cached[2]])
        self.assertEqual(session.count(target), 1)
        resumed, resumed_target, resumed_old = self.refresh('b', 2)
        self.assertTrue(resumed.resume_required)
        self.assertEqual(resumed_target, target)
        staged = resumed.documents(resumed_old, resumed_target, [1, 2])
        self.assertEqual(staged[1]['description'], 'updated shelter metadata')
        self.assertEqual(staged[1]['image_vector'], source['image_vector'])
        self.assertEqual(staged[1]['animal_vector'], source['animal_vector'])
        self.assertEqual(staged[2], cached[2])
        self.assertEqual(self.fingerprints(base), before)
        session.publish(ALIAS, target, old, 2, lambda: None)
        after = self.fingerprints(base)
        self.assertEqual(after[0][4:], before[0][4:])
        self.assertEqual(after[1], before[1])
        self.assertEqual([hit['_source']['id'] for hit in self.search()], [2])
        self.assertEqual(self.store.documents(None, base, [1])[1]['status'], 'ADOPTED')

    def test_absent_animal_is_removed_only_after_every_source_id_is_verified(self):
        base = self.publish('a', [document(1), document(2), document(3)])
        before = self.fingerprints(base)
        metadata = self.build_state(base)
        session, target, old = self.refresh('b', 2)
        self.stage(session, target, old, [document(1)])
        self.assertEqual(session.count(target), 0)
        with self.assertRaisesRegex(RuntimeError, 'count|complete|seen'):
            session.publish(ALIAS, target, old, 2, lambda: None)
        self.assertEqual(self.fingerprints(base), before)
        self.assertEqual(self.build_state(base), metadata)
        self.assertEqual([hit['_source']['id'] for hit in self.search()], [1, 2, 3])
        self.stage(session, target, old, [document(3)])
        session.publish(ALIAS, target, old, 2, lambda: None)
        self.assertEqual(self.fingerprints(base), [before[0], before[2]])
        self.assertEqual([hit['_source']['id'] for hit in self.search()], [1, 3])
        self.assert_published(base, 'b', 2)

    def test_cancellation_before_commit_preserves_documents_metadata_and_pool(self):
        base = self.publish('a', [document(1), document(2)])
        before = self.fingerprints(base)
        metadata = self.build_state(base)
        session, target, old = self.refresh('b', 2)
        self.stage(session, target, old, [self.row(1, description='changed'), document(3)])
        checks = []
        def cancel_before_commit():
            checks.append(True)
            if len(checks) == 2:
                raise GalleryBuildCancelled('controlled stop after merge')
        with self.assertRaises(GalleryBuildCancelled):
            session.publish(ALIAS, target, old, 2, cancel_before_commit)
        self.assertEqual(len(checks), 2)
        self.assertEqual(self.fingerprints(base), before)
        self.assertEqual(self.build_state(base), metadata)
        self.assertEqual([hit['_source']['id'] for hit in self.search()], [1, 2])
        self.assertEqual(session.count(target), 2)
        self.assertEqual(self.pool.get_stats()['pool_available'], 1)
        session.publish(ALIAS, target, old, 2, lambda: None)
        self.assertEqual([hit['_source']['id'] for hit in self.search()], [1, 3])

    def test_sql_merge_failure_rolls_back_documents_and_metadata_then_can_retry(self):
        from psycopg import sql
        base = self.publish('a', [document(1), document(2)])
        before = self.fingerprints(base)
        metadata = self.build_state(base)
        session, target, old = self.refresh('b', 2)
        self.stage(session, target, old, [self.row(1, description='changed'), document(3)])
        with self.store.connection() as conn:
            conn.execute(
                'CREATE OR REPLACE FUNCTION pg_temp.fail_incremental_merge() RETURNS trigger LANGUAGE plpgsql AS $$ '
                'BEGIN IF NEW.build_key=TG_ARGV[0] AND NEW.animal_id=3 THEN '
                "RAISE EXCEPTION 'controlled incremental merge failure'; END IF; RETURN NEW; END $$"
            )
            conn.execute(sql.SQL(
                'CREATE TRIGGER incremental_merge_failure BEFORE INSERT ON lost_gallery_documents '
                'FOR EACH ROW EXECUTE FUNCTION pg_temp.fail_incremental_merge({})'
            ).format(sql.Literal(base)))
        try:
            with self.assertRaisesRegex(Exception, 'controlled incremental merge failure'):
                session.publish(ALIAS, target, old, 2, lambda: None)
            self.assertEqual(self.fingerprints(base), before)
            self.assertEqual(self.build_state(base), metadata)
            self.assertEqual([hit['_source']['id'] for hit in self.search()], [1, 2])
            self.assertEqual(session.count(target), 2)
            self.assertEqual(self.pool.get_stats()['pool_available'], 1)
        finally:
            with self.store.connection() as conn:
                conn.execute('DROP TRIGGER incremental_merge_failure ON lost_gallery_documents')
        session.publish(ALIAS, target, old, 2, lambda: None)
        self.assert_published(base, 'b', 2)
        self.assertEqual([hit['_source']['id'] for hit in self.search()], [1, 3])

    def test_second_session_rejects_changed_fingerprint_even_when_head_key_is_stable(self):
        base = self.publish('a', [document(1)])
        first, first_target, first_old = self.refresh('b', 1)
        other, other_target, other_old = self.refresh('c', 1)
        self.stage(first, first_target, first_old, [self.row(1, description='first publication')])
        self.stage(other, other_target, other_old, [self.row(1, description='stale publication')])
        first.publish(ALIAS, first_target, first_old, 1, lambda: None)
        committed = self.fingerprints(base)
        with self.assertRaisesRegex(RuntimeError, 'changed|conflict'):
            other.publish(ALIAS, other_target, other_old, 1, lambda: None)
        self.assertEqual(self.fingerprints(base), committed)
        self.assert_published(base, 'b', 1)

    def test_extra_alias_forces_full_generation_and_preserves_its_snapshot(self):
        base = self.publish('a', [document(1)])
        protected_alias = ALIAS + '-rollback'
        with self.store.connection() as conn:
            conn.execute('INSERT INTO lost_gallery_heads(alias,build_key) VALUES (%s,%s)', (protected_alias, base))
        before = self.fingerprints(base)
        session, target, old = self.refresh('b', 1)
        self.assertIs(session, self.store)
        self.assertEqual(target, gallery_target('b' * 64))
        self.stage(session, target, old, [document(2)])
        session.publish(ALIAS, target, old, 1, lambda: None)
        self.assertEqual(self.head(protected_alias), base)
        self.assertEqual(self.fingerprints(base), before)
        self.assertEqual(self.head(), target)

    def test_alias_added_after_preparation_rejects_in_place_publication(self):
        base = self.publish('a', [document(1)])
        before = self.fingerprints(base)
        metadata = self.build_state(base)
        session, target, old = self.refresh('b', 1)
        self.stage(session, target, old, [document(2)])
        protected_alias = ALIAS + '-rollback'
        with self.store.connection() as conn:
            conn.execute('INSERT INTO lost_gallery_heads(alias,build_key) VALUES (%s,%s)', (protected_alias, base))
        with self.assertRaisesRegex(RuntimeError, 'exclusive|alias'):
            session.publish(ALIAS, target, old, 1, lambda: None)
        self.assertEqual(self.head(protected_alias), base)
        self.assertEqual(self.fingerprints(base), before)
        self.assertEqual(self.build_state(base), metadata)
        self.assertEqual([hit['_source']['id'] for hit in self.search()], [1])
        self.assertEqual(session.count(target), 1)
        self.assertEqual(session.publication_stats, {})

    def test_search_keeps_old_rows_when_same_key_is_refreshed_between_its_queries(self):
        from contextlib import contextmanager
        from types import SimpleNamespace
        from unittest.mock import patch

        base = self.publish('a', [document(1)])
        # A separate guarded pool publishes while the reader holds its only
        # connection, matching actual concurrent service sessions.
        writer = SimpleNamespace()
        gallery_fixture.PostgresqlGalleryTest.setUpClass.__func__(writer)
        try:
            session, target = writer.store.refresh_store(ALIAS, gallery_target('b' * 64), 'b' * 64, 1)
            old = session.begin(ALIAS, target, 'b' * 64, 1)
            self.stage(session, target, old, [document(2)])
            original = self.store.connection
            publications = []

            @contextmanager
            def intercept(**kwargs):
                with original(**kwargs) as connection:
                    class Proxy:
                        def execute(inner, sql, *args):
                            result = connection.execute(sql, *args)
                            if sql.startswith('SELECT b.metadata,b.completed FROM lost_gallery_heads'):
                                session.publish(ALIAS, target, old, 1, lambda: None)
                                publications.append(True)
                            return result
                    yield Proxy()

            with patch.object(self.store, 'connection', intercept):
                hits = self.search()
            self.assertEqual(publications, [True])
            self.assertEqual([hit['_source']['id'] for hit in hits], [1])
            self.assertEqual([hit['_source']['id'] for hit in self.search()], [2])
            self.assert_published(base, 'b', 1)
        finally:
            gallery_fixture.PostgresqlGalleryTest.tearDownClass.__func__(writer)

    def test_model_and_color_version_mismatches_force_full_generation(self):
        for field in ('model_version', 'coat_color_version', 'metadata_version'):
            with self.subTest(field=field):
                self.setUp()
                base = self.publish('a', [document(1)])
                with self.store.connection() as conn:
                    conn.execute(
                        'UPDATE lost_gallery_builds SET metadata=jsonb_set(metadata,ARRAY[%s],%s::jsonb) '
                        'WHERE build_key=%s', (field, '"previous-version"', base)
                    )
                session, target, old = self.refresh('b', 1)
                self.assertIs(session, self.store)
                self.assertEqual(target, gallery_target('b' * 64))
                self.assertEqual(old, base)

    def test_source_hash_reversion_refreshes_the_stable_key_with_original_content(self):
        original = document(1)
        base = self.publish('a', [original])
        changed = self.row(1, source_sha256='b' * 64, image_vector=[0., 1.] + [0.] * 1022)
        session, target, old = self.refresh('b', 1)
        self.stage(session, target, old, [changed])
        session.publish(ALIAS, target, old, 1, lambda: None)
        reverse, reverse_target, reverse_old = self.refresh('a', 1)
        self.assertEqual(reverse_old, base)
        self.assertNotEqual(reverse_target, base)
        self.stage(reverse, reverse_target, reverse_old, [original])
        reverse.publish(ALIAS, reverse_target, reverse_old, 1, lambda: None)
        self.assertEqual(self.store.documents(None, base, [1])[1], original)
        self.assert_published(base, 'a', 1)

    def test_protected_alias_hash_reversion_uses_distinct_full_target_without_overwriting_base(self):
        original = document(1)
        base = self.publish('a', [original])
        session, target, old = self.refresh('b', 1)
        self.stage(session, target, old, [self.row(1, description='protected newer snapshot')])
        session.publish(ALIAS, target, old, 1, lambda: None)
        protected_alias = ALIAS + '-rollback'
        with self.store.connection() as conn:
            conn.execute('INSERT INTO lost_gallery_heads(alias,build_key) VALUES (%s,%s)', (protected_alias, base))
        protected_rows = self.fingerprints(base)
        protected_metadata = self.build_state(base)

        full, full_target, full_old = self.refresh('a', 1)
        self.assertIs(full, self.store)
        self.assertEqual(full_old, base)
        self.assertNotEqual(full_target, gallery_target('a' * 64))
        self.assertEqual(self.store.build_target(gallery_target('a' * 64), 'a' * 64, 1), full_target)
        self.stage(full, full_target, full_old, [original])
        full.publish(ALIAS, full_target, full_old, 1, lambda: None)

        self.assertEqual(self.head(protected_alias), base)
        self.assertEqual(self.fingerprints(base), protected_rows)
        self.assertEqual(self.build_state(base), protected_metadata)
        self.assertEqual(self.store.documents(None, base, [1])[1]['description'], 'protected newer snapshot')
        self.assertEqual(self.store.documents(None, full_target, [1])[1], original)
        self.assert_published(full_target, 'a', 1)

    def test_committed_publication_retry_recognizes_current_snapshot_without_new_delta(self):
        base = self.publish('a', [document(1)])
        session, target, old = self.refresh('b', 1)
        self.stage(session, target, old, [self.row(1, description='committed')])
        session.publish(ALIAS, target, old, 1, lambda: None)
        before = self.fingerprints(base)
        metadata = self.build_state(base)
        with self.audited_document_writes():
            retry, retry_target, retry_old = self.refresh('b', 1)
            self.assertEqual(retry_old, base)
            self.assertTrue(retry.completed(retry_target, 1))
            self.assertTrue(retry.already_current)
            self.assertEqual(retry.count(retry_target), 0)
            self.assertEqual(retry.result_index(retry_target), base)
            retry.publish(ALIAS, retry_target, retry_old, 1, lambda: None)
            self.assertEqual(self.writes(), [])
        self.assertEqual(self.fingerprints(base), before)
        self.assertEqual(self.build_state(base), metadata)
        with self.store.connection() as conn:
            self.assertEqual(conn.execute('SELECT count(*) FROM lost_gallery_builds').fetchone(), (1,))


if __name__ == '__main__':
    unittest.main()
