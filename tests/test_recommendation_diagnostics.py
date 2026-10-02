import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.routers.recommendation import router


class RecommendationDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {'INTERNAL_API_KEY': 'test-internal-key'})
        env.start()
        self.addCleanup(env.stop)
        self.app = FastAPI()
        self.app.include_router(router, prefix='/internal/animals')
        self.client = TestClient(self.app)
        self.addCleanup(self.client.close)

    def request(self):
        return self.client.get('/internal/animals/73/similar?species=DOG',
                               headers={'X-Internal-Api-Key': 'test-internal-key'})

    def test_unready_gallery_logs_reason_without_querying_storage(self):
        self.app.state.gallery_refresh = SimpleNamespace(ready=False)
        with patch('app.routers.recommendation.recommend_animals') as query:
            with self.assertLogs('app.routers.recommendation', level='WARNING') as logs:
                response = self.request()
        self.assertEqual(response.status_code, 503)
        query.assert_not_called()
        self.assertIn('animal_id=73', logs.output[0])
        self.assertIn('reason=GALLERY_NOT_READY', logs.output[0])
        self.assertNotIn('test-internal-key', logs.output[0])

    def test_database_failure_logs_type_without_message_or_traceback(self):
        failure = RuntimeError('private-database-details token=private-test-token')
        with patch('app.routers.recommendation.recommend_animals', side_effect=failure):
            with self.assertLogs('app.routers.recommendation', level='WARNING') as logs:
                response = self.request()
        self.assertEqual(response.status_code, 503)
        self.assertIn('animal_id=73', logs.output[0])
        self.assertIn('reason=QUERY_FAILED', logs.output[0])
        self.assertIn('exception_type=RuntimeError', logs.output[0])
        self.assertNotIn('private', logs.output[0])
        self.assertNotIn('test-internal-key', logs.output[0])
        self.assertNotIn('private', response.text)
        self.assertIsNone(logs.records[0].exc_info)

    def test_success_keeps_candidate_order_and_emits_no_failure_log(self):
        with patch('app.routers.recommendation.recommend_animals', return_value=[9, 4]):
            with self.assertNoLogs('app.routers.recommendation', level='WARNING'):
                response = self.request()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [9, 4])

    def test_source_features_and_gallery_contract_failures_have_distinct_safe_reasons(self):
        from app.services.pg_gallery_store import RecommendationGalleryUnavailable, RecommendationSourceUnavailable
        for exception_type, reason in [(RecommendationSourceUnavailable, 'SOURCE_FEATURES_NOT_READY'),
                                       (RecommendationGalleryUnavailable, 'GALLERY_CONTRACT_UNAVAILABLE')]:
            with self.subTest(reason=reason):
                with patch('app.routers.recommendation.recommend_animals',
                           side_effect=exception_type('private-database-details')):
                    with self.assertLogs('app.routers.recommendation', level='WARNING') as logs:
                        response = self.request()
                self.assertEqual(response.status_code, 503)
                self.assertIn(f'reason={reason}', logs.output[0])
                self.assertNotIn('private', logs.output[0])
                self.assertNotIn('private', response.text)

    def test_storage_guards_classify_missing_gallery_and_missing_source(self):
        from unittest.mock import Mock
        from app.services.lost_gallery import CONTRACT
        from app.services.pg_gallery_store import (PostgresqlGalleryStore,
            RecommendationGalleryUnavailable, RecommendationSourceUnavailable)
        store = PostgresqlGalleryStore(Mock())
        with patch.object(store, 'connection') as connection:
            query = connection.return_value.__enter__.return_value.execute
            query.return_value.fetchone.return_value = None
            with self.assertRaises(RecommendationGalleryUnavailable):
                store.recommendation_candidates('animals-lost-test', 73, 'DOG', 'model-test')
            head = ('build-test', {'contract': CONTRACT, 'model_version': 'model-test'}, True)
            query.return_value.fetchone.side_effect = [head, None]
            with self.assertRaises(RecommendationSourceUnavailable):
                store.recommendation_candidates('animals-lost-test', 73, 'DOG', 'model-test')
