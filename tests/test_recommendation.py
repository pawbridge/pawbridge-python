import os
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.routers.recommendation import router
from app.services.recommendation import recommend_animals
from app.services.pg_gallery_store import GalleryUnavailable


class RecommendationTest(unittest.TestCase):
    def test_authenticated_route_preserves_id_order_and_validates_species(self):
        app = FastAPI()
        app.include_router(router, prefix='/internal/animals')
        with patch.dict(os.environ, {'INTERNAL_API_KEY': 'test-key'}), TestClient(app) as client:
            with patch('app.routers.recommendation.recommend_animals', return_value=[9, 4]) as query:
                self.assertEqual(client.get('/internal/animals/73/similar?species=DOG').status_code, 401)
                query.assert_not_called()
                headers = {'X-Internal-Api-Key': 'test-key'}
                response = client.get('/internal/animals/73/similar?species=DOG', headers=headers)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), [9, 4])
                query.assert_called_once_with(73, 'DOG')
                self.assertEqual(client.get('/internal/animals/73/similar?species=ETC', headers=headers).status_code, 422)
                self.assertEqual(client.get('/internal/animals/0/similar?species=DOG', headers=headers).status_code, 422)

    def test_unprepared_gallery_and_database_failures_are_503_without_exposing_details(self):
        app = FastAPI()
        app.include_router(router, prefix='/internal/animals')
        with patch.dict(os.environ, {'INTERNAL_API_KEY': 'test-key'}), TestClient(app) as client:
            with patch('app.routers.recommendation.recommend_animals', side_effect=RuntimeError('private database detail')) as query:
                response = client.get('/internal/animals/1/similar?species=CAT', headers={'X-Internal-Api-Key': 'test-key'})
                self.assertEqual(response.status_code, 503)
                self.assertNotIn('private', response.text)
                app.state.gallery_refresh = SimpleNamespace(ready=False)
                query.reset_mock()
                self.assertEqual(client.get('/internal/animals/1/similar?species=CAT', headers={'X-Internal-Api-Key': 'test-key'}).status_code, 503)
                query.assert_not_called()

    def test_stored_color_reranking_and_six_result_bound_use_no_encoder(self):
        import numpy as np
        from PIL import Image
        from app.services.coat_color import describe
        with Image.new('RGB', (20, 20), 'red') as red, Image.new('RGB', (20, 20), 'blue') as blue:
            mask = np.ones((20, 20), dtype=bool)
            source_color, different_color = describe(red, mask), describe(blue, mask)
        hits = [{'_source': {'id': i, 'coat_color': source_color}, '_score': 1.94} for i in range(2, 10)]
        hits.insert(0, {'_source': {'id': 10, 'coat_color': different_color}, '_score': 1.95})
        store = Mock()
        store.recommendation_candidates.return_value = ({'coat_color': source_color}, hits)
        env = {'LOST_STORAGE_BACKEND': 'postgresql', 'LOST_SEARCH_COAT_COLOR_WEIGHT': '0.1',
               'LOST_SEARCH_VISUAL_PROFILE': 'sam3-animal-focus', 'LOST_SEARCH_INDEX': 'animals-lost-dinov3-sam3-test'}
        with patch.dict(os.environ, env), patch('app.services.recommendation.get_postgresql_store', return_value=store),                 patch('app.services.dinov3.get_encoder', side_effect=AssertionError('No inference permitted')):
            self.assertEqual(recommend_animals(1, 'DOG'), [2, 3, 4, 5, 6, 7])
            store.recommendation_candidates.assert_called_once()
            self.assertEqual(store.recommendation_candidates.call_args.args[1:3], (1, 'DOG'))
        with patch.dict(os.environ, {'LOST_STORAGE_BACKEND': 'elasticsearch'}), self.assertRaises(GalleryUnavailable):
            recommend_animals(1, 'DOG')

    def test_visual_floor_is_applied_before_reranking_truncates_candidates(self):
        import numpy as np
        from PIL import Image
        from app.services.coat_color import describe
        with Image.new('RGB', (20, 20), 'red') as red, Image.new('RGB', (20, 20), 'blue') as blue:
            mask = np.ones((20, 20), dtype=bool)
            query_color, other_color = describe(red, mask), describe(blue, mask)
        store = Mock()
        hits = [{'_source': {'id': i, 'coat_color': query_color}, '_score': 1.59} for i in range(2, 25)]
        hits.append({'_source': {'id': 30, 'coat_color': other_color}, '_score': 1.61})
        store.recommendation_candidates.return_value = ({'coat_color': query_color}, hits)
        env = {'LOST_STORAGE_BACKEND': 'postgresql', 'LOST_SEARCH_COAT_COLOR_WEIGHT': '0.1',
               'LOST_SEARCH_VISUAL_PROFILE': 'sam3-animal-focus', 'LOST_SEARCH_INDEX': 'animals-lost-dinov3-sam3-test'}
        with patch.dict(os.environ, env), patch('app.services.recommendation.get_postgresql_store', return_value=store):
            self.assertEqual(recommend_animals(1, 'DOG'), [30])

    def test_postgresql_chatbot_entry_point_does_not_load_or_expose_legacy_embedding(self):
        script = """import sys
from app.main import app
assert 'app.services.embedding' not in sys.modules
assert 'app.es.client' not in sys.modules
paths = {r.path for r in app.routes}
assert '/api/v1/animals/similar' not in paths
assert '/api/v1/animals/batch/embeddings' not in paths
assert '/internal/chatbot/messages' in paths
"""
        result = subprocess.run([sys.executable, '-c', script], env={**os.environ, 'LOST_STORAGE_BACKEND': 'postgresql'},
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
