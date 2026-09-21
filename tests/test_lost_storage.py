import io
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from PIL import Image
from app.services import lost_storage
from app.services.lost_search import search_photo
from app.services.pg_gallery_store import PostgresqlGalleryStore


class LostStorageTest(unittest.TestCase):
    def test_default_storage_does_not_create_a_postgresql_pool(self):
        with patch.dict(os.environ,{},clear=True), patch.object(lost_storage,'get_postgresql_store') as factory:
            with lost_storage.storage_session() as store:
                self.assertIsNone(store)
            factory.assert_not_called()

    def test_unknown_backend_fails_closed_instead_of_falling_back(self):
        with patch.dict(os.environ,{'LOST_STORAGE_BACKEND':'misspelled'}):
            with self.assertRaises(RuntimeError):
                with lost_storage.storage_session(): self.fail('Cannot serve an unknown backend')

    def test_startup_failure_closes_created_pool(self):
        store=Mock()
        with patch.dict(os.environ,{'LOST_STORAGE_BACKEND':'postgresql'}), \
                patch.object(lost_storage,'get_postgresql_store',return_value=store), \
                patch.object(lost_storage,'_store',store):
            with self.assertRaises(RuntimeError):
                with lost_storage.storage_session(): raise RuntimeError('startup failed')
            store.pool.close.assert_called_once()
            self.assertIsNone(lost_storage._store)

    def test_postgresql_retrieval_keeps_auxiliary_ranking_and_resolved_option(self):
        image=io.BytesIO(); Image.new('RGB',(8,8),'white').save(image,format='PNG')
        calls=[]
        def encode(*args):
            calls.append('inference')
            return SimpleNamespace(vector=[1.]+[0.]*1023,animal_vector=None,model_version='test',focus_status='test',coat_color=None)
        def retrieve(*args):
            self.assertEqual(calls,['inference'])
            self.assertEqual(args[3],('NOTICE','PROTECT','ADOPTED','RETURNED'))
            self.assertEqual(args[4],200)
            return [{'_score':1.8,'_source':{'id':1}},
                    {'_score':1.795,'_source':{'id':2,'happen_place':'서울'}}]
        store=Mock();store.search.side_effect=retrieve
        with patch.dict(os.environ,{'LOST_STORAGE_BACKEND':'postgresql','LOST_SEARCH_COAT_COLOR_WEIGHT':'0'}), \
                patch('app.services.dinov3.get_encoder',return_value=SimpleNamespace(encode_with_metadata=encode)), \
                patch('app.services.dinov3.gallery_index',return_value='test'), \
                patch.object(lost_storage,'get_postgresql_store',return_value=store):
            result=search_photo(image.getvalue(),'DOG',region='서울',include_adopted_or_returned=True)
        self.assertEqual([row['animalId'] for row in result['candidates']],[2,1])

    def test_corrupt_vector_cannot_reach_database(self):
        pool=Mock();store=PostgresqlGalleryStore(pool)
        for vector in ([0.]*1024,[1.]*384,[float('nan')]*1024):
            with self.subTest(length=len(vector)), self.assertRaises(RuntimeError):
                store.search('test',SimpleNamespace(vector=vector,animal_vector=None),'DOG',('PROTECT',))
        pool.connection.assert_not_called()
