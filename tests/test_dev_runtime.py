import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('check_dev_runtime', Path(__file__).resolve().parents[1]/'scripts/check_dev_runtime.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class DevRuntimeTest(unittest.TestCase):
    def test_dedicated_configuration_and_rejected_production_fallbacks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = {'PAWBRIDGE_ENVIRONMENT': 'dev', 'LOST_STORAGE_BACKEND': 'postgresql', 'LOST_PG_DSN': 'postgresql://pawbridge_dev_vector@127.0.0.1:15433/pawbridge', 'LOST_SEARCH_INDEX': 'animals-lost-dinov3-sam3-dev-v1', 'LOST_GALLERY_STATE_DIR': str(root/'state'), 'LOST_GALLERY_PHOTO_ROOT': str(root/'photos'), 'LOST_GALLERY_SYNC_ENABLED': 'false', 'LOST_DEV_GPU_ENABLED': 'true', 'INTERNAL_API_KEY': 'test-placeholder'}
            module.validate(env, root)
            for key, bad in [('LOST_PG_DSN', 'postgresql://pawbridge_vector@127.0.0.1:30432/pawbridge'), ('LOST_GALLERY_STATE_DIR', '/home/shyu/pawbridge-ai/state'), ('LOST_GALLERY_SYNC_ENABLED', 'true'), ('LOST_DEV_GPU_ENABLED', 'false'), ('LOST_SEARCH_INDEX', 'animals-lost-dinov3-sam3-v1')]:
                with self.subTest(key=key), self.assertRaises(ValueError): module.validate({**env, key: bad}, root)
