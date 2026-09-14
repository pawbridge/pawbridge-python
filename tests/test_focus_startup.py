import copy
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch
from app.lost_main import validate_focus_gallery


class FocusStartupTest(unittest.TestCase):
    def test_only_populated_matching_dual_vector_gallery_can_start(self):
        index = "animals-lost-dinov3-focus-eval-v2"
        mapping = {"_meta": {"model_version": "dual-v2"}, "properties": {
            "image_vector": {"type": "dense_vector", "dims": 1024},
            "animal_vector": {"type": "dense_vector", "dims": 1024}}}
        bad_version = copy.deepcopy(mapping); bad_version["_meta"]["model_version"] = "old"
        bad_dims = copy.deepcopy(mapping); bad_dims["properties"]["animal_vector"]["dims"] = 384
        module = types.ModuleType("app.es.client"); module.es = MagicMock()
        client = module.es.options.return_value
        with patch.dict(os.environ, {"LOST_SEARCH_VISUAL_PROFILE":"animal-focus", "LOST_SEARCH_INDEX":index}), patch.dict(sys.modules, {"app.es.client":module}):
            for schema, count in [(bad_version,1),(bad_dims,1),(mapping,0)]:
                client.indices.get_mapping.return_value = {index:{"mappings":schema}}
                client.count.return_value = {"count": count}
                with self.assertRaises(RuntimeError):
                    validate_focus_gallery(types.SimpleNamespace(model_version="dual-v2"))
            client.indices.get_mapping.return_value = {index:{"mappings":mapping}}
            client.count.return_value = {"count": 200}
            validate_focus_gallery(types.SimpleNamespace(model_version="dual-v2"))

    def test_sam3_lifespan_checks_gallery_before_accepting_requests(self):
        import asyncio
        from unittest.mock import AsyncMock
        from app.lost_main import lifespan
        async def enter():
            async with lifespan(None):
                self.fail("Invalid SAM 3 gallery must not become ready")
        with patch.dict(os.environ, {"INTERNAL_API_KEY": "test-key", "LOST_SEARCH_VISUAL_PROFILE": "sam3-animal-focus", "LOST_SEARCH_INDEX": "animals-lost-dinov3-sam3-eval-v1"}), patch("app.lost_main.anyio.to_thread.run_sync", new_callable=AsyncMock, side_effect=[object(), RuntimeError("gallery mismatch")]) as run:
            with self.assertRaisesRegex(RuntimeError, "gallery mismatch"):
                asyncio.run(enter())
            self.assertEqual(run.call_count, 2)
