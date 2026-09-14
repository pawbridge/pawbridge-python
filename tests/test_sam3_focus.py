import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
# CPU image CI intentionally omits PyTorch. These two runtime contracts are
# exercised without skips in the dedicated CUDA environment.
try:
    import torch
except ModuleNotFoundError as error:
    if error.name != 'torch':
        raise
    torch = None
from PIL import Image
from app.services.sam3_focus import prepare_prediction, load_mapped_checkpoint
from app.services.dinov3 import DinoV3Encoder, gallery_index


class Sam3ContractTest(unittest.TestCase):
    def test_each_profile_rejects_the_other_profiles_gallery(self):
        indices = {"original": "animals-lost-dinov3-eval-v1",
                   "animal-focus": "animals-lost-dinov3-focus-eval-v1",
                   "sam3-animal-focus": "animals-lost-dinov3-sam3-eval-v1"}
        for profile in indices:
            for owner, index in indices.items():
                with self.subTest(profile=profile, index=index), patch.dict(os.environ, {
                        "LOST_SEARCH_VISUAL_PROFILE": profile, "LOST_SEARCH_INDEX": index}):
                    if profile == owner:
                        self.assertEqual(gallery_index(), index)
                    else:
                        with self.assertRaises(RuntimeError):
                            gallery_index()

    def test_single_animal_masks_background_and_preserves_original(self):
        with Image.new("RGB", (120, 120), "red") as image:
            original = image.tobytes()
            mask = np.zeros((1, 120, 120), dtype=bool)
            mask[0, 20:100, 20:100] = True
            result = prepare_prediction(image, [[10, 10, 110, 110]], mask, [.9])
            try:
                self.assertEqual(result.status, "animal_mask")
                self.assertEqual(result.image.size, (256, 256))
                self.assertEqual(result.image.getpixel((0, 0)), (124, 116, 104))
                self.assertEqual(image.tobytes(), original)
            finally:
                result.image.close()

    def test_missing_and_multiple_detections_keep_full_photo(self):
        with Image.new("RGB", (120, 120), "red") as image:
            for count, expected in [(0, "original_no_confident_animal"),
                                    (2, "original_multiple_animals")]:
                with self.subTest(count=count):
                    result = prepare_prediction(image, np.zeros((count, 4)),
                                                np.zeros((count, 120, 120)), np.ones(count))
                    try:
                        self.assertEqual(result.status, expected)
                        self.assertEqual(result.image.getpixel((0, 0)), (255, 0, 0))
                    finally:
                        result.image.close()

    def test_malformed_model_output_is_failure_not_no_detection(self):
        with Image.new("RGB", (120, 120)) as image:
            for boxes, masks, scores in [([[0, 0, 100, 100]], np.zeros((0, 120, 120)), []),
                                         ([[0, 0, 100, 100]], np.zeros((1, 120, 120)), [np.nan]),
                                         ([[0, 0, 100, 100]], np.full((1, 120, 120), 2), [.9])]:
                with self.subTest(scores=scores), self.assertRaises(RuntimeError):
                    prepare_prediction(image, boxes, masks, scores)

    @unittest.skipIf(torch is None, "Requires the dedicated PyTorch runtime")
    def test_mapped_loader_requires_all_active_keys_and_rejects_unknown_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pt"
            source = {"detector.weight": torch.full((2, 2), .5),
                      "detector.bias": torch.full((2,), .2),
                      "detector.backbone.vision_backbone.sam2_convs.0.weight": torch.ones(1)}
            model = torch.nn.Linear(2, 2)
            torch.save({"model": source}, path)
            load_mapped_checkpoint(model, path)
            self.assertTrue(torch.equal(model.weight, source["detector.weight"]))
            self.assertTrue(torch.equal(model.bias, source["detector.bias"]))
            for changed in [{k: v for k, v in source.items() if k != "detector.bias"},
                            dict(source, **{"detector.unexpected": torch.ones(1)})]:
                torch.save(changed, path)
                with self.assertRaises(RuntimeError):
                    load_mapped_checkpoint(model, path)

    @unittest.skipIf(torch is None, "Requires the dedicated PyTorch runtime")
    def test_gpu_oom_releases_lock_and_next_request_can_run(self):
        encoder = object.__new__(DinoV3Encoder)
        from app.services.inference_gate import InferenceGate
        encoder.gate = InferenceGate()
        with patch.object(encoder, "_encode_image", side_effect=[torch.cuda.OutOfMemoryError(), "ok"]), patch.object(torch.cuda, "empty_cache") as clear:
            with self.assertRaisesRegex(RuntimeError, "GPU memory exhausted"):
                encoder.encode_with_metadata(None, "DOG")
            self.assertFalse(encoder.gate.active)
            clear.assert_called_once_with()
            self.assertEqual(encoder.encode_with_metadata(None, "DOG"), "ok")
