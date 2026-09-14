import os
import unittest
from unittest.mock import patch
from app.services.dinov3 import gallery_index, validate_vector, visual_profile


class DinoV3ContractTest(unittest.TestCase):
    def test_gallery_cannot_point_at_existing_animals_or_multiple_indices(self):
        for index in ("animals", "animals-*", "animals-lost-dinov3-v1,animals", ""):
            with self.subTest(index=index), patch.dict(os.environ, {"LOST_SEARCH_INDEX": index}):
                with self.assertRaises(RuntimeError):
                    gallery_index()
        with patch.dict(os.environ, {"LOST_SEARCH_INDEX": "animals-lost-dinov3-eval-v1"}):
            self.assertEqual(gallery_index(), "animals-lost-dinov3-eval-v1")

    def test_output_requires_1024_finite_normalized_values(self):
        valid = [1.0] + [0.0] * 1023
        self.assertEqual(validate_vector(valid), valid)
        for vector in ([0.0]*384, [0.0]*1024, [float("nan")]+valid[1:], [2.0]+valid[1:]):
            with self.subTest(length=len(vector)), self.assertRaises(RuntimeError):
                validate_vector(vector)


class FocusProfileTest(unittest.TestCase):
    def test_focus_requires_a_separate_index_and_cannot_reuse_original_vectors(self):
        with patch.dict(os.environ, {"LOST_SEARCH_VISUAL_PROFILE": "animal-focus", "LOST_SEARCH_INDEX": "animals-lost-dinov3-eval-v1"}):
            with self.assertRaises(RuntimeError):
                gallery_index()
        with patch.dict(os.environ, {"LOST_SEARCH_VISUAL_PROFILE": "animal-focus", "LOST_SEARCH_INDEX": "animals-lost-dinov3-focus-eval-v1"}):
            self.assertEqual(gallery_index(), "animals-lost-dinov3-focus-eval-v1")
        with patch.dict(os.environ, {"LOST_SEARCH_VISUAL_PROFILE": "original", "LOST_SEARCH_INDEX": "animals-lost-dinov3-focus-eval-v1"}):
            with self.assertRaises(RuntimeError):
                gallery_index()
        with patch.dict(os.environ, {"LOST_SEARCH_VISUAL_PROFILE": "typo"}):
            with self.assertRaises(RuntimeError):
                visual_profile()


class DualEmbeddingTest(unittest.TestCase):
    def test_full_photo_channel_is_identical_when_focus_succeeds_or_falls_back(self):
        from types import SimpleNamespace
        from PIL import Image
        from app.services.dinov3 import DinoV3Encoder
        from app.services.animal_focus import FocusResult, FOCUS_VERSION
        full_views = []
        for status in ["animal_mask", "original_multiple_animals", "original_suspect_mask"]:
            with self.subTest(status=status), Image.new("RGB", (40, 80), "red") as image:
                encoder = object.__new__(DinoV3Encoder)
                from app.services.inference_gate import InferenceGate
                encoder.gate = InferenceGate()
                encoder.model_version = FOCUS_VERSION
                encoder.focus = SimpleNamespace(prepare=lambda *_: FocusResult(Image.new("RGB", (256,256), "blue"), status))
                calls = []
                def vector_for(view):
                    calls.append(view.tobytes())
                    return [1.,0.] if len(calls) == 1 else [0.,1.]
                encoder._vector_for = vector_for
                result = encoder.encode_with_metadata(image, "DOG")
                full_views.append(calls[0])
                self.assertEqual(result.vector, [1.,0.])
                self.assertEqual(result.animal_vector, [0.,1.] if status == "animal_mask" else None)
                self.assertEqual(result.focus_status, status)
                self.assertEqual(result.model_version, FOCUS_VERSION)
                self.assertEqual(len(calls), 2 if status == "animal_mask" else 1)
        self.assertEqual(full_views[0], full_views[1])
        self.assertEqual(full_views[1], full_views[2])
