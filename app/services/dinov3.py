"""Pinned, local-only DINOv3 encoder for the isolated lost-animal gallery."""
from dataclasses import dataclass
import hashlib
import math
import os
import re
import threading
from pathlib import Path

MODEL_NAME = "vit_large_patch16_dinov3.lvd1689m"
MODEL_VERSION = "dinov3-large-30c1109-avg-256-center-fp16-l2-v1"
WEIGHTS_SHA256 = "45172f209c9583c40538afc26b60a07033e6fcc2e8c30228338e6b2e932e7941"
DIMENSIONS = 1024
_encoder = None
_load_lock = threading.Lock()


def visual_profile():
    profile = os.getenv("LOST_SEARCH_VISUAL_PROFILE", "original")
    if profile not in {"original", "animal-focus", "sam3-animal-focus"}:
        raise RuntimeError("Unknown lost-search visual profile")
    return profile


@dataclass
class AnimalEmbedding:
    vector: list[float]
    model_version: str
    focus_status: str
    animal_vector: list[float] | None = None


def gallery_index():
    index = os.getenv("LOST_SEARCH_INDEX", "animals-lost-dinov3-large-v1")
    if not re.fullmatch(r"animals-lost-dinov3-[a-z0-9][a-z0-9-]{0,80}", index):
        raise RuntimeError("A separate DINOv3 gallery index is required")
    profile = visual_profile()
    index_profile = ("sam3-animal-focus" if index.startswith("animals-lost-dinov3-sam3-")
                     else "animal-focus" if index.startswith("animals-lost-dinov3-focus-")
                     else "original")
    if profile != index_profile:
        raise RuntimeError("Visual profile requires its own matching gallery index")
    return index


def validate_vector(vector):
    if len(vector) != DIMENSIONS or not all(math.isfinite(v) for v in vector):
        raise RuntimeError("Invalid DINOv3 output")
    if abs(sum(v * v for v in vector) - 1.0) > 0.001:
        raise RuntimeError("DINOv3 output is not normalized")
    return vector


class DinoV3Encoder:
    def __init__(self):
        import torch
        import timm
        if not torch.cuda.is_available():
            raise RuntimeError("DINOv3 requires CUDA; CPU fallback is disabled")
        checkpoint = Path(os.environ["DINOV3_CHECKPOINT"])
        with checkpoint.open("rb") as source:
            digest = hashlib.sha256()
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != WEIGHTS_SHA256:
            raise RuntimeError("DINOv3 checkpoint hash mismatch")
        torch.set_num_threads(2)
        total = torch.cuda.get_device_properties(0).total_memory
        limit_gib = 7.0 if visual_profile() == "sam3-animal-focus" else 5.5
        torch.cuda.set_per_process_memory_fraction(min(1.0, limit_gib * 1024**3 / total), 0)
        self.focus = None
        self.model_version = MODEL_VERSION
        # SAM first avoids retaining DINO's CPU allocation during SAM construction.
        if visual_profile() == "sam3-animal-focus":
            from app.services.sam3_focus import Sam3Focus, FOCUS_VERSION
            self.focus = Sam3Focus()
            self.model_version = FOCUS_VERSION
        self.model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=0,
                                      checkpoint_path=str(checkpoint)).eval().to("cuda")
        config = timm.data.resolve_model_data_config(self.model)
        expected = {"input_size": (3, 256, 256), "interpolation": "bicubic",
                    "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225),
                    "crop_pct": 1.0, "crop_mode": "center"}
        if config != expected or self.model.global_pool != "avg":
            raise RuntimeError("DINOv3 preprocessing or pooling changed")
        self.transform = timm.data.create_transform(**config, is_training=False)
        if visual_profile() == "animal-focus":
            from app.services.animal_focus import AnimalFocus, FOCUS_VERSION
            self.focus = AnimalFocus()
            self.model_version = FOCUS_VERSION
        from app.services.inference_gate import InferenceGate
        self.gate = InferenceGate()

    def encode(self, image, species=None):
        return self.encode_with_metadata(image, species).vector

    def _vector_for(self, view):
        import torch
        import torch.nn.functional as functional
        tensor = self.transform(view).unsqueeze(0).to("cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            vector = functional.normalize(self.model(tensor).float(), dim=-1)
        return validate_vector(vector.squeeze(0).cpu().tolist())

    def encode_with_metadata(self, image, species=None, *, background=False):
        # Hold one lock across detection and both embeddings; no concurrent GPU jobs.
        import torch
        with self.gate.acquire(background=background):
            try:
                return self._encode_image(image, species)
            except torch.cuda.OutOfMemoryError:
                pass
            # Leave the exception scope first so failed inference tensors are released.
            torch.cuda.empty_cache()
            raise RuntimeError("GPU memory exhausted during lost-animal search")

    def _encode_image(self, image, species):
        if self.focus is None:
            return AnimalEmbedding(self._vector_for(image), self.model_version, "original_profile")
        from app.services.animal_focus import square_image
        # Always preserve the same full-photo channel, even if segmentation changes
        # after JPEG recompression or is ambiguous on only one side of the search.
        with square_image(image) as original:
            full_vector = self._vector_for(original)
        prepared = self.focus.prepare(image, species)
        try:
            animal_vector = self._vector_for(prepared.image) if prepared.status == "animal_mask" else None
            return AnimalEmbedding(full_vector, self.model_version, prepared.status, animal_vector)
        finally:
            prepared.image.close()


def get_encoder():
    global _encoder
    with _load_lock:
        if _encoder is None:
            _encoder = DinoV3Encoder()
        return _encoder
