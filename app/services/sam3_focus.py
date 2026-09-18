"""Pinned SAM 3 image-only segmentation; no model downloads or image synthesis."""
from contextlib import nullcontext
import hashlib
import os
from pathlib import Path

from app.services.animal_focus import FocusResult, prepare_focus, square_image

CHECKPOINT_SHA256 = "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
FOCUS_VERSION = "dinov3-large-sam39999e234-dual-pad256-v1"


def load_mapped_checkpoint(model, checkpoint_path):
    import torch
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    if "model" in checkpoint and isinstance(checkpoint["model"], dict):
        checkpoint = checkpoint["model"]
    state = {key.removeprefix("detector."): value for key, value in checkpoint.items()
             if key.startswith("detector.")}
    required = set(model.state_dict())
    extra = set(state) - required
    # The pinned checkpoint also carries inactive interactive-predictor convolutions.
    if not required <= set(state) or any(
            not key.startswith("backbone.vision_backbone.sam2_convs.") for key in extra):
        raise RuntimeError("SAM 3 checkpoint does not match the image-only model")
    model.load_state_dict({key: state[key] for key in required}, strict=True, assign=True)


def prepare_prediction(image, boxes, masks, scores):
    import numpy as np
    boxes, masks, scores = np.asarray(boxes), np.asarray(masks), np.asarray(scores)
    if (scores.ndim != 1 or boxes.shape != (len(scores), 4)
            or masks.shape != (len(scores), image.height, image.width)
            or not np.isfinite(boxes).all() or not np.isfinite(scores).all()
            or not np.isfinite(masks).all()
            or (scores < 0).any() or (scores > 1).any()
            or (masks < 0).any() or (masks > 1).any()):
        raise RuntimeError("Invalid SAM 3 prediction")
    if len(scores) == 0:
        return FocusResult(square_image(image), "original_no_confident_animal")
    if len(scores) > 1:
        # Keep the identity embedding conservative (the full photo), but retain
        # color evidence from the primary foreground instead of silently making
        # every multi-detection candidate color-neutral. Mask area weighted by
        # confidence favors the main subject while the descriptor records that
        # it came from a primary, not unambiguous, mask.
        areas = (masks >= .5).sum(axis=(1, 2))
        selected = max(range(len(scores)), key=lambda i: (areas[i] * scores[i], scores[i]))
        box = boxes[selected].tolist()
        box[0], box[1] = max(0, box[0]), max(0, box[1])
        box[2], box[3] = min(image.width, box[2]), min(image.height, box[3])
        focused = prepare_focus(image, box, masks[selected].astype("float32"),
                                color_source="primary_mask")
        try:
            return FocusResult(square_image(image), "original_multiple_animals",
                               focused.coat_color)
        finally:
            focused.image.close()
    # SAM's floating point boxes can extend slightly past the image boundary.
    box = boxes[0].tolist()
    box[0], box[1] = max(0, box[0]), max(0, box[1])
    box[2], box[3] = min(image.width, box[2]), min(image.height, box[3])
    return prepare_focus(image, box, masks[0].astype("float32"))


class Sam3Focus:
    def __init__(self):
        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("SAM 3 requires a BF16-capable CUDA GPU")
        checkpoint = Path(os.environ["SAM3_CHECKPOINT"])
        digest = hashlib.sha256()
        with checkpoint.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != CHECKPOINT_SHA256:
            raise RuntimeError("SAM 3 checkpoint hash mismatch")
        # Build on CPU without loading weights, then assign the mapped tensors.
        # This avoids global builder monkey-patches and a second full CPU copy.
        model = build_sam3_image_model(device="cpu", checkpoint_path=None,
                                      load_from_HF=False, enable_inst_interactivity=False,
                                      compile=False)
        load_mapped_checkpoint(model, checkpoint)
        self.model = model.eval().to("cuda")
        self.processor = Sam3Processor(self.model, confidence_threshold=0.5)
        self._species_text = {}

    def _set_species_prompt(self, state, species):
        # Same pinned processor path as set_text_prompt, except for the two fixed
        # text-only outputs. The encoder gate serializes use of this model.
        if species not in {"DOG", "CAT"}:
            raise ValueError("Animal focus requires DOG or CAT")
        if "backbone_out" not in state:
            raise ValueError("Image features are required before a species prompt")
        if species not in self._species_text:
            outputs = self.model.backbone.forward_text([species.lower()], device=self.processor.device)
            self._species_text[species] = {key: value.detach().clone() for key, value in outputs.items()}
        # Grounding receives private tensors/dict so per-image state cannot poison
        # the next request. Only text features, never photo features, are cached.
        state["backbone_out"].update({key: value.clone() for key, value in self._species_text[species].items()})
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()
        return self.processor._forward_grounding(state)

    def prepare(self, image, species, *, prepared_image=None):
        import torch
        from PIL import Image
        if species not in {"DOG", "CAT"}:
            raise ValueError("Animal focus requires DOG or CAT")
        context = image.copy() if prepared_image is None else nullcontext(prepared_image)
        with context as working:
            if prepared_image is None:
                working.thumbnail((1024, 1024), Image.Resampling.BICUBIC)
            elif working.mode != "RGB" or max(working.size) > 1024:
                raise ValueError("Prepared SAM 3 image must be bounded RGB")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                state = self.processor.set_image(working)
                prediction = self._set_species_prompt(state, species)
            boxes = prediction["boxes"].detach().float().cpu().numpy()
            masks = prediction["masks"].detach().cpu().numpy()
            scores = prediction["scores"].detach().float().cpu().numpy()
            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks[:, 0]
            return prepare_prediction(working, boxes, masks, scores)
