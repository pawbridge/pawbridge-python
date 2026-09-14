"""Pinned SAM 3 image-only segmentation; no model downloads or image synthesis."""
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
    if len(scores) != 1:
        status = "original_multiple_animals" if len(scores) > 1 else "original_no_confident_animal"
        return FocusResult(square_image(image), status)
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

    def prepare(self, image, species):
        import torch
        from PIL import Image
        if species not in {"DOG", "CAT"}:
            raise ValueError("Animal focus requires DOG or CAT")
        with image.copy() as working:
            working.thumbnail((1024, 1024), Image.Resampling.BICUBIC)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                state = self.processor.set_image(working)
                prediction = self.processor.set_text_prompt(state=state, prompt=species.lower())
            boxes = prediction["boxes"].detach().float().cpu().numpy()
            masks = prediction["masks"].detach().cpu().numpy()
            scores = prediction["scores"].detach().float().cpu().numpy()
            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks[:, 0]
            return prepare_prediction(working, boxes, masks, scores)
