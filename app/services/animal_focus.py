"""Conservative animal masking for the opt-in lost-search embedding profile.

A passed geometry check is not an anatomical completeness guarantee. Ambiguous
or suspect masks retain the full image; inference/configuration failures surface.
"""
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path

from PIL import Image, ImageOps

CHECKPOINT_SHA256 = "73cbd0190fcbe3ba339921fbce2c3a0b6bb9126c9a133c85e43a2a8e060a109e"
FOCUS_VERSION = "dinov3-large-mrcnn73cbd019-dual-pad256-v3"
BACKGROUND = (124, 116, 104)


@dataclass
class FocusResult:
    image: Image.Image
    status: str
    coat_color: dict | None = None


def square_image(image):
    return ImageOps.pad(image, (256, 256), method=Image.Resampling.BICUBIC, color=BACKGROUND)


def choose_detection(labels, scores, species, boxes=None, masks=None):
    """Never silently pick one animal when another plausible target is present."""
    if species not in {"DOG", "CAT"}:
        raise ValueError("Animal focus requires DOG or CAT")
    if len(labels) != len(scores) or any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
        raise RuntimeError("Invalid animal detector output")
    target = 18 if species == "DOG" else 17  # pinned COCO categories
    plausible = [i for i, (label, score) in enumerate(zip(labels, scores)) if label == target and score >= .3]
    if not plausible or max(scores[i] for i in plausible) < .7:
        return None, "original_no_confident_animal"
    if len(plausible) == 1:
        return plausible[0], "animal_mask"
    if boxes is None or masks is None:
        return None, "original_multiple_animals"
    if len(boxes) != len(labels):
        raise RuntimeError("Invalid animal detector boxes")
    anchor = max(plausible, key=lambda i: scores[i])
    # Collapse only near-contained body-part detections of the strongest instance.
    # Separate foregrounds, collages and uncertain overlaps remain ambiguous.
    if not all(i == anchor or same_instance(boxes[anchor], masks[anchor], boxes[i], masks[i])
               for i in plausible):
        return None, "original_multiple_animals"
    confident = [i for i in plausible if scores[i] >= .7]
    selected = max(confident, key=lambda i: (boxes[i][2]-boxes[i][0])*(boxes[i][3]-boxes[i][1]))
    return selected, "animal_mask"


def same_instance(first_box, first_mask, second_box, second_mask):
    import numpy as np
    for box in (first_box, second_box):
        if len(box) != 4 or not all(math.isfinite(v) for v in box) or box[2] <= box[0] or box[3] <= box[1]:
            raise RuntimeError("Invalid animal box")
    a, b = np.asarray(first_mask), np.asarray(second_mask)
    if a.shape != b.shape or a.ndim != 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise RuntimeError("Invalid animal masks")
    if (a < 0).any() or (a > 1).any() or (b < 0).any() or (b > 1).any():
        raise RuntimeError("Animal mask outside probability range")
    area = lambda box: (box[2]-box[0])*(box[3]-box[1])
    intersection = (max(0, min(first_box[2], second_box[2])-max(first_box[0], second_box[0]))
                    * max(0, min(first_box[3], second_box[3])-max(first_box[1], second_box[1])))
    if intersection / min(area(first_box), area(second_box)) < .9:
        return False
    a, b = a >= .5, b >= .5
    smaller = min(int(a.sum()), int(b.sum()))
    return smaller > 0 and np.logical_and(a, b).sum() / smaller >= .9


def prepare_focus(image, box, mask, *, color_source="single_mask"):
    """Return a bounded, aspect-preserving view; never modify the source image."""
    import numpy as np
    width, height = image.size
    if len(box) != 4 or not all(math.isfinite(x) for x in box):
        raise RuntimeError("Invalid animal box")
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise RuntimeError("Animal box outside image")
    probabilities = np.asarray(mask)
    if probabilities.shape != (height, width) or not np.isfinite(probabilities).all():
        raise RuntimeError("Invalid animal mask")
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise RuntimeError("Animal mask outside probability range")
    # Conservative pilot gates, versioned with the profile; not calibrated accuracy thresholds.
    # A small animal in a large photo is precisely where cropping can help.
    # Require usable detector-input pixels, not 45% of the whole photo.
    if min(x2-x1, y2-y1) < 64:
        return FocusResult(square_image(image), "original_small_region")
    bounds = (math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2))
    foreground = probabilities >= .5
    region = foreground[bounds[1]:bounds[3], bounds[0]:bounds[2]]
    ys, xs = np.nonzero(region)
    if (len(xs) == 0 or not .2 <= region.mean() <= .98
            or (xs.max()-xs.min()+1)/region.shape[1] < .75
            or (ys.max()-ys.min()+1)/region.shape[0] < .75):
        return FocusResult(square_image(image), "original_suspect_mask")
    from app.services.coat_color import describe
    color_mask = np.zeros_like(foreground)
    color_mask[bounds[1]:bounds[3], bounds[0]:bounds[2]] = probabilities[bounds[1]:bounds[3], bounds[0]:bounds[2]] >= .9
    color = describe(image, color_mask, source=color_source)
    # Only use the selected instance, with a small context margin around its box.
    dx, dy = (x2-x1)*.1, (y2-y1)*.1
    crop = (max(0, math.floor(x1-dx)), max(0, math.floor(y1-dy)),
            min(width, math.ceil(x2+dx)), min(height, math.ceil(y2+dy)))
    with Image.fromarray((foreground*255).astype("uint8")) as alpha:
        with Image.new("RGB", image.size, BACKGROUND) as background:
            with Image.composite(image, background, alpha) as masked:
                with masked.crop(crop) as view:
                    return FocusResult(square_image(view), "animal_mask", color)


class AnimalFocus:
    def __init__(self):
        import torch
        from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
        checkpoint = Path(os.environ["ANIMAL_FOCUS_CHECKPOINT"])
        digest = hashlib.sha256()
        with checkpoint.open("rb") as source:
            for chunk in iter(lambda: source.read(1024*1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != CHECKPOINT_SHA256:
            raise RuntimeError("Animal focus checkpoint hash mismatch")
        # No automatic network download at startup or on a user's request.
        self.model = maskrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None,
                                            box_detections_per_img=20).eval()
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        self.model.to("cuda")

    def prepare(self, image, species):
        import torch
        from torchvision.transforms.functional import pil_to_tensor
        if species not in {"DOG", "CAT"}:
            raise ValueError("Animal focus requires DOG or CAT")
        with image.copy() as working:
            working.thumbnail((1024, 1024), Image.Resampling.BICUBIC)
            tensor = pil_to_tensor(working).float().div_(255).to("cuda")
            with torch.inference_mode():
                result = self.model([tensor])[0]
            labels, scores = result["labels"].tolist(), result["scores"].tolist()
            boxes = result["boxes"].tolist()
            target = 18 if species == "DOG" else 17
            masks = {i: result["masks"][i, 0].detach().cpu().numpy()
                     for i in range(len(labels)) if labels[i] == target and scores[i] >= .3}
            selected, status = choose_detection(labels, scores, species, boxes, masks)
            if selected is None:
                return FocusResult(square_image(working), status)
            return prepare_focus(working, boxes[selected], masks[selected])
