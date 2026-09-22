"""Offline model workers. Never imported by a serving route or gallery publisher."""
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image, ImageOps

from app.services.animal_focus import BACKGROUND, square_image
from app.services.lost_search import decode_photo
from .dataset import digest, json_read
from .features import DIM, save_features, unit

SERVICE = 'pawbridge-lost-search.service'


def require_idle_service():
    state = subprocess.run(['systemctl', 'show', SERVICE, '--property=ActiveState', '--value'],
                           check=True, capture_output=True, text=True, timeout=10).stdout.strip()
    if state != 'inactive':
        raise RuntimeError('Production AI must be explicitly stopped before GPU evaluation; state=' + state)


def configure_gpu():
    require_idle_service()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA evaluation requires an available GPU')
    torch.set_num_threads(2)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1., 7 * 1024**3 / total))
    # Both precisions use the same TF32 policy, isolating the autocast change.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch


def crop_bounds(box, size):
    x1, y1, x2, y2 = box
    dx, dy = (x2-x1)*.1, (y2-y1)*.1
    return (max(0, math.floor(x1-dx)), max(0, math.floor(y1-dy)),
            min(size[0], math.ceil(x2+dx)), min(size[1], math.ceil(y2+dy)))


def mask_view(mask, box, size):
    with mask.crop(crop_bounds(box, mask.size)) as crop:
        return ImageOps.pad(crop, (size, size), method=Image.Resampling.NEAREST, color=0)


def native_view(original, small_mask, box, size):
    """Map the original SAM mask to source pixels, not an upscaled 256 input."""
    sx, sy = original.width / small_mask.width, original.height / small_mask.height
    native_box = [box[0]*sx, box[1]*sy, box[2]*sx, box[3]*sy]
    with small_mask.resize(original.size, Image.Resampling.NEAREST) as mask:
        with Image.new('RGB', original.size, BACKGROUND) as background:
            with Image.composite(original, background, mask) as foreground:
                with foreground.crop(crop_bounds(native_box, original.size)) as cropped:
                    image = ImageOps.pad(cropped, (size, size), method=Image.Resampling.BICUBIC, color=BACKGROUND)
        padded_mask = mask_view(mask, native_box, size)
    return image, padded_mask


def pool_tokens(tokens, prefix, mask, mode):
    """Operate on post-norm DINO tokens. Foreground weights exclude pad/background."""
    patches = np.asarray(tokens[prefix:], dtype=np.float32)
    side = math.isqrt(len(patches))
    if patches.shape != (side*side, DIM) or prefix < 1:
        raise ValueError('Expected square DINO patch tokens')
    if mask is None:
        weights = np.ones(len(patches), dtype=np.float32)
        selected = np.empty(0, dtype=int)
    else:
        alpha = np.asarray(mask, dtype=np.float32) / 255.
        if alpha.shape != (side*16, side*16):
            raise ValueError('Mask and patch grid differ')
        weights = alpha.reshape(side, 16, side, 16).mean(axis=(1, 3)).ravel()
        selected = np.flatnonzero(weights >= .5)
    if mode == 'cls':
        vector = tokens[0]
    elif mode == 'foreground' and mask is not None and weights.sum() > 1e-6:
        vector = np.average(patches, axis=0, weights=weights)
    elif mode in {'avg', 'foreground'}:
        vector = patches.mean(axis=0)
    else:
        raise ValueError('Unknown pooling')
    if len(selected) > 256:
        selected = selected[np.linspace(0, len(selected)-1, 256, dtype=int)]
    coords = np.column_stack(((selected % side + .5)/side, (selected // side + .5)/side))
    return unit(vector), unit(patches[selected]), coords.astype(np.float32)


def sam_worker(dataset, output):
    torch = configure_gpu()
    from app.services.sam3_focus import Sam3Focus, prepare_prediction
    started = time.perf_counter()
    focus = Sam3Focus()
    startup = time.perf_counter() - started
    records = []
    for index, photo in enumerate(dataset.photos):
        directory = output / 'segmentation' / photo.id
        directory.mkdir(parents=True)
        with decode_photo(photo.path.read_bytes()) as original:
            with original.copy() as working:
                working.thumbnail((1024, 1024), Image.Resampling.BICUBIC)
                torch.cuda.synchronize()
                begin = time.perf_counter()
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    state = focus.processor.set_image(working)
                    prediction = focus._set_species_prompt(state, photo.species)
                boxes = prediction['boxes'].detach().float().cpu().numpy()
                masks = prediction['masks'].detach().float().cpu().numpy()
                scores = prediction['scores'].detach().float().cpu().numpy()
                if masks.ndim == 4 and masks.shape[1] == 1:
                    masks = masks[:, 0]
                prepared = prepare_prediction(working, boxes, masks, scores)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - begin
                try:
                    prepared.image.save(directory / 'legacy.png')
                    box = None
                    if prepared.status == 'animal_mask':
                        box = boxes[0].tolist()
                        box = [max(0, box[0]), max(0, box[1]), min(working.width, box[2]), min(working.height, box[3])]
                        with Image.fromarray((masks[0] >= .5).astype('uint8')*255) as mask:
                            mask.save(directory / 'mask.png')
                    record = {'image_id': photo.id, 'focus_status': prepared.status,
                              'coat_color': prepared.coat_color, 'box': box,
                              'seconds': elapsed, 'first_image': index == 0}
                    (directory / 'metadata.json').write_text(json.dumps(record, allow_nan=False))
                    records.append({key: record[key] for key in ('image_id', 'focus_status', 'seconds', 'first_image')})
                finally:
                    prepared.image.close()
                    del state, prediction, boxes, masks, scores
    write_stage(output, 'sam', startup, records, torch)


def write_stage(output, stage, startup, records, torch):
    import resource
    groups = {}
    for record in records:
        groups.setdefault(record.get('variant', stage), []).append(record)
    rates = {}
    for name, cohort in groups.items():
        warm = [row['seconds'] for row in cohort if not row['first_image']]
        total = sum(warm)
        rates[name] = {'warm_images': len(warm), 'warm_seconds': total,
                       'images_per_second': len(warm)/total if total else None,
                       'images_per_minute': 60*len(warm)/total if total else None,
                       'median_seconds': float(np.median(warm)) if warm else None,
                       'p95_seconds': float(np.percentile(warm, 95)) if warm else None}
    (output / f'{stage}-timing.json').write_text(json.dumps({
        'stage': stage, 'startup_seconds': startup, 'records': records, 'rates': rates,
        'peak_process_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(),
        'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
        'tf32': False, 'cpu_threads': torch.get_num_threads()}, allow_nan=False, indent=2))


def dino_worker(dataset, output, variants, checkpoint):
    torch = configure_gpu()
    import timm
    from app.services.dinov3 import MODEL_NAME, WEIGHTS_SHA256
    if digest(checkpoint) != WEIGHTS_SHA256:
        raise ValueError('DINO checkpoint differs from pinned weights')
    begin = time.perf_counter()
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=0,
                              checkpoint_path=str(checkpoint)).eval().to('cuda')
    config = timm.data.resolve_model_data_config(model)
    expected = {'input_size': (3, 256, 256), 'interpolation': 'bicubic',
                'mean': (.485, .456, .406), 'std': (.229, .224, .225),
                'crop_pct': 1., 'crop_mode': 'center'}
    if config != expected or model.global_pool != 'avg' or model.num_prefix_tokens != 5:
        raise RuntimeError('Pinned DINO preprocessing/token contract changed')
    startup = time.perf_counter() - begin
    records = []

    def encode(image, variant, mask=None):
        transform = timm.data.create_transform(**dict(config, input_size=(3, image.height, image.width)), is_training=False)
        tensor = transform(image).unsqueeze(0).to('cuda')
        context = torch.autocast('cuda', dtype=torch.float16) if variant.precision == 'fp16' else nullcontext()
        with torch.inference_mode(), context:
            tokens = model.forward_features(tensor)
            # Production forward_head performs its average in the activation
            # dtype before conversion to float32. Keep that exact baseline.
            head = model.forward_head(tokens).float()
        values = tokens[0].float().cpu().numpy()
        vector, patches, coordinates = pool_tokens(values, model.num_prefix_tokens, mask, variant.pooling)
        if variant.pooling == 'avg' or (variant.pooling == 'foreground' and mask is None):
            vector = unit(head[0].cpu().numpy())
        return vector, patches, coordinates

    for variant in variants:
        inputs = output / 'inputs' / variant.name
        inputs.mkdir(parents=True)
        for index, photo in enumerate(dataset.photos):
            segment = output / 'segmentation' / photo.id
            metadata = json_read(segment / 'metadata.json')
            torch.cuda.synchronize()
            begin = time.perf_counter()
            with decode_photo(photo.path.read_bytes()) as original:
                with square_image(original) as full:
                    full_vector, _, _ = encode(full, variant)
                animal, patches, coordinates = None, np.empty((0, DIM), np.float32), np.empty((0, 2), np.float32)
                if metadata['focus_status'] == 'animal_mask':
                    with Image.open(segment / 'mask.png') as mask:
                        if variant.source == 'native':
                            view, alpha = native_view(original, mask, metadata['box'], variant.size)
                        else:
                            with Image.open(segment / 'legacy.png') as legacy:
                                view = legacy.copy()
                            alpha = mask_view(mask, metadata['box'], 256)
                        with view, alpha:
                            animal, patches, coordinates = encode(view, variant, alpha)
                            view.save(inputs / f'{photo.id}.jpg', quality=90)
                else:
                    with square_image(original) as full:
                        full.save(inputs / f'{photo.id}.jpg', quality=90)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - begin
            record = {'image_id': photo.id, 'focus_status': metadata['focus_status'],
                      'variant': variant.name, 'seconds': elapsed, 'first_image': index == 0}
            save_features(output / 'features' / variant.name / photo.id, photo, variant,
                          full=full_vector, animal=animal, patches=patches, coordinates=coordinates,
                          metadata=dict(metadata, weights_sha256=WEIGHTS_SHA256))
            records.append(record)
    write_stage(output, 'dino', startup, records, torch)
