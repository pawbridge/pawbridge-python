"""Versioned foreground Lab distributions; not an animal identity classifier."""
from functools import lru_cache
import math
import os

VERSION = "foreground-lab32-v1"
BINS = 32
MIN_PIXELS = 128
MAX_SAMPLES = 8192


@lru_cache(maxsize=1)
def _transform():
    from PIL import ImageCms
    return ImageCms.buildTransformFromOpenProfiles(
        ImageCms.createProfile("sRGB"), ImageCms.createProfile("LAB"), "RGB", "LAB")


def describe(image, foreground):
    """Ignore fill/background; sample a bounded number of confident inner pixels."""
    import numpy as np
    from PIL import Image, ImageCms, ImageFilter
    mask = np.asarray(foreground, dtype=bool)
    if mask.shape != (image.height, image.width):
        raise ValueError("Color mask does not match image")
    with Image.fromarray(mask.astype("uint8") * 255) as alpha:
        # Drop one boundary pixel; this does not guarantee fur-only segmentation.
        with alpha.filter(ImageFilter.MinFilter(3)) as inner:
            selected = np.flatnonzero(np.asarray(inner).ravel())
    count = len(selected)
    if count < MIN_PIXELS:
        return None
    if count > MAX_SAMPLES:
        selected = selected[np.linspace(0, count - 1, MAX_SAMPLES, dtype=int)]
    pixels = np.asarray(image).reshape(-1, 3)[selected]
    with Image.fromarray(pixels.reshape(1, -1, 3), "RGB") as sample:
        with ImageCms.applyTransform(sample, _transform()) as converted:
            lab = np.asarray(converted).reshape(-1, 3).astype("float64")
    # Soft bins avoid a sudden mismatch across an arbitrary histogram boundary.
    coordinates = lab / 255 * (BINS - 1)
    lower = coordinates.astype(int)
    fractions = coordinates - lower
    histogram = []
    for channel in range(3):
        h = np.bincount(lower[:, channel], weights=1 - fractions[:, channel], minlength=BINS)
        h += np.bincount(np.minimum(lower[:, channel] + 1, BINS - 1),
                         weights=fractions[:, channel], minlength=BINS)
        histogram.extend((h / len(pixels)).tolist())
    return {"version": VERSION, "pixels": count, "histogram": histogram}


def valid(value):
    if not isinstance(value, dict) or value.get("version") != VERSION:
        return False
    pixels, histogram = value.get("pixels"), value.get("histogram")
    if type(pixels) is not int or not MIN_PIXELS <= pixels <= 1024 * 1024:
        return False
    if not isinstance(histogram, list) or len(histogram) != 3 * BINS:
        return False
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in histogram):
        return False
    return all(abs(sum(histogram[i:i+BINS]) - 1) < 1e-5 for i in range(0, 3*BINS, BINS))


def mismatch(first, second):
    """Bounded distance, or None when color comparison is unavailable.

    Lab marginal earth-mover distances retain dark/light and mixed-color mass.
    A 12 L* / 4 a*,b* dead band softens exposure/white-balance perturbations.
    These are provisional tolerances, not calibrated same-animal thresholds.
    """
    if not valid(first) or not valid(second):
        return None
    distances = []
    for channel, scale, tolerance, normalizer in ((0, 100, 12, 60), (1, 255, 4, 40), (2, 255, 4, 40)):
        cumulative = distance = 0.
        for i in range(channel * BINS, (channel + 1) * BINS):
            cumulative += first["histogram"][i] - second["histogram"][i]
            distance += abs(cumulative)
        distances.append(max(0., distance * scale / (BINS - 1) - tolerance) / normalizer)
    return min(1., math.sqrt(sum(d*d for d in distances) / 3))


def ranking_weight():
    # Opt in only after offline evaluation and a complete gallery color backfill.
    try:
        weight = float(os.getenv("LOST_SEARCH_COAT_COLOR_WEIGHT", "0"))
    except ValueError as error:
        raise RuntimeError("Invalid coat-color ranking weight") from error
    if not math.isfinite(weight) or not 0 <= weight <= .2:
        raise RuntimeError("Coat-color ranking weight must be between 0 and 0.2")
    return weight
