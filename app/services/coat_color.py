"""Versioned foreground Lab distributions; not an animal identity classifier."""
from functools import lru_cache
from itertools import product
import math
import os

VERSION = "foreground-lab32-joint8-v2"
MARGINAL_BINS = 32
JOINT_BINS = 8
JOINT_TOLERANCE = .08
MIN_PIXELS = 128
MAX_SAMPLES = 8192
SOURCES = frozenset(("single_mask", "primary_mask"))


@lru_cache(maxsize=1)
def _transform():
    from PIL import ImageCms
    return ImageCms.buildTransformFromOpenProfiles(
        ImageCms.createProfile("sRGB"), ImageCms.createProfile("LAB"), "RGB", "LAB")


def describe(image, foreground, *, source="single_mask"):
    """Ignore fill/background; sample a bounded number of confident inner pixels."""
    import numpy as np
    from PIL import Image, ImageCms, ImageFilter
    if source not in SOURCES:
        raise ValueError("Unknown coat-color source")
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
    coordinates = lab / 255 * (MARGINAL_BINS - 1)
    lower = coordinates.astype(int)
    fractions = coordinates - lower
    marginal = []
    for channel in range(3):
        h = np.bincount(lower[:, channel], weights=1 - fractions[:, channel],
                        minlength=MARGINAL_BINS)
        h += np.bincount(np.minimum(lower[:, channel] + 1, MARGINAL_BINS - 1),
                         weights=fractions[:, channel], minlength=MARGINAL_BINS)
        marginal.extend((h / len(pixels)).tolist())

    # Marginal histograms cannot distinguish color combinations with identical
    # per-channel totals. A softly binned joint Lab distribution keeps that
    # correlation while remaining bounded and deterministic.
    joint_lab = lab.copy()
    # Preserve relative light/dark pattern in the joint distribution without
    # charging twice for a uniform exposure shift. Absolute lightness remains
    # available in the marginal distribution with its established tolerance.
    joint_lab[:, 0] = np.clip(lab[:, 0] - np.median(lab[:, 0]) + 127.5, 0, 255)
    joint_coordinates = joint_lab / 255 * (JOINT_BINS - 1)
    joint_lower = joint_coordinates.astype(int)
    joint_fractions = joint_coordinates - joint_lower
    joint = np.zeros((JOINT_BINS, JOINT_BINS, JOINT_BINS), dtype="float64")
    for offsets in product((0, 1), repeat=3):
        indices = [np.minimum(joint_lower[:, channel] + offsets[channel], JOINT_BINS - 1)
                   for channel in range(3)]
        weights = np.ones(len(lab), dtype="float64")
        for channel, offset in enumerate(offsets):
            weights *= (joint_fractions[:, channel] if offset
                        else 1 - joint_fractions[:, channel])
        np.add.at(joint, tuple(indices), weights)
    joint /= len(pixels)
    return {"version": VERSION, "source": source, "pixels": count,
            "marginal": marginal, "joint": joint.ravel().tolist()}


def valid(value):
    if not isinstance(value, dict) or value.get("version") != VERSION:
        return False
    pixels, marginal, joint = value.get("pixels"), value.get("marginal"), value.get("joint")
    if type(pixels) is not int or not MIN_PIXELS <= pixels <= 1024 * 1024:
        return False
    if value.get("source") not in SOURCES:
        return False
    if not isinstance(marginal, list) or len(marginal) != 3 * MARGINAL_BINS:
        return False
    if not isinstance(joint, list) or len(joint) != JOINT_BINS ** 3:
        return False
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
           for v in marginal + joint):
        return False
    return (all(abs(sum(marginal[i:i+MARGINAL_BINS]) - 1) < 1e-5
                for i in range(0, 3*MARGINAL_BINS, MARGINAL_BINS))
            and abs(sum(joint) - 1) < 1e-5)


def _marginal_distance(first, second):
    distances = []
    for channel, scale, tolerance, normalizer in ((0, 100, 12, 60),
                                                   (1, 255, 4, 40),
                                                   (2, 255, 4, 40)):
        cumulative = distance = 0.
        for i in range(channel * MARGINAL_BINS, (channel + 1) * MARGINAL_BINS):
            cumulative += first["marginal"][i] - second["marginal"][i]
            distance += abs(cumulative)
        distances.append(max(0., distance * scale / (MARGINAL_BINS - 1) - tolerance)
                         / normalizer)
    return min(1., math.sqrt(sum(d*d for d in distances) / 3))


def _joint_distance(first, second):
    import numpy as np
    a = np.asarray(first["joint"], dtype="float64").reshape(
        JOINT_BINS, JOINT_BINS, JOINT_BINS)
    b = np.asarray(second["joint"], dtype="float64").reshape(
        JOINT_BINS, JOINT_BINS, JOINT_BINS)

    def shift_lightness(histogram, offset):
        if offset == 0:
            return histogram
        shifted = np.zeros_like(histogram)
        if offset > 0:
            shifted[offset:] = histogram[:-offset]
            shifted[-1] += histogram[-offset:].sum(axis=0)
        else:
            amount = -offset
            shifted[:-amount] = histogram[amount:]
            shifted[0] += histogram[:amount].sum(axis=0)
        return shifted

    # One coarse lightness bin is the v1 12 L* exposure tolerance expressed in
    # the joint space. Chroma is not shifted because coat hue is useful evidence.
    distances = []
    for offset in (-1, 0, 1):
        coefficient = np.sqrt(a * shift_lightness(b, offset)).sum()
        distances.append(math.sqrt(max(0., 1 - min(1., float(coefficient)))))
    distance = min(distances)
    return max(0., distance - JOINT_TOLERANCE) / (1 - JOINT_TOLERANCE)


def mismatch(first, second):
    """Bounded distance, or None when color comparison is unavailable.

    Marginal earth-mover distance retains the v1 exposure tolerance. Joint Lab
    distance additionally distinguishes different color combinations that have
    the same per-channel totals. These remain ranking evidence, not an identity
    threshold.
    """
    if not valid(first) or not valid(second):
        return None
    marginal = _marginal_distance(first, second)
    joint = _joint_distance(first, second)
    # v2 adds evidence; it must not dilute a difference that the established
    # marginal comparison already detected.
    return max(marginal, joint)


def ranking_weight():
    # Opt in only after offline evaluation and a complete gallery color backfill.
    try:
        weight = float(os.getenv("LOST_SEARCH_COAT_COLOR_WEIGHT", "0"))
    except ValueError as error:
        raise RuntimeError("Invalid coat-color ranking weight") from error
    if not math.isfinite(weight) or not 0 <= weight <= .2:
        raise RuntimeError("Coat-color ranking weight must be between 0 and 0.2")
    return weight
