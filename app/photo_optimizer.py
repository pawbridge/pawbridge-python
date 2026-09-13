"""Bounded, CPU-only photo optimization; no network or storage side effects."""

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_INPUT_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 16_000_000
MAX_MPO_FRAMES = 16
WEBP_QUALITY = 90
WEBP_METHOD = 4
MIME_TYPES = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}


class InvalidPhoto(ValueError):
    pass


class PhotoTooLarge(InvalidPhoto):
    pass


@dataclass(frozen=True)
class OptimizedPhoto:
    data: bytes
    content_type: str
    source_sha256: str
    stored_sha256: str
    # Display dimensions, after applying EXIF orientation even on original fallback.
    width: int
    height: int
    recipe: str


def _encode_webp(photo: Image.Image) -> bytes:
    output = BytesIO()
    options = {}
    if photo.info.get("icc_profile"):
        options["icc_profile"] = photo.info["icc_profile"]
    photo.save(output, format="WEBP", quality=WEBP_QUALITY, method=WEBP_METHOD, **options)
    return output.getvalue()


def _preserve_mpo(source: Image.Image, data: bytes) -> OptimizedPhoto:
    # MPO may carry auxiliary JPEG frames. Re-encoding only the first would discard them.
    try:
        if not 1 <= source.n_frames <= MAX_MPO_FRAMES:
            raise PhotoTooLarge("MPO exceeds frame limit")
        width, height = source.size
        if source.getexif().get(274) in (5, 6, 7, 8):
            width, height = height, width
        pixels = 0
        for frame in range(source.n_frames):
            source.seek(frame)
            pixels += source.width * source.height
            if pixels > MAX_PIXELS:
                raise PhotoTooLarge("MPO exceeds total pixel limit")
            source.load()
    except InvalidPhoto:
        raise
    except (ValueError, EOFError) as exc:
        raise InvalidPhoto("MPO frame could not be decoded") from exc
    digest = sha256(data).hexdigest()
    return OptimizedPhoto(
        data=data,
        content_type="image/jpeg",
        source_sha256=digest,
        stored_sha256=digest,
        width=width,
        height=height,
        recipe="original-v1",
    )


def optimize_photo(data: bytes) -> OptimizedPhoto:
    if len(data) > MAX_INPUT_BYTES:
        raise PhotoTooLarge("Photo exceeds 10 MiB")
    if not data:
        raise InvalidPhoto("Photo is empty")

    try:
        with Image.open(BytesIO(data)) as source:
            if source.format == "MPO":
                return _preserve_mpo(source, data)
            if source.format not in MIME_TYPES:
                raise InvalidPhoto("Only JPEG, MPO, static PNG and WebP are supported")
            content_type = MIME_TYPES[source.format]
            if source.width * source.height > MAX_PIXELS:
                raise PhotoTooLarge("Photo exceeds 16 million pixels")
            if getattr(source, "is_animated", False):
                raise InvalidPhoto("Animated photos are not supported")
            source.verify()
        with Image.open(BytesIO(data)) as source:
            source.load()
            photo = ImageOps.exif_transpose(source)
    except Image.DecompressionBombError as exc:
        raise PhotoTooLarge("Photo exceeds pixel limit") from exc
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise InvalidPhoto("Photo could not be decoded") from exc

    try:
        stored = data
        recipe = "original-v1"
        # Avoid another lossy WebP generation and implicit CMYK/16-bit conversion.
        if content_type != "image/webp" and photo.mode in ("RGB", "RGBA"):
            # Encoding failures propagate to the caller for retry, never fake success.
            if photo.mode == "RGB" and "transparency" in photo.info:
                with photo.convert("RGBA") as transparent_photo:
                    encoded = _encode_webp(transparent_photo)
            else:
                encoded = _encode_webp(photo)
            if len(encoded) < len(data):
                stored = encoded
                content_type = "image/webp"
                recipe = "webp-q90-m4-fullsize-v1"
        return OptimizedPhoto(
            data=stored,
            content_type=content_type,
            source_sha256=sha256(data).hexdigest(),
            stored_sha256=sha256(stored).hexdigest(),
            width=photo.width,
            height=photo.height,
            recipe=recipe,
        )
    finally:
        photo.close()
