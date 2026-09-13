import hashlib
import unittest
from io import BytesIO
from unittest.mock import patch

from PIL import Image, ImageOps

from app.photo_optimizer import InvalidPhoto, PhotoTooLarge, optimize_photo


def photo_bytes(mode="RGB", size=(180, 120), format="PNG", **options):
    image = Image.new(mode, size, (23, 101, 205, 90) if mode == "RGBA" else 120)
    output = BytesIO()
    # Uncompressed PNG makes the smaller-output branch deterministic.
    if format == "PNG":
        options.setdefault("compress_level", 0)
    image.save(output, format=format, **options)
    return output.getvalue()


def mpo_bytes(*, orientation=1, frames=2):
    output = BytesIO()
    images = [Image.new("RGB", (24, 16), color) for color in ("red", "blue", "green")[:frames]]
    exif = Image.Exif()
    exif[274] = orientation
    try:
        images[0].save(output, format="MPO", save_all=True, append_images=images[1:], exif=exif)
        return output.getvalue()
    finally:
        for image in images:
            image.close()


class PhotoOptimizerTests(unittest.TestCase):
    def test_smaller_webp_preserves_full_frame_and_dimensions(self):
        source = Image.new("RGB", (180, 120), "green")
        # Distinct borders expose cropping or stretching.
        source.paste("red", (0, 0, 15, 120))
        source.paste("blue", (165, 0, 180, 120))
        output = BytesIO()
        source.save(output, format="PNG", compress_level=0)
        original = output.getvalue()
        result = optimize_photo(original)
        self.assertLess(len(result.data), len(original))
        self.assertEqual(result.content_type, "image/webp")
        self.assertEqual(result.recipe, "webp-q90-m4-fullsize-v1")
        self.assertEqual((result.width, result.height), (180, 120))
        with Image.open(BytesIO(result.data)) as actual:
            self.assertEqual(actual.size, source.size)
            self.assertGreater(actual.getpixel((5, 60))[0], 230)
            self.assertGreater(actual.getpixel((175, 60))[2], 230)
        self.assertEqual(result.source_sha256, hashlib.sha256(original).hexdigest())
        self.assertEqual(result.stored_sha256, hashlib.sha256(result.data).hexdigest())

    def test_equal_or_larger_output_keeps_exact_original_and_mime(self):
        original = photo_bytes()
        for encoded in (original, original + b"larger"):
            with self.subTest(size=len(encoded)), patch("app.photo_optimizer._encode_webp", return_value=encoded):
                result = optimize_photo(original)
                self.assertEqual(result.data, original)
                self.assertEqual(result.content_type, "image/png")
                self.assertEqual(result.recipe, "original-v1")
                self.assertEqual(result.source_sha256, result.stored_sha256)

    def test_mpo_preserves_all_original_frames_and_primary_display_orientation(self):
        for orientation, size in ((1, (24, 16)), (6, (16, 24))):
            with self.subTest(orientation=orientation):
                original = mpo_bytes(orientation=orientation)
                with patch("app.photo_optimizer._encode_webp") as encode:
                    result = optimize_photo(original)
                encode.assert_not_called()
                self.assertEqual(result.data, original)
                self.assertEqual(result.content_type, "image/jpeg")
                self.assertEqual(result.recipe, "original-v1")
                self.assertEqual((result.width, result.height), size)
                self.assertEqual(result.source_sha256, hashlib.sha256(original).hexdigest())
                self.assertEqual(result.stored_sha256, result.source_sha256)
                with Image.open(BytesIO(result.data)) as actual:
                    self.assertEqual(actual.n_frames, 2)
                    for frame in range(actual.n_frames):
                        actual.seek(frame)
                        actual.load()

    def test_mpo_rejects_damaged_secondary_frame(self):
        original = mpo_bytes()
        secondary = original.find(b"\xff\xd8", 2)
        self.assertGreater(secondary, 0)
        damaged = original[:secondary] + b"broken secondary frame"
        with self.assertRaises(InvalidPhoto):
            optimize_photo(damaged)

    def test_mpo_enforces_frame_and_cumulative_pixel_limits(self):
        original = mpo_bytes()
        for setting, limit in (("MAX_MPO_FRAMES", 1), ("MAX_PIXELS", 24 * 16)):
            with self.subTest(setting=setting), patch("app.photo_optimizer." + setting, limit):
                with self.assertRaises(PhotoTooLarge):
                    optimize_photo(original)

    def test_jpeg_is_supported(self):
        original = photo_bytes(format="JPEG")
        result = optimize_photo(original)
        with Image.open(BytesIO(result.data)) as actual:
            self.assertEqual(actual.size, (180, 120))
        self.assertLessEqual(len(result.data), len(original))

    def test_existing_webp_is_not_recompressed(self):
        original = photo_bytes(format="WEBP")
        with patch("app.photo_optimizer._encode_webp") as encode:
            result = optimize_photo(original)
        encode.assert_not_called()
        self.assertEqual(result.data, original)
        self.assertEqual(result.content_type, "image/webp")

    def test_cmyk_photo_is_preserved_without_color_conversion(self):
        original = photo_bytes(mode="CMYK", format="JPEG")
        result = optimize_photo(original)
        self.assertEqual(result.data, original)
        self.assertEqual(result.content_type, "image/jpeg")

    def test_alpha_channel_is_preserved(self):
        original = photo_bytes(mode="RGBA")
        result = optimize_photo(original)
        self.assertEqual(result.content_type, "image/webp")
        with Image.open(BytesIO(result.data)) as actual:
            self.assertEqual(actual.getextrema()[3], (90, 90))

    def test_png_transparent_color_is_not_flattened(self):
        original = photo_bytes(transparency=(120, 0, 0))
        with Image.open(BytesIO(original)) as source:
            expected_alpha = source.convert("RGBA").getchannel("A").tobytes()
        result = optimize_photo(original)
        self.assertEqual(result.content_type, "image/webp")
        with Image.open(BytesIO(result.data)) as actual:
            self.assertEqual(actual.convert("RGBA").getchannel("A").tobytes(), expected_alpha)

    def test_exif_orientation_is_applied_once_without_crop(self):
        exif = Image.Exif()
        exif[274] = 6
        original = photo_bytes(exif=exif)
        result = optimize_photo(original)
        self.assertEqual(result.content_type, "image/webp")
        self.assertEqual((result.width, result.height), (120, 180))
        with Image.open(BytesIO(result.data)) as actual:
            self.assertEqual(actual.size, (120, 180))
            self.assertEqual(ImageOps.exif_transpose(actual).size, actual.size)

    def test_original_fallback_reports_display_orientation(self):
        exif = Image.Exif()
        exif[274] = 6
        original = photo_bytes(exif=exif)
        with patch("app.photo_optimizer._encode_webp", return_value=original):
            result = optimize_photo(original)
        self.assertEqual(result.data, original)
        self.assertEqual((result.width, result.height), (120, 180))

    def test_icc_profile_is_preserved(self):
        from PIL import ImageCms
        profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        result = optimize_photo(photo_bytes(icc_profile=profile))
        self.assertEqual(result.content_type, "image/webp")
        with Image.open(BytesIO(result.data)) as actual:
            self.assertEqual(actual.info["icc_profile"], profile)

    def test_empty_corrupt_truncated_and_unsupported_photo_are_rejected(self):
        valid = photo_bytes()
        for original in (b"", b"not a photo", valid[:100], photo_bytes(format="GIF")):
            with self.subTest(size=len(original)), self.assertRaises(InvalidPhoto):
                optimize_photo(original)

    def test_animation_is_rejected_without_silently_selecting_first_frame(self):
        output = BytesIO()
        Image.new("RGB", (20, 20), "red").save(
            output, format="PNG", save_all=True,
            append_images=[Image.new("RGB", (20, 20), "blue")], duration=100,
        )
        with self.assertRaises(InvalidPhoto):
            optimize_photo(output.getvalue())

    def test_byte_and_pixel_caps_reject_before_encoding(self):
        with patch("app.photo_optimizer.MAX_INPUT_BYTES", 2), self.assertRaises(PhotoTooLarge):
            optimize_photo(b"123")
        original = photo_bytes()
        with patch("app.photo_optimizer.MAX_PIXELS", 100), patch("app.photo_optimizer._encode_webp") as encode:
            with self.assertRaises(PhotoTooLarge):
                optimize_photo(original)
            encode.assert_not_called()

    def test_encoder_failure_is_not_reported_as_successful_optimization(self):
        with patch("app.photo_optimizer._encode_webp", side_effect=OSError("encoder failure")):
            with self.assertRaises(OSError):
                optimize_photo(photo_bytes())


if __name__ == "__main__":
    unittest.main()
