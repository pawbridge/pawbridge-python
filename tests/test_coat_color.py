import copy
import os
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageEnhance
from app.services.coat_color import describe, mismatch, ranking_weight, valid, VERSION
from app.services.lost_search import rank_candidates


def swatch(color, background="blue"):
    image = Image.new("RGB", (120, 120), background)
    image.paste(color, (20, 20, 100, 100))
    mask = np.zeros((120, 120), dtype=bool)
    mask[20:100, 20:100] = True
    return image, mask


def descriptor(color):
    image, mask = swatch(color)
    with image:
        return describe(image, mask)


class CoatColorTest(unittest.TestCase):
    def test_background_and_small_boundary_changes_do_not_change_foreground(self):
        first, mask = swatch((110, 70, 35), "blue")
        second, _ = swatch((110, 70, 35), "green")
        with first, second:
            second.paste("red", (20, 20, 100, 21))
            a, b = describe(first, mask), describe(second, mask)
        self.assertTrue(valid(a))
        self.assertEqual(a, b)
        self.assertEqual(mismatch(a, b), 0)

    def test_dark_vs_tan_and_black_vs_white_differ_more_than_exposure_change(self):
        dark, mask = swatch((45, 40, 35))
        with dark, ImageEnhance.Brightness(dark).enhance(1.3) as brighter:
            a, b = describe(dark, mask), describe(brighter, mask)
        same = mismatch(a, b)
        self.assertLess(same, .08)
        self.assertGreater(mismatch(a, descriptor((165, 110, 60))), same + .25)
        self.assertGreater(mismatch(descriptor("black"), descriptor("white")), .5)

    def test_mixed_coat_is_not_collapsed_to_average_gray_and_small_mark_has_limited_effect(self):
        gray, mask = swatch((128, 128, 128))
        mixed, _ = swatch("white")
        mixed.paste("black", (20, 20, 60, 100))
        marked, _ = swatch((128, 128, 128))
        marked.paste("red", (50, 50, 60, 60))
        with gray, mixed, marked:
            a, b, c = [describe(image, mask) for image in (gray, mixed, marked)]
        self.assertGreater(mismatch(a, b), .15)
        self.assertLess(mismatch(a, c), .05)

    def test_unavailable_wrong_version_or_invalid_features_are_not_a_mismatch(self):
        good = descriptor("black")
        corrupt = copy.deepcopy(good); corrupt["histogram"][0] = float("nan")
        for bad in (None, {}, dict(good, version="old"), corrupt):
            self.assertIsNone(mismatch(good, bad))
        image, mask = swatch("black")
        with image:
            mask[:] = False
            self.assertIsNone(describe(image, mask))

    def test_color_reorders_close_candidates_without_hiding_them_or_changing_raw_image_score(self):
        black = descriptor((45, 40, 35)); tan = descriptor((165, 110, 60))
        hits = [{"_score": 1.88, "_source": {"id": 1, "coat_color": tan}},
                {"_score": 1.87, "_source": {"id": 2, "coat_color": black}}]
        base = rank_candidates(hits, coat_color=black)
        enabled = rank_candidates(hits, coat_color=black, color_weight=.12)
        self.assertEqual([r["animalId"] for r in base], [1, 2])
        self.assertEqual([r["animalId"] for r in enabled], [2, 1])
        self.assertAlmostEqual(enabled[1]["imageScore"], .88)
        self.assertEqual(rank_candidates(hits, color_weight=.12), base)
        hits[0]["_source"].pop("coat_color")
        self.assertEqual([r["animalId"] for r in rank_candidates(hits, coat_color=black, color_weight=.12)], [1, 2])

    def test_ranking_is_opt_in_and_invalid_weight_is_not_silently_enabled(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(ranking_weight(), 0)
        for value in ("nan", "inf", "-0.1", "0.3", "typo"):
            with self.subTest(value=value), patch.dict(os.environ, {"LOST_SEARCH_COAT_COLOR_WEIGHT": value}):
                with self.assertRaises(RuntimeError): ranking_weight()
