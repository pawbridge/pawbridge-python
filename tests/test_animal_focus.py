import unittest
import numpy as np
from PIL import Image
from app.services.animal_focus import choose_detection, prepare_focus, square_image, BACKGROUND


class AnimalFocusTest(unittest.TestCase):
    def test_one_confident_target_is_selected_but_another_plausible_target_is_not_ignored(self):
        self.assertEqual(choose_detection([17, 18], [.99, .9], "DOG"), (1, "animal_mask"))
        self.assertEqual(choose_detection([17, 18], [.99, .9], "CAT"), (0, "animal_mask"))
        for scores in ([.9, .8], [.9, .4]):
            self.assertEqual(choose_detection([18, 18], scores, "DOG"), (None, "original_multiple_animals"))

    def test_missing_or_uncertain_animal_preserves_original_instead_of_picking_another_species(self):
        for labels, scores in [([], []), ([17], [.99]), ([18], [.6])]:
            self.assertEqual(choose_detection(labels, scores, "DOG"), (None, "original_no_confident_animal"))
        with self.assertRaises(ValueError):
            choose_detection([18], [.9], "ETC")
        with self.assertRaises(RuntimeError):
            choose_detection([18], [float("nan")], "DOG")

    def test_full_portrait_keeps_top_and_bottom_in_a_square_without_mutating_source(self):
        with Image.new("RGB", (40, 80), "red") as image:
            image.paste("blue", (0, 60, 40, 80))
            before = image.tobytes()
            with square_image(image) as result:
                self.assertEqual(result.size, (256, 256))
                self.assertEqual(result.getpixel((128, 0)), (255, 0, 0))
                self.assertEqual(result.getpixel((128, 255)), (0, 0, 255))
                self.assertEqual(result.getpixel((0, 128)), BACKGROUND)
            self.assertEqual(image.tobytes(), before)

    def test_single_valid_mask_suppresses_background_without_rewriting_source(self):
        y, x = np.mgrid[:100, :100]
        mask = (((x-50)/39)**2 + ((y-50)/39)**2 < 1).astype(float)
        with Image.new("RGB", (100, 100), "red") as image:
            before = image.tobytes()
            result = prepare_focus(image, [10, 10, 90, 90], mask)
            with result.image as view:
                self.assertEqual(result.status, "animal_mask")
                self.assertEqual(view.getpixel((128, 128)), (255, 0, 0))
                self.assertEqual(view.getpixel((0, 0)), BACKGROUND)
            self.assertEqual(image.tobytes(), before)

    def test_small_region_or_partial_mask_uses_full_photo_and_records_fallback(self):
        partial = np.zeros((100, 100)); partial[20:40, 20:80] = 1
        for box, mask, reason in [([10, 20, 90, 50], partial, "original_small_region"),
                                  ([10, 10, 90, 90], partial, "original_suspect_mask")]:
            with Image.new("RGB", (100, 100), "red") as image:
                result = prepare_focus(image, box, mask)
                with result.image as view:
                    self.assertEqual(result.status, reason)
                    self.assertEqual(view.getpixel((0, 0)), (255, 0, 0))

    def test_invalid_detector_geometry_is_an_error_not_a_successful_fallback(self):
        with Image.new("RGB", (100, 100)) as image:
            for box, mask in [([10, 10, 110, 90], np.zeros((100,100))),
                              ([10, 10, 90, 90], np.zeros((50,50))),
                              ([10, 10, 90, 90], np.full((100,100), float("nan")))]:
                with self.subTest(box=box), self.assertRaises(RuntimeError):
                    prepare_focus(image, box, mask)


class DetectionGroupingTest(unittest.TestCase):
    def test_nested_body_part_detections_keep_the_larger_confident_animal_region(self):
        body = np.zeros((100,100)); body[10:90,10:90] = 1
        head = np.zeros((100,100)); head[20:40,30:70] = 1
        boxes = [[10,10,90,90],[30,20,70,40]]
        # The high-scoring head must not replace the larger, confident full-body region.
        self.assertEqual(choose_detection([18,18],[.9,.99],"DOG",boxes,{0:body,1:head}), (0,"animal_mask"))

    def test_nested_boxes_with_separate_foregrounds_and_collages_remain_ambiguous(self):
        left = np.zeros((100,100)); left[10:90,10:40] = 1
        right = np.zeros((100,100)); right[20:80,50:80] = 1
        for boxes in ([[10,10,90,90],[20,20,80,80]], [[10,10,40,90],[50,20,80,80]]):
            self.assertEqual(choose_detection([17,17],[.95,.9],"CAT",boxes,{0:left,1:right}),
                             (None,"original_multiple_animals"))

    def test_clear_animal_is_not_rejected_just_because_it_occupies_a_small_photo_fraction(self):
        y,x = np.mgrid[:500,:500]
        mask = (((x-250)/49)**2 + ((y-250)/49)**2 < 1).astype(float)
        with Image.new("RGB",(500,500),"red") as image:
            result=prepare_focus(image,[200,200,300,300],mask)
            with result.image as view:
                self.assertEqual(result.status,"animal_mask")
                self.assertEqual(view.getpixel((128,128)),(255,0,0))
