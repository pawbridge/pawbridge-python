import io
import os
import unittest
from datetime import date
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from app.routers import lost_search as api
from app.services.lost_search import decode_photo, InvalidPhoto, rank_candidates, auxiliary_evidence


def photo(format="PNG"):
    buf = io.BytesIO()
    with Image.new("RGB", (16, 16), "white") as img:
        img.save(buf, format=format)
    return buf.getvalue()


def hit(id, score, **fields):
    return {"_score": score + 1, "_source": {"id": id, **fields}}


class PhotoTest(unittest.TestCase):
    def test_jpeg_and_png_decode_to_rgb(self):
        for fmt in ("PNG", "JPEG"):
            with self.subTest(fmt=fmt), decode_photo(photo(fmt)) as img:
                self.assertEqual(img.mode, "RGB")
                self.assertEqual(img.size, (16, 16))

    def test_mpo_uses_primary_still_image(self):
        buffer = io.BytesIO()
        with Image.new("RGB", (16, 12), "red") as primary, Image.new("RGB", (16, 12), "blue") as auxiliary:
            primary.save(buffer, format="MPO", save_all=True, append_images=[auxiliary])
        with decode_photo(buffer.getvalue()) as image:
            self.assertEqual(image.size, (16, 12))
            red, _, blue = image.getpixel((8, 6))
            self.assertGreater(red - blue, 200)

    def test_other_formats_and_corrupt_bytes_are_rejected(self):
        for data in (photo("GIF"), b"not an image", b"", photo()[:30]):
            with self.subTest(data_length=len(data)), self.assertRaises(InvalidPhoto):
                decode_photo(data)

    def test_exif_orientation_is_applied_and_metadata_removed(self):
        buffer = io.BytesIO()
        with Image.new("RGB", (12, 8), "white") as image:
            exif = Image.Exif()
            exif[274] = 6
            exif[315] = "test-owner"
            image.save(buffer, format="JPEG", exif=exif)
        with decode_photo(buffer.getvalue()) as image:
            self.assertEqual(image.size, (8, 12))
            self.assertEqual(image.info, {})
            self.assertEqual(dict(image.getexif()), {})

    def test_animated_png_is_rejected(self):
        buffer = io.BytesIO()
        with Image.new("RGB", (16, 16), "white") as first, Image.new("RGB", (16, 16), "black") as second:
            first.save(buffer, format="PNG", save_all=True, append_images=[second], duration=100, loop=0)
        with self.assertRaises(InvalidPhoto):
            decode_photo(buffer.getvalue())

    def test_pixel_limit_checked_before_decode(self):
        with patch("app.services.lost_search.MAX_PIXELS", 100):
            with self.assertRaises(InvalidPhoto):
                decode_photo(photo())


class RankingTest(unittest.TestCase):
    def test_no_conditions_preserves_visual_order(self):
        ranked = rank_candidates([hit(1, .9), hit(2, .8)])
        self.assertEqual([c["animalId"] for c in ranked], [1, 2])
        self.assertEqual(ranked[0]["matchedEvidence"], [])

    def test_auxiliary_match_promotes_close_candidate_without_excluding_others(self):
        ranked = rank_candidates([hit(1, .8), hit(2, .795, happen_place="상주시 동문동")], region="상주시")
        self.assertEqual([c["animalId"] for c in ranked], [2, 1])

    def test_auxiliary_matches_do_not_overturn_large_visual_gap(self):
        ranked = rank_candidates([hit(1, .9), hit(2, .5, happen_date="2026-09-10", happen_place="상주시", color="흰색")], date(2026, 9, 8), "상주시", "흰색")
        self.assertEqual([c["animalId"] for c in ranked], [1, 2])

    def test_unknown_and_mismatched_metadata_do_not_remove_candidates(self):
        ranked = rank_candidates([hit(1, .9, happen_date="unknown"), hit(2, .8, happen_date="2020-01-01", status="ADOPTED")], date(2026, 9, 8), "상주시", "흰색")
        self.assertEqual(len(ranked), 2)
        self.assertTrue(all(not c["matchedEvidence"] for c in ranked))

    def test_blank_region_is_not_matching_evidence(self):
        self.assertEqual(auxiliary_evidence({"happen_place": "상주시"}, region="   "), [])

    def test_shelter_address_is_not_discovery_evidence(self):
        self.assertEqual(auxiliary_evidence({"shelter_address": "상주시"}, region="상주시"), [])

    def test_result_size_is_bounded(self):
        self.assertEqual(len(rank_candidates([hit(i, .9) for i in range(30)])), 20)


class LostApiTest(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(api.router, prefix="/internal/animals")
        self.client = TestClient(app)
        self.env = patch.dict(os.environ, {"INTERNAL_API_KEY": "test-key"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def post(self, fields=None, data=None, headers=None):
        return self.client.post("/internal/animals/lost-candidates", files={"image": ("photo.png", data or photo(), "image/png")}, data=fields or {"species": "DOG"}, headers=headers if headers is not None else {"X-Internal-Api-Key": "test-key"})

    def test_valid_request_passes_all_conditions_to_search(self):
        with patch.object(api, "search_photo", return_value={"candidates": []}) as search:
            response = self.post({"species": "DOG", "lostDate": "2026-09-08", "region": "상주시", "description": "갈색 귀"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(search.call_args.args[1:], ("DOG", date(2026,9,8), "상주시", "갈색 귀"))

    def test_unauthenticated_request_does_not_search(self):
        with patch.object(api, "search_photo") as search:
            self.assertEqual(self.post(headers={}).status_code, 401)
        search.assert_not_called()

    def test_invalid_species_date_and_long_description_do_not_search(self):
        for fields in ({"species":"ETC"}, {"species":"DOG","lostDate":"bad"}, {"species":"CAT","description":"x"*501}):
            with self.subTest(fields=str(fields)[:40]), patch.object(api,"search_photo") as search:
                self.assertEqual(self.post(fields).status_code,422)
                search.assert_not_called()

    def test_empty_optional_fields_are_allowed(self):
        with patch.object(api, "search_photo", return_value={"candidates": []}):
            self.assertEqual(self.post({"species":"CAT","lostDate":"","region":"","description":""}).status_code,200)

    def test_whitespace_optional_text_is_normalized(self):
        with patch.object(api, "search_photo", return_value={"candidates": []}) as search:
            response = self.post({"species": "DOG", "region": "  ", "description": "  갈색 귀  "})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(search.call_args.args[3:], (None, "갈색 귀"))

    def test_image_and_request_size_limits(self):
        with patch.object(api,"search_photo") as search:
            self.assertEqual(self.post(data=b"x"*(5*1024*1024+1)).status_code,413)
            self.assertEqual(self.post(data=b"x"*(6*1024*1024)).status_code,413)
        search.assert_not_called()

    def test_failure_is_not_empty_success(self):
        for error, expected in [(InvalidPhoto("bad photo"),422),(RuntimeError("upstream"),503)]:
            with patch.object(api,"search_photo", side_effect=error):
                response=self.post()
            self.assertEqual(response.status_code,expected)

    def test_busy_search_is_rejected(self):
        with patch.object(api._gate,"locked", return_value=True), patch.object(api,"search_photo") as search:
            self.assertEqual(self.post().status_code,503)
        search.assert_not_called()


class RetrievalContractTest(unittest.TestCase):
    def test_query_has_species_but_no_status_date_or_region_exclusion(self):
        import sys
        import types
        from unittest.mock import MagicMock, create_autospec
        from elasticsearch import Elasticsearch
        from app.services.lost_search import search_photo
        encoder = types.ModuleType("app.services.dinov3")
        encoder.get_encoder = MagicMock()
        encoder.get_encoder.return_value.encode_with_metadata.return_value = types.SimpleNamespace(vector=[.1, .2], model_version="test-model", focus_status="original_multiple_animals", animal_vector=None, coat_color=None)
        encoder.gallery_index = lambda: "animals-lost-dinov3-large-v1"
        encoder.MODEL_VERSION = "test-model"
        module = types.ModuleType("app.es.client")
        module.INDEX_NAME = "animals"
        module.es = create_autospec(Elasticsearch, instance=True)
        module.es.options.return_value = create_autospec(Elasticsearch, instance=True)
        module.es.options.return_value.search.return_value = {"hits":{"hits":[hit(7,.8,status="ADOPTED")]}}
        with patch.dict(sys.modules, {"app.services.dinov3":encoder, "app.es.client":module}):
            result = search_photo(photo(), "DOG", date(2026,9,8), "상주시", "흰색")
        args = module.es.options.return_value.search.call_args.kwargs
        self.assertEqual(args["query"]["script_score"]["query"]["bool"]["filter"], [
            {"term":{"species":"DOG"}}, {"term":{"model_version":"test-model"}}, {"exists":{"field":"image_vector"}}, {"exists":{"field":"id"}}])
        self.assertEqual(result["candidates"][0]["animalId"],7)
        self.assertEqual(args["size"],200)
        self.assertEqual(args["index"], "animals-lost-dinov3-large-v1")
        module.es.options.assert_called_once_with(request_timeout=15, max_retries=0)
        module.es.update.assert_not_called()
        module.es.index.assert_not_called()

    def test_animal_channel_is_only_combined_when_the_gallery_document_has_it(self):
        import sys
        import types
        from unittest.mock import MagicMock
        from app.services.lost_search import search_photo, ANIMAL_REGION_WEIGHT
        encoder = types.ModuleType("app.services.dinov3")
        encoder.get_encoder = MagicMock()
        encoder.get_encoder.return_value.encode_with_metadata.return_value = types.SimpleNamespace(
            vector=[1., 0.], animal_vector=[0., 1.], model_version="dual-v2", focus_status="animal_mask", coat_color=None)
        encoder.gallery_index = lambda: "animals-lost-dinov3-focus-eval-v2"
        module = types.ModuleType("app.es.client")
        module.es = MagicMock()
        module.es.options.return_value.search.return_value = {"hits": {"hits": []}}
        with patch.dict(sys.modules, {"app.services.dinov3": encoder, "app.es.client": module}):
            search_photo(photo(), "DOG")
        args = module.es.options.return_value.search.call_args.kwargs
        script = args["query"]["script_score"]["script"]
        self.assertEqual(script["params"], {"vector": [1.,0.], "animal": [0.,1.], "weight": ANIMAL_REGION_WEIGHT})
        self.assertIn("doc['animal_vector'].size() != 0", script["source"])
        self.assertIn("return 1.0 + original;", script["source"])
        self.assertIn({"term": {"model_version": "dual-v2"}}, args["query"]["script_score"]["query"]["bool"]["filter"])
        self.assertEqual(encoder.get_encoder.return_value.encode_with_metadata.call_args.args[1], "DOG")

    def test_partial_elasticsearch_response_is_failure(self):
        import sys
        import types
        from unittest.mock import MagicMock
        from app.services.lost_search import search_photo
        encoder = types.ModuleType("app.services.dinov3")
        encoder.get_encoder = MagicMock()
        encoder.get_encoder.return_value.encode_with_metadata.return_value = types.SimpleNamespace(vector=[.1, .2], model_version="test-model", focus_status="original_multiple_animals", animal_vector=None, coat_color=None)
        encoder.gallery_index = lambda: "animals-lost-dinov3-large-v1"
        encoder.MODEL_VERSION = "test-model"
        module = types.ModuleType("app.es.client")
        module.INDEX_NAME = "animals"
        module.es = MagicMock()
        for response in ({"timed_out":True},{"_shards":{"failed":1}}):
            module.es.options.return_value.search.return_value=response
            with patch.dict(sys.modules,{"app.services.dinov3":encoder,"app.es.client":module}):
                with self.assertRaises(RuntimeError):
                    search_photo(photo(),"CAT")

    def test_query_and_gallery_color_features_drive_returned_order(self):
        import sys
        import types
        from unittest.mock import MagicMock
        from app.services.dinov3 import AnimalEmbedding
        from app.services.lost_search import search_photo
        from tests.test_coat_color import descriptor
        dark, tan = descriptor((45, 40, 35)), descriptor((165, 110, 60))
        encoder = MagicMock()
        encoder.encode_with_metadata.return_value = AnimalEmbedding([1., 0.], "test", "animal_mask", [1., 0.], dark)
        module = types.ModuleType("app.es.client"); module.es = MagicMock()
        module.es.options.return_value.search.return_value = {"hits": {"hits": [hit(1, .88, coat_color=tan), hit(2, .87, coat_color=dark)]}}
        with patch("app.services.dinov3.get_encoder", return_value=encoder), patch("app.services.dinov3.gallery_index", return_value="test-gallery"), patch.dict(sys.modules, {"app.es.client": module}), patch.dict(os.environ, {"LOST_SEARCH_COAT_COLOR_WEIGHT": ".12"}):
            result = search_photo(photo(), "DOG")
        self.assertEqual([c["animalId"] for c in result["candidates"]], [2, 1])
        self.assertIn("coat_color", module.es.options.return_value.search.call_args.kwargs["source"])
