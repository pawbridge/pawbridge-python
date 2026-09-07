import importlib
import os
import sys
import unittest
from unittest.mock import create_autospec, patch


class ElasticsearchClientConfigurationTest(unittest.TestCase):
    def setUp(self):
        with patch.dict(os.environ, {"ES_URL": "http://localhost:9200"}, clear=True):
            sys.modules.pop("app.es.client", None)
            self.client_module = importlib.import_module("app.es.client")

    def test_https_connection_uses_ca_and_basic_authentication(self):
        environment = {
            "ES_URL": "https://store-search-es-http.databases.svc:9200",
            "ES_USERNAME": "animal-reader",
            "ES_PASSWORD": "secret",
            "ES_CA_CERT_PATH": "/etc/animal-search-ca/ca.crt",
        }

        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(self.client_module, "Elasticsearch") as elasticsearch,
        ):
            self.client_module.create_elasticsearch_client()

        elasticsearch.assert_called_once_with(
            environment["ES_URL"],
            basic_auth=(environment["ES_USERNAME"], environment["ES_PASSWORD"]),
            ca_certs=environment["ES_CA_CERT_PATH"],
        )

    def test_https_connection_requires_basic_authentication(self):
        environment = {
            "ES_URL": "https://store-search-es-http.databases.svc:9200",
            "ES_CA_CERT_PATH": "/etc/animal-search-ca/ca.crt",
        }

        with patch.dict(os.environ, environment, clear=True):
            with self.assertRaisesRegex(RuntimeError, "requires ES_USERNAME and ES_PASSWORD"):
                self.client_module.create_elasticsearch_client()

    def test_https_connection_requires_ca_certificate(self):
        environment = {
            "ES_URL": "https://store-search-es-http.databases.svc:9200",
            "ES_USERNAME": "animal-reader",
            "ES_PASSWORD": "secret",
        }

        with patch.dict(os.environ, environment, clear=True):
            with self.assertRaisesRegex(RuntimeError, "requires ES_CA_CERT_PATH"):
                self.client_module.create_elasticsearch_client()

    def test_save_vector_uses_elasticsearch_8_doc_parameter(self):
        elasticsearch = create_autospec(self.client_module.Elasticsearch, instance=True)

        with patch.object(self.client_module, "es", elasticsearch):
            saved = self.client_module.save_animal_vector(42, [0.1, 0.2])

        self.assertTrue(saved)
        elasticsearch.update.assert_called_once_with(
            index="animals",
            id="42",
            doc={"image_vector": [0.1, 0.2]},
        )

    def test_vector_search_uses_elasticsearch_8_query_parameters(self):
        elasticsearch = create_autospec(self.client_module.Elasticsearch, instance=True)
        elasticsearch.search.return_value = {
            "hits": {"hits": [{"_source": {"id": 43}}]}
        }

        with patch.object(self.client_module, "es", elasticsearch):
            result = self.client_module.knn_search(
                [0.1, 0.2],
                exclude_id=42,
                species="DOG",
                k=6,
            )

        self.assertEqual(result, [43])
        search_arguments = elasticsearch.search.call_args.kwargs
        self.assertEqual(search_arguments["index"], "animals")
        self.assertEqual(search_arguments["size"], 7)
        self.assertEqual(search_arguments["source"], ["id"])
        self.assertEqual(
            search_arguments["query"]["script_score"]["script"]["params"]["query_vector"],
            [0.1, 0.2],
        )


if __name__ == "__main__":
    unittest.main()
