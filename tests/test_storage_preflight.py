"""Backend dispatch and redacted startup failures; no GPU or external I/O."""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock

spec = importlib.util.spec_from_file_location("storage_preflight", Path(__file__).parents[1] / "deploy/check_storage.py")
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)

class StartupTests(unittest.TestCase):
    def test_postgresql_never_checks_elasticsearch(self):
        with patch.dict(os.environ, {"LOST_STORAGE_BACKEND": "postgresql"}), patch.object(preflight, "check_postgresql") as pg, patch.object(preflight, "check_elasticsearch") as es, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(0, preflight.main())
            pg.assert_called_once_with()
            es.assert_not_called()

    def test_default_preserves_elasticsearch_and_does_not_import_pg_driver(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(preflight, "check_postgresql") as pg, patch.object(preflight, "check_elasticsearch") as es, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(0, preflight.main())
            es.assert_called_once_with()
            pg.assert_not_called()

    def test_unknown_backend_and_driver_failures_are_closed_and_redacted(self):
        for backend in ("typo", "postgresql", "elasticsearch"):
            with self.subTest(backend=backend), patch.dict(os.environ, {"LOST_STORAGE_BACKEND": backend}), patch.object(preflight, "check_postgresql", side_effect=RuntimeError("password=do-not-print")) as pg, patch.object(preflight, "check_elasticsearch", side_effect=RuntimeError("signed-url-do-not-print")) as es, contextlib.redirect_stderr(io.StringIO()) as output:
                self.assertEqual(1, preflight.main())
                self.assertNotIn("do-not-print", output.getvalue())
                if backend == "typo":
                    pg.assert_not_called()
                    es.assert_not_called()

    def test_es_red_or_timed_out_health_prevents_start(self):
        for health in ('{"status":"red"}', '{"status":"yellow","timed_out":true}'):
            response = MagicMock()
            response.__enter__.return_value = io.StringIO(health)
            with self.subTest(health=health), patch.dict(os.environ, {"ES_URL":"http://127.0.0.1:19201"}), patch.object(preflight.urllib.request, "urlopen", return_value=response):
                with self.assertRaises(RuntimeError):
                    preflight.check_elasticsearch()

if __name__ == "__main__":
    unittest.main()
