import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("update_python_ai_image_values.py")
SPEC = importlib.util.spec_from_file_location("update_python_ai_image_values", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


REPOSITORY = "dorosiya/pawbridge-python-ai-service"
TAG = "sha-" + "a" * 40
DIGEST = "sha256:" + "b" * 64


class UpdatePythonAiImageValuesTest(unittest.TestCase):
    def test_adds_image_mapping_when_missing(self) -> None:
        original = "env:\n  LLM_PROVIDER: gemini\n"

        updated = MODULE.update_values(original, REPOSITORY, TAG, DIGEST)

        self.assertEqual(
            updated,
            "image:\n"
            f"  repository: {REPOSITORY}\n"
            f"  tag: {TAG}\n"
            f"  digest: {DIGEST}\n"
            "\n"
            "env:\n"
            "  LLM_PROVIDER: gemini\n",
        )

    def test_updates_managed_keys_and_preserves_other_image_settings(self) -> None:
        original = (
            "image:\n"
            "  repository: old/repository\n"
            "  tag: k8s-v1\n"
            "  pullPolicy: Always\n"
            "env:\n"
            "  LLM_PROVIDER: gemini\n"
        )

        updated = MODULE.update_values(original, REPOSITORY, TAG, DIGEST)

        self.assertEqual(
            updated,
            "image:\n"
            f"  repository: {REPOSITORY}\n"
            f"  tag: {TAG}\n"
            f"  digest: {DIGEST}\n"
            "  pullPolicy: Always\n"
            "env:\n"
            "  LLM_PROVIDER: gemini\n",
        )

    def test_update_is_idempotent(self) -> None:
        original = "secretRef: python-ai-service-secrets\n"

        first = MODULE.update_values(original, REPOSITORY, TAG, DIGEST)
        second = MODULE.update_values(first, REPOSITORY, TAG, DIGEST)

        self.assertEqual(second, first)

    def test_preserves_crlf_and_final_newline(self) -> None:
        original = "image:\r\n  pullPolicy: Always\r\nenv: {}\r\n"

        updated = MODULE.update_values(original, REPOSITORY, TAG, DIGEST)

        self.assertTrue(updated.endswith("\r\n"))
        self.assertNotIn("\n", updated.replace("\r\n", ""))
        self.assertIn(f"  digest: {DIGEST}\r\n", updated)

    def test_rejects_invalid_immutable_reference(self) -> None:
        invalid_references = [
            ("repository with spaces", TAG, DIGEST),
            (REPOSITORY, "latest", DIGEST),
            (REPOSITORY, TAG, "sha256:short"),
        ]

        for repository, tag, digest in invalid_references:
            with self.subTest(repository=repository, tag=tag, digest=digest):
                with self.assertRaises(ValueError):
                    MODULE.update_values("env: {}\n", repository, tag, digest)

    def test_rejects_duplicate_managed_keys(self) -> None:
        original = "image:\n  tag: first\n  tag: second\nenv: {}\n"

        with self.assertRaisesRegex(ValueError, "duplicate image key: tag"):
            MODULE.update_values(original, REPOSITORY, TAG, DIGEST)

    def test_file_update_preserves_crlf(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            values_file = Path(directory) / "values.yaml"
            values_file.write_bytes(b"image:\r\n  pullPolicy: Always\r\nenv: {}\r\n")

            changed = MODULE.update_file(values_file, REPOSITORY, TAG, DIGEST)
            content = values_file.read_bytes()

            self.assertTrue(changed)
            self.assertIn(b"\r\n", content)
            self.assertNotIn(b"\n", content.replace(b"\r\n", b""))


if __name__ == "__main__":
    unittest.main()
