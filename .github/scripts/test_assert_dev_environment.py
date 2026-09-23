import json
from pathlib import Path
import tempfile
import unittest
from assert_dev_environment import verify

class EnvironmentContractTest(unittest.TestCase):
    def test_rejects_legacy_and_production_routes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'environments').mkdir()
            for runtime, route in [('kubernetes', 'environments/dev/isolated-values/animal-service.yaml'), ('local-compose', 'environments/prod/values/animal-service.yaml')]:
                contract = {'version': 2, 'runtime': runtime, 'composeProject': 'pawbridge-dev', 'branch': 'dev', 'services': {'animal-service': {'devValues': route}}}
                (root/'environments/environment-contract.json').write_text(json.dumps(contract))
                with self.assertRaises(ValueError): verify(root, 'animal-service')

    def test_accepts_only_the_existing_isolated_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            relative = 'environments/dev/isolated-values/animal-service.yaml'
            path = root/relative
            path.parent.mkdir(parents=True)
            (root/'environments/environment-contract.json').write_text(json.dumps({'version': 2, 'runtime': 'local-compose', 'composeProject': 'pawbridge-dev', 'branch': 'dev', 'services': {'animal-service': {'devValues': relative}}}))
            with self.assertRaises(ValueError): verify(root, 'animal-service')
            path.write_text('replicaCount: 0')
            self.assertEqual(relative, verify(root, 'animal-service'))
            path.unlink(); path.symlink_to(root/'environments/environment-contract.json')
            with self.assertRaises(ValueError): verify(root, 'animal-service')
