"""Exercise publishing-step cwd and obsolete-PR branch scoping without GitHub writes."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
CASES = [('python-ai-image-ci.yml', 'python-ai-service', 'chore/python-ai-image-')]

class PublishingEnvironmentTest(unittest.TestCase):
    def test_guard_runs_from_effective_workflow_directory(self):
        for name, service, legacy in CASES:
            with self.subTest(workflow=name), tempfile.TemporaryDirectory() as temporary:
                workspace = Path(temporary)
                (workspace / '.github/scripts').mkdir(parents=True)
                shutil.copyfile(ROOT / '.github/scripts/assert_dev_environment.py', workspace / '.github/scripts/assert_dev_environment.py')
                relative = 'environments/dev/isolated-values/' + service + '.yaml'
                values = workspace / 'infra' / relative
                values.parent.mkdir(parents=True)
                values.write_text('image: {}\n')
                (workspace / 'infra/environments/environment-contract.json').write_text(json.dumps({'version': 2, 'runtime': 'local-compose', 'composeProject': 'pawbridge-dev', 'branch': 'dev', 'services': {service: {'devValues': relative}}}))
                (workspace / service).mkdir()
                text = (ROOT / '.github/workflows' / name).read_text()
                guard = next(block for block in re.split(r'\n      - name: ', text) if block.startswith('Require isolated dev destination\n'))
                override = re.search(r'^        working-directory: (.+)$', guard, re.M)
                default = re.search(r'^        working-directory: (.+)$', text.split('    steps:')[0], re.M)
                directory = (override or default).group(1) if (override or default) else '${{ github.workspace }}'
                directory = directory.replace('${{ github.workspace }}', str(workspace))
                cwd = Path(directory) if Path(directory).is_absolute() else workspace / directory
                command = re.search(r'^        run: (.+)$', guard, re.M).group(1)
                result = subprocess.run(['bash', '-euc', command], cwd=cwd, env=dict(os.environ, TARGET_SERVICE=service), capture_output=True, text=True, timeout=10)
                self.assertEqual(0, result.returncode, result.stderr)
                self.assertEqual(relative, result.stdout.strip())

    def test_dev_cleanup_excludes_legacy_branches_and_preserves_commit_suffix(self):
        sha = 'a' * 40
        for name, service, legacy in CASES:
            with self.subTest(workflow=name):
                text = (ROOT / '.github/workflows' / name).read_text()
                branch = re.search(r'^          branch=(.+)$', text, re.M).group(0).strip()
                suffix = re.search(r'^            obsolete_sha=(.+)$', text, re.M).group(0).strip()
                command = branch + '\nobsolete_branch="$branch"\n' + suffix + '\nprintf "%s\\n%s\\n" "$branch" "$obsolete_sha"'
                result = subprocess.run(['bash', '-euc', command], env=dict(os.environ, GITHUB_SHA=sha, SERVICE_DIRECTORY=service), capture_output=True, text=True, check=True)
                generated, extracted = result.stdout.splitlines()
                normalized = text.replace('\\"', '"').replace('${SERVICE_DIRECTORY}', service)
                pattern = re.search(r'test\("(\^chore/[^"\n]+)"\)', normalized).group(1)
                self.assertRegex(generated, pattern)
                self.assertIsNone(re.fullmatch(pattern, legacy + sha[:12]))
                self.assertEqual(sha[:12], extracted)
