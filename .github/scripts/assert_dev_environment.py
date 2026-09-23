#!/usr/bin/env python3
"""Fail closed before an image bot edits any infrastructure checkout."""
import json
from pathlib import Path
import sys


def verify(root, service):
    root = Path(root)
    contract = json.loads((root / 'environments/environment-contract.json').read_text())
    if contract.get('version') != 2 or contract.get('runtime') != 'local-compose' or contract.get('composeProject') != 'pawbridge-dev' or contract.get('branch') != 'dev':
        raise ValueError('isolated dev contract is required before publishing an Infra PR')
    expected = 'environments/dev/isolated-values/' + service + '.yaml'
    if contract['services'][service]['devValues'] != expected:
        raise ValueError('image updates must target isolated dev values only')
    path = root / expected
    if not path.is_file() or path.is_symlink() or root.resolve() not in path.resolve().parents:
        raise ValueError('dev values must be a regular file inside the infra checkout')
    return expected


if __name__ == '__main__':
    print(verify(sys.argv[1], sys.argv[2]))
