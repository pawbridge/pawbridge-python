#!/usr/bin/env python3
"""Check a dev GPU process before loading models; never contacts a database."""
import os
from pathlib import Path
from urllib.parse import urlsplit, unquote


def validate(env, root=None):
    root = Path(root or Path.home()/'pawbridge-ai-dev').resolve()
    if env.get('PAWBRIDGE_ENVIRONMENT') != 'dev':
        raise ValueError('explicit dev environment required')
    if env.get('LOST_STORAGE_BACKEND') != 'postgresql':
        raise ValueError('dev must use its PostgreSQL gallery')
    dsn = urlsplit(env.get('LOST_PG_DSN', ''))
    if (dsn.scheme not in ('postgresql', 'postgres') or dsn.hostname not in ('localhost', '127.0.0.1')
            or dsn.port in (None, 5432) or dsn.path != '/pawbridge' or dsn.query or dsn.fragment
            or not unquote(dsn.username or '').startswith('pawbridge_dev_')):
        raise ValueError('dedicated dev role and explicit loopback forwarded port required')
    if env.get('LOST_SEARCH_INDEX') != 'animals-lost-dinov3-sam3-dev-v1':
        raise ValueError('dedicated dev gallery required')
    for key in ('LOST_GALLERY_STATE_DIR', 'LOST_GALLERY_PHOTO_ROOT'):
        path = Path(env.get(key, '')).resolve()
        if root not in path.parents:
            raise ValueError('dev state paths must remain under the dev runtime root')
    if env.get('LOST_GALLERY_SYNC_ENABLED', 'false').lower() != 'false':
        raise ValueError('automatic gallery mutation is disabled in the initial dev profile')
    if env.get('LOST_DEV_GPU_ENABLED') != 'true':
        raise ValueError('GPU allocation must be explicitly enabled after capacity review')
    if not env.get('INTERNAL_API_KEY'):
        raise ValueError('dedicated internal authentication required')


if __name__ == '__main__':
    validate(os.environ)
    print('dev runtime configuration verified; DB identity and GPU capacity still require runtime verification')
