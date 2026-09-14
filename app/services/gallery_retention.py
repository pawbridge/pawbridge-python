"""Retain active/previous builds; clean only journaled, unaliased owned indices."""
import json
import re
from pathlib import Path

from app.services.gallery_source import atomic_json
from app.services.lost_gallery import PREFIX, CONTRACT, FOCUS_VERSION, gallery_target


class GalleryRetention:
    def __init__(self, es, state_dir, alias):
        if not re.fullmatch(PREFIX + r'[a-z0-9][a-z0-9-]{0,60}', alias):
            raise ValueError('Invalid retention alias')
        self.es = es.options(request_timeout=30, max_retries=0)
        self.path = Path(state_dir) / (alias + '-retention.json')
        self.journal = {'known': [], 'published': []}
        if self.path.exists():
            if self.path.is_symlink() or self.path.stat().st_size > 16384:
                raise ValueError('Invalid retention journal')
            value = json.loads(self.path.read_text())
            if (set(value) != {'known', 'published'} or any(not isinstance(value[k], list) for k in value)
                    or any(not isinstance(name, str) or not re.fullmatch(PREFIX + r'build-[a-f0-9]{24}', name)
                           for names in value.values() for name in names)
                    or len(value['known']) > 9 or len(value['published']) > 2
                    or not set(value['published']).issubset(value['known'])):
                raise ValueError('Invalid retention journal')
            self.journal = value

    def save(self):
        atomic_json(self.path, self.journal)

    def prune(self, keep):
        for name in list(self.journal['known']):
            if name in keep:
                continue
            if self.es.indices.exists(index=name):
                # Never delete an index in use by any alias, including another local gallery.
                if self.es.indices.get_alias(index=name)[name].get('aliases'):
                    continue
                meta = self.es.indices.get_mapping(index=name)[name]['mappings'].get('_meta', {})
                if meta.get('contract') != CONTRACT or meta.get('model_version') != FOCUS_VERSION:
                    raise RuntimeError('Refusing to clean an index with a different owner')
                self.es.indices.delete(index=name)
            self.journal['known'].remove(name)
            self.save()

    def prepare(self, snapshot_hash):
        target = gallery_target(snapshot_hash)
        self.prune(set(self.journal['published']) | {target})
        if target not in self.journal['known']:
            if len(self.journal['known']) >= 8:
                raise RuntimeError('Gallery retention is blocked by referenced builds')
            self.journal['known'].append(target)
            self.save()  # Record before creation, so interrupted staging can be cleaned next time.

    def published(self, index):
        self.journal['published'] = [name for name in self.journal['published'] if name != index] + [index]
        self.journal['published'] = self.journal['published'][-2:]
        self.save()
        self.prune(set(self.journal['published']))
