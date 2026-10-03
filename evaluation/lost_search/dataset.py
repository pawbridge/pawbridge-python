"""Bounded, local, content-addressed evaluation inputs with explicit ground truth."""
from dataclasses import dataclass
from datetime import date
import hashlib
import json
from pathlib import Path
import re

MAX_BYTES = 5 * 1024 * 1024
MAX_QUERIES = 30
MAX_GALLERY = 300
IDENTIFIER = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def json_read(path, limit=2 * 1024 * 1024):
    with Path(path).open('rb') as source:
        raw = source.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('JSON exceeds its size limit')
    return json.loads(raw, parse_constant=lambda value: fail('Non-finite JSON'))


def fail(message):
    raise ValueError(message)


def optional_date(value):
    if value is None:
        return None
    if not isinstance(value, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
        raise ValueError('Dates must be ISO dates or null')
    return date.fromisoformat(value)


@dataclass(frozen=True)
class Photo:
    id: str
    path: Path
    sha256: str
    role: str
    species: str
    partition: str
    identity: str | None
    capture: str | None
    truth: str
    metadata: dict


@dataclass(frozen=True)
class Dataset:
    name: str
    sha256: str
    photos: tuple[Photo, ...]

    @property
    def queries(self):
        return tuple(p for p in self.photos if p.role == 'query')

    @property
    def gallery(self):
        return tuple(p for p in self.photos if p.role == 'gallery')


def load_dataset(path):
    path = Path(path).resolve()
    raw = json_read(path)
    if (not isinstance(raw, dict) or raw.get('version') != 1
            or not isinstance(raw.get('name'), str) or not IDENTIFIER.fullmatch(raw['name'])):
        raise ValueError('Expected version 1 and a safe dataset name')
    entries = raw.get('photos')
    if not isinstance(entries, list) or not 2 <= len(entries) <= MAX_QUERIES + MAX_GALLERY:
        raise ValueError('Dataset must contain 2..330 images')
    photos, ids, hashes, animal_ids = [], set(), set(), set()
    partitions = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError('Each photo entry must be an object')
        image_id = entry.get('id', '')
        if not isinstance(image_id, str) or not IDENTIFIER.fullmatch(image_id) or image_id in ids:
            raise ValueError('Image IDs must be safe and unique')
        role, species = entry.get('role'), entry.get('species')
        partition = entry.get('partition')
        if role not in {'query', 'gallery'} or species not in {'DOG', 'CAT'}:
            raise ValueError('Invalid role or species')
        if partition not in {'tune', 'holdout', 'case'}:
            raise ValueError('Expected tune, holdout, or unlabelled case partition')
        name = entry.get('file')
        if not isinstance(name, str) or Path(name).is_absolute():
            raise ValueError('Image paths must be relative to the manifest')
        image_path = (path.parent / name).resolve()
        if not image_path.is_relative_to(path.parent) or not image_path.is_file():
            raise ValueError('Image must be a local file inside the dataset directory')
        if not 0 < image_path.stat().st_size <= MAX_BYTES:
            raise ValueError('Image exceeds the 5 MiB limit')
        sha = entry.get('sha256')
        if not isinstance(sha, str) or not re.fullmatch('[a-f0-9]{64}', sha) or digest(image_path) != sha:
            raise ValueError('Image content does not match its manifest hash')
        if sha in hashes:
            raise ValueError('Duplicate image bytes cannot be separate benchmark photos')
        # Reject invalid/oversized decoded images before reserving a GPU window.
        from app.services.lost_search import decode_photo
        with decode_photo(image_path.read_bytes()):
            pass
        identity, capture = entry.get('identity'), entry.get('capture')
        for identifier in (identity, capture):
            if identifier is not None and (not isinstance(identifier, str) or not IDENTIFIER.fullmatch(identifier)):
                raise ValueError('Invalid identity or capture ID')
        if identity is not None:
            if identity in partitions and partitions[identity] != partition:
                raise ValueError('One identity cannot cross evaluation partitions')
            partitions[identity] = partition
        truth = entry.get('truth', 'unknown')
        if truth not in {'present', 'absent', 'unknown'}:
            raise ValueError('Invalid ground-truth state')
        if role == 'query' and truth != 'unknown':
            if not identity or not capture or not entry.get('verification') or partition == 'case':
                raise ValueError('Scored queries require identity, capture, verification and a labelled partition')
        if partition == 'case' and truth != 'unknown':
            raise ValueError('Unlabelled cases are not accuracy evidence')
        if not isinstance(entry.get('metadata', {}), dict):
            raise ValueError('Photo metadata must be an object')
        metadata = dict(entry.get('metadata', {}))
        for field in ('happen_date', 'lost_date'):
            optional_date(metadata.get(field))
        if len(json.dumps(metadata)) > 8000:
            raise ValueError('Oversized photo metadata')
        for key in ('happen_place', 'color', 'special_mark', 'description', 'region', 'query_description'):
            if key in metadata and (not isinstance(metadata[key], str) or len(metadata[key]) > 1000):
                raise ValueError('Invalid text metadata')
        if type(metadata.get('include_adopted_or_returned', False)) is not bool:
            raise ValueError('Status selection must be boolean')
        if role == 'gallery':
            animal_id = metadata.get('animal_id')
            if type(animal_id) is not int or animal_id <= 0 or animal_id in animal_ids:
                raise ValueError('One positive, unique animal ID is required per gallery photo')
            if metadata.get('status') not in {'NOTICE', 'PROTECT', 'ADOPTED', 'RETURNED', 'EUTHANIZED', 'NATURAL_DEATH', 'OTHER', None}:
                raise ValueError('Invalid gallery status')
            animal_ids.add(animal_id)
        if not isinstance(entry.get('source'), str) or not entry['source'].strip():
            raise ValueError('Each image requires a source/provenance description')
        ids.add(image_id)
        hashes.add(sha)
        photos.append(Photo(image_id, image_path, sha, role, species, partition, identity, capture, truth, metadata))
    dataset = Dataset(raw['name'], digest(path), tuple(photos))
    if not 1 <= len(dataset.queries) <= MAX_QUERIES or not 1 <= len(dataset.gallery) <= MAX_GALLERY:
        raise ValueError('Expected 1..30 queries and 1..300 gallery images')
    for query in dataset.queries:
        positives = [p for p in dataset.gallery if query.identity and p.identity == query.identity]
        if query.truth == 'present' and not positives:
            raise ValueError('A present query needs a gallery positive')
        if query.truth == 'absent' and positives:
            raise ValueError('An absent query cannot have a gallery positive')
        if query.truth == 'present' and any(not p.capture or p.capture == query.capture or p.species != query.species for p in positives):
            raise ValueError('Positive photos must be same species and different captures')
    return dataset


def audit(dataset):
    return {'name': dataset.name, 'manifest_sha256': dataset.sha256,
            'scope': 'bounded_pilot_not_full_production_gallery',
            'query_count': len(dataset.queries), 'gallery_count': len(dataset.gallery),
            'present_queries': sum(p.truth == 'present' for p in dataset.queries),
            'absent_queries': sum(p.truth == 'absent' for p in dataset.queries),
            'unverified_queries': sum(p.truth == 'unknown' for p in dataset.queries),
            'total_source_bytes': sum(p.path.stat().st_size for p in dataset.photos)}
