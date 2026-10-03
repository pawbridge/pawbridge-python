"""Small on-disk feature records. No database, credentials, or model downloads."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from .dataset import digest, json_read

DIM = 1024
FEATURE_VERSION = 1


@dataclass(frozen=True)
class Variant:
    name: str
    precision: str = 'fp32'
    pooling: str = 'avg'
    size: int = 256
    source: str = 'legacy'

    def __post_init__(self):
        if self.precision not in {'fp16', 'fp32'} or self.pooling not in {'avg', 'cls', 'foreground'}:
            raise ValueError('Invalid extraction variant')
        if self.size not in {256, 384, 512} or self.source not in {'legacy', 'native'}:
            raise ValueError('Invalid resolution or crop source')
        if self.source == 'legacy' and self.size != 256:
            raise ValueError('Upscaling the legacy 256 crop is not a resolution experiment')

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


VARIANTS = (
    Variant('baseline-fp16', precision='fp16'),
    Variant('precision-fp32'),
    Variant('cls-fp32', pooling='cls'),
    Variant('foreground-fp32', pooling='foreground'),
    Variant('native-256-fp32', pooling='foreground', source='native'),
    Variant('native-384-fp32', pooling='foreground', size=384, source='native'),
    Variant('native-512-fp32', pooling='foreground', size=512, source='native'),
)


def unit(array):
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError('Non-finite features')
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    if (norms < 1e-10).any():
        raise ValueError('Zero features')
    return array / norms


def check_vectors(full, animal, patches, coordinates):
    if full.shape != (DIM,) or animal.shape not in {(0,), (DIM,)}:
        raise ValueError('Invalid embedding dimensions')
    if patches.ndim != 2 or patches.shape[1] != DIM or not 0 <= len(patches) <= 256:
        raise ValueError('Invalid patch dimensions or count')
    if coordinates.shape != (len(patches), 2) or not np.isfinite(coordinates).all() or ((coordinates < 0) | (coordinates > 1)).any():
        raise ValueError('Invalid patch coordinates')
    for values in (full[None, :], animal.reshape(-1, DIM), patches):
        if not np.isfinite(values).all() or not np.allclose(np.linalg.norm(values, axis=1), 1, atol=.001):
            raise ValueError('Features must be finite unit vectors')


def save_features(directory, photo, variant, *, full, animal, patches, coordinates, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    arrays = {'full': np.asarray(full, dtype=np.float32),
              'animal': np.asarray([] if animal is None else animal, dtype=np.float32),
              'patches': np.asarray(patches, dtype=np.float32).reshape(-1, DIM),
              'coordinates': np.asarray(coordinates, dtype=np.float32).reshape(-1, 2)}
    check_vectors(**arrays)
    destination = directory / 'features.npz'
    np.savez(destination, **arrays)
    record = dict(metadata, version=FEATURE_VERSION, image_id=photo.id, image_sha256=photo.sha256,
                  variant=asdict(variant), fingerprint=variant.fingerprint, features_sha256=digest(destination))
    (directory / 'metadata.json').write_text(json.dumps(record, ensure_ascii=False, allow_nan=False, indent=2))


def load_features(directory, photo, variant):
    directory = Path(directory)
    metadata = json_read(directory / 'metadata.json')
    if (metadata.get('version') != FEATURE_VERSION or metadata.get('image_id') != photo.id
            or metadata.get('image_sha256') != photo.sha256 or metadata.get('fingerprint') != variant.fingerprint):
        raise ValueError('Feature provenance differs from this image/variant')
    path = directory / 'features.npz'
    if digest(path) != metadata.get('features_sha256'):
        raise ValueError('Feature file checksum mismatch')
    with zipfile.ZipFile(path) as archive:
        if {i.filename for i in archive.infolist()} != {'full.npy', 'animal.npy', 'patches.npy', 'coordinates.npy'}:
            raise ValueError('Unexpected feature archive members')
        if sum(i.file_size for i in archive.infolist()) > 2 * 1024 * 1024:
            raise ValueError('Feature archive exceeds its expanded size limit')
    with np.load(path, allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in ('full', 'animal', 'patches', 'coordinates')}
    check_vectors(**arrays)
    return arrays, metadata
