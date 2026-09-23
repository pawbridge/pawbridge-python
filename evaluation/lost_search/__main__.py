"""python -m evaluation.lost_search --help"""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from .dataset import audit, digest, json_read, load_dataset
from .features import VARIANTS, load_features


@contextmanager
def exclusive_lock(kind):
    import fcntl
    path = Path('/tmp') / f'pawbridge-lost-quality-{os.getuid()}-{kind}.lock'
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another local quality evaluation owns this resource') from None
        yield
    finally:
        os.close(descriptor)


def source_fingerprint():
    root = Path(__file__).resolve().parents[2]
    paths = sorted(Path(__file__).parent.glob('*.py')) + [root / 'app/services' / name for name in
            ('dinov3.py', 'sam3_focus.py', 'animal_focus.py', 'coat_color.py', 'lost_search.py')]
    content = '\n'.join(str(path.relative_to(root)) + ':' + digest(path) for path in paths)
    return hashlib.sha256(content.encode()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2))


def supervise(command, *, deadline, rss_limit=8 * 1024**3):
    """One child, no parallel models; terminate the whole child group on failure."""
    process = subprocess.Popen(command, start_new_session=True)
    try:
        while process.poll() is None:
            if time.monotonic() >= deadline:
                raise TimeoutError('Evaluation wall-clock deadline exceeded')
            try:
                lines = Path(f'/proc/{process.pid}/status').read_text().splitlines()
                rss = next(int(line.split()[1])*1024 for line in lines if line.startswith('VmRSS:'))
            except (FileNotFoundError, StopIteration):
                rss = 0
            if rss > rss_limit:
                raise MemoryError('Evaluation worker exceeded its 8 GiB RSS budget')
            time.sleep(.2)
        if process.returncode:
            raise RuntimeError(f'Evaluation worker exited with status {process.returncode}')
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)


def extract(args, dataset, variants):
    from .extract import require_idle_service
    if args.confirm_exclusive_gpu != 'APPROVED_OFFLINE_GPU_WINDOW':
        raise ValueError('An explicitly approved exclusive GPU window is required')
    require_idle_service()
    from app.services.dinov3 import WEIGHTS_SHA256
    from app.services.sam3_focus import CHECKPOINT_SHA256
    # Check both files before any model loading; never download a missing model.
    for path, expected in ((args.dino_checkpoint, WEIGHTS_SHA256), (args.sam_checkpoint, CHECKPOINT_SHA256)):
        if path is None or not path.is_file() or digest(path) != expected:
            raise ValueError('Existing, pinned local checkpoints are required')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.output.parent).free < 5 * 1024**3:
        raise ValueError('At least 5 GiB of free space is required for bounded artifacts')
    args.output.mkdir(exist_ok=False)
    started = time.monotonic()
    state = {'status': 'running', 'execution': 'real_gpu_offline', 'dataset': audit(dataset),
             'variants': [asdict(v) for v in variants], 'source_sha256': source_fingerprint(),
             'weights_sha256': {'dino': WEIGHTS_SHA256, 'sam': CHECKPOINT_SHA256},
             'python': sys.version.split()[0], 'scope': 'pilot_not_production_accuracy',
             'resource_limits': {'cuda_gib': 7, 'worker_rss_gib': 8, 'cpu_threads': 2,
                                 'seconds': args.timeout_seconds},
             'timing_note': 'Sequential SAM then DINO workers; not public API latency. First image of each variant is marked cold.'}
    state_path = args.output / 'run.json'
    write_json(state_path, state)
    try:
        for stage in ('sam', 'dino'):
            command = [sys.executable, '-m', 'evaluation.lost_search', '_worker',
                       '--manifest', str(args.manifest.resolve()), '--output', str(args.output.resolve()),
                       '--stage', stage, '--dino-checkpoint', str(args.dino_checkpoint.resolve()),
                       '--sam-checkpoint', str(args.sam_checkpoint.resolve()),
                       '--variants', ','.join(v.name for v in variants)]
            supervise(command, deadline=started + args.timeout_seconds)
        state['status'] = 'completed'
        state['sam'] = json_read(args.output / 'sam-timing.json')
        state['dino'] = json_read(args.output / 'dino-timing.json')
    except BaseException as error:
        state['status'] = 'failed'
        state['error_type'] = type(error).__name__
        raise
    finally:
        state['wall_seconds'] = time.monotonic() - started
        write_json(state_path, state)


def compare(args, dataset, variants):
    from .ranking import RANKINGS, evaluate_query, metrics
    from .report import write_report
    from app.services.dinov3 import WEIGHTS_SHA256
    state = json_read(args.features / 'run.json')
    if (state.get('status') != 'completed' or state.get('dataset', {}).get('manifest_sha256') != dataset.sha256
            or state.get('source_sha256') != source_fingerprint()):
        raise ValueError('Only complete features from this exact dataset and code can be compared')
    if any(asdict(v) not in state['variants'] for v in variants):
        raise ValueError('Requested variant was not extracted')
    evaluations = []
    for variant in variants:
        def get_features(photo):
            features, metadata = load_features(args.features / 'features' / variant.name / photo.id, photo, variant)
            if metadata.get('weights_sha256') != WEIGHTS_SHA256:
                raise ValueError('Feature weights differ from pinned DINO')
            return features, metadata
        for ranking in RANKINGS:
            results = [evaluate_query(query, dataset.gallery, get_features, ranking) for query in dataset.queries]
            evaluations.append({'variant': asdict(variant), 'ranking': asdict(ranking),
                                'metrics': metrics(results), 'queries': results})
    write_report(args.output, dataset, evaluations, state, feature_root=args.features)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Bounded local lost-animal quality evaluation. No serving/DB writes.')
    parser.add_argument('command', choices=('validate', 'preview', 'extract', 'compare', '_worker'))
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--features', type=Path)
    parser.add_argument('--variants', default=','.join(v.name for v in VARIANTS[:4]))
    parser.add_argument('--dino-checkpoint', type=Path)
    parser.add_argument('--sam-checkpoint', type=Path)
    parser.add_argument('--confirm-exclusive-gpu')
    parser.add_argument('--timeout-seconds', type=int, default=600)
    parser.add_argument('--stage', choices=('sam', 'dino'))
    args = parser.parse_args(argv)
    if not 1 <= args.timeout_seconds <= 600:
        parser.error('timeout-seconds must be within 1..600')
    requested = args.variants.split(',')
    variants = [variant for variant in VARIANTS if variant.name in requested]
    if len(variants) != len(requested):
        parser.error('Unknown or repeated variant')
    if args.command != 'validate' and args.output is None:
        parser.error('--output is required')
    if args.command == 'compare' and args.features is None:
        parser.error('--features is required')
    dataset = load_dataset(args.manifest)
    if args.command == 'validate':
        print(json.dumps(audit(dataset), indent=2))
    elif args.command == 'preview':
        from .report import write_report
        write_report(args.output, dataset, [], {'execution': 'input_inventory_only', 'dataset': audit(dataset)})
    elif args.command == 'extract':
        with exclusive_lock('session'):
            extract(args, dataset, variants)
    elif args.command == 'compare':
        compare(args, dataset, variants)
    else:
        if args.stage is None or args.sam_checkpoint is None or args.dino_checkpoint is None:
            parser.error('Worker requires a stage and both local checkpoints')
        # Worker also checks service state: invoking it directly cannot bypass it.
        from .extract import dino_worker, sam_worker
        os.environ['SAM3_CHECKPOINT'] = str(args.sam_checkpoint)
        with exclusive_lock('gpu'):
            if args.stage == 'sam':
                sam_worker(dataset, args.output)
            else:
                dino_worker(dataset, args.output, variants, args.dino_checkpoint)


if __name__ == '__main__':
    main()
