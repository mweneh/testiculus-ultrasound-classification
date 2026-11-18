"""Prepare, verify or download pinned checkpoint release assets; never publish.

Downloading/integrity checks use only the standard library. Model-load checks
import the existing ML runtime lazily and do not execute a model forward pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / 'release/manifest.json'
MODEL_NAMES = {'custom_cnn', 'resnet18', 'mobilenet_v3_small'}
MANIFEST_ASSET = 'release_manifest.json'
CHECKSUM_ASSET = 'SHA256SUMS'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate manifest key: {key}')
        result[key] = value
    return result


def read_manifest(path: Path = DEFAULT_MANIFEST) -> tuple[dict, bytes]:
    raw = Path(path).read_bytes()
    manifest = json.loads(raw, object_pairs_hook=_unique_object)
    if manifest.get('format_version') != 1:
        raise ValueError('Unsupported release manifest format')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', manifest.get('repository', '')):
        raise ValueError('Invalid GitHub owner/repository')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', manifest.get('tag', '')):
        raise ValueError('Use a pinned release tag, not a URL or path')
    if manifest['tag'] == 'latest':
        raise ValueError('A mutable latest alias is not a pinned release')
    assets = manifest.get('assets')
    if not isinstance(assets, dict) or not assets:
        raise ValueError('Release assets must be a nonempty mapping')
    for name, spec in assets.items():
        if (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name)
                or name in {MANIFEST_ASSET, CHECKSUM_ASSET}):
            raise ValueError('Unsafe or reserved asset filename')
        if (not isinstance(spec, dict)
                or not re.fullmatch(r'[a-f0-9]{64}', spec.get('sha256', ''))
                or type(spec.get('size_bytes')) is not int or spec['size_bytes'] < 1):
            raise ValueError(f'Invalid hash/size for {name}')
        source = PurePosixPath(spec.get('source', ''))
        if source.is_absolute() or '..' in source.parts or not source.parts or '\\' in str(source):
            raise ValueError('Asset source must be a relative path within the repository')
    models = manifest.get('models', {})
    if set(models) != MODEL_NAMES:
        raise ValueError('Require all three frozen model records')
    for model in models.values():
        if model.get('checkpoint_asset') not in assets or model.get('threshold_asset') not in assets:
            raise ValueError('Model references an unknown asset')
        threshold = model.get('threshold')
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError('Invalid frozen threshold')
        if type(model.get('parameters')) is not int or model['parameters'] < 1:
            raise ValueError('Invalid parameter count')
    return manifest, raw


def checksum_text(manifest: dict, raw: bytes) -> str:
    hashes = {name: spec['sha256'] for name, spec in manifest['assets'].items()}
    hashes[MANIFEST_ASSET] = hashlib.sha256(raw).hexdigest()
    return ''.join(f'{hashes[name]}  {name}\n' for name in sorted(hashes))


def check_file(path: Path, spec: dict) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Require a regular asset file: {path.name}')
    if path.stat().st_size != spec['size_bytes'] or sha256(path) != spec['sha256']:
        raise ValueError(f'SHA-256/size mismatch: {path.name}')


def verify_integrity(package: Path, manifest: dict, raw: bytes) -> dict:
    package = Path(package)
    expected = set(manifest['assets']) | {MANIFEST_ASSET, CHECKSUM_ASSET}
    if {p.name for p in package.iterdir()} != expected:
        raise ValueError('Package must contain exactly the pinned release assets')
    for name, spec in manifest['assets'].items():
        check_file(package / name, spec)
    for name, content in [(MANIFEST_ASSET, raw),
                          (CHECKSUM_ASSET, checksum_text(manifest, raw).encode())]:
        path = package / name
        if path.is_symlink() or not path.is_file() or path.read_bytes() != content:
            raise ValueError(f'Package metadata differs from trusted source: {name}')
    return {'integrity_verified': True, 'tag': manifest['tag'],
            'assets': {name: spec['sha256'] for name, spec in manifest['assets'].items()},
            'manifest_sha256': hashlib.sha256(raw).hexdigest()}


def verify_models(package: Path, manifest: dict) -> dict:
    """Strict CPU state loading and metadata checks, with no forward or images."""
    import platform
    import torch
    import torchvision
    from src.training import CLASS_TO_LABEL, _preprocessing_metadata, load_checkpoint
    from src.threshold_config import resolve_threshold

    models = {}
    for name, spec in manifest['models'].items():
        path = package / spec['checkpoint_asset']
        before = sha256(path)
        model, checkpoint = load_checkpoint(path, device='cpu')
        if checkpoint.get('model_type') != name or checkpoint.get('class_to_idx') != CLASS_TO_LABEL:
            raise ValueError(f'Checkpoint model/class mapping mismatch: {name}')
        if checkpoint.get('preprocessing') != _preprocessing_metadata(name):
            raise ValueError(f'Checkpoint preprocessing mismatch: {name}')
        parameters = sum(p.numel() for p in model.parameters())
        if parameters != spec['parameters'] or any(m.training for m in model.modules()):
            raise ValueError(f'Checkpoint parameters/evaluation mode mismatch: {name}')
        if not all(bool(torch.isfinite(t).all()) for t in model.state_dict().values() if t.is_floating_point()):
            raise ValueError(f'Nonfinite model state: {name}')
        threshold = resolve_threshold(name, threshold_config=package / spec['threshold_asset'])
        if threshold != spec['threshold'] or sha256(path) != before:
            raise ValueError(f'Frozen threshold or checkpoint changed: {name}')
        models[name] = {'loads': True, 'parameters': parameters, 'threshold': threshold,
                        'preprocessing_matches': True, 'finite_state': True,
                        'checkpoint_sha256': before, 'device': 'cpu'}
        del model, checkpoint
    return {'models': models, 'model_forward_performed': False,
            'test_images_accessed': False, 'training_performed': False,
            'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                            'torchvision': str(torchvision.__version__)}}


def _new_destination(path: Path) -> Path:
    # Inspect the original path before resolving, including dangling symlinks.
    path = Path(path).absolute()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f'Refusing to replace an existing destination: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def prepare_package(output: Path, *, manifest_path=DEFAULT_MANIFEST,
                    source_root=ROOT) -> dict:
    manifest, raw = read_manifest(manifest_path)
    output = _new_destination(output)
    source_root = Path(source_root).resolve()
    # Validate every source before creating or loading any packaged checkpoint.
    sources = {}
    for name, spec in manifest['assets'].items():
        path = source_root / spec['source']
        if not path.resolve().is_relative_to(source_root):
            raise ValueError('Asset source resolves outside the repository')
        check_file(path, spec)
        sources[name] = path
    with tempfile.TemporaryDirectory(prefix='.release-stage-', dir=output.parent) as work:
        staged = Path(work) / 'payload'
        staged.mkdir()
        for name, source in sources.items():
            shutil.copyfile(source, staged / name)  # No reserialization or tensor changes.
        (staged / MANIFEST_ASSET).write_bytes(raw)
        (staged / CHECKSUM_ASSET).write_text(checksum_text(manifest, raw), encoding='utf-8')
        result = verify_integrity(staged, manifest, raw)
        result.update(verify_models(staged, manifest))
        for name, path in sources.items():
            check_file(path, manifest['assets'][name])
        if output.exists() or output.is_symlink():
            raise FileExistsError('Destination appeared during preparation')
        staged.rename(output)
    return result


def _download_one(url: str, path: Path, spec: dict) -> None:
    request = Request(url, headers={'User-Agent': 'TesticulUS-checkpoint-downloader/1',
                                    'Accept': 'application/octet-stream'})
    digest = hashlib.sha256()
    size = 0
    with urlopen(request, timeout=60) as response, path.open('xb') as output:
        while block := response.read(1024 * 1024):
            size += len(block)
            if size > spec['size_bytes']:
                raise ValueError(f'Download exceeds pinned size: {path.name}')
            output.write(block)
            digest.update(block)
    if size != spec['size_bytes'] or digest.hexdigest() != spec['sha256']:
        raise ValueError(f'Download SHA-256/size mismatch: {path.name}')


def download_package(output: Path, *, manifest_path=DEFAULT_MANIFEST,
                     verify_load=False) -> dict:
    """Download a fixed public GitHub Release into a new directory after validation."""
    manifest, raw = read_manifest(manifest_path)
    output = _new_destination(output)
    base = f"https://github.com/{manifest['repository']}/releases/download/{manifest['tag']}"
    specs = {name: dict(spec) for name, spec in manifest['assets'].items()}
    for name, content in [(MANIFEST_ASSET, raw),
                          (CHECKSUM_ASSET, checksum_text(manifest, raw).encode())]:
        specs[name] = {'sha256': hashlib.sha256(content).hexdigest(), 'size_bytes': len(content)}
    with tempfile.TemporaryDirectory(prefix='.release-download-', dir=output.parent) as work:
        staged = Path(work) / 'payload'
        staged.mkdir()
        try:
            for name, spec in specs.items():
                _download_one(f'{base}/{name}', staged / name, spec)
        except HTTPError as error:
            if error.code == 404:
                raise RuntimeError('Pinned public release/asset is unavailable. '
                                   'It may not be published yet; no files were installed.') from error
            raise
        result = verify_integrity(staged, manifest, raw)
        if verify_load:
            result.update(verify_models(staged, manifest))
        if output.exists() or output.is_symlink():
            raise FileExistsError('Destination appeared during download')
        staged.rename(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare', help='Copy and verify existing checkpoints; no upload')
    prep.add_argument('--output', type=Path)
    verify = sub.add_parser('verify', help='Verify exact assets and strict CPU checkpoint loading')
    verify.add_argument('--package', type=Path, required=True)
    download = sub.add_parser('download', help='Install the pinned public release, when published')
    download.add_argument('--output', type=Path)
    download.add_argument('--verify-load', action='store_true')
    args = parser.parse_args()
    manifest, raw = read_manifest(args.manifest)
    if args.command == 'prepare':
        destination = args.output or ROOT / 'artifacts/releases' / manifest['tag']
        result = prepare_package(destination, manifest_path=args.manifest)
    elif args.command == 'download':
        destination = args.output or ROOT / 'artifacts/checkpoints' / manifest['tag']
        result = download_package(destination, manifest_path=args.manifest, verify_load=args.verify_load)
    else:
        result = verify_integrity(args.package, manifest, raw)
        result.update(verify_models(args.package, manifest))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
