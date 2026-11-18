"""Pinned release assembly/download integrity, without model evaluation or training."""
import copy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import subprocess
from urllib.error import HTTPError

import pytest

from src import release_assets as release


@pytest.fixture
def release_fixture(tmp_path, monkeypatch):
    root = tmp_path / 'source'
    root.mkdir()
    assets = {}
    models = {}
    for index, name in enumerate(sorted(release.MODEL_NAMES)):
        weight_name = f'{name}.pth'
        config_name = f'{name}.json'
        for filename, content in [(weight_name, f'checkpoint-{name}'.encode()),
                                  (config_name, b'{"synthetic": true}\n')]:
            (root / filename).write_bytes(content)
            assets[filename] = {'source': filename, 'sha256': hashlib.sha256(content).hexdigest(),
                                'size_bytes': len(content)}
        models[name] = {'checkpoint_asset': weight_name, 'threshold_asset': config_name,
                        'threshold': .2 + .1 * index, 'parameters': index + 1}
    manifest = {'format_version': 1, 'repository': 'owner/project', 'tag': 'models-v1.0.0',
                'assets': assets, 'models': models}
    path = root / 'manifest.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    monkeypatch.setattr(release, 'verify_models', lambda *args: {'loads_checked_in_fixture': True})
    return root, path, manifest


def assemble(tmp_path, fixture):
    root, path, _ = fixture
    package = tmp_path / 'package'
    result = release.prepare_package(package, manifest_path=path, source_root=root)
    return package, result


def fake_github(monkeypatch, package):
    requested = []
    def open_asset(request, timeout):
        requested.append(request.full_url)
        assert request.full_url.startswith('https://github.com/owner/project/releases/download/models-v1.0.0/')
        assert timeout == 60
        return BytesIO((package / request.full_url.rsplit('/', 1)[1]).read_bytes())
    monkeypatch.setattr(release, 'urlopen', open_asset)
    return requested


def test_prepare_is_deterministic_and_preserves_sources(tmp_path, release_fixture):
    root, manifest, spec = release_fixture
    before = {name: (root / a['source']).read_bytes() for name, a in spec['assets'].items()}
    package, result = assemble(tmp_path, release_fixture)
    second = tmp_path / 'second'
    release.prepare_package(second, manifest_path=manifest, source_root=root)
    assert result['integrity_verified']
    assert result['loads_checked_in_fixture']
    assert sorted(p.name for p in package.iterdir()) == sorted(p.name for p in second.iterdir())
    for p in package.iterdir():
        assert p.read_bytes() == (second / p.name).read_bytes()
    for name, a in spec['assets'].items():
        assert (package / name).read_bytes() == before[name] == (root / a['source']).read_bytes()
    assert (package / 'release_manifest.json').read_bytes() == manifest.read_bytes()


def test_corrupted_source_aborts_before_loading(tmp_path, release_fixture, monkeypatch):
    root, manifest, spec = release_fixture
    source = root / next(iter(spec['assets'].values()))['source']
    source.write_bytes(b'corrupted')
    monkeypatch.setattr(release, 'verify_models', lambda *a: pytest.fail('Must check hashes before loading'))
    with pytest.raises(ValueError, match='mismatch'):
        release.prepare_package(tmp_path / 'package', manifest_path=manifest, source_root=root)
    assert not (tmp_path / 'package').exists()


@pytest.mark.parametrize('extra', [False, True])
def test_verify_rejects_changed_or_unexpected_assets(tmp_path, release_fixture, extra):
    package, _ = assemble(tmp_path, release_fixture)
    manifest, raw = release.read_manifest(release_fixture[1])
    if extra:
        (package / 'unreviewed.txt').write_text('do not upload')
    else:
        (package / next(iter(manifest['assets']))).write_bytes(b'changed')
    with pytest.raises(ValueError):
        release.verify_integrity(package, manifest, raw)


@pytest.mark.parametrize('name', ['SHA256SUMS', 'release_manifest.json'])
def test_verify_rejects_changed_control_files(tmp_path, release_fixture, name):
    package, _ = assemble(tmp_path, release_fixture)
    manifest, raw = release.read_manifest(release_fixture[1])
    (package / name).write_bytes(b'changed')
    with pytest.raises(ValueError, match='metadata differs'):
        release.verify_integrity(package, manifest, raw)


def test_prepare_and_download_refuse_existing_destinations(tmp_path, release_fixture, monkeypatch):
    package, _ = assemble(tmp_path, release_fixture)
    before = {p.name: p.read_bytes() for p in package.iterdir()}
    monkeypatch.setattr(release, 'urlopen', lambda *a, **k: pytest.fail('No download allowed'))
    with pytest.raises(FileExistsError):
        release.prepare_package(package, manifest_path=release_fixture[1], source_root=release_fixture[0])
    with pytest.raises(FileExistsError):
        release.download_package(package, manifest_path=release_fixture[1])
    assert {p.name: p.read_bytes() for p in package.iterdir()} == before


def test_mocked_download_installs_complete_verified_release(tmp_path, release_fixture, monkeypatch):
    package, _ = assemble(tmp_path, release_fixture)
    requested = fake_github(monkeypatch, package)
    installed = tmp_path / 'installed'
    result = release.download_package(installed, manifest_path=release_fixture[1], verify_load=True)
    assert result['integrity_verified'] and result['loads_checked_in_fixture']
    assert len(requested) == len(list(package.iterdir()))
    for p in package.iterdir():
        assert (installed / p.name).read_bytes() == p.read_bytes()


@pytest.mark.parametrize('oversized', [False, True])
def test_bad_download_never_installs_partial_release(tmp_path, release_fixture, monkeypatch, oversized):
    package, _ = assemble(tmp_path, release_fixture)
    name = next(iter(release_fixture[2]['assets']))
    raw = (package / name).read_bytes()
    (package / name).write_bytes(raw + b'too large' if oversized else b'X' * len(raw))
    fake_github(monkeypatch, package)
    installed = tmp_path / 'installed'
    with pytest.raises(ValueError, match='(mismatch|exceeds)'):
        release.download_package(installed, manifest_path=release_fixture[1])
    assert not installed.exists()
    assert not list(tmp_path.glob('.release-download-*'))


def test_unpublished_release_fails_without_installing(tmp_path, release_fixture, monkeypatch):
    def missing(request, timeout):
        raise HTTPError(request.full_url, 404, 'Not Found', {}, None)
    monkeypatch.setattr(release, 'urlopen', missing)
    with pytest.raises(RuntimeError, match='may not be published'):
        release.download_package(tmp_path / 'installed', manifest_path=release_fixture[1])
    assert not (tmp_path / 'installed').exists()


@pytest.mark.parametrize('change', ['traversal', 'url_tag', 'latest', 'source', 'duplicate'])
def test_manifest_rejects_unsafe_names_and_duplicate_keys(release_fixture, change):
    _, path, original = release_fixture
    manifest = copy.deepcopy(original)
    first = next(iter(manifest['assets']))
    if change == 'traversal':
        manifest['assets']['../outside.pth'] = manifest['assets'].pop(first)
    elif change == 'url_tag':
        manifest['tag'] = 'https://untrusted.example/file'
    elif change == 'latest':
        manifest['tag'] = 'latest'
    elif change == 'source':
        manifest['assets'][first]['source'] = '../outside.pth'
    if change == 'duplicate':
        path.write_text('{"format_version":1,"format_version":1}')
    else:
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        release.read_manifest(path)


def test_committed_manifest_matches_frozen_records_and_binary_ignore_rules():
    manifest, _ = release.read_manifest()
    root = release.ROOT
    original = json.loads((root / 'reports/artifact_verification.json').read_text())
    freeze = json.loads((root / 'reports/mobilenet_v3_small_freeze.json').read_text())
    benchmark = json.loads((root / 'reports/model_efficiency.json').read_text())
    for name, model in manifest['models'].items():
        asset = manifest['assets'][model['checkpoint_asset']]
        expected = (freeze['sources']['checkpoint']['sha256'] if name == 'mobilenet_v3_small'
                    else original['models'][name]['checkpoint_sha256'])
        assert asset['sha256'] == expected == benchmark['models'][name]['checkpoint_sha256']
        config = manifest['assets'][model['threshold_asset']]
        assert release.sha256(root / config['source']) == config['sha256']
        assert subprocess.run(['git', 'check-ignore', '-q', asset['source']], cwd=root).returncode == 0
        assert subprocess.run(['git', 'check-ignore', '-q', f"artifacts/releases/{manifest['tag']}/{model['checkpoint_asset']}"], cwd=root).returncode == 0
    tracked = subprocess.check_output(['git', 'ls-files'], cwd=root, text=True).splitlines()
    assert not any(p.endswith(('.pth', '.pt', '.ckpt', '.onnx')) for p in tracked)


def test_actual_frozen_checkpoints_load_without_model_execution(tmp_path, monkeypatch):
    manifest, _ = release.read_manifest()
    if any(not (release.ROOT / spec['source']).is_file() for spec in manifest['assets'].values()):
        pytest.skip('Original checkpoint assets are supplied separately')
    import torch
    from src.models.architectures import CustomCNN
    from torchvision.models.resnet import ResNet
    from torchvision.models.mobilenetv3 import MobileNetV3
    def forbidden(*a, **k):
        pytest.fail('Release verification must not run model inference')
    for cls in (CustomCNN, ResNet, MobileNetV3):
        monkeypatch.setattr(cls, 'forward', forbidden)
    monkeypatch.setattr(torch.hub, 'load_state_dict_from_url', forbidden)
    result = release.prepare_package(tmp_path / 'actual-release')
    assert result['integrity_verified']
    assert not result['model_forward_performed']
    assert not result['test_images_accessed']
    assert not result['training_performed']
    assert all(m['loads'] and m['preprocessing_matches'] and m['finite_state']
               for m in result['models'].values())
