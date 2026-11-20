import copy
import hashlib
from io import BytesIO
import json
from pathlib import Path
from urllib.request import Request

import pytest

from src import private_assets, release_assets, serve

REVISION = 'a' * 40


@pytest.fixture
def package(tmp_path):
    # Small transport fixtures, never deserialized as checkpoints.
    manifest, _ = release_assets.read_manifest()
    manifest = copy.deepcopy(manifest)
    payloads = {}
    for name, spec in manifest['assets'].items():
        data = f'fixture:{name}'.encode()
        payloads[name] = data
        spec['size_bytes'] = len(data)
        spec['sha256'] = hashlib.sha256(data).hexdigest()
    raw = (json.dumps(manifest) + '\n').encode()
    manifest_path = tmp_path / 'trusted_manifest.json'
    manifest_path.write_bytes(raw)
    payloads['release_manifest.json'] = raw
    payloads['SHA256SUMS'] = release_assets.checksum_text(manifest, raw).encode()
    folder = tmp_path / 'package'
    folder.mkdir()
    for name, data in payloads.items():
        (folder / name).write_bytes(data)
    return folder, manifest_path, manifest, raw, payloads


class Hub:
    def __init__(self, payloads, private=True, revision=REVISION):
        self.payloads = payloads
        self.private = private
        self.revision = revision
        self.calls = []

    def open(self, request, timeout):
        assert timeout == 30
        assert request.get_header('Authorization') == 'Bearer fake-test-token'
        self.calls.append(request.full_url)
        if '/api/models/' in request.full_url:
            return BytesIO(json.dumps({'private': self.private, 'sha': self.revision}).encode())
        return BytesIO(self.payloads[request.full_url.rsplit('/', 1)[1]])


def download(tmp_path, package, hub=None, **kwargs):
    return private_assets.download_private_package(
        'owner/private-models', REVISION, 'fake-test-token', tmp_path / 'cache',
        manifest_path=package[1], opener=hub or Hub(package[4]), **kwargs)


def test_private_download_checks_all_assets_and_reuses_only_verified_cache(tmp_path, package):
    hub = Hub(package[4])
    path = download(tmp_path, package, hub)
    assert path.name == REVISION
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in path.iterdir())
    for name, data in package[4].items():
        assert (path / name).read_bytes() == data
    before = len(hub.calls)
    assert download(tmp_path, package, hub) == path
    assert len(hub.calls) == before + 1  # Still checks repository privacy.
    (path / next(iter(package[4]))).write_bytes(b'corrupt')
    with pytest.raises(RuntimeError):
        download(tmp_path, package, hub)


@pytest.mark.parametrize('case', ['public', 'wrong_revision', 'corrupt', 'oversize', 'network'])
def test_failures_leave_no_installed_package_or_credentials(tmp_path, package, case):
    payloads = dict(package[4])
    first = next(iter(payloads))
    if case == 'corrupt':
        payloads[first] = b'x' * len(payloads[first])
    if case == 'oversize':
        payloads[first] += b'x'
    hub = Hub(payloads, private=case != 'public', revision='b' * 40 if case == 'wrong_revision' else REVISION)
    if case == 'network':
        def fail(*args, **kwargs):
            raise RuntimeError('fake-test-token private/signed-url')
        hub.open = fail
    with pytest.raises(RuntimeError) as caught:
        download(tmp_path, package, hub)
    assert 'fake-test-token' not in str(caught.value)
    assert not (tmp_path / 'cache' / REVISION).exists()
    assert list((tmp_path / 'cache').iterdir()) == []


@pytest.mark.parametrize('repo,revision,token', [('../bad', REVISION, 'token'),
                         ('owner/repo', 'main', 'token'), ('owner/repo', REVISION, '')])
def test_invalid_identifiers_fail_before_network(tmp_path, repo, revision, token):
    with pytest.raises(ValueError):
        private_assets.download_private_package(repo, revision, token, tmp_path / 'cache')


def test_redirects_strip_credentials_and_reject_plain_http():
    redirect = private_assets.SafeHubRedirects()
    req = Request('https://huggingface.co/owner/repo/resolve/main/file',
                  headers={'Authorization': 'Bearer fake-test-token'})
    same = redirect.redirect_request(req, None, 302, 'Found', {}, 'https://huggingface.co/new')
    assert same.get_header('Authorization') == 'Bearer fake-test-token'
    cdn = redirect.redirect_request(req, None, 302, 'Found', {}, 'https://cdn.example/file?signature=x')
    assert cdn.get_header('Authorization') is None
    with pytest.raises(ValueError):
        redirect.redirect_request(req, None, 302, 'Found', {}, 'http://cdn.example/file')


def test_space_selects_subset_and_downloaded_thresholds_without_exposing_token(package, monkeypatch):
    folder, _, manifest, raw, _ = package
    monkeypatch.setattr(serve, 'read_manifest', lambda: (manifest, raw))
    monkeypatch.setenv('MODEL_PACKAGE_DIR', str(folder))
    monkeypatch.setenv('HF_TOKEN', 'fake-test-token')
    monkeypatch.setenv('MODEL_TYPES', 'resnet18,mobilenet_v3_small')
    for name in manifest['models']:
        for suffix in ('CHECKPOINT', 'THRESHOLD_CONFIG'):
            monkeypatch.setenv(f'{name.upper()}_{suffix}', '/wrong/old.pth')
    for key, value in {'DEVICE': 'cuda', 'PUBLIC_MODE': '0', 'PORT': '7860',
                       'THRESHOLD_CONFIG': '/wrong/config.json'}.items():
        monkeypatch.setenv(key, value)
    status = serve.configure_space()
    import os
    assert status['configured_model_count'] == 2
    assert os.environ['RESNET18_THRESHOLD_CONFIG'] == str(folder / 'validation_thresholds.json')
    assert os.environ['MOBILENET_V3_SMALL_THRESHOLD_CONFIG'] == str(folder / 'mobilenet_v3_small_thresholds.json')
    assert os.environ['DEVICE'] == 'cpu'
    assert os.environ['PUBLIC_MODE'] == '1'
    assert 'CUSTOM_CNN_CHECKPOINT' not in os.environ
    assert 'HF_TOKEN' not in os.environ


def test_space_rejects_static_storage_overlap_and_unknown_models(package, monkeypatch):
    folder, _, manifest, raw, _ = package
    monkeypatch.setattr(serve, 'read_manifest', lambda: (manifest, raw))
    monkeypatch.setenv('MODEL_PACKAGE_DIR', str(folder))
    monkeypatch.setenv('FRONTEND_DIST', str(folder.parent))
    with pytest.raises(ValueError, match='separate'):
        serve.configure_space()
    monkeypatch.setenv('MODEL_TYPES', 'unknown')
    with pytest.raises(ValueError, match='MODEL_TYPES'):
        serve.configure_space()
