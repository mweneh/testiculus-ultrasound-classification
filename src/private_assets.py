"""Download pinned assets from an approved private Hugging Face model repo.

No upload, credential persistence, Hub SDK cache, or model execution. The
trusted hashes remain in the reviewed source, not in downloaded metadata.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from src.release_assets import (CHECKSUM_ASSET, MANIFEST_ASSET, checksum_text,
                                check_file, read_manifest, verify_integrity)


class SafeHubRedirects(HTTPRedirectHandler):
    """Hub asset redirects may use signed CDN URLs; never send them our token."""

    def redirect_request(self, request, fp, code, msg, headers, newurl):
        target = urlsplit(newurl)
        if target.scheme != 'https' or target.username or target.password:
            raise ValueError('Unsafe asset redirect')
        redirected = super().redirect_request(request, fp, code, msg, headers, newurl)
        if redirected is not None and target.netloc != 'huggingface.co':
            redirected.remove_header('Authorization')
        return redirected


def _fetch(url, token, *, opener):
    return opener.open(Request(url, headers={
        'Authorization': f'Bearer {token}',
        'User-Agent': 'TesticulUS-private-checkpoints/1',
    }), timeout=30)


def download_private_package(repo_id: str, revision: str, token: str, cache: Path,
                             *, manifest_path=None, opener=None) -> Path:
    """Require a private repository at an immutable commit; verify every byte.

    Call only after the owner approves the account, hosting and applicable
    rights. Network exception text (which may contain signed URLs) is withheld.
    """
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*', repo_id):
        raise ValueError('Require a Hugging Face namespace/model-repository ID')
    if not re.fullmatch(r'[a-f0-9]{40}', revision):
        raise ValueError('MODEL_REVISION must be a full immutable 40-character commit SHA')
    if not token or any(char.isspace() for char in token):
        raise ValueError('Set the server-side HF_TOKEN secret')
    manifest, raw = read_manifest() if manifest_path is None else read_manifest(manifest_path)
    cache = Path(cache).absolute()
    if cache.is_symlink():
        raise ValueError('Private cache must not be a symlink')
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    if cache.stat().st_uid != os.getuid():
        raise ValueError('Private cache must be owned by the application user')
    cache.chmod(0o700)
    destination = cache / revision
    opener = opener or build_opener(SafeHubRedirects())
    try:
        # Check privacy even when reusing a valid cache after a restart.
        with _fetch(f'https://huggingface.co/api/models/{repo_id}/revision/{revision}',
                    token, opener=opener) as response:
            metadata = response.read(1024 * 1024 + 1)
            if len(metadata) > 1024 * 1024:
                raise ValueError('Model repository metadata is too large')
            info = json.loads(metadata)
        if info.get('private') is not True or info.get('sha') != revision:
            raise ValueError('Require a private model repository at the pinned revision')
        if destination.exists() or destination.is_symlink():
            if destination.is_symlink():
                raise ValueError('Cached package must not be a symlink')
            verify_integrity(destination, manifest, raw)
            return destination
        specs = dict(manifest['assets'])
        for name, data in [(MANIFEST_ASSET, raw),
                           (CHECKSUM_ASSET, checksum_text(manifest, raw).encode())]:
            specs[name] = {'size_bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        with tempfile.TemporaryDirectory(prefix='.download-', dir=cache) as work:
            staged = Path(work) / 'package'
            staged.mkdir(mode=0o700)
            deadline = time.monotonic() + 180
            for name, spec in specs.items():
                url = f'https://huggingface.co/{repo_id}/resolve/{revision}/{name}'
                path = staged / name
                with _fetch(url, token, opener=opener) as response, path.open('xb') as output:
                    path.chmod(0o600)
                    size = 0
                    while block := response.read(1024 * 1024):
                        size += len(block)
                        if size > spec['size_bytes'] or time.monotonic() > deadline:
                            raise ValueError('Asset exceeds pinned size or download deadline')
                        output.write(block)
                # Check before even moving the file into the installed package.
                check_file(path, spec)
            verify_integrity(staged, manifest, raw)
            staged.rename(destination)
        return destination
    except Exception:
        raise RuntimeError('Private checkpoint installation failed; check access, privacy, revision and hashes') from None
