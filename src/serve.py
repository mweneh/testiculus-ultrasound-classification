"""Start the local service or the opt-in, CPU-only Hugging Face profile."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

from src.private_assets import download_private_package
from src.release_assets import read_manifest, verify_integrity


def configure_space() -> dict:
    """Verify the private package before importing the model-serving process."""
    started = time.monotonic()
    manifest, raw = read_manifest()
    names = os.environ.get('MODEL_TYPES', 'custom_cnn,resnet18,mobilenet_v3_small').split(',')
    if not names or len(set(names)) != len(names) or not set(names).issubset(manifest['models']):
        raise ValueError('MODEL_TYPES must list one or more distinct supported model IDs')
    # Reject an accidental mount inside a publicly served directory.
    default_frontend = Path(__file__).resolve().parents[1] / 'frontend/dist'
    frontend = Path(os.environ.get('FRONTEND_DIST', str(default_frontend))).resolve()
    local_package = os.environ.get('MODEL_PACKAGE_DIR')
    cache = Path(os.environ.get('MODEL_CACHE_DIR', str(Path.home() / '.local/share/testiculus')))
    location = Path(local_package) if local_package else cache
    resolved = location.resolve()
    if resolved.is_relative_to(frontend) or frontend.is_relative_to(resolved):
        raise ValueError('Model storage and frontend static files must be separate')
    try:
        if local_package:
            package = Path(local_package)
        else:
            package = download_private_package(
                os.environ.get('MODEL_REPO_ID', ''), os.environ.get('MODEL_REVISION', ''),
                os.environ.get('HF_TOKEN', ''), cache,
            )
        verify_integrity(package, manifest, raw)
    finally:
        # Do not keep a download credential in the serving process environment.
        os.environ.pop('HF_TOKEN', None)
    for name, record in manifest['models'].items():
        for suffix in ('CHECKPOINT', 'THRESHOLD_CONFIG'):
            os.environ.pop(f'{name.upper()}_{suffix}', None)
        if name in names:
            os.environ[f'{name.upper()}_CHECKPOINT'] = str(package / record['checkpoint_asset'])
            os.environ[f'{name.upper()}_THRESHOLD_CONFIG'] = str(package / record['threshold_asset'])
    os.environ.pop('THRESHOLD_CONFIG', None)
    os.environ['DEVICE'] = 'cpu'
    os.environ['PUBLIC_MODE'] = '1'
    os.environ.setdefault('PORT', '7860')
    return {'package_verified': True, 'configured_model_count': len(names),
            'package_startup_seconds': round(time.monotonic() - started, 3)}


def main():
    mode = os.environ.get('DEPLOYMENT_MODE', 'local')
    if mode not in {'local', 'space'}:
        raise ValueError('DEPLOYMENT_MODE must be local or space')
    if mode == 'space':
        try:
            status = configure_space()
        except Exception:
            raise SystemExit('Space startup failed; check the private package configuration and access') from None
        print(json.dumps(status), flush=True)
    import uvicorn
    uvicorn.run('src.api:app', host='0.0.0.0', port=int(os.environ.get('PORT', '8000')),
                workers=1, access_log=mode != 'space', timeout_keep_alive=5,
                timeout_graceful_shutdown=30, limit_concurrency=16 if mode == 'space' else None)


if __name__ == '__main__':
    main()
