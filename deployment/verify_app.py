"""Compare a running application with local CPU inference on ONE permitted image.

The operator must supply an anonymized, licensed non-test image. This is an
application smoke check, not a repeat evaluation or performance benchmark.
No images or heatmaps are saved. No authentication token is sent to the app.
"""
from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
from pathlib import Path
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import uuid


def verify(base: str, image_path: Path, package: Path, *, allow_local=False) -> dict:
    from PIL import Image
    from src.inference import CheckpointPredictor
    from src.release_assets import read_manifest, sha256, verify_integrity

    parsed = urlsplit(base)
    local = parsed.scheme == 'http' and parsed.hostname in {'127.0.0.1', 'localhost'}
    if ((parsed.scheme != 'https' and not (allow_local and local))
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in {'', '/'}):
        raise ValueError('Use a bare HTTPS app URL (or explicitly allow a localhost URL)')
    base = base.rstrip('/')
    manifest, raw = read_manifest()
    verify_integrity(package, manifest, raw)
    data = Path(image_path).read_bytes()
    if len(data) > 5 * 1024 * 1024:
        raise ValueError('Demonstration image exceeds upload limit')

    def request(path, *, payload=None, content_type=None):
        headers = {'Accept': 'application/json'}
        if content_type:
            headers['Content-Type'] = content_type
        start = time.perf_counter()
        with urlopen(Request(base + path, data=payload, headers=headers), timeout=60) as response:
            body = response.read(1024 * 1024 + 1)
            if len(body) > 1024 * 1024:
                raise ValueError('Unexpectedly large application response')
        return body, time.perf_counter() - start

    health = json.loads(request('/health')[0])
    assert health['status'] == 'ready'
    assert health['models'] and set(health['models']).issubset(manifest['models'])
    boundary = 'TesticulUS' + uuid.uuid4().hex
    form = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
            'filename="demonstration.png"\r\nContent-Type: application/octet-stream\r\n\r\n').encode()
    form += data + f'\r\n--{boundary}--\r\n'.encode()
    content_type = f'multipart/form-data; boundary={boundary}'
    results = {}
    with Image.open(BytesIO(data)) as image:
        for name in health['models']:
            spec = manifest['models'][name]
            predictor = CheckpointPredictor(package / spec['checkpoint_asset'],
                                           threshold_config=package / spec['threshold_asset'], device='cpu')
            expected = predictor.predict(image)
            predicted, latency = request(f'/predict?model_type={name}', payload=form, content_type=content_type)
            result = json.loads(predicted)
            for key in ('model_type', 'prediction', 'class_name', 'threshold', 'threshold_source'):
                assert result[key] == expected[key], key
            delta = abs(result['probability_class_1'] - expected['probability_class_1'])
            assert delta <= 1e-5
            assert result['threshold'] == spec['threshold'] == health['models'][name]['threshold']
            explained, cam_latency = request(f'/explain?model_type={name}', payload=form, content_type=content_type)
            explanation = json.loads(explained)
            assert all(explanation[key] == value for key, value in result.items())
            details = explanation['gradcam']
            assert details['target_class'] == result['prediction']
            assert details['heatmap_shape'] == [224, 224]
            with Image.open(BytesIO(base64.b64decode(details['heatmap_png_base64'], validate=True))) as heatmap:
                assert heatmap.format == 'PNG' and heatmap.size == (224, 224) and heatmap.mode == 'L'
            results[name] = {'prediction': result['prediction'], 'threshold': result['threshold'],
                             'probability_class_1': result['probability_class_1'],
                             'local_probability_absolute_difference': delta,
                             'prediction_http_seconds': latency, 'explain_http_seconds': cam_latency,
                             'gradcam_verified': True}
    return {'url': base, 'loaded_model_count': len(results), 'models': results,
            'image_sha256': sha256(image_path), 'single_non_test_demo': True,
            'held_out_test_evaluation': False, 'training': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--allow-local', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify(args.url, args.image, args.package, allow_local=args.allow_local), indent=2))


if __name__ == '__main__':
    main()
