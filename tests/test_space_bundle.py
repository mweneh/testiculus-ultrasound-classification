from pathlib import Path

import pytest

from deployment.prepare_space import FIXED_FILES, prepare


@pytest.fixture
def source(tmp_path):
    root = tmp_path / 'source'
    for name in (*FIXED_FILES, 'src/api.py', 'frontend/src/App.jsx',
                 'frontend/src/App.test.jsx', 'frontend/src/test/setup.js',
                 'frontend/node_modules/pkg/index.js', 'src/__pycache__/api.pyc',
                 'training_runs/custom_cnn/best_model.pth', 'data/raw/image.png',
                 '.env', 'frontend/public/private.png',
                 'deployment/huggingface/SPACE_README.md'):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('source fixture')
    return root


def test_source_bundle_excludes_binaries_data_secrets_dependencies_and_tests(source, tmp_path):
    output = tmp_path / 'bundle'
    files = prepare(output, root=source, source_revision='reviewed-sha')
    assert (output / 'README.md').read_text() == 'source fixture'
    assert (output / 'SOURCE_REVISION.txt').read_text() == 'reviewed-sha\n'
    assert 'src/api.py' in files
    assert 'frontend/src/App.jsx' in files
    assert all((output / name).is_file() for name in FIXED_FILES)
    assert not any(p.suffix in {'.pth', '.png', '.pyc'} for p in output.rglob('*'))
    assert not (output / '.env').exists()
    assert not (output / 'frontend/node_modules').exists()
    assert not (output / 'frontend/src/App.test.jsx').exists()
    assert not (output / 'frontend/src/test').exists()
    with pytest.raises(FileExistsError):
        prepare(output, root=source, source_revision='sha')


def test_bundle_rejects_source_symlinks_before_copying(source, tmp_path):
    path = source / 'src/api.py'
    path.unlink()
    path.symlink_to(source / '.env')
    with pytest.raises(ValueError, match='regular source'):
        prepare(tmp_path / 'bundle', root=source, source_revision='sha')
    assert not (tmp_path / 'bundle').exists()
