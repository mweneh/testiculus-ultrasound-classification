"""Prepare an allowlisted Docker Space source directory; never upload files."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
FIXED_FILES = (
    'Dockerfile', '.dockerignore', 'requirements-api.txt', 'requirements-serving.txt',
    'reports/validation_thresholds.json', 'reports/mobilenet_v3_small_thresholds.json',
    'reports/mobilenet_exploratory_results.json', 'report/evidence.json', 'release/manifest.json',
    'frontend/package.json', 'frontend/package-lock.json', 'frontend/index.html',
    'frontend/vite.config.js', 'frontend/tailwind.config.js', 'frontend/postcss.config.js',
    'frontend/public/THIRD_PARTY_NOTICES.txt',
)


def prepare(output: Path, *, root=ROOT, source_revision: str) -> list[str]:
    root, output = Path(root).resolve(), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError('Refusing to replace an existing Space source directory')
    files = list(FIXED_FILES)
    for directory, suffixes in [('src', {'.py'}), ('frontend/src', {'.js', '.jsx', '.css'})]:
        for path in sorted((root / directory).rglob('*')):
            relative = path.relative_to(root)
            if (path.suffix in suffixes and '.test.' not in path.name
                    and 'test' not in relative.parts):
                files.append(relative.as_posix())
    for name in [*files, 'deployment/huggingface/SPACE_README.md']:
        path = root / name
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError(f'Require a regular source file: {name}')
        if path.stat().st_size > 5 * 1024 * 1024:
            raise ValueError(f'Unexpectedly large source file: {name}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.space-source-', dir=output.parent) as work:
        staged = Path(work) / 'source'
        staged.mkdir()
        for name in files:
            target = staged / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, target)
        shutil.copyfile(root / 'deployment/huggingface/SPACE_README.md', staged / 'README.md')
        (staged / 'SOURCE_REVISION.txt').write_text(source_revision + '\n')
        staged.rename(output)
    return sorted(files + ['README.md', 'SOURCE_REVISION.txt'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/huggingface-space')
    args = parser.parse_args()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT).strip():
        raise SystemExit('Commit/review source changes before preparing a pinned Space bundle')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    files = prepare(args.output, source_revision=revision)
    print(f'Prepared {len(files)} source files at {args.output}; revision {revision}. No upload performed.')


if __name__ == '__main__':
    main()
