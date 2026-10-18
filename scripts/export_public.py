"""Export current public source files without local state or Git history."""

import argparse
from pathlib import Path
import shutil
import subprocess


def export_public(root, destination):
    root = root.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise ValueError('Destination already exists; choose a new directory.')
    output = subprocess.check_output(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=root
    )
    paths = sorted(set(Path(name.decode()) for name in output.split(b'\0') if name))
    roots = {'src', 'config', 'scripts', 'tests', 'docs', 'docker'}
    files = {'.gitignore', '.dockerignore', 'README.md', 'Dockerfile',
             'docker-compose.yml', 'environment.yml', 'pyproject.toml',
             'requirements.txt', 'LICENSE', 'NOTICE', 'data/.gitkeep', 'outputs/.gitkeep'}
    private = {'scripts/run_full_slam.sh', 'scripts/run_unity_slam.sh',
               'scripts/run_exvivo.sh', 'scripts/run_unity.sh'}
    selected = []
    for relative in paths:
        name = relative.as_posix()
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'Unsafe source path: {relative}')
        if name not in files and relative.parts[0] not in roots:
            continue
        if name in private or name.startswith(('scripts/local/', 'config/local/')):
            continue
        if '__pycache__' in relative.parts or relative.suffix in ('.pyc', '.log'):
            continue
        source = root / relative
        if not source.exists():
            continue
        if source.resolve() != source.absolute():
            raise ValueError(f'Symlinks are not allowed in public exports: {relative}')
        if source.is_file():
            selected.append(relative)
    destination.mkdir(parents=True)
    for relative in selected:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, target)
    return len(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', nargs='?', default='public_release',
                        help='New directory, relative to the project root by default.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = Path(args.destination)
    if not destination.is_absolute():
        destination = root / destination
    try:
        count = export_public(root, destination)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'Export failed: {error}\n')
    print(f'Exported {count} files to {destination}; no Git history included.')


if __name__ == '__main__':
    main()
