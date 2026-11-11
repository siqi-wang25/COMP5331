"""Reconstruct the recorded SMORE runtime without changing the team framework."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def direct_environment():
    env = {
        key: value for key, value in os.environ.items()
        if not key.lower().endswith('_proxy')
        and key != 'SSH_AUTH_SOCK'
        and not key.startswith('GIT_CONFIG')
        and not key.startswith('GIT_SSH')
    }
    env.update(NO_PROXY='*', no_proxy='*', GIT_TERMINAL_PROMPT='0',
               GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull,
               GIT_SSH_COMMAND='ssh -o ClearAllForwardings=yes -o ForwardAgent=no')
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='New, nonexistent runtime directory')
    parser.add_argument('--source', type=Path, help='Optional existing local base checkout; no network')
    parser.add_argument('--verify-only', action='store_true', help='Verify the 35 archived PR files only')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'task4/migration_manifest.json').read_text())
    for item in manifest['files']:
        path = ROOT / item['path']
        if digest(path) != item['sha256']:
            raise SystemExit(f'Archived file changed: {item["path"]}')
    if args.verify_only:
        print(json.dumps({'verified_original_pr_files': len(manifest['files']),
                          'source_commit': manifest['source_commit']}))
        return
    if args.output is None:
        parser.error('--output is required unless --verify-only is used')
    destination = args.output.expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        parser.error(f'Refusing to overwrite existing path: {destination}')
    if args.source is not None:
        source = args.source.expanduser().resolve(strict=True)
        if not source.is_dir():
            parser.error('--source must be a local Git checkout directory')
        source = str(source)
    else:
        source = manifest['base_repository']
    env = direct_environment()
    git = ['git', '-c', 'http.proxy=', '-c', 'https.proxy=']
    subprocess.run(git + ['clone', '--no-hardlinks', '--no-checkout', source,
                         str(destination)], env=env, check=True)
    subprocess.run(git + ['-C', str(destination), 'checkout', '--detach',
                         manifest['base_commit']], env=env, check=True)
    for item in manifest['files']:
        target = destination / item['original_path']
        if target.exists() or target.is_symlink():
            raise SystemExit(f'Unexpected file in pinned base: {target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / item['path'], target)
        target.chmod(0o755 if item['git_mode'] == '100755' else 0o644)
        if digest(target) != item['sha256']:
            raise SystemExit(f'Copied file checksum mismatch: {target}')
    protocol = json.loads((destination / 'task4/audit_20261031/fixed_protocol.json').read_text())
    for relative, expected in protocol['file_sha256'].items():
        if digest(destination / relative) != expected:
            raise SystemExit(f'Fixed training source/config mismatch: {relative}')
    shutil.copyfile(ROOT / 'task4/migration_manifest.json',
                    destination / 'task4/migration_manifest.json')
    print(json.dumps({'runtime': str(destination), 'verified_original_pr_files': len(manifest['files']),
                      'verified_protocol_files': len(protocol['file_sha256']),
                      'base_commit': manifest['base_commit'],
                      'recorded_source_commit': manifest['source_commit']}))
    print('Runtime prepared; no dependencies, datasets, or training jobs were started.')
    print('Commit the reconstructed src/ and task4/ files before new training to record a clean source commit.')


if __name__ == '__main__':
    main()
