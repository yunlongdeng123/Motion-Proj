"""验证归档 payload 与恢复清单；解包仅允许空的新目录。"""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import shutil
import tarfile


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_name(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or not p.parts:
        raise ValueError('Unsafe archive member: ' + name)
    return p


def verify_archive(archive):
    with tarfile.open(archive, 'r:*') as tar:
        members = tar.getmembers()
        if len({m.name for m in members}) != len(members):
            raise ValueError('Duplicate archive member')
        for m in members:
            safe_name(m.name)
            if not m.isfile():
                raise ValueError('Archive permits regular files only: ' + m.name)
        manifest = json.load(tar.extractfile('restore_manifest.json'))
        expected = {r['archive_path']: r for r in manifest['files']}
        actual = {m.name for m in members if m.name != 'restore_manifest.json'}
        if actual != set(expected):
            raise ValueError('Archive file set differs from manifest')
        for name, record in expected.items():
            m = tar.getmember(name)
            if m.size != record['bytes']:
                raise ValueError('Size differs: ' + name)
            h = hashlib.sha256()
            with tar.extractfile(m) as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    h.update(block)
            if h.hexdigest() != record['sha256']:
                raise ValueError('Content differs: ' + name)
    return manifest


def verify_payload(manifest, payload_root):
    root = Path(payload_root).resolve()
    for r in manifest['files']:
        rel = safe_name(r['archive_path'])
        p = root.joinpath(*rel.parts)
        if not p.is_file() or p.stat().st_size != r['bytes'] or sha(p) != r['sha256']:
            raise ValueError('Restored payload mismatch: ' + r['archive_path'])
    return len(manifest['files'])


def extract_verified(archive, destination):
    manifest = verify_archive(archive)
    dest = Path(destination).resolve()
    if dest.exists() and any(dest.iterdir()):
        raise ValueError('Restore destination must be empty: ' + str(dest))
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, 'r:*') as tar:
        for member in tar.getmembers():
            rel = safe_name(member.name)
            target = dest.joinpath(*rel.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as src, open(target, 'xb') as out:
                shutil.copyfileobj(src, out)
    verify_payload(manifest, dest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive')
    parser.add_argument('--manifest')
    parser.add_argument('--payload-root')
    parser.add_argument('--extract-to')
    args = parser.parse_args()
    if args.archive:
        result = extract_verified(args.archive, args.extract_to) if args.extract_to else verify_archive(args.archive)
        print(json.dumps(dict(complete=True, archive=args.archive, files=len(result['files']), extracted_to=args.extract_to)))
    elif args.manifest and args.payload_root:
        result = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
        print(json.dumps(dict(complete=True, files=verify_payload(result, args.payload_root))))
    else:
        parser.error('Use --archive [--extract-to NEW_EMPTY_DIRECTORY] or --manifest with --payload-root')


if __name__ == '__main__':
    main()
