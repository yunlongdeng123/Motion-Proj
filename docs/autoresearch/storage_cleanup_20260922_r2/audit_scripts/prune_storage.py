"""按已审阅清单清理普通文件；先记账，再删除，拒绝路径漂移和活动文件。"""
import argparse
import json
import os
from pathlib import Path
import stat
import time

ROOT = Path('/root/autodl-tmp').resolve()
AUDIT = ROOT/'cleanup_manifests/20260922-r2'
ALLOWED = [ROOT/'models/worldsim_v74_mainfig', ROOT/'data/worldsim_v4', ROOT/'runs']

def open_paths():
    found = set()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            for fd in (proc/'fd').iterdir():
                try:
                    found.add(os.readlink(fd).removesuffix(' (deleted)'))
                except OSError:
                    pass
            for line in (proc/'maps').read_text().splitlines():
                parts = line.split(maxsplit=5)
                if len(parts) == 6 and parts[5].startswith('/'):
                    found.add(parts[5].removesuffix(' (deleted)'))
        except (OSError, PermissionError):
            pass
    return found

def describe(p, category, restore):
    s = p.lstat()
    assert stat.S_ISREG(s.st_mode) and p.resolve() == p
    return dict(path=str(p), bytes=s.st_size, allocated=s.st_blocks*512,
                inode=s.st_ino, device=s.st_dev, mtime_ns=s.st_mtime_ns,
                links=s.st_nlink, category=category, restore=restore)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('plan', type=Path)
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    rows = json.loads(args.plan.read_text())['files']
    active = open_paths()
    seen = set()
    for row in rows:
        p = Path(row['path'])
        assert str(p) not in seen
        seen.add(str(p))
        assert p.resolve() == p and any(p.is_relative_to(a) for a in ALLOWED), p
        s = p.lstat()
        assert stat.S_ISREG(s.st_mode), p
        assert [s.st_size, s.st_ino, s.st_dev, s.st_mtime_ns, s.st_nlink] == [row[k] for k in ['bytes','inode','device','mtime_ns','links']], p
        assert str(p) not in active, f'active: {p}'
    summary = {'phase': args.plan.stem, 'files': len(rows), 'logical_bytes': sum(r['bytes'] for r in rows),
               'expected_released_bytes': sum(r['allocated'] for r in rows if r['links'] == 1),
               'applied': False}
    if args.apply:
        before = os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize
        with (AUDIT/(args.plan.stem+'-deletions.jsonl')).open('x') as log:
            for i, row in enumerate(rows):
                if i % 100 == 0:
                    active = open_paths()
                p = Path(row['path'])
                assert str(p) not in active
                s = p.lstat()
                assert p.resolve() == p and s.st_ino == row['inode'] and s.st_size == row['bytes'] and s.st_mtime_ns == row['mtime_ns']
                log.write(json.dumps({'event':'delete_intent', 'path':str(p), 'time':time.time()})+'\n')
                log.flush()
                os.fsync(log.fileno())
                p.unlink()
                log.write(json.dumps({'event':'deleted', 'path':str(p)})+'\n')
            log.flush()
            os.fsync(log.fileno())
        after = os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize
        summary.update(applied=True, available_before_bytes=before, available_after_bytes=after,
                       actual_available_increase_bytes=after-before)
        (AUDIT/(args.plan.stem+'-result.json')).write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)

if __name__ == '__main__':
    main()
