"""One explicitly authorized cleanup; never a general retention daemon.

Delete only full/real_only's 5K..25K four checkpoint files. Keep 30K, all
publication/evaluation assets, failed campaigns and both other arms unchanged.
No CUDA, optimizer, training/control process mutation, or recursive deletion.
"""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil

from evaluate_v3_final49 import (CHANNELS, COMPONENT_PREFIXES, FRAMES,
                                RETENTION_VERSION, STEPS, read, write_new)

EXPECTED_CAMPAIGN = Path('/root/autodl-tmp/drivingeditor_repro/campaign/'
                        'r15_gn11_stage1_v3_serial_r1_20261021')
ARMS = ('full', 'real_only')


def fingerprint(path):
    stat = path.stat()
    return dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, inode=stat.st_ino)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def process_guard(campaign, targets):
    """Reject a full/real worker or any process holding an exact removal file."""
    names = {str(p) for p in targets}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            command = (proc/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            if any(str(campaign/'arms'/a/'run30k') in command for a in ARMS):
                raise RuntimeError('a protected arm has an active process: '+proc.name)
            for fd in (proc/'fd').iterdir():
                try:
                    if os.readlink(fd) in names:
                        raise RuntimeError('checkpoint is open in process: '+proc.name)
                except FileNotFoundError:
                    pass
        except (FileNotFoundError, ProcessLookupError):
            pass


def make_plan(campaign):
    campaign = Path(campaign).resolve()
    removed, protected, kept = [], {}, []
    for arm in ARMS:
        run = campaign/'arms'/arm/'run30k'
        if run.resolve() != run:
            raise RuntimeError('symlinked arm/run directories are not cleanup targets')
        state = read(run/'status.json')
        if (state['status'] != 'complete' or state['iteration'] != 30000
                or state['arm'] != arm or state['second_stage_updates'] != 0
                or state['optimizer_step_counts'] != dict.fromkeys(
                    ('static', 'dynamic', 'environment'), 30000)):
            raise RuntimeError('arm is not completed fresh30K: '+arm)
        for step in STEPS:
            event_path = run/'checkpoint_events'/f'iteration_{step:06d}.json'
            event = read(event_path)
            if (event['status'] != 'complete' or event['iteration'] != step
                    or set(event['checkpoints']) != set(COMPONENT_PREFIXES)
                    or len(event['frames']) != 49
                    or {f['image_name'] for f in event['frames']} != set(FRAMES)):
                raise RuntimeError('incomplete publication')
            for name, value in event['checkpoints'].items():
                path = run/f'{COMPONENT_PREFIXES[name]}{step}.pth'
                if (value['path'] != str(path) or path.is_symlink()
                        or path.resolve().parent != run.resolve() or not path.is_file()
                        or path.stat().st_size != value['bytes'] or value['bytes'] <= 0
                        or path.stat().st_nlink != 1):
                    raise RuntimeError('unexpected checkpoint target: '+str(path))
                item = dict(arm=arm, iteration=step, component=name, path=str(path),
                            event_sha256=sha256(event_path), **fingerprint(path))
                if step < 30000:
                    removed.append(item)
                else:
                    item['sha256'] = sha256(path)
                    kept.append(item)
            for frame in event['frames']:
                if set(frame['channels']) != CHANNELS:
                    raise RuntimeError('wrong publication channels')
                for path in list(frame['channels'].values())+[frame['road_float'], frame['hybrid_metrics']]:
                    if not Path(path).is_file():
                        raise RuntimeError('missing evaluation asset')
        # Both arms have ended. Protect every non-removal file, not only models.
        targets = {x['path'] for x in removed}
        for path in run.rglob('*'):
            if path.is_file() and str(path) not in targets:
                protected[str(path)] = fingerprint(path)
    if len(removed) != 40 or len(kept) != 8:
        raise RuntimeError('wrong exact cleanup count')
    return dict(version=RETENTION_VERSION, status='planned', campaign=str(campaign),
                authorization='user_requested_full_real_only_keep_latest_30k',
                keep_iteration=30000, removed=removed, kept=kept,
                protected=protected, reclaim_bytes=sum(x['bytes'] for x in removed),
                planned_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())


def execute(plan):
    campaign = Path(plan['campaign']).resolve()
    if campaign != EXPECTED_CAMPAIGN:
        raise RuntimeError('execution is restricted to the user-named campaign')
    intent = campaign/'checkpoint_retention.intent.json'
    receipt = campaign/'checkpoint_retention.json'
    if intent.exists() or receipt.exists():
        raise RuntimeError('cleanup already attempted; inspect receipt, never blind retry')
    targets = [Path(x['path']) for x in plan['removed']]
    process_guard(campaign, targets)
    for item in plan['removed']:
        path = Path(item['path'])
        if path.is_symlink() or fingerprint(path) != {k:item[k] for k in ('bytes','mtime_ns','inode')}:
            raise RuntimeError('target changed since planning')
    for path, state in plan['protected'].items():
        if fingerprint(Path(path)) != state:
            raise RuntimeError('protected asset changed since planning')
    free_before = shutil.disk_usage(campaign).free
    write_new(intent, plan)
    deleted = []
    try:
        for item in plan['removed']:
            path = Path(item['path'])
            run = campaign/'arms'/item['arm']/'run30k'
            expected = run/f"{COMPONENT_PREFIXES[item['component']]}{item['iteration']}.pth"
            if (item['arm'] not in ARMS or item['iteration'] not in STEPS[:-1]
                    or run.resolve() != run or path != expected
                    or path.resolve().parent != run.resolve()
                    or path.is_symlink()
                    or fingerprint(path) != {k:item[k] for k in ('bytes','mtime_ns','inode')}):
                raise RuntimeError('exact deletion guard failed')
            path.unlink()  # Exact regular file only; no glob, recursion or directory removal.
            deleted.append(item['path'])
        for path, state in plan['protected'].items():
            if fingerprint(Path(path)) != state:
                raise RuntimeError('protected file changed during cleanup')
        for item in plan['kept']:
            if sha256(Path(item['path'])) != item['sha256']:
                raise RuntimeError('latest30K checksum changed')
        result = dict(plan, status='complete', protected_assets_unchanged=True,
                      latest30k_sha256_unchanged=True, free_before_bytes=free_before,
                      free_after_bytes=shutil.disk_usage(campaign).free,
                      completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write_new(receipt, result)
        return result
    except BaseException as error:
        write_new(campaign/'checkpoint_retention.failure.json',
                  dict(status='failed', deleted=deleted, error=repr(error)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--execute-user-authorized-cleanup', action='store_true')
    args = parser.parse_args()
    plan = make_plan(args.campaign)
    result = execute(plan) if args.execute_user_authorized_cleanup else plan
    print(json.dumps({key:result[key] for key in ('status','reclaim_bytes','keep_iteration')}))


if __name__ == '__main__':
    main()
