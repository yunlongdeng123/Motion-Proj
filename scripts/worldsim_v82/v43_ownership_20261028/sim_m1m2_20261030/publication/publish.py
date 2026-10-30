"""Bounded publication, verification and staging. No jobs, commit, push or power action."""
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
import hashlib
import json
import shutil
import subprocess
import tarfile

REPO = Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
BASE = Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/sim_mechanism_20261030/publication_m1m2')
PREFIXES = ('docs/autoresearch/v43_ownership_20261028/sim_m1m2_20261030/',
            'scripts/worldsim_v82/v43_ownership_20261028/sim_m1m2_20261030/',
            'docs/autoresearch/v43_ownership_20261028/gt_protocol_r5_20261030/',
            'scripts/worldsim_v82/v43_ownership_20261028/gt_protocol_r5_20261030/')
UNRELATED = 'scripts/run_worldsim_v72_e3_render_bridge.py'


def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args]).decode('utf8')


def main():
    with tarfile.open(BASE / 'publication.tar.gz', 'r:gz') as archive:
        manifest = json.loads(archive.extractfile('PUBLICATION_MANIFEST.json').read())
        assert manifest['task_id'] == 'WS-V43-SIM-M1-M2-20261030'
        members = archive.getmembers()
        assert all(m.isfile() for m in members)
        assert len({m.name for m in members}) == len(members)
        assert {m.name for m in members} == set(manifest['files']) | {'PUBLICATION_MANIFEST.json'}
        payload = {}
        for name in manifest['files']:
            p = PurePosixPath(name)
            assert not p.is_absolute() and '..' not in p.parts and '\\' not in name
            assert name == 'docs/RESEARCH_STATUS.md' or name.startswith(PREFIXES)
            assert (REPO / name).resolve().is_relative_to(REPO.resolve())
            payload[name] = archive.extractfile(name).read()
            assert hashlib.sha256(payload[name]).hexdigest() == manifest['sha256'][name]
    initial_staged = set(git('diff', '--cached', '--name-only', '-z').split('\0')) - {''}
    assert initial_staged <= set(payload) | {'docs/EXPERIMENTS.md'}, 'Unrelated staged changes preserved; stop.'
    before = (REPO / UNRELATED).read_bytes()
    backup = BASE / ('before_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
    backup.mkdir()
    (backup / 'status.txt').write_text(git('status', '--short'))
    for name, data in payload.items():
        target = REPO / name
        if target.exists():
            old = backup / name
            old.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, old)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        assert hashlib.sha256(target.read_bytes()).hexdigest() == manifest['sha256'][name]
    index = REPO / 'docs/EXPERIMENTS.md'
    shutil.copy2(index, backup / 'EXPERIMENTS.md')
    content = index.read_text()
    if manifest['task_id'] not in content:
        index.write_text(content.rstrip() + '\n\n' + manifest['experiment_entry'] + '\n')
    allowed = sorted([*payload, 'docs/EXPERIMENTS.md'])
    git('add', '--', *allowed)
    git('diff', '--cached', '--check')
    staged = set(git('diff', '--cached', '--name-only', '-z').split('\0')) - {''}
    assert staged <= set(allowed)
    assert (REPO / UNRELATED).read_bytes() == before
    (BASE / 'staged_stat.txt').write_text(git('diff', '--cached', '--stat'))
    receipt = {'published_files': len(payload), 'sha256_verified': len(payload), 'backup': str(backup),
               'stage_allowlist': allowed, 'staged_count': len(staged), 'power_actions': 0,
               'known_unrelated_dirty_path_preserved': UNRELATED}
    (BASE / 'publication_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'stage_allowlist'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
