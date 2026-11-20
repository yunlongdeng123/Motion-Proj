"""由 root 发布后显式执行；备份并应用文档，不触 index。"""
from pathlib import Path
import datetime
import hashlib
import json
import shutil
import subprocess
import sys

R = Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(path):
    plan_path = Path(path)
    plan = json.loads(plan_path.read_text())
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=R, text=True).strip()
    if head != plan['expected_active_head']:
        raise RuntimeError('active HEAD changed')
    pending = []
    for rel, entry in plan['working_docs'].items():
        target, candidate = R / rel, Path(entry['candidate'])
        if target.read_bytes() == candidate.read_bytes():
            continue
        if sha(target) != entry['before_sha256']:
            raise RuntimeError('working document changed since preparation; preserve it: ' + rel)
        if rel != 'docs/RESEARCH_STATUS.md' and not candidate.read_bytes().startswith(target.read_bytes()):
            raise RuntimeError('append must preserve all original bytes: ' + rel)
        pending.append((rel, target, candidate))
    backup = plan_path.parent / ('working_apply_before_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    for rel, target, candidate in pending:
        saved = backup / rel
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, saved)
    for rel, target, candidate in pending:
        temp = target.with_name(target.name + '.teacher-detail-prepared-tmp')
        temp.write_bytes(candidate.read_bytes())
        temp.replace(target)
    result = dict(applied=[entry[0] for entry in pending], backup=str(backup), research_status_same_snapshot=True,
                  parallel_2D_status_preserved=bool(plan['parallel_status_preserved_in_working_candidate']),
                  experiments_and_F06_original_bytes_preserved_as_prefix=True, git_index_used=False,
                  HEAD_unchanged=True, failure_index_regeneration=str(plan_path.parent / 'failure_index_regeneration.json'))
    (plan_path.parent / 'working_docs_applied.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main(sys.argv[1])
