"""T收口：仅核对已取回的封存文件；无实验、无solver、无远端写入。"""
import hashlib
import json
import shlex
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parent
remote = '/root/autodl-tmp/drivingeditor_repro/campaign/scene008_v44_a5_road_fusion_integration_20261107_r1/M4_child_material_capacity_v1/'
pairs = []
for folder, suffix in [('T_run_v1', 'T_child_original_metrics_QP_v1/'),
                       ('released_source_snapshot', 'qp_source_released_v1/')]:
    for p in sorted((root/folder).iterdir()):
        if p.is_file(): pairs.append((p, remote+suffix+p.name))
pairs += [(root/'T_run_v1_console.log', remote+'T_child_original_metrics_QP_v1_console.log'),
          (root/'T_child_original_metrics_QP_single_solve_started.json', remote+'T_child_original_metrics_QP_single_solve_started.json')]
cmd = 'sha256sum -- '+' '.join(shlex.quote(p) for _, p in pairs)
r = subprocess.run(['ssh', 'wm-vgpu-1011', cmd], check=True, capture_output=True, text=True)
expected = {}
for line in r.stdout.splitlines():
    digest, path = line.split(None, 1); expected[path.lstrip(' *')] = digest
rows = []
for local, path in pairs:
    data = local.read_bytes(); digest = hashlib.sha256(data).hexdigest()
    assert expected[path] == digest, str(local)
    rows.append(dict(local=str(local), remote=path, bytes=len(data), sha256=digest, verified=True))
out = dict(status='PASS_ALL_T_RELEASED_SOURCE_AND_RUN_FILES_RETRIEVED_EXACT',
    files=rows, total_files=len(rows), total_bytes=sum(r['bytes'] for r in rows),
    solver_calls=0, QP_calls=0, remote_writes=0, GPU_forwards=0)
(root/'T_retrieval_verification.json').write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps({k: out[k] for k in ('status', 'total_files', 'total_bytes')}))
