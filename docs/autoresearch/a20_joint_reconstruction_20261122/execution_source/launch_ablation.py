from pathlib import Path
import subprocess
root = Path('/root/autodl-tmp/drivingeditor_repro/campaign/a20_joint_reconstruction_20261122_r1')
run = root/'execution_r3'
command = ['/root/autodl-tmp/envs/drivingeditor/bin/python', '-u', str(root/'runtime/post_geometry_ablation.py'),
    '--runtime-dir', str(run/'source'), '--checkpoint', str(run/'geometry/checkpoints/geometry_step0040.pt'),
    '--evidence', str(run/'geometry/contribution/step0040.npz'),
    '--init', str(root/'initialization/joint_actor_init.npz'), '--manifest', str(root/'initialization/manifest.json'),
    '--source-manifest', str(root/'native_bindings.json'), '--context-plan', str(root/'context_plan.json'),
    '--prior-contract', '/root/autodl-tmp/drivingeditor_repro/campaign/a20_sd15_sameview_20261122_r1/student/contract.json',
    '--path-map-file', str(run/'source/remote_path_map.json'), '--output', str(run/'post_geometry_ablation')]
with (run/'post_geometry_ablation.log').open('w') as log:
    code = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT).returncode
raise SystemExit(code)
