"""封装原生闭环试跑证据；保留实际执行与拟合尚未完成的边界。"""
import argparse,json,shutil,subprocess,tarfile
from pathlib import Path
N=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-NATIVE-LIDAR-01/20260915-r1')
I=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-INTERACTION-01/20260915-r1')
B=Path('/root/autodl-tmp/external/worldsim_simimpact')
ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--fit',type=Path,required=True);ap.add_argument('--checkpoint-dir',type=Path,required=True);args=ap.parse_args()
reg=json.loads((args.run/'registration.json').read_text());assert reg['completed']
dest=args.run/'export'
if dest.exists():raise RuntimeError(f'Preserve existing export {dest}')
dest.mkdir();shutil.copy2(args.run/'registration.json',args.run/'registration.before_milestone5.json')
reg.update(fit=str(args.fit),checkpoint_dir=str(args.checkpoint_dir),
    policy_checkpoint=str(I/'lidar_policy/assets/transfuser_seed_0.ckpt'),
    source_revisions={r:subprocess.check_output(['git','-C',str(B/r),'rev-parse','HEAD'],text=True).strip() for r in ['neurad-studio','SplatAD_splat','NAVSIM']},
    failure_ledger_delta='V74-H2-F18',source_snapshot=str(args.run/'source_snapshot'),
    scientific_status='Integration pilot only. RGB-channel replacement reproduces the observed replay gap; LiDAR-only effects small. No geometry causal claim, final 30001-step fit remains in progress.')
(args.run/'registration.json').write_text(json.dumps(reg,indent=2))
source=args.run/'source_snapshot';source.mkdir()
for name in ['native_splatad_bridge.py','run_splatad_closed_loop.py','audit_native_closed_loop.py']:
    shutil.copy2(N/'scripts'/name,source/name)
for name in ['registration.json','sensor_interface_qa.json','policy_execution_summary.json','downstream_audit.json']:
    shutil.copy2(args.run/name,dest/name)
shutil.copy2(N/'native_policy_runtime.json',dest/'native_policy_runtime.json')
shutil.copy2(N/'closed_loop/reference'/f"{reg['scene']}.json",dest/'log_reference.json')
views=json.loads((I/'inputs'/f"{reg['scene']}.json").read_text())['views']
shutil.copy2(views[0]['image'],dest/'real_front_initial.jpg')
shutil.copy2(args.run/'replay/frame_00/front.jpg',dest/'render_front_initial.jpg')
for condition in ['closedloop_raw','closedloop_median']:
    target=dest/condition;target.mkdir()
    shutil.copy2(args.run/condition/'executed_states.npz',target/'executed_states.npz')
    for i in range(reg['steps']):shutil.copy2(args.run/condition/f'frame_{i:02d}/front.jpg',target/f'front_{i:02d}.jpg')
with tarfile.open(args.run/'export.tar.gz','w:gz') as f:
    for p in dest.rglob('*'):
        if p.is_file():f.add(p,arcname=str(p.relative_to(dest)))
print('EXPORTED',dest,flush=True)
