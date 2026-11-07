from pathlib import Path
import argparse,json,os,subprocess,time
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_v44_a5_c49_rgb_20261107_r1')
p=argparse.ArgumentParser();p.add_argument('--step',type=int,choices=[500,2000],required=True);a=p.parse_args();stage=a.step
mf=R/f'formal_export{stage}_r1/scene_manifest.json';cp=R/f'formal2000_r1/state{stage}.pth';assert mf.exists() and cp.exists()
env=os.environ.copy()
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[k]='1'
cmd=['/root/autodl-tmp/envs/drivingeditor/bin/python','-B',str(R/'source/render_C49_H2_material_transplant.py'),'--scene-manifest',str(mf),'--checkpoint',str(cp),'--step',str(stage)]
with (R/f'H2_material{stage}.log').open('x') as log:proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=R,env=env,start_new_session=True,stdin=subprocess.DEVNULL)
record={'pid':proc.pid,'step':stage,'command':cmd,'unix':time.time(),'native_forwards_planned':8,'dummy_backwards':0,'optimizer_updates':0,'fixed_H2_qualifications_reused':True};(R/'contract'/f'H2_material{stage}_launch.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
