from pathlib import Path
import hashlib,json,os,subprocess,time
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_v44_a5_road_fusion_integration_20261107_r1');L=R/'J4_road_parent_operator/lp_single_v1';M=R/'J4_road_parent_operator/J_candidate_material_v1/scene_manifest.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(M)=='ec974e28dc811c88ce356032e5afb30fc038d8c44f12a6fcb3cae06b1af1db19'
assert sha(L/'J_candidate_parent_RGB.npz')=='eed9adb3d85bdf9858b73a04323fe61ad8f55b6ebf2ed2bab4ad255b4e263e3d'
assert json.loads((L/'result.json').read_text())['status']=='FEASIBLE_CPU_CERTIFIED_PENDING_REAL8F'
source=R/'source/render_J_single_LP_candidate.py';env=os.environ.copy()
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[k]='1'
cmd=['/root/autodl-tmp/envs/drivingeditor/bin/python','-B',str(source),'--scene-manifest',str(M),'--candidate',str(L/'J_candidate_parent_RGB.npz'),'--certificate',str(L/'result.json')]
with (R/'J4_road_parent_operator/lp_candidate_render_v1.log').open('x') as log:p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=R,env=env,start_new_session=True,stdin=subprocess.DEVNULL)
record={'pid':p.pid,'command':cmd,'source_sha256':sha(source),'candidate_sha256':sha(L/'J_candidate_parent_RGB.npz'),'single_LP_certificate_sha256':sha(L/'result.json'),'scene_manifest_sha256':sha(M),'unix':time.time(),'planned_native_forwards':8,'backwards':0,'optimizer_updates':0,'no_automatic49':True};(R/'contract/J_single_LP_candidate_launch.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
