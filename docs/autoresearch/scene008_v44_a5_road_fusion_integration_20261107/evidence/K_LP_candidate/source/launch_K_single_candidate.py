from pathlib import Path
import hashlib, json, os, subprocess, time

R = Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_v44_a5_road_fusion_integration_20261107_r1')
L = R / 'J4_road_parent_operator/K_single_v1'
M = R / 'J4_road_parent_operator/K_candidate_material_v1/scene_manifest.json'
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

assert sha(M) == '9f768a44d90f7486ee4ad9346e487d80555be85a733a39cb2c6d0fb9b3936beb'
assert sha(L/'K_candidate_parent_RGB.npz') == '41a570ba2f5c7544b8ddeaef730142368f660234a3b53721d88570b0dea1903a'
assert sha(L/'result.json') == 'f22d1c51c961491b4cd7606120d3682cd457490937547a41927ae4dab6a03add'
assert json.loads((L/'result.json').read_text())['status'] == 'FEASIBLE_CPU_CERTIFIED_PENDING_REAL8F'
source = R/'source/render_K_single_LP_candidate.py'
compile(source.read_text(), str(source), 'exec', dont_inherit=True)
env = os.environ.copy()
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    env[k] = '1'
cmd = ['/root/autodl-tmp/envs/drivingeditor/bin/python','-B',str(source),'--scene-manifest',str(M),'--candidate',str(L/'K_candidate_parent_RGB.npz'),'--certificate',str(L/'result.json')]
with (R/'J4_road_parent_operator/K_candidate_render_v1.log').open('x') as log:
    p = subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=R,env=env,start_new_session=True,stdin=subprocess.DEVNULL)
record = {'pid':p.pid,'command':cmd,'source_sha256':sha(source),'candidate_sha256':sha(L/'K_candidate_parent_RGB.npz'),'single_LP_certificate_sha256':sha(L/'result.json'),'scene_manifest_sha256':sha(M),'unix':time.time(),'planned_native_forwards':8,'backwards':0,'Adam_updates':0,'no_automatic49':True}
(R/'contract/K_single_LP_candidate_launch.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
