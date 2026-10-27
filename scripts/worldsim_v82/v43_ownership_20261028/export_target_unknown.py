"""First-scene RGB, exact learned row parameters and measured background cloud."""
import json,shutil,tarfile,hashlib,argparse
from pathlib import Path
import numpy as np
from segmentation import ROOT,write
from export_diagnostics import pack
from probe_target_unknown import CTX,OLD

def main():
    p=argparse.ArgumentParser();p.add_argument('--name',default='target_unknown_delivery_v1');p.add_argument('--conditional',action='store_true');a=p.parse_args();R=ROOT/'target_unknown_v2';D=ROOT/a.name;D.mkdir(exist_ok=False);index=[]
    T=np.loadtxt(OLD/'motion_scene_inputs_v1'/CTX/'transform_matrix.txt');Ti=np.linalg.inv(T);scale=np.linalg.norm(T[:3,0])
    plans=[('conditional_joint',R/'conditional_pair/joint/formal'),('conditional_split',R/'conditional_pair/frozen_split/formal')] if a.conditional else [('joint199',R/'formal199/joint/formal'),('split199',R/'formal199/frozen_split/formal'),('joint_dense199',R/'dense199/joint/formal')]
    for name,run in plans:
        assert json.loads((run/'status.json').read_text())['phase']=='complete';dst=D/name;dst.mkdir();shutil.copytree(run/'native_rgb',dst/'native_rgb');shutil.copy2(run/'summary.json',dst/'summary.json');shutil.copy2(run.parent/'restore/restore_verification.json',dst/'restore_verification.json');d=np.load(run/'decomposition_rows.npz');xyz=d['background_xyz_model'].astype(float)@Ti[:3,:3].T+Ti[:3,3];N=len(xyz)
        adapter=pack(dict(ids=d['static_ids'].astype('int32'),opacity_factor=d['background_retention'].astype('float32').ravel(),dc_rgb_delta=(d['static_dc_delta'][:,0]*.28209479177387814).astype('float32'),**({'time_phi':d['time_phi'].astype('float32'),'raws':np.arange(77,126,dtype='int32')} if 'time_phi' in d.files else {})))
        cloud=dict(total=N,sample=N,fullPopulation=True,cycle=1.,row_namespace='v43_measured_background',branch='static',coordinate_space='world',data=pack(dict(xyz=xyz.astype('float32'),rgb=np.round(np.clip(d['background_dc'][:,0]*.28209479177387814+.5,0,1)*255).astype('uint8'),opacity=d['background_opacity'].astype('float32').ravel(),ids=np.arange(N,dtype='int32'),scales=(d['background_sigma_model']/scale).astype('float32'),rotation=np.broadcast_to(np.eye(3,dtype='float32'),(N,3,3)).copy(),t=np.zeros(N,'float32'),sigma_t=np.full(N,1000.,'float32'),velocity=np.zeros((N,3),'float32'))))
        write(dst/'adapter.json',adapter);write(dst/'cloud.json',cloud)
        if (run/'novel_camera').exists():shutil.copytree(run/'novel_camera',dst/'novel_camera')
        index.append(dict(name=name,frames=49,step=10000,source_run=str(run)))
    if a.conditional:
        for k in [75,95]:shutil.copytree(R/'time_local_probe'/('purity'+str(k))/'native_rgb',D/('time_local'+str(k))/'native_rgb')
        shutil.copy2(R/'time_local_probe/masks.json',D/'time_local_masks.json');shutil.copy2(R/'time_local_probe/summary.json',D/'time_local_summary.json')
        for name in ['conditional_pair','conditional_prior_pilot']:shutil.copy2(R/name/'controller.json',D/(name+'_controller.json'))
        write(D/'index.json',dict(context=CTX,models_exported=False,experiments=index,quality_accepted=False,human_verdict=None))
        finish(D,a.name);return
    for control in ['baseline','target_upper_bound','target_preserve_observed_bg']:
        shutil.copytree(R/'evidence_probe'/control/'native_rgb',D/control/'native_rgb')
    for name,path in [('probe',R/'evidence_probe'),('memory49',R/'clean_memory'),('memory199_v1',R/'clean_memory199'),('memory199',R/'clean_memory199v2'),('dense_memory',R/'clean_memory199_dense_r2')]:shutil.copy2(path/'summary.json',D/(name+'_summary.json'))
    ev=np.load(R/'evidence_probe/evidence.npz');ids=np.flatnonzero(ev['target_candidate']|ev['background_protected']);role=np.full(len(ids),2,np.uint8);role[ev['background_protected'][ids]]=0;role[ev['mixed_target_background'][ids]]=3;role[ev['vehicle_candidate'][ids]]=1
    write(D/'target_evidence.json',pack(dict(ids=ids.astype('int32'),role=role,target_frames=ev['count_target'][ids],background_frames=ev['count_depth_background'][ids])))
    write(D/'suppression.json',dict(target_upper_bound=np.flatnonzero(ev['target_candidate']).tolist(),target_preserve_observed_bg=np.flatnonzero(ev['target_candidate']&~ev['background_protected']).tolist()))
    for name in ['formal199','dense199','pilot49']:shutil.copy2(R/name/'controller.json',D/(name+'_controller.json'))
    write(D/'index.json',dict(context=CTX,models_exported=False,experiments=index,quality_accepted=False,human_verdict=None))
    finish(D,a.name)

def finish(D,name):
    manifest={str(f.relative_to(D)):dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in D.rglob('*') if f.is_file()};write(D/'manifest.json',manifest);archive=ROOT/(name+'.tar.gz')
    with tarfile.open(archive,'w:gz') as t:t.add(D,arcname=name)
    write(ROOT/(name+'_receipt.json'),dict(archive=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),models_exported=False));print(json.dumps(dict(name=name,files=len(manifest),bytes=archive.stat().st_size)))

if __name__=='__main__':main()
