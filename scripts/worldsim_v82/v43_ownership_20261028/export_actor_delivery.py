"""CPU export of complete actor evaluation nodes + small visualization fields."""
import argparse,json,shutil,time,hashlib,tarfile
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from segmentation import ROOT,CONTEXTS,write
from export_diagnostics import pack
def main():
    a=argparse.ArgumentParser();a.add_argument('--name',required=True);p=a.parse_args();dest=ROOT/p.name;dest.mkdir(exist_ok=False);index=[]
    for ctx in CONTEXTS:
        root=ROOT/'actor_campaign_v1'/ctx/'formal10k';dataset=ROOT/'datasets'/ctx;manifest=json.loads((dataset/'manifest.json').read_text());nodes=[]
        out=dest/ctx;out.mkdir();write(out/'dataset_manifest.json',manifest)
        for step in [1000,5000,10000]:
            source=root/'evaluation'/('%06d'%step);checkpoint=root/('checkpoint%06d.pth'%step)
            if not (source/'summary.json').exists() or not checkpoint.exists():continue
            summary=json.loads((source/'summary.json').read_text());assert len(summary['frames'])==49
            payload=torch.load(str(checkpoint),map_location='cpu');state=payload['actor'];n=len(state['xyz']);quat=state['quat'].numpy();quat/=np.linalg.norm(quat,axis=1,keepdims=True)
            cloud=dict(total=n,sample=n,fullPopulation=True,cycle=1.,row_namespace='v43_measured_actor',data=pack(dict(xyz=state['xyz'].numpy().astype('<f4'),rgb=(np.clip(state['dc'].numpy()[:,0]*.28209479177387814+.5,0,1)*255).round().astype('uint8'),opacity=state['opacity'].sigmoid().numpy().astype('<f4').ravel(),ids=np.arange(n,dtype='<i4'),scales=state['log_scale'].exp().numpy().astype('<f4'),rotation=Rotation.from_quat(np.roll(quat,-1,axis=1)).as_matrix().astype('<f4'),t=np.zeros(n,'<f4'),sigma_t=np.full(n,1000.,'<f4'),velocity=np.zeros((n,3),'<f4'))))
            target=out/('%06d'%step);shutil.copytree(source,target);write(target/'cloud.json',cloud)
            poses={}
            for f in manifest['frames']:
                m=np.eye(4);m[:3,:3]=f['actor_rotation_world'];m[:3,3]=f['actor_center_world'];poses[str(f['raw'])]=m.tolist()
            write(target/'poses_world.json',poses);nodes.append(dict(step=step,summary=summary,cloud_rows=n,checkpoint_bytes=checkpoint.stat().st_size,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()))
        index.append(dict(context=ctx,nodes=nodes))
    write(dest/'index.json',index);files={str(f.relative_to(dest)):dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in dest.rglob('*') if f.is_file()};write(dest/'manifest.json',files)
    archive=ROOT/(p.name+'.tar.gz')
    with tarfile.open(archive,'w:gz') as t:t.add(dest,arcname=p.name)
    write(ROOT/(p.name+'_receipt.json'),dict(archive=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),models_exported=False,visualization_fields_only=True));print(json.dumps(index))
if __name__=='__main__':main()
