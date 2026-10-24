"""Export existing 49-frame evidence for a CPU-only V42 propagation audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import argparse,json,sys,time,tarfile
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    sys.path.insert(0,str(a.campaign/'tools'))
    from prepare_v41_dense import masks,cloud_and_boxes
    from prepare_v41_revision import native_rgb
    dense=Path(str(a.campaign)+'_dense_r7');dm=json.loads((dense/'manifest.json').read_text());bundle=Path(dm['source_bundle']);m=json.loads(bundle.read_text())
    anchor=np.asarray(m['coordinate_anchor']);inv=np.linalg.inv(anchor)
    plane=np.asarray(json.loads(Path(m['original_readiness']).read_text())['SGCM_geometry']['ground_fit']['plane_anchor'])
    source=Path(m['original_engineering']).parent/'inputs/scene0008_089_137'
    with np.load(bundle.parent/m['units']) as z:
        keys=['world','normal_world','surface_supported','base_weight','provenance','keys','xy_local','unit_ids']
        u={k:z[k] for k in keys}
    u['local']=u['world']@inv[:3,:3].T+inv[:3,3];u['normal_local']=u['normal_world']@anchor[:3,:3]
    np.savez_compressed(a.output/'units.npz',**u)
    meta=dict(frame_order=m['frame_order'],actors=m['actors'],anchor=anchor.tolist(),ground_plane=plane.tolist(),source_bundle=str(bundle),GPU_used=False,optimizer_updates=0)
    for i,fid in enumerate(m['frame_order']):
        fm=m['frames'][fid];cloud,ground,boxes=cloud_and_boxes(source,fid,inv,plane)
        with np.load(dense/'frames'/f'{fid}.npz') as z:
            fields={k:z[k] for k in ['actor_labels','deletion_valid','deletion_source_ids','deletion_target_rgb','unit_id','fixed_surface_depth_m','supported','vehicle_body','vehicle_edge']}
        fields.update(rgb=native_rgb(fm['original_rgb']),guard=masks(fm)|fields['vehicle_body'].any(0)|fields['vehicle_edge'].any(0),K=np.asarray(fm['intrinsic']),w2l=np.asarray(fm['world_to_camera'])@anchor,cloud=cloud@inv[:3,:3].T+inv[:3,3],ground=ground)
        np.savez_compressed(a.output/f'{fid}.npz',**fields)
        print(json.dumps({'exported':fid,'index':i}),flush=True)
    (a.output/'manifest.json').write_text(json.dumps(meta,indent=2))
    with tarfile.open(str(a.output)+'.tar.gz','w:gz') as t: t.add(a.output,arcname=a.output.name)
    print(json.dumps({'archive':str(a.output)+'.tar.gz'}),flush=True)
if __name__=='__main__':main()
