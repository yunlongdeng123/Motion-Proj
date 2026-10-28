"""Frozen native per-row world trajectory and view color; no label feedback."""
import argparse,json,time
from pathlib import Path
import numpy as np
from probe_detection import load_native,kwargs,authorized,write,torch,cpu,exclusive

def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();authorized();a.output.mkdir(parents=True,exist_ok=False)
    ids=np.unique(np.concatenate([np.load(x)['ids'] for x in a.evidence.glob('[0-9][0-9][0-9].npz')])).astype('int32');np.save(a.output/'ids.npy',ids)
    started=time.monotonic()
    with exclusive(),torch.no_grad():
        torch.set_num_threads(1);n,actor,ds,frames=load_native();idx=torch.from_numpy(ids.astype('int64')).cuda();Ti=np.linalg.inv(n.T)
        from utils.sh_utils import eval_sh
        for i,f in enumerate(frames):
            authorized();cam,st,dy=kwargs(n,actor,f,i);xyz=st['means3D'][idx]
            world=cpu(xyz).astype('float64')@Ti[:3,:3].T+Ti[:3,3]
            if st.get('colors_precomp') is not None:rgb=st['colors_precomp'][idx]
            else:
                direction=xyz-cam.camera_center;direction=direction/direction.norm(dim=1,keepdim=True).clamp_min(1e-8)
                rgb=(eval_sh(n.static.active_sh_degree,st['shs'][idx].transpose(1,2),direction)+.5).clamp_min(0)
            np.savez_compressed(a.output/('%03d.npz'%f['raw']),world=world,rgb=cpu(rgb),sigma_m=cpu(st['scales'][idx].amax(1))/n.scale,opacity=cpu(st['opacities'][idx]).reshape(-1))
        assert n.unchanged()
    write(a.output/'contract.json',dict(schema='frozen-row-observation-v1',raws=[f['raw'] for f in frames],row_count=len(ids),namespace='original_static_rows',world_dtype='float64',source_checkpoint_unchanged=True,training_updates=0,view_color='native SH evaluation, lower clamped as native rasterizer',edited_rgb_used=False,wall_seconds=time.monotonic()-started))
    print('EXPORTED',len(ids),len(frames),flush=True)
if __name__=='__main__':main()
