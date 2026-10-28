"""Post-intervention dark-pixel tracing, explicitly excluded from labels/training."""
import argparse,sys,time,json
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from probe_detection import load_native,kwargs,attribution,authorized,write,torch,cpu,exclusive

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    authorized();a.output.mkdir(parents=True,exist_ok=False);records=[];start=time.monotonic()
    lum=np.array([.2126,.7152,.0722],np.float32)
    with exclusive():
        torch.set_num_threads(1);n,actor,ds,frames=load_native();ns=len(n.static._xyz)
        for i,f in enumerate(frames):
            raw=f['raw'];root=a.run/f'{raw:03d}'
            if not root.exists():continue
            cam,st,dy=kwargs(n,actor,f,i);domain=np.load(root/'domains.npz')['maps']
            for stage in ['existing','gates_open_same_domain','broad_once','peel_01']:
                authorized();source=root/stage;state=np.load(source/'maps.npz');arr=state['native_rgb'];display=np.clip(arr,0,1)
                brightness=(display*lum[:,None,None]).sum(0);dark=brightness<.2
                # Only a diagnostic query; dark != vehicle/shadow ground truth.
                masks=np.stack([dark*(domain[2]>0),dark*(domain[3]>0),dark*(domain[1]>0),np.ones_like(dark)]).astype('float32')
                rem=np.load(source/'contributors.npz')['removed_ids'];kw={**st,'opacities':st['opacities'].clone()};kw['opacities'][torch.from_numpy(rem.astype('int64')).cuda()]=0
                out,w,c=attribution(n,cam,kw,masks);w=cpu(w);ids=np.flatnonzero(w[:,2]>0)
                dest=a.output/f'{raw:03d}'/stage;dest.mkdir(parents=True)
                np.savez_compressed(dest/'contributors.npz',ids=ids.astype('int32'),weights=w[ids],dark_mask=dark,diagnostic_regions=masks[:3])
                regions={}
                for j,name in enumerate(['body','effect','broad']):
                    region=domain[[2,3,1][j]]>0;mask=masks[j]>0
                    g=state['gaussian_rgb'];e=state['environment_rgb']
                    g_lum=(np.clip(g,0,1)*lum[:,None,None]).sum(0)
                    regions[name]=dict(dark_pixels=int(mask.sum()),dark_threshold_counts={str(t):int(((brightness<t)&region).sum()) for t in [.1,.2,.3]},
                        alpha_mass=float(state['alpha'][mask].sum()),gaussian_rows_above_1e4=int((w[:,j]>1e-4).sum()),
                        native_rgb_integral=arr[:,mask].sum(1).tolist(),gaussian_rgb_integral=g[:,mask].sum(1).tolist(),environment_rgb_integral=e[:,mask].sum(1).tolist(),
                        negative_environment_pixels=int(((e<0).any(0)&mask).sum()),would_not_be_dark_without_environment=int(((g_lum>=.2)&mask).sum()))
                overlay=Image.fromarray(np.uint8(np.clip(display.transpose(1,2,0)*255,0,255)));overlay.save(dest/'native.png')
                Image.fromarray(np.uint8(masks[2]*255)).save(dest/'diagnostic_dark_mask.png')
                record=dict(raw=raw,stage=stage,regions=regions,adjoint_closure=c,diagnostic_masks_from_post_render=True,
                    labels_or_proposals_modified=False,darkness_is_not_vehicle_ground_truth=True)
                write(dest/'receipt.json',record);records.append(record);print('DARK',raw,stage,regions['body']['dark_pixels'],regions['body']['gaussian_rows_above_1e4'],flush=True)
                del out,w
        assert n.unchanged()
    write(a.output/'summary.json',dict(records=records,channels=['dark_body','dark_effect','dark_broad','total'],training_updates=0,used_to_generate_labels=False,source_checkpoint_unchanged=True,wall_seconds=time.monotonic()-start))

if __name__=='__main__':main()
