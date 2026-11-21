"""Once-only reserve selection from actual activation alpha, no GPU or new parameter search."""
from pathlib import Path
import argparse,json,hashlib,time
import numpy as np
P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_surface_adaptive_20261121_r1')
def main(root):
 out=P/'proposal';manifest=out/'append_manifest.json'
 if manifest.exists():print(manifest.read_text());return
 with np.load(out/'append_reserve.npz')as z:d={k:z[k]for k in z.files}
 alpha={r:np.load(root/f'{r:03d}.npz')['road_alpha']for r in[86,110]};fit={r:np.load(out/f'domains/{r:03d}.npz')['fit']for r in[86,110]};keep=np.zeros(len(d['world_xyz']),bool)
 for i in range(len(keep)):
  for j,r in enumerate([86,110]):
   if d['support_frame_ids'][i,j]!=r:continue
   x,y=d['source_pixels'][i,j];y0,y1=max(0,y-1),min(768,y+2);x0,x1=max(0,x-1),min(1152,x+2)
   if np.any(fit[r][y0:y1,x0:x1]&(alpha[r][y0:y1,x0:x1]<.95)):keep[i]=True;break
 selected={k:v[keep]if getattr(v,'ndim',0)>0 and len(v)==len(keep)else v for k,v in d.items()};dest=out/'append.npz';np.savez_compressed(dest,**selected)
 data=dict(schema='adaptive_surface_append_selection.v1',complete=True,reserve=len(keep),appended_new_geometry=int(keep.sum()),discarded_after_activation=int((~keep).sum()),rule='single pass fixed source center3x3 neighborhood AND fixed legal fit mask, actual activation alpha<.95 in either lawful support view; no teacherRGB or new thresholds',activation_render_root=str(root),append_npz=str(dest),append_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),selected_reserve_ids=np.flatnonzero(keep).tolist(),CPU_only=True,CUDA_calls=0)
 temp=manifest.with_suffix('.json.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(manifest);print(json.dumps(data))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--activation-render-root',type=Path,required=True);a=ap.parse_args();main(a.activation_render_root)
