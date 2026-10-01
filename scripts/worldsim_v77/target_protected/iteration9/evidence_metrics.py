"""几何其他帧支持/无支持/未知分开计，不能称精确真实纹理对应。"""
from pathlib import Path
import sys
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).parent));from temporal_factory import T,R8,read,dump
from temporal_metrics import cell_pixels
O=T/'r10'
def main():
 plan=read(O/'evaluation_plan.json');sources={c['source_id']:c for c in read(R8/'factory/source_manifest.json')['clips']};cases=[]
 for c in plan['cases']:
  if c['kind']!='synthetic' or c['type']=='background':continue
  folder=Path(c['folder']);pair=read(folder/'pair_manifest.json');src=sources[pair['source_id']];load=lambda r,i:np.asarray(Image.open(folder/r/f'{i:03}.png'))
  holes=[load('model_hole',i)>0 for i in range(10)];ys=[load('Y',i).astype('float32')/255 for i in range(10)];tokens=pair['protected_instances'];masks=[];known=[];support=[]
  for i in range(10):
   masks.append(np.zeros((576,1024),bool));known.append(np.zeros((576,1024),bool));support.append(np.zeros((576,1024),bool))
  for tok in tokens:
   ms=[np.asarray(Image.open(folder/'protected'/f'{i:03}_{tok}.png'))>0 for i in range(10)];cells=[];visible=[]
   for i,(m,h,f) in enumerate(zip(ms,holes,src['frames'])):
    actor=next(a for a in f['actors'] if a['instance_token']==tok);yy,xx,keys=cell_pixels(f,actor,m);cells.append((yy,xx,keys));visible.append(set(keys[~h[yy,xx]].tolist()))
   for i,(m,h,(yy,xx,keys)) in enumerate(zip(ms,holes,cells)):
    other=set().union(*(v for j,v in enumerate(visible) if j!=i));hit=np.isin(keys,list(other));masks[i]|=m&h;known[i][yy[h[yy,xx]],xx[h[yy,xx]]]=True;support[i][yy[hit&h[yy,xx]],xx[hit&h[yy,xx]]]=True
  row={'eval_id':c['eval_id'],'scene':c['receiver_scene'],'suite':c['suite'],'process_family':c.get('process_family','r8_validation_control'),'frames':[],'arms':{}}
  regions={'geometric_other_frame_support':support,'no_geometric_other_frame_support':[k&~s for k,s in zip(known,support)],'unknown_correspondence':[m&~k for m,k in zip(masks,known)]}
  for i in range(10):row['frames'].append({'frame':i,**{r:int(v[i].sum()) for r,v in regions.items()},'protected_hidden':int(masks[i].sum())})
  for arm in plan['arms']:
   vals={k:[] for k in regions}
   for i in range(10):
    raw=np.asarray(Image.open(O/'evaluation'/c['eval_id']/(arm+'_native')/f'{i:05}.png')).astype('float32')/255;err=np.abs(raw-ys[i]).mean(-1)
    for k,v in regions.items():
     if v[i].any():vals[k].append(float(err[v[i]].mean()))
   row['arms'][arm]={k:float(np.mean(v)) if v else None for k,v in vals.items()}
  cases.append(row)
 dump(O/'evidence_conditioned_metrics.json',{'cases':cases,'semantics':'true Y visible SAM2 B pixels intersect annotated oriented GT cuboid16-cell; other-frame support approximate, not exact surface/texture validation','real_evidence':'UNKNOWN hidden RGB without actor-free GT','background_evidence':'not measured, remainsUNKNOWN','human_verdict':None});print('evidence cases',len(cases))
if __name__=='__main__':main()
