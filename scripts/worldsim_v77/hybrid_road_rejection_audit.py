"""r34：把零真实支持拆成射线、来源视野、actor包络和地面采样四步；不改补景。"""
import sys,time,datetime
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
import hybrid_road_plane_control as p
import hybrid_road_patch_control as rp
from repair_common import dump,hull_mask
import numpy as np,cv2
from PIL import Image,ImageDraw
cv2.setNumThreads(4)
ROOT=p.BASE/'r34'

def main():
 assert not ROOT.exists();ROOT.mkdir();start=time.time()
 dump(ROOT/'registration.json',dict(task_id='WS-V77-HYBRID-BG-20260927',run_id='r34',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scene='official_000',target='12',queries=[0,15],sources='query+5,+10,+20 CAM0',question='Which unchanged r27 check rejects actual DELETE road rays?',intervention='Read-only attribution of existing geometric checks. No height correction, RGB writeback, threshold changes or inference.',roles='Original processed camera/LiDAR and GT envelopes; plane is local f0 assumption, not hidden surface truth. RGB used only for visual review.',limits='Envelope overlap is not exact instance occlusion. Unsupported sampled ground does not imply never observed. Audit does not authorize writing rejected source RGB.',resources=dict(cpu_threads=4,gpu_forwards=0),failure_ledger_refs=['V77-F02'],human_verdict=None,background_input_dir=None))
 rows=[]
 for q in [0,15]:
  _,c,k,im,_,_,_=rp.source(q);domain=np.array(Image.open(rp.OLD/'mask'/f'{q:05}.png'))>0
  y,x=np.where(domain);pix=np.stack([x,y,np.ones_like(x)],1);rays=pix@np.linalg.inv(k).T@c[:3,:3].T;den=rays@rp.NORMAL;depth=-(c[:3,3]@rp.NORMAL+rp.OFFSET)/np.where(np.abs(den)>1e-8,den,np.nan);world=c[:3,3]+rays*depth[:,None];eligible=np.isfinite(depth)&(depth>3)&(depth<50)
  panel=Image.new('RGB',(1536,1000),(12,20,30));draw=ImageDraw.Draw(panel)
  for j,s in enumerate([q+5,q+10,q+20]):
   fr,sc,sk,rgb,exclude,ground,_=rp.source(s);uv,z=p.project(world,sc,sk);inview=eligible&(z>.2)&np.isfinite(uv).all(1)&(uv[:,0]>=0)&(uv[:,0]<1023)&(uv[:,1]>=0)&(uv[:,1]<575)
   ij=np.rint(np.nan_to_num(uv,nan=-1,posinf=-1,neginf=-1)).astype(int);ids=np.flatnonzero(inview);ex=np.zeros(len(x),bool);support=ex.copy();ex[ids]=exclude[ij[ids,1],ij[ids,0]];support[ids]=ground[ij[ids,1],ij[ids,0]]
   stages=dict(domain=len(x),invalid_plane_ray=int((~eligible).sum()),eligible_plane_ray=int(eligible.sum()),outside_source_view=int((eligible&~inview).sum()),in_source_view=int(inview.sum()),actor_envelope=int((inview&ex).sum()),no_actor_but_no_ground=int((inview&~ex&~support).sum()),accepted=int((inview&support).sum()))
   assert stages['domain']==sum(stages[n] for n in ['invalid_plane_ray','outside_source_view','actor_envelope','no_actor_but_no_ground','accepted'])
   actors=[]
   for b in fr['all_boxes']:
    m=hull_mask(b,sc,sk,p.HW,pad=3,ground_extend=.5);n=int(m[ij[ids,1],ij[ids,0]].sum())
    if n:actors.append(dict(actor=b['actor_id'],category=b['category'],projected_query_pixels=n))
   rows.append(dict(query=q,source=s,counts=stages,actors=actors))
   # 原图与投影落点分开保存，叠色只用于定位，不能当补景输出。
   overlay=rgb.copy();color=np.tile(np.array([0,210,220],np.uint8),(len(ids),1));color[ex[ids]]=[255,65,65];color[support[ids]]=[40,255,60];overlay[ij[ids,1],ij[ids,0]]=color
   qview=im.copy();qview[y,x]=[110,20,150];qview[y[eligible],x[eligible]]=[0,210,220]
   for row,(name,img) in enumerate([('query purple=invalid cyan=valid ray',qview),('source original',rgb),('source red=actor cyan=no ground green=pass',overlay)]):
    tile=Image.fromarray(img).resize((512,288));panel.paste(tile,(j*512,row*326+26));draw.text((j*512+5,row*326+5),f'q{q} -> s{s}: {name}',fill='white')
   Image.fromarray(overlay).save(ROOT/f'q{q:03}_source{s:03}_overlay.png');Image.fromarray(rgb).save(ROOT/f'q{q:03}_source{s:03}_original.png')
  panel.save(ROOT/f'q{q:03}_contact.jpg',quality=97)
 dump(ROOT/'state.json',dict(state='complete_pending_visual_review',seconds=time.time()-start,rows=rows,human_verdict=None,background_input_dir=None));print(rows,flush=True)
if __name__=='__main__':main()
