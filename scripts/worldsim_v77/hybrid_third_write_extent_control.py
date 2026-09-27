"""r30：冻结r28，只拆写回范围、边缘渐变和可见邻车包络保护。"""
import sys,time,datetime
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from hybrid_road_plane_control import BASE,get,HW
from repair_common import dump,hull_mask
import numpy as np,cv2
from PIL import Image,ImageDraw
cv2.setNumThreads(4);ROOT=BASE/'r30';SOURCE=BASE/'r28'

def composite(original,native,alpha):
 return np.rint(native.astype('float32')*alpha[...,None]+original.astype('float32')*(1-alpha[...,None])).clip(0,255).astype('uint8')

def main():
 assert not ROOT.exists();ROOT.mkdir();beg=time.time()
 dump(ROOT/'registration.json',dict(task_id='WS-V77-HYBRID-BG-20260927',run_id='r30',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scene='official_000',target='12',camera=0,frames=list(range(10)),
  question='Does the car-shaped writeback boundary, rather than frozen native generation, cause the third-scene bright seam?',
  inputs='Exact r28 native, original RGB, precise write mask and model rectangle. GT boxes only supply approximate visible non-target preservation envelope; no image regeneration or extra hidden truth.',
  arms=dict(rect_hard='Change only final write extent from precise instance to whole existing model rectangle.',rect_feather='Same rectangle, fixed8px smoothstep blend inside outer boundary; old precise target mask stays full native weight.',rect_feather_keep='Same feather, zero weight on other GT actor envelopes outside original target write mask. Target-occluded part is not protected from original source.'),
  fixed=dict(feather_px=8,cpu_threads=4,size=[1024,576],gpu_forwards=0),
  limits='GT envelope minus delete mask approximates visible preserved actors, not exact SAM or depth visibility. Rectangle can modify static barriers/road beyond target; inspect original/native/composites and preserve all controls. No all-neighbor preservation claim from box contract.',
  stop_rule='One fixed extent/feather/protect decomposition. Reject structure or neighbor damage; no parameter grid, seed change or new generator. Retain r28 if all controls worsen.',
  failure_ledger_refs=['V77-F02'],previous_goal_turn='progress: r28 third-scene improvement and r29 counterexample committed at7aa8b52d',human_verdict=None,background_input_dir=None))
 names=['rect_hard','rect_feather','rect_feather_keep'];folders=names+['alpha','protect','scope']
 for name in folders:(ROOT/name).mkdir()
 contacts=[Image.new('RGB',(2000,1250),(12,20,30)) for _ in range(2)];rows=[]
 for f in range(10):
  original=np.array(Image.open(SOURCE/'input'/f'{f:05}.png'));native=np.array(Image.open(SOURCE/'native'/f'{f:05}.png'));old=np.array(Image.open(SOURCE/'final'/f'{f:05}.png'));m=np.array(Image.open(SOURCE/'model_mask'/f'{f:05}.png'))>0;write=np.array(Image.open(SOURCE/'write_mask'/f'{f:05}.png'))>0
  fr,c,k,actual,_=get(f);assert np.array_equal(original,actual);protect=np.zeros(HW,bool)
  for b in fr['all_boxes']:
   if b['actor_id']!='12':protect|=hull_mask(b,c,k,HW)
  protect&=m&~write
  dist=cv2.distanceTransform(m.astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE);t=np.clip(dist/8,0,1);alpha=t*t*(3-2*t);alpha[write]=1;kept=alpha.copy();kept[protect]=0
  alphas=dict(rect_hard=m.astype('float32'),rect_feather=alpha,rect_feather_keep=kept);images={};checks={}
  for name,a in alphas.items():
   result=composite(original,native,a);assert np.array_equal(result[~m],original[~m]);assert np.array_equal(result[write],native[write]);images[name]=result;Image.fromarray(result).save(ROOT/name/f'{f:05}.png');checks[name]=dict(outside_model_changed=0,target_write_exact=True,changed_outside_old_write=int(np.any(result!=original,-1)[m&~write].sum()),changed_protect_pixels=int(np.any(result!=original,-1)[protect].sum()))
  assert checks['rect_feather_keep']['changed_protect_pixels']==0
  Image.fromarray(np.rint(kept*255).astype('uint8')).save(ROOT/'alpha'/f'{f:05}.png');Image.fromarray(np.uint8(protect)*255).save(ROOT/'protect'/f'{f:05}.png')
  scope=original.copy();scope[m]=(.5*scope[m]+.5*np.array([0,180,160])).astype('uint8');scope[write]=(.3*original[write]+.7*np.array([235,200,0])).astype('uint8');scope[protect]=[80,120,255];Image.fromarray(scope).save(ROOT/'scope'/f'{f:05}.png')
  y,x=np.where(write);left=int(np.clip((x.min()+x.max())/2-160,0,704));top=int(np.clip((y.min()+y.max())/2-90,0,396));box=(left,top,left+320,top+180)
  for j,(name,arr) in enumerate([('original',original),('r28 precise write',old)]+[(n,images[n]) for n in names]):
   tile=Image.new('RGB',(400,250),(12,20,30));tile.paste(Image.fromarray(arr).crop(box).resize((400,225)),(0,25));ImageDraw.Draw(tile).text((5,5),f'f{f} {name}',fill='white');contacts[f//5].paste(tile,(j*400,(f%5)*250))
  rows.append(dict(frame=f,old_write_pixels=int(write.sum()),model_pixels=int(m.sum()),protected_pixels=int(protect.sum()),checks=checks))
 for i,im in enumerate(contacts):im.save(ROOT/f'all10_part{i+1}.jpg',quality=97)
 dump(ROOT/'state.json',dict(state='complete_pending_visual_review',seconds=time.time()-beg,frames=10,arms=names,rows=rows,human_verdict=None,background_input_dir=None));print('R30_COMPLETE',time.time()-beg,flush=True)
if __name__=='__main__':main()
