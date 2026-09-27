"""r10：目标删除后的GT盒射线候选；包络只说明支持，不当可见mesh真值。"""
import sys
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from hybrid_common import *
from repair_common import camera
from video_review import scene_frame
from PIL import ImageDraw
R2=ROOT;ROOT=ROOT.parent/'r10'
def intersections(c2w,k,box):
 yy,xx=np.mgrid[:H,:W];dc=np.stack([xx,yy,np.ones_like(xx)],-1)@np.linalg.inv(k).T
 pose=np.array(box['pose']);rot=pose[:3,:3];origin=(c2w[:3,3]-pose[:3,3])@rot;direction=dc@c2w[:3,:3].T@rot
 half=np.array(box['size_lwh'])/2
 # 相机z参数化；平行且在slab外的射线必须拒绝。
 parallel=np.abs(direction)<1e-10;outside=np.any(parallel&(np.abs(origin)>half),axis=-1)
 div=np.where(parallel,1.,direction);a=(-half-origin)/div;b=(half-origin)/div
 lo=np.where(parallel,-np.inf,np.minimum(a,b)).max(-1);hi=np.where(parallel,np.inf,np.maximum(a,b)).min(-1)
 hit=(hi>=np.maximum(lo,.1))&~outside
 return np.where(hit,np.maximum(lo,.1),np.inf)
# 非同实现镜像的解析平面/盒强控制：中心射线前面z4，离轴射线不命中。
test=dict(pose=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,5],[0,0,0,1]]),size_lwh=[2,2,2]);z=intersections(np.eye(4),np.array([[100,0,W//2],[0,100,H//2],[0,0,1]]),test);assert z[H//2,W//2]==4 and not np.isfinite(z[0,0])
summary=[]
for s in read(R2/'registration.json')['scenes']:
 name=s['name'];out=ROOT/name;frames=read(OLD/name/'frames.json');rows=[];sheet=Image.new('RGB',(960*3,536*4))
 for q,i in enumerate([0,7,15,29]):
  f=s['source_frames'][i];fr=frames[str(f)];c,k=camera(fr,s['camera'],HW);orig=rgb(R2/name/'rgb'/f'{i:05}.png');core=mask(OLD/name/'sam'/f'core_{i:05}.png');depth=np.full(HW,np.inf);ids=np.full(HW,-1,np.int32);targetdepth=None;boxes={}
  for b in fr['all_boxes']:
   if not b['category'].startswith('vehicle.'):continue
   rect=project_bbox(b['pose'],b['size_lwh'],c,k,HW)
   if rect is None:continue
   zz=intersections(c,k,b);aid=int(b['actor_id']);boxes[aid]=b
   if b['actor_id']==s['actor']:targetdepth=zz;continue
   hit=zz<depth;depth[hit]=zz[hit];ids[hit]=aid
  assert targetdepth is not None
  supported=core&np.isfinite(depth)&(depth>targetdepth+.1);unknown=core&~supported
  stats=[];maskcolor=orig.copy();maskcolor[unknown]=(.25*orig[unknown]+.75*np.array([240,110,30])).astype('uint8')
  for aid in sorted(set(ids[supported])):
   area=supported&(ids==aid);color=np.array([(aid*73+43)%180+60,(aid*113+9)%180+60,(aid*37+97)%180+60]);maskcolor[area]=(.15*orig[area]+.85*color).astype('uint8');stats.append(dict(actor=int(aid),pixels=int(area.sum()),median_target_box_z=float(np.median(targetdepth[area])),median_other_box_z=float(np.median(depth[area]))))
  overlay=Image.fromarray(orig);d=ImageDraw.Draw(overlay)
  top=sorted(stats,key=lambda a:-a['pixels'])[:3]
  for r in top:
   b=boxes[r['actor']];rect=project_bbox(b['pose'],b['size_lwh'],c,k,HW);d.rectangle(rect,outline='cyan',width=2);d.text((rect[0],rect[1]-12),f'neighbor {r["actor"]}',fill='cyan')
  target=boxes[int(s['actor'])];rect=project_bbox(target['pose'],target['size_lwh'],c,k,HW);d.rectangle(rect,outline='yellow',width=2);d.text((rect[0],rect[3]+2),f'DELETE {s["actor"]}',fill='yellow')
  candidate=Image.fromarray(rgb(R2.parent/'r3'/name/'final'/f'{i:05}.png'));d=ImageDraw.Draw(candidate)
  for r in top:
   b=boxes[r['actor']];rect=project_bbox(b['pose'],b['size_lwh'],c,k,HW);d.rectangle(rect,outline='cyan',width=2);d.text((rect[0],rect[1]-12),str(r['actor']),fill='cyan')
  for j,im in enumerate([overlay,Image.fromarray(maskcolor),candidate]):
   ImageDraw.Draw(im).text((8,8),f'{name} f{f} '+['source: target yellow / behind cyan','GT-box support after target removal / orange unknown','r3 candidate + expected neighbor boxes'][j],fill='yellow');sheet.paste(im,(j*960,q*536))
  rows.append(dict(frame=f,target_mask_pixels=int(core.sum()),behind_vehicle_box_support=int(supported.sum()),fraction=float(supported.sum()/core.sum()),no_box_support=int(unknown.sum()),actors=stats))
 sheet.resize((1920,1429)).save(out/'occluded_actor_audit.jpg',quality=96);dump(out/'occluded_actor_audit.json',rows);summary.append(dict(scene=name,frames=rows))
 # 查看原日志更大视角变化，给fence与邻车提供参考；没有生成或捏造GT。
 data=Path(s['data']);inst=read(data/'instances/instances_info.json');refs=[0,25,50,65,94,130,170];refsheet=Image.new('RGB',(960*2,536*4))
 for q,f in enumerate(refs):
  p=data/'images'/f'{f:03}_{s["camera"]}.jpg'
  if not p.exists():continue
  fr=scene_frame(s['spec'],f,inst);im=Image.fromarray(rgb(p));d=ImageDraw.Draw(im);c,k=camera(fr,s['camera'],HW)
  for b in fr['all_boxes']:
   if b['actor_id'] not in [s['actor'],'52','34']:continue
   rect=project_bbox(b['pose'],b['size_lwh'],c,k,HW)
   if rect:d.rectangle(rect,outline='yellow' if b['actor_id']==s['actor'] else 'cyan',width=2);d.text((rect[0],rect[1]-12),str(b['actor_id']),fill='yellow')
  d.text((8,8),f'{name} factual f{f}',fill='yellow');refsheet.paste(im,((q%2)*960,(q//2)*536))
 refsheet.resize((1280,1429)).save(out/'factual_context.jpg',quality=96)
dump(ROOT/'occlusion_summary.json',dict(scenes=summary,scope='GT box-volume ray candidates only; not proof of exact visible neighbor pixels or correct generated identity',guard_implication='visible-mask IoU can miss legitimate newly revealed area; require actor-specific support and appearance check',human_verdict=None));print('OCCLUSION AUDIT COMPLETE')
