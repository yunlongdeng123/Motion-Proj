"""全三十帧重新渲染/重载，三关键帧呈现三秒过程，不把单帧QA当视频认证。"""
from pathlib import Path
import sys,copy
from collections import Counter
sys.path.insert(0,str(Path(__file__).parent.parent/'iteration9'))
from long_factory import geometry,protections,exact,old,R8,T,ROOT,read,dump
from temporal_metrics import process
from render_pairs import label,save,encode,font
from data_contract import assert_pair_pixels,masked_condition_from_x
import numpy as np,cv2
from PIL import Image,ImageDraw

O=T/'r16'

def select():
 dest=O/'selected.json'
 if dest.exists():return read(dest)['selected']
 pool=[c for path in sorted((O/'planned').glob('*.json')) for c in read(path)['candidates']]
 unique={(p['source_id'],p['asset'],p['offset_longitudinal_m'],p['offset_lateral_m'],p['relative_speed_mps'],p['trajectory_policy']):p for p in pool}
 selected=copy.deepcopy(list(unique.values()))
 for i,p in enumerate(selected):p.update(case_id=f'L{i+1:03}',edge_mode=['near_hard','feather_05','feather_10'][i%3],mask_dilation_px=[2,3,4][i%3],candidate_role='independent_QA_pending',speed_signed_mps=p['relative_speed_mps'])
 dump(dest,{'selected':selected,'summary':{'cases':len(selected),'process_counts':dict(Counter(p['process_family'] for p in selected)),'type_counts':dict(Counter(p['type'] for p in selected)),'selection_uses_model_outputs':False},'speed_signed_mps_alias_is_relative_delta_not_world_speed':True})
 return selected

def main():
 cv2.setNumThreads(1);geo=geometry();out=O/'data_review';(out/'contacts').mkdir(parents=True,exist_ok=True);rows=[];checks=[]
 for p in select():
  cid=p['case_id'];sid=p['source_id'];src=geo.sources[sid];geo.prepare(sid);pm=protections(geo,sid);mesh=dict(np.load(R8/'assets'/(p['asset']+'.npz')))
  alphas=[old.silhouette(mesh['vertices'],mesh['faces'],f['actor'],sf) for f,sf in zip(p['frames'],src['frames'])];q,why=exact(geo,p,alphas,pm);assert q is not None,why
  dest=O/'synthetic'/cid;preview=out/'assets'/cid
  dest.mkdir(parents=True,exist_ok=True);preview.mkdir(parents=True,exist_ok=True)
  for role in ['Y','X','alpha','influence','model_hole','protected','condition_preview']:(dest/role).mkdir(exist_ok=True)
  panels=[];metrics=[];hs=[]
  for i,(f,aa) in enumerate(zip(src['frames'],alphas)):
   with Image.open(ROOT/'rgb'/f['filename']) as im:y=np.asarray(im.convert('RGB').resize((1024,576),Image.Resampling.LANCZOS)).copy()
   alpha=aa.astype('float32');sigma={'near_hard':0,'feather_05':.5,'feather_10':1}[p['edge_mode']]
   if sigma:alpha=cv2.GaussianBlur(alpha,(3,3),sigma)
   influence=alpha>1/65535;x=np.rint(y*(1-alpha[...,None])+np.array([96,110,127])*alpha[...,None]).clip(0,255).astype('uint8');radius=p['mask_dilation_px'];h=cv2.dilate(influence.astype('uint8'),np.ones((2*radius+1,2*radius+1),'uint8'))>0;hs.append(h)
   assert_pair_pixels(y,x,influence,h);assert not h[512:].any();cc=masked_condition_from_x(x[None],h[None])[0];assert np.array_equal(cc,masked_condition_from_x(y[None],h[None])[0]);cp=np.rint((cc+1)*127.5).clip(0,255).astype('uint8')
   bm={t:m[i] for t,m in pm.items()};ann=label(x,aa,{t:bm[t] for t in q['protected_instances']});ratios={t:float((h&m).sum()/m.sum()) for t,m in bm.items()};assert max(ratios.values(),default=0)<=.85
   for role,a in [('Y',y),('X',x),('alpha',np.rint(alpha*255).astype('uint8')),('influence',influence.astype('uint8')*255),('model_hole',h.astype('uint8')*255),('condition_preview',cp)]:save(dest/role/f'{i:03}.png',a)
   for t,m in bm.items():save(dest/'protected'/f'{i:03}_{t}.png',m.astype('uint8')*255)
   # 真实磁盘产物重载；不是仅相信内存中的规划。
   xx=np.asarray(Image.open(dest/'X'/f'{i:03}.png'));yy=np.asarray(Image.open(dest/'Y'/f'{i:03}.png'));hh=np.asarray(Image.open(dest/'model_hole'/f'{i:03}.png'))>0
   assert np.array_equal(yy,y);assert not np.any((xx!=yy).any(-1)&~hh);xx2=xx.copy();yy2=yy.copy();xx2[hh]=0;yy2[hh]=0;assert np.array_equal(xx2,yy2)
   panel={'gt':y,'input':x,'labels':ann,'condition':cp};panels.append(panel)
   for role,a in panel.items():Image.fromarray(a).save(preview/f'{i:03}_{role}.jpg',quality=94)
   metrics.append({'frame':i,'hole_fraction':float(h.mean()),'hole_overlap':ratios,'A_in_H':True,'masked_X_equals_masked_Y':True,'actual_disk_RGB_leak':0,'GT_exact_real_redecode':True})
  active={t:pm[t] for t in q['protected_instances']};ba={t:[next(a for a in f['actors'] if a['instance_token']==t) for f in src['frames']] for t in active};proc=process(src['frames'],[f['actor'] for f in p['frames']],hs,active,ba)
  sheet=Image.new('RGB',(1600,1150),(14,21,31));draw=ImageDraw.Draw(sheet);draw.text((12,8),f'{cid} | {src["scene"]} | {p["process_family"]} | relative speed delta {p["relative_speed_mps"]}m/s',font=font(22),fill='white')
  for n,i in enumerate([0,15,29]):
   draw.text((12,45+n*350),f'f{i} | time {(src["frames"][i]["timestamp"]-src["frames"][0]["timestamp"])/1e6:.2f}s',font=font(18),fill='white')
   for j,(role,a) in enumerate(panels[i].items()):
    draw.text((j*400+10,70+n*350),role,font=font(18),fill='white');sheet.paste(Image.fromarray(a).resize((400,225)),(j*400,100+n*350))
  sheet.save(out/'contacts'/f'{cid}_process.jpg',quality=95)
  for role in ['gt','input','labels','condition']:encode(preview,role)
  row=p|{'folder':str(dest),'camera':src['camera'],'review_frame':15,'review_contact':f'contacts/{cid}_process.jpg','frame_count':30,'temporal_process':proc,'pixel_metrics':metrics,'videos':{r:f'assets/{cid}/{r}.mp4' for r in ['gt','input','labels','condition']},'human_verdict':None};dump(dest/'pair_manifest.json',row);rows.append(row)
  checks.append({'case_id':cid,'technical_pass':True,'frames_checked':30,'RGB_leak':0,'GT_exact':True,'ego_hole_pixels':0,'min_GT_clearance_m':p['min_GT_clearance_m'],'depth_revalidated':True,'process':proc,'independent_QA':'pending','human_verdict':None})
  dump(out/'synthetic_manifest.json',{'clips':rows});dump(O/'technical_checks.json',{'stage':'running','cases':checks});print('RENDER',cid,proc['sweep_over_any_B'],flush=True)
 dump(O/'technical_checks.json',{'stage':'complete','cases':checks,'checked_frames':len(checks)*30,'all_pass':True});print('DONE',len(rows))
if __name__=='__main__':main()
