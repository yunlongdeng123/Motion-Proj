"""r39：r38第二来源未通过后，只审查已通过几何检查的f50后视来源。"""
import sys,time,datetime
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp/work');sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
import hybrid_rear_road_evidence_control as rear
from repair_common import dump,read,hull_mask
import numpy as np
from PIL import Image,ImageDraw
ROOT=rear.p.BASE/'r39'

def main():
 assert not ROOT.exists();ROOT.mkdir();start=time.time()
 failed=rear.p.BASE/'r38';assert not (failed/'state.json').exists();record=read(failed/'source075/plane_fit.json');assert record['heldout_q90_m']>=.12
 dump(failed/'state.json',dict(state='stopped_source_geometry_validation',failed_source={'frame':75,'camera':5},heldout_q90_m=record['heldout_q90_m'],maximum_q90_m=.12,accepted_source50_partial_retained=True,no_model_or_background_write=True,human_verdict=None,background_input_dir=None))
 dump(ROOT/'registration.json',dict(task_id='WS-V77-HYBRID-BG-20260927',run_id='r39',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scene='official_000',target='12',queries=[0,10],source={'frame':50,'camera':5},parent='r38 source75 q90=0.1634m fails fixed0.12m check; no threshold relaxation. Source50 had already passed4.2mm median/26.2mm q90.',question='Does the individually valid late rear source supply actual DELETE road RGB?',intervention='Only remove invalid source75; no geometry or photometric modification. Diagnostic single-source coverage, not consensus evidence or a complete DELETE.',roles='Original future RGB/LiDAR/poses/GT as offline source. Query visible RGB evaluation only. Same local-road ROI and mesh projection as r38.',stop_rule='Inspect visible road and actual hole; reject misalignment/illumination, do not inject unreliable RGB into DriveEditor/Omega.',resources=dict(cpu_threads=4,gpu_forwards=0),failure_ledger_refs=['V77-F02'],human_verdict=None,background_input_dir=None))
 rear.ROOT=ROOT;rear.mesh.source=rear.source;rows=[]
 for mode in ['visible','hole']:
  for q in [0,10]:
   fr,c,k,original,_=rear.p.get(q)
   if mode=='visible':domain=np.array(Image.open(rear.p.ROOT/f'f{q:03}/evaluation_mask.png'))>0
   else:
    domain=np.array(Image.open(rear.mesh.OLD/'mask'/f'{q:05}.png'))>0
    for b in fr['all_boxes']:
     if str(b['actor_id'])!='12':domain&=~hull_mask(b,c,k,rear.p.HW,pad=3,ground_extend=.5)
   rgb,mask,depth,mx,my,tri,info=rear.mesh.project_pair(q,50,domain);out=ROOT/mode/f'f{q:03}';out.mkdir(parents=True);view=original.copy();view[domain]=[110,20,140];view[mask]=rgb[mask]
   for name,im in [('original',original),('rgb',rgb),('mask',np.uint8(mask)*255),('domain',np.uint8(domain)*255),('diagnostic',view)]:Image.fromarray(im).save(out/f'{name}.png')
   np.savez_compressed(out/'source050_map.npz',mx=mx,my=my,depth=depth,triangle=tri);row=dict(mode=mode,query=q,denominator=int(domain.sum()),covered=int(mask.sum()),coverage=float(mask.sum()/max(1,domain.sum())))
   if mode=='visible':
    err=np.abs(rgb.astype('float32')-original).mean(-1);oldrgb=np.array(Image.open(rear.p.ROOT/f'f{q:03}/nearest_rgb.png'));oldmask=np.array(Image.open(rear.p.ROOT/f'f{q:03}/nearest_mask.png'))>0;shared=mask&oldmask;olderr=np.abs(oldrgb.astype('float32')-original).mean(-1);row.update(mean_mae=float(err[mask].mean()) if mask.any() else None,shared_pixels=int(shared.sum()),r25_mae_shared=float(olderr[shared].mean()) if shared.any() else None,rear_mae_shared=float(err[shared].mean()) if shared.any() else None)
   rows.append(row);dump(out/'metrics.json',row)
 sheet=Image.new('RGB',(1200,4*365),(12,20,30));dr=ImageDraw.Draw(sheet)
 for j,row in enumerate(rows):
  d=ROOT/row['mode']/f'f{row["query"]:03}';m=np.array(Image.open(d/'domain.png'))>0
  if row['mode']=='hole':y,x=np.where(m);left=int(np.clip((x.min()+x.max())/2-192,0,640));top=int(np.clip((y.min()+y.max())/2-108,0,360));box=(left,top,left+384,top+216)
  else:box=(0,0,1024,576)
  for col,key in enumerate(['original','diagnostic']):sheet.paste(Image.open(d/f'{key}.png').crop(box).resize((600,337)),(col*600,j*365+26));dr.text((col*600+5,j*365+5),f'{row["mode"]} f{row["query"]}: {key}',fill='white')
 sheet.save(ROOT/'contact.jpg',quality=97);dump(ROOT/'state.json',dict(state='complete_pending_visual_review',seconds=time.time()-start,queries=rows,source=rear.CACHE[50][-1],human_verdict=None,background_input_dir=None));print(rows,flush=True)
if __name__=='__main__':main()
