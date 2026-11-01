import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import pathlib,json,collections,math
import numpy as np
from PIL import Image,ImageDraw,ImageFont
H=pathlib.Path('/root/autodl-tmp/codex_tmp/v43_case15_isolation_20261101_r1');O=H/'mask_proposals_r3_inherited';O.mkdir(exist_ok=True);rows=json.loads((H/'mask_probe_input_r2.json').read_text())
P=pathlib.Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027/motion_scene_inputs_v1');result={'mode':'annotation-assisted overlap proposals from already predicted DEVA IDs; not GT, not formal target mask admission','ranking':'inherited .65*IoU+.35*containment from persistent_world_20261027/code/prepare_native_scene.py:50; proposal ranking only, no admission threshold','new_inference':0,'rows':[]};font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',11)
for ctx in dict.fromkeys(r['context_id'] for r in rows):
 cases=[r for r in rows if r['context_id']==ctx];root=P/ctx;prep=json.loads((root/'dynamic_multi_actor_preparation.json').read_text());fids=[f['frame_id'] for f in prep['actors'][0]['frames']];out={r['case']:{'case':r['case'],'track':r['track'],'source':str(root/'mask/mask_0/Annotations'),'frames':[],'human_verdict':None,'mask_admitted':False} for r in cases};sheets={r['case']:Image.new('RGB',(7*224,7*178),'#fff') for r in cases}
 for i,fid in enumerate(fids):
  labels=np.asarray(Image.open(root/'mask/mask_0/Annotations'/(fid+'.png')));assert labels.ndim==2
  counts=np.bincount(labels.reshape(-1).astype(np.int32));rgb=Image.open(root/'image/image_0'/(fid+'.png')).convert('RGB')
  for r in cases:
   row=r['frames'][i];x0,y0,x1,y1=[int(x) for x in row['bbox']];x0=max(0,x0);x1=min(labels.shape[1]-1,x1);y0=max(0,y0);y1=min(labels.shape[0]-1,y1);inside=np.bincount(labels[y0:y1+1,x0:x1+1].reshape(-1).astype(np.int32),minlength=len(counts));ids=np.flatnonzero(inside);candidates=[]
   for ident in ids:
    if ident==0:continue
    a=int(inside[ident]);total=int(counts[ident]);rect=(x1-x0+1)*(y1-y0+1);iou=a/(total+rect-a);candidates.append({'instance_id':int(ident),'pixels':total,'inside_projected_box_pixels':a,'inside_fraction':a/total,'box_IoU':iou,'inherited_proposal_score':.65*iou+.35*a/total})
   candidates.sort(key=lambda v:v['inherited_proposal_score'],reverse=True);best=candidates[0] if candidates else None;rec={'frame_id':fid,'raw':row['raw'],'projected_box':[x0,y0,x1,y1],'top3':candidates[:3],'automatic_identity_accepted':False};out[r['case']]['frames'].append(rec)
   im=rgb.copy()
   if best:
    mask=labels==best['instance_id'];tint=np.asarray(im).copy();tint[mask]=(tint[mask]*.55+np.array([255,145,20])*.45).astype(np.uint8);im=Image.fromarray(tint)
    if i in [0,24,48]:Image.fromarray((mask*255).astype(np.uint8)).save(O/(r['case']+'_raw'+str(row['raw'])+'_proposal.png'))
   ImageDraw.Draw(im).rectangle([x0,y0,x1,y1],outline='#30b4ff',width=3);cx=(x0+x1)/2;cy=(y0+y1)/2;cw=max(x1-x0+70,160);ch=max(y1-y0+70,cw*.67);cw=max(cw,ch/.67);im=im.crop((int(cx-cw/2),int(cy-ch/2),int(cx+cw/2),int(cy+ch/2))).resize((224,150));sheet=sheets[r['case']];px=(i%7)*224;py=(i//7)*178;sheet.paste(im,(px,py));draw=ImageDraw.Draw(sheet);draw.text((px+3,py+152),f"raw{row['raw']} ID{best['instance_id'] if best else '-'} IoU{best['box_IoU'] if best else 0:.2f}",fill='black',font=font)
  print(ctx,fid,flush=True) if i%24==0 else None
 for key,rec in out.items():
  best=[f['top3'][0]['instance_id'] for f in rec['frames'] if f['top3']];rec['top1_instance_ID_frame_counts']=dict(collections.Counter(best));rec['frames_with_any_overlap_proposal']=len(best);rec['status']='visual identity and ambiguity review pending; proposed masks are not admitted';sheets[key].save(O/(key+'_49_proposals.jpg'),quality=88);result['rows'].append(rec)
(O/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print('MASK_PROPOSALS_COMPLETE',len(result['rows']),flush=True)
