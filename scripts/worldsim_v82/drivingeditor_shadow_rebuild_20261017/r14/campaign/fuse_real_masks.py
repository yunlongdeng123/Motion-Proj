"""复用49帧真实原生SAM3预测；逐区域对称证据融合，不重跑模型。"""
from pathlib import Path
import sys,json,time
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import binary_erosion
from PIL import Image,ImageDraw
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3');D=R/'ownership_rebuild_r14';S=R/'ownership_controls_r12_sam3';O=D/'shared_masks_r14'
sys.path.insert(0,str(R/'implementation_r14/shadow_rebuild_v3'))
from shadow_rebuild.mask_fusion import fuse_regions
m=json.loads((R/'data_r2/manifest.json').read_text());k=len(m['actor_ids']);targets=[5,6,12,18,25,26,37];rawpred=S/'shared_sam3_masks_r1/native_predictions';roadtree=cKDTree(np.load(S/'actual_world_road_support.npz')['xyz']);start=time.monotonic()
def inputs(row):
    actual=np.load(S/'raw_measurements'/(row['frame_id']+'.npz'));pred=np.load(rawpred/(row['frame_id']+'.npz'));old=np.load(R/'data_r2'/row['supervision']);pixel=actual['pixel'].astype(int)
    allowed=actual['BUILD'] if row['split']=='train' else actual['QUERY'];ids=np.flatnonzero(allowed&actual['in_image']);bins=pixel[ids,1]*480+pixel[ids,0];depth=np.full(320*480,np.inf);np.minimum.at(depth,bins,actual['camera_z'][ids]);front=ids[actual['camera_z'][ids]<=depth[bins]+.1]
    worldroad=roadtree.query(actual['world'][front],workers=1)[0]<.06;body=np.zeros((320,480,k),bool);road=np.zeros((320,480),bool);pp=pixel[front];road[pp[worldroad,1],pp[worldroad,0]]=True
    own=actual['candidate_owner'][front];good=(own>=0)&~worldroad
    body[pp[good,1],pp[good,0],own[good]]=True
    hits=np.zeros((k,len(pred['masks'])),float)
    for j,mask in enumerate(pred['masks']):
        take=good&mask[pp[:,1],pp[:,0]]
        hits[:,j]=np.bincount(own[take],minlength=k)
    # Raw box candidates alone do not create any foreground pixel. They only
    # verify a visual proposal against actual non-road visible measurements.
    return actual,pred,old,body,road,hits
votes={}
for row in m['frames']:
    if row['split']!='train':continue
    _,p,_,_,_,h=inputs(row)
    for j,tid in enumerate(p['tracker_ids']):
        if h[:,j].sum()>=3:votes.setdefault(int(tid),np.zeros(k))[:]+=h[:,j]
mapping={tid:int(v.argmax()) for tid,v in votes.items() if v.max()>=6 and v.max()/max(1,v.sum())>=.8}
O.mkdir(exist_ok=False);records=[];source_totals=np.zeros((k,4),np.int64)
for index,row in enumerate(m['frames']):
    actual,pred,old,body,road,hits=inputs(row);sam=np.zeros((320,480,k),bool);sam_known=np.zeros(k,bool);score=hits*hits/np.maximum(1,hits.sum(0,keepdims=True))
    if row['split']=='train':
        for j,tid in enumerate(pred['tracker_ids']):
            if int(tid) in mapping:
                actor=mapping[int(tid)];score[:,j]=-1;score[actor,j]=max(.1,hits[actor,j])
            else:score[:,j]=-1
    if score.shape[1]:
        ii,jj=linear_sum_assignment(-score)
        for a,j in zip(ii,jj):
            if score[a,j]<.1 or pred['scores'][j]<.5:continue
            if row['split']!='train' and hits[a,j]<3:continue
            sam[...,a]=pred['masks'][j];sam_known[a]=True
    result=fuse_regions(old['masks'],old['mask_known'],sam,sam_known,body,road)
    np.savez_compressed(O/(row['frame_id']+'.npz'),**result,SAM_prediction=sam)
    counts=np.stack([(result['source']==i).sum((0,1)) for i in range(4)],-1);source_totals+=counts
    rec=dict(frame_id=row['frame_id'],split=row['split'],unknown_pixels=int(result['uncertainty'].sum()),overlap_valid_pixels=int((result['masks'].sum(-1)>1).sum()),
        actors=[dict(actor=a,old_pixels=int(old['masks'][...,a].sum()),SAM_associated=bool(sam_known[a]),SAM_pixels=int(sam[...,a].sum()),fused_positive=int(result['masks'][...,a].sum()),valid_pixels=int(result['mask_valid'][...,a].sum()),old_only=int(counts[a,1]),SAM_only=int(counts[a,2]),agreement=int(counts[a,3])) for a in targets])
    records.append(rec);(O/(row['frame_id']+'_record.json')).write_text(json.dumps(rec,indent=2))
    # True RGB overlays: orange DEVA, blue SAM3, green fused; red uncertainty.
    if int(row['frame_id'][-3:]) in [101,105,107,113,121,125,133,137]:
        original=Image.open(R/'data_r2'/row['rgb']).convert('RGB');panels=[]
        for title,maskset,color in [('RGB',None,None),('DEVA',old['masks'],(255,170,0)),('SAM3 matched',sam,(50,150,255)),('fused / red unknown',result['masks'],(30,230,90))]:
            im=original.copy()
            if maskset is not None:
                pixels=np.asarray(im).copy()
                for a in targets:
                    mask=maskset[...,a]
                    # Both entry and exit boundaries, sides, holes and image edges.
                    edge=mask&~binary_erosion(mask,structure=np.ones((3,3),bool),border_value=0)
                    pixels[mask]=np.rint(.72*pixels[mask]+.28*np.array(color)).astype(np.uint8)
                    pixels[edge]=color
                im=Image.fromarray(pixels)
            dr=ImageDraw.Draw(im)
            if title.startswith('fused'):
                yy,xx=np.where(result['uncertainty']);take=(xx+yy)%5==0
                for x,y in zip(xx[take],yy[take]):dr.point((int(x),int(y)),fill=(255,60,70))
            dr.rectangle((0,0,260,18),fill=(10,20,30));dr.text((4,2),title,fill='white');panels.append(im)
        canvas=Image.new('RGB',(1920,320));[canvas.paste(im,(j*480,0)) for j,im in enumerate(panels)];canvas.save(O/(row['frame_id']+'_review.png'))
    (D/'input_progress.json').write_text(json.dumps(dict(phase='REGIONAL_MASK_FUSION',frames=index+1,total=49,scene_training_updates=0)))
    print(json.dumps({'frame':index+1,'unknown_pixels':rec['unknown_pixels'],'SAM_positive':[x['SAM_only'] for x in rec['actors']]}),flush=True)
summary=dict(status='REGIONAL_MASK_FUSION_DONE_REQUIRES_REAL_INPUT_REVIEW',frames=records,train_only_tracker_mapping=mapping,source_totals=source_totals.tolist(),source_codes={'0':'not positive / uncertain','1':'verified DEVA only','2':'verified SAM3 only','3':'verified agreement'},
    policy='equal connected-component/actual visible non-road LiDAR evidence; no old-label priority, no whole-frame fallback; disagreement/identity overlap unknown only at affected pixels',
    native_predictions_reused=True,no_NEW_model_inference=True,QUERY_isolated=True,QUERY_not_used_in_train_mapping=True,worldroad_is_geometric_proxy=True,box_is_candidate_not_body_truth=True,scene_training_updates=0,seconds=time.monotonic()-start)
(O/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({x:y for x,y in summary.items() if x!='frames'}),flush=True)
