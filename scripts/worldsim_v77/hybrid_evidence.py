"""原RGB局部LK/RANSAC与缓存Ω点重投影；保留逐像素源位置，禁止生成图当证据。"""
import time
from scipy.spatial import cKDTree
from hybrid_common import *
from repair_common import hull_mask,camera
from video_review import scene_frame
cv2.setNumThreads(4)

def geometry_candidates(s,out,frame,dmask):
 old=OLD/s['name'];index=read(old/'donor_index.json');cam,k=camera(frame,s['camera'],HW)
 zbest=np.full(H*W,np.inf);color=np.zeros((H*W,3),np.uint8);sourceid=np.full(H*W,-1,np.int16);pidbest=np.full(H*W,-1,np.int32);projected=[]
 with np.load(old/'donors.npz') as z:
  for j,rec in enumerate(index):
   pts=z[f'{j}_points'];cp=transform(pts,np.linalg.inv(cam));q=cp@k.T;uv=np.rint(q[:,:2]/np.maximum(q[:,2:],1e-6)).astype(int)
   valid=(cp[:,2]>.5)&(uv[:,0]>=1)&(uv[:,0]<W-1)&(uv[:,1]>=1)&(uv[:,1]<H-1);ids=np.flatnonzero(valid);ii=[];zz=[];pp=[]
   for dx in [-1,0,1]:
    for dy in [-1,0,1]:
     x=uv[ids,0]+dx;y=uv[ids,1]+dy;v=dmask[y,x];ii.append(y[v]*W+x[v]);zz.append(cp[ids[v],2]);pp.append(ids[v])
   ix=np.concatenate(ii);dep=np.concatenate(zz);pids=np.concatenate(pp)
   if len(ix):
    order=np.argsort(dep);_,first=np.unique(ix[order],return_index=True);sel=order[first];ix,dep,pids=ix[sel],dep[sel],pids[sel]
   col=z[f'{j}_rgb'][pids];projected.append((ix,dep,pids,col));better=dep<zbest[ix];use=ix[better];zbest[use]=dep[better];color[use]=col[better];sourceid[use]=j;pidbest[use]=pids[better]
  corroborate=np.full(H*W,-1,np.int16);frames=np.array([r['frame'] for r in index]+[-1000]);basef=frames[sourceid]
  for j,(ix,dep,pids,col) in enumerate(projected):
   good=(np.abs(dep-zbest[ix])<=.25)&(np.max(np.abs(col.astype('int16')-color[ix].astype('int16')),axis=1)<=25)&(np.abs(index[j]['frame']-basef[ix])>=5);corroborate[ix[good]]=j
  accepted=cv2.erode((corroborate>=0).reshape(HW).astype('uint8'),np.ones((3,3),np.uint8))>0
  flat=np.flatnonzero(accepted);uvsource=np.zeros((len(flat),2),np.int16)
  for j in np.unique(sourceid[flat]):
   take=sourceid[flat]==j;uvsource[take]=z[f'{j}_uv'][pidbest[flat[take]]]
 return color.reshape(H,W,3),accepted,dict(target_index=flat,source_id=sourceid[flat],source_point=pidbest[flat],source_uv=uvsource,second_source=corroborate[flat]),index

def main():
 allrows=[];started=time.time()
 for s in read(ROOT/'registration.json')['scenes']:
  out=ROOT/s['name'];data=pathlib.Path(s['data']);inst=read(data/'instances/instances_info.json');cache={};pairrows=[];rows=[]
  if (out/'evidence_stats.json').exists() and len(read(out/'evidence_stats.json'))==30:
   rows=read(out/'evidence_stats.json');allrows.append(dict(scene=s['name'],frames=30,delete_pixels=sum(r['delete_pixels'] for r in rows),observed_delete_pixels=sum(r['observed_delete_pixels'] for r in rows),temporal_delete_pixels=sum(r['temporal_delete_pixels'] for r in rows),geometry_delete_pixels=sum(r['geometry_delete_pixels'] for r in rows)));continue
  for sub in ['observed','residual_delete','residual_generate','evidence','provenance']:(out/sub).mkdir(exist_ok=True)
  def source(f):
   if f not in cache:
    fr=scene_frame(s['spec'],f,inst);im=rgb(data/'images'/f'{f:03}_{s["camera"]}.jpg');cam,k=camera(fr,s['camera'],HW);exclude=np.zeros(HW,bool)
    for b in fr['all_boxes']:exclude|=hull_mask(b,cam,k,HW,pad=3,ground_extend=.6)
    cache[f]=(fr,im,cv2.cvtColor(im,cv2.COLOR_RGB2GRAY),exclude)
   return cache[f]
  validframes=set(inst[s['actor']]['frame_annotations']['frame_idx'])
  for i,f in enumerate(s['source_frames']):
   fr,original,gray,excluded=source(f);delete=mask(out/'delete'/f'{i:05}.png');protect=mask(out/'protect'/f'{i:05}.png');generate=mask(out/'generate'/f'{i:05}.png')
   y,x=np.where(generate);roi=np.zeros(HW,np.uint8);roi[max(0,y.min()-100):min(H,y.max()+101),max(0,x.min()-100):min(W,x.max()+101)]=1
   target_feature=(roi>0)&~excluded&~generate;points=cv2.goodFeaturesToTrack(gray,800,.005,5,mask=target_feature.astype('uint8')*255,blockSize=5)
   candidates=[]
   for delta in [-3,3,-6,6,-9,9,-15,15,-24,24,-36,36]:
    sf=f+delta
    if sf not in validframes:continue
    sfr,sim,sg,se=source(sf);record=dict(target_frame=f,source_frame=sf,method='local_LK_RANSAC_homography',accepted=False)
    if points is None or len(points)<20:pairrows.append({**record,'reason':'few_background_features'});continue
    p1,st,e=cv2.calcOpticalFlowPyrLK(gray,sg,points,None,winSize=(25,25),maxLevel=4,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.01))
    p2,st2,_=cv2.calcOpticalFlowPyrLK(sg,gray,p1,None,winSize=(25,25),maxLevel=4,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.01))
    a=points[:,0];b=p1[:,0];fb=np.linalg.norm(p2[:,0]-a,axis=1);u=np.rint(b).astype(int);inside=(u[:,0]>=2)&(u[:,0]<W-2)&(u[:,1]>=2)&(u[:,1]<H-2)
    ok=st[:,0].astype(bool)&st2[:,0].astype(bool)&(fb<.75)&inside
    ids=np.flatnonzero(ok);ok[ids]&=~se[u[ids,1],u[ids,0]]
    a,b=a[ok],b[ok]
    if len(a)<20:pairrows.append({**record,'reason':'few_consistent_static_tracks','tracks':len(a)});continue
    matrix,inliers=cv2.findHomography(b,a,cv2.RANSAC,1.5)
    if matrix is None:pairrows.append({**record,'reason':'ransac_failed'});continue
    take=inliers[:,0]>0;project=cv2.perspectiveTransform(b[:,None],matrix)[:,0];err=np.linalg.norm(project-a,axis=1)
    record.update(tracks=len(a),inliers=int(take.sum()),fraction=float(take.mean()),median_error=float(np.median(err[take])))
    if take.sum()<20 or take.mean()<.75 or np.median(err[take])>.7:pairrows.append({**record,'reason':'nonplanar_or_bad_local_fit'});continue
    hull=np.zeros(HW,np.uint8);cv2.fillConvexPoly(hull,cv2.convexHull(np.rint(a[take]).astype('int32')),1)
    yy,xx=np.mgrid[:H,:W];coords=np.stack([xx,yy],axis=-1).astype('float32');su=cv2.perspectiveTransform(coords.reshape(-1,1,2),np.linalg.inv(matrix)).reshape(H,W,2)
    ix=np.rint(su[...,0]).astype(int);iy=np.rint(su[...,1]).astype(int);valid=(ix>=2)&(ix<W-2)&(iy>=2)&(iy<H-2);ix=np.clip(ix,0,W-1);iy=np.clip(iy,0,H-1)
    valid &= ~se[iy,ix];warped=sim[iy,ix]
    # 仅在拟合内点覆盖范围插值；不将全局H外推到隐藏区域。
    valid &= hull>0
    anchor=(roi>0)&~excluded&~generate&valid
    photometric=np.abs(warped.astype('int16')-original.astype('int16')).mean(axis=-1)
    photo=float(np.median(photometric[anchor])) if anchor.sum()>200 else 999.
    record.update(anchor_pixels=int(anchor.sum()),median_anchor_rgb_error=photo)
    if photo>12:pairrows.append({**record,'reason':'photometric_or_parallax_mismatch'});continue
    valid &= generate
    # 局部支持必须靠近真实内点，避免大空洞仅被凸包包住。
    py,px=np.where(valid)
    if len(px):dist,_=cKDTree(a[take]).query(np.stack([px,py],1),workers=4);valid[py[dist>48],px[dist>48]]=False
    record.update(accepted=True,candidate_pixels=int(valid.sum()),matrix=matrix.tolist());pairrows.append(record)
    if valid.any():candidates.append((sf,warped,valid,np.stack([ix,iy],-1).astype('int16')))
   accepted=np.zeros(HW,bool);observed_rgb=original.copy();src=np.full(HW,-1,np.int16);uv=np.full((H,W,2),-1,np.int16);second=np.full(HW,-1,np.int16)
   for j,(sf,col,valid,suv) in enumerate(candidates):
    corroborated=np.zeros(HW,bool);secondf=np.full(HW,-1,np.int16)
    for sf2,col2,valid2,_ in candidates[j+1:]:
     if abs(sf2-sf)<5:continue
     agree=valid&valid2&(np.max(np.abs(col.astype('int16')-col2.astype('int16')),axis=-1)<=20);corroborated|=agree;secondf[agree]=sf2
    use=cv2.erode(corroborated.astype('uint8'),np.ones((3,3),np.uint8))>0;use &= ~accepted
    accepted|=use;observed_rgb[use]=col[use];src[use]=sf;uv[use]=suv[use];second[use]=secondf[use]
   temporal_count=int((accepted&delete).sum())
   grgb,gm,gprov,gindex=geometry_candidates(s,out,fr,generate)
   use=gm&~accepted;observed_rgb[use]=grgb[use];accepted|=gm
   masks=mask_contract(delete,protect,accepted,16)
   for k in ['observed','residual_delete','residual_generate']:write_mask(out/k/f'{i:05}.png',masks[k])
   Image.fromarray(observed_rgb).save(out/'evidence'/f'{i:05}.png')
   np.savez_compressed(out/'provenance'/f'{i:05}.npz',temporal_source=src,temporal_uv=uv,temporal_second=second,geometry_used=use,**{'geometry_'+k:v for k,v in gprov.items()})
   dump(out/'geometry_source_index.json',gindex)
   row=dict(frame=f,delete_pixels=int(delete.sum()),generate_pixels=int(generate.sum()),temporal_delete_pixels=temporal_count,geometry_delete_pixels=int((use&delete).sum()),observed_delete_pixels=int((accepted&delete).sum()),observed_generate_pixels=int(accepted.sum()),residual_delete_pixels=int(masks['residual_delete'].sum()),valid_2d_pairs=len(candidates))
   rows.append(row);dump(out/'evidence_stats.json',rows);dump(out/'temporal_pair_audit.json',pairrows);print(s['name'],i,row,flush=True)
  allrows.append(dict(scene=s['name'],frames=30,delete_pixels=sum(r['delete_pixels'] for r in rows),observed_delete_pixels=sum(r['observed_delete_pixels'] for r in rows),temporal_delete_pixels=sum(r['temporal_delete_pixels'] for r in rows),geometry_delete_pixels=sum(r['geometry_delete_pixels'] for r in rows)))
 dump(ROOT/'evidence_summary.json',dict(scenes=allrows,elapsed_seconds=time.time()-started,human_verdict=None))
if __name__=='__main__':main()
