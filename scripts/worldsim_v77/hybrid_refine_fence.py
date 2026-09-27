"""有界POC修正：图像审阅选定的镂空栏杆mask，LK/RANSAC跟踪其平面。
这14个提示为助手图像提示，不宣称完全自动分割；旧自动mask完整保留。
"""
import shutil
from hybrid_common import *
out=ROOT/'scene_0255';backup=out/'mask_attempt_02_degenerate_fence';backup.mkdir(exist_ok=False)
for sub in ['core','delete','protect','generate','static','dynamic']:
 shutil.copytree(out/sub,backup/sub)
shutil.copy2(out/'mask_review.jpg',backup/'mask_review.jpg');shutil.copy2(out/'mask_stats.json',backup/'mask_stats.json')
first=rgb(out/'rgb/00000.png');g0=cv2.cvtColor(first,cv2.COLOR_RGB2GRAY);m0=mask(out/'fence_probe/0.png')
p0=cv2.goodFeaturesToTrack(g0,300,.002,3,mask=m0.astype('uint8')*255,blockSize=3)
assert len(p0)>=12
tracked=p0.copy();valid=np.ones(len(p0),bool);previous=g0;rows=[]
for i in range(30):
 im=rgb(out/'rgb'/f'{i:05}.png');gray=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
 if i:
  nxt,st,_=cv2.calcOpticalFlowPyrLK(previous,gray,tracked,None,winSize=(21,21),maxLevel=4)
  back,st2,_=cv2.calcOpticalFlowPyrLK(gray,previous,nxt,None,winSize=(21,21),maxLevel=4)
  valid &= st[:,0].astype(bool)&st2[:,0].astype(bool)&(np.linalg.norm(back[:,0]-tracked[:,0],axis=1)<.8)
  tracked=nxt
 if valid.sum()<8:raise RuntimeError(f'fence tracking insufficient f{i}')
 affine,inl=cv2.estimateAffinePartial2D(p0[valid,0],tracked[valid,0],method=cv2.RANSAC,ransacReprojThreshold=2.)
 matrix=np.vstack([affine,[0.,0.,1.]]) if affine is not None else None
 assert matrix is not None and int(inl.sum())>=8
 projected=cv2.perspectiveTransform(p0[valid],matrix)[:,0];err=np.linalg.norm(projected-tracked[valid,0],axis=1)
 assert np.median(err[inl[:,0]>0])<1.5
 assert .5 < np.linalg.det(matrix[:2,:2]) < 2.5
 fence=cv2.warpPerspective(m0.astype('uint8'),matrix,(W,H),flags=cv2.INTER_NEAREST)>0
 static=mask(out/'mask_attempt_01_solid_fence/static'/f'{i:05}.png')
 # 只在原栏杆框随H运动的范围中替换自动实心栏杆；其他设施保护保留。
 box=np.zeros(HW,np.uint8);box[299:391,347:578]=1;region=cv2.warpPerspective(box,matrix,(W,H),flags=cv2.INTER_NEAREST)>0
 static=(static&~region)|fence
 dynamic=mask(out/'dynamic'/f'{i:05}.png');protect=static|dynamic
 oldcore=mask(OLD/'scene_0255'/'sam'/f'core_{i:05}.png');core=oldcore&~protect
 delete=(cv2.dilate(core.astype('uint8'),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))>0)&~protect
 mm=mask_contract(delete,protect,np.zeros(HW,bool),16)
 for k in ['delete','protect','generate']:write_mask(out/k/f'{i:05}.png',mm[k])
 write_mask(out/'core'/f'{i:05}.png',core);write_mask(out/'static'/f'{i:05}.png',static)
 rows.append(dict(frame=65+i,old_core=int(oldcore.sum()),core=int(core.sum()),core_removed_by_protect=int((oldcore&protect).sum()),delete=int(delete.sum()),protect=int(protect.sum()),generate=int(mm['generate'].sum()),fence_tracks=int(valid.sum()),fence_inliers=int(inl.sum()),fence_median_error=float(np.median(err[inl[:,0]>0])),fence_homography=matrix.tolist()))
 previous=gray
sheet=Image.new('RGB',(W*2,H*3))
for j,i in enumerate([0,15,29]):
 im=rgb(out/'rgb'/f'{i:05}.png');an=im.copy();p=mask(out/'protect'/f'{i:05}.png');g=mask(out/'generate'/f'{i:05}.png');d=mask(out/'delete'/f'{i:05}.png')
 for m,col in [(p,[0,220,230]),(g,[60,80,255]),(d,[255,210,0])]:an[m]=(.45*an[m]+.55*np.array(col)).astype('uint8')
 sheet.paste(Image.fromarray(im),(0,j*H));sheet.paste(Image.fromarray(an),(W,j*H))
sheet.resize((1280,1072)).save(out/'mask_review.jpg',quality=94)
dump(out/'mask_stats.json',rows);dump(out/'fence_refinement.json',dict(method='assistant-reviewed SAM2 hollow candidate0 + bounded LK/RANSAC similarity foreground-mask tracking',automatic=False,selected_candidate=0,source='fence_probe/prompts.json',frames=30,old_mask_preserved=str(backup),rejected='unconstrained homography was degenerate despite low inlier residual'))
print('FENCE_REFINED',min(r['fence_inliers'] for r in rows),max(r['fence_median_error'] for r in rows))
