from pathlib import Path
import cv2,numpy as np,json
from skimage.metrics import structural_similarity
from PIL import Image,ImageDraw
root=Path('C:/Users/dengyunlong/Documents/Codex/2026-10-31/g');w=root/'work/horizon_trace';out=root/'outputs/HorizonForge_Waymo_Source_Trace_20261031/visual_verification';out.mkdir(parents=True,exist_ok=True)
raw=w/'sorted_val035_dense/front_576x384';files=sorted(raw.glob('front_*.jpg'));assert len(files)==198
cap=cv2.VideoCapture(str(w/'scene_035_original.mp4'));queries={}
for n in [0,30,60,89]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,n);ok,img=cap.read();assert ok;queries[n]=img
cap.release();rows=[]
for f in files:
 x=cv2.imread(str(f));assert x.shape[:2]==(384,576);g=cv2.cvtColor(x,cv2.COLOR_BGR2GRAY).astype(np.float32);b=g.ravel();raw_index=int(f.stem.split('_')[-1])
 for n,q in queries.items():
  a=cv2.cvtColor(q,cv2.COLOR_BGR2GRAY).astype(np.float32).ravel();ncc=float(np.mean((a-a.mean())*(b-b.mean()))/(a.std()*b.std()));mae=float(np.abs(x.astype(np.float32)-q.astype(np.float32)).mean());rows.append({'query_frame':n,'raw_frame':raw_index,'ncc_gray':ncc,'mae_rgb_0_255':mae,'raw_path':str(f)})
orb=cv2.ORB_create(nfeatures=2500);bf=cv2.BFMatcher(cv2.NORM_HAMMING);summaries=[]
for n,q in queries.items():
 qgray=cv2.cvtColor(q,cv2.COLOR_BGR2GRAY);k1,d1=orb.detectAndCompute(qgray,None);ranked=sorted((r for r in rows if r['query_frame']==n),key=lambda r:r['mae_rgb_0_255']);best20=ranked[:20]
 for r in best20:
  im=cv2.imread(r['raw_path']);gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);r['ssim_gray']=float(structural_similarity(qgray,gray,data_range=255));k2,d2=orb.detectAndCompute(gray,None);good=[]
  if d1 is not None and d2 is not None:good=[m for pair in bf.knnMatch(d1,d2,k=2) if len(pair)==2 for m,j in [pair] if m.distance<.75*j.distance]
  r['orb_ratio_matches']=len(good);r['ransac_inliers']=0;r['ransac_median_reprojection_px']=None
  if len(good)>=4:
   a=np.float32([k1[m.queryIdx].pt for m in good]);b=np.float32([k2[m.trainIdx].pt for m in good]);h,mask=cv2.findHomography(a,b,cv2.RANSAC,3)
   if h is not None and mask is not None:
    valid=mask.ravel().astype(bool);r['ransac_inliers']=int(valid.sum());r['ransac_inlier_fraction']=float(valid.mean())
    if valid.any():r['ransac_median_reprojection_px']=float(np.median(np.linalg.norm(cv2.perspectiveTransform(a.reshape(-1,1,2),h).reshape(-1,2)[valid]-b[valid],axis=1)))
 best=ranked[0];summaries.append({'query_frame':n,'best':best,'second':ranked[1],'mae_gap_to_second':ranked[1]['mae_rgb_0_255']-best['mae_rgb_0_255'],'full_metrics_computed_for_top_mae_count':20})
 other=cv2.imread(best['raw_path']);sheet=Image.new('RGB',(1152,416),'white');sheet.paste(Image.fromarray(cv2.cvtColor(q,cv2.COLOR_BGR2RGB)),(0,32));sheet.paste(Image.fromarray(cv2.cvtColor(other,cv2.COLOR_BGR2RGB)),(576,32));d=ImageDraw.Draw(sheet);d.text((8,8),f'HF035 video frame {n}',fill='black');d.text((584,8),f'candidate original frame {best["raw_frame"]} (analysis copy)',fill='black');sheet.save(out/f'dense_query_{n:03d}_best.png')
 print(n,best['raw_frame'],round(best['mae_rgb_0_255'],3),round(best['ssim_gray'],5),round(best['ncc_gray'],5),best['ransac_inliers'],'gap',round(ranked[1]['mae_rgb_0_255']-best['mae_rgb_0_255'],3),flush=True)
manifest=json.loads((w/'sorted_val035_dense/manifest.json').read_text(encoding='utf8'));result={'context_id':manifest['context_id'],'raw_representation':manifest['representation'],'raw_manifest':str(w/'sorted_val035_dense/manifest.json'),'candidate_frame_count':len(files),'query_frame_indices':list(queries),'method':'MAE and NCC all 198 candidates per query; gray SSIM and ORB ratio 0.75 / RANSAC 3px on 20 lowest MAE per query','summary':summaries,'comparisons':rows,'limitations':'Candidate frames are resized and JPEG-encoded analysis copies; no claims of bitwise source equality. ORB is not calculated on candidates beyond the top 20 by MAE.'};(out/'dense_035_metrics.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
