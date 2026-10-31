from pathlib import Path
import cv2, numpy as np,json,re,hashlib
from skimage.metrics import structural_similarity
from PIL import Image,ImageDraw
root=Path('C:/Users/dengyunlong/Documents/Codex/2026-10-31/g');work=root/'work/horizon_trace';raw=work/'raw_candidates/sorted_val_035';out=root/'outputs/HorizonForge_Waymo_Source_Trace_20261031/visual_verification';out.mkdir(parents=True,exist_ok=True)
video=work/'scene_035_original.mp4';target=[0,30,60,89];cap=cv2.VideoCapture(str(video));queries={}
for i in target:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read()
 if not ok:raise RuntimeError(f'Cannot decode query {i}')
 queries[i]=im
cap.release()
files=sorted(raw.glob('front_[0-9][0-9][0-9].jpg')); orb=cv2.ORB_create(nfeatures=2500);bf=cv2.BFMatcher(cv2.NORM_HAMMING); qfeatures={}
for i,img in queries.items():qfeatures[i]=orb.detectAndCompute(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),None)
records=[]
for path in files:
 raw_index=int(path.stem.split('_')[-1]);im=cv2.imread(str(path)); resized=cv2.resize(im,(576,384),interpolation=cv2.INTER_AREA);gray=cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY);kp2,d2=orb.detectAndCompute(gray,None)
 for qi,query in queries.items():
  qgray=cv2.cvtColor(query,cv2.COLOR_BGR2GRAY);k1,d1=qfeatures[qi];mae=float(np.mean(np.abs(query.astype(np.float32)-resized.astype(np.float32))))
  a=qgray.astype(np.float32).ravel();b=gray.astype(np.float32).ravel();std=float(a.std()*b.std()); ncc=float(np.mean((a-a.mean())*(b-b.mean()))/std) if std else None
  ssim=float(structural_similarity(qgray,gray,data_range=255));good=[]
  if d1 is not None and d2 is not None:
   good=[m for pair in bf.knnMatch(d1,d2,k=2) if len(pair)==2 for m,n in [pair] if m.distance < 0.75*n.distance]
  inliers=0;frac=0.;median_error=None;matrix=None
  if len(good)>=4:
   src=np.float32([k1[m.queryIdx].pt for m in good]);dst=np.float32([kp2[m.trainIdx].pt for m in good]);h,mask=cv2.findHomography(src,dst,cv2.RANSAC,3.0)
   if h is not None and mask is not None:
    valid=mask.ravel().astype(bool);inliers=int(valid.sum());frac=inliers/len(good);matrix=h.tolist()
    if inliers:
     projected=cv2.perspectiveTransform(src.reshape(-1,1,2),h).reshape(-1,2);median_error=float(np.median(np.linalg.norm(projected[valid]-dst[valid],axis=1)))
  records.append({'query_frame':qi,'raw_frame':raw_index,'raw_path':str(path),'mae_rgb_0_255':mae,'ssim_gray':ssim,'ncc_gray':ncc,'orb_ratio_075_matches':len(good),'ransac_inliers':inliers,'ransac_inlier_fraction':frac,'ransac_median_reprojection_px':median_error,'homography_query_to_raw':matrix})
rankings={}
for qi in target:
 r=[x for x in records if x['query_frame']==qi];ranked=sorted(r,key=lambda x:(x['ssim_gray'],-x['mae_rgb_0_255']),reverse=True);rankings[str(qi)]={'ssim_ranked_top10':ranked[:10],'orb_ranked_top10':sorted(r,key=lambda x:(x['ransac_inliers'],x['ssim_gray']),reverse=True)[:10]}
 # contact sheet retains decoded query and resized original, with labels outside images.
 match=ranked[0];source=cv2.imread(match['raw_path']);resized=cv2.resize(source,(576,384),interpolation=cv2.INTER_AREA);sheet=Image.new('RGB',(1152,416),'white');sheet.paste(Image.fromarray(cv2.cvtColor(queries[qi],cv2.COLOR_BGR2RGB)),(0,32));sheet.paste(Image.fromarray(cv2.cvtColor(resized,cv2.COLOR_BGR2RGB)),(576,32));draw=ImageDraw.Draw(sheet);draw.text((8,8),f'HF035 video frame {qi}',fill='black');draw.text((584,8),f'Waymo candidate frame {match["raw_frame"]} (best SSIM)',fill='black');sheet.save(out/f'query_{qi:03d}_best_ssim.png')
manifest=json.loads((raw/'manifest.json').read_text(encoding='utf8'))
result={'query_video':str(video),'candidate_label':'sorted_val_035','candidate_context_id':manifest['context_id'],'candidate_manifest':str(raw/'manifest.json'),'query_frames':target,'candidate_frame_count':len(files),'candidate_frames':[int(x.stem.split('_')[-1]) for x in files],'resizing':'source 1920x1280 resized directly to 576x384 using INTER_AREA; no crop','methods':'gray SSIM and NCC; RGB MAE; ORB 2500 features, Lowe ratio 0.75, homography RANSAC 3 px','candidate_manifest_original_jpeg_unchanged':True,'comparisons':records,'rankings':rankings,'limitations':'Only decoded image matching. Similar location or isolated features are insufficient to establish same context; identify temporal correspondence across frames and examine scores and geometry. JPEG, encoding and resizing affect values.'}
(out/'matching_metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
print('candidate frames',len(files));
for q in target:
 x=rankings[str(q)]['ssim_ranked_top10'][0];print(q,'best raw',x['raw_frame'],'SSIM',round(x['ssim_gray'],4),'MAE',round(x['mae_rgb_0_255'],2),'NCC',round(x['ncc_gray'],4),'ORB inliers',x['ransac_inliers'])
