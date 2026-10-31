from pathlib import Path
import argparse,cv2,numpy as np,json,re
from skimage.metrics import structural_similarity
from PIL import Image,ImageDraw
p=argparse.ArgumentParser();p.add_argument('candidate_root');p.add_argument('output_dir');args=p.parse_args()
candidate_root=Path(args.candidate_root);out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
root=Path('C:/Users/dengyunlong/Documents/Codex/2026-10-31/g/work/horizon_trace');queries=sorted(root.glob('scene_*_first.png'));files=sorted(candidate_root.rglob('camera1.jpg'))+sorted(candidate_root.rglob('front_000_original.jpg'))
if len(queries)!=11 or not files:raise ValueError(f'Expected 11 queries and at least one camera1.jpg; got {len(queries)} / {len(files)}')
orb=cv2.ORB_create(nfeatures=2500);bf=cv2.BFMatcher(cv2.NORM_HAMMING);candidates=[]
for path in files:
 image=cv2.imread(str(path));
 if image is None:raise ValueError(f'Unreadable {path}')
 image=cv2.resize(image,(576,384),interpolation=cv2.INTER_AREA);gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);k,d=orb.detectAndCompute(gray,None);candidates.append((path,image,gray,k,d))
records=[];summary=[]
for query in queries:
 scene=re.search(r'scene_(\d+)_first',query.name).group(1);im=cv2.imread(str(query));
 if im.shape[:2]!=(384,576):raise ValueError(f'Unexpected dimensions {query}')
 gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);k1,d1=orb.detectAndCompute(gray,None);a=gray.astype(np.float32).ravel();this=[]
 for path,img,g,k2,d2 in candidates:
  b=g.astype(np.float32).ravel();den=float(a.std()*b.std());ncc=float(np.mean((a-a.mean())*(b-b.mean()))/den) if den else None
  mae=float(np.mean(np.abs(im.astype(np.float32)-img.astype(np.float32))));ssim=float(structural_similarity(gray,g,data_range=255));good=[]
  if d1 is not None and d2 is not None:good=[m for pair in bf.knnMatch(d1,d2,k=2) if len(pair)==2 for m,n in [pair] if m.distance<0.75*n.distance]
  inliers=0;fraction=0.;error=None
  if len(good)>=4:
   src=np.float32([k1[m.queryIdx].pt for m in good]);dst=np.float32([k2[m.trainIdx].pt for m in good]);h,mask=cv2.findHomography(src,dst,cv2.RANSAC,3.0)
   if h is not None and mask is not None:
    valid=mask.ravel().astype(bool);inliers=int(valid.sum());fraction=inliers/len(good)
    if inliers:
     predicted=cv2.perspectiveTransform(src.reshape(-1,1,2),h).reshape(-1,2);error=float(np.median(np.linalg.norm(predicted[valid]-dst[valid],axis=1)))
  item={'scene':scene,'candidate':str(path.relative_to(candidate_root)),'mae_rgb_0_255':mae,'ssim_gray':ssim,'ncc_gray':ncc,'orb_ratio_matches':len(good),'ransac_inliers':inliers,'ransac_inlier_fraction':fraction,'ransac_median_reprojection_px':error};records.append(item);this.append(item)
 ranked=sorted(this,key=lambda x:(x['ssim_gray'],-x['mae_rgb_0_255']),reverse=True);first=ranked[0];second=ranked[1] if len(ranked)>1 else None;gap=first['ssim_gray']-second['ssim_gray'] if second else None
 strong=first['ssim_gray']>=0.90 and first['mae_rgb_0_255']<=8 and first['ncc_gray'] is not None and first['ncc_gray']>=0.99 and first['ransac_inliers']>=100 and gap is not None and gap>=0.10
 summary.append({'scene':scene,'top':first,'second':second,'ssim_gap_to_second':gap,'conservative_strong_match':strong,'review_note':'Screening lead only; verify provenance, images and temporal evidence.' if strong else 'Thresholds not met; no automatic mapping.'})
 # Raw and query are only decoded/resized for inspection; originals remain untouched.
 chosen=next(c for c in candidates if str(c[0].relative_to(candidate_root))==first['candidate']);sheet=Image.new('RGB',(1152,416),'white');sheet.paste(Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)),(0,32));sheet.paste(Image.fromarray(cv2.cvtColor(chosen[1],cv2.COLOR_BGR2RGB)),(576,32));draw=ImageDraw.Draw(sheet);draw.text((8,8),f'HF {scene} frame 0',fill='black');draw.text((584,8),first['candidate'][:75],fill='black');sheet.save(out/f'scene_{scene}_top.png')
 print(scene,first['candidate'],'SSIM',round(first['ssim_gray'],4),'gap',round(gap,4) if gap is not None else None,'inliers',first['ransac_inliers'],'strong',strong,flush=True)
result={'candidate_root':str(candidate_root),'query_count':len(queries),'candidate_count':len(files),'comparison_count':len(records),'resize':'direct proportional 576x384 INTER_AREA, no crop','features':'gray SSIM and NCC; RGB MAE; ORB 2500, ratio 0.75, homography RANSAC 3px','conservative_screen_thresholds':{'ssim_min':0.90,'mae_max':8,'ncc_min':0.99,'ransac_inliers_min':100,'ssim_gap_to_second_min':0.10},'limitations':'Uncalibrated conservative screening thresholds; first-frame agreement does not establish complete sequence or dataset split.','summary':summary,'all_comparisons':records}
(out/'cross_candidate_metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
