"""CPU图像特征定位用户截图，保存候选及像素证据，不据截图猜case ID。"""
import json, argparse
from pathlib import Path
import cv2,numpy as np

def main(root,delivery):
    cv2.setNumThreads(2)
    shot=next((root/'raw').glob('*.png')); q=cv2.imread(str(shot));sift=cv2.SIFT_create(nfeatures=1200)
    kq,dq=sift.detectAndCompute(cv2.cvtColor(q,cv2.COLOR_BGR2GRAY),None);bf=cv2.BFMatcher()
    manifest=json.loads((delivery/'review_manifest.json').read_text());rows=[]
    for c in manifest['clips']:
        for f in c['preview_frames']:
            im=cv2.imread(str(delivery/f['input']));kp,dp=sift.detectAndCompute(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),None)
            matches=[a for a,b in bf.knnMatch(dq,dp,k=2) if a.distance<.7*b.distance]
            if len(matches)<8:continue
            a=np.float32([kq[m.queryIdx].pt for m in matches]);b=np.float32([kp[m.trainIdx].pt for m in matches]);H,inl=cv2.findHomography(a,b,cv2.RANSAC,2)
            if H is None:continue
            rows.append({'case_id':c['case_id'],'frame':f['frame'],'inliers':int(inl.sum()),'matches':len(matches),'donor_source_id':c['donor_source_id'],'input':f['input'],'homography_screenshot_to_input':H.tolist()})
    rows.sort(key=lambda r:r['inliers'],reverse=True)
    (root/'screenshot_matches.json').write_text(json.dumps(rows[:20],indent=2))
    print(json.dumps(rows[:5],indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--delivery',type=Path,required=True);a=p.parse_args();main(a.root,a.delivery)
