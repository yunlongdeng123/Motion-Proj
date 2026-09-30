"""r4空间预案的固定连续轮廓；保持原曝光、位姿与6px洞上界。"""
import sys
from pathlib import Path
import cv2,numpy as np
from scipy.ndimage import median_filter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from geometry_factory import read,source_masks
from build_pairs import actor_contact,warp_matrix
from iteration2.donor_gate import assert_donor

def inputs(parent):
    sources={c['source_id']:c for c in read(parent/'native10_factory/source_manifest.json')['clips']}
    config=read(parent/'planner_config.json');donor_root=Path(config['donor_root'])
    donors={c['source_id']:c for c in read(donor_root/'source_manifest.json')['clips']}
    plan=read(parent/'selected_pair_plan.json');assert plan['summary']['complete']
    qa={c['case_id']:c for c in read(parent/'independent_preflight_reviews.json')['cases']}
    return sources,donors,donor_root,plan['plans'],qa

def silhouettes(pair,donor,donor_root):
    masks=source_masks(donor_root,donor['source_id']);assert_donor(donor,masks)
    assert len(masks)==len(pair['frames'])==len(donor['frames'])==10
    contact=median_filter(np.array([actor_contact(m,np.array(f['actors'][0]['projection']['box_xyxy'])) for m,f in zip(masks,donor['frames'])]),size=5,mode='nearest')
    frames=[]
    for m,df,p,c in zip(masks,donor['frames'],pair['frames'],contact):
        a=cv2.warpAffine(m.astype('uint8'),warp_matrix(df,p,c),(1024,576),flags=cv2.INTER_NEAREST)>0
        h=cv2.dilate(a.astype('uint8'),np.ones((13,13),'uint8'))>0
        assert a.any() and not h[512:].any()
        frames.append((a,h))
    return masks,contact,frames

def normalized_masks(masks,boxes):
    normalized=[];metrics=[]
    for i,(m,b) in enumerate(zip(masks,boxes)):
        b=np.array(b);sx=128/(b[2]-b[0]);sy=128/(b[3]-b[1])
        n=cv2.warpAffine(m.astype('uint8'),np.float32([[sx,0,-b[0]*sx],[0,sy,-b[1]*sy]]),(128,128),flags=cv2.INTER_NEAREST)>0
        prev=normalized[-1] if normalized else n
        metrics.append({'frame':i,'pixels':int(m.sum()),'normalized_iou':float((n&prev).sum()/max(1,(n|prev).sum())),
                        'normalized_area_ratio':float(n.sum()/max(1,prev.sum()))})
        normalized.append(n)
    return metrics
