"""便宜的供体入口检查：先挡明确失败，再投入配对、合成与AI检查。"""
import json,argparse
from pathlib import Path
import numpy as np
import cv2

def load_policy():return json.loads(Path(__file__).with_name('donor_reuse_policy.json').read_text())
def check_masks(instance,masks,policy=None):
    policy=policy or load_policy();reasons=[];rows=[]
    if instance in policy['blocked_instances']:reasons.append('human_reported_source_blocked')
    for i,raw in enumerate(masks):
        m=np.asarray(raw,dtype=np.uint8);ys,xs=np.where(m)
        if m.shape!=(policy['frame_height'],1024):raise ValueError('policy calibrated only for1024x576')
        if not len(ys):reasons.append('empty_mask');rows.append({'frame':i,'empty':True});continue
        n,_,stats,_=cv2.connectedComponentsWithStats(m,8);areas=stats[1:,cv2.CC_STAT_AREA];secondary=1-float(areas.max())/len(ys)
        bottom=int(ys.max());edge=min(int(xs.min()),int(ys.min()),1023-int(xs.max()),575-bottom)
        if bottom+policy['clearance_px']>=576-policy['source_bottom_exclusion_px']:reasons.append('source_bottom_or_ego_ambiguity')
        if edge<policy['clearance_px']:reasons.append('image_boundary')
        if secondary>policy['max_secondary_component_fraction']:reasons.append('secondary_component_needs_new_source_or_mask')
        rows.append({'frame':i,'bottom_px':bottom,'edge_clearance_px':edge,'secondary_fraction':secondary})
    return {'eligible_for_pairing':not reasons,'reasons':sorted(set(reasons)),'frames':rows,'visual_quality_certified':False}

def assert_donor(source,masks):
    result=check_masks(source['actors'][0]['instance_token'],masks)
    if not result['eligible_for_pairing']:raise ValueError('donor admission failed '+source['source_id']+': '+','.join(result['reasons']))
    return result

def main(root,factory):
    cv2.setNumThreads(1);qa=json.loads((factory/'mask_review/independent_mask_reviews.json').read_text())['clips'];sources={c['source_id']:c for c in json.loads((factory/'source_manifest.json').read_text())['clips']};rows=[]
    for q in qa:
        if q['donor_mask_status']!='pass':continue
        sid=q['source_id'];masks=[cv2.imread(str(p),0)>0 for p in sorted((factory/'segmented'/sid/'sam2_raw').glob('*.png'))]
        r=check_masks(sources[sid]['actors'][0]['instance_token'],masks);rows.append({'source_id':sid,'instance_token':sources[sid]['actors'][0]['instance_token']}|r)
    result={'policy':load_policy(),'sources':rows,'checked_sources':len(rows),'mechanical_eligible':sum(r['eligible_for_pairing'] for r in rows),'not_visual_re_review':True}
    (root/'donor_gate_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print({k:v for k,v in result.items() if k not in ('sources','policy')})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--factory',type=Path,required=True);a=p.parse_args();main(a.root,a.factory)
