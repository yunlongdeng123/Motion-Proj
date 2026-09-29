"""按已保存真实数据逐actor全帧审计；不重写原manifest和拒绝证据。"""
import argparse
import bisect
import json
from collections import Counter
from pathlib import Path
import numpy as np

def main(root):
    manifest=json.loads((root/'source_manifest.json').read_text())
    rows=[]
    for c in manifest['clips']:
        stamps=c['keyframe_timestamps'];actors=[]
        for a in c['actors']:
            reasons=[];token=a['instance_token']
            for f in c['frames']:
                i=bisect.bisect_right(stamps,f['timestamp'])
                if not 0<i<len(stamps):reasons.append({'frame':f['frame'],'reason':'annotation_extrapolation'});continue
                if not 0<stamps[i]-stamps[i-1]<=600000:reasons.append({'frame':f['frame'],'reason':'annotation_gap_gt600ms'})
                r=next((r for r in f['actors'] if r['instance_token']==token),None)
                if r is None or not r['projection']:
                    reasons.append({'frame':f['frame'],'reason':'missing_projection'});continue
                p=r['projection']
                if not(p['border']>=8 and p['width']>=72 and p['height']>=40 and .006<=p['area_frac']<=.18):
                    reasons.append({'frame':f['frame'],'reason':'size_or_border_gate'})
                m=np.asarray(f['camera_to_world']);k=np.asarray(f['intrinsics_1024'])
                if not(np.isfinite(m).all() and np.isfinite(k).all() and np.allclose(m[3],[0,0,0,1]) and abs(np.linalg.det(m[:3,:3])-1)<1e-5):
                    reasons.append({'frame':f['frame'],'reason':'camera_invalid'})
            actors.append({'instance_token':token,'pass':not reasons,'reasons':reasons})
        rows.append({'source_id':c['source_id'],'scene':c['scene'],'primary_geometry_pass':actors[0]['pass'],
                     'actors':actors,'source_status':'pending_visual_review' if actors[0]['pass'] else 'geometry_reject'})
    out={'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r1','scope':'all saved source frames; interpolated annotations only; not SAM2 or placement',
         'clips':rows,'counts':dict(Counter(r['source_status'] for r in rows)),
         'source_manifest_unchanged':True,'subagent_review_pending':True,'synthetic_quality':'not_created'}
    (root/'source_geometry_validation.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(out['counts']),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
