"""把潜在邻车包络风险转成实际SAM2待检输入；不把包络交叠直接判分割失败。"""
from pathlib import Path
import os,sys,copy,shutil
from collections import defaultdict,Counter
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77/target_protected')
import cv2,numpy as np
from scipy.ndimage import median_filter
from geometry_factory import Geometry,read,dump,source_masks,projection
from build_pairs import actor_contact,evaluate
from iteration2.donor_gate import check_masks
from iteration5.planning import normalized_masks
from exact_pairs import O,F,T
def main():
    cv2.setNumThreads(1);geo=Geometry(F);need=defaultdict(Counter);rejections=Counter()
    for sid,c in geo.sources.items():
        if not sid.startswith('N'):continue
        plans=read(O/'preflight'/f'{sid}.json')['plans']+read(O/'cross_preflight'/f'{sid}.json')['plans']
        if not plans:continue
        dm=source_masks(F,sid);stats=normalized_masks(dm,[f['actors'][0]['projection']['box_xyxy'] for f in c['frames']])
        if not check_masks(c['actors'][0]['instance_token'],dm)['eligible_for_pairing'] or any(s['normalized_iou']<.8 or not .85<=s['normalized_area_ratio']<=1.15 for s in stats):continue
        geo.prepare(sid);known={c['actors'][0]['instance_token']:dm}
        for a in c['actors'][1:]:
            jid=sid+'_'+a['instance_token'][:8]
            if (F/'segmented_secondary'/jid/'sam2_raw').exists():known[a['instance_token']]=source_masks(F,sid,jid)
        dcache={}
        for tr in plans:
            ds=tr['donor_source_id'];d=geo.sources[ds]
            if ds not in dcache:
                m=source_masks(F,ds);g=check_masks(d['actors'][0]['instance_token'],m)
                if not g['eligible_for_pairing']:dcache[ds]=None
                else:
                    ct=median_filter(np.array([actor_contact(mm,np.array(f['actors'][0]['projection']['box_xyxy'])) for mm,f in zip(m,d['frames'])]),size=5,mode='nearest');dcache[ds]=(m,ct)
            if dcache[ds] is None:continue
            q,why=evaluate(geo,tr,*dcache[ds],known,diagnose_unreviewed=True,ego_bottom_guard_px=64)
            if q is None:continue
            for hit in q['unverified_actor_envelope_hits']:need[sid][hit['instance_token']]+=1
    manifest=read(F/'source_manifest.json');src={c['source_id']:c for c in manifest['clips']};jobs=[];accepted=[]
    for sid,counts in need.items():
        c=src[sid]
        for tok,n in counts.most_common():
            arr=[next((a for a in obs if a['instance_token']==tok),None) for obs in geo.obstacles[sid]]
            if any(a is None or a.get('interpolation_uncertain') for a in arr):rejections['context_track_incomplete']+=1;continue
            if not arr[0]['category'].startswith('vehicle.'):rejections['nonvehicle_context_conservatively_excluded']+=1;continue
            ps=[a['_projection'] for a in arr]
            if any(p is None or p['box'][2]-p['box'][0]<72 or p['box'][3]-p['box'][1]<40 or min(p['box'][0],p['box'][1],1024-p['box'][2],576-p['box'][3])<8 for p in ps):rejections['context_not_well_observed_size_border']+=1;continue
            # 当前10帧真实插值区间的端点均需高visibility；不要求窗外四秒都完整。
            context=geo.context[sid]['frames'];anns=[a for f in context for a in f['annotations'] if a['instance_token']==tok]
            activekeys={i for f in c['frames'] for i in [np.searchsorted(c['keyframe_timestamps'],f['timestamp'])-1,np.searchsorted(c['keyframe_timestamps'],f['timestamp'])]}
            if any(not any(a['instance_token']==tok and int(a['visibility_token'])==4 for a in context[i]['annotations']) for i in activekeys):rejections['context_visibility_not4']+=1;continue
            if any(a['instance_token']==tok for a in c['actors']):continue
            c['actors'].append({'instance_token':tok,'category':arr[0]['category'],'context_extension':'actual 10-frame window, high visibility at its bracket keyframes'})
            for f,a,p in zip(c['frames'],arr,ps):
                f['actors'].append({k:copy.deepcopy(a[k]) for k in ['instance_token','category','translation','rotation','size']}|{'projection':{'box_xyxy':p['box'].tolist(),'depth':p['center_depth']}})
            jid=sid+'_'+tok[:8]
            accepted.append({'source_id':sid,'instance_token':tok,'job_id':jid,'candidate_hit_count':n,'category':arr[0]['category']})
    backup=O/'before_context_refine';backup.mkdir(exist_ok=True)
    for path in [F/'source_manifest.json',F/'pair_candidates.json',O/'all_exact_candidates.json',O/'mask_numeric_audit.json',O/'sam_queue.json']:
        dst=backup/path.name
        if not dst.exists():shutil.copy2(path,dst)
    dump(F/'source_manifest.json',manifest)
    dump(O/'context_refinement.json',{'potential_context_instances':sum(len(c) for c in need.values()),'added_well_observed_tracks':accepted,'rejections':dict(rejections),'method':'query precise neighboring instances for conservative envelope overlap; no mask clipping/threshold relaxation; final all-frame technical gates and independent QA still required'})
    print('CONTEXT',len(accepted),'sources',len({r['source_id'] for r in accepted}),'rejections',dict(rejections),flush=True)
if __name__=='__main__':main()
