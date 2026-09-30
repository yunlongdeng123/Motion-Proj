"""接真实SAM2结果与独立QA，才调用原精确遮挡关卡；默认缺输入则等待。"""
import argparse,sys
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from geometry_factory import Geometry,read,dump
from build_pairs import evaluate
from iteration4.segment_receivers import reusable
from planning import inputs,silhouettes,normalized_masks

def review_blockers(job,review):
    if review is None:return ['missing_independent_receiver_mask_review']
    if any(review.get(k)!=job[k] for k in ['job_id','source_id','instance_token']):return ['independent_review_provenance_mismatch']
    if review.get('mask_status')!='pass':return ['receiver_mask_'+str(review.get('mask_status','pending'))]
    return []

def main(parent,root):
    cv2=__import__('cv2');cv2.setNumThreads(1);root.mkdir(exist_ok=True,parents=True)
    sources,donors,donor_root,pairs,source_qa=inputs(parent)
    queue={j['job_id']:j for j in read(parent/'gpu_queue.json')['jobs']}
    qa_file=parent/'receiver_mask_reviews.json'
    reviews=read(qa_file)['jobs'] if qa_file.exists() else []
    byjob={r['job_id']:r for r in reviews};assert len(byjob)==len(reviews),'重复独立mask评审ID'
    geo=None;rows=[];selected=[];template=[]
    for pair_index,pair in enumerate(pairs):
        sid=pair['source_id'];required=[];blockers=[];protected={};numeric=[]
        if source_qa[sid]['source_status']!='pass':blockers.append('source_preflight_not_pass')
        for tok in pair['required_protected_instances']:
            jid=sid+'_'+tok[:8];job=queue[jid];required.append(jid)
            template.append({'job_id':jid,'source_id':sid,'instance_token':tok,'mask_status':None,'note':'','human_verdict':None})
            dest=parent/'segmented_receivers'/jid
            if not reusable(dest,job):blockers.append(jid+':missing_or_incomplete_SAM2_provenance');continue
            blockers+=['%s:%s'%(jid,b) for b in review_blockers(job,byjob.get(jid))]
            masks=[]
            for i in range(10):
                with Image.open(dest/'sam2_raw'/f'{i:05}.png') as im:masks.append(np.asarray(im)>0)
            stats=normalized_masks(masks,job['box_prompts'])
            bad=[m for m in stats if m['pixels']==0 or m['normalized_iou']<.8 or not .85<=m['normalized_area_ratio']<=1.15]
            numeric.append({'job_id':jid,'pass':not bad,'metrics':stats})
            if bad:blockers.append(jid+':mask_numeric_continuity_gate')
            protected[tok]=masks
        row={'case_id':sid,'required_jobs':required,'blockers':blockers,'numeric_masks':numeric,
             'status':'waiting_inputs' if blockers else 'pending_exact_geometry','render_allowed':False,'training_ready':False,'human_verdict':None}
        if not blockers:
            if geo is None:geo=Geometry(parent/'native10_factory')
            donor=donors[pair['donor_source_id']];geo.sources[donor['source_id']]=donor
            for f in donor['frames']:f['_w2c']=np.linalg.inv(f['camera_to_world'])
            assert geo.prepare(sid)['pass']
            rebuilt,reason=geo.trajectory(sid,pair['offset_longitudinal_m'],pair['offset_lateral_m'],donor['source_id'],pair['placement_mode'],0)
            assert rebuilt is not None,reason
            for a,b in zip(rebuilt['frames'],pair['frames']):assert np.allclose(a['box'],b['box'],atol=1e-5) and np.allclose(a['actor']['translation'],b['actor']['translation'],atol=1e-7)
            dmasks,contacts,_=silhouettes(pair,donor,donor_root)
            quality,reason=evaluate(geo,rebuilt,dmasks,contacts,protected,ego_bottom_guard_px=64)
            if quality is None:row.update(status='exact_gate_reject',blockers=[reason])
            elif quality['type']!=pair['planned_type']:row.update(status='exact_gate_reject',blockers=['planned_type_differs_from_real_mask_type'])
            else:
                # 在看到SAM2/合成结果之前按候选原序固定旧边缘配方。
                p=rebuilt|quality|{'case_id':sid,'human_verdict':None,'training_ready':False,'quality_status':'exact_gate_pass_pending_actual_render_and_independent_QA',
                                  'edge_mode':['near_hard','feather_05','feather_10'][pair_index%3],
                                  'extra_motion_blur_px':[0,.4,.8][pair_index%3],'mask_dilation_px':[2,3,4][pair_index%3]}
                selected.append(p);row.update(status='exact_gate_pass_pending_actual_render',render_allowed=True)
        rows.append(row)
    result={'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r5','parent_run':str(parent),'cases':rows,
            'status_counts':dict(Counter(r['status'] for r in rows)),'exact_gate_pass':len(selected),'training_ready':0,'GPU_calls':0,
            'rule_scope':'original build_pairs.evaluate; no threshold changes, GT envelope not substituted for true SAM2 masks'}
    dump(root/'exact_admission.json',result);dump(root/'receiver_mask_reviews.template.json',{'jobs':template,'template_only':True})
    dump(root/'exact_pair_candidates.json',{'config':{'run_id':'r5','image_bottom_guard_px':64},'selected':selected,'qualified_synthetic_count':0})
    print('EXACT_ADMISSION',result['status_counts'],'render_candidates',len(selected))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.parent,a.root)
