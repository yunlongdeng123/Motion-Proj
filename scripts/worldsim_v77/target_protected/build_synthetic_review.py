"""最终合成质量页：独立通过才可人工逐帧打分，失败保留诊断入口。"""
import argparse,json
from pathlib import Path
from collections import Counter
from geometry_factory import read,dump

def build(root,out):
    full=read(out/'synthetic_manifest.json');qa_path=out/'independent_synthetic_reviews.json'
    reviews={c['case_id']:c for c in read(qa_path)['clips']} if qa_path.exists() else {}
    valpath=out/'delivery_validation.json';validation={c['case_id']:c for c in read(valpath)['cases']} if valpath.exists() else {}
    rows=[]
    for r in full['clips']:
        q=reviews.get(r['case_id'],{'synthetic_status':'pending','reviewed_frames':[],'issues':[],'note':'尚未完成独立合成质检'})
        if not validation.get(r['case_id'],{}).get('pass'):
            q=dict(q,synthetic_status='engineering_pending_or_reject',note=q['note']+'；实际产物工程门槛未通过或尚未检查，暂不能准入。')
        rows.append({k:r[k] for k in ['case_id','source_id','scene','donor_source_id','donor_scene','camera','type','donor_instance','receiver_primary','protected_instances','edge_mode','extra_motion_blur_px','mask_dilation_px','min_GT_clearance_m','max_ground_support_distance_m','max_view_yaw_delta_deg','max_view_pitch_delta_deg','videos','preview_frames','contacts','review_frames']}|{'review':q,'human_verdict':None,'training_ready':False,
          'remaining_protected_min':{t:min(m['remaining_protected_fraction'][t] for m in r['pixel_metrics']) for t in r['pixel_metrics'][0]['remaining_protected_fraction']},
          'pixel_contract_pass':all(m['pixel_contract'] for m in r['pixel_metrics']),
          'condition_scope':r['condition_scope'],'frame_count':len(r['preview_frames']),
          'placement_mode':r.get('placement_mode','world_offset'),'source_window':r.get('source_window'),
          'donor_window':r.get('donor_window'),'cohort':r.get('cohort','30-frame pilot')})
    data={'task_id':full['task_id'],'run_id':full['run_id'],'human_scoring_scope':'final_synthetic_per_frame_v1',
          'clips':rows,'counts':dict(Counter(r['review']['synthetic_status'] for r in rows)),
          'type_counts':dict(Counter(r['type'] for r in rows)),'training_ready':0,
          'scene_count':len({r['scene'] for r in rows}),'passed_scene_count':len({r['scene'] for r in rows if r['review']['synthetic_status']=='pass'}),
          'frame_counts':sorted({len(r['preview_frames']) for r in rows})}
    dump(out/'review_manifest.json',data)
    text=Path(__file__).with_name('synthetic_review.html').read_text(encoding='utf-8').replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('</','<\\/'))
    (out/'index.html').write_text(text,encoding='utf-8',newline='\n')
    print('HTML',len(rows),data['counts'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();build(a.root,a.out)
