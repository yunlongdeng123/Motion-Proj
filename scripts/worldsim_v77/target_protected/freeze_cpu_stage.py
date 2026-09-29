"""CPU收口：汇合真实文件、几何和独立来源审核，生成待GPU队列，不运行模型。"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def dump(p, obj):
    p=Path(p); tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    tmp.replace(p)


def main(root):
    m=read(root/'source_manifest.json')
    g={r['source_id']:r for r in read(root/'source_geometry_validation.json')['clips']}
    review=read(root/'subagent_source_reviews.json')
    assert (review['task_id'],review['run_id'])==(m['task_id'],m['run_id'])
    assert review['review_stage']=='source_only_cpu'
    qa={r['source_id']:r for r in review['clips']}
    assert len(qa)==len(review['clips'])
    extraction=read(root/'extract_state.json')
    assert extraction['state'] in ['complete','missing_or_bad'], '解包和解码核验尚未结束'
    assert not extraction['bad'], '损坏文件必须先修复；不能被skip-old-files掩盖'
    available={c['source_id'] for c in m['clips'] if all((root/'rgb'/f['filename']).is_file() for f in c['frames'])}
    assert set(qa)==available, '每个已齐备来源都须独立检查；缺图来源不能有视觉通过记录'
    jobs=[]; decisions=[]; missing_sources=[]
    for c in m['clips']:
        sid=c['source_id']; geom=g[sid]['primary_geometry_pass']
        assert len(c['frames'])==30
        if sid not in available:
            missing_sources.append({'source_id':sid,'scene':c['scene'],'reason':'source_rgb_not_materialized_from_selected_archive',
                                    'missing_filenames':[f['filename'] for f in c['frames'] if not (root/'rgb'/f['filename']).is_file()],
                                    'source_review':'not_reviewable','admitted':False})
            continue
        q=qa[sid]
        assert {0,15,29}<=set(q['reviewed_frames'])
        assert set(q['reviewed_frames'])<=set(range(30))
        assert q['note'].strip()
        for key in ['receiver_status','donor_status']:
            assert q[key] in ['pass','reject','uncertain']
            assert geom or q[key]!='pass', (sid,'几何硬拒绝不能被视觉通过覆盖')
        roles=[role for role,key in [('receiver','receiver_status'),('donor','donor_status')] if geom and q[key]=='pass']
        decisions.append({'source_id':sid,'scene':c['scene'],'instance_token':c['actors'][0]['instance_token'],
                          'geometry_pass':geom,'eligible_source_roles':roles,'source_review':q,
                          'synthetic_quality':'not_created','human_verdict':None})
        if not roles:continue
        jobs.append({'source_id':sid,'scene':c['scene'],'instance_token':c['actors'][0]['instance_token'],
                     'role':'primary_reviewed_actor','eligible_source_roles':roles,'prompt_frame':15,
                     'frame_filenames':[f['filename'] for f in c['frames']],
                     'purpose':'source_instance_masks_only; not synthetic-pair approval'})
    assert jobs
    context={c['source_id']:c for c in read(root/'source_context.json')['clips']}
    context_missing=[]
    for job in jobs:
        for frame in context[job['source_id']]['frames']:
            for row in frame['sensors'].values():
                if row['materialize'] and not (root/'rgb'/row['filename']).is_file():
                    context_missing.append({'source_id':job['source_id'],'filename':row['filename']})
    assert not context_missing, '准入来源的相机/LiDAR/传感器参考文件未齐备'
    queue={'task_id':m['task_id'],'run_id':m['run_id'],'stage':'awaiting_gpu','seed':42,
           'model':'SAM2.1_hiera_large','architecture_change':False,'jobs':jobs,
           'secondary_actor_policy':'Unreviewed secondary candidates are excluded; dense-pair actors need separate review.',
           'next_gates':['raw_mask_review','3D_placement_and_ground_support','synthesized_pair_rule_checks',
                         'independent_each_case_sampled_review','human_every_frame_review']}
    dump(root/'segmentation_queue.json',queue)
    inv=read(root/'source_file_inventory.json')
    summary={'task_id':m['task_id'],'run_id':m['run_id'],'state':'cpu_complete_awaiting_gpu',
             'selected_source_scenes':len(m['clips'])+len(m['sampling_rejected']),
             'continuous_rgb_sources':len(m['clips']),'sampling_rejected':len(m['sampling_rejected']),
             'available_rgb_sources':len(available),'source_rgb_missing':missing_sources,
             'geometry_pass_among_available':sum(g[sid]['primary_geometry_pass'] for sid in available),
             'independent_reviewed':len(qa),
             'receiver':dict(Counter(q['receiver_status'] for q in qa.values())),
             'donor':dict(Counter(q['donor_status'] for q in qa.values())),
             'gpu_segmentation_jobs':len(jobs),'gpu_frames':30*len(jobs),
             'queued_context_reference_missing':context_missing,
             'source_scene_count':len({c['scene'] for c in m['clips'] if c['source_id'] in available}),
             'source_log_counts':dict(Counter(c['log_token'] for c in m['clips'] if c['source_id'] in available)),
             'source_location_counts':dict(Counter(c['location'] for c in m['clips'] if c['source_id'] in available)),
             'camera_counts':dict(Counter(c['camera'] for c in m['clips'] if c['source_id'] in available)),
             'inventory':{'all_files':len(inv),'jpeg_files':sum(r['filename'].endswith('.jpg') for r in inv),
                          'sensor_binary_files':sum(not r['filename'].endswith('.jpg') for r in inv),
                          'actual_bytes':sum(r['bytes'] for r in inv),'missing':len(extraction['missing']),
                          'bad':len(extraction['bad'])},
             'planned_synthetic_case_budget':50,'qualified_synthetic_pairs':0,'human_approved_pairs':0,
             'model_calls_this_cpu_stage':0,'training_runs':0,
             'scope':'Source factory pilot, concentrated in a few Boston logs; not representative generalization evaluation.',
             'count_boundary':'50 is the next synthesis target, not 50 approved cases; pair feasibility depends on instance masks and grounded placement.',
             'failure_ledger_refs':['V77-F02'],'decisions':decisions}
    dump(root/'source_rgb_missing.json',missing_sources)
    dump(root/'cpu_closeout.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='decisions'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
