"""独立输入质检后补齐20景；拒绝只退队列，原始资料和评分保留。"""
from common import *
import argparse
from collections import defaultdict


def all_candidates():
    cases = read(O/'manifest.json')['cases']
    extra = O/'quality_candidates.json'
    if extra.exists():
        seen = {c['case_id'] for c in cases}
        cases += [c for c in read(extra)['cases'] if c['case_id'] not in seen]
    return cases


def stage(request):
    # common 为旧基线入口扩展搜索路径；本轮同名 prepare 必须优先本目录。
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from prepare import materialize_case
    from review import generate_assets
    state = read(O/'CPU_selection_geometry.json')
    sources = {s['source_id']: s for s in state['sources']}
    cases = all_candidates()
    used = {(c['scene'], c['instance_token']) for c in cases}
    next_id = max(int(c['case_id'][1:]) for c in cases) + 1
    new = []
    for c in read(request)['candidates']:
        assert c in state['pool'] and c['scene'] not in state['reserve']
        key = (c['scene'], c['instance_token'])
        assert key not in used, ('重复车辆实例', key)
        cid = f'R{next_id:03}'; next_id += 1; used.add(key)
        row = materialize_case(c, sources[c['source_id']], cid)
        row.update(structural_audit_eligible=False, input_visual_review='pending independent review')
        dump(Path(row['folder'])/'case.json', row)
        generate_assets(row, cpu_only=True)
        cases.append(row); new.append(row)
        dump(O/'quality_candidates.json', {'cases': cases, 'GPU_jobs': 0, 'training_steps': 0})
        print('QA_CANDIDATE', cid, row['scene'], flush=True)
    dump(O/'quality_batch.json', {'case_ids': [c['case_id'] for c in new],
        'cases': [{k: c[k] for k in ['case_id','scene','camera','instance_token']} for c in new],
        'scope': 'real input RGB quality only; f00/f05/f09; no masks or generated outputs'})


def apply_review(review_path):
    review = read(review_path)
    reviews = {r['case_id']: r for r in review['cases']}
    cases = all_candidates()
    assert len(reviews) == len(review['cases'])
    assert set(reviews) == {c['case_id'] for c in cases}, '所有候选必须独立复核'
    by_scene = defaultdict(list)
    for c in cases:
        r = reviews[c['case_id']]
        assert r['status'] in ['pass','reject','uncertain']
        assert (r['status'] == 'pass') == (r['score'] == 2)
        assert r['status'] != 'reject' or r['score'] in [0,1]
        c.update(input_quality_review=r, input_quality_score=r['score'],
            input_visual_review=r['concise_reason'],
            B_evidence_status=r['protected_reference_quality'], human_verdict=None)
        if r['status'] == 'pass': by_scene[c['scene']].append(c)
    old = read(O/'manifest.json')
    original_order = read(O/'CPU_selection_geometry.json')['scenes']
    scene_order = list(dict.fromkeys(original_order + [c['scene'] for c in cases]))
    qualified = [s for s in scene_order if len(by_scene[s]) >= 2]
    assert len(qualified) >= 20, ('尚未补足20个各含2个2分车辆的场景', len(qualified))
    scenes = qualified[:20]
    selected = [c for s in scenes for c in by_scene[s][:2]]
    selected += [by_scene[s][2] for s in scenes if len(by_scene[s]) > 2][:10]
    admitted = {c['case_id'] for c in selected}
    for c in cases:
        c['structural_audit_eligible'] = c['case_id'] in admitted
        c['quality_queue_role'] = ('admitted' if c['case_id'] in admitted else
            'uncertain' if reviews[c['case_id']]['status'] == 'uncertain' else
            'rejected' if reviews[c['case_id']]['status'] == 'reject' else 'qualified_standby')
        dump(Path(c['folder'])/'case.json', c)
    # manifest 只含本次准入队列；历史样本保存于独立归档，不让旧拒绝例进入GPU。
    archive = [c for c in cases if c['case_id'] not in admitted]
    rejected = [c['case_id'] for c in archive if c['quality_queue_role'] == 'rejected']
    uncertain = [c['case_id'] for c in archive if c['quality_queue_role'] == 'uncertain']
    old.update(cases=selected, scene_count=20, case_count=len(selected),
        eligible_scenes=20, eligible_cases=len(selected),
        input_excluded_cases=rejected, quality_uncertain_cases=uncertain,
        input_quality_scope='independent gpt-6-sol xhigh, no fast; actual RGB f00/f05/f09 and available B reference; not mask or output review',
        input_candidate_count=len(cases), input_reviewed_scene_count=len({c['scene'] for c in cases}),
        quality_score_rule='2 admitted;0/1 rejected;uncertain never admitted;max3 targets/scene')
    assert all(c['input_quality_score'] == 2 for c in selected)
    assert not set(scenes) & set(old['reserve_scenes'])
    dump(O/'quality_candidates.json', {'cases': cases, 'GPU_jobs':0, 'training_steps':0})
    dump(O/'input_quality_review.json', review)
    dump(O/'quality_archive.json', {'cases':archive, 'rejected':rejected,'uncertain':uncertain,
        'policy':'remove from active queue; original data and independent scores retained'})
    dump(O/'input_visual_review.json', {'reviewer':review['reviewer'],
        'scope':old['input_quality_scope'], 'cases':{c['case_id']:reviews[c['case_id']] for c in selected},
        'candidate_count':len(cases), 'rejected':rejected, 'uncertain':uncertain, 'human_verdict':None})
    dump(O/'manifest.json', old)
    sampling = read(O/'sampling.json')
    sampling.update(DEV_scenes=scenes, input_quality_score_rule=old['quality_score_rule'],
        active_case_ids=[c['case_id'] for c in selected], independent_reviewed_candidates=len(cases),
        input_quality_scope=old['input_quality_scope'])
    dump(O/'sampling.json', sampling)
    dump(O/'controller_state.json', {'stage':'CPU_quality_complete_pending_delivery',
        'cases':len(selected),'scenes':20,'GPU_jobs':0,'training_steps':0,
        'reviewed_candidates':len(cases),'rejected':rejected,'uncertain':uncertain,'human_verdict':None})
    print('QUALITY_READY', len(selected), 20, 'rejected',len(rejected),'uncertain',len(uncertain),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=['stage','apply'])
    p.add_argument('input', type=Path); a = p.parse_args()
    (stage if a.phase == 'stage' else apply_review)(a.input)
