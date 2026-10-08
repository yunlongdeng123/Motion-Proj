"""r51：追加真实单车 DELETE 输入，复用 r50 的固定 SAM/官方推理。"""
from pathlib import Path
import argparse
import collections
import html
import json
import os
import random
import sys

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
REPO = Path('/root/autodl-tmp/motion_proj_v77')
BASE = REPO / 'scripts/worldsim_v77/target_protected/iteration17'
sys.path.insert(0, str(BASE))
import common

OLD = common.O
O = common.T / 'r51'
common.O = O
from common import read, dump, TASK
sys.path.insert(0, str(BASE))

SEED = 771051
COUNT = 40


def page():
    m = read(O / 'manifest.json')
    root = O / 'review'
    root.mkdir(exist_ok=True)
    cards = []
    for c in m['cases']:
        cid = c['case_id']
        q = c.get('input_quality_review')
        score = f"{q['score']:.1f}" if q and q['score'] is not None else '待检查'
        reason = q['concise_reason'] if q else '尚未做独立图像质检，不能进入 GPU。'
        b = q['protected_reference_reason'] if q else '后车框只为几何候选，不保证实际可见或完整。'
        columns = []
        for name, title in [('original', '原视频：黄框删除 A / 绿框后车 B'),
                            ('model_input', '完整 SAM 与实际生成洞'),
                            ('native', '官方权重原生 DELETE'),
                            ('delete', '固定规则最终写回')]:
            p = root / cid / f'{name}.mp4'
            content = (f'<video controls muted preload="none" src="{cid}/{name}.mp4"></video>'
                       if p.exists() else '<p class="pending">未运行。当前没有生成结果。</p>')
            columns.append(f'<div><b>{title}</b>{content}</div>')
        refs = (f'<a href="{cid}/protected_reference.jpg">真实后车参考</a>'
                if (root / cid / 'protected_reference.jpg').exists() else '没有合格的后车参考候选')
        cards.append(f'''<article id="{cid}"><h2>{cid} · {c['scene']} · {c['camera']}</h2>
<p>输入质检 {score}：{html.escape(reason)}</p><p>后车证据：{html.escape(b)} · {refs}</p>
<p>目标实例 <code>{c['instance_token']}</code>；prompt f{c['prompt_frame']:02}。
每例恢复独立原视频，只删除这一辆车；绿框是保留对象的粗投影。</p>
<button onclick="sync(this)">同步播放 / 重头播放</button><div class="videos">{''.join(columns)}</div>
<details><summary>f00 / f05 / f09 原图与目标 crop</summary>
<img loading="lazy" src="{cid}/inputs_f00_f05_f09.jpg"></details></article>''')
    scenes = len({c['scene'] for c in m['cases']})
    exclusions = ''
    if (O / 'quality_exclusions.json').exists():
        rejected = read(O / 'quality_exclusions.json')['cases']
        reasons = ''.join(f"<li>{c['case_id']} · {c['scene']}：{html.escape(c['input_quality_review']['concise_reason'])}</li>" for c in rejected)
        exclusions = f'<details><summary>{len(rejected)} 例退队列：不运行 GPU</summary><ul>{reasons}</ul></details>'
    body = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>r51 真实 DELETE 新输入</title>
<style>body{{font:16px/1.65 system-ui;background:#111925;color:#e7edf6;max-width:1800px;margin:25px auto;padding:15px}}a{{color:#87ceff}}article{{background:#1b2738;padding:18px;margin:24px 0}}.videos{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}}video,img{{width:100%}}.pending{{color:#a5b6c8;padding:36px 8px}}button{{margin:8px 0;padding:8px}}.flow{{display:flex;flex-wrap:wrap;align-items:center;gap:12px;padding:20px;background:#25354b}}.flow span{{border:1px solid #6d8fbe;padding:8px}}@media(max-width:1100px){{.videos{{grid-template-columns:repeat(2,1fr)}}}}</style>
<h1>r51 · {scenes} 个新场景 / {len(m['cases'])} 个单车 DELETE 输入</h1>
<p>阶段：{html.escape(m['phase'])}。本页当前是 CPU 准备页；原生输出和最终写回未运行时明确留空。</p>
<div class="flow"><span>nuScenes train 真实 RGB</span>→<span>清晰单车输入质检</span>→<span>完整 SAM / 身份检查</span>→<span>冻结官方 DriveEditor</span>→<span>原生生成 ∥ 最终写回</span></div>
<p>新场景与 r50 审核过的全部 38 景分离，旧 5 景隔离集不动。既有 RGB 缓存受先前采样影响，本批用于开发排查，不能当最终泛化评测。</p>
<p>固定 r46 官方原权重 + r21 sam_full_v2，seed42 / 25steps / 10帧 / 1024×576。
没有 Adapter、训练或逐例调参。生成失败与写回损伤分开；只有入口合格且原生已失败的案例才候选匹配造数。训练真值始终使用真实视频。</p>
<p><a href="manifest.json">准入清单</a> · <a href="sampling.json">采样规则</a>
 · <a href="input_quality_review.json">独立输入质检</a></p>{exclusions}
{''.join(cards)}
<script>function sync(b){{const v=[...b.closest('article').querySelectorAll('video')];for(const x of v){{x.currentTime=0;x.play().catch(()=>{{}});}}}}</script></html>'''
    (root / 'index.html').write_text(body, encoding='utf-8')
    for name in ['manifest.json', 'sampling.json', 'input_quality_review.json', 'cpu_check.json', 'quality_exclusions.json', 'r50_failure_split_user.json']:
        if (O / name).exists():
            dump(root / name, read(O / name))


def stage():
    import prepare as prior_prepare
    import review as prior_review
    if (O / 'input_candidates.json').exists():
        print('ALREADY_STAGED; no duplicate candidates', flush=True)
        return
    state = read(OLD / 'CPU_selection_geometry.json')
    old_cases = read(OLD / 'quality_candidates.json')['cases']
    seen = {c['scene'] for c in old_cases}
    reserve = set(state['reserve'])
    pool = [c for c in state['pool'] if c['scene'] not in seen | reserve]
    by_scene = collections.defaultdict(list)
    for c in pool:
        by_scene[c['scene']].append(c)
    # 优先较大、清楚的目标，含后车布局仅作优先级，不作为遮挡真值。
    rows = [sorted(cs, key=lambda c: (c['median_width'] < 140 or c['median_height'] < 65,
                                     not c['behind_vehicle_proxy'], -c['median_area_fraction'],
                                     c['source_id'], c['instance_token']))[0]
            for cs in by_scene.values()]
    random.Random(SEED).shuffle(rows)
    rows.sort(key=lambda c: (c['median_width'] < 140 or c['median_height'] < 65,
                             not c['behind_vehicle_proxy'], -c['median_area_fraction']))
    chosen = rows[:COUNT]
    assert len(chosen) == COUNT
    assert len({c['scene'] for c in chosen}) == COUNT
    sources = {s['source_id']: s for s in state['sources']}
    sampling = {'seed': SEED, 'population': 'r50 cached RGB/SDK geometry; no output used for selection',
        'unused_scene_candidate_count': len(pool), 'unused_scene_count': len(by_scene),
        'candidate_count': COUNT, 'candidate_scenes': [c['scene'] for c in chosen],
        'excluded_r50_reviewed_scenes': sorted(seen), 'reserve_scenes': sorted(reserve),
        'rule': 'one distinct actor per new scene; prefer median width>=140/height>=65, then behind-car proxy and size; original r50 geometry gates retained; independent RGB quality>=2 required',
        'remaining_scene_candidates': [c for c in rows[COUNT:]], 'hidden_GT': None}
    O.mkdir(parents=True, exist_ok=True)
    dump(O / 'sampling.json', sampling)
    dump(O / 'controller_state.json', {'stage': 'CPU_sampling', 'pid': os.getpid(), 'GPU_jobs': 0, 'training_steps': 0})
    start_id = max(int(c['case_id'][1:]) for c in old_cases) + 1
    cases = []
    for i, c in enumerate(chosen):
        cid = f'R{start_id+i:03}'
        row = prior_prepare.materialize_case(c, sources[c['source_id']], cid)
        row.update(structural_audit_eligible=False, input_quality_score=None,
                   input_visual_review='pending independent image review')
        dump(Path(row['folder']) / 'case.json', row)
        prior_review.generate_assets(row, cpu_only=True)
        cases.append(row)
        print('INPUT', cid, row['scene'], flush=True)
    m = read(OLD / 'manifest.json')
    for name in ['eligible_cases', 'eligible_scenes', 'input_excluded_cases', 'quality_uncertain_cases']:
        m.pop(name, None)
    m.update(run_id='r51', cases=cases, case_count=len(cases), scene_count=len(cases),
        phase='CPU candidate input review pending', input_candidate_count=len(cases),
        input_reviewed_scene_count=0, max_new_windows=len(cases),
        quality_score_rule='independent score>=2.0 admitted; lower/uncertain rejected; one target/scene',
        training_steps=0, adapter=False, temporal_module_change=False,
        selection_rule=sampling['rule'], input_quality_scope='pending independent original RGB review',
        stop='CPU preparation stops before GPU; SAM identities checked before fixed DELETE; no automatic training',
        failure_ledger_delta='none; new model outputs not generated')
    dump(O / 'input_candidates.json', {'cases': cases})
    dump(O / 'manifest.json', m)
    dump(O / 'controller_state.json', {'stage': 'CPU_inputs_ready_pending_independent_review',
        'cases': len(cases), 'scenes': len(cases), 'GPU_jobs': 0, 'training_steps': 0})
    page()
    print('CPU_CANDIDATES', len(cases), flush=True)


def apply_quality(path):
    q = read(path)
    cases = read(O / 'input_candidates.json')['cases']
    qs = {c['case_id']: c for c in q['cases']}
    assert len(qs) == len(q['cases']) and set(qs) == {c['case_id'] for c in cases}
    approved = []
    for c in cases:
        r = qs[c['case_id']]
        passed = r['score'] is not None and r['score'] >= 2 and r['status'] == 'pass'
        assert passed or r['status'] in ['reject', 'uncertain']
        c.update(input_quality_review=r, input_quality_score=2 if passed else r['score'],
            structural_audit_eligible=passed, input_visual_review=r['concise_reason'],
            B_evidence_status=r['protected_reference_quality'], human_verdict=None)
        dump(Path(c['folder']) / 'case.json', c)
        if passed:
            approved.append(c)
    assert 20 <= len(approved) <= 40, ('不放宽门槛，需补候选', len(approved))
    m = read(O / 'manifest.json')
    m.update(cases=approved, case_count=len(approved), scene_count=len(approved),
        eligible_cases=len(approved), eligible_scenes=len(approved),
        input_reviewed_scene_count=len(cases), max_new_windows=len(approved),
        quality_score_rule='independent score>=2.0 admitted; lower/uncertain rejected; one target/scene',
        phase='CPU ready; waiting user GPU', input_quality_scope=q['scope'],
        input_excluded_cases=[c['case_id'] for c in cases if c not in approved])
    dump(O / 'manifest.json', m)
    dump(O / 'input_candidates.json', {'cases': cases})
    dump(O / 'input_quality_review.json', q)
    dump(O / 'quality_exclusions.json', {'cases': [{k:c[k] for k in ['case_id','scene','input_quality_review']} for c in cases if c not in approved]})
    check_cpu()
    page()


def check_cpu():
    import cv2
    import numpy as np
    from PIL import Image
    from nuscenes.utils.splits import train
    cv2.setNumThreads(1)
    m = read(O / 'manifest.json')
    s = read(O / 'sampling.json')
    cs = m['cases']
    assert 20 <= len(cs) <= 40 and len({c['scene'] for c in cs}) == len(cs)
    assert {c['scene'] for c in cs}.issubset(train)
    assert not {c['scene'] for c in cs} & set(s['excluded_r50_reviewed_scenes'] + s['reserve_scenes'])
    assert all(c['input_quality_score'] == 2 and c['structural_audit_eligible'] for c in cs)
    assert all(Path(m[k]).is_file() for k in ['model_checkpoint', 'SAM_checkpoint'])
    decoded = 0
    for c in cs:
        assert len({f['sample_data_token'] for f in c['frames']}) == 10
        dt = np.diff([f['timestamp'] for f in c['frames']]) / 1e6
        assert np.all((dt >= .04) & (dt <= .17)) and .08 <= np.median(dt) <= .12
        for f in c['frames']:
            assert f['target']['instance_token'] == c['instance_token']
            if f['is_key_frame']:
                assert f['pose_source'] == 'SDK associated keyframe'
            with Image.open(Path(c['folder']) / 'rgb' / f"{f['frame']:05}.jpg") as im:
                assert im.size == (1024, 576)
        v = cv2.VideoCapture(str(O / 'review' / c['case_id'] / 'original.mp4'))
        count = 0
        while True:
            ok, im = v.read()
            if not ok:
                break
            assert im.shape == (576, 1024, 3)
            count += 1
        v.release()
        assert count == 10
        decoded += count
    result = {'task_id': TASK, 'run_id': 'r51', 'CPU_ready': True, 'case_count': len(cs),
        'scene_count': len(cs), 'new_vs_all_r50_reviewed_scenes': True,
        'independent_input_quality_only_2plus': True, 'actual_video_decoded_frames': decoded,
        'checkpoint_exists': True, 'GPU_jobs': 0, 'training_steps': 0, 'human_verdict': None}
    dump(O / 'cpu_check.json', result)
    dump(O / 'controller_state.json', dict(result, stage='CPU_ready_waiting_user_GPU'))
    print('CPU_READY', len(cs), len(cs), decoded, flush=True)


def gpu_phase(phase):
    import gpu as prior_gpu
    # 模型与合成代码完全复用 r50；实例 gate 文件仍是 DELETE 的必需输入。
    prior_gpu.O = O
    lock = open(O / 'GPU.lock', 'a')
    import fcntl
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (prior_gpu.masks if phase == 'masks' else prior_gpu.delete)()


def assets():
    import review as prior_review
    for c in read(O / 'manifest.json')['cases']:
        prior_review.generate_assets(c, cpu_only=False)
    page()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['stage', 'apply-quality', 'check', 'masks', 'delete', 'assets'])
    p.add_argument('--review', type=Path)
    a = p.parse_args()
    if a.phase == 'stage':
        stage()
    elif a.phase == 'apply-quality':
        assert a.review is not None
        apply_quality(a.review)
    elif a.phase == 'check':
        check_cpu()
    elif a.phase in ['masks', 'delete']:
        gpu_phase(a.phase)
    else:
        assets()
