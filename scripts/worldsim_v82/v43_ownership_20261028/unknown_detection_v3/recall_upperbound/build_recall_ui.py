"""从正式三帧产物构建独立召回上界报告；保留旧实验，数字只读真实receipt。"""
from __future__ import annotations
import argparse
import json
import re
import shutil
import struct
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]
OUT = BASE / 'outputs/V43_Ownership_20261027/gpu_v43/recall_ceiling_v1'
STAGES = ('actor_omitted', 'existing', 'gates_open_same_domain', 'broad_once', 'peel_01')
RAWS = (98, 113, 114)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def png(path):
    with path.open('rb') as f:
        head = f.read(24)
    assert head[:8] == b'\x89PNG\r\n\x1a\n' and struct.unpack('>II', head[16:24]) == (480, 320), path


def architecture():
    groups = [
        ('原观测与冻结输入', ['RGB / LiDAR', 'static / Actor', 'EnvMap']),
        ('固定目标查询域', ['原body与旧域', '车底effect', '按body扩域']),
        ('原生召回干预', ['同域门槛控制', '初始支撑剥离', 'after重算补删']),
        ('分量与边界核对', ['高斯 G / alpha', '透射 × 环境', '闭合与来源'])
    ]
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 240" width="1600" height="240"><rect width="1600" height="240" rx="12" fill="#eef4f8"/>']
    for i, (title, components) in enumerate(groups):
        x = 18 + i * 397
        svg.append(f'<rect x="{x}" y="22" width="370" height="196" rx="10" fill="white" stroke="#91adbf"/><text x="{x+185}" y="57" text-anchor="middle" font-family="Microsoft YaHei,sans-serif" font-size="21" fill="#183547">{title}</text>')
        for j, component in enumerate(components):
            xx = x + 12 + j * 116
            fill = '#f3eafb' if i == 1 and j == 1 else '#e9f1f7'
            svg.append(f'<rect x="{xx}" y="91" width="105" height="80" rx="6" fill="{fill}" stroke="#b6c9d6"/><text x="{xx+52.5}" y="137" text-anchor="middle" font-family="Microsoft YaHei,sans-serif" font-size="14" fill="#24485e">{component}</text>')
            if j < 2:
                svg.append(f'<path d="M{xx+106} 131h3" stroke="#3e6d8c"/><polygon points="{xx+109},127 {xx+114},131 {xx+109},135" fill="#3e6d8c"/>')
        if i < 3:
            svg.append(f'<path d="M{x+372} 121h15" stroke="#3e6d8c" stroke-width="2"/><polygon points="{x+387},116 {x+395},121 {x+387},126" fill="#3e6d8c"/>')
    svg.append('</svg>')
    return ''.join(svg)


def block(source, name, body, insertion):
    start, end = f'<!-- {name}-start -->', f'<!-- {name}-end -->'
    value = start + body + end
    if start in source:
        return re.sub(re.escape(start) + '.*?' + re.escape(end), lambda _: value, source, count=1, flags=re.S)
    assert insertion in source, name
    return source.replace(insertion, insertion + value, 1)


def build(source, gate_env, output, attach=True, ledger=None, dark=None):
    summary = read(source / 'summary.json')
    assert summary['schema'] == 'target-recall-ceiling-v1'
    assert summary['checkpoint_frozen'] and summary['training_updates'] == 0
    assert summary['semantic_labels_modified'] is False and summary['environment_modified'] is False
    assert summary['actor_checkpoint_and_tensors_unchanged'] and summary['initial_weights_dense']
    data = dict(schema='recall-ceiling-ui-v1', raws=list(RAWS), channels=summary['channels'], frames={}, environment_source=None, ledger_source=None, dark_source=None, provenance=dict(run=str(source.resolve()), summary='evidence/summary.json', state_population=15, intervention_only=True, independent_semantic_recall=False, transfer_validated=False, background_damage_is_logged_cost=True))
    planned = []
    validation = dict(schema='recall-ceiling-ui-validation-v1', source=str(source.resolve()), frame_count=3, stage_count=5, real_images=0, receipts=0, full_row_namespace=summary['static_row_namespace'], static_rows=summary['static_rows'], checkpoint_frozen=True, training_updates=0, checks=[], browserQA=False, human_verdict=None)
    validation['browser_verification_used_for_acceptance'] = False
    if (output / 'ui_validation.json').exists():
        previous_validation = read(output / 'ui_validation.json')
        if 'excluded_browser_attempt' in previous_validation:
            validation['excluded_browser_attempt'] = previous_validation['excluded_browser_attempt']
    for raw in RAWS:
        directory = source / f'{raw:03d}'
        frame = dict(images={}, stages={}, final=read(directory / 'final.json'), final_path=f'evidence/{raw}/final.json')
        planned.append((directory / 'final.json', frame['final_path']))
        domains = np.load(directory / 'domains.npz')['maps'].astype(bool)
        environment = np.load(directory / 'environment.npz')
        env, bg = environment['rgb'], environment['bg']
        assert env.shape == (3, 320, 480) and bg.shape == (3, 1, 1)
        frame['environment_stats'] = dict(minimum=float(env.min()), maximum=float(env.max()), source='environment.npz / rgb / float32')
        for name, file in [('original', 'original.png'), ('domain', 'domain_overlay.png'), ('environment_source', 'environment_display.png')]:
            path = directory / file; png(path)
            frame['images'][name] = f'assets/{raw}/{file}'; planned.append((path, frame['images'][name])); validation['real_images'] += 1
        previous_removed = set()
        for stage in STAGES:
            path = directory / stage
            receipt = read(path / 'receipt.json')
            assert receipt['raw'] == raw and receipt['stage'] == stage
            assert receipt['environment_unmodified'] and receipt['background_veto'] is False and receipt['labels_for_training'] is False
            assert receipt['actor_removed'] == 6241
            arrays_path = path / ('float32maps.npz' if (path / 'float32maps.npz').exists() else 'maps.npz')
            maps = np.load(arrays_path)
            alpha, gaussian, env_term, native = [maps[k] for k in ('alpha', 'gaussian_rgb', 'environment_rgb', 'native_rgb')]
            assert all(a.dtype == np.float32 for a in (alpha, gaussian, env_term, native))
            assert alpha.shape == (320, 480) and gaussian.shape == env_term.shape == native.shape == (3, 320, 480)
            closure = float(np.max(np.abs(gaussian + (1-alpha)[None] * bg + env_term - native)))
            assert closure < 2e-5, (raw, stage, closure)
            env_closure = float(np.max(np.abs(env_term - (1-alpha)[None]*env)))
            assert env_closure < 2e-5
            contributors = np.load(path / 'contributors.npz')
            ids, weights, removed = contributors['ids'], contributors['weights'], contributors['removed_ids']
            assert len(ids) == len(weights) == len(contributors['colors']) and weights.shape[1] == 6
            assert np.all(np.diff(ids) > 0) and np.all((ids >= 0) & (ids < 918850))
            removed_set = set(map(int, removed)); assert not removed_set.intersection(map(int, ids))
            assert previous_removed.issubset(removed_set), (raw, stage, 'non-monotonic removal')
            previous_removed = removed_set
            assert len(removed) == receipt['removed_static']
            assert int((weights[:, 1] > summary['mass_epsilon']).sum()) == receipt['contributing_static_in_broad']
            for j, name in enumerate(summary['channels']):
                mask = domains[j]
                assert abs(float(alpha[mask].mean(dtype=np.float64)) - receipt['regions'][name]['alpha_mean']) < 2e-6
                assert int(((env_term < 0).any(0) & mask).sum()) == receipt['regions'][name]['negative_environment_pixels']
            item = dict(receipt=receipt, receipt_path=f'evidence/{raw}/{stage}/receipt.json', images={}, float32_stats=dict(native_minimum=float(native.min()), native_maximum=float(native.max()), clipped_native_pixels=int(((native<0)|(native>1)).any(0).sum()), source=str(arrays_path.name)))
            planned.append((path / 'receipt.json', item['receipt_path']))
            for name, file in [('native', 'native.png'), ('gaussian', 'gaussian_only.png'), ('environment', 'environment_term.png'), ('alpha', 'alpha.png')]:
                image = path / file; png(image); relative = f'assets/{raw}/{stage}/{file}'
                planned.append((image, relative)); item['images'][name] = relative; validation['real_images'] += 1
            frame['stages'][stage] = item; validation['receipts'] += 1
            validation['checks'].append(dict(raw=raw, stage=stage, removed_static=len(removed), after_active_rows=len(ids), component_max_abs_error=closure, environment_max_abs_error=env_closure, body_alpha=receipt['regions']['body']['alpha_mean']))
        final = frame['stages'][frame['final']['final_stage']]['receipt']
        assert final['regions']['body']['alpha_mean'] == 0 and final['regions']['effect']['alpha_mean'] == 0 and final['regions']['broad_domain']['alpha_mean'] == 0
        data['frames'][str(raw)] = frame
    planned.append((source / 'summary.json', 'evidence/summary.json'))
    if (source / 'recall_visual_review.json').exists():
        review = read(source / 'recall_visual_review.json')
        assert review['human_verdict'] is None and review['ground_truth'] is False
        data['visual_review'] = review
        planned.append((source / 'recall_visual_review.json', 'evidence/ai_visual_review.json'))
    if gate_env is not None:
        env_summary = read(gate_env / 'summary.json')
        assert env_summary['schema'] == 'recall-gate-and-environment-source-v1'
        assert env_summary['source_checkpoint_unchanged'] and env_summary['environment_modified'] is False and env_summary['training_updates'] == 0
        assert {r['raw'] for r in env_summary['receipts']} == set(RAWS)
        assert max(r['environment_integral_relative_error'] for r in env_summary['receipts']) < 1e-5
        data['environment_source'] = env_summary
        planned.append((gate_env / 'summary.json', 'evidence/environment_source_summary.json'))
        data['provenance']['environment_source'] = str(gate_env.resolve())
    validation['core_png_images'] = validation['real_images']
    if dark is not None:
        dark_summary = read(dark / 'summary.json')
        assert dark_summary['training_updates'] == 0 and dark_summary['source_checkpoint_unchanged'] and dark_summary['used_to_generate_labels'] is False
        assert len(dark_summary['records']) == 12
        data['dark_source'] = dark_summary
        planned.append((dark / 'summary.json', 'evidence/dark/summary.json'))
        for record in dark_summary['records']:
            raw, stage = record['raw'], record['stage']
            assert record['diagnostic_masks_from_post_render'] and record['labels_or_proposals_modified'] is False and record['darkness_is_not_vehicle_ground_truth']
            original = dark / f'{raw:03d}' / stage
            assert read(original / 'receipt.json') == record
            item = dict(receipt=record, receipt_path=f'evidence/dark/{raw}/{stage}.json', mask_path=f'assets/{raw}/{stage}/diagnostic_dark_mask.png')
            planned.append((original / 'receipt.json', item['receipt_path']))
            png(original / 'diagnostic_dark_mask.png'); planned.append((original / 'diagnostic_dark_mask.png', item['mask_path']))
            data['frames'][str(raw)]['stages'][stage]['dark_diagnostic'] = item
            if stage == 'peel_01':
                for region in record['regions'].values():
                    assert region['alpha_mass'] == 0 and np.count_nonzero(region['gaussian_rgb_integral']) == 0
                    assert np.array_equal(region['native_rgb_integral'], region['environment_rgb_integral'])
        validation.update(dark_diagnostic_records=12, diagnostic_mask_pngs=12, final_dark_integrals_equal_environment=True)
        validation['real_images'] += 12
        data['provenance']['dark_diagnostic_source'] = str(dark.resolve())
    if ledger is not None:
        ledger_summary = read(ledger / 'summary.json'); verify = read(ledger / 'verification.json')
        assert ledger_summary['schema'] == 'recall-upperbound-rule-miss-ledger-v1' and verify['all_invariants_passed']
        assert ledger_summary['background_veto'] is False and ledger_summary['labels_for_training'] is False
        mass = read(ledger / 'gate_relaxation_mass_audit.json'); visibility = read(ledger / 'visibility_by_roi.json')
        data['ledger_source'] = dict(summary=ledger_summary, verification=verify)
        qindex = {q['raw']: q for q in ledger_summary['queries']}
        mindex = {q['raw']: {s['stage']: s for s in q['stages']} for q in mass['queries']}
        vindex = {q['raw']: {s['stage']: s for s in q['stages']} for q in visibility['queries']}
        for original in ledger.rglob('*'):
            if original.is_file() and original.suffix in ('.json', '.npz'):
                planned.append((original, 'evidence/ledger/' + original.relative_to(ledger).as_posix()))
        for raw in RAWS:
            frame = data['frames'][str(raw)]; frame['relaxations_path'] = f'evidence/ledger/{raw:03d}/rule_relaxations.npz'
            for record in qindex[raw]['stages']:
                stage = record['stage']; item = frame['stages'][stage]
                assert record['remaining_broad_contributors'] == item['receipt']['contributing_static_in_broad']
                rows = np.load(ledger / f'{raw:03d}' / f'{stage}_misses.npz', allow_pickle=False)
                ids, after, before, bits = rows['ids'], rows['after_weights'], rows['initial_weights'], rows['category_bits']
                assert len(ids) == record['remaining_broad_contributors'] and np.all((ids >= 0) & (ids < 918850))
                chosen = set()
                for flag in (0, 1, 2, 4, 8, 16):
                    positions = np.arange(len(ids)) if not flag else np.flatnonzero(bits & flag)
                    chosen.update(map(int, positions[np.argsort(after[positions, 1])[::-1][:25]]))
                examples = []
                names = rows['gate_names'].tolist()
                for index in sorted(chosen, key=lambda i: float(after[i, 1]), reverse=True):
                    states = rows['gate_states'][index]
                    examples.append(dict(row=int(ids[index]), category_bits=int(bits[index]), initial_broad_mass=float(before[index, 1]), after_broad_mass=float(after[index, 1]), initial_weights=before[index].tolist(), after_weights=after[index].tolist(), weight_channels=summary['channels'], original_selected=bool(rows['original_selected'][index]), removed=bool(rows['removed'][index]), failed_gates=[name for name, value in zip(names, states) if value == 0], unknown_gates=[name for name, value in zip(names, states) if value < 0], gate_states={name: int(value) for name, value in zip(names, states)}))
                item.update(ledger=record, gate_mass=mindex[raw][stage], visibility=vindex[raw][stage], row_examples=examples, miss_rows_path=f'evidence/ledger/{raw:03d}/{stage}_misses.npz')
        validation.update(ledger_stage_checks=15, ledger_invariants_passed=True, ledger_rows_include_background=True)
        data['provenance']['rule_miss_ledger_source'] = str(ledger.resolve())
    output.mkdir(parents=True, exist_ok=True)
    for original, relative in planned:
        target = output / relative; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(original, target)
    (output / 'data.js').write_text('window.RECALL_CEILING_DATA=' + json.dumps(data, ensure_ascii=False, separators=(',', ':')) + ';\n', encoding='utf8')
    shutil.copy2(HERE / 'recall_ceiling.html', output / 'index.html')
    shutil.copy2(HERE / 'recall_ceiling_viewer.js', output / 'viewer.js')
    (output / 'architecture.svg').write_text(architecture(), encoding='utf8')
    (output / 'ui_validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding='utf8')
    if attach:
        shared = BASE / 'outputs/GS_Scene_Diagnostics_20261027/index.html'
        scene = shared.parent / '14705303724557273004_3105_000_3125_000/index.html'
        links = [
            (shared, '<main id="workspace">', '<div class="notice"><strong>新阶段 · 召回优先：</strong><a href="../V43_Ownership_20261027/gpu_v43/recall_ceiling_v1/index.html">三帧召回上界与高斯／环境分解 →</a>　道路误选只记录；高斯消隐与语义召回、环境影身份分别判断。</div>'),
            (scene, '<main>', '<section class="notice"><strong>新阶段 · 召回优先：</strong><a href="../../V43_Ownership_20261027/gpu_v43/recall_ceiling_v1/index.html">查看三帧实际原生RGB、高斯与环境分量 →</a>　此前“双目标未通过”记录保留；本轮道路误选作为成本，不否决召回上界。</section>'),
            (BASE / 'outputs/V43_Ownership_20261027/index.html', '</nav>', '<div class="notice"><a href="gpu_v43/recall_ceiling_v1/index.html">新增：召回优先上界、after真实贡献与高斯／环境来源分解 →</a></div>'),
            (BASE / 'outputs/V43_Ownership_20261027/gpu_v43/index.html', '</nav>', '<div class="notice"><a href="recall_ceiling_v1/index.html">新阶段：三帧召回上界与原环境归因 →</a></div>')
        ]
        for file, insertion, html in links:
            if file.exists():
                text = file.read_text(encoding='utf8')
                if insertion not in text and '<main>' in text: insertion = '<main>'
                if insertion not in text: insertion = '</h1>'
                file.write_text(block(text, 'recall-ceiling-link', html, insertion), encoding='utf8')
        old = BASE / 'outputs/V43_Ownership_20261027/gpu_v43/target_detection_v3/index.html'
        if old.exists():
            text = old.read_text(encoding='utf8')
            note = '<div class="card warn"><strong>历史阶段｜此前双目标验收</strong><p>本页“背景伤害导致未通过”对应此前“完整去影并保护背景”的双目标合同，旧数字和失败记录保留。用户现已改为召回优先：允许更宽unknown，道路误选只记录为代价。<a href="../recall_ceiling_v1/index.html"><strong>查看新召回上界及高斯／环境分解报告 →</strong></a></p></div>'
            old.write_text(block(text, 'recall-history-note', note, '<main>'), encoding='utf8')
    return validation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=HERE / 'recall_ceiling_r1')
    parser.add_argument('--gate-env', type=Path, default=HERE / 'recall_gate_env_r1')
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--ledger', type=Path, default=HERE / 'ledger_r1')
    parser.add_argument('--dark', type=Path, default=HERE / 'recall_dark_sources_r1')
    parser.add_argument('--no-attach', action='store_true')
    args = parser.parse_args()
    result = build(args.source.resolve(), args.gate_env.resolve() if args.gate_env.exists() else None, args.output.resolve(), not args.no_attach, args.ledger.resolve() if args.ledger.exists() else None, args.dark.resolve() if args.dark.exists() else None)
    print(json.dumps({key: value for key, value in result.items() if key not in ('checks', 'excluded_browser_attempt')}, ensure_ascii=False, indent=2))
