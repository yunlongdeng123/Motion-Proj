"""将已完成的1920采集栅格源质量核查原位接入原报告。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

from PIL import Image

HERE = Path(__file__).resolve().parent
BEGIN = '<!-- SOURCE_QUALITY_AUDIT_BEGIN -->'
END = '<!-- SOURCE_QUALITY_AUDIT_END -->'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def embedded_data(html):
    match = re.search(r'\bconst\s+D\s*=\s*', html)
    data, n = json.JSONDecoder().raw_decode(html[match.end():])
    return data, html[match.end():match.end() + n]


def integrate(index, dataset, receipt_path, audit_path, summary_path=None, refit_path=None):
    index, dataset, receipt_path = [Path(p).resolve() for p in (index, dataset, receipt_path)]
    receipt = json.loads(receipt_path.read_text(encoding='utf8'))
    if receipt.get('complete') is not True or receipt.get('render_wh') != [1920, 1280]:
        raise ValueError('1920原尺寸核查未完成')
    rows = receipt['rows']
    if {(r['patch'], r['raw']) for r in rows} != {(p, f) for p in ('patch_00', 'patch_01', 'patch_02') for f in (86, 110)}:
        raise ValueError('核查组合不是完整六组')
    original = index.read_text(encoding='utf8')
    data_before, literal_before = embedded_data(original)
    catalog = data_before['catalog']
    before_sha = sha(index)
    prefix = Path('source_quality_audit')
    assets = {}

    def asset(source, relative):
        source = Path(source).resolve()
        dest = (index.parent / prefix / relative).resolve()
        if not dest.is_relative_to(index.parent.resolve()):
            raise ValueError('资源目标越界')
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists() or sha(dest) != sha(source):
            shutil.copy2(source, dest)
        rel = dest.relative_to(index.parent).as_posix()
        assets[rel] = {'sha256': sha(dest), 'bytes': dest.stat().st_size}
        return rel

    backup = HERE / 'backups' / 'index.before_source_quality.html'
    backup.parent.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        shutil.copy2(index, backup)

    cases = []
    for r in sorted(rows, key=lambda r: (r['patch'], r['raw'])):
        folder = Path('native1920') / r['patch'] / f"{r['raw']:03d}"
        case = dict(r)
        for key, name in [('photo', 'observation.png'), ('gaussians', 'isolated.png'),
                          ('full_road', 'full_road.png'), ('core_mask', 'core_mask.png'),
                          ('score_mask', 'score_mask.png'), ('alpha', 'isolated_alpha.png')]:
            source = dataset / folder / name
            with Image.open(source) as image:
                if list(image.size) != r['crop_wh']:
                    raise ValueError('核查图像尺寸不一致: ' + str(source))
            case[key] = asset(source, folder / name)
        case['HF_RMS_ratio'] = r['GS_HF_RMS'] / r['photo_HF_RMS'] if r['photo_HF_RMS'] > 0 else None
        cases.append(case)
    payload = {
        'schema': 'source_quality_native1920_report.v1',
        'run_id': 'road_gaussian_source_quality_20261122_r1', 'complete': True,
        'render_wh': receipt['render_wh'], 'cases': cases,
        'resources': {k: receipt[k] for k in ('nativeF', 'nativeB', 'optimizer_updates', 'new_diffusion_calls')},
        'receipt': asset(receipt_path, Path('native1920/receipt.json')),
        'architecture': asset(HERE / 'architecture_native1920.svg', Path('architecture_native1920.svg')),
        'summary': None,
    }
    if refit_path:
        refit_path = Path(refit_path).resolve()
        refit = json.loads(refit_path.read_text(encoding='utf8'))
        if (refit.get('complete') is not True or refit.get('formal_optimizer_updates') != 400
                or refit.get('geometry_scales_opacity_unchanged') is not True
                or refit.get('target_migration_executed') is not False):
            raise ValueError('源RGB诊断没有满足完成合同')
        refit_relative = refit_path.parent.relative_to(dataset)
        originals = {r['raw']: r for r in refit['original']}
        fitted = {r['raw']: r for r in refit['refit']}
        if set(originals) != {86, 98, 110, 137, 173} or set(originals) != set(fitted):
            raise ValueError('源RGB诊断缺少五帧实测')
        fit_cases = []
        for raw in sorted(originals):
            before, after = originals[raw], fitted[raw]
            if before['bbox_xyxy'] != after['bbox_xyxy'] or before['score_n'] != after['score_n']:
                raise ValueError('源RGB诊断评价域不一致')
            folder = refit_relative / 'original' / f'{raw:03d}'
            fitted_folder = refit_relative / 'refit' / f'{raw:03d}'
            with Image.open(dataset / folder / 'observation.png') as image:
                size = list(image.size)
            case = {'patch': 'patch_00', 'raw': raw, 'role': before['role'], 'crop_wh': size,
                    'metric_original': before, 'metric_refit': after}
            with Image.open(dataset / folder / 'core_mask.png') as mask:
                case['core_pixels'] = sum(mask.convert('L').histogram()[128:])
            for key, name in [('photo', 'observation.png'), ('gaussians', 'isolated.png'),
                              ('core_mask', 'core_mask.png'), ('score_mask', 'score_mask.png')]:
                source = dataset / folder / name
                with Image.open(source) as image:
                    if list(image.size) != size:
                        raise ValueError('源RGB图像尺寸不一致')
                case[key] = asset(source, folder / name)
            source = dataset / fitted_folder / 'isolated.png'
            with Image.open(source) as image:
                if list(image.size) != size:
                    raise ValueError('拟合前后图像尺寸不一致')
            case['refit'] = asset(source, fitted_folder / 'isolated.png')
            fit_cases.append(case)
        payload['native_cases'] = cases
        payload['cases'] = fit_cases + [r for r in cases if r['patch'] != 'patch_00']
        payload['source_refit'] = {
            'complete': True, 'contract': refit['contract'], 'cases': fit_cases,
            'resources': {k: refit[k] for k in ('nativeF', 'nativeB', 'formal_optimizer_updates',
                          'discarded_smoke_and_restore_updates', 'source_core_color_parameters')},
            'receipt': asset(refit_path, refit_relative / 'receipt.json'),
            'geometry_scales_opacity_unchanged': True, 'target_migration_executed': False,
        }
        payload['architecture'] = asset(HERE / 'architecture_source_rgb_fit.svg', Path('architecture_source_rgb_fit.svg'))
    metadata = [
        ('独立1920源材质审计', Path('reference_audit/source_quality_native1920.json')),
        ('199帧照片资格与排序', Path('source_view_ranking/manifest.json')),
        ('源块检索范围审计', Path('source_view_ranking/retrieval_audit.json')),
        ('199帧参考排序摘要', Path('source_view_ranking/summary.json')),
        ('源RGB重拟合独立审计', Path('reference_audit/source_rgb_refit_qualified_r2_independent_cpu.json')),
        ('照片UV一致性 · 近似投影诊断', Path('reference_audit/photo_pair_uv_consistency.json')),
        ('本轮累计计算记录', Path('resource_ledger_cumulative.json')),
    ]
    payload['evidence'] = [{'name': name, 'path': asset(dataset / p, p)}
                           for name, p in metadata if (dataset / p).is_file()]
    if (dataset / 'resource_ledger_cumulative.json').is_file():
        payload['resource_ledger'] = json.loads((dataset / 'resource_ledger_cumulative.json').read_text(encoding='utf8'))
    if summary_path:
        summary_path = Path(summary_path).resolve()
        summary = json.loads(summary_path.read_text(encoding='utf8'))
        if summary.get('complete') is not True:
            raise ValueError('审计摘要未完成')
        payload['summary'] = summary
        payload['summary_evidence'] = asset(summary_path, Path('audit_summary.json'))
    clean = re.sub(re.escape(BEGIN) + r'.*?' + re.escape(END), '', original, flags=re.S)
    literal = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace('</', '<\\/')
    viewer = (HERE / 'source_quality_viewer.js').read_text(encoding='utf8')
    block = BEGIN + '\n<script id="source-quality-audit-data" type="application/json">' + literal + '</script>\n<script>\n' + viewer + '\n</script>\n' + END
    position = clean.rfind('</html>')
    result = clean[:position] + block + '\n' + clean[position:] if position >= 0 else clean + block
    data_after, literal_after = embedded_data(result)
    if literal_before != literal_after or data_before != data_after:
        raise AssertionError('既有内嵌数据发生变化')
    temp = index.with_name(index.name + '.source-quality.tmp')
    temp.write_text(result, encoding='utf8')
    os.replace(temp, index)
    audit = {
        'schema': 'source_quality_native1920_integration.v1', 'passed': True,
        'index': str(index), 'backup': str(backup), 'before_sha256': before_sha, 'after_sha256': sha(index),
        'existing_D_literal_unchanged': True, 'existing_catalog_unchanged': True,
        'existing_records': len(catalog['models']), 'original_records': len([m for m in catalog['models'] if m.get('batch') != 'gaussian_transfer']),
        'existing_metric_rows': len(catalog['rows']), 'native_pairs': len(cases),
        'display_cases': len(payload['cases']), 'source_refit_frames': len(payload.get('source_refit', {}).get('cases', [])),
        'new_assets': assets, 'source_formal_updates': 400 if refit_path else 0,
        'source_discarded_updates': refit['discarded_smoke_and_restore_updates'] if refit_path else 0,
        'target_migration_updates': 0, 'new_diffusion_calls': 0,
    }
    Path(audit_path).write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--summary', type=Path)
    parser.add_argument('--refit-receipt', type=Path)
    args = parser.parse_args()
    audit = integrate(args.index, args.dataset, args.receipt, args.audit, args.summary, args.refit_receipt)
    print(json.dumps({k: audit[k] for k in ('passed', 'existing_D_literal_unchanged', 'existing_records', 'original_records', 'existing_metric_rows', 'native_pairs', 'source_refit_frames')}, ensure_ascii=False))
