"""Validate real source-colour results; append to the single report only with --apply."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil

from PIL import Image

HERE = Path(__file__).resolve().parent
DEFAULT_INDEX = Path(r'C:\Users\dengyunlong\Documents\Codex\2026-11-16\xian\outputs\road_material_three_rounds_20261121\index.html')
BEGIN = '<!-- SOURCE_FIDELITY_CHAIN_BEGIN -->'
END = '<!-- SOURCE_FIDELITY_CHAIN_END -->'
OLD_BEGIN = '<!-- SOURCE_QUALITY_AUDIT_BEGIN -->'
OLD_END = '<!-- SOURCE_QUALITY_AUDIT_END -->'
PREFIX = Path('source_fidelity_chain_r2')
REQUIRED_STAGES = ('photo', 'sampled_rgb', 'loaded_rgb', 'native_render')
ALL_STAGES = REQUIRED_STAGES + ('original_render',)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf8')


def existing_state(html):
    match = re.search(r'\bconst\s+D\s*=\s*', html)
    if not match:
        raise ValueError('Missing original D literal')
    data, length = json.JSONDecoder().raw_decode(html[match.end():])
    literal = html[match.end():match.end() + length]
    block = re.search(re.escape(OLD_BEGIN) + r'.*?' + re.escape(OLD_END), html, re.S)
    if not block:
        raise ValueError('Missing previous source-quality audit')
    catalog = data['catalog']
    counts = (len(catalog['models']), len([m for m in catalog['models'] if m.get('batch') != 'gaussian_transfer']), len(catalog['rows']))
    if counts != (58, 55, 744):
        raise ValueError(f'Unexpected original catalog counts: {counts}')
    return {'records': counts[0], 'historical_records': counts[1], 'metric_rows': counts[2],
            'D_literal_sha256': digest(literal.encode('utf8')),
            'source_quality_block_sha256': digest(block[0].encode('utf8'))}


def number(value, label, *, nonnegative=False, optional=False):
    if optional and value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Expected finite number: ' + label)
    if nonnegative and value < 0:
        raise ValueError('Expected nonnegative number: ' + label)


def build_payload(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    source = json.loads(manifest_path.read_text(encoding='utf8'))
    if source.get('schema') != 'source_fidelity_chain_result.v1' or source.get('complete') is not True:
        raise ValueError('Real complete source_fidelity_chain_result.v1 result required')
    for field in ('geometry_unchanged', 'cameras_unchanged'):
        if source.get(field) is not True:
            raise ValueError('Fixed-geometry contract failed: ' + field)
    if source.get('render_wh') != [1920, 1280] or source.get('target_migration_executed') is not False:
        raise ValueError('Expected native 1920 source-only result')
    if not source.get('run_id') or not source.get('color_encoding'):
        raise ValueError('run_id and color_encoding required')
    for key in ('new_diffusion_calls', 'new_target_migration_updates'):
        if source.get(key) != 0:
            raise ValueError('Expected zero: ' + key)
    patches = source['patches']
    if {p['id'] for p in patches} != {'patch_00', 'patch_02'} or len(patches) != 2:
        raise ValueError('Both source 00 and 02 must be represented equally')
    assets = []

    def asset(value, relative, expected_wh=None):
        path = Path(value)
        if not path.is_absolute():
            path = manifest_path.parent / path
        path = path.resolve(strict=True)
        if not path.is_file():
            raise ValueError('Expected file: ' + str(path))
        if expected_wh is not None:
            with Image.open(path) as im:
                if list(im.size) != expected_wh:
                    raise ValueError('Native crop mismatch: ' + str(path))
        relative = (PREFIX / relative).as_posix()
        raw = path.read_bytes()
        assets.append({'source': str(path), 'relative': relative, 'sha256': digest(raw), 'bytes': len(raw)})
        return relative

    cases = []
    combinations = set()
    for case in source['cases']:
        patch, raw = case['patch'], case['raw']
        if patch not in {'patch_00', 'patch_02'} or not isinstance(raw, int) or raw < 0:
            raise ValueError('Invalid patch or camera frame')
        if (patch, raw) in combinations:
            raise ValueError('Duplicate patch/frame')
        combinations.add((patch, raw))
        box, wh = case['bbox_xyxy'], case['crop_wh']
        if (len(box) != 4 or any(type(v) is not int for v in box)
                or not (0 <= box[0] < box[2] <= 1920 and 0 <= box[1] < box[3] <= 1280)
                or wh != [box[2] - box[0], box[3] - box[1]]):
            raise ValueError('Invalid native crop')
        item = {key: case[key] for key in ('patch', 'raw', 'bbox_xyxy', 'crop_wh')}
        item['role'] = case.get('role', '')
        item['variants'] = {}
        if set(case['variants']) != {'core', 'halo'}:
            raise ValueError('Both core and halo variants required')
        for variant in ('core', 'halo'):
            view = case['variants'][variant]
            if not view.get('scope') or not view.get('label'):
                raise ValueError('Actual row-set or mask scope must be explicit')
            if not set(REQUIRED_STAGES).issubset(view['images']) or set(view['images']) - set(ALL_STAGES):
                raise ValueError('Expected photo/sample/load/native stages')
            prefix = Path('cases') / patch / f'{raw:03d}' / variant
            dest = {'label': view['label'], 'scope': view['scope'], 'images': {}}
            for stage in ALL_STAGES:
                if stage in view['images']:
                    original_path = view['images'][stage]
                    dest['images'][stage] = asset(original_path, prefix / (stage + Path(original_path).suffix.lower()), wh)
            if view.get('display_mask'):
                dest['display_mask'] = asset(view['display_mask'], prefix / 'display_mask.png', wh)
            rows = view['metrics']
            if not any(m['stage'] == 'native_render' for m in rows):
                raise ValueError('Native rendering metrics required')
            for row in rows:
                if row['stage'] not in ALL_STAGES[1:] or row['stage'] not in view['images']:
                    raise ValueError('Metric stage lacks matching image')
                if not row.get('domain') or type(row['score_n']) is not int or row['score_n'] <= 0:
                    raise ValueError('Explicit metric domain and denominator required')
                for name in ('RGB_MAE', 'HF_cosine', 'HF_RMS_ratio', 'sigma_native_px'):
                    number(row.get(name), name, nonnegative=name in {'RGB_MAE', 'HF_RMS_ratio', 'sigma_native_px'}, optional=True)
                if row.get('RGB_bias') is not None:
                    if len(row['RGB_bias']) != 3:
                        raise ValueError('RGB_bias must contain 3 channels')
                    for value in row['RGB_bias']:
                        number(value, 'RGB_bias')
                if row['stage'] in {'original_render', 'native_render'} and not re.fullmatch(r'[0-9a-f]{64}', row.get('score_mask_sha256', '')):
                    raise ValueError('Rendering metrics require a score-mask digest')
                if row.get('HF_cosine') is not None and not row.get('sigma_native_px'):
                    raise ValueError('HF metric needs native-pixel sigma')
            for before in [m for m in rows if m['stage'] == 'original_render']:
                after = [m for m in rows if m['stage'] == 'native_render' and m.get('sigma_native_px') == before.get('sigma_native_px')]
                if len(after) != 1 or any(before.get(k) != after[0].get(k) for k in ('domain', 'score_n', 'score_mask_sha256')):
                    raise ValueError('Before/after rendering domains differ')
            dest['metrics'] = rows
            item['variants'][variant] = dest
        cases.append(item)
    if {c['patch'] for c in cases} != {'patch_00', 'patch_02'}:
        raise ValueError('Real cases required for both sources')
    checks = []
    for i, check in enumerate(source['checks']):
        if check['patch'] not in {'patch_00', 'patch_02'} or check.get('geometry_unchanged') is not True:
            raise ValueError('Invalid source numerical proof')
        if type(check.get('point_ids_equal')) is not bool or type(check.get('point_count')) is not int or check['point_count'] <= 0:
            raise ValueError('Point count and ID equality must be measured')
        for name in ('RGB_max_abs', 'RGB_mean_abs'):
            number(check.get(name), name, nonnegative=True, optional=not check['point_ids_equal'])
        proof = dict(check)
        proof['evidence'] = asset(check['evidence'], Path('evidence') / f'check_{i:02d}{Path(check["evidence"]).suffix}')
        checks.append(proof)
    if {c['patch'] for c in checks} != {'patch_00', 'patch_02'}:
        raise ValueError('Save/load proof required for both sources')
    for row in source['resources']:
        if not row.get('process'):
            raise ValueError('Resource process label required')
        for key in ('nativeF', 'nativeB', 'formal_updates', 'extra_updates'):
            if type(row.get(key)) is not int or row[key] < 0:
                raise ValueError('Actual nonnegative integer resource count required: ' + key)
    evidence = [{'name': e['name'], 'path': asset(e['path'], Path('evidence') / f'extra_{i:02d}{Path(e["path"]).suffix}')}
                for i, e in enumerate(source.get('evidence', []))]
    payload = {key: source[key] for key in ('run_id', 'complete', 'render_wh', 'geometry_unchanged', 'cameras_unchanged',
                                          'target_migration_executed', 'color_encoding', 'new_diffusion_calls',
                                          'new_target_migration_updates', 'resources')}
    payload.update(schema='source_fidelity_chain_report.v1', patches=sorted(patches, key=lambda p: p['id']),
                   cases=sorted(cases, key=lambda c: (c['patch'], c['raw'])), checks=checks, evidence=evidence,
                   findings=source.get('findings', []), metric_note=source.get('metric_note', ''),
                   review_note=source.get('review_note', ''),
                   architecture=asset(HERE / 'architecture_components.svg', Path('architecture_components.svg')))
    return payload, assets


def apply(index, payload, assets):
    original = index.read_text(encoding='utf8')
    before = existing_state(original)
    clean = re.sub(re.escape(BEGIN) + r'.*?' + re.escape(END), '', original, flags=re.S)
    literal = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace('</', '<\\/')
    viewer_name = 'source_fidelity_actual_viewer.js' if 'photo_review' in payload else 'source_fidelity_viewer.js'
    viewer = (HERE / viewer_name).read_text(encoding='utf8')
    block = BEGIN + '\n<script id="source-fidelity-chain-data" type="application/json">' + literal + '</script>\n<script>\n' + viewer + '\n</script>\n' + END
    position = clean.rfind('</html>')
    if position < 0:
        raise ValueError('Single original HTML missing closing html tag')
    result = clean[:position] + block + '\n' + clean[position:]
    if existing_state(result) != before:
        raise AssertionError('Original catalog or previous audit changed')
    backup = HERE / 'backups' / ('index.' + digest(original.encode('utf8'))[:16] + '.before_source_fidelity.html')
    backup.parent.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        shutil.copy2(index, backup)
    for plan in assets:
        dest = (index.parent / plan['relative']).resolve()
        if not dest.is_relative_to((index.parent / PREFIX).resolve()):
            raise ValueError('Asset path escaped its new report directory')
        src = Path(plan['source'])
        if digest(src.read_bytes()) != plan['sha256']:
            raise ValueError('Asset changed after validation')
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        if digest(dest.read_bytes()) != plan['sha256']:
            raise ValueError('Copied asset failed hash verification')
    temporary = index.with_name(index.name + '.source-fidelity.tmp')
    temporary.write_text(result, encoding='utf8')
    os.replace(temporary, index)
    return dict(passed=True, index=str(index), backup=str(backup), existing_state_unchanged=True,
                before=before, after=existing_state(index.read_text(encoding='utf8')),
                before_html_sha256=digest(original.encode('utf8')), after_html_sha256=digest(index.read_bytes()),
                cases=len(payload.get('cases', payload.get('photo_review', {}).get('rows', []))),
                formal_solvers=len(payload.get('solvers', [])), camera_controls=len(payload.get('camera_controls', [])),
                static_controls=len(payload.get('static_controls', [])), copied_assets=len(assets))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, default=DEFAULT_INDEX)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    snapshot = existing_state(args.index.read_text(encoding='utf8'))
    snapshot.update(index=str(args.index.resolve()), html_sha256=digest(args.index.read_bytes()))
    if not args.manifest:
        if args.apply:
            parser.error('--apply requires real --manifest')
        write_json(HERE / 'baseline_snapshot.json', snapshot)
        print(json.dumps(snapshot))
    else:
        payload, assets = build_payload(args.manifest)
        write_json(HERE / 'adapted_payload.json', payload)
        write_json(HERE / 'asset_plan.json', assets)
        if args.apply:
            audit = apply(args.index.resolve(), payload, assets)
            write_json(HERE / 'integration_audit.json', audit)
            print(json.dumps(audit))
        else:
            print(json.dumps({'validated': True, 'applied': False, 'cases': len(payload['cases']), 'assets': len(assets), 'existing': snapshot}))
