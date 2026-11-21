"""Independent CPU-only QA of delivered HTML; never changes report or metrics."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import re
import time
import traceback
from urllib.parse import unquote, urlparse

from playwright.sync_api import sync_playwright

W = Path(__file__).resolve().parent
OUT = W.parents[1] / 'outputs/road_material_three_rounds_20261121'
INDEX = OUT / 'index.html'
BATCH_COUNTS = dict(legacy_1=11, legacy_2=11, legacy_3=8,
                    common_1=4, common_2=7, common_3=4, adaptive=5, residual=5)
RAWS = ['86', '110', '134', '179', '198']
DEFAULT = ['common_3::R3_G_NUMERIC', 'residual::appearance']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def payload():
    text = INDEX.read_text(encoding='utf-8-sig')
    start = re.search(r'\bconst\s+D\s*=\s*', text)
    assert start, 'missing embedded D payload'
    return json.JSONDecoder().raw_decode(text[start.end():])[0]


def declared_images(obj):
    result = set()
    def walk(v):
        if isinstance(v, dict):
            for x in v.values(): walk(x)
        elif isinstance(v, list):
            for x in v: walk(x)
        elif isinstance(v, str) and re.search(r'\.(png|webp|jpe?g|svg)$', v, re.I):
            if not Path(v).is_absolute() and not v.startswith(('http:', 'https:')):
                result.add(v)
    walk(obj)
    return result


def source_records():
    files = [OUT / p for p in ['report_data.json', 'common1152/report_data.json',
             'surface_adaptive/report_data.json', 'surface_residual_r2/report_data.json']]
    old, common, adaptive, residual = [load(p) for p in files]
    assert [len(old['metrics']), len(common['rows']), len(adaptive['rows']), len(residual['rows'])] == [294, 270, 90, 90]
    expected = {}
    for mid, m in old['methods'].items():
        frames = {}
        for raw, views in m['images'].items():
            frames[raw] = {}
            for view, image in views.items():
                frames[raw][view] = image['overview']
                for region, crop in image.get('crops', {}).items():
                    frames[raw][region + '_' + view + '_crop'] = crop['path']
        expected['legacy_' + m['group'] + '::' + mid] = frames
    for label, data in [('common', common), ('adaptive', adaptive), ('residual', residual)]:
        for m in data['models']:
            batch = 'common_' + str(m['round']) if label == 'common' else label
            expected[batch + '::' + m['id']] = m['frames']
    assert len(expected) == 55
    rows, gaps = defaultdict(list), defaultdict(list)
    for row in old['metrics']: rows['legacy_' + row['id'].split(':')[0]].append(row)
    for label, data in [('common', common), ('adaptive', adaptive), ('residual', residual)]:
        batches = {m['id']: 'common_' + str(m['round']) if label == 'common' else label for m in data['models']}
        for row in data['rows']: rows[batches[row['id']]].append(row)
        for row in data.get('gaps', []): gaps[batches[row['id']]].append(row)
    return files, expected, rows, gaps


def check_catalog(d, expected, original_rows, original_gaps):
    c = d['catalog']
    assert c['schema'] == 'road_report_catalog.v1'
    assert c['metric_recomputed'] is False
    assert {b['id']: b['count'] for b in c['batches']} == BATCH_COUNTS
    assert len(c['models']) == 55 and {m['id'] for m in c['models']} == set(expected)
    assert c['default_pair'] == DEFAULT
    assert len(c['rows']) == 744 and len(c['gaps']) == 150
    assert dict(Counter(m['batch'] for m in c['models'])) == BATCH_COUNTS
    for m in c['models']:
        assert m['id'] == m['batch'] + '::' + m['original_id']
        assert m['frames'] == expected[m['id']], m['id'] + ': image mapping changed'
    for key, original in [('rows', original_rows), ('gaps', original_gaps)]:
        actual = defaultdict(list)
        for row in c[key]: actual[row['batch']].append(row)
        for batch in BATCH_COUNTS:
            assert len(actual[batch]) == len(original[batch]), (key, batch)
            for source, target in zip(original[batch], actual[batch]):
                assert target['id'] == batch + '::' + source['id']
                for name, value in source.items():
                    if name not in ['id', 'source', 'domain']:
                        assert target.get(name) == value, (batch, source['id'], name)
                if batch.startswith('legacy_') and key == 'rows':
                    assert target['source_path'] == source['source']
                    assert target['source'] == source['region']
                    for old, new in [('mae', 'RGB_MAE'), ('hf', 'HF_cosine'), ('energy', 'HF_energy_ratio'),
                                     ('edge', 'edge_F1'), ('flat', 'flat_pred_RMS')]:
                        assert target[new] == source.get(old), (batch, source['id'], new)
                    assert target['core_n'] == source.get('core'), (batch, source['id'], 'HF core alias')
                    assert target['score_n'] == source.get('score_n', source.get('n')), (batch, source['id'], 'score domain alias')
                else: assert target.get('source') == source.get('source')
    paths = declared_images(d)
    missing = [p for p in sorted(paths) if not (OUT / p).is_file()]
    assert not missing, missing
    return c, len(paths)


def browser_check(c, expected):
    errors, failed_images = [], []
    counts = Counter()
    batches = {b['id']: b for b in c['batches']}
    models = {m['id']: m for m in c['models']}
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path='C:/Program Files/Google/Chrome/Application/chrome.exe',
            headless=True, args=['--disable-gpu', '--allow-file-access-from-files'])
        page = browser.new_page(viewport=dict(width=1500, height=1050))
        page.set_default_timeout(15000)
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.on('requestfailed', lambda r: failed_images.append(r.url) if r.resource_type == 'image' else None)
        page.goto(INDEX.as_uri(), wait_until='load')
        assert page.locator('#batch').input_value() == 'all'
        assert [page.locator('#' + k).input_value() for k in ['left', 'right']] == DEFAULT
        assert page.evaluate('D.catalog.rows.length') == 744
        for selector in ['#left', '#right']:
            assert page.locator(selector + ' option').count() == 55
            assert page.locator(selector + ' optgroup').count() == 8
            assert set(page.locator(selector + ' option').evaluate_all('es=>es.map(e=>e.value)')) == set(expected)
        for batch, n in BATCH_COUNTS.items():
            page.select_option('#batch', batch)
            for selector in ['#left', '#right']:
                values = page.locator(selector + ' option').evaluate_all('es=>es.map(e=>e.value)')
                assert len(values) == n and all(v.startswith(batch + '::') for v in values)
            counts['batch_filters'] += 1
        page.select_option('#batch', 'all')
        assert page.locator('#left option').count() == 55
        for mid in ['legacy_2::2:M01', 'legacy_2::2:M11', 'legacy_3::3:G_BUILD_NUMERIC',
                    'common_2::R2_M01', 'common_2::R2_M11', 'adaptive::appearance', 'residual::appearance']:
            assert page.locator('#left option[value="' + mid + '"]').count() == 1

        def gallery_paths():
            assert page.locator('#gallery figure').count() == 3
            page.wait_for_function('[...document.querySelectorAll("#gallery img")].every(i=>i.complete&&i.naturalWidth>0)')
            values = page.locator('#gallery img').evaluate_all('es=>es.map(e=>e.getAttribute("src"))')
            result = []
            for value in values:
                if value.startswith('file:'):
                    full = Path(unquote(urlparse(value).path).lstrip('/'))
                    value = full.relative_to(OUT).as_posix()
                result.append(value.replace('\\', '/'))
            assert not re.search(r'\b(?:undefined|NaN)\b', page.locator('#gallery').inner_text())
            return result

        page.select_option('#right', 'residual::appearance')
        page.select_option('#region', 'C')
        for index, (mid, frames) in enumerate(expected.items()):
            page.select_option('#left', mid)
            for raw in RAWS:
                page.select_option('#raw', raw)
                for view in ['road', 'joint']:
                    page.select_option('#view', view)
                    for mode in ['full', 'detail']:
                        page.click('#' + mode)
                        want = frames[raw].get('C_' + view + '_crop') if mode == 'detail' else None
                        want = want or frames[raw][view]
                        assert want in gallery_paths(), (mid, raw, view, mode, want)
                        assert page.locator('#gallery figure').nth(1).locator('img').count() == 1
                        counts['road_joint_combinations'] += 1
                page.select_option('#view', 'alpha')
                page.click('#full')
                paths = gallery_paths()
                if frames[raw].get('alpha'):
                    assert frames[raw]['alpha'] in paths, (mid, raw, 'alpha')
                    assert page.locator('#gallery figure').nth(1).locator('img').count() == 1
                    counts['available_alpha_combinations'] += 1
                else:
                    assert page.locator('#gallery figure').nth(1).locator('img').count() == 0
                    counts['legal_missing_alpha_combinations'] += 1
            if (index + 1) % 10 == 0: print('checked_models', index + 1, flush=True)

        # Teacher/ref follows the right record's source, including legal absent teachers.
        page.select_option('#left', DEFAULT[0])
        page.select_option('#view', 'road')
        page.click('#full')
        for mid in expected:
            page.select_option('#right', mid)
            teacher = batches[models[mid]['batch']]['teacher']
            for raw in ['86', '134']:
                page.select_option('#raw', raw)
                for region in ['A', 'C', 'S']:
                    page.select_option('#region', region)
                    paths = gallery_paths()
                    ref_region = 'IDEAL' if models[mid]['original_id'].endswith('IDEAL_ORACLE') else region
                    wanted = teacher.get('frames', {}).get(raw, {}).get(ref_region)
                    student = {expected[x][raw]['road'] for x in [DEFAULT[0], mid]}
                    reference_paths = [x for x in paths if x not in student]
                    if wanted: assert reference_paths == [wanted], (mid, raw, region, reference_paths, wanted)
                    else: assert not reference_paths, (mid, raw, region, 'unexpected teacher fallback', reference_paths)
                    counts['right_teacher_combinations'] += 1

        for batch in BATCH_COUNTS:
            page.select_option('#batch', batch)
            page.select_option('#raw', '110')
            for region in ['A', 'C', 'S']:
                page.select_option('#region', region)
                for scale in ['1', '2', '4']:
                    # Preserve actual HTML values such as '1.0'.
                    value = next(v for v in page.locator('#scale option').evaluate_all('es=>es.map(e=>e.value)') if float(v) == float(scale))
                    page.select_option('#scale', value)
                    gallery_paths()
                    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
                    counts['region_scale_combinations'] += 1

        page.select_option('#batch', 'all')
        for key, value in zip(['left', 'right'], DEFAULT): page.select_option('#' + key, value)
        page.select_option('#raw', '110'); page.select_option('#view', 'road'); page.select_option('#region', 'C')
        page.click('#detail'); gallery_paths()
        page.screenshot(path=str(W / 'qa_unified_desktop.png'), full_page=True)
        page.set_viewport_size(dict(width=390, height=1000))
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
        page.screenshot(path=str(W / 'qa_unified_mobile.png'), full_page=True)
        for mid in ['legacy_3::3:G_BUILD_NUMERIC', 'common_2::R2_M11', 'adaptive::appearance', 'residual::appearance']:
            page.select_option('#left', mid); gallery_paths()
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'), ('mobile overflow', mid)
            counts['mobile_source_checks'] += 1
        links = page.locator('a[href]').evaluate_all('es=>es.map(e=>e.getAttribute("href"))')
        missing = [x for x in links if x and not x.startswith(('http:', 'https:', '#')) and not (OUT / x.split('#')[0]).is_file()]
        assert not missing, missing
        browser.close()
    assert not errors, errors
    assert not failed_images, failed_images
    return dict(counts), errors


def main():
    start = time.perf_counter()
    files, expected, rows, gaps = source_records()
    protected = files + [INDEX] + sorted(OUT.glob('history*.html'))
    before = {str(p.relative_to(OUT)): digest(p) for p in protected}
    d = payload()
    source_images = set().union(*(declared_images(load(p)) for p in files))
    assert all((OUT / p).is_file() for p in source_images), 'missing source-declared image'
    c, images = check_catalog(d, expected, rows, gaps)
    counts, errors = browser_check(c, expected)
    after = {str(p.relative_to(OUT)): digest(p) for p in protected}
    assert before == after, 'report/source changed during independent QA'
    return dict(passed=True, records=55, batches=BATCH_COUNTS, metric_rows=744, gap_rows=150,
                declared_images=images, source_declared_images=len(source_images), checks=counts, JavaScript_errors=errors,
                source_and_HTML_unchanged=True, source_SHA256=after,
                seconds=time.perf_counter()-start, new_GPU_calls=0, scientific_metrics_recomputed=False)


if __name__ == '__main__':
    try:
        result = main()
    except Exception:
        result = dict(passed=False, error=traceback.format_exc(), new_GPU_calls=0, scientific_metrics_recomputed=False)
        (W / 'qa_unified_catalog.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        raise
    (W / 'qa_unified_catalog.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)
