"""CPU saved-render audit. Fixed domains stay separate from material additions.

No model/rasterizer import, fitting, GPU work, source mutation, or main-report edit.
The default run audits the five already saved controls. --candidate adds a new
saved 4xKD candidate after its source and run contract have been reviewed.
"""
import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
OLD = WS / 'outputs/scene008_v44_a5_road_fusion_integration_20261107'
C49 = WS / 'outputs/scene008_v44_a5_c49_rgb_20261107'
WORLD = Path('C:/Users/dengyunlong/Documents/Codex/2026-11-05/j-ao-j/outputs/scene008_road_fusion_bake/delivery/repro/red_review/world_masks49_v1')
RAWS = [94, 105, 128, 137]
STATES = ['keep', 'delete']
EDGES = {'near_x': 1, 'far_x': 1, 'lateral_y': 0}
FROZEN = ['alpha', 'depth_sum', 'depth_m', 'group_alpha_T', 'added_owner_rgb_T']
POINTS = [(128, 301, 221, 'registered_white'), (128, 310, 222, 'old_L_adjacent_dark'),
          (137, 300, 227, 'registered_white'), (137, 311, 228, 'old_L_adjacent_dark')]
CROPS = [(105, 'keep', (0, 160, 175, 320), 'a5 roof / original crop'),
         (128, 'keep', (65, 185, 165, 255), 'retained a6 / visible wheel'),
         (128, 'delete', (165, 188, 208, 234), 'registered primary128 seam'),
         (128, 'delete', (276, 204, 326, 240), 'white + adjacent dark / raw128'),
         (137, 'delete', (114, 220, 211, 238), 'registered near137 seam'),
         (137, 'delete', (276, 209, 325, 246), 'white + adjacent dark / raw137')]


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def load(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def csv_write(path, rows):
    assert rows
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def chw(a):
    a = np.asarray(a)
    if a.shape == (320, 480, 3):
        a = a.transpose(2, 0, 1)
    assert a.shape == (3, 320, 480) and np.isfinite(a).all()
    return a.astype(np.float32, copy=False)


def means(a, m):
    a = np.asarray(a)[m].astype(np.float64)
    assert np.isfinite(a).all()
    return float(a.mean()) if a.size else None


def upper(a):
    y = np.nonzero(a)[0]
    return a & (np.arange(320)[:, None] <= y.min() + .25 * (y.max() - y.min())) if len(y) else a.copy()


def domains(obs, target):
    a5, a6, a37 = obs['actor_labels'] == 1, obs['visible'][0], obs['visible'][1]
    road, edit, protect = obs['road'], obs['road_edit'], obs['protect']
    return dict(a5_RGB=a5, a5_upper_RGB=upper(a5), a6_RGB=a6, a37_RGB=a37,
                protected_nonroad_RGB=protect & ~a6 & ~a37 & ~road & ~edit,
                legacy_protected_RGB=protect & ~a6 & ~a37,
                original_road_RGB=road, target_protect_RGB=target['protect_mask'],
                target_fit_RGB=target['fit_mask'])


def pair_masks(masks, *, primary=False):
    pairs = []
    for edge, axis in EDGES.items():
        minus = masks[('band_' if primary else '') + edge + '_minus']
        plus = masks[('band_' if primary else '') + edge + '_plus']
        selected = minus | plus
        bins = np.zeros(minus.shape, np.int32)
        bins[selected] = np.floor(masks['local_xyz'][..., axis][selected] / .2).astype(np.int32)
        for number in np.unique(bins[selected]):
            left, right = minus & (bins == number), plus & (bins == number)
            if left.sum() >= 3 and right.sum() >= 3:
                pairs.append((edge, int(number), left, right))
    return pairs


def font(size):
    return ImageFont.truetype('C:/Windows/Fonts/arial.ttf', size)


def render_png(path):
    if path.is_file():
        im = Image.open(path).convert('RGB')
    else:
        # Supplementary saved float packages omit Cwarm PNGs. This is only an
        # 8-bit display conversion; all metrics above use unclamped float RGB.
        a = chw(load(path.with_name(path.stem + '_float.npz'))['rgb']).transpose(1, 2, 0)
        im = Image.fromarray(np.round(a.clip(0, 1) * 255).astype(np.uint8))
    assert im.size == (480, 320)
    return im


def overlay(im, mask, colour, opacity=.65):
    arr = np.asarray(im).copy().astype(np.float64)
    arr[mask] = arr[mask] * (1 - opacity) + np.asarray(colour) * opacity
    return Image.fromarray(np.round(arr).astype(np.uint8))


def draw_distribution(out, cached, observations, targets, variants):
    image = Image.new('RGB', (1980, 1530), '#f4f6f8')
    d = ImageDraw.Draw(image)
    d.text((14, 10), 'Fixed four-frame protection / supervision distribution', font=font(24), fill='#153342')
    d.text((14, 43), 'GT; original fixed masks; Cwarm newroad group7>0 support in KEEP / DELETE. No candidate-dependent denominator.', font=font(17), fill='#344f5c')
    headers = ['Original observation', 'fit=green; protect=yellow; originalroad=cyan', 'Cwarm KEEP road support=red', 'Cwarm DELETE road support=red']
    for col, title in enumerate(headers):
        d.text((14 + col * 491, 77), title, font=font(16), fill='#153342')
    for row, raw in enumerate(RAWS):
        top = 135 + row * 345
        obs, target = observations[raw], targets[raw]
        gt = Image.fromarray(np.round(chw(obs['rgb']).transpose(1, 2, 0).clip(0, 1) * 255).astype(np.uint8))
        dm = domains(obs, target)
        m = overlay(gt, dm['original_road_RGB'], (0, 210, 220))
        m = overlay(m, target['fit_mask'], (0, 220, 75))
        m = overlay(m, target['protect_mask'], (255, 218, 0))
        cells = [gt, m]
        counts = ['raw%03d' % raw, 'fit=%d  protect=%d  protectNR=%d' % (target['fit_mask'].sum(), target['protect_mask'].sum(), dm['protected_nonroad_RGB'].sum())]
        for state in STATES:
            support = cached[(raw, state, 'Cwarm')]['group_alpha_T'][7] > 0
            cells.append(overlay(gt, support, (255, 45, 35)))
            counts.append('road>0=%d; protectedNR overlap=%d' % (support.sum(), (support & dm['protected_nonroad_RGB']).sum()))
        for col, (im, count) in enumerate(zip(cells, counts)):
            d.text((14 + col * 491, top - 24), count, font=font(16), fill='#153342')
            image.paste(im, (14 + col * 491, top))
    image.save(out / 'protection_distribution4.png')


def draw_details(out, variants, filename='native_details.png'):
    cell_w, row_h = 282, 222
    board = Image.new('RGB', (18 + cell_w * len(variants), 130 + row_h * len(CROPS)), '#f4f6f8')
    d = ImageDraw.Draw(board)
    d.text((14, 10), 'Saved native crops / fixed before new candidate', font=font(23), fill='#153342')
    d.text((14, 42), 'Nearest display zoom; same xyxy across all columns. Full-image metrics remain in domain_metrics.csv.', font=font(16), fill='#344f5c')
    for col, (name, folder) in enumerate(variants.items()):
        left = 14 + col * cell_w
        d.text((left, 78), name, font=font(20), fill='#153342')
        for row, (raw, state, box, label) in enumerate(CROPS):
            top = 115 + row * row_h
            d.text((left, top), f'{raw} {state}: {label}', font=font(14), fill='#153342')
            d.text((left, top + 20), f'xyxy={box}', font=font(14), fill='#344f5c')
            im = render_png(folder / f'frames/{raw:03d}/{state}.png').crop(box)
            ratio = min(266 / im.width, 166 / im.height)
            im = im.resize((round(im.width * ratio), round(im.height * ratio)), Image.Resampling.NEAREST)
            board.paste(im, (left + (266 - im.width) // 2, top + 45))
    board.save(out / filename)


def draw_metrics(out, rows, names):
    panels = [('protected_nonroad_RGB', 'original', 'protectedNR / original full fixed domain'),
              ('protected_nonroad_RGB', 'Cwarm_newroad_support', 'protectedNR / fixed road-affected subset'),
              ('original_road_RGB', 'original', 'originalroad / original full fixed domain'),
              ('original_road_RGB', 'Cwarm_newroad_support', 'originalroad / fixed road-affected subset')]
    cell, label, panel_h = 112, 135, 350
    panel_w = label + cell * len(names) + 28
    im = Image.new('RGB', (panel_w * 2, panel_h * 2 + 80), '#f4f6f8')
    d = ImageDraw.Draw(im)
    d.text((14, 10), 'RGB MAE to original observation / full four-frame KEEP + DELETE', font=font(22), fill='#153342')
    d.text((14, 39), 'Values are absolute MAE. Red/cyan tint = increase/decrease relative to Cwarm; denominators are in CSV.', font=font(16), fill='#344f5c')
    lookup = {(r['raw'], r['state'], r['variant'], r['domain'], r['scope']): r for r in rows}
    for pi, (domain, scope, title) in enumerate(panels):
        ox, oy = (pi % 2) * panel_w, 80 + (pi // 2) * panel_h
        d.text((ox + 14, oy), title, font=font(18), fill='#153342')
        for j, name in enumerate(names):
            d.text((ox + label + j * cell + 5, oy + 31), name, font=font(16), fill='#153342')
        for i, (raw, state) in enumerate((r, s) for r in RAWS for s in STATES):
            y = oy + 60 + i * 32
            d.text((ox + 14, y + 5), f'{raw} {state}', font=font(16), fill='#153342')
            for j, name in enumerate(names):
                row = lookup[(raw, state, name, domain, scope)]
                delta, value = row['delta_MAE_vs_Cwarm'], row['MAE_to_observation']
                mag = min(abs(delta or 0) * 50, 1)
                colour = (255, round(250 - 80 * mag), round(245 - 100 * mag)) if (delta or 0) > 1e-6 else (round(245 - 90 * mag), 250, 250)
                x = ox + label + j * cell
                d.rectangle((x, y, x + cell - 4, y + 28), fill=colour)
                text = 'empty' if value is None else f'{value:.5f}'
                d.text((x + 5, y + 5), text, font=font(16), fill='#153342')
    im.save(out / 'fourframe_domain_MAE.png')


def asset_comparison(reference, candidate):
    a, b = load(reference), load(candidate)
    assert set(a) == set(b), 'Candidate asset field set changed'
    allowed = {'render_colours_srgb', 'parent_render_colours_srgb'}
    rows = []
    for key in a:
        exact = a[key].dtype == b[key].dtype and a[key].shape == b[key].shape and a[key].tobytes() == b[key].tobytes()
        rows.append(dict(field=key, byte_exact=exact, RGB_change_allowed=key in allowed))
    assert all(r['byte_exact'] for r in rows if not r['RGB_change_allowed']), 'Non-RGB asset field changed'
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate', action='append', help='NAME=directory containing frames/094/{keep,delete}_float.npz; repeat for a fixed pair')
    parser.add_argument('--reference-asset', type=Path)
    parser.add_argument('--candidate-asset', type=Path)
    args = parser.parse_args()
    assert not args.output.exists(), 'Do not overwrite prior verification evidence'
    variants = dict(latest21K=HERE / 'inputs/latest21K', Cwarm=OLD / 'evidence/H2_source_warm',
                    formal2K=C49 / 'evidence/H2_material2000_retry1',
                    K=OLD / 'evidence/K_LP_candidate', L=OLD / 'evidence/L_Cwarm12_candidate')
    for specification in args.candidate or []:
        name, path = specification.split('=', 1)
        assert name not in variants
        variants[name] = Path(path).resolve()
    contract = read(OLD / 'evidence/independent_evaluation_contract_v1.json')
    registry = read(WORLD / 'registry.json')
    assert sha(WORLD / 'registry.json') == contract['road_domains']['world_registry_SHA256']
    assert RAWS == contract['pilot_raws']
    original_L = read(OLD / 'evidence/review/L_candidate_fixed4560_cpu.json')
    cert = read(OLD / 'evidence/review/K_single_LP_result_v1.json')
    operator = read(OLD / 'evidence/J_operator/operator_manifest.json')
    assert len(operator['seam_records']) == len(cert['per_bin_constraints']) == 106
    obs, targets, cached, metric = {}, {}, {}, {}
    registered_support = []
    domain_rows, invariants, points, seam_rows, original_gates = [], [], [], [], []
    orig = {r['raw']: r for r in registry['frames']}
    for raw in RAWS:
        obs[raw] = load(HERE / f'inputs/observations/{raw:03d}.npz')
        targets[raw] = load(HERE / f'inputs/targets/raw{raw:03d}.npz')
        assert not (targets[raw]['fit_mask'] & targets[raw]['protect_mask']).any()
        np.testing.assert_array_equal(chw(obs[raw]['rgb']), chw(targets[raw]['observation_rgb']))
        assert sha(WORLD / f'{raw:03d}.npz') == orig[raw]['mask_sha256']
        pairs = [('world', *p) for p in pair_masks(load(WORLD / f'{raw:03d}.npz'))]
        for edge in EDGES:
            actual_pairs = [p for p in pairs if p[1] == edge]
            registered_pairs = orig[raw]['edges'][edge]['bins']
            assert [(p[2], int(p[3].sum()), int(p[4].sum())) for p in actual_pairs] == [
                (p['tangent_bin'], p['minus_count'], p['plus_count']) for p in registered_pairs]
            registered_support.append(dict(raw=raw, edge=edge, paired_bins=len(actual_pairs),
                                           status='SUPPORTED' if actual_pairs else 'UNSUPPORTED',
                                           unsupported_score=None))
        if raw == 128:
            primary_path = WORLD / 'primary_raw128/fixed_masks_raw128.npz'
            expected_sha = next(r['sha256'] for r in contract['source_references'] if r['relative'].endswith('/fixed_masks_raw128.npz'))
            assert sha(primary_path) == expected_sha
            primary = load(primary_path)
            assert int(primary['original_road'].sum()) == 1385
            pairs += [('primary128', *p) for p in pair_masks(primary, primary=True)]
            pp = [p for p in pairs if p[0] == 'primary128' and p[1] == 'near_x']
            assert len(pp) == 6 and sum(int(p[3].sum()) for p in pp) == 35 and sum(int(p[4].sum()) for p in pp) == 33
        gt = chw(obs[raw]['rgb'])
        dm = domains(obs[raw], targets[raw])
        for state in STATES:
            for name, folder in variants.items():
                value = load(folder / f'frames/{raw:03d}/{state}_float.npz')
                for field in ['alpha', 'depth_sum', 'depth_m']:
                    value[field] = value[field].squeeze()
                assert all(np.isfinite(a).all() for a in value.values())
                assert value['rgb'].dtype == np.float32
                assert value['group_alpha_T'].shape == ((7 if name == 'latest21K' else 8), 320, 480)
                cached[(raw, state, name)] = value
            warm = cached[(raw, state, 'Cwarm')]
            support = warm['group_alpha_T'][7] > 0
            for name in variants:
                value = cached[(raw, state, name)]
                err = np.abs(value['rgb'] - gt).mean(0)
                for domain, fixed in dm.items():
                    if domain == 'a37_RGB' and state == 'delete':
                        continue
                    for scope, mask in [('original', fixed), ('Cwarm_newroad_support', fixed & support)]:
                        selected = err[mask].astype(np.float64)
                        score = means(err, mask)
                        metric[(raw, state, name, domain, scope)] = score
                        domain_rows.append(dict(raw=raw, state=state, variant=name, domain=domain, scope=scope,
                                                pixels=int(mask.sum()), MAE_to_observation=score,
                                                pixel_error_p95=float(np.quantile(selected, .95)) if selected.size else None,
                                                pixel_error_max=float(selected.max()) if selected.size else None))
                if name != 'latest21K':
                    for field in FROZEN:
                        a, b = value[field], warm[field]
                        exact = a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
                        invariants.append(dict(raw=raw, state=state, variant=name, field=field,
                                               byte_exact_to_Cwarm=exact,
                                               max_abs_to_Cwarm=float(np.abs(a-b).max())))
                for kind, edge, number, minus, plus in pairs:
                    residual = value['rgb'] - gt
                    jump = residual[:, plus].mean(1) - residual[:, minus].mean(1)
                    seam_rows.append(dict(raw=raw, state=state, variant=name, kind=kind, edge=edge, tangent_bin=number,
                                          minus_pixels=int(minus.sum()), plus_pixels=int(plus.sum()),
                                          score32=float(np.abs(jump).mean()),
                                          score64=float(np.abs(jump.astype(np.float64)).mean())))
                for point_raw, x, y, role in POINTS:
                    if point_raw == raw:
                        rgb = value['rgb'][:, y, x].astype(np.float64)
                        target = gt[:, y, x].astype(np.float64)
                        points.append(dict(raw=raw, state=state, variant=name, role=role, x=x, y=y,
                                           MAE_to_observation=float(np.abs(rgb-target).mean()),
                                           red=float(rgb[0]), green=float(rgb[1]), blue=float(rgb[2]),
                                           original_protect=bool(obs[raw]['protect'][y, x]),
                                           original_road=bool(obs[raw]['road'][y, x]),
                                           target_protect=bool(targets[raw]['protect_mask'][y, x]),
                                           fixed_Cwarm_newroad_alphaT=float(warm['group_alpha_T'][7, y, x])))
    assert all(r['byte_exact_to_Cwarm'] for r in invariants), 'Frozen rendered geometry/semantic RGB changed'
    for r in domain_rows:
        key = (r['raw'], r['state'], r['variant'], r['domain'], r['scope'])
        score = metric[key]
        for ref in ['latest21K', 'Cwarm']:
            base = metric[(*key[:2], ref, *key[3:])]
            r['delta_MAE_vs_' + ref] = score - base if score is not None else None
    threshold = contract['gates']['for_each_AA_and_C_vs_latest_per_frame_state']

    def gate(name, value, maximum, **context):
        original_gates.append(dict(name=name, value=value, threshold=maximum,
                                   passed=None if value is None else value <= maximum, **context))

    seams = {(r['raw'], r['state'], r['variant'], r['kind'], r['edge'], r['tangent_bin']): r for r in seam_rows}
    for name in list(variants)[1:]:
        for raw in RAWS:
            dm = domains(obs[raw], targets[raw])
            guard = ~obs[raw]['road'] & ~obs[raw]['road_edit'] & (obs[raw]['protect'] | dm['a5_RGB'] | dm['a6_RGB'] | dm['a37_RGB'])
            for state in STATES:
                b, c = cached[(raw, state, 'latest21K')], cached[(raw, state, name)]
                mapping = [('a5_RGB', 'a5_full_RGB_MAE_delta_max'), ('a5_upper_RGB', 'a5_upper_RGB_MAE_delta_max'),
                           ('a6_RGB', 'a6_RGB_MAE_delta_max'), ('protected_nonroad_RGB', 'protected_nonroad_RGB_MAE_delta_max'),
                           ('original_road_RGB', 'original_road_RGB_MAE_delta_max')]
                for domain, field in mapping + ([('a37_RGB', 'KEEP_a37_RGB_MAE_delta_max')] if state == 'keep' else []):
                    bv, cv = metric[(raw, state, 'latest21K', domain, 'original')], metric[(raw, state, name, domain, 'original')]
                    gate(name + ':' + domain, cv-bv if cv is not None else None, threshold[field], raw=raw, state=state, variant=name)
                measured = obs[raw]['depth_valid'][0]
                target = obs[raw]['depth_m'][0]
                bd, cd = means(np.abs(b['depth_m']-target), measured), means(np.abs(c['depth_m']-target), measured)
                gate(name + ':a6_depth_m', cd-bd if cd is not None else None, threshold['a6_measured_depth_MAE_delta_max_m'], raw=raw, state=state, variant=name)
                gate(name + ':nonroad_guard:alpha_pixel_max_abs', float(np.abs(c['alpha']-b['alpha'])[guard].max()), threshold['fixed_nonroad_alpha_pixel_max_abs_delta'], raw=raw, state=state, variant=name)
                gate(name + ':nonroad_guard:depth_pixel_max_abs_m', float(np.abs(c['depth_m']-b['depth_m'])[guard].max()), threshold['fixed_nonroad_depth_pixel_max_abs_delta_m'], raw=raw, state=state, variant=name)
                for t in [.9, .5]:
                    gate(name + ':new_a6_holes_alpha' + str(t), int((dm['a6_RGB'] & (b['group_alpha_T'][0] >= t) & (c['group_alpha_T'][0] < t)).sum()), 0, raw=raw, state=state, variant=name)
        for state in STATES:
            for edge in ['near_x', 'lateral_y']:
                def pool(variant):
                    return float(np.mean([r['score32'] for r in seam_rows if r['variant'] == variant and r['state'] == state and r['kind'] == 'world' and r['edge'] == edge]))
                gate(name + ':' + edge + ':equal_bin_pooled', pool(name)-pool('latest21K'), 1e-6, state=state, variant=name)
            def primary(variant):
                return float(np.mean([r['score32'] for r in seam_rows if r['variant'] == variant and r['state'] == state and r['kind'] == 'primary128' and r['edge'] == 'near_x']))
            b, c = primary('latest21K'), primary(name)
            gate(name + ':raw128:primary_near_no_regression', c-b, 1e-6, state=state, variant=name)
            gate(name + ':raw128:primary_near_absolute_reduction', c-b, -1e-4, state=state, variant=name)
            gate(name + ':raw128:primary_near_relative_reduction', (c-b)/b, -.05, state=state, variant=name)
    l_gates = [r for r in original_gates if r['variant'] == 'L']
    def gate_key(r):
        return r['name'], r.get('raw'), r.get('state')
    golden = {gate_key(r): r for r in original_L['numeric_gates']}
    assert len(l_gates) == len(golden) == 94
    for r in l_gates:
        old = golden[gate_key(r)]
        assert r['threshold'] == old['threshold'] and r['passed'] == old['passed'], (r, old)
        assert r['value'] == old['value'] if r['value'] is None else abs(r['value']-old['value']) < 1e-12, (r, old)
    pools, bins = [], []
    for name in list(variants)[1:]:
        scores = {}
        for registered in operator['seam_records']:
            key = (registered['raw'], registered['state'], name, registered['kind'], registered['edge'], registered['tangent_bin'])
            actual = seams[key]
            assert actual['minus_pixels'] == registered['minus_pixels'] and actual['plus_pixels'] == registered['plus_pixels']
            scores[registered['operator_row']] = actual['score64']
        for q in cert['per_bin_constraints']:
            score = scores[q['operator_row']]
            bins.append(dict(variant=name, operator_row=q['operator_row'], raw=q['raw'], state=q['state'],
                             kind=q['kind'], edge=q['edge'], tangent_bin=q['tangent_bin'],
                             actual_score=score, original_target=q['original_no_regression_target'],
                             passed=score <= q['original_no_regression_target']))
        for q in cert['pools']:
            score = float(np.mean([scores[i] for i in q['rows']]))
            pools.append(dict(variant=name, name=q['name'], state=q['state'], bins=q['bins'],
                              actual_score=score, original_target=q['original_target'], passed=score <= q['original_target']))
    asset = asset_comparison(args.reference_asset, args.candidate_asset) if args.reference_asset and args.candidate_asset else None
    assert bool(args.reference_asset) == bool(args.candidate_asset)
    counts = {}
    for name in list(variants)[1:]:
        g = [r for r in original_gates if r['variant'] == name]
        b = [r for r in bins if r['variant'] == name]
        p = [r for r in pools if r['variant'] == name]
        counts[name] = dict(original94=dict(PASS=sum(r['passed'] is True for r in g), FAIL=sum(r['passed'] is False for r in g), unsupported=sum(r['passed'] is None for r in g)),
                            K_six_pools_pass=sum(r['passed'] for r in p), K106_pass=sum(r['passed'] for r in b),
                            geometry_semantic_colour_byte_exact_to_Cwarm=True)
    summary = dict(schema='SAVED_FOURFRAME_MATERIAL_PROTECTION_CPU_V1', status='COMPLETE_INDEPENDENT_CPU_AUDIT',
                   raw_ids=RAWS, states=STATES, variants={k: str(v) for k, v in variants.items()}, results=counts,
                   original94_golden_L_reproduction=True, original94_threshold_changes=False,
                   fixed_world_seam_support=registered_support,
                   fixed_affected_subset='Original registered domain intersect fixed Cwarm group_alpha_T[7]>0; supplementary only',
                   point_registry=[dict(raw=r, x=x, y=y, role=role) for r, x, y, role in POINTS],
                   points_were_fixed_from_old_L_before_new_candidate=True,
                   rendered_invariants=FROZEN, serialized_asset_nonRGB_exact=bool(asset) if asset is not None else None,
                   crop_display='Saved native PNG where present; otherwise saved float RGB with display-only clip[0,1] and round255. Metrics never use PNGs.',
                   candidate_source_identity_requires_own_runtime_tuple_and_model_receipt=True,
                   GPU_forwards=0, backwards=0, optimizer_updates=0, models_loaded=False,
                   quality_accepted=False, human_verdict=None,
                   limits=['All four frames are BUILD. No49 candidate claim.',
                           'Original94 includes topology differences vs21K; material scope and old geometry failures stay separate.',
                           'Saved geometry/semantic RGB exactness is compared to fixed H2 Cwarm, not the original7group topology.',
                           'New-domain subset does not replace any old gate or grant a hidden-GT claim.'])
    out = args.output.resolve()
    out.mkdir(parents=True)
    csv_write(out / 'domain_metrics.csv', domain_rows)
    csv_write(out / 'rendered_frozen_invariants.csv', invariants)
    csv_write(out / 'registered_points.csv', points)
    csv_write(out / 'original94_gates.csv', original_gates)
    csv_write(out / 'K_six_pools.csv', pools)
    csv_write(out / 'K106_bins.csv', bins)
    if asset:
        csv_write(out / 'serialized_asset_invariants.csv', asset)
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    draw_distribution(out, cached, obs, targets, variants)
    draw_details(out, variants)
    draw_metrics(out, domain_rows, list(variants))
    architecture = '<svg xmlns="http://www.w3.org/2000/svg" width="1060" height="140" viewBox="0 0 1060 140"><defs><marker id="a" markerWidth="9" markerHeight="9" refX="8" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#335966"/></marker></defs><rect width="1060" height="140" fill="#f4f6f8"/><g fill="white" stroke="#335966" stroke-width="2"><rect x="10" y="20" width="240" height="95" rx="8"/><rect x="315" y="20" width="210" height="95" rx="8"/><rect x="590" y="20" width="210" height="95" rx="8"/><rect x="865" y="20" width="185" height="95" rx="8"/></g><g stroke="#335966" stroke-width="2" marker-end="url(#a)"><path d="M250 67 H310"/><path d="M525 67 H585"/><path d="M800 67 H860"/></g><g font-family="Arial" font-size="16" text-anchor="middle" fill="#153342"><text x="130" y="48">Saved 4-frame KEEP / DELETE</text><text x="130" y="74">GT + fixed masks + Cwarm</text><text x="130" y="98">21K / 2K / K / L / candidate</text><text x="420" y="52">CPU arrays only</text><text x="420" y="81">Byte exact frozen checks</text><text x="695" y="52">Original94 + K6 / 106</text><text x="695" y="81">Full + road-affected MAE</text><text x="958" y="52">CSV + fixed crops</text><text x="958" y="81">Separate root verdict</text></g></svg>'
    (out / 'architecture.svg').write_text(architecture, encoding='utf-8')
    body = f'<!doctype html><html><meta charset="utf-8"><title>Independent material protection audit</title><style>body{{font:16px Arial;max-width:1450px;margin:28px auto;color:#153342}}img{{max-width:100%;height:auto}}pre{{background:#f4f6f8;padding:16px}}a{{color:#156280}}</style><h1>Independent saved four-frame material protection audit</h1><p>原注册94门保留；材料相对同一 Cwarm 的冻结几何/语义色与新保护增量分开。所有图为既有真实渲染的固定显示，未新增前向或训练。</p><img src="architecture.svg"><pre>{html.escape(json.dumps(counts,indent=2))}</pre><p>完整原域和固定道路受影响子域分别计分，后者不替代任何旧域。邻暗点来自旧L输出，先于新候选固定。完整质量与人工结论仍为未接受。</p><p><a href="domain_metrics.csv">Full domain metrics CSV</a> · <a href="original94_gates.csv">Original94 gates</a> · <a href="rendered_frozen_invariants.csv">Frozen rendered invariants</a> · <a href="registered_points.csv">Fixed points</a> · <a href="K_six_pools.csv">K6 pools</a> · <a href="K106_bins.csv">K106 bins</a></p><h2>Fixed mask distribution</h2><img src="protection_distribution4.png"><h2>Full four-frame MAE</h2><img src="fourframe_domain_MAE.png"><h2>Fixed native crops</h2><img src="native_details.png"></html>'
    (out / 'audit.html').write_text(body, encoding='utf-8')
    print(json.dumps(dict(output=str(out), results=counts, golden_L94=True), ensure_ascii=False))


if __name__ == '__main__':
    main()
