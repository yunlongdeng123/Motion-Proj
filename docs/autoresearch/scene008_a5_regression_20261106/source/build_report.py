"""Rebuild the Scene008 a5 report from retrieved evidence, without running models.

Usage: python -B source/build_report.py
Only writes index.html, figures/, comparison.csv and report_data.json under the
report root. Missing evaluations remain visibly pending. Historical four-frame
receipts are never promoted to fresh all49 measurements.
"""
import argparse
import csv
import hashlib
import html
import json
import math
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ANCHORS = (89, 97, 105, 109)
CROP = (0, 160, 175, 320)
PIXELS = 121128
DEFINITIONS = (
    ('full5', 'Full5 B30K（既有参考）', 'reference_full5_B30k', 'full5_30k'),
    ('single10k', '仅 a37 · 原10K', 'reference_single10k', 'single10k'),
    ('dual5k', '完整 a37+a6 · 5K', 'dual5k', None),
    ('dual10k', '完整 a37+a6 · 10K', 'dual10k', None),
    ('current10k', 'Current 原生10K（沿革）', None, 'current10k'),
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def fmt(value, digits=5):
    return f'{value:.{digits}f}' if finite_number(value) else '—'


def h(value):
    return html.escape(str(value), quote=True)


class Builder:
    def __init__(self, root):
        self.root = root
        self.evidence = root / 'evidence'
        self.consumed = set()
        self.errors = []
        self.figure_records = []

    def relative(self, path):
        return Path(path).resolve().relative_to(self.root).as_posix()

    def read(self, path):
        path = Path(path)
        if not path.is_file():
            return None
        self.consumed.add(path.resolve())
        try:
            def reject(value):
                raise ValueError('non-finite JSON constant: ' + value)
            return json.loads(path.read_text(encoding='utf-8-sig'), parse_constant=reject)
        except (ValueError, OSError) as error:
            self.errors.append(dict(path=self.relative(path), error=str(error)))
            return None

    def link(self, path, label):
        path = Path(path)
        if not path.is_absolute():
            path = self.root / path
        if path.exists():
            return f'<a href="{quote(self.relative(path), safe="/._-")}">{h(label)}</a>'
        return f'<span class="missing">{h(label)}：待取回</span>'

    def dataset(self, identifier, label, fresh_folder, legacy_folder):
        aliases = {
            'reference_single10k': ('reference_a5_single10k_retry1',),
            'dual5k': ('a5_evaluation_5k',),
            'dual10k': ('a5_evaluation_10k',),
        }
        if fresh_folder:
            candidates = (fresh_folder,) + aliases.get(fresh_folder, ())
            fresh_folder = next((name for name in candidates if (self.evidence / name / 'summary.json').is_file()),
                                next((name for name in candidates if (self.evidence / name).is_dir()), fresh_folder))
        folder = self.evidence / fresh_folder if fresh_folder else None
        summary = self.read(folder / 'summary.json') if folder else None
        integrity = self.read(folder / 'source_integrity.json') if folder else None
        legacy = self.evidence / legacy_folder if legacy_folder else None
        receipt = self.read(legacy / 'receipt.json') if legacy else None
        problems = []
        if summary is not None:
            if not str(summary.get('status', '')).startswith('PASS'):
                problems.append('评价状态未通过')
            if summary.get('frame_count') != 49 or summary.get('raw_ids') != list(range(89, 138)):
                problems.append('不是完整原49帧')
            domain = summary.get('fixed_denominator', {})
            if domain.get('pixels') != PIXELS or summary.get('all49', {}).get('pixels') != PIXELS:
                problems.append('固定可见分母不符')
            if len(str(domain.get('sha256', ''))) != 64:
                problems.append('固定域哈希缺失')
            if not summary.get('camera_checks', {}).get('all49_passed'):
                problems.append('相机/时间核对未通过')
            for section in ('all49', 'boundary_all49'):
                if not finite_number(summary.get(section, {}).get('keep_rgb_mae')):
                    problems.append(section + ' RGB指标不完整')
            if integrity is None or not str(integrity.get('status', '')).startswith('PASS'):
                problems.append('源文件完整性回执未通过/待取回')
        validated = summary is not None and not problems
        status = ('49帧已核验' if validated else '49帧待复核：' + '；'.join(problems)
                  if summary is not None else '仅历史4帧；49帧待取回'
                  if receipt is not None and fresh_folder else '仅历史4帧沿革'
                  if receipt is not None else '待评价/取回')
        return dict(id=identifier, label=label, fresh_folder=fresh_folder,
                    legacy_folder=legacy_folder, folder=folder, legacy=legacy,
                    summary=summary, integrity=integrity, legacy_receipt=receipt,
                    validated=validated, status=status, problems=problems)

    def load_datasets(self):
        datasets = [self.dataset(*entry) for entry in DEFINITIONS]
        domains = [item['summary']['fixed_denominator']['sha256'] for item in datasets if item['validated']]
        if len(set(domains)) > 1:
            for item in datasets:
                if item['validated']:
                    item['validated'] = False
                    item['status'] = '固定域哈希跨模型不一致，暂停数字比较'
                    item['problems'].append('跨模型固定域哈希不一致')
        return datasets

    def image_path(self, dataset, raw, state):
        folder = dataset['folder']
        summary = dataset['summary'] or {}
        candidates = []
        if folder:
            if state == 'delete':
                delete_name = summary.get('delete_name')
                if isinstance(delete_name, str) and Path(delete_name).name == delete_name:
                    candidates.append(folder / 'frames' / f'{raw:03d}' / f'{delete_name}.png')
                # Stable native static output, where a result used a generic name.
                candidates += [folder / 'frames' / f'{raw:03d}' / name
                               for name in ('delete.png', 'delete_a37.png', 'delete_a37_a6.png')]
            else:
                candidates.append(folder / 'frames' / f'{raw:03d}' / f'{state}.png')
        legacy = dataset['legacy']
        if legacy:
            if dataset['id'] == 'full5':
                name = f'raw{raw}_keep.png' if state == 'keep' else f'raw{raw}_delete_a37_a6.png'
                if state != 'gt':
                    candidates.append(legacy / name)
            else:
                name = f'raw{raw}_gt.png' if state == 'gt' else f'raw{raw}_native_{state}.png'
                candidates.append(legacy / name)
        return next((path for path in candidates if path.is_file()), None)

    def load_image(self, path):
        if path is None:
            return None
        self.consumed.add(path.resolve())
        with Image.open(path) as source:
            if source.size != (480, 320):
                raise ValueError(f'Not a native 480x320 anchor: {path}: {source.size}')
            return source.convert('RGB').copy()

    def make_figures(self, datasets):
        figures = self.root / 'figures'
        figures.mkdir(exist_ok=True)
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 15) if Path('C:/Windows/Fonts/arial.ttf').is_file() else ImageFont.load_default()
        small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 12) if Path('C:/Windows/Fonts/arial.ttf').is_file() else ImageFont.load_default()
        names = ['Observed GT', 'Full5 B30K', 'a37 original10K', 'a37+a6 fresh5K', 'a37+a6 fresh10K', 'Current10K']
        for raw in ANCHORS:
            gt_path = next((self.image_path(item, raw, 'gt') for item in datasets
                            if self.image_path(item, raw, 'gt') is not None), None)
            # No clean-delete full-image GT exists. Full5 selective ownership
            # replay is not the requested old subset training; native all-five
            # deletion would remove a5 itself. Neither belongs in the main row.
            pairs = [(gt_path, None)] + [(self.image_path(item, raw, 'keep'),
                                         None if item['id'] == 'full5' else self.image_path(item, raw, 'delete')) for item in datasets]
            width, height = 64 + 6 * 195, 440
            canvas = Image.new('RGB', (width, height), '#f3f7f9')
            draw = ImageDraw.Draw(canvas)
            draw.text((14, 12), f'raw{raw} | fixed crop (0,160)-(175,320) | original pixels, no image enhancement', fill='#193b4a', font=font)
            draw.text((6, 118), 'KEEP', fill='#193b4a', font=small)
            draw.text((6, 316), 'DELETE', fill='#193b4a', font=small)
            sources = []
            assert len(names) == len(pairs)
            for index, (name, pair) in enumerate(zip(names, pairs)):
                left = 64 + index * 195
                draw.text((left + 7, 42), name, fill='#193b4a', font=font)
                for row, path in enumerate(pair):
                    top = 67 + row * 185
                    image = self.load_image(path)
                    if image is None:
                        draw.rectangle((left + 6, top, left + 181, top + 160), fill='#e5ebee', outline='#c3d0d6')
                        if index == 0 and row == 1:
                            draw.text((left + 12, top + 68), 'No clean-delete GT', fill='#627680', font=font)
                        elif index == 1 and row == 1:
                            draw.text((left + 12, top + 58), 'KEEP reference only', fill='#627680', font=font)
                            draw.text((left + 24, top + 81), 'not comparable', fill='#627680', font=font)
                        else:
                            draw.text((left + 22, top + 68), 'Not retrieved', fill='#627680', font=font)
                    else:
                        canvas.paste(image.crop(CROP), (left + 6, top))
                    sources.append(dict(column=name, state='keep' if row == 0 else 'delete',
                                        path=self.relative(path) if path else None))
            draw.text((72, 418), 'DELETE: no clean GT; Full5 is KEEP-only. Subset models remove their own declared pooled actors.', fill='#526d7a', font=small)
            path = figures / f'raw{raw:03d}_comparison.png'
            canvas.save(path)
            self.figure_records.append(dict(raw=raw, path=self.relative(path), size=[width, height],
                                            crop=list(CROP), sources=sources))

    def old_anchor_rows(self, datasets):
        rows = []
        for item in datasets:
            receipt = item['legacy_receipt'] or {}
            for row in receipt.get('rows', []):
                stage = row.get('stage', row.get('operation'))
                if row.get('raw') not in ANCHORS or stage not in ('keep', 'delete_a37_a6', 'native_keep', 'native_delete'):
                    continue
                rows.append(dict(model=item['label'], raw=row['raw'], stage=stage,
                                 pixels=row.get('a5_pixels', row.get('a5_label1_pixels')),
                                 rgb_mae=row.get('a5_RGB_MAE'), alpha=row.get('a5_alpha_mean'),
                                 change=row.get('a5_RGB_change_MAE_vs_keep', row.get('a5_delta_MAE'))))
        return rows

    def receipt_records(self):
        directory = self.evidence / 'run_receipts'
        rows = []
        if not directory.is_dir():
            return rows
        for path in sorted(directory.rglob('*.json')):
            data = self.read(path)
            if not isinstance(data, dict):
                continue
            fields = {key: data[key] for key in ('status', 'mode', 'arm', 'iteration', 'updates', 'actual_updates',
                                                'training_updates', 'seconds', 'elapsed_seconds')
                      if key in data and isinstance(data[key], (str, int, float, bool, type(None)))}
            rows.append(dict(path=self.relative(path), fields=fields))
        return rows

    def upper25_metrics(self, dataset):
        """Compute a fixed original-mask upper-band metric from saved floats."""
        if not dataset['validated']:
            return None
        paths = [dataset['folder'] / 'a5_float' / f'{raw:03d}.npz' for raw in range(89, 138)]
        if not all(path.is_file() for path in paths):
            return None
        count = total = 0
        sums = dict(keep_rgb_error=0., delete_rgb_error=0., change=0., alpha=0.)
        domain_hash, gt_hash = hashlib.sha256(), hashlib.sha256()
        for raw, path in enumerate(paths, 89):
            self.consumed.add(path.resolve())
            with np.load(path, allow_pickle=False) as data:
                yx = data['yx']
                keep, deleted, gt = [np.asarray(data[key], dtype=np.float64) for key in ('keep_rgb', 'delete_rgb', 'gt_rgb')]
                alpha = data['keep_alpha']
            if yx.shape != (len(gt), 2) or keep.shape != gt.shape or deleted.shape != gt.shape or gt.shape[1:] != (3,):
                raise ValueError(f'Invalid fixed-pixel float schema: {path}')
            if not all(np.isfinite(value).all() for value in (keep, deleted, gt, alpha)):
                raise ValueError(f'Nonfinite saved float: {path}')
            total += len(gt)
            gt_hash.update(np.asarray(raw, dtype='<i2').tobytes())
            gt_hash.update(np.ascontiguousarray(gt, dtype='<f4').tobytes())
            domain_hash.update(np.asarray(raw, dtype='<i2').tobytes())
            if not len(gt):
                continue
            select = yx[:, 0] <= yx[:, 0].min() + .25 * (yx[:, 0].max()-yx[:, 0].min())
            domain_hash.update(np.ascontiguousarray(yx[select], dtype='<i2').tobytes())
            a, b, g = np.clip(keep[select], 0, 1), np.clip(deleted[select], 0, 1), gt[select]
            count += int(select.sum())
            sums['keep_rgb_error'] += float(np.abs(a-g).mean(1).sum())
            sums['delete_rgb_error'] += float(np.abs(b-g).mean(1).sum())
            sums['change'] += float(np.abs(a-b).mean(1).sum())
            sums['alpha'] += float(alpha[select].astype(np.float64).sum())
        if total != PIXELS or not count:
            raise ValueError('Engineering upper25 metric lost the original121128 fixed pixels')
        return dict(pixels=count, full_pixel_frames=total, keep_rgb_mae=sums['keep_rgb_error']/count,
                    delete_rgb_mae=sums['delete_rgb_error']/count, keep_delete_rgb_change=sums['change']/count,
                    keep_mean_alpha=sums['alpha']/count, upper25_indices_sha256=domain_hash.hexdigest(),
                    gt_rgb_sha256=gt_hash.hexdigest(),
                    rule='Intersection of original a5 mask with the upper25% HEIGHT of that mask bounding box; not the top25% pixel count, not render-alpha selection, not semantic roof GT')

    def make_engineering_figure(self, original, baseline, fixed, kind='time_shift'):
        paths = [self.image_path(original, 105, 'gt'), self.image_path(original, 105, 'keep'),
                 self.image_path(baseline, 105, 'keep'), self.image_path(fixed, 105, 'keep')]
        names = ('Observed GT', 'Original single10K', 'Unpatched11977',
                 'Screen-only11977' if kind == 'screen_radius' else 'Time-only11977')
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 15) if Path('C:/Windows/Fonts/arial.ttf').is_file() else ImageFont.load_default()
        canvas = Image.new('RGB', (780, 228), '#f3f7f9')
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 8), 'raw105 | engineering comparison only | fixed crop, original pixels', fill='#193b4a', font=font)
        for index, (path, name) in enumerate(zip(paths, names)):
            left = index * 195
            draw.text((left + 8, 37), name, fill='#193b4a', font=font)
            image = self.load_image(path)
            if image is None:
                draw.rectangle((left+8, 60, left+183, 220), fill='#e5ebee', outline='#c3d0d6')
                draw.text((left+22, 128), 'Pending replay', fill='#627680', font=font)
            else:
                canvas.paste(image.crop(CROP), (left+8, 60))
        path = self.root / f'figures/raw105_{kind}_engineering.png'
        canvas.save(path)
        return dict(path=self.relative(path), size=[780, 228], crop=list(CROP),
                    sources=[self.relative(path) if path else None for path in paths])

    def engineering_data(self, datasets):
        original = next(item for item in datasets if item['id'] == 'single10k')
        baseline = self.dataset('smoke11977', '未修11977（既存smoke1977步）', 'single_smoke11977', None)
        fixed = self.dataset('candidate11977', 'time-only11977（隔离1977步回放）', 'time_shift_fix/fixed', None)
        pair = self.read(self.evidence / 'time_shift_fix/paired_result.json')
        isolated_contract = self.read(self.evidence / 'time_shift_fix/isolated_fix_contract.json')
        shift = self.read(self.evidence / 'static_time_shift_cpu_probe.json')
        screen = self.read(self.evidence / 'screen_prune_cpu_delivery_probe.json')
        source_audit = self.read(self.evidence / 'single_bug_source_audit.json')
        checkpoint = self.read(self.evidence / 'single_bug_checkpoint_audit.json')
        completed_accounting = self.read(self.evidence / 'run_receipts/dual/execution_accounting_final.json')
        rows = []
        for item in (original, baseline, fixed):
            rows.append(dict(id=item['id'], label=item['label'], validated=item['validated'], status=item['status'],
                             summary=item['summary'] if item['validated'] else None, upper25=self.upper25_metrics(item)))
        alignment = dict(same_initial_complete_state=(pair or {}).get('same_initial_complete_state'),
                         same_camera_shift=(pair or {}).get('identical_camera_time_shift_schedule', (pair or {}).get('same_camera_shift')),
                         compared_updates=(pair or {}).get('compared_updates'))
        errors = []
        if pair is not None:
            if not str(pair.get('status', '')).startswith('COMPLETE'):
                errors.append('paired_result未完成')
            if alignment['same_initial_complete_state'] is not True or alignment['same_camera_shift'] is not True:
                errors.append('同起点或相机/shift断言缺失/失败')
            if alignment['compared_updates'] != 1977:
                errors.append('配对更新数不是1977')
            if pair.get('baseline_endpoint') != 11977 or pair.get('fixed_endpoint') != 11977:
                errors.append('端点不是共同11977')
            if not baseline['validated'] or not fixed['validated']:
                errors.append('两端49帧或完整性未齐')
            elif baseline['summary']['fixed_denominator'] != fixed['summary']['fixed_denominator']:
                errors.append('两端固定域不同')
            a, b = rows[1]['upper25'], rows[2]['upper25']
            if a is None or b is None:
                errors.append('上25%原始float待取回')
            elif (a['upper25_indices_sha256'], a['gt_rgb_sha256']) != (b['upper25_indices_sha256'], b['gt_rgb_sha256']):
                errors.append('两端上部固定域或原观测RGB不同')
        ready = pair is not None and not errors
        delta = None
        if ready:
            a, b = rows[1], rows[2]
            delta = dict(all49_keep_rgb_mae=b['summary']['all49']['keep_rgb_mae']-a['summary']['all49']['keep_rgb_mae'],
                         boundary_keep_rgb_mae=b['summary']['boundary_all49']['keep_rgb_mae']-a['summary']['boundary_all49']['keep_rgb_mae'],
                         upper25_keep_rgb_mae=b['upper25']['keep_rgb_mae']-a['upper25']['keep_rgb_mae'])
        return dict(status='PAIRED_CANDIDATE_REPLAY_VERIFIED' if ready else 'CANDIDATE_REPLAY_PENDING_OR_NOT_PAIRED',
                    paired_ready=ready, pair_assertions=alignment, pair_errors=errors,
                    adoption_decision='NOT_ADOPTED_NEGATIVE_A5_REPLAY' if ready and delta['all49_keep_rgb_mae'] >= 0 else
                                      'NO_AUTOMATIC_ADOPTION_FROM_ENGINEERING_PASS' if ready else 'PENDING_COMPLETE_PAIR',
                    paired_result=pair, isolated_contract=isolated_contract, rows=rows, delta_fixed_minus_baseline=delta,
                    shift_cpu_proof=shift, screen_prune_cpu_proof=screen,
                    source_audit_status=(source_audit or {}).get('status'),
                    checkpoint_summary=dict(topology=(checkpoint or {}).get('topology'),
                        raw105_roof_family_overlap=(checkpoint or {}).get('raw105_roof_family_overlap'),
                        native_fixed_anchor_metrics=(checkpoint or {}).get('native_fixed_anchor_metrics')),
                    engineering_accounting=dict(original_unpatched_smoke_updates=1977,
                        candidate_real_smoke=32, candidate_restore_probe=2, candidate_replay=1977,
                        candidate_new_updates=2011, formal_updates_added=0,
                        prior_completed_engineering_updates=(completed_accounting or {}).get('smoke_updates_total'),
                        execution_claim='complete receipts' if ready else 'planned; do not treat as all executed'),
                    completed_accounting=completed_accounting,
                    figure=self.make_engineering_figure(original, baseline, fixed))

    def screen_radius_data(self, datasets):
        original = next(item for item in datasets if item['id'] == 'single10k')
        baseline = self.dataset('smoke11977', '未修11977（共同旧基线）', 'single_smoke11977', None)
        fixed = self.dataset('screen11977', 'screen-only11977（独立1977回放）', 'screen_radius_fix/fixed', None)
        pair = self.read(self.evidence / 'screen_radius_fix/paired_result.json')
        contract = self.read(self.evidence / 'screen_radius_fix/isolated_fix_contract.json')
        cpu = self.read(self.evidence / 'screen_radius_candidate_cpu_test.json')
        global49 = self.read(self.evidence / 'screen_radius_fix/global49_comparison.json')
        smoke = self.read(self.evidence / 'screen_radius_fix/smoke_summary.json')
        topology_coverage = self.read(self.evidence / 'screen_radius_fix/smoke_topology_coverage.json')
        rows = []
        for item in (baseline, fixed):
            upper = self.engineering['rows'][1]['upper25'] if item['id'] == 'smoke11977' else self.upper25_metrics(item)
            rows.append(dict(id=item['id'], label=item['label'], validated=item['validated'], status=item['status'],
                             summary=item['summary'] if item['validated'] else None, upper25=upper))
        alignment = dict(same_initial_complete_state=(pair or {}).get('same_initial_complete_state'),
                         same_camera_shift=(pair or {}).get('identical_camera_time_shift_schedule', (pair or {}).get('same_camera_shift')),
                         compared_updates=(pair or {}).get('compared_updates'))
        original_renderer = '35b2863da1bf9173812207dbd06fbbecdb88929f95b50a7968db3b5b2aa60ced'
        renderer = 'gaussian_renderer/__init__.py'
        source = (contract or {}).get('source_sha256', {})
        isolated = (contract or {}).get('isolated_engine_sha256', {})
        core = ('train.py', 'gaussian_renderer/__init__.py', 'scene/__init__.py', 'scene/gaussian_model.py',
                'scene/cameras.py', 'scene/paper_sgcm.py', 'utils/hybrid30k_losses.py',
                'utils/hybrid30k_runtime.py', 'utils/paper_checkpoint.py')
        missing_core = [name for name in core if name not in source or name not in isolated]
        changed_other_core = [name for name in core if name != 'scene/gaussian_model.py' and
                              name in source and name in isolated and source[name] != isolated[name]]
        source_assertions = dict(changed_files=(contract or {}).get('changed_files'),
                                 original_renderer_sha256=source.get(renderer),
                                 isolated_renderer_sha256=isolated.get(renderer),
                                 candidate_gaussian_model_sha256=isolated.get('scene/gaussian_model.py'),
                                 expected_candidate_gaussian_sha256=(cpu or {}).get('candidate_sha256'),
                                 native_core_files=list(core), missing_native_core_files=missing_core,
                                 unexpectedly_changed_native_core=changed_other_core,
                                 unrelated_omitted_auxiliary_directories_allowed=True)
        errors = []
        if pair is not None:
            if not str(pair.get('status', '')).startswith('COMPLETE'):
                errors.append('screen paired_result未完成')
            if alignment['same_initial_complete_state'] is not True or alignment['same_camera_shift'] is not True:
                errors.append('screen同起点或相机/shift断言缺失/失败')
            if alignment['compared_updates'] != 1977 or pair.get('baseline_endpoint') != 11977 or pair.get('fixed_endpoint') != 11977:
                errors.append('screen不是1977步、共同11977端点')
            if not baseline['validated'] or not fixed['validated']:
                errors.append('screen两端49帧或来源完整性未齐')
            elif baseline['summary']['fixed_denominator'] != fixed['summary']['fixed_denominator']:
                errors.append('screen两端固定域不同')
            a, b = rows[0]['upper25'], rows[1]['upper25']
            if a is None or b is None:
                errors.append('screen上部原始float待取回')
            elif (a['upper25_indices_sha256'], a['gt_rgb_sha256']) != (b['upper25_indices_sha256'], b['gt_rgb_sha256']):
                errors.append('screen两端上部域或GT不同')
            if sorted((contract or {}).get('changed_files', [])) != ['scene/gaussian_model.py']:
                errors.append('screen缺少仅修改gaussian_model的隔离合同')
            if source.get(renderer) != original_renderer or isolated.get(renderer) != original_renderer:
                errors.append('screen renderer不是两端原35b，不能排除叠加time改动')
            if missing_core or changed_other_core:
                errors.append('screen原生核心文件缺失或有额外改动')
            if not cpu or not str(cpu.get('status', '')).startswith('PASS') or cpu.get('case_count') != 18 or isolated.get('scene/gaussian_model.py') != cpu.get('candidate_sha256'):
                errors.append('screen候选源码未与CPU18案例版本绑定')
            if pair.get('additional_engineering_updates') != 2107 or pair.get('added_formal_updates') != 0:
                errors.append('screen2107工程/正式0的记账未核对')
            if not str((smoke or {}).get('status', '')).startswith('PASS') or (smoke or {}).get('smoke_updates') != 128 or (smoke or {}).get('resume_extra_updates') != 2:
                errors.append('screen真实128+2准入回执未核对')
            if not str((topology_coverage or {}).get('status', '')).startswith('PASS') or not (topology_coverage or {}).get('generic10100'):
                errors.append('screen首个10100变更节点覆盖未核对')
        ready = pair is not None and not errors
        delta = None
        if ready:
            a, b = rows
            delta = dict(all49_keep_rgb_mae=b['summary']['all49']['keep_rgb_mae']-a['summary']['all49']['keep_rgb_mae'],
                         boundary_keep_rgb_mae=b['summary']['boundary_all49']['keep_rgb_mae']-a['summary']['boundary_all49']['keep_rgb_mae'],
                         upper25_keep_rgb_mae=b['upper25']['keep_rgb_mae']-a['upper25']['keep_rgb_mae'])
        return dict(status='PAIRED_SCREEN_ONLY_REPLAY_VERIFIED' if ready else 'SCREEN_ONLY_REPLAY_PENDING_OR_NOT_PAIRED',
                    paired_ready=ready, pair_assertions=alignment, source_assertions=source_assertions, pair_errors=errors,
                    paired_result=pair, isolated_contract=contract, cpu_candidate_test=cpu,
                    global49_comparison=global49,
                    real_GPU_smoke=smoke, smoke_topology_coverage=topology_coverage,
                    rows=rows, delta_fixed_minus_baseline=delta,
                    adoption_decision='NOT_ADOPTED_NEGATIVE_A5_REPLAY' if ready and delta['all49_keep_rgb_mae'] >= 0 else
                                      'NO_AUTOMATIC_A5_ADOPTION_FROM_BOOKKEEPING_FIX' if ready else 'PENDING_COMPLETE_PAIR',
                    engineering_accounting=dict(real_smoke=128, restore_probe=2, replay=1977, new_engineering_updates=2107,
                                                formal_updates_added=0, execution_claim='complete receipts' if ready else 'planned; do not count pending as completed'),
                    figure=self.make_engineering_figure(original, baseline, fixed, kind='screen_radius'))

    def render_screen_radius(self, data):
        rows = []
        for row in data['rows']:
            if row['validated']:
                summary, upper = row['summary'], row['upper25']
                cells = [h(row['label']), fmt(summary['all49']['keep_rgb_mae']), fmt(summary['boundary_all49']['keep_rgb_mae']),
                         fmt((upper or {}).get('keep_rgb_mae')), fmt(summary['all49']['keep_delete_rgb_change'], 7)]
            else:
                cells = [h(row['label'])] + ['待运行/待取回'] * 4
            rows.append('<tr>'+''.join('<td>'+value+'</td>' for value in cells)+'</tr>')
        if data['paired_ready']:
            delta = data['delta_fixed_minus_baseline']
            state = 'screen-only配对、固定域和独立源码合同已核验。候选−未修 ΔMAE：全域 '+fmt(delta['all49_keep_rgb_mae'],7)+\
                    '，内边界 '+fmt(delta['boundary_keep_rgb_mae'],7)+'，原mask上部 '+fmt(delta['upper25_keep_rgb_mae'],7)+'。这是工程回放数值，不自动证明a5已修复。'
            if data['adoption_decision'] == 'NOT_ADOPTED_NEGATIVE_A5_REPLAY':
                relative = delta['all49_keep_rgb_mae']/data['rows'][0]['summary']['all49']['keep_rgb_mae']
                state = '负结果：screen-only全域KEEP MAE上升'+f'{relative:.1%}'+\
                        '，raw105仍缺顶、暗糊并有蓝色残影；该模型不采纳为a5修复。'+state
        else:
            state = 'screen-radius独立候选仍待运行或完整配对回执；不提前认定有效，也不将2107预算计作全部完成。'
            if data['pair_errors']:
                state += ' 当前未满足：'+'；'.join(data['pair_errors'])+'。'
        prior = self.engineering['engineering_accounting'].get('prior_completed_engineering_updates')
        complete = (prior or 0) + (2011 if self.engineering['paired_ready'] else 0) + (2107 if data['paired_ready'] else 0)
        accounting = '已有完整回执可确认的工程更新：'+str(complete)+'；运行中/缺回执部分另记，不等于其实际消耗为0。' if prior is not None else '既有工程总账待取回。'
        global49 = data.get('global49_comparison') or {}
        global_table = ''
        if all(isinstance(global49.get(role), dict) and global49[role].get('frames') == 49 and
               all(finite_number(global49[role].get(key)) for key in ('psnr', 'ssim', 'lpips', 'l1'))
               for role in ('baseline', 'candidate')):
            global_rows = []
            for role, label in (('baseline', '未修11977'), ('candidate', 'screen-only11977')):
                value = global49[role]
                global_rows.append('<tr>'+''.join('<td>'+cell+'</td>' for cell in
                    (label, fmt(value['psnr'], 4), fmt(value['ssim'], 5), fmt(value['lpips'], 5), fmt(value['l1'], 6)))+'</tr>')
            global_table = '<h3>全画面原生49训练视角重建</h3><div class="scroll"><table><thead><tr><th>独立工程端点</th><th>PSNR / dB ↑</th><th>SSIM ↑</th><th>LPIPS ↓</th><th>RGB L1 ↓</th></tr></thead><tbody>'+''.join(global_rows)+'</tbody></table></div><p>screen-only的全局重建也退化；正确修复统计窗口不等于该回放模型达到a5或全场景质量要求。这些仍是训练视角指标，不是独立泛化结果。</p>'
        links = [
            ('evidence/screen_radius_candidate_cpu_test.json', '候选18案例CPU核验'),
            ('source/probe_screen_radius_candidate_cpu.py', '候选CPU探针'),
            ('source/screen_radius_candidate.patch', 'screen-radius单变量patch'),
            ('evidence/screen_radius_fix/isolated_fix_contract.json', 'screen独立源码合同'),
            ('evidence/screen_radius_fix/paired_result.json', 'screen配对结果'),
            ('evidence/screen_radius_fix/fixed/summary.json', 'screen49帧评价'),
            ('evidence/screen_radius_fix/fixed/source_integrity.json', 'screen来源完整性'),
            ('evidence/screen_radius_fix/smoke_summary.json', 'screen真实128步与恢复准入'),
            ('evidence/screen_radius_fix/global49_comparison.json', '原生全局49指标与来源'),
        ]
        return f'''<section id="screen-radius"><h2>第二个独立代码验证：screen-radius窗口</h2>
<p>已复现的缺陷是普通densify先清零累计屏幕半径，令原max_screen_size裁剪失效。候选只在gaussian_model内两处调整：postfix保留旧行半径、新行置0；普通prune消费后清空窗口。prune_only及阈值None/0的原语义保留。CPU18案例通过仍不证明a5因果。</p>
<p class="muted">该半径次序继承旧实现，历史V82-F06已记录append会重置半径、不能冒称持续屏幕大点约束。本次18例CPU、128步真实GPU与1977步回放是补充其对a5恢复作用的证据，不是新引入commit或新科学发现；不改变原a6/RCR-1采用范围。</p>
<p><strong>这是从同一原10K另起的screen-only回放，不是在time候选上继续。</strong>renderer保持原35b版本；actor、20px阈值、损失、采样、学习率不变，不叠加time_shift改动。</p>
<p class="notice">{h(state)}</p><div class="scroll"><table><thead><tr><th>独立工程角色</th><th>全49 KEEP MAE</th><th>固定内边界 MAE</th><th>原mask上部 MAE</th><th>全49 KEEP↔DELETE变化</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p class="muted">使用与第一工程区相同的原mask固定域及包围框高度上25%交集；各候选分别核对完整起点、1977次相机/shift、原观测与源码隔离。两个候选的结果不进入正式5K/10K表。</p>
<figure class="scroll"><img src="{data['figure']['path']}" alt="raw105 screen-only独立候选与未修11977同裁剪KEEP比较"><figcaption>只展示raw105 KEEP原像素裁剪；screen结果缺项保留灰格。第一time图与负结果继续保留在上一节。</figcaption></figure>
{global_table}
<p>候选的18例CPU与真实GPU128步（覆盖首个10100变更节点）、完整恢复及下一入口可以通过工程验收；这不升级为a5质量通过。</p>
<p>新增工程预算128步真实smoke＋2步恢复＋1977步回放＝2107；首个10100增密节点需覆盖变更代码。正式新增仍只有Dual 10,000步，Single不追加。若两个候选均完整完成，工程总数为2481＋2011＋2107＝6599；这个条件总数不是pending部分的完成声明。</p>
<p>{h(accounting)}</p><details><summary>screen-only复现、patch与回执</summary><ul>{''.join('<li>'+self.link(path,label)+'</li>' for path,label in links)}</ul></details></section>'''

    def render_early_bounds(self):
        if self.early_bounds is None:
            return '<section><h2>最早形成与后续退化的因果边界</h2><p class="missing">最早时点与源码条件凭据待取回，尚不补造因果结论。</p></section>'
        return f'''<section id="causal-bounds"><h2>最早形成与后续退化：两个问题不能混为一谈</h2>
<p>已存Single 1K图就有车顶欠重建，尚无证据表明完整车顶曾先形成后被删除。static首次opacity reset在3K，9K reset最多影响后续恢复，不能单独解释1K现象。20px屏幕裁剪要求iteration&gt;3000且为100的倍数，最早3100启用，因此该bookkeeping缺陷也不能单独解释最早1K欠顶。</p>
<p>time-only从原10K出发的短回放负结果，只说明本轮候选未修好，不等于排除它从头训练时的贡献。screen统计缺陷可复现，但其修复模型仍失败；不能把代码正确性证明替代a5根因证明。</p>
<p>当前线索落在a5退出dynamic后，原始观测支撑、监督责任/权重和静态时间表示共同改变，尚不能唯一归因某个commit。两个代码候选短回放未修复a5。已完成的补真实支撑5K有有限改善但不足以恢复车顶；新研究分别检验原轨迹训练时长与全static域RGB监督强度，不预判训练条件的因果作用。</p>
<p>{self.link('evidence/early_causal_bounds.json', '最早时点与源码条件')} · {self.link('evidence/single_bug_source_audit.json', '支持范围与监督的源码证据')}</p></section>'''

    def support_ablation_data(self):
        preparation = self.read(self.evidence / 'a5_real_support_input_preparation.json') or {}
        support_contract = preparation.get('data', {}).get('support_contract.json', {}).get('value', {})
        diagnostic_comparison = self.read(self.evidence / 'support_round1_1k_comparison.json')
        comparison5k = self.read(self.evidence / 'support_round1_5k_comparison.json')
        diagnostic_rows = []
        for identifier,label in (('support_control1k','Control：原单车输入 1K'),('support_full_a5_1k','Support：真实a5支撑 1K')):
            item = self.dataset(identifier,label,identifier,None)
            diagnostic_rows.append(dict(id=identifier,label=label,folder=identifier,validated=item['validated'],
                                        status=item['status'],summary=item['summary'] if item['validated'] else None,
                                        upper25=self.upper25_metrics(item)))
        previews = []
        for node in (1000, 5000):
            control = self.evidence / ('support_control'+str(node//1000)+'k') / 'frames/105'
            support = self.evidence / ('support_full_a5_'+str(node//1000)+'k') / 'frames/105'
            sources = [next((folder/'gt.png' for folder in (control,support) if (folder/'gt.png').is_file()),None),
                       control/'keep.png', support/'keep.png']
            if not any(path is not None and path.is_file() for path in sources[1:]):
                continue
            canvas = Image.new('RGB', (585, 218), '#f3f7f9')
            draw = ImageDraw.Draw(canvas)
            font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 13) if Path('C:/Windows/Fonts/arial.ttf').is_file() else ImageFont.load_default()
            draw.text((8,5), f'raw105 | fresh {node} | KEEP | fixed native crop, no enhancement', fill='#193b4a', font=font)
            for index,(label,path) in enumerate(zip(('Observed GT','Control','Real a5 support'),sources)):
                draw.text((8+195*index,28),label,fill='#193b4a',font=font)
                image = self.load_image(path) if path is not None and path.is_file() else None
                if image is not None:
                    canvas.paste(image.crop(CROP),(8+195*index,50))
                else:
                    draw.rectangle((8+195*index,50,183+195*index,210),fill='#e5ebee')
                    draw.text((16+195*index,115),'Not retrieved',fill='#627680',font=font)
            target = self.root/'figures'/f'raw105_support_{node//1000}k.png'
            canvas.save(target)
            previews.append(dict(iteration=node,path=self.relative(target),size=list(canvas.size),
                                 crop=list(CROP),sources=[self.relative(path) if path is not None and path.is_file() else None for path in sources]))
        rows = []
        for identifier, label, folder in (
                ('support_control5k', 'Control：原单车输入 fresh5K', 'support_control5k'),
                ('support_full_a5_5k', 'Support：补齐真实a5源点 fresh5K', 'support_full_a5_5k')):
            item = self.dataset(identifier, label, folder, None)
            rows.append(dict(id=identifier, label=label, folder=folder, validated=item['validated'],
                             status=item['status'], summary=item['summary'] if item['validated'] else None,
                             upper25=self.upper25_metrics(item)))
        all_rows = diagnostic_rows+rows
        fixed_hashes = {row['summary']['fixed_denominator']['sha256'] for row in all_rows if row['validated']}
        upper_hashes = {(row['upper25']['upper25_indices_sha256'],row['upper25']['gt_rgb_sha256']) for row in all_rows if row['upper25']}
        if len(fixed_hashes)>1 or len(upper_hashes)>1:
            raise ValueError('Support-ablation fixed GT domains differ across arms or nodes')
        pair_assertions = dict(same_5000_camera_time_shift=(comparison5k or {}).get('identical_camera_time_shift_schedule') is True,
                               records_5000=(comparison5k or {}).get('records')==5000,
                               fixed_pixels_121128=(comparison5k or {}).get('fixed_mask_pixels')==PIXELS,
                               method_and_losses_unchanged=(comparison5k or {}).get('method_and_losses_unchanged') is True)
        pair_ready = all(row['validated'] for row in rows) and all(pair_assertions.values()) and str((comparison5k or {}).get('status','')).startswith('PASS')
        return dict(status='SUPPORT_5K_EVALUATIONS_RETRIEVED' if all(row['validated'] for row in rows) else 'SUPPORT_EXPERIMENT_RUNNING_OR_PENDING',
                    both_5k_retrieved=all(row['validated'] for row in rows), rows=rows,
                    method='Original unmodified engine; dynamic onlya37; a5 remains temporal static',
                    intended_single_factor='Append missing real a5 xyz+time observations; retain all original1496 sampled rows',
                    point_budget='Adds19916 real rows: native static591672→611588(+3.366%); total690124→710040(+2.886%). Not an equal-point-count geometry control',
                    input_preparation=preparation.get('status'), support_contract=support_contract,
                    preview_figures=previews,
                    paired5k_comparison=comparison5k,paired5k_assertions=pair_assertions,paired5k_ready=pair_ready,
                    raw105_5k_agent_observation='Both5K KEEP crops remain dark/incomplete with missing roof/windshield; support shows more outline but is not repaired. This is an anchor review, not all49 human review.' if pair_ready else None,
                    diagnostic1k=dict(rows=diagnostic_rows,paired_comparison=diagnostic_comparison,
                                      raw105_agent_observation='Both KEEP images show almost no recognizable blue a5 body at1K; not a5 restoration. Only this anchor was visually assessed here.' if all(row['validated'] for row in diagnostic_rows) else None),
                    diagnostic_nodes=[1000,5000], planned_fresh_updates_per_arm=5000,
                    learning_rate_horizon=30000, prediction='No repair effect assumed')

    def render_support_ablation(self):
        rows = []
        for row in self.support_ablation['diagnostic1k']['rows']+self.support_ablation['rows']:
            value = (row['summary'] or {}).get('all49', {})
            boundary = (row['summary'] or {}).get('boundary_all49', {})
            raw105 = next((entry['visible'] for entry in (row['summary'] or {}).get('frames',[]) if entry.get('raw')==105),{})
            rows.append('<tr><td>'+h(row['label'])+'</td><td>'+h(row['status'])+'</td><td>'+fmt(value.get('keep_rgb_mae'))+'</td><td>'+fmt(boundary.get('keep_rgb_mae'))+'</td><td>'+fmt((row.get('upper25') or {}).get('keep_rgb_mae'))+'</td><td>'+fmt(raw105.get('keep_rgb_mae'))+'</td><td>'+fmt(value.get('keep_delete_rgb_change'),7)+'</td></tr>')
        links = [
            ('evidence/support_control5k/summary.json', 'Control5K评价'),
            ('evidence/support_full_a5_5k/summary.json', 'Support5K评价'),
            ('evidence/support_control1k/anchors/105_crop.png', 'Control1K先看片'),
            ('evidence/support_full_a5_1k/anchors/105_crop.png', 'Support1K先看片'),
            ('evidence/a5_real_support_input_preparation.json', '真实支撑输入合同与CPU核对'),
            ('evidence/support_round1_1k_comparison.json', '1K实际日程与配对摘要'),
            ('evidence/support_round1_5k_comparison.json', '5K实际日程与配对摘要'),
            ('evidence/new_goal_authorization.json', '历史有界验证授权（已被新指令覆盖）'),
        ]
        authorization = ('<details><summary>本轮范围与资源收尾授权（以最新登记为准）</summary><pre>'+h(json.dumps(self.new_goal_authorization, ensure_ascii=False, indent=2))+'</pre></details>'
                         if self.new_goal_authorization else '')
        previews = ''.join('<figure><img src="'+item['path']+'" alt="raw105支撑初始化同节点KEEP比较"><figcaption>fresh '+str(item['iteration'])+'：GT / Control / Support，同175×160原像素窗口；这里只抽查raw105，不代表49帧人工验收。</figcaption></figure>' for item in self.support_ablation['preview_figures'])
        firstlook = ('<p>已实际核对1K raw105：两组KEEP中蓝色SUV均几乎尚未形成，数值略有变化不等于车顶或挡风玻璃恢复。继续原合同到5K，不将1K现象当作5K结论。</p>'
                     if all(row['validated'] for row in self.support_ablation['diagnostic1k']['rows']) else '')
        result = ''
        if self.support_ablation['paired5k_ready']:
            pair = self.support_ablation['paired5k_comparison']
            improvement = 1-pair['support']['all49']['keep_rgb_mae']/pair['control']['all49']['keep_rgb_mae']
            anchor_improvement = 1-pair['support']['raw105']['keep_rgb_mae']/pair['control']['raw105']['keep_rgb_mae']
            result = '<p><strong>Round1完成：补齐真实支撑有一定支持作用，但5K未修复a5。</strong>全49固定域误差下降'+f'{improvement:.1%}'+'，raw105仅下降'+f'{anchor_improvement:.1%}'+'；两张已目视的raw105车顶/挡风玻璃仍不完整，Support只增加了暗色轮廓。两组raw105总alpha≥0.5覆盖均为100%，仍不能代表车体完整。5000次实际camera/shift完全相同；这轮不能排除支撑与监督的联合作用。</p>'
            firstlook = firstlook.replace('继续原合同到5K，不将1K现象当作5K结论。','1K结论与5K终点评价分列。')
        heading = 'Round1：初始化支撑单变量实验（已完成5K）' if self.support_ablation['paired5k_ready'] else '正在执行：初始化支撑单变量实验'
        return f'''<section id="support-experiment"><h2>{heading}</h2>
<p>停止继续追代码改动，直接检验a5退出动态池后的真实观测支撑。两组都从头训练，使用<strong>原未修engine、仅a37动态、a5仍在static</strong>；Control沿用原single输入，Support只补入原a5完整21,358真实源点中缺失的xyz＋time观测，原sample的1,496行全部保留。</p>
<p>损失、采样、30K学习率horizon和删除语义保持原合同；1K先看片，5K终点。输入合同核对追加19,916真实行；原1,496行中有1,442个唯一xyz＋time，保留重复采样后，a5共21,412行、覆盖21,358个唯一真实观测。static初始池591,672→611,588（＋3.366%），全模型初始池690,124→710,040（＋2.886%）。这是支撑与局部密度/容量干预，<strong>不是同点数几何对照</strong>。原生KNN会随新增邻点重新计算初始scale，不主张旧参数逐项相同；也不先预判补点会修复。两个既有代码候选负结果保留在下方，不与此实验叠加。</p>
{result}<div class="scroll"><table><thead><tr><th>新实验组 / 节点</th><th>当前证据</th><th>全49 KEEP MAE</th><th>边界 KEEP MAE</th><th>固定上部 KEEP MAE</th><th>raw105 KEEP MAE</th><th>KEEP↔DELETE变化</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div><p class="muted">RGB MAE均为0–1通道均值、越小越好；边界7,530像素帧。固定上部是原始a5 mask包围盒高度上25%与该mask的交集（25,906像素帧），由固定GT确定，不按render选择，也不是语义车顶真值。</p>{firstlook}<div class="scroll">{previews}</div>
<p>{' · '.join(self.link(path,label) for path,label in links)}</p><p class="muted">先依据同相机1K/5K图片观察车顶与挡风玻璃；结果未到时不填写改善结论。新实验账本与上一代码诊断阶段的正式10000／工程6599分开记录。旧回执中的权限只描述当时现场，后续执行与资源收尾以最新授权登记为准。</p>{authorization}</section>'''

    def research_data(self, datasets):
        """New research results enter only through validated native evaluations."""
        by_id = {item['id']:item for item in datasets}
        trajectory = [(10000,by_id['single10k']),
                      (11977,self.dataset('single11977','原single · 11977工程节点','single_smoke11977',None))]
        for step in (15000,20000,30000):
            folder = 'single_longrun'+str(step//1000)+'k'
            trajectory.append((step,self.dataset(folder,'原single续训 · '+str(step//1000)+'K',folder,None)))
        duration = []
        cells = []
        early_cells = []
        all_items = [item for _,item in trajectory]
        for step,item in trajectory:
            duration.append(dict(iteration=step,id=item['id'],label=item['label'],folder=item['fresh_folder'],
                                 validated=item['validated'],status='未执行（范围调整；不下30K恢复结论）' if step==30000 and self.scope_change else item['status'],
                                 summary=item['summary'] if item['validated'] else None,
                                 upper25=self.upper25_metrics(item)))
        for support,weight,folder in ((False,1,'support_control5k'),(True,1,'support_full_a5_5k'),
                                     (False,5,'loss5_control5k'),(True,5,'loss5_full_a5_5k')):
            item = self.dataset(folder,('完整真实支撑' if support else '原采样支撑')+' · staticRGB='+str(weight),folder,None)
            all_items.append(item)
            cells.append(dict(initialization='appended_real_support' if support else 'original_sample',
                              static_rgb_scalar=weight,nominal_global_plus_static_rgb_scalar=1+weight,
                              label=item['label'],folder=folder,validated=item['validated'],status=item['status'],
                              summary=item['summary'] if item['validated'] else None,
                              upper25=self.upper25_metrics(item)))
            early_folder = folder.replace('5k','1k')
            early = self.dataset(early_folder,item['label']+' · 1K',early_folder,None)
            all_items.append(early)
            early_cells.append(dict(label=early['label'],folder=early_folder,validated=early['validated'],
                                    status=early['status'],summary=early['summary'] if early['validated'] else None,
                                    upper25=self.upper25_metrics(early)))
        domains = {item['summary']['fixed_denominator']['sha256'] for item in all_items if item['validated']}
        if len(domains)>1:
            raise ValueError('Research evaluations do not share the original fixed a5 domain')
        upper_domains = {(row['upper25']['upper25_indices_sha256'],row['upper25']['gt_rgb_sha256'])
                         for row in duration+cells+early_cells if row['upper25']}
        if len(upper_domains)>1:
            raise ValueError('Research upper-region GT/index domains differ')
        pairing = self.read(self.evidence / 'loss2x2_comparison.json')
        visual_review=self.read(self.evidence / 'research_agent_visual_review_20261107.json')
        for cell in cells:
            reviews=[row for row in (visual_review or {}).get('reviews',[]) if row.get('iteration')==5000 and
                     (row.get('image','').startswith(cell['folder']+'/') or
                      (cell['static_rgb_scalar']==1 and row.get('run')=='support_control_vs_full_a5'))]
            cell['agent_body_restored']=reviews[-1].get('agent_body_restored') if reviews else None
        by_cell = {(row['initialization'],row['static_rgb_scalar']):row for row in cells}
        effects = []
        for section,label in (('all49','全a5 MAE'),('boundary_all49','边界 MAE'),('upper25','固定上部 MAE')):
            def value(initialization,weight):
                row = by_cell[(initialization,weight)]
                return ((row['upper25'] or {}).get('keep_rgb_mae') if section=='upper25' else
                        (row['summary'] or {}).get(section,{}).get('keep_rgb_mae'))
            y00,y10,y01,y11 = [value(i,w) for i,w in (('original_sample',1),('appended_real_support',1),('original_sample',5),('appended_real_support',5))]
            complete = all(finite_number(v) for v in (y00,y10,y01,y11))
            effects.append(dict(metric=label,y00=y00,y10=y10,y01=y01,y11=y11,
                                support_effect_at_weight1=y10-y00 if finite_number(y10) and finite_number(y00) else None,
                                support_effect_at_weight5=y11-y01 if finite_number(y11) and finite_number(y01) else None,
                                rgb_weight_effect_original=y01-y00 if finite_number(y01) and finite_number(y00) else None,
                                rgb_weight_effect_support=y11-y10 if finite_number(y11) and finite_number(y10) else None,
                                marginal_support=((y10-y00)+(y11-y01))/2 if complete else None,
                                marginal_rgb_weight=((y01-y00)+(y11-y10))/2 if complete else None,
                                interaction=(y11-y10)-(y01-y00) if complete else None))
        return dict(status='HISTORICAL_RESEARCH_FINISHING_SCOPE_MOVED_TO_ENGINEERING' if self.scope_change else 'RESEARCH_IN_PROGRESS_RESULTS_NOT_ASSUMED',
                    authorization=self.research_authorization,contract=self.research_contract,
                    duration=dict(question='Does this unchanged native single trajectory recover with more iterations?',
                                  source_iteration=11977,target_iteration=20000 if self.scope_change else 30000,
                                  planned_new_updates=8023 if self.scope_change else 18023,
                                  scope_change=self.scope_change,
                                  rows=duration,not_a_fresh_replication=True,
                                  rng_boundary_audit=self.read(self.evidence / 'single_longrun_rng_boundary_audit.json'),
                                  execution_verification=self.read(self.evidence / 'single_longrun_execution_verification.json'),
                                  counterfactual_without_extra_evaluation_bitwise_identity_claimed=False),
                    static_rgb_supervision=dict(question='Initialization support × whole-static RGB scalar',
                                                cells=cells,early1k_cells=early_cells,
                                                contract=self.read(self.evidence / 'loss2x2_contract.json'),
                                                paired_audit=pairing,
                                                descriptive_effects=effects,
                                                all_four_evaluations_retrieved=all(row['validated'] for row in cells),
                                                causal_effect_established=False,
                                                scope='Whole-static RGB scalar1→5 only; global1+static nominal2→6. Not the complete dynamic training conditions.'),
                    fixed_domains=dict(full=PIXELS,boundary=7530,upper_bbox_height25=25906),
                    conditional_branch_control_preparation=self.read(self.evidence / 'branch_control_cpu_preparation_20261107.json'),
                    agent_visual_review=visual_review,
                    quality_accepted=False,human_verdict=None)

    def make_research_figures(self, data):
        architecture = '''<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="440" viewBox="0 0 1400 440" role="img" aria-labelledby="t d">
<title id="t">Scene008：身份划分、初始化与训练目标的研究组件</title><desc id="d">真实观测经过身份划分进入动态待删集合与时间静态分支。固定初始化和RGB目标的实验先检查KEEP完整性，再检查原生DELETE保留。新结果未到不推断因果。</desc>
<defs><marker id="a" markerWidth="9" markerHeight="9" refX="8" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="#607d8b"/></marker><style>text{font-family:'Microsoft YaHei',Arial,sans-serif;fill:#193b4a}.title{font-size:23px;font-weight:600}.body{font-size:18px}.small{font-size:16px;fill:#526d7a}.b{fill:white;stroke:#8aabb9;stroke-width:1.7}.arrow{fill:none;stroke:#607d8b;stroke-width:2;marker-end:url(#a)}</style></defs>
<rect width="1400" height="440" fill="#f5f8fa"/><text x="28" y="42" class="title">Scene008 a5：先验证KEEP车体，再检查DELETE保留</text>
<text x="28" y="74" class="small">同原生10K已有差异；初始化支撑与静态训练条件是待实验区分的因素</text>
<rect class="b" x="25" y="140" width="220" height="170" rx="12"/><text x="135" y="178" text-anchor="middle" class="title">真实输入</text><text x="135" y="215" text-anchor="middle" class="body">49帧RGB / LiDAR</text><text x="135" y="248" text-anchor="middle" class="small">a5源点21,358</text><text x="135" y="278" text-anchor="middle" class="small">原采样1,496行</text>
<rect class="b" x="275" y="140" width="230" height="170" rx="12"/><text x="390" y="178" text-anchor="middle" class="title">身份划分</text><text x="390" y="215" text-anchor="middle" class="body">a37进入共同动态池</text><text x="390" y="248" text-anchor="middle" class="small">指定身份 = 待删集合</text><text x="390" y="278" text-anchor="middle" class="small">a5仍由时间static解释</text>
<rect class="b" x="535" y="125" width="290" height="200" rx="12"/><text x="680" y="162" text-anchor="middle" class="title">初始化与RGB目标</text><text x="680" y="199" text-anchor="middle" class="body">原采样 / 补真实支撑</text><text x="680" y="232" text-anchor="middle" class="body">staticRGB 1 / 5 · fresh5K</text><text x="680" y="265" text-anchor="middle" class="small">其余目标与身份保持原合同</text><text x="680" y="297" text-anchor="middle" class="small">另：原single轨迹续训至30K</text>
<rect class="b" x="855" y="140" width="230" height="170" rx="12"/><text x="970" y="178" text-anchor="middle" class="title">KEEP完整性</text><text x="970" y="215" text-anchor="middle" class="body">原始a5固定可见域</text><text x="970" y="248" text-anchor="middle" class="small">颜色 / 边界 / 固定上部</text><text x="970" y="278" text-anchor="middle" class="small">同相机车顶与挡风玻璃</text>
<rect class="b" x="1115" y="140" width="260" height="170" rx="12"/><text x="1245" y="178" text-anchor="middle" class="title">DELETE保留</text><text x="1245" y="215" text-anchor="middle" class="body">原生render_s移除a37池</text><text x="1245" y="248" text-anchor="middle" class="small">a5不在删除集合</text><text x="1245" y="278" text-anchor="middle" class="small">与KEEP同一固定域比较</text>
<path class="arrow" d="M245 225H275"/><path class="arrow" d="M505 225H535"/><path class="arrow" d="M825 225H855"/><path class="arrow" d="M1085 225H1115"/>
<text x="28" y="366" class="small">121,128像素帧；边界7,530；包围盒高度上25%与原mask交集25,906。总alpha不是车体完整性。</text>
<text x="28" y="400" class="small">Full5历史10K PNG单独核对；工程候选负结果保留。研究进行中，GPU保持开机；本轮未用重置卡、未关机。</text></svg>'''
        if self.scope_change:
            architecture=architecture.replace('另：原single轨迹续训至30K','另：原single实际续训止20K').replace('研究进行中，GPU保持开机；本轮未用重置卡、未关机。','主线转工程；GPU保持开机。')
        architecture_path = self.root / 'figures/research_components.svg'
        architecture_path.write_text(architecture,encoding='utf-8')
        curve_path = self.root / 'figures/single_duration_curve.svg'
        width,height = 940,340
        colors = [('all49','keep_rgb_mae','#007a92','Full a5'),('boundary_all49','keep_rgb_mae','#b46b28','Boundary'),('upper25','keep_rgb_mae','#7757a3','Fixed upper')]
        points = []
        for row in data['duration']['rows']:
            if row['validated']:
                for section,key,color,label in colors:
                    value = (row['upper25'] or {}).get(key) if section=='upper25' else row['summary'].get(section,{}).get(key)
                    if finite_number(value):
                        points.append((row['iteration'],value,color,label))
        maximum = max([point[1] for point in points]+[.02])
        ymax = math.ceil(maximum/.02)*.02
        x = lambda iteration:75+(iteration-10000)/20000*820
        y = lambda value:267-value/ymax*215
        svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="940" height="340" fill="#f5f8fa"/><g font-family="Arial,sans-serif" fill="#193b4a"><text x="20" y="27" font-size="19">Native single continuation | actual retrieved native-float nodes only</text>']
        for tick in range(5):
            value = ymax*tick/4
            svg.append(f'<path d="M75 {y(value):.2f}H895" stroke="#dce6eb"/><text x="15" y="{y(value)+5:.2f}" font-size="13">{value:.3f}</text>')
        for step in (10000,11977,15000,20000,30000):
            svg.append(f'<text x="{x(step):.2f}" y="290" text-anchor="middle" font-size="12">{step}</text>')
        for _,_,color,label in colors:
            sequence = sorted((step,value) for step,value,c,l in points if c==color)
            if sequence:
                svg.append('<polyline fill="none" stroke="'+color+'" stroke-width="2" points="'+' '.join(f'{x(step):.2f},{y(value):.2f}' for step,value in sequence)+'"/>')
                for step,value in sequence:
                    svg.append(f'<circle cx="{x(step):.2f}" cy="{y(value):.2f}" r="4" fill="{color}"/>')
        for index,(_,_,color,label) in enumerate(colors):
            svg.append(f'<text x="{90+210*index}" y="320" fill="{color}" font-size="14">{label} MAE (0-1)</text>')
        svg.append('</g></svg>')
        curve_path.write_text(''.join(svg),encoding='utf-8')
        preview=Image.new('RGB',(1015,442),'#f3f7f9')
        draw=ImageDraw.Draw(preview)
        font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',13) if Path('C:/Windows/Fonts/arial.ttf').is_file() else ImageFont.load_default()
        draw.text((8,7),'raw105 | KEEP only | original a5 crop | all cells keep a5 in static',fill='#193b4a',font=font)
        names=['Observed GT','Orig / RGB1','Support / RGB1','Orig / RGB5','Support / RGB5']
        preview_sources=[]
        for col,name in enumerate(names):
            draw.text((40+195*col,31),name,fill='#193b4a',font=font)
        for node,rows in ((1000,data['static_rgb_supervision']['early1k_cells']),(5000,data['static_rgb_supervision']['cells'])):
            top=57+(0 if node==1000 else 190)
            draw.text((5,top+68),'1K' if node==1000 else '5K',fill='#193b4a',font=font)
            folders=[self.evidence/row['folder'] for row in rows]
            gt=next((folder/'frames/105/gt.png' for folder in folders if (folder/'frames/105/gt.png').is_file()),None)
            sources=[gt]+[folder/'frames/105/keep.png' for folder in folders]
            for col,path in enumerate(sources):
                image=self.load_image(path) if path is not None and path.is_file() else None
                left=40+195*col
                if image is not None:
                    preview.paste(image.crop(CROP),(left,top))
                else:
                    draw.rectangle((left,top,left+175,top+160),fill='#e5ebee')
                    draw.text((left+15,top+70),'Not retrieved',fill='#627680',font=font)
                preview_sources.append(dict(iteration=node,column=names[col],path=self.relative(path) if path is not None and path.is_file() else None))
        preview_path=self.root/'figures/raw105_static_rgb_2x2.png'
        preview.save(preview_path)
        data['static_rgb_supervision']['raw105_figure']=dict(path=self.relative(preview_path),size=list(preview.size),crop=list(CROP),sources=preview_sources)
        duration_preview=Image.new('RGB',(1170,226),'#f3f7f9')
        dd=ImageDraw.Draw(duration_preview)
        dd.text((8,7),'raw105 | same single trajectory | KEEP | original crop, no enhancement',fill='#193b4a',font=font)
        gt=self.evidence/'reference_single10k/frames/105/gt.png'
        duration_sources=[('Observed GT',gt)]+[(str(row['iteration']),self.evidence/row['folder']/'frames/105/keep.png') for row in data['duration']['rows']]
        for col,(label,path) in enumerate(duration_sources):
            dd.text((8+195*col,34),label,fill='#193b4a',font=font)
            image=self.load_image(path) if path.is_file() else None
            if image is not None:
                duration_preview.paste(image.crop(CROP),(8+195*col,58))
            else:
                dd.rectangle((8+195*col,58,183+195*col,218),fill='#e5ebee')
                dd.text((20+195*col,125),'Not executed' if label=='30000' and self.scope_change else 'Not retrieved',fill='#627680',font=font)
        duration_path=self.root/'figures/raw105_single_duration.png'
        duration_preview.save(duration_path)
        data['duration']['raw105_figure']=dict(path=self.relative(duration_path),size=list(duration_preview.size),crop=list(CROP),sources=[dict(label=label,path=self.relative(path) if path.is_file() else None) for label,path in duration_sources])
        return [self.relative(architecture_path),self.relative(curve_path),self.relative(preview_path),self.relative(duration_path)]

    def research_accounting_data(self):
        """Accumulate completed new updates from returned phase receipts once."""
        base=self.accounting_round1 or {}
        if base.get('completed_formal_updates',{}).get('total')!=20000 or base.get('completed_engineering_total')!=6599:
            raise ValueError('The preserved Round1 accounting baseline is missing or inconsistent')
        entries=[]
        pending=[]
        lifecycle={}
        for arm in ('control','support'):
            rel='loss2x2_preparation/'+arm+'_1k_lifecycle.json'
            value=self.read(self.evidence/rel)
            if value:
                lifecycle[arm]=(value,'evidence/'+rel)
        specifications=[]
        for end,start in ((15000,11977),(20000,15000),(30000,20000)):
            if end==30000 and self.scope_change:
                continue
            folder='single_longrun'+str(end//1000)+'k'
            specifications.append(('single_continuation',start,end,[folder+'/run_receipts/returned.json'],None))
        for arm,stem in (('control','loss5_control'),('support','loss5_full_a5_')):
            for end,start in ((1000,0),(5000,1000)):
                node=str(end//1000)+'k'
                candidates=[stem+node+'/run_receipts/formal'+node+'/returned.json',stem+node+'/run_receipts/returned.json']
                specifications.append(('staticRGB5_'+arm,start,end,candidates,(arm,arm+'/formal'+node+'/returned.json')))
        seen=set()
        for family,start,end,candidates,fallback in specifications:
            receipt=source=None
            json_key=None
            for rel in candidates:
                receipt=self.read(self.evidence/rel)
                if receipt is not None:
                    source='evidence/'+rel
                    break
            if receipt is None and fallback and fallback[0] in lifecycle:
                packed,source=lifecycle[fallback[0]]
                receipt=packed.get(fallback[1])
                json_key=fallback[1]
            if receipt is None:
                pending.append(dict(family=family,from_iteration=start,to_iteration=end,reason='completed returned receipt not retrieved; not a claim of zero actual consumption'))
                continue
            if receipt.get('status')!='PASS_COMPLETE_BOUNDARY' or receipt.get('source_iteration')!=start or receipt.get('iteration')!=end or receipt.get('actual_additional_iterations')!=end-start:
                raise ValueError('Research phase returned receipt differs from the registered update interval: '+str(source))
            key=(family,start,end)
            if key in seen:
                raise ValueError('Duplicate research interval would double-count updates')
            seen.add(key)
            entries.append(dict(family=family,from_iteration=start,to_iteration=end,completed_updates=end-start,
                                source=source,json_key=json_key,source_sha256=sha256(self.root/source),returned=receipt))
        formal=20000+sum(row['completed_updates'] for row in entries)
        return dict(status='PASS_COMPLETED_RECEIPT_ACCOUNTING_RESEARCH_IN_PROGRESS',
                    generated_utc=datetime.now(timezone.utc).isoformat(),
                    base=dict(source='evidence/execution_accounting_through_round1_20261107.json',
                              sha256=sha256(self.evidence/'execution_accounting_through_round1_20261107.json'),formal_updates=20000,engineering_updates=6599,total_updates=26599),
                    completed_research_formal_intervals=entries,
                    completed_formal_updates=formal,completed_engineering_updates=6599,completed_total_updates=formal+6599,
                    preserved_engineering_entries=base['completed_engineering_updates'],
                    incomplete_or_unretrieved_intervals=pending,
                    scope='Completed returned phase receipts only; in-flight or unretrieved updates are not presumed zero. Prior1977 engineering is not recounted.',
                    GPU_keep_on=True,shutdown=False,reset_card=False,quality_accepted=False,human_verdict=None)

    def render_research(self):
        data = self.research
        duration_rows = []
        for row in data['duration']['rows']:
            summary = row['summary'] or {}
            duration_rows.append('<tr><td>'+h(row['label'])+'</td><td>'+h(row['status'])+'</td><td>'+fmt(summary.get('all49',{}).get('keep_rgb_mae'))+'</td><td>'+fmt(summary.get('boundary_all49',{}).get('keep_rgb_mae'))+'</td><td>'+fmt((row['upper25'] or {}).get('keep_rgb_mae'))+'</td></tr>')
        cell_rows = []
        for row in data['static_rgb_supervision']['cells']:
            summary = row['summary'] or {}
            raw105=next((frame.get('visible',{}) for frame in summary.get('frames',[]) if frame.get('raw')==105),{})
            visual='未恢复' if row['agent_body_restored'] is False else '代理观察恢复；非用户验收' if row['agent_body_restored'] is True else '未登记'
            cell_rows.append('<tr><td>'+h(row['label'])+'</td><td>'+str(row['nominal_global_plus_static_rgb_scalar'])+'</td><td>'+h(row['status'])+'</td><td>'+fmt(summary.get('all49',{}).get('keep_rgb_mae'))+'</td><td>'+fmt(summary.get('boundary_all49',{}).get('keep_rgb_mae'))+'</td><td>'+fmt((row['upper25'] or {}).get('keep_rgb_mae'))+'</td><td>'+fmt(raw105.get('keep_rgb_mae'))+'</td><td>'+visual+'</td></tr>')
        effects = ''.join('<tr><td>'+h(row['metric'])+'</td><td>'+fmt(row['support_effect_at_weight1'])+'</td><td>'+fmt(row['rgb_weight_effect_original'])+'</td><td>'+fmt(row['rgb_weight_effect_support'])+'</td><td>'+fmt(row['marginal_support'])+'</td><td>'+fmt(row['marginal_rgb_weight'])+'</td><td>'+fmt(row['interaction'])+'</td></tr>' for row in data['static_rgb_supervision']['descriptive_effects'])
        early_rows=[]
        for row in data['static_rgb_supervision']['early1k_cells']:
            summary=row['summary'] or {}
            early_rows.append('<tr><td>'+h(row['label'])+'</td><td>'+h(row['status'])+'</td><td>'+fmt(summary.get('all49',{}).get('keep_rgb_mae'))+'</td><td>'+fmt((row['upper25'] or {}).get('keep_rgb_mae'))+'</td><td>'+fmt(summary.get('all49',{}).get('keep_mean_alpha'))+'</td></tr>')
        duration_reviews=''.join('<p>'+h(str(row.get('iteration'))+'代理目视：'+row.get('agent_observation',''))+'</p>' for row in (data['agent_visual_review'] or {}).get('reviews',[]) if 'single_longrun' in row.get('image',''))
        accounting=self.research_accounting
        finished=not accounting['incomplete_or_unretrieved_intervals'] and self.scope_change is not None
        historical_state='已完成的历史诊断' if finished else '历史诊断收尾'
        pair_state='两条scalar对照均已逐项核对5000次实际相机、time_shift与学习率，零不匹配；这仍是单场景单seed的干预描述，不是唯一根因证明。' if data['static_rgb_supervision']['paired_audit'] else '完整日程与唯一配重差异审计到达前，不把四格数值当成已完成因果验证。'
        scope_notice='<p class="notice">当前主线是完整五身份重建与身份隔离，目标为正常完整的a5车顶/挡风玻璃。本页保留已完成的历史研究；Single止20K，30K未执行、不对其恢复下结论。原a5+a37备选与opacity0均0训练、不再启动。旧研究负结果不代判新重建结果，编辑或静态残影不是本分支主验收。</p>' if self.scope_change else ''
        return f'''<section id="research"><h2>{historical_state}：初始化支撑与静态RGB监督强度</h2>{scope_notice}
<p>用户引用的同原生10K PNG已显示Full5与Single的车顶差异，因此不能仅用“30K对10K”解释。初始化支撑与静态训练条件仍待区分；既有证据尚未确定二者各自的因果贡献。</p>
<p>后续诊断评价优先看raw97/105/109三帧，确认接线与现象后再扩49；已在执行的49帧评价继续，不打断。训练仍按原49帧采样合同进行。三帧pilot只作快速检查，不冒充全49指标。</p>
<p>截至已完成Round1，累计新增正式20,000步、工程6,599步，共26,599步；新研究进行中的更新未提前计作完成。旧凭据不覆盖。按最新要求<strong>GPU保持开机</strong>；本轮未使用重置卡、未关机，历史条件授权原文保留。{self.link('evidence/execution_accounting_through_round1_20261107.json','Round1累计账本与来源SHA')}</p>
<p>本研究阶段已取回完成回执累计：正式<strong>{accounting['completed_formal_updates']:,}</strong>＋工程<strong>6,599</strong>＝<strong>{accounting['completed_total_updates']:,}</strong>更新。仅累加每个实际阶段一次；新工程接口的smoke/正式更新独立记账，未混入这里。{self.link('evidence/research_execution_accounting_current.json','逐阶段完成账目')}</p>
<img class="architecture" src="figures/research_components.svg" alt="身份划分到初始化与训练目标、KEEP完整性，再到DELETE保留的组件与数据流">
<h3>原Single实际恢复链：15K / 20K（30K未执行）</h3><p>从未修11977完整状态延续，保持原初始化、目标、动态身份和30K学习率时钟；不重新训练已花的1977步。范围调整后止于20K，新增上限8,023更新，未有完成回执的部分不计作完成。这个控制只检验实际到达节点的训练时长，不能外推30K。</p>
<div class="scroll"><table><thead><tr><th>实际iteration / 节点</th><th>状态</th><th>KEEP MAE</th><th>边界 MAE</th><th>固定上部 MAE</th></tr></thead><tbody>{''.join(duration_rows)}</tbody></table></div>
<img class="architecture" src="figures/single_duration_curve.svg" alt="只连接已经取回的同轨迹原生浮点指标，不填未来节点">
<figure class="scroll"><img src="figures/raw105_single_duration.png" alt="原single各实际节点的同相机KEEP固定裁剪"><figcaption>同轨迹raw105原像素裁剪；未来节点保留空位，误差曲线下降不等于车顶恢复。</figcaption></figure>{duration_reviews}
<p class="muted">曲线只包含已核验的原生浮点评价节点；未取回的节点保持空白，30K因范围调整未执行，未插值或外推。11977是既有工程节点；均值下降仍需与同帧车顶图片一起判断。</p>
<details><summary>实际20K结束与额外评价的随机状态边界</summary><p>15K额外原生评价会消耗Torch CPU RNG。所枚举的15K后存活训练路径未发现显式Torch CPU随机消费者，但有限源码审计不能证明未加15K节点的反事实逐位同轨。这里只断言实际已保存checkpoint的完整恢复零差；未改或重放既有基线。{self.link('evidence/single_longrun_rng_boundary_audit.json','CPU随机状态与实际恢复审计')} · {self.link('evidence/single_longrun_execution_verification.json','实际8023更新与20K停止、30K未执行')}</p></details>
<h3>静态RGB监督强度消融：初始化 × scalar 1 / 5</h3><p>两条新5K臂只将<strong>全static域RGB scalar从1改到5</strong>；原global RGB系数1保留，a5静态像素的名义global＋static RGB系数由2变6。它影响背景，且不等于复制动态分支的alpha100、mask域、时间扰动或其他先验。a5保持static，a37共同动态池及删除语义不变。</p>
<div class="scroll"><table><thead><tr><th>初始化 / staticRGB</th><th>名义global＋static系数</th><th>5K状态</th><th>KEEP MAE</th><th>边界 MAE</th><th>固定上部 MAE</th><th>raw105 MAE</th><th>代理raw105车体检查</th></tr></thead><tbody>{''.join(cell_rows)}</tbody></table></div>
<figure class="scroll"><img src="figures/raw105_static_rgb_2x2.png" alt="同一原生裁剪的初始化乘静态RGB权重KEEP对照"><figcaption>上排1K，下排5K；GT及四格均用相同原像素裁剪。缺项保留灰格，1K不升级为5K结论，只抽查raw105。</figcaption></figure>
<div class="scroll"><table><thead><tr><th>描述性差值（MAE）</th><th>scalar1下补点</th><th>原采样下配重</th><th>补点下配重</th><th>补点边际差</th><th>配重边际差</th><th>交互项</th></tr></thead><tbody>{effects}</tbody></table></div><p class="muted">y00=原采样/1，y10=补点/1，y01=原采样/5，y11=补点/5；交互=(y11−y10)−(y01−y00)。边际差为另一因素两水平的平均差，误差差值为负表示下降；缺格不填。仅单场景单seed描述，不做统计显著性结论。</p>
<p>既有scalar1两臂与新增scalar5两臂构成2×2。提高RGB系数也会相对弱化未变系数的静态duration/velocity等项；若恢复，只能归因这项目标平衡干预有效，不能独立排除prior的作用。补点同时改变支撑、局部密度/容量与KNN初始scale；只有一场景、一seed。{pair_state}</p>
<details><summary>1K早期检查（单独列示，不作为5K终点）</summary><table><thead><tr><th>1K节点</th><th>证据</th><th>KEEP MAE</th><th>固定上部 MAE</th><th>平均总alpha</th></tr></thead><tbody>{''.join(early_rows)}</tbody></table><p>平均误差或总alpha变化不能替代车体目视；已取回的RGB5原采样1K中仍几乎没有可辨a5车体。{self.link('evidence/research_agent_visual_review_20261107.json','本轮代理目视记录（人类判定为空）')}</p></details>
<details><summary>历史备选分支：仅CPU准备，0训练，已取消启动</summary><p>原生a5+a37动态分支备选没有训练或入队；新工程主线不再启动它。a5的21,358个唯一归一化xyz/time与fullsupport相同，但静态原生0.1m排除还移走2,053个非精确a5匹配的近邻行，其语义未知。加54重复行差异，总点数710,040对707,933（约0.30%）。KNN域和原生目标/分支责任整包改变，不能独立隔离prior。原生DELETE会删a5+a37，不是a37-only编辑方案。{self.link('evidence/branch_control_cpu_preparation_20261107.json','CPU准备历史凭据')} · {self.link('source/prepare_a5_branch_control.py','输入准备源码')}</p></details>
<p>{self.link('evidence/research_authorization_20261107.json','本轮授权（GPU保持开机）')} · {self.link('evidence/research_20261107_contract.json','研究合同')} · {self.link('evidence/loss2x2_contract.json','监督强度消融合同')} · {self.link('evidence/loss2x2_comparison.json','2×2实际配对审计')} · {self.link('source/aggregate_fixed_a5_regions.py','固定上部CPU汇总源码')}</p>
<details><summary>新研究指标与可识别性边界</summary><p>49帧均为训练视图重建诊断，并非独立泛化评价；a5只在raw89..112的24帧中有固定可见像素。全域121,128、内侧1px边界7,530、固定上部25,906像素帧。上部由原mask包围盒高度上25%确定，不是按render或误差挑像素。KEEP完整性与DELETE变化分别记录；总alpha不能证明车体完整。Full5 cached PNG继续单列，不并入新float表。human_verdict为空。</p><pre>{h(json.dumps(self.research_contract,ensure_ascii=False,indent=2))}</pre></details></section>'''

    def render_engineering(self, data):
        rows = []
        for row in data['rows']:
            if row['validated']:
                summary, upper = row['summary'], row['upper25']
                cells = [h(row['label']), fmt(summary['all49']['keep_rgb_mae']),
                         fmt(summary['boundary_all49']['keep_rgb_mae']),
                         fmt((upper or {}).get('keep_rgb_mae')),
                         fmt(summary['all49']['keep_delete_rgb_change'], 7)]
            else:
                cells = [h(row['label'])] + ['待运行/待取回'] * 4
            rows.append('<tr>' + ''.join('<td>'+value+'</td>' for value in cells) + '</tr>')
        pair = data['pair_assertions']
        alignment = ('共同完整起点：' + ('通过' if pair['same_initial_complete_state'] is True else '待核对') +
                     '；1977次相机/shift日程：' + ('通过' if pair['same_camera_shift'] is True else '待核对'))
        if data['paired_ready']:
            delta = data['delta_fixed_minus_baseline']
            state = '配对断言与固定域核验已通过。候选−未修的 ΔMAE：全可见域 '+fmt(delta['all49_keep_rgb_mae'], 7)+\
                    '，边界 '+fmt(delta['boundary_keep_rgb_mae'], 7)+'，上25% '+fmt(delta['upper25_keep_rgb_mae'], 7)+'。负值仅表示该数值下降，不自动表示车顶恢复。'
            if data['adoption_decision'] == 'NOT_ADOPTED_NEGATIVE_A5_REPLAY':
                relative = delta['all49_keep_rgb_mae']/data['rows'][1]['summary']['all49']['keep_rgb_mae']
                state = '负结果：time-only全可见域KEEP MAE上升'+f'{relative:.1%}'+\
                        '，raw105未见车顶恢复，本轮明确不采纳为a5修复。工程PASS仅证明配对和执行完成。'+state
        else:
            state = '候选验证等待运行或完整取回 paired_result.json；此处不预判改善，也不把未配对数值作因果结论。'
            if data['pair_errors']:
                state += ' 当前未满足：'+'；'.join(data['pair_errors'])+'。'
        shift = data['shift_cpu_proof'] or {}
        count = shift.get('sampler_log', {}).get('nonzero_time_shifts')
        checkpoint = data['checkpoint_summary']
        points = []
        for item in checkpoint.get('native_fixed_anchor_metrics') or []:
            value = next((row for row in item['frames'] if row['raw'] == 105), None)
            if value:
                points.append(value['upper25'])
        numeric = ''
        if len(points) == 2:
            numeric = '<p>raw105固定上部2091像素：10K→11977的KEEP MAE '+fmt(points[0]['keep_rgb_mae'])+'→'+fmt(points[1]['keep_rgb_mae'])+\
                      '，总alpha均值 '+fmt(points[0]['keep_mean_alpha'])+'→'+fmt(points[1]['keep_mean_alpha'])+'；alpha≥0.5均为100%，图中车顶仍缺失并有蓝色错位残影。</p>'
        links = [
            ('evidence/static_time_shift_cpu_probe.json', 'static分支CPU证明'),
            ('source/probe_static_time_shift.py', 'shift复现源码'),
            ('source/static_time_shift_candidate.patch', '两行候选patch'),
            ('evidence/single_bug_checkpoint_audit.json', '检查点支持与时间核审计'),
            ('source/probe_single_checkpoints_cpu.py', '检查点CPU探针'),
            ('evidence/single_bug_source_audit.json', '源码审计JSON'),
            ('evidence/single_bug_source_audit.md', '源码调查说明'),
            ('evidence/screen_prune_cpu_delivery_probe.json', 'screen-radius独立CPU复现'),
            ('source/probe_screen_prune_cpu.py', 'screen-radius探针'),
            ('source/probe_a5_lineage_cpu.py', 'a5源谱系探针'),
            ('source/probe_a5_source_time_cpu.py', 'a5点源/时间探针'),
            ('evidence/time_shift_fix/isolated_fix_contract.json', '隔离合同'),
            ('evidence/time_shift_fix/paired_result.json', '配对结果'),
            ('evidence/time_shift_fix/fixed/summary.json', '候选49帧评价'),
            ('evidence/time_shift_fix/fixed/source_integrity.json', '候选来源完整性'),
        ]
        evidence = ''.join('<li>'+self.link(path, label)+'</li>' for path, label in links)
        upper_count = next((row['upper25']['pixels'] for row in data['rows'] if row['upper25'] is not None), None)
        upper_size = str(upper_count)+f' / {PIXELS}（{upper_count/PIXELS:.1%}）' if upper_count is not None else '待原始float完整取回'
        return f'''<section id="single-bug"><h2>单车并行排查：现象、分支行为与候选验证分开</h2>
<p>仅a37原10K与额外1977步的未修11977都在KEEP中缺顶。11977的全49帧MAE虽由0.062096降至0.043573，车顶仍明显缺失，前上方还有蓝色半透明/错位残影；这不是已恢复的车体。</p>{numeric}
<div class="scroll"><table><thead><tr><th>当前判定</th><th>已复现证据</th><th>a5因果与本轮处理</th></tr></thead><tbody>
<tr><td>分支行为 / 候选接线缺陷<br>static time_shift</td><td>原10K实际有{h(count if count is not None else '待核对')}次非零shift；原static输出变化为0，dynamic正常响应。候选仅移除两处覆写None，None-shift输出保持不变。</td><td>上游2024已有该不对称；缺少static必须与dynamic采用相同自监督的设计合同。尚未证明它造成a5缺损，下面只做单变量候选验证。</td></tr>
<tr><td>已复现实现缺陷<br>screen-radius统计清零</td><td>densify先清零max_radii2D，再检查max_screen_size；CPU半径[100,1]、阈值20，正常路径应删1却删0，prune_only对照删1。</td><td>a5因果未知。当前正式主实验及该time-shift候选回放均未施加此项修复；不把两项改动堆叠。</td></tr>
</tbody></table></div>
<p>检查点审计未发现所查几何/时间/opacity参数的NaN或Inf；10K→11977尚未跨12K opacity-reset节点。两快照不能定位唯一有害训练事件。投影支持proxy也不能当成CUDA逐像素归因。</p>
<img class="architecture" src="bug_time_shift.svg" alt="同原10K完整状态、相同1977相机shift日程，未修与单变量候选独立对照，工程开销另记">
<h3>隔离工程对照（不进入正式5K/10K表）</h3><p class="notice">{h(state)}</p><p>{h(alignment)}。即使相机/shift相同，也不宣称增密后的随机流和逐位轨迹完全相同。</p>
<div class="scroll"><table><thead><tr><th>工程角色</th><th>全49可见域<br>KEEP RGB MAE</th><th>固定内边界<br>KEEP MAE</th><th>原mask上部区域<br>KEEP MAE（定义见下）</th><th>全49 KEEP↔DELETE<br>平均RGB变化</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p class="muted">上部区域＝原始a5 mask包围框<strong>高度上25%</strong>与该mask的交集；阈值只由固定GT mask决定，不是按像素数取最上25%，也不是render选择。当前计数{h(upper_size)}，不是语义车顶真值。数值只从已保存float计算，缺float保持空缺；没有用PNG估算。</p>
<figure class="scroll"><img src="{data['figure']['path']}" alt="raw105原10K、未修11977、候选11977同裁剪工程比较"><figcaption>只比较KEEP的同一原像素裁剪；候选未取回时保持占位。</figcaption></figure>
<p>未修1977步属于此前smoke。time-only另有32步真实smoke＋2步恢复探针＋1977步隔离回放，共2011工程更新；正式预算不变。完整回执到达才记完成；工程完成和科学修复通过是两件事。</p>
<details><summary>复现源码、候选patch与审计凭据</summary><ul>{evidence}</ul></details></section>'''

    def comparison_rows(self, datasets):
        rows = []
        for item in datasets:
            for region, key in (('a5_visible_all49', 'all49'), ('a5_inner_boundary_1px_all49', 'boundary_all49')):
                row = dict(model_id=item['id'], label=item['label'], status=item['status'], region=region,
                           views=49 if item['validated'] else '', source=f'evidence/{item["fresh_folder"]}/summary.json' if item['fresh_folder'] else '')
                values = item['summary'][key] if item['validated'] else {}
                for name in ('pixels', 'keep_rgb_mae', 'delete_rgb_mae', 'keep_delete_rgb_change',
                             'keep_mean_alpha', 'delete_mean_alpha'):
                    row[name] = values.get(name, '')
                row['keep_alpha_ge_0.5'] = values.get('keep_coverage', {}).get('0.5', '')
                row['delete_alpha_ge_0.5'] = values.get('delete_coverage', {}).get('0.5', '')
                row['a5_is_in_deleted_pool'] = 5 in (item['summary'] or {}).get('dynamic_aliases', []) if item['validated'] else ''
                row['delete_comparison_applicable'] = item['id'] != 'full5'
                if item['id'] == 'full5':
                    for key in ('delete_rgb_mae', 'keep_delete_rgb_change', 'delete_mean_alpha', 'delete_alpha_ge_0.5'):
                        row[key] = ''
                rows.append(row)
        return rows

    def render_cached_native10k(self):
        summary = self.cached_native10k
        table = figure = interpretation = ''
        models = (summary or {}).get('models', {})
        integrity = (summary or {}).get('source_integrity', '')
        if isinstance(integrity, dict):
            integrity = integrity.get('status', '')
        valid = (summary or {}).get('status') == 'PASS_CACHED_NATIVE10K_PNG_SAME_BUDGET_COMPARISON' and \
                (summary or {}).get('fixed_denominator', {}).get('pixels') == PIXELS and \
                str(integrity).startswith('PASS') and all(
                    models.get(key, {}).get('frame_count') == 49 and models[key].get('pixels') == PIXELS and
                    models[key].get('native_checkpoint_reloaded') is False
                    for key in ('full5_native10k', 'single_native10k'))
        if valid:
            state = '两套已有原生PNG在同10K、同49相机/时间与固定121128像素域的CPU核对已通过。'
            rows = []
            for key, label in (('full5_native10k', 'Full5 cached 10K'), ('single_native10k', 'Single cached 10K')):
                values = models[key]
                rows.append('<tr>'+''.join('<td>'+value+'</td>' for value in (
                    h(label), '49 / '+str(values['pixels']), fmt(values.get('keep_rgb_mae')),
                    str(values.get('boundary_pixels')), fmt(values.get('boundary_keep_rgb_mae')),
                    '仍在' if values.get('checkpoint_components_still_available') else '四件权重已清理'))+'</tr>')
            table = '<div class="scroll"><table><thead><tr><th>历史PNG来源</th><th>帧 / 可见像素</th><th>KEEP RGB MAE</th><th>内边界像素</th><th>边界KEEP MAE</th><th>权重状态（本次均未加载）</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></div>'
            interpretation = '<p>全可见域里Full5 cached10K误差较低，但其边界误差反而较高；不能挑一个指标称全面更好。同10K减少了训练长度混淆，仍未单独隔离输入支撑、监督责任/损失与RNG轨迹差异，不能据此认定唯一代码根因。</p>'
            image = self.evidence / 'cached_native10k_comparison/anchors/105_crop.png'
            if image.is_file():
                self.consumed.add(image.resolve())
                figure = '<figure class="scroll"><img src="'+self.relative(image)+'" alt="raw105已有原生10K PNG同裁剪GT、Full5与Single"><figcaption>raw105：GT、Full5 cached10K、Single cached10K；已存原生uint8 PNG，未恢复或重新渲染Full5权重。</figcaption></figure>'
        else:
            state = '已有原生10K PNG的CPU同口径核对正在取回；当前不填入未齐的数字。'
        return f'''<section id="cached-10k"><h2>同10K历史PNG证据（独立口径）</h2>
<p>Full5 B10K的四件权重虽已清理，原生checkpoint发布事件和49帧PNG仍保留。这里仅对已存原生uint8 PNG，与Single10K的已存PNG按相同原始mask、相机和时间做CPU核对；<strong>未恢复或重新渲染Full5 10K权重</strong>。</p>
<p>{h(state)}</p>{table}{interpretation}{figure}<p class="muted">本节RGB MAE为PNG uint8/255相对固定float观测RGB的通道平均绝对误差，单位0–1；不是原生float重放分数。该证据不是隐藏表面真值，不混入新鲜float主表。</p>
<p>{self.link('evidence/cached_native10k_comparison/summary.json', '同10K历史PNG摘要')} · {self.link('source/compare_cached_native10k.py', '只读CPU核对源码')}</p></section>'''

    def write_csv(self, rows):
        with (self.root / 'comparison.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def render(self, datasets, old_rows, receipts, contract, history, audit):
        by_id = {item['id']: item for item in datasets}
        pending = [by_id[name]['label'] for name in ('single10k', 'dual5k', 'dual10k') if not by_id[name]['validated']]
        if by_id['single10k']['validated'] and by_id['dual5k']['validated']:
            state = '已复现删除前缺损 / 候选验证单独列账'
            reviewed_dual10 = any(row.get('a5_visual_quality_accepted') is False and str(row.get('source', '')).startswith('dual10k/')
                                 for row in (self.visual_review or {}).get('reviews', []))
            if by_id['dual10k']['validated'] and reviewed_dual10:
                lead = '仅a37原10K与完整a37+a6的5K、10K均未通过raw105车体的代理目视：车顶/挡风玻璃仍缺失；10K虽增加蓝色外观，仍伴错位残影。KEEP/DELETE都存在缺损，删除不是首次形成步骤。'
            else:
                lead = '仅a37原10K与完整a37+a6的5K均已在KEEP阶段复现a5缺顶；删除操作不是缺损首次出现的步骤。'
                lead += '双车10K实际评价已取回，见固定域表与四锚图；' if by_id['dual10k']['validated'] else '双车10K仍待评价；'
            if self.cached_native10k is not None and any(row.get('full5_top_structure_visible') is True
                    for row in (self.visual_review or {}).get('reviews', [])):
                lead += '同10K历史PNG中的Full5虽模糊，车顶/挡风玻璃结构仍在；这减少了30K对10K的预算混淆。'
            if self.engineering['paired_ready'] and self.engineering['adoption_decision'] == 'NOT_ADOPTED_NEGATIVE_A5_REPLAY':
                relative = self.engineering['delta_fixed_minus_baseline']['all49_keep_rgb_mae']/self.engineering['rows'][1]['summary']['all49']['keep_rgb_mae']
                lead += 'time-only隔离验证反而使KEEP MAE上升'+f'{relative:.1%}'+ '，未见车顶修复，明确不采纳。'
            else:
                lead += 'time-only候选结果单独列示，不自动判为修复。' if self.engineering['paired_ready'] else 'time-shift单变量候选验证仍待完整配对回执，尚未证明它造成a5缺损。'
            if self.screen_radius['paired_ready'] and self.screen_radius['adoption_decision'] == 'NOT_ADOPTED_NEGATIVE_A5_REPLAY':
                relative = self.screen_radius['delta_fixed_minus_baseline']['all49_keep_rgb_mae']/self.screen_radius['rows'][0]['summary']['all49']['keep_rgb_mae']
                lead += 'screen-only也使KEEP MAE上升'+f'{relative:.1%}'+ '，仍缺顶且全局重建退化，同样不采纳。两个独立短回放都未修好a5，尚未唯一归因某个commit。'
                state = '诊断队列已结束 / a5质量未通过，两个候选不采纳'
            else:
                lead += 'screen-only独立验证结果已取回，另表报告。' if self.screen_radius['paired_ready'] else 'screen-radius独立验证仍等待完整取回，不叠加time改动。'
        elif pending:
            state = '实验进行中 / 评价待取回'
            lead = '已有 raw105 原生 KEEP 图可见：a5 的车顶缺损在仅 a37 动态的原10K中已经存在。当前仍待取回' + '、'.join(pending) + '的完整核验，不能提前宣称修复或认定某个提交为唯一原因。'
        else:
            state = '诊断节点已取回 / 结论按实际证据更新'
            lead = 'Single 原10K与完整双车5K/10K的49帧评价已取回，下表给出固定域数值，四个锚帧保留同一裁剪。工程评价通过不等于车体完整性通过；质量结论仍需结合全序列与边界目视。'
        state = '研究进行中 / 初始化支撑与静态RGB监督分别验证'
        lead = '同原生10K的Full5与Single已经存在车顶差异，不能仅用30K对10K解释。补齐真实a5初始化支撑的5K控制有有限改善（全域误差下降约10.1%，raw105约1.4%），仍未恢复车顶。现在分别检验原single轨迹续训到30K，以及固定初始化的静态RGB监督强度；新结果按实际取回更新，尚不认定训练条件为确定根因。time/radius两个独立短回放负结果保留，均未作为a5修复模型采用。'
        if self.scope_change:
            state='历史诊断保留 / 主线转向完整五身份重建与身份隔离'
            lead='最新任务是固定完整五身份建模集合，恢复a5车顶与挡风玻璃，并保持身份在拓扑、保存和完整恢复中正确。本页保留初始化与静态RGB监督诊断的正负证据；Single止20K，30K未执行、不下30K恢复结论；备选分支与opacity0均0训练。旧研究现象不直接判定新重建是否合格，编辑过滤仅为建模集合不变的辅助证据。'
        if contract and contract.get('fixed', {}).get('total_formal_updates') == 30000:
            stale = '<p class="notice">取回的 experiment_contract.json 仍保留旧30K计划；本页已按用户最新指令取消延伸，合同文件待同步。目录名中的30k不代表当前预算。</p>'
        else:
            stale = ''
        table = []
        for item in datasets:
            if item['validated']:
                a, b = item['summary']['all49'], item['summary']['boundary_all49']
                deleted = fmt(a.get('delete_rgb_mae'))
                if item['id'] == 'full5':
                    deleted = '—<br><small>仅KEEP参考</small>'
                    boundary_pair = fmt(b.get('keep_rgb_mae')) + ' / —'
                    change = '不适用'
                    coverage_pair = fmt(a.get('keep_coverage', {}).get('0.5')) + ' / —'
                else:
                    boundary_pair = fmt(b.get('keep_rgb_mae')) + ' / ' + fmt(b.get('delete_rgb_mae'))
                    change = fmt(a.get('keep_delete_rgb_change'), 7)
                    coverage_pair = fmt(a.get('keep_coverage', {}).get('0.5')) + ' / ' + fmt(a.get('delete_coverage', {}).get('0.5'))
                if item['id'] != 'full5' and not item['summary'].get('a5_preservation_expected', True):
                    deleted += '<br><small>删除池含 a5，非保留质量</small>'
                table.append('<tr><td>' + h(item['label']) + '<br><small>' + h(item['status']) + '</small></td>' +
                             f'<td>{a["pixels"]:,}<br><small>边界 {b["pixels"]:,}</small></td>' +
                             f'<td>{fmt(a.get("keep_rgb_mae"))}</td><td>{deleted}</td>' +
                             f'<td>{boundary_pair}</td><td>{change}</td><td>{coverage_pair}</td></tr>')
            else:
                table.append(f'<tr><td>{h(item["label"])}</td><td colspan="6" class="missing">{h(item["status"])}</td></tr>')
        anchor_table = ''.join('<tr>' + ''.join(f'<td>{h(value)}</td>' for value in
                            (row['model'], row['raw'], row['stage'], row['pixels'], fmt(row['rgb_mae']), fmt(row['change'], 7), fmt(row['alpha']))) + '</tr>'
                            for row in old_rows)
        figures = ''.join(f'<figure id="raw{item["raw"]}"><img src="{item["path"]}" alt="raw{item["raw"]}同裁剪KEEP和DELETE比较">'
                          f'<figcaption>raw{item["raw"]} · 上排 KEEP，下排 DELETE；175×160 原像素裁剪，无增强。缺项保持灰色占位。</figcaption></figure>' for item in self.figure_records)
        artifacts = []
        for item in datasets:
            parts = [h(item['label'])]
            if item['fresh_folder']:
                parts += [self.link(f'evidence/{item["fresh_folder"]}/{name}', label) for name, label in
                          (('summary.json', '49帧摘要'), ('source_integrity.json', '完整性'), ('source_manifest_before.json', '来源SHA清单'))]
            if item['legacy_folder']:
                parts.append(self.link(f'evidence/{item["legacy_folder"]}/receipt.json', '历史4帧凭据'))
            artifacts.append('<li>' + ' · '.join(parts) + '</li>')
        receipt_html = ''.join('<li>' + self.link(row['path'], row['path'].split('/')[-1]) +
                               ' <small>' + h(json.dumps(row['fields'], ensure_ascii=False)) + '</small></li>' for row in receipts)
        if not receipts:
            receipt_html = '<li class="missing">run_receipts：尚未取回，未知工程失败更新数不记作0。</li>'
        native_count = sum(item['validated'] for item in datasets)
        completed = self.engineering.get('completed_accounting') or {}
        completed_ledger = ''
        if completed.get('status') == 'PASS_COMPLETE_REDUCED_BUDGET_ACCOUNTING':
            completed_ledger = '<p>已取回实账：Single新增正式0步，Dual正式'+str(completed.get('dual_fresh_formal_updates'))+'步；既有smoke/恢复合计'+str(completed.get('smoke_updates_total'))+'工程更新。'+\
                               ('time-only2011工程已完整执行，但负结果不采纳；screen-only2107另按独立回执记账。' if self.engineering['paired_ready'] else
                                'time-only2011与screen-only2107均按各自完整回执另记，不提前并入已执行总数。')+'</p>'
        audit_content = '<pre>' + h(json.dumps(audit, ensure_ascii=False, indent=2)) + '</pre>' if audit is not None else '<p class="missing">远端历史扫描完整回执待取回；不把未找到的旧run写成已存在。</p>'
        history_content = '<pre>' + h(json.dumps((history or {}).get('uncertainties', []), ensure_ascii=False, indent=2)) + '</pre>'
        initial = self.link('evidence/initial_raw105.png', '用户原始 raw105 证据图')
        registration = self.registration or {}
        registration_text = 'task '+str(registration.get('task_id', 'SCENE008-A5-REGRESSION-20261106'))+\
                            '；failure ledger '+str(registration.get('failure_ledger_delta', 'V82-F06已有记录，本轮更新补充证据'))
        style = 'body{margin:0;background:#f3f7f9;color:#193b4a;font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1320px;margin:auto;padding:28px}section{background:#fff;border:1px solid #dde7ec;border-radius:12px;padding:24px;margin:20px 0}h1{font-size:30px;line-height:1.35;margin:12px 0}h2{font-size:22px;margin:0 0 12px}p{margin:10px 0}small,.muted{color:#607783;font-size:13px}.tag{font-size:13px;background:#dcecf2;border-radius:20px;padding:5px 12px;display:inline-block}a{color:#006b83}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;border-bottom:1px solid #dce6eb;text-align:left;vertical-align:top}th{background:#edf4f7;white-space:nowrap}.scroll{overflow:auto}.missing{color:#946322}.notice{background:#fff4de;border-left:4px solid #c99435;padding:12px}figure{margin:20px 0}figure img{display:block;width:100%;min-width:1000px}figcaption{font-size:13px;color:#607783;margin-top:7px}.architecture{width:100%;display:block}details{margin:15px 0;border-top:1px solid #e0e8ed;padding-top:12px}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;word-break:break-word;background:#f4f7f9;padding:12px;font:12px/1.5 monospace}code{word-break:break-word;background:#eef3f5}li{margin:7px 0}.nav{display:flex;gap:14px;flex-wrap:wrap}footer{font-size:13px;color:#607783;padding:12px}@media(max-width:700px){main{padding:12px}section{padding:16px}h1{font-size:25px}}'
        return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Scene008 a5 缺损研究：初始化与静态RGB监督</title><style>{style}</style></head><body><main>
<span class="tag">{h(state)}</span><h1>Scene008 a5 缺损研究：初始化与静态RGB监督</h1><p>{h(lead)}</p>
<p class="muted">基线主表已核对的新鲜49帧组：{native_count}；新研究节点另列。这里没有独立测试集或删除后干净背景GT。</p>{stale}
{self.render_research()}
{self.render_support_ablation()}
<section><h2>先把旧逻辑说清楚</h2><p>旧输入的 <code>points3d_dynamic_*.ply</code> 全部进入<strong>共同动态池</strong>，对应掩码取并集；原生 <code>render_s</code> 移除整个池。指定 actor 就是这次共同待删集合。仅 a37 / a37+a6 的子集模型中，a5 交给带时间参数的 static/background 解释。</p>
<p>前一诊断阶段沿用Single原10K，完整原始a6（11,625点）与a37（98,452点）完成fresh10K并在5K、10K评价。新研究从未修Single11977节点精确续训，独立列在上方；历史取消延伸与暂停指令不代表当前授权。学习率horizon仍为30K。Full5 B30K仅作既有参考，<strong>不能称同训练长度配对</strong>；其10K权重已清理，但原生PNG与发布事件仍在，下方另列同10K历史图像核对。</p>
<img class="architecture" src="architecture.svg" alt="原始观测与旧engine子集，经smoke恢复核验，在5K和10K评价a5固定域"></section>
<section id="anchors"><h2>四个锚帧：同一固定裁剪</h2><p>固定窗口用于定位raw105左侧蓝色SUV（a5）的缺损；raw89时a5主体尚未进入此窗，不能用这一小窗判其完整性。raw97及出窗中的raw109仍能看到形体混合/蓝色残影。这里只做锚帧抽查，不是49帧人工验收。裁剪不改变全图原始a5标签数值域。</p><nav class="nav">{''.join(f'<a href="#raw{raw}">raw{raw}</a>' for raw in ANCHORS)}</nav>
<div class="scroll">{figures}</div>
<p class="muted">Full5只展示KEEP：历史owner-selective回放不能替代旧子集训练，原生删全五车又会删a5，因此其DELETE格禁用。GT的DELETE格没有整图clean-delete真值。Single删除a37；完整Dual删除a37+a6；Current沿革列只删除a37。各列删除集合不同，不能把DELETE行当成同一编辑任务排名。Full5优先使用新鲜原生49帧；未取回时其KEEP沿用既有四帧回放。</p><p>{initial}</p></section>
<section id="metrics"><h2>全49帧固定域：颜色与边界</h2><p>扫描49个训练视图，a5在raw89..112的24帧有原始 <code>actor_labels==1</code> 可见像素，共121,128像素帧；不在场的帧不补造像素。边界为同一标签向内1像素环。RGB MAE为sRGB通道平均绝对误差（0–1，越小越好）；按固定像素数加权。这是重建诊断而非独立泛化评价。新鲜同浮点评价优先，历史4帧数字不填入本表。</p>
<div class="scroll"><table><thead><tr><th>模型 / 状态</th><th>可见域像素帧</th><th>KEEP RGB MAE</th><th>DELETE RGB MAE</th><th>边界 MAE<br>KEEP / DELETE</th><th>KEEP↔DELETE<br>平均RGB变化</th><th>alpha≥0.5覆盖率<br>KEEP / DELETE</th></tr></thead><tbody>{''.join(table)}</tbody></table></div>
<p class="notice">总alpha可被道路、背景或错误表面补满。覆盖率高、删车前后平均变化小，都不能单独证明a5车顶完整、深度正确或所有权正确。</p>
<p>{self.link('comparison.csv', '比较表 CSV')} · {self.link('report_data.json', '报告数据与文件清单')}</p>
<details><summary>历史四帧原始数字：仅作沿革，不冒充49帧</summary><div class="scroll"><table><thead><tr><th>来源</th><th>raw</th><th>操作</th><th>a5像素</th><th>RGB MAE</th><th>相对KEEP变化</th><th>平均alpha</th></tr></thead><tbody>{anchor_table}</tbody></table></div><p class="muted">Full5删除两车来自历史owner-selective工程分配，不是单/双对象子集训练，不参与主比较。以上独立脚本的浮点/clamp口径与新表不保证一致；不混合计算全49帧均值。</p></details></section>
{self.render_cached_native10k()}
{self.render_engineering(self.engineering)}
{self.render_screen_radius(self.screen_radius)}
{self.render_early_bounds()}
<section><h2>目前可以说到哪里</h2><p>raw105 的历史原生 KEEP 对照已显示：Full5 B30K中的a5较完整，仅a37原10K中的a5车顶已经缺损；缺损早于后加道路和完整a6资产替换。这个时间顺序支持先检查训练输入与表示分配，但尚不能区分训练长度、初始支撑、监督分支和随机轨迹各自的作用。</p>
<p>Nov4双对象试验只把1,023个a6种子从static迁入dynamic，不能代表本轮完整11,625点a6输入。<code>3fefd98</code>等配方提交只能帮助定位历史，不能在没有逐版本对照的情况下称为确切bug提交。</p>
<details><summary>历史证据与未决项</summary><p>{self.link('evidence/historical_report_findings.json', '本机报告证据与精确路径')} · {self.link('evidence/remote_history_audit.json', '远端历史扫描')}</p>{history_content}{audit_content}</details></section>
<section><h2>已完成训练与额外开销分开记账</h2><div class="scroll"><table><thead><tr><th>前阶段组</th><th>已完成正式更新</th><th>评价节点</th><th>额外开销</th></tr></thead><tbody><tr><td>Single · 仅a37</td><td>沿用原10K；前阶段新增正式0</td><td>原10K新鲜49帧评价</td><td>11977的额外1977保留工程账，后续精确续训不重记</td></tr><tr><td>Dual · 完整a37+a6</td><td>fresh 10,000更新</td><td>5K、10K</td><td>独立smoke、恢复探针、工程失败与零更新评价另记</td></tr><tr><td>Round1 原输入 / 补真实支撑</td><td>各fresh5,000；共10,000更新</td><td>1K、5K；1K→5K完整恢复</td><td>1K是正式轨迹节点，不重复加算</td></tr></tbody></table></div><p>累计已完成正式20,000＋工程6,599＝26,599更新，范围截至Round1；当前研究新增步骤另按实际回执计账。{self.link('evidence/execution_accounting_through_round1_20261107.json','累计账目与旧凭据引用')}</p>
<p>诊断节点发布不改原30K学习率时钟。旧目录中即使仍含 <code>30k</code>，也不表示继续执行30K。没有回执的工程失败更新数保持未知，不计成0。</p>{completed_ledger}<details><summary>逐项执行回执与预算合同</summary><p>{self.link('evidence/experiment_contract.json', '实验合同')} · {self.link('evidence/run_receipts/dual/execution_accounting_final.json', '已完成实账')} · {self.link('evidence/agent_visual_review.json', '代理目视记录（human_verdict为空）')}</p><ul>{receipt_html}</ul></details></section>
<section><h2>原资产与可复核性</h2><p>新鲜评价逐项记录实际读取源的SHA256、大小与mtime，并核对原生保存图重放；该范围不等于整份远端数据从未写入。旧loader已出现 <code>transform_matrix.txt</code> 同值重写，写入事件与权重只读核验分开列账，具体前后凭据以工程回执为准。历史四帧回执的大小/mtime与张量版本检查，不自动升级为全文件SHA核验。</p>
<details><summary>摘要、来源清单与源码</summary><ul>{''.join(artifacts)}</ul><p>{self.link('source/evaluate_a5.py', '原生评价源码')} · {self.link('source/build_report.py', '可重复构建源码')} · {self.link('architecture.svg', '组件图 SVG')} · {self.link('report_data.json', '本页manifest与验收')}</p></details></section>
<footer>同场景训练视角诊断；没有隐藏表面真值或跨场景泛化结论。a5质量未通过，human_verdict为空。{h(registration_text)} · {self.link('evidence/research_registration.json', '研究登记')} · {self.link('evidence/execution_accounting_through_round1_20261107.json','截至Round1累计账目')} · {self.link('evidence/run_receipts/compute_accounting_through_screen_candidate.json', '上一代码阶段实账')} · {self.link('evidence/run_receipts/final_readonly_verification.json', '上一阶段只读核验')}。更新：{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</footer>
</main></body></html>'''

    def qa(self):
        class Links(HTMLParser):
            def __init__(self):
                super().__init__()
                self.links, self.images = [], []
            def handle_starttag(self, tag, attributes):
                attributes = dict(attributes)
                if 'href' in attributes:
                    self.links.append(attributes['href'])
                if tag == 'img' and 'src' in attributes:
                    self.images.append(attributes['src'])
        parser = Links()
        parser.feed((self.root / 'index.html').read_text(encoding='utf-8'))
        checked, images = [], []
        for value in parser.links + parser.images:
            parsed = urlsplit(value)
            if parsed.scheme or value.startswith('#'):
                continue
            path = (self.root / unquote(parsed.path)).resolve()
            if not path.is_relative_to(self.root) or not path.is_file():
                raise ValueError(f'Broken or nonlocal report link: {value}')
            checked.append(value)
        for value in parser.images:
            path = self.root / unquote(urlsplit(value).path)
            if path.suffix.lower() == '.svg':
                element = ET.parse(path).getroot()
                size = [int(element.attrib['width']), int(element.attrib['height'])]
            else:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    size = list(image.size)
            images.append(dict(path=value, size=size))
        return dict(status='PASS', checked_relative_links=len(checked), broken_links=[], images=images)

    def build(self):
        history = self.read(self.evidence / 'historical_report_findings.json')
        contract = self.read(self.evidence / 'experiment_contract.json')
        audit = self.read(self.evidence / 'remote_history_audit.json')
        self.visual_review = self.read(self.evidence / 'agent_visual_review.json')
        self.cached_native10k = self.read(self.evidence / 'cached_native10k_comparison/summary.json')
        self.early_bounds = self.read(self.evidence / 'early_causal_bounds.json')
        self.registration = self.read(self.evidence / 'research_registration.json')
        self.historical_goal_authorization = self.read(self.evidence / 'new_goal_authorization.json')
        self.research_authorization = self.read(self.evidence / 'research_authorization_20261107.json')
        self.research_contract = self.read(self.evidence / 'research_20261107_contract.json')
        self.scope_change = self.read(self.evidence / 'reconstruction_scope_change_20261107.json') or self.read(self.evidence / 'engineering_scope_change_20261107.json')
        self.accounting_round1 = self.read(self.evidence / 'execution_accounting_through_round1_20261107.json')
        self.new_goal_authorization = self.research_authorization or self.historical_goal_authorization
        self.final_accounting = self.read(self.evidence / 'run_receipts/compute_accounting_through_screen_candidate.json')
        self.final_readonly = self.read(self.evidence / 'run_receipts/final_readonly_verification.json')
        datasets = self.load_datasets()
        self.make_figures(datasets)
        self.engineering = self.engineering_data(datasets)
        self.screen_radius = self.screen_radius_data(datasets)
        self.support_ablation = self.support_ablation_data()
        self.research = self.research_data(datasets)
        self.research_figures = self.make_research_figures(self.research)
        self.research_accounting = self.research_accounting_data()
        if self.scope_change and not self.research_accounting['incomplete_or_unretrieved_intervals']:
            self.research['status']='HISTORICAL_DIAGNOSTICS_COMPLETE_SCOPE_MOVED_TO_ENGINEERING'
        (self.evidence/'research_execution_accounting_current.json').write_text(
            json.dumps(self.research_accounting,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        old_rows = self.old_anchor_rows(datasets)
        receipts = self.receipt_records()
        comparison = self.comparison_rows(datasets)
        self.write_csv(comparison)
        page = self.render(datasets, old_rows, receipts, contract, history, audit)
        (self.root / 'index.html').write_text(page, encoding='utf-8')
        # QA follows final paths; report_data is a declared generated target.
        (self.root / 'report_data.json').write_text('{}\n', encoding='utf-8')
        qa = self.qa()
        self.consumed.update(path.resolve() for pattern in ('*.py', '*.patch') for path in (self.root / 'source').glob(pattern))
        if (self.evidence / 'single_bug_source_audit.md').is_file():
            self.consumed.add((self.evidence / 'single_bug_source_audit.md').resolve())
        inputs = [dict(path=self.relative(path), bytes=path.stat().st_size, sha256=sha256(path))
                  for path in sorted(self.consumed)]
        outputs = [self.root / 'index.html', self.root / 'comparison.csv', self.root / 'architecture.svg',
                   self.root / 'bug_time_shift.svg', self.root / self.engineering['figure']['path'],
                   self.root / self.screen_radius['figure']['path'],
                   Path(__file__).resolve(), *sorted((self.root / 'figures').glob('raw*_comparison.png'))]
        outputs.extend(self.root/item['path'] for item in self.support_ablation['preview_figures'])
        outputs.extend(self.root/path for path in self.research_figures)
        outputs.append(self.evidence/'research_execution_accounting_current.json')
        formal_ready = all(item['validated'] for item in datasets if item['id'] in ('single10k', 'dual5k', 'dual10k'))
        both_ready = formal_ready and self.engineering['paired_ready'] and self.screen_radius['paired_ready']
        both_negative = both_ready and self.engineering['adoption_decision'] == 'NOT_ADOPTED_NEGATIVE_A5_REPLAY' and self.screen_radius['adoption_decision'] == 'NOT_ADOPTED_NEGATIVE_A5_REPLAY'
        prior_engineering = self.engineering['engineering_accounting'].get('prior_completed_engineering_updates')
        completed_engineering = None if prior_engineering is None else prior_engineering+(2011 if self.engineering['paired_ready'] else 0)+(2107 if self.screen_radius['paired_ready'] else 0)
        if both_ready and self.final_accounting is not None:
            if self.final_accounting.get('formal_new_updates') != 10000 or self.final_accounting.get('total_engineering_updates') != completed_engineering:
                raise ValueError('Final receipt and per-trial completed update totals disagree')
        completed_formal = (10000 if formal_ready else 0)+(10000 if self.support_ablation['paired5k_ready'] else 0)
        if self.accounting_round1 and (self.accounting_round1.get('completed_formal_updates',{}).get('total') != completed_formal or
                                      self.accounting_round1.get('completed_engineering_total') != completed_engineering):
            raise ValueError('Round1 completed accounting disagrees with validated experiment receipts')
        data = dict(status='HISTORICAL_DIAGNOSTICS_SCOPE_MOVED_TO_IDENTITY_EDIT_ENGINEERING' if self.scope_change else 'RESEARCH_IN_PROGRESS_INITIALIZATION_AND_STATIC_RGB_NOT_YET_CAUSALLY_RESOLVED',
                    prior_code_trials_status='COMPLETE_A5_QUALITY_NOT_ACCEPTED' if both_negative else 'PRIOR_EVIDENCE_PENDING',
                    generated_utc=datetime.now(timezone.utc).isoformat(), quality_accepted=False, human_verdict=None,
                     a5_root_cause='IDENTITY_PARTITION_COUPLES_INPUT_AND_OBJECTIVE_MECHANISM_PENDING',
                     historical_commit_attribution='NOT_UNIQUELY_ATTRIBUTED_TO_A_COMMIT',
                     completed_accounting=dict(formal_new_updates=self.research_accounting['completed_formal_updates'],
                                              completed_receipted_engineering_updates=completed_engineering,
                                               total_new_training_updates=self.research_accounting['completed_total_updates'],
                                               scope=self.research_accounting['scope'],
                                               receipt=self.research_accounting,
                                               preserved_round1_receipt=self.accounting_round1,
                                              code_trial_GPU_diagnostics_ended=both_ready,
                                              all_authorized_GPU_diagnostics_ended=False,
                                               historical_research_diagnostics_complete=not self.research_accounting['incomplete_or_unretrieved_intervals'] and self.scope_change is not None,
                                              pending_is_not_counted_as_completed=True),
                    current_budget=dict(single_original_steps=10000, single_additional_formal_steps=0,
                                         scope='Prior baseline/code-trial phase; newly authorized support budgets are listed in support_ablation',
                                        single_existing_smoke_extra=1977, dual_fresh_formal_steps=10000,
                                        dual_diagnostic_nodes=[5000, 10000], lr_horizon=30000,
                                        full5_B30k_role='historical reference, not matched training length',
                                        full5_B10k='four checkpoint weights cleaned; native event and cached49 PNG retained, CPU uint8 comparison separate', directory_names_are_not_budgets=True),
                     fixed_domain=dict(rule='actor_labels==1', expected_pixel_frames=PIXELS, visible_frames=24,
                                       visible_raw_ids=list(range(89,113)), evaluated_training_views=49,
                                       independent_generalization_evaluation=False,crop_view_only=list(CROP)),
                    evaluations=[dict(id=item['id'], label=item['label'], folder=item['fresh_folder'],
                                      validated=item['validated'], status=item['status'], problems=item['problems'],
                                      summary=item['summary'] if item['validated'] else None) for item in datasets],
                    comparison=comparison, legacy_four_anchor_rows=old_rows,
                    figures=self.figure_records, engineering_diagnosis=self.engineering,
                    screen_radius_diagnosis=self.screen_radius,
                    support_ablation=self.support_ablation,
                     research=self.research,
                     engineering_scope_change=self.scope_change,
                    current_stage_authorization=self.new_goal_authorization,
                     historical_goal_authorization=self.historical_goal_authorization,
                    cached_native10k_comparison=self.cached_native10k,
                    early_causal_bounds=self.early_bounds,
                    research_registration=self.registration, final_accounting_receipt=self.final_accounting,
                    final_readonly_verification=self.final_readonly,
                    agent_visual_review=self.visual_review,
                    execution_receipts=receipts, input_parse_errors=self.errors,
                    input_manifest=inputs,
                    output_manifest=[dict(path=self.relative(path), bytes=path.stat().st_size, sha256=sha256(path)) for path in outputs],
                    qa=qa, manifest_exclusion='report_data.json omits its own hash to avoid self-reference')
        serialized = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        (self.root / 'report_data.json').write_text(serialized, encoding='utf-8')
        # Reparse the delivered JSON with strict non-finite handling.
        json.loads(serialized, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        print(json.dumps(dict(status=data['status'], qa=qa, parse_errors=self.errors), ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    options = parser.parse_args()
    # Avoid Windows strict-realpath directory probes; ordinary read/write checks
    # below establish availability in the explicitly chosen report directory.
    root = options.root.absolute()
    if not (root / 'evidence').is_dir() or not (root / 'architecture.svg').is_file():
        raise FileNotFoundError('Expected evidence/ and architecture.svg in report root')
    Builder(root).build()


if __name__ == '__main__':
    main()
