"""Publish actual paired v44 GPU results without inventing a quality verdict.

The report uses relative PNG paths and embedded metadata. It opens directly from
file:// without fetch, a local server, a browser dependency, or large float NPZs.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import hashlib
import html
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import struct
from typing import Any


REPORT = Path(__file__).resolve().parent
WORKSPACE = REPORT.parents[2]
DEFAULT_CPU = WORKSPACE / 'outputs/scene008_v44_rgb_asset_integration'
DEFAULT_CROPS = WORKSPACE / 'outputs/scene008_vehicle_appearance/diagnosis/frame_compare_manifest.json'
MARKER = 'data-v44-results-report="1"'
PILOT = [105, 113, 130, 137]
STATES = ('keep', 'delete')
VARIANTS = ('current', 'v44')
METRICS = ('a6_RGB', 'a6_depth_m', 'road_RGB', 'a37_RGB', 'protected_RGB')
LABELS = {'a6_RGB': 'a6 RGB MAE', 'a6_depth_m': 'a6 深度 MAE (m)',
          'road_RGB': '道路 RGB MAE', 'a37_RGB': 'a37 RGB MAE',
          'protected_RGB': 'protected RGB MAE'}
SPLITS = {'ALL': '本次全部帧', 'BUILD40': 'BUILD40',
          'HELD9_internal_development': 'HELD9 · 内部开发'}


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def number(value: Any, digits: int = 6) -> str:
    return '—' if value is None else f'{value:.{digits}f}'


def count(value: int) -> str:
    return f'{value:,}'


def delta(value: Any) -> str:
    if value is None:
        return '—'
    style = 'delta-up' if value > 0 else 'delta-down' if value < 0 else ''
    return f'<span class="{style}">{value:+.6f}</span>'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    def invalid(value: str) -> None:
        raise ValueError(f'Non-finite JSON value {value} in {path}')
    return json.loads(path.read_text(encoding='utf-8-sig'), parse_constant=invalid)


def write_json(path: Path, value: Any) -> None:
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(content, encoding='utf-8')
    temporary.replace(path)


def copy_exact(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if digest(source) != digest(target):
            raise ValueError(f'Existing evidence has different bytes: {target}')
    else:
        shutil.copy2(source, target)


def same_number(actual: Any, expected: Any, where: str) -> None:
    if actual is None or expected is None:
        if actual is not expected:
            raise ValueError(f'Null mismatch: {where}')
    elif not math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-10):
        raise ValueError(f'Metric mismatch: {where}: {actual} != {expected}')


def png_size(path: Path) -> tuple[int, int]:
    with path.open('rb') as stream:
        header = stream.read(24)
    if header[:8] != b'\x89PNG\r\n\x1a\n' or header[12:16] != b'IHDR':
        raise ValueError(f'Not a PNG: {path}')
    return struct.unpack('>II', header[16:24])


def validate_box(box: Any, label: str) -> list[int]:
    if not isinstance(box, (list, tuple)) or len(box) != 4 or not all(isinstance(x, int) for x in box):
        raise ValueError(f'Invalid registered crop: {label}')
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= 480 and 0 <= y0 < y1 <= 320):
        raise ValueError(f'Crop is outside 480x320: {label}')
    return list(box)


def relative_png(root: Path, entry: dict, expected: str) -> str:
    relative = PurePosixPath(entry['path'])
    if relative.is_absolute() or '..' in relative.parts or relative.as_posix() != expected:
        raise ValueError(f'Unexpected result PNG path: {relative}; expected {expected}')
    source = root.joinpath(*relative.parts)
    if png_size(source) != (480, 320) or digest(source) != entry['sha256']:
        raise ValueError(f'PNG size/hash mismatch: {source}')
    return relative.as_posix()


@dataclass
class Dataset:
    root: Path
    prefix: str
    receipts: dict
    comparison: dict
    frames: dict
    manifest: dict
    comparison_sha256: str

    @property
    def mode(self) -> str:
        return self.comparison['mode']

    @property
    def raws(self) -> list[int]:
        return self.manifest['raw_ids']

    def link(self, relative: str) -> str:
        return f'{self.prefix}/{relative}'


def load_dataset(root: Path, label: str, crops: dict) -> Dataset:
    root = root.resolve(strict=True)
    comparison = read_json(root / 'comparison.json')
    comparison_sha256 = digest(root / 'comparison.json')
    prefix = f'data/{label}_{comparison_sha256[:12]}'
    receipts = {variant: read_json(root / variant / 'receipt.json') for variant in VARIANTS}
    if comparison.get('method') != 'v44' or set(comparison.get('variants', [])) != set(VARIANTS):
        raise ValueError('Expected actual current/v44 comparison')
    mode = comparison['mode']
    if mode not in ('pilot', 'full49'):
        raise ValueError(f'Unknown result mode: {mode}')
    expected_raws = PILOT if mode == 'pilot' else list(range(89, 138))
    for variant, receipt in receipts.items():
        if receipt['raws'] != expected_raws or receipt['mode'] != mode or receipt['variant'] != variant:
            raise ValueError(f'Receipt frame/mode mismatch: {variant}')
        if receipt.get('status') != 'RENDERED_NOT_HUMAN_ACCEPTED':
            raise ValueError(f'Unexpected render status: {variant}')
        for key in ('neural_weight_updates', 'geometry_updates', 'colour_fit_updates', 'native_or_road_training_updates'):
            if receipt.get(key) != 0:
                raise ValueError(f'Out-of-contract training/update counter: {variant}/{key}')
    if receipts['current']['persistent_SF_hash'] != receipts['v44']['persistent_SF_hash']:
        raise ValueError('Persistent current/v44 S/F asset hash differs')
    frames = {variant: {frame['raw']: frame for frame in receipt['frames']}
              for variant, receipt in receipts.items()}
    if any(sorted(items) != expected_raws or len(receipts[variant]['frames']) != len(items)
           for variant, items in frames.items()):
        raise ValueError('Missing or duplicate frame records')
    manifest = dict(schema='v44-paired-viewer-v1', mode=mode, raw_ids=expected_raws,
                    default_raw=113, default_state='delete', width=480, height=320,
                    comparison_sha256=comparison_sha256, frames={})
    for raw in expected_raws:
        before, after = frames['current'][raw], frames['v44'][raw]
        for key in ('camera_pose_identity', 'actual_camera_pose', 'split'):
            if before[key] != after[key]:
                raise ValueError(f'Paired {key} differs at raw{raw}')
        item = dict(raw=raw, split=before['split'], views={}, current={}, v44={})
        if str(raw) not in crops.get('frames', {}):
            raise ValueError(f'No inherited common body crop for raw{raw}; do not invent one')
        item['views']['crop'] = validate_box(crops['frames'][str(raw)]['crop_xyxy'], f'raw{raw}/body')
        for variant in VARIANTS:
            frame = frames[variant][raw]
            if read_json(root / variant / f'{raw:03d}' / 'metrics.json') != frame:
                raise ValueError(f'Frame metrics differ from receipt: {variant}/{raw}')
            states = frame['states']
            if states['keep']['keep_delete_asset_rows'] != states['delete']['keep_delete_asset_rows']:
                raise ValueError(f'keep/delete asset rows differ: {variant}/{raw}')
            for state in STATES:
                metric = states[state]
                if metric['meta']['native_a6_rows_input'] != 0:
                    raise ValueError(f'Native a6 was not removed: {variant}/{raw}/{state}')
                if state == 'delete' and metric['meta']['native_a37_packed_rows'] != 0:
                    raise ValueError(f'a37 remains in delete: {variant}/{raw}')
                if metric['group_alpha_conservation_maxabs'] >= 2e-5:
                    raise ValueError(f'Group alpha does not conserve: {variant}/{raw}/{state}')
                relative = relative_png(root, metric['PNG'], f'{variant}/{raw:03d}/{state}.png')
                item[variant][state] = dict(path=f'{prefix}/{relative}', sha256=metric['PNG']['sha256'])
                for field, view in (('blue_fixed_ROI', 'blue'), ('silver_fixed_ROI', 'silver')):
                    if field in metric:
                        box = validate_box(metric[field]['xyxy'], f'{variant}/{raw}/{field}')
                        if view in item['views'] and item['views'][view] != box:
                            raise ValueError(f'ROI differs between paired states: {raw}/{view}')
                        item['views'][view] = box
            observed = root / variant / f'{raw:03d}' / 'observed_a37_present.png'
            if png_size(observed) != (480, 320):
                raise ValueError(f'Observed PNG has wrong dimensions: {observed}')
        observed_current = root / 'current' / f'{raw:03d}' / 'observed_a37_present.png'
        observed_candidate = root / 'v44' / f'{raw:03d}' / 'observed_a37_present.png'
        if digest(observed_current) != digest(observed_candidate):
            raise ValueError(f'Original observation differs across paired runs: raw{raw}')
        item['observed'] = dict(path=f'{prefix}/current/{raw:03d}/observed_a37_present.png',
                                sha256=digest(observed_current))
        manifest['frames'][str(raw)] = item
    pairs = {(item['raw'], item['state']): item for item in comparison['comparisons']}
    if len(comparison['comparisons']) != len(pairs) or set(pairs) != {(r, s) for r in expected_raws for s in STATES}:
        raise ValueError('Missing or duplicate per-frame comparisons')
    for (raw, state), paired in pairs.items():
        for key, change in paired['changes'].items():
            before, after = (frames[v][raw]['states'][state][key] for v in VARIANTS)
            if before['pixels'] != after['pixels'] or change['pixels'] != before['pixels']:
                raise ValueError(f'Paired fixed denominator differs: {raw}/{state}/{key}')
            same_number(change['before'], before['mean'], f'{raw}/{state}/{key}/before')
            same_number(change['after'], after['mean'], f'{raw}/{state}/{key}/after')
            expected_delta = None if before['mean'] is None else after['mean'] - before['mean']
            same_number(change['delta'], expected_delta, f'{raw}/{state}/{key}/delta')
    for pooled in comparison['pixel_weighted_pooled']:
        state, split, key = pooled['state'], pooled['split'], pooled['metric']
        selected = [raw for raw in expected_raws if split == 'ALL' or frames['current'][raw]['split'] == split]
        n = sum(frames['current'][raw]['states'][state][key]['pixels'] for raw in selected)
        sums = [sum(frames[v][raw]['states'][state][key]['sum'] for raw in selected) for v in VARIANTS]
        if n != pooled['pixels'] or len(selected) != pooled['frames']:
            raise ValueError(f'Pooled fixed denominator differs: {state}/{split}/{key}')
        for name, expected in [('before', sums[0] / n if n else None), ('after', sums[1] / n if n else None),
                               ('delta', (sums[1] - sums[0]) / n if n else None)]:
            same_number(pooled[name], expected, f'pooled/{state}/{split}/{key}/{name}')
    return Dataset(root, prefix, receipts, comparison, frames, manifest, comparison_sha256)


def copy_dataset(dataset: Dataset, output: Path) -> list[dict]:
    copied = []
    # Only the original paired-run deliverables. Root-created review boards are
    # separate evidence and must not become native render counts by accident.
    relatives = {'comparison.json'}
    for variant in VARIANTS:
        relatives.add(f'{variant}/receipt.json')
        if (dataset.root / variant / 'progress.json').is_file():
            relatives.add(f'{variant}/progress.json')
        for raw in dataset.raws:
            relatives.add(f'{variant}/{raw:03d}/metrics.json')
            for state in STATES:
                relatives.add(f'{variant}/{raw:03d}/{state}.png')
    for raw in dataset.raws:
        relatives.add(f'current/{raw:03d}/observed_a37_present.png')
    for relative in sorted(relatives):
        source = dataset.root / relative
        target = output / dataset.prefix / relative
        copy_exact(source, target)
        copied.append(dict(path=f'{dataset.prefix}/{relative}', sha256=digest(source), bytes=source.stat().st_size))
    return copied


def validate_review(review: dict, dataset: Dataset) -> None:
    """A pilot review must never acquire full49 scope by being reused."""
    accepted = review.get('quality_accepted')
    if accepted is not None and not isinstance(accepted, bool):
        raise ValueError('visual-review quality_accepted must be true, false or null')
    if review.get('run') != dataset.root.name:
        raise ValueError(f'Visual review run does not match {dataset.root.name}')
    if 'mode' in review and review['mode'] != dataset.mode:
        raise ValueError('Visual review mode does not match result')
    if 'raws' in review and review['raws'] != dataset.raws:
        raise ValueError('Visual review raw IDs do not match result')
    if 'comparison_sha256' in review and review['comparison_sha256'] != dataset.comparison_sha256:
        raise ValueError('Visual review comparison SHA does not match result')
    if review.get('full49_executed') is not (dataset.mode == 'full49'):
        raise ValueError('Visual review full49 execution flag does not match its scope')
    if dataset.mode == 'full49' and (review.get('mode') != 'full49' or review.get('raws') != dataset.raws
                                    or review.get('comparison_sha256') != dataset.comparison_sha256):
        raise ValueError('Full49 visual review requires explicit mode, all raw IDs and comparison SHA binding')


def seed_cpu_archive(output: Path, source: Path) -> dict:
    """Keep the old CPU HTML and every relative asset it already depended on."""
    output.mkdir(parents=True, exist_ok=True)
    archived = output / 'cpu_stage.html'
    original = source / 'cpu_stage.html'
    if not original.exists():
        original = source / 'index.html'
    if not archived.exists():
        if not original.is_file() or MARKER in original.read_text(encoding='utf-8-sig'):
            raise ValueError('CPU report archive is missing; provide --cpu-stage-root with the original CPU report')
        copy_exact(original, archived)
    for folder in ('assets', 'evidence', 'code'):
        source_folder = source / folder
        if source_folder.is_dir() and source_folder.resolve() != (output / folder).resolve():
            for file in source_folder.rglob('*'):
                if file.is_file() and '__pycache__' not in file.parts and file.suffix.lower() not in ('.npz', '.pyc'):
                    copy_exact(file, output / folder / file.relative_to(source_folder))
    copy_exact(source / 'architecture.svg', output / 'architecture.svg')
    if (source / 'README.md').is_file() and not (output / 'README_cpu_stage.md').exists():
        copy_exact(source / 'README.md', output / 'README_cpu_stage.md')
    if (source / 'RUN_COMMANDS.txt').is_file():
        copy_exact(source / 'RUN_COMMANDS.txt', output / 'RUN_COMMANDS.txt')
    return dict(path='cpu_stage.html', sha256=digest(archived))


def table(headers: list[str], rows: list[list[str]], *, caption: str | None = None) -> str:
    head = ''.join(f'<th scope="col">{esc(header)}</th>' for header in headers)
    body = ''.join('<tr>' + ''.join(f'<td>{cell}</td>' for cell in row) + '</tr>' for row in rows)
    cap = f'<caption>{esc(caption)}</caption>' if caption else ''
    return f'<div class="scroll"><table>{cap}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def card(dataset: Dataset, raw: int, state: str, column: str, *, initial: bool = False, lazy: bool = False) -> str:
    frame = dataset.manifest['frames'][str(raw)]
    entry = frame['observed'] if column == 'observed' else frame[column][state]
    labels = {'current': 'current：当前车体＋旧背景', 'v44': 'v44：同一车体＋RCR-1 背景',
              'observed': '原始实拍：a37 仍在，非删车 GT'}
    path = esc(entry['path'])
    identity = f' id="v44-image-{column}"' if initial else ''
    loading = ' loading="lazy"' if lazy else ''
    return (f'<article class="v44-card"><h3>{esc(labels[column])}</h3><div class="v44-window">'
            f'<img{identity} src="{path}" width="480" height="320"{loading} '
            f'alt="raw{raw} / {state} a37 / {esc(labels[column])}" data-raw="{raw}" '
            f'data-state="{state}" data-column="{column}"></div><div class="v44-card-footer">'
            f'<a href="{path}" target="_blank" rel="noopener">打开原尺寸 PNG</a><code>{path}</code></div></article>')


def viewer(dataset: Dataset, static_page: str) -> str:
    pilot = dataset.mode == 'pilot'
    dataset.manifest['static_page'] = static_page
    slider = (f'<label for="v44-range">四帧 pilot 索引 0–3</label><input id="v44-range" type="range" '
              'min="0" max="3" step="1" value="1" data-v44-control disabled>' if pilot else
              '<label for="v44-range">raw89–137 · 完整49帧</label><input id="v44-range" type="range" '
              'min="89" max="137" step="1" value="113" data-v44-control disabled>')
    selected = '索引 1 / raw113' if pilot else '选择 raw113'
    jumps = ''.join(f'<button type="button" data-v44-raw="{raw}" data-v44-control disabled>{raw}</button>' for raw in PILOT)
    initial = ''.join(card(dataset, 113, 'delete', column, initial=True) for column in ('current', 'v44', 'observed'))
    data = json.dumps(dataset.manifest, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    data = data.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    return f'''<section id="v44-compare" aria-label="本次GPU成对逐帧对照"><h2>{'四帧 pilot' if pilot else '49帧'}同步对照</h2>
<p class="note">current 与 v44 使用同一冻结车体、同一帧相机与轨迹。保留／删除仅开关 a37。原始实拍列仍有 a37，不能当删除后的隐藏道路真值。初始 raw113 的真实图片直接写在 src 中，无 JavaScript 也可见。</p>
<div class="v44-controls"><button id="v44-prev" type="button" data-v44-control disabled>上一帧</button><div class="v44-range-row">{slider}<output id="v44-requested" for="v44-range">{selected}</output></div><button id="v44-next" type="button" data-v44-control disabled>下一帧</button>{jumps}
<label for="v44-state">a37 <select id="v44-state" data-v44-control disabled><option value="delete" selected>delete 删除</option><option value="keep">keep 保留</option></select></label>
<label for="v44-view">窗口 <select id="v44-view" data-v44-control disabled><option value="full" selected>全景 480×320</option><option value="crop">同帧共用车体裁剪</option><option value="blue">固定蓝 ROI 框</option><option value="silver">固定银 ROI 框</option></select></label>
<label><input id="v44-observed" type="checkbox" checked data-v44-control disabled>显示原始实拍第三列</label></div>
<p id="v44-current">当前显示：raw113 / delete a37 / 全景 480×320 / 含原始实拍列</p>
<p id="v44-status" role="status" aria-live="polite">静态初始 raw113 已写入真实图片 src；切换需要 JavaScript。整组图片预解码成功后才同步更新。</p>
<div class="v44-stage"><div id="v44-grid" class="v44-grid" style="--v44-columns:3">{initial}</div><div id="v44-overlay" hidden aria-live="polite"></div></div>
<p id="v44-crop-note" class="note">全景 480×320。所有列均为本次实际结果 PNG；可打开原尺寸查看。</p>
<p class="links"><a id="v44-static-current" href="{esc(static_page)}#raw113-delete">无需JavaScript的全部静态帧</a> · <a href="{dataset.prefix}/viewer_manifest.json">真实图片映射与 SHA256</a> · <a href="code/report/v44_compare_controller.js">切换脚本</a></p>
<noscript><p class="note">当前显示 raw113 / delete 全景。逐帧和 keep/delete 可通过静态页面查看。</p></noscript>
<script id="v44-viewer-data" type="application/json">{data}</script></section>'''


def metric_sections(dataset: Dataset) -> str:
    pooled_rows = []
    for item in dataset.comparison['pixel_weighted_pooled']:
        pooled_rows.append([esc(item['state']), esc(SPLITS.get(item['split'], item['split'])), esc(LABELS[item['metric']]),
                            count(item['frames']), count(item['pixels']), number(item['before']),
                            number(item['after']), delta(item['delta'])])
    pooled = table(['状态', '域', '指标（越低越好）', '帧', '像素数', 'current', 'v44', 'Δ v44−current'], pooled_rows)
    regression_rows = []
    for state in STATES:
        selected = [item for item in dataset.comparison['comparisons'] if item['state'] == state]
        for key in METRICS:
            available = [item for item in selected if key in item['changes'] and item['changes'][key]['delta'] is not None]
            worse = [item for item in available if item['changes'][key]['delta'] > 0]
            if not available:
                continue
            worst = max(available, key=lambda item: item['changes'][key]['delta'])
            regression_rows.append([esc(state), esc(LABELS[key]), f'{len(worse)} / {len(available)}',
                                    ', '.join('raw' + str(item['raw']) for item in worse) or '无正增量',
                                    f'raw{worst["raw"]}: {delta(worst["changes"][key]["delta"])}'])
    regressions = table(['状态', '指标', '误差增加帧数', '误差增加帧', '最大逐帧 Δ'], regression_rows)
    frame_rows = []
    for item in dataset.comparison['comparisons']:
        cells = [f'raw{item["raw"]}', esc(item['state'])]
        for key in METRICS:
            change = item['changes'].get(key)
            cells.append('—' if change is None else f'<span class="metric">{number(change["before"])} → {number(change["after"])}</span><br>Δ {delta(change["delta"])}<br><span class="small">n={count(change["pixels"])}</span>')
        outside = item['protected_outside_change']
        cells.append(f'{number(outside["mean_channel_absolute"]["mean"])} / {number(outside["max_channel_absolute"])}<br>'
                     f'<span class="small">最大通道变化&gt;0.02: {count(outside["pixels_max_channel_change_gt002"])} / {count(outside["mean_channel_absolute"]["pixels"])}</span>')
        frame_path = f'{item["raw"]:03d}/metrics.json'
        cells.append(f'<a href="{dataset.link("current/" + frame_path)}">current</a> / '
                     f'<a href="{dataset.link("v44/" + frame_path)}">v44</a>')
        frame_rows.append(cells)
    per_frame = table(['帧', '状态', 'a6 RGB', 'a6 深度 (m)', '道路 RGB', 'a37 RGB', 'protected RGB',
                       '非目标区版本间变化：均值 / 最大值', '原始 metrics'], frame_rows)
    return f'''<section id="pooled"><h2>像素加权汇总与逐帧回退</h2>
<p>以下误差使用固定原始观测掩码，以浮点误差 sum / 像素 count 汇总。Δ 为 v44−current，正值表示该项误差增加；不能用平均改善覆盖局部回退。深度为同次排序后 alpha 归一化深度，背景改动仍可改变它。</p>
{pooled}<p class="note">HELD9 已参与内部开发，不是独立泛化测试。本次四个 pilot raw 全属于 HELD9 时，ALL 与 HELD 行相同，也不代表覆盖全部九帧或49帧。delete 道路 target 是跨时刻重观测 proxy；始终被遮挡区域没有独立隐藏 GT。</p>
<h3>逐帧误差增加</h3>{regressions}<h3>完整逐帧数据</h3>{per_frame}
<p class="note">“非目标区版本间变化”衡量两版本差异，使用 protect 减两车 reviewed／候选框；它不是该区域准确率，不能单独判断变化更好或更坏。保护域与道路域的 pooled 改善不证明其他车辆局部颜色没有回退。</p></section>'''


def blue_section(dataset: Dataset) -> str:
    rows = []
    for raw in dataset.raws:
        for state in STATES:
            pair = [dataset.frames[v][raw]['states'][state].get('blue_fixed_ROI') for v in VARIANTS]
            if not all(pair):
                continue
            before, after = pair
            b, a = before['absolute_excess'], after['absolute_excess']
            if b['pixels'] != a['pixels']:
                raise ValueError(f'Blue fixed denominator differs at raw{raw}/{state}')
            change = None if b['mean'] is None else a['mean'] - b['mean']
            rows.append([f'raw{raw}', esc(state), esc(str(before['xyxy'])), count(b['pixels']), number(b['mean']),
                         number(a['mean']), delta(change), f'{count(before["positive_excess_gt005"])} → {count(after["positive_excess_gt005"])}',
                         '空域／不可评价' if not b['pixels'] else '固定道路候选掩码'])
    return '<section id="blue"><h2>预登记蓝 ROI：保留原始分母</h2><p>蓝残差为 B−(R+G)/2，相对重观测道路 proxy。矩形内再与原始 road_edit、ground proxy 相交并排除 a6／a37；不会为得到数值而扩大或改换原分母。n=0 时均值保持 null／“—”，不得写成误差为0或去蓝成功。</p>' + table(
        ['帧', '状态', '[x0,y0,x1,y1)', '有效像素 n', 'current 绝对蓝残差', 'v44 绝对蓝残差', 'Δ', '正蓝残差>0.05 像素', '域状态'], rows) + '</section>'


def observed_road_section(dataset: Dataset, audits: list[tuple[dict, str]]) -> str:
    matched = next(((audit, path) for audit, path in audits
                    if audit['comparison_sha256'] == dataset.comparison_sha256 and
                    audit.get('observed_road_blue_diagnostics')), None)
    if matched is None:
        return ''
    audit, audit_path = matched
    observations = audit['observed_road_blue_diagnostics']
    if {(item['raw'], item['state']) for item in observations} != {(raw, state) for raw in dataset.raws for state in STATES}:
        raise ValueError('Observed-road diagnostics are incomplete')
    if len(observations) != 2 * len(dataset.raws):
        raise ValueError('Observed-road diagnostics contain duplicate frames')
    pooled_rows, split_rows, common_rows, frame_rows, summary_rows = [], [], [], [], []
    for item in audit['observed_road_blue_pooled']:
        before, after = item['current'], item['v44']
        cells = [esc(item['state']), esc(SPLITS.get(item['split'], item['split'])),
                 count(item['frames']), count(item['pixels']),
                 number(before['rgb_mae']['mean']) + ' → ' + number(after['rgb_mae']['mean']),
                 number(before['absolute_blue_error']['mean']) + ' → ' + number(after['absolute_blue_error']['mean']),
                 delta(after['absolute_blue_error']['mean'] - before['absolute_blue_error']['mean']),
                 f'{count(before["positive_blue_error_gt005"])} → {count(after["positive_blue_error_gt005"])}']
        if item['domain'] == 'fixed_observed_road':
            (pooled_rows if item['split'] == 'ALL' else split_rows).append(cells)
        else:
            common_rows.append([esc(item['domain'])] + cells)
    for state in STATES:
        selected = [item for item in observations if item['state'] == state]
        mean_deltas = [(item['raw'], item['fixed_observed_road']['v44']['absolute_blue_error']['mean'] -
                        item['fixed_observed_road']['current']['absolute_blue_error']['mean']) for item in selected]
        count_deltas = [(item['raw'], item['fixed_observed_road']['v44']['positive_blue_error_gt005'] -
                         item['fixed_observed_road']['current']['positive_blue_error_gt005']) for item in selected]
        worst = max(mean_deltas, key=lambda item: item[1])
        summary_rows.append([esc(state), f'{sum(value < 0 for _, value in mean_deltas)} / {len(selected)}',
                             f'{sum(value < 0 for _, value in count_deltas)} / {len(selected)}',
                             f'raw{worst[0]}：{delta(worst[1])}'])
    for item in observations:
        domain = item['fixed_observed_road']
        before, after = domain['current'], domain['v44']
        frame_rows.append([f'raw{item["raw"]}', esc(item['state']), esc(SPLITS.get(item['split'], item['split'])),
                           count(domain['pixels']), number(before['absolute_blue_error']['mean']),
                           number(after['absolute_blue_error']['mean']),
                           delta(after['absolute_blue_error']['mean'] - before['absolute_blue_error']['mean']),
                           f'{count(before["positive_blue_error_gt005"])} → {count(after["positive_blue_error_gt005"])}'])
    headers = ['状态', '域', '帧', '固定像素数', 'RGB MAE current→v44', '绝对蓝误差 current→v44', '绝对蓝误差 Δ', '正蓝误差>0.05 像素']
    return f'''<section id="road-blue"><h2>本轮主结果：原观测可见道路去蓝</h2>
<p>以原始可见道路RGB为target，固定排除原观测车辆与reviewed区域；没有统一去除真实天空或蓝色车辆。绝对蓝误差为 |ΔB−(ΔR+ΔG)/2|（Δ=预测−原RGB）。该域独立命名为fixed_observed_road，不替换下方原预登记蓝ROI空域，也不评价始终遮挡的隐藏道路。</p>
{table(headers, pooled_rows)}<h3>逐帧核对：均值与尾部同时检查</h3>
{table(['状态', '绝对蓝均值下降帧数', '>0.05尾部计数减少帧数', '最大逐帧绝对蓝误差 Δ'], summary_rows)}
<p class="note">此处道路域的逐帧改善不等于全图无回退。a6深度、a37与hood局部尾部的变化在下方逐项保留；HELD9仍是内部开发域。</p>
<details><summary>展开BUILD／HELD分层与完整{len(observations)}组逐帧道路蓝指标</summary>{table(headers, split_rows)}
{table(['帧', '状态', '域', '像素数', 'current绝对蓝误差', 'v44绝对蓝误差', 'Δ', '正蓝误差>0.05 像素'], frame_rows)}
<h3>共同C2排除道路域（带预测条件的补充口径）</h3>{table(['补充域'] + headers, common_rows)}</details>
<p class="links"><a href="{esc(audit_path)}">完整原始RGB支持域审查JSON</a></p></section>'''


def silver_section(dataset: Dataset, has_layer_replay: bool = False) -> str:
    rows, group_rows = [], []
    for raw in dataset.raws:
        for state in STATES:
            pair = [dataset.frames[v][raw]['states'][state].get('silver_fixed_ROI') for v in VARIANTS]
            if not all(pair):
                continue
            before, after = pair
            b, a = before['brightness'], after['brightness']
            if b['pixels'] != a['pixels']:
                raise ValueError(f'Silver fixed denominator differs at raw{raw}/{state}')
            alpha = lambda values: ' / '.join(number(value) for value in values)
            rows.append([f'raw{raw}', esc(state), esc(str(before['xyxy'])), count(b['pixels']), number(b['mean']),
                         number(a['mean']), delta(a['mean'] - b['mean']), alpha(before['group_alpha_mean']), alpha(after['group_alpha_mean'])])
            actual = after.get('actual_group_RGB_mean')
            if actual is not None:
                decomposition = dataset.frames['v44'][raw]['states'][state]['layer_decomposition']
                if len(actual) != len(decomposition['groups']):
                    raise ValueError('Actual RGB group order/length mismatch')
                for name, rgb in zip(decomposition['groups'], actual):
                    group_rows.append([f'raw{raw}', esc(state), esc(name), ' / '.join(number(x) for x in rgb),
                                       number(sum(rgb) / 3), number(decomposition['maxabs'], 9)])
                env = after['env_RGB_mean']
                group_rows.append([f'raw{raw}', esc(state), 'environment × T', ' / '.join(number(x) for x in env),
                                   number(sum(env) / 3), number(decomposition['maxabs'], 9)])
    brightness = table(['帧', '状态', '完整 ROI', '像素数', 'current 亮度均值', 'v44 亮度均值', '亮度 Δ（非误差）',
                        'current α×T：asset/a37/road/static', 'v44 α×T：asset/a37/road/static'], rows)
    groups = table(['帧', '状态', 'v44 实际分组', 'ROI均值 R / G / B × T', '三通道均值', '像素RGB重构最大误差'], group_rows)
    replay_note = ('本节主运行current receipt只含group alpha×T；下方另列的三次真实forward重放补齐了指定ROI的current分组RGB，可在其已核对范围内比较层贡献。'
                   if has_layer_replay else 'current本轮只保存group alpha×T，没有同口径的actual group RGB；不能据此推算各层版本间净变化。')
    return f'''<div id="silver-statistics"><h3>银 ROI：完整预登记矩形</h3>
<p>按用户最新范围，本轮先解决蓝紫斑；银色残片保留到下一轮，以下诊断不作为继续49帧的门槛，也不宣称已修复。</p>
<p>整个预登记矩形作为分母；不按预测亮度挑亮点。亮度增加或减少本身都不是修复准确率：此处没有独立删车后的隐藏道路真值，也不沿用历史315亮点作为新版本分母。</p>{brightness}
<h3>v44 同一次 forward、同一次联合深度排序下的 RGB×T</h3>{groups}
<p class="note">以上为v44绝对贡献。{replay_note} 不把“v44 static贡献大”直接写成“static导致净增亮”；α×T不能替代真实RGB×T。keep状态没有额外保存actual group RGB，缺失项不补0。</p></div>'''


def supplementary_section(audits: list[tuple[dict, str]]) -> str:
    if not audits:
        return ''
    support_rows, hood_rows, links = [], [], []
    for audit, path in audits:
        label = str(audit.get('mode', '诊断'))
        links.append(f'<a href="{esc(path)}">{esc(label)}独立CPU核对与支持域原始JSON</a>')
        for item in audit.get('blue_support_diagnostics', []):
            for key, name in [('fixed_observed_background', '原始固定可见背景'),
                              ('legacy_compare_principle_common_C2_exclusion', '共同C2排除（预测条件域）')]:
                domain = item.get(key)
                if domain is None:
                    continue
                before, after = domain['current'], domain['v44']
                support_rows.append([esc(label), f'raw{item["raw"]}', esc(item['state']), esc(name),
                                     count(domain['pixels']),
                                     number(before['rgb_mae']['mean']) + ' → ' + number(after['rgb_mae']['mean']),
                                     number(before['absolute_blue_error']['mean']) + ' → ' + number(after['absolute_blue_error']['mean']),
                                     f'{count(before["positive_blue_error_gt005"])} → {count(after["positive_blue_error_gt005"])}'])
        for item in audit.get('posthoc_hood_diagnostics', []):
            before, after = item['current'], item['v44']
            hood_rows.append([esc(label), f'raw{item["raw"]}', esc(item['state']), esc(str(item['xyxy'])), count(item['pixels']),
                              number(before['rgb_mae']['mean']) + ' → ' + number(after['rgb_mae']['mean']),
                              number(before['absolute_blue_error']['mean']) + ' → ' + number(after['absolute_blue_error']['mean']),
                              f'{count(before["positive_blue_error_gt005"])} → {count(after["positive_blue_error_gt005"])}'])
    support = table(['来源', '帧', '状态', '独立命名的域', '像素数', 'RGB MAE current→v44', '绝对蓝误差 current→v44', '正蓝误差>0.05 像素'], support_rows)
    hood = table(['来源', '帧', '状态', '事后矩形', '像素数', 'RGB MAE current→v44', '绝对蓝误差 current→v44', '正蓝误差>0.05 像素'], hood_rows)
    return f'''<section id="support"><h2>补充有支持观测域：与原蓝ROI空域分开</h2>
<p>以下补充统计从已保存float结果复算，target是原始可见RGB；保留原蓝道路域n=0。固定可见背景域排除原观测车辆；共同C2排除域再要求两端asset α×T≤0.01，其分母带预测条件，不能当作原合同固定域，也不是隐藏删车GT。</p>{support}
<h3>raw105左侧来车hood：事后开发诊断</h3>{hood}
<p class="note">hood矩形由目视后指定，可能混入车体及非车像素，不是整车mask或正式接受门槛。该蓝色车辆的矩形RGB／平均蓝误差可以改善，同时正蓝误差&gt;0.05的尾部像素增加；不能把局部色感或尾部增多写成整车平均误差回退。</p>
<p class="links">{' · '.join(links)}</p></section>'''


def layer_section(layer_data: tuple[dict, str] | None) -> str:
    if layer_data is None:
        return ''
    receipt, prefix = layer_data
    rows, images = [], []
    for item in receipt['rows']:
        for name, group in item['groups'].items():
            rows.append([f'raw{item["raw"]}', esc(name),
                         ' / '.join(number(value) for value in group['current_RGB']),
                         ' / '.join(number(value) for value in group['v44_RGB']),
                         ' / '.join(f'{value:+.6f}' for value in group['delta_RGB']),
                         f'{group["delta_mean_brightness"]:+.6f}'])
        image = f'{prefix}/raw{item["raw"]}_same_sort_layers.png'
        images.append(f'<figure><figcaption>raw{item["raw"]} · 指定ROI的实际分层重放</figcaption><a href="{image}"><img loading="lazy" src="{image}" alt="raw{item["raw"]} current与v44同排序RGB层来源图"></a></figure>')
    content = table(['帧', '层', 'current RGB×T', 'v44 RGB×T', 'Δ RGB×T', 'Δ 三通道均值'], rows)
    forwards = receipt['actual_native_forward_calls']
    return f'''<section id="layers"><h2>补充分层重放：已知局部问题的来源记录</h2>
<p>针对pilot的delete a37指定ROI额外{forwards}次真实native forward，耗时{number(receipt['seconds'], 3)}秒；无训练、无参数改动。state依据已归档diagnose_pilot_layers.py中固定delete=True及delete_float对齐检查。current重放RGB与已保存pilot结果逐像素一致后，才与v44同排序贡献比较。只覆盖raw105 hood与raw130／137完整银ROI；该附加pilot诊断不扩大为49帧分层证据，也不扩展成全景因果结论。</p>{content}
<p class="note">各层RGB×T差值可相加重构输出差值；environment×T同时受环境RGB与透过率影响，未把二者拆开。该记录描述成像贡献变化，不证明训练伪影的根因。银残片留下一轮处理。</p>
<details><summary>展开三张实际来源图</summary>{''.join(images)}</details><p class="links"><a href="{prefix}/receipt.json">三次forward原始receipt</a> · <a href="code/source/diagnose_pilot_layers.py">delete状态的重放源码</a></p></section>'''


def review_section(review: dict | None, review_path: str | None, dataset: Dataset) -> str:
    is_pilot = dataset.mode == 'pilot'
    accepted = None if review is None else review.get('quality_accepted')
    quality = '尚未给出人工画质接受结论' if accepted is None else '未接受整体画质' if accepted is False else '画质接受状态：true（见独立视觉审查收据）'
    engine = '成对GPU渲染已完成；工程结论以执行收据为准' if review is None else str(review.get('engine_status', '工程结论见执行收据'))
    summary = '等待独立视觉审查记录；数值与工程检查不会自动填入 human verdict。' if review is None else str(review.get('summary', '视觉审查结论见原始收据。'))
    observations = [] if review is None else review.get('observations', [])
    observation_html = ''
    if observations:
        observation_html = '<ul class="review-list">' + ''.join('<li>' + esc(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)) + '</li>' for value in observations) + '</ul>'
    evidence = f'<p class="links"><a href="{esc(review_path)}">独立视觉审查原始JSON</a></p>' if review_path else ''
    identity = ''
    if review:
        identity = '<p class="note">审查角色：' + esc(review.get('reviewer', '见原始收据'))
        if 'agent_assessment' in review:
            identity += '；agent assessment：<code>' + esc(review['agent_assessment']) + '</code>'
        identity += '；human verdict：<code>' + esc(json.dumps(review.get('human_verdict'), ensure_ascii=False)) + '</code>。</p>'
        if 'continuation_decision' in review:
            identity += '<p>' + esc(review['continuation_decision']) + '</p>'
    stage = '本页只有 raw105／113／130／137 四帧 pilot，不是完整49帧。' if is_pilot else '本页展示 raw89–137 完整49帧真实GPU配对。'
    return f'''<div class="status"><p><strong>{stage}</strong><br>{esc(engine)}。<strong>{esc(quality)}</strong>。</p><p>{esc(summary)}</p></div>
<section id="review"><h2>视觉判定与执行边界</h2><p><strong>engine pass ≠ human 画质接受。</strong>资产冻结与 RGB 指标接近只描述车体输入及特定评价域，不等于全景无回退。视觉审查需覆盖银亮片、其他车辆、天空和非目标区域。</p>{identity}{observation_html}{evidence}
<p>RCR-1 原生10K与匹配道路1K／gates作为已保存组件接入；原11×9世界色场是 <strong>frozen-transfer，未重标定</strong>。本轮无 native／road／asset 训练、无PVD更新。HELD属于内部开发域，没有独立隐藏GT，也不作独立泛化声明。AutoDL 按用户要求保持开机；发布器不管理电源。</p></section>'''


def evidence_section(dataset: Dataset, pilot: Dataset | None) -> str:
    rows = []
    datasets = [('本页主结果', dataset)] + ([('先行pilot（另存）', pilot)] if pilot else [])
    for label, item in datasets:
        for variant in VARIANTS:
            receipt = item.receipts[variant]
            max_alpha = max(frame['states'][state]['group_alpha_conservation_maxabs']
                            for frame in item.frames[variant].values() for state in STATES)
            rows.append([esc(label), esc(variant), esc(receipt['mode']), count(len(item.raws)),
                         number(receipt['seconds'], 3), number(max_alpha, 9),
                         '<a href="' + item.link(variant + '/receipt.json') + '">' + esc(receipt['status']) + '</a>'])
    updates = table(['阶段', '版本', '模式', '帧', '实际秒数', 'α×T守恒最大误差', '原始receipt'], rows)
    current, candidate = (dataset.receipts[v] for v in VARIANTS)
    bindings = table(['冻结项', 'current', 'v44'], [
        ['canonical asset SHA256', '<code>' + esc(current['asset_sha256']) + '</code>', '<code>' + esc(candidate['asset_sha256']) + '</code>'],
        ['posterior SHA256', '<code>' + esc(current['posterior_sha256']) + '</code>', '<code>' + esc(candidate['posterior_sha256']) + '</code>'],
        ['persistent S/F SHA256', '<code>' + esc(current['persistent_SF_hash']) + '</code>', '<code>' + esc(candidate['persistent_SF_hash']) + '</code>'],
        ['rasterizer kernel SHA256', '<code>' + esc(current['kernel_sha256']) + '</code>', '<code>' + esc(candidate['kernel_sha256']) + '</code>']])
    return f'''<section id="evidence"><h2>工程收据与可复核文件</h2>{updates}
<p class="note">pilot 与正式结果分别记账。每个版本 receipt 的 neural／geometry／colour-fit／native-or-road training updates 均为0；这是已保存组件的配对渲染，不是新的训练实验。</p>{bindings}
<p class="links"><a href="{dataset.link('comparison.json')}">原始comparison（pooled＋逐帧）</a> · <a href="publication_manifest.json">交付清单与哈希</a> · <a href="code/report/publish_v44_results.py">本报告发布器</a> · <a href="cpu_stage.html">CPU阶段原报告与历史证据</a> · <a href="code/source/evaluate_v44.py">成对渲染入口</a> · <a href="code/contract/input_contract.json">输入契约</a></p>
<p class="note">最终HTML包只复制真实PNG、JSON指标／收据和报告代码，不复制大float NPZ。完整float与components保留在原结果目录，报告中的RGB分组数字来自原metrics，不由PNG反推。</p></section>'''


def historical_section(historical_reviews: list[str]) -> str:
    groups = []
    for raw in (113, 137):
        cards = []
        for kind, title in [('reference', '历史 RCR-1 采用前／旧 M2'), ('candidate', '历史 RCR-1 去蓝后／同一旧 M2')]:
            path = f'assets/RCR1_{kind}_{raw}.png'
            cards.append(f'<figure><figcaption>{esc(title)}</figcaption><a href="{path}"><img loading="lazy" src="{path}" width="480" height="320" alt="历史raw{raw} {esc(title)}"></a></figure>')
        groups.append(f'<h3>历史 raw{raw}</h3><div class="triple">' + ''.join(cards) + '</div>')
    review_links = ''.join(f'<p class="links"><a href="{esc(path)}">用户更新范围前的pilot暂缓审查（仅执行历史）</a></p>' for path in historical_reviews)
    return '''<section id="history"><h2>历史去蓝证据（与本次GPU配对分开）</h2><p>以下是旧M2的历史配对，只说明RCR-1组件的来源；不能冒充当前车体上的v44结果或复用其历史评价分母。历史银亮片诊断与先前暗团归因保留在CPU阶段报告。</p><details><summary>展开历史 raw113／137 图像</summary>''' + ''.join(groups) + '</details><p class="links"><a href="cpu_stage.html">完整CPU阶段报告／历史证据</a> · <a href="evidence/historical_rcr1_silver315_contributions.json">历史315点银亮片归因</a></p>' + review_links + '</section>'


def render_report(dataset: Dataset, pilot: Dataset | None, review: dict | None,
                  review_path: str | None, static_page: str, timestamp: str,
                  audits: list[tuple[dict, str]], layer_data: tuple[dict, str] | None,
                  historical_reviews: list[str]) -> str:
    style = (REPORT / 'v44_results.css').read_text(encoding='utf-8')
    script = (REPORT / 'v44_compare_controller.js').read_text(encoding='utf-8')
    pilot_link = '<a href="pilot.html">先行pilot独立页面</a>' if pilot else ''
    title = 'v44 · 四帧GPU pilot' if dataset.mode == 'pilot' else 'v44 · 49帧GPU配对'
    road_link = '<a href="#road-blue">道路去蓝</a>' if any(audit.get('observed_road_blue_diagnostics') and
                  audit['comparison_sha256'] == dataset.comparison_sha256 for audit, _ in audits) else ''
    body = (f'<div class="eyebrow">scene008 / v44 · 实际GPU结果 · 冻结当前C2／RGBA／SurfaceLOD车体</div>'
            f'<h1>{title}：当前车体与RCR-1背景接入</h1>'
            f'<nav><a href="#v44-compare">逐帧对照</a><a href="#review">视觉判定</a>{road_link}<a href="#pooled">指标与回退</a>'
            '<a href="#blue">蓝ROI</a><a href="#silver">银ROI与真实分组</a><a href="#evidence">收据</a>'
            f'<a href="cpu_stage.html">CPU阶段／历史</a>{pilot_link}</nav>'
            + review_section(review, review_path, dataset)
            + viewer(dataset, static_page)
            + '<section id="architecture"><h2>组件与数据流</h2><p>原始RGB／LiDAR／轨迹提供固定观测和评价域；当前车体与RCR-1背景在同一次native Gaussian排序中成像。删除只移除a37，所有其余层保留。</p><div class="scroll"><img class="architecture" src="architecture.svg" alt="原始观测、RCR-1已保存背景、冻结当前车体、统一深度排序与RGB输出"></div></section>'
            + observed_road_section(dataset, audits) + metric_sections(dataset) + blue_section(dataset) + supplementary_section(audits)
            + '<section id="silver"><h2>银色残片：留到下一轮</h2><p>按用户最新范围，本轮先解决蓝紫斑；银残片仍在，保留诊断，未宣称已修复。</p><details><summary>展开完整ROI统计与pilot分层证据（非本轮门槛）</summary>'
            + silver_section(dataset, has_layer_replay=layer_data is not None) + layer_section(layer_data) + '</details></section>'
            + evidence_section(dataset, pilot) + historical_section(historical_reviews)
            + f'<footer>生成时间：{esc(timestamp)}。真实GPU结果与历史图片分开保存；工程完成不代填人工画质接受。AutoDL保持开机。</footer>')
    return f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(title)}</title><style>{style}</style></head><body {MARKER}><main>{body}</main><script>{script}</script></body></html>\n'


def render_static(dataset: Dataset, return_page: str) -> str:
    sections = []
    for raw in dataset.raws:
        for state in STATES:
            cards = ''.join(card(dataset, raw, state, column, lazy=True) for column in ('current', 'v44', 'observed'))
            sections.append(f'<section class="static-frame" id="raw{raw}-{state}"><h2>raw{raw} / {state} a37</h2><div class="v44-grid" style="--v44-columns:3">{cards}</div><p class="links"><a href="{return_page}#v44-compare">返回主对照</a> · <a href="#top">回到帧导航</a></p></section>')
    navigation = ' · '.join(f'<a href="#raw{raw}-delete">raw{raw}</a>' for raw in dataset.raws)
    title = '四帧pilot静态对照（不是49帧）' if dataset.mode == 'pilot' else '49帧静态对照'
    style = (REPORT / 'v44_results.css').read_text(encoding='utf-8')
    return f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><style>{style}</style></head><body><main id="top"><h1>{title}</h1><p>current／v44／原始实拍使用同帧相机。原始实拍仍有a37，不是删车隐藏GT。所有图片为实际480×320 PNG；本页无需JavaScript。</p><nav>{navigation}</nav>' + ''.join(sections) + '</main></body></html>\n'


def preserve_previous_gpu_page(path: Path) -> None:
    if path.is_file() and MARKER in path.read_text(encoding='utf-8-sig'):
        copy_exact(path, path.with_name(f'{path.stem}_stage_{digest(path)[:12]}.html'))


def publish(args: argparse.Namespace) -> dict:
    output = args.output.resolve()
    results = args.results.resolve(strict=True)
    if output == results or output.is_relative_to(results) or results.is_relative_to(output):
        raise ValueError('--results and --output must be separate trees')
    crops = read_json(args.crop_manifest)
    dataset = load_dataset(results, 'results', crops)
    pilot = load_dataset(args.pilot, 'pilot', crops) if args.pilot else None
    if pilot and pilot.mode != 'pilot':
        raise ValueError('--pilot must point to an actual pilot result')
    review = read_json(args.visual_review) if args.visual_review else None
    if review is not None:
        validate_review(review, dataset)
    if args.pilot_visual_review and pilot is None:
        raise ValueError('--pilot-visual-review requires --pilot')
    pilot_review = read_json(args.pilot_visual_review) if args.pilot_visual_review else None
    if pilot_review is not None:
        validate_review(pilot_review, pilot)
    cpu_archive = seed_cpu_archive(output, args.cpu_stage_root.resolve(strict=True))
    copied = copy_dataset(dataset, output)
    if pilot:
        copied += copy_dataset(pilot, output)
    review_path = None
    if args.visual_review:
        review_path = f'evidence/visual_review_{digest(args.visual_review)[:12]}.json'
        copy_exact(args.visual_review, output / review_path)
        copied.append(dict(path=review_path, sha256=digest(args.visual_review), bytes=args.visual_review.stat().st_size))
    pilot_review_path = None
    if args.pilot_visual_review:
        pilot_review_path = f'evidence/pilot_visual_review_{digest(args.pilot_visual_review)[:12]}.json'
        copy_exact(args.pilot_visual_review, output / pilot_review_path)
        copied.append(dict(path=pilot_review_path, sha256=digest(args.pilot_visual_review), bytes=args.pilot_visual_review.stat().st_size))
    historical_reviews = []
    for history_file in args.historical_review:
        read_json(history_file)
        history_path = f'evidence/historical_review_{digest(history_file)[:12]}.json'
        copy_exact(history_file, output / history_path)
        historical_reviews.append(history_path)
        copied.append(dict(path=history_path, sha256=digest(history_file), bytes=history_file.stat().st_size))
    audits = []
    for audit_file in args.support_audit:
        audit = read_json(audit_file)
        matching = next((item for item in [dataset, pilot] if item is not None and
                         item.comparison_sha256 == audit.get('comparison_sha256')), None)
        if matching is None:
            raise ValueError(f'Support audit does not bind to a provided result: {audit_file}')
        for variant, expected in audit.get('source_sha256', {}).items():
            if variant not in VARIANTS or digest(matching.root / variant / 'receipt.json') != expected:
                raise ValueError(f'Support audit source receipt hash differs: {audit_file}/{variant}')
        audit_path = f'evidence/{audit_file.stem}_{digest(audit_file)[:12]}.json'
        copy_exact(audit_file, output / audit_path)
        copied.append(dict(path=audit_path, sha256=digest(audit_file), bytes=audit_file.stat().st_size))
        audits.append((audit, audit_path))
    layer_data = None
    if args.layer_results:
        layer_root = args.layer_results.resolve(strict=True)
        layer_receipt = read_json(layer_root / 'receipt.json')
        if layer_receipt.get('status') != 'PASS_THREE_FORWARD_SOURCE_DIAGNOSTICS':
            raise ValueError('Unverified supplemental layer replay')
        if layer_receipt['training_updates'] != 0 or layer_receipt['method_parameters_changed']:
            raise ValueError('Supplemental layer replay changed the method')
        for row in layer_receipt['rows']:
            if row['marker_parity_maxabs']['RGB'] != 0 or row['delta_additivity_maxabs'] >= 2e-5:
                raise ValueError('Supplemental layer replay does not match the saved result')
        layer_prefix = f'data/layers_{digest(layer_root / "receipt.json")[:12]}'
        for source in sorted(layer_root.rglob('*')):
            if source.is_file() and source.suffix.lower() in ('.png', '.json'):
                relative = f'{layer_prefix}/{source.relative_to(layer_root).as_posix()}'
                copy_exact(source, output / relative)
                copied.append(dict(path=relative, sha256=digest(source), bytes=source.stat().st_size))
        layer_data = (layer_receipt, layer_prefix)
    for filename in ('publish_v44_results.py', 'v44_compare_controller.js', 'v44_results.css', 'qa_v44_report.py'):
        source = REPORT / filename
        if source.exists():
            target = output / 'code/report' / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    for source in sorted((REPORT.parent / 'source').rglob('*.py')):
        if '__pycache__' not in source.parts:
            copy_exact(source, output / 'code/source' / source.relative_to(REPORT.parent / 'source'))
    crop_path = 'evidence/body_crop_manifest.json'
    copy_exact(args.crop_manifest, output / crop_path)
    timestamp = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    static_page = f'frames_static_{dataset.comparison_sha256[:12]}.html'
    report = render_report(dataset, pilot, review, review_path, static_page, timestamp, audits, layer_data, historical_reviews)
    write_text(output / static_page, render_static(dataset, 'index.html'))
    write_json(output / dataset.prefix / 'viewer_manifest.json', dataset.manifest)
    if pilot:
        pilot_static = f'pilot_static_{pilot.comparison_sha256[:12]}.html'
        pilot_audits = [(audit, path) for audit, path in audits if audit['comparison_sha256'] == pilot.comparison_sha256]
        pilot_report = render_report(pilot, None, pilot_review, pilot_review_path, pilot_static, timestamp, pilot_audits, layer_data, historical_reviews)
        write_text(output / pilot_static, render_static(pilot, 'pilot.html'))
        write_json(output / pilot.prefix / 'viewer_manifest.json', pilot.manifest)
        preserve_previous_gpu_page(output / 'pilot.html')
        write_text(output / 'pilot.html', pilot_report)
    preserve_previous_gpu_page(output / 'index.html')
    write_text(output / 'index.html', report)
    manifest = dict(schema='v44-results-publication-v1', generated_at=timestamp,
                    mode=dataset.mode, raw_ids=dataset.raws, main_comparison_sha256=dataset.comparison_sha256,
                    pilot_comparison_sha256=pilot.comparison_sha256 if pilot else None,
                    cpu_stage=cpu_archive, crop_manifest=dict(path=crop_path, sha256=digest(args.crop_manifest)),
                    visual_review=review_path, quality_accepted=None if review is None else review.get('quality_accepted'),
                    pilot_visual_review=pilot_review_path,
                    historical_reviews=historical_reviews,
                    support_audits=[path for _, path in audits],
                    supplemental_layer_replay=None if layer_data is None else layer_data[1] + '/receipt.json',
                    independent_hidden_GT=False, HELD_role='internal_development',
                    road_field_mode='frozen_transfer_from_RCR1_M2_not_recalibrated',
                    engine_pass_is_human_acceptance=False, float_NPZ_copied=False,
                    full49_executed=dataset.mode == 'full49', shutdown_performed=False,
                    files=copied)
    write_json(output / 'publication_manifest.json', manifest)
    readme = f'''# v44 实际GPU配对结果

主结果模式：{dataset.mode}。帧：{', '.join(map(str, dataset.raws))}。
{'本页只有四帧pilot，完整49帧未执行。' if dataset.mode == 'pilot' else '本页为完整49帧实际渲染。'}

打开 index.html；初始raw113真实PNG无需JavaScript可见。拖动、双状态、共用车体裁剪、蓝/银ROI与可选原始实拍列使用本地相对PNG，不fetch，不依赖外部资源。

![组件图](architecture.svg)

engine pass 不等于 human 画质接受。当前车体C2/固定RGBA/SurfaceLOD冻结；RCR-1已保存native10K与匹配road1K/gates接入。11×9色场仅frozen-transfer，未重新标定。HELD是内部开发域，没有独立隐藏删车GT。主运行current没有同口径actual group RGB；仅在另存的真实分层重放核对范围内比较净层差。银残片留下一轮，本轮先解决蓝紫斑。

CPU原报告保存在cpu_stage.html，历史图像及证据保留。本包只带PNG、JSON和代码；大floatNPZ及components留在原结果目录。AutoDL按用户要求保持开机。
'''
    write_text(output / 'README.md', readme)
    return dict(index=str(output / 'index.html'), mode=dataset.mode, raw_ids=dataset.raws,
                copied_PNG_JSON=len(copied), cpu_stage_sha256=cpu_archive['sha256'],
                quality_accepted=manifest['quality_accepted'], full49_executed=manifest['full49_executed'])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True, help='Local retrieved actual current/v44 result directory')
    parser.add_argument('--pilot', type=Path, help='Optional separate pilot directory alongside a full49 main result')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--visual-review', type=Path, help='Independent review JSON; absent means no human verdict')
    parser.add_argument('--pilot-visual-review', type=Path, help='Separate pilot review; never reused as the main full49 verdict')
    parser.add_argument('--historical-review', type=Path, action='append', default=[], help='Original superseded pilot review, archived as history only')
    parser.add_argument('--support-audit', type=Path, action='append', default=[], help='Independent audit JSON; repeat for pilot and full49')
    parser.add_argument('--layer-results', type=Path, help='Optional verified three-forward supplemental layer replay')
    parser.add_argument('--crop-manifest', type=Path, default=DEFAULT_CROPS)
    parser.add_argument('--cpu-stage-root', type=Path, default=DEFAULT_CPU)
    args = parser.parse_args()
    print(json.dumps(publish(args), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
