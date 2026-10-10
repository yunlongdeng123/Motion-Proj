#!/usr/bin/env python3
"""从真实 DGGT 参考修复诊断产物构建逐帧只读审核页。"""

from __future__ import annotations

import argparse
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
from urllib.parse import unquote, urlsplit


REPO = Path(__file__).resolve().parents[2]
DEFAULT_RUN = Path('/root/autodl-tmp/runs/worldsim_v81/WS-V81-DGGT-WAYMO-INFERENCE-20261011/r1')
DEFAULT_OUTPUT = REPO / 'outputs/v81-dggt-waymo/reference_refinement'
DIAGNOSTIC = Path('diagnostics/reference_refinement_r1')
BRANCHES = ('noop', 'delete')
CONDITIONS = ('official_public_single', 'official_mv_self', 'official_mv_legal_ref')
LABELS = {
    'noop': '不编辑对照',
    'delete': '删除目标车',
    'official_public_single': '公开单图重放',
    'official_mv_self': '官方双图 · 自参考',
    'official_mv_legal_ref': '官方双图 · 原 RGB 参考',
}
STYLE = """
:root{color-scheme:dark;font-family:system-ui,-apple-system,'Segoe UI',sans-serif;background:#0b111a;color:#ecf1f7}
*{box-sizing:border-box}body{margin:0;line-height:1.55}main{max-width:1660px;margin:auto;padding:30px 24px 80px}
h1,h2,h3{line-height:1.22}h1{font-size:clamp(29px,3vw,45px);margin:8px 0 12px}h2{font-size:25px;margin:42px 0 14px}
p{color:#b7c5d5;max-width:1000px}a{color:#82d6f7}a:hover{color:white}.eyebrow{color:#79cfc0;letter-spacing:.14em;font-size:12px;font-weight:750;text-transform:uppercase}
.hero{background:linear-gradient(130deg,#142838,#101923 62%,#1b2833);border:1px solid #29404f;border-radius:20px;padding:28px 30px}
.badge{display:inline-block;border-radius:999px;padding:5px 11px;background:#173b34;color:#8aead4;font-size:13px;font-weight:700}
.badge.pending{background:#3a3020;color:#ffd185}.meta{display:flex;flex-wrap:wrap;gap:12px;margin:22px 0}
.card{flex:1 1 215px;border:1px solid #30404e;background:#15212d;border-radius:13px;padding:15px 17px;min-width:0}
.card small{display:block;color:#8fa2b5;margin-bottom:7px}.card strong{display:block;font-size:18px;overflow-wrap:anywhere}
.flow{display:flex;gap:8px;align-items:stretch;flex-wrap:wrap;margin:25px 0}.module{background:#152838;border:1px solid #376078;border-radius:12px;padding:13px 17px;min-width:150px;flex:1;color:#e2edf6;text-align:center}.arrow{align-self:center;color:#73c5d1;font-size:24px;font-weight:700}
.notice{border-left:4px solid #e7af61;background:#30271e;padding:13px 17px;border-radius:7px;color:#f5dcc1;max-width:1100px}
.section-lead{max-width:1150px}.matrix-wrap{overflow-x:auto;border:1px solid #2d3c4b;border-radius:14px;background:#101a25}
table{border-collapse:collapse;width:100%}.matrix{min-width:1180px;table-layout:fixed}.matrix th,.matrix td{border-bottom:1px solid #283746;border-right:1px solid #283746;padding:10px;vertical-align:top}.matrix th{background:#1a2b39;text-align:left;color:#d6e7f2}.matrix th:first-child{width:190px}.matrix td img{display:block;width:100%;height:auto;border-radius:6px;background:#0a1118}.matrix td a{display:block}.matrix .rowlabel{font-weight:700;color:#dbe9f4}.matrix .rowlabel span{display:block;font-weight:400;font-size:12px;color:#91a8b9;margin-top:5px}
.data{max-width:100%;overflow-x:auto;border:1px solid #2d3c4b;border-radius:12px}.data th,.data td{padding:10px 12px;text-align:left;border-bottom:1px solid #2b3c4b;white-space:nowrap}.data th{background:#1a2b39}.data td{color:#c8d6e3}
details{border:1px solid #30404e;background:#111d29;border-radius:12px;margin:12px 0;padding:12px 16px}summary{cursor:pointer;font-weight:700;color:#cfe6f3}pre{overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;color:#b9ccdb;font-size:12px}
.footer{margin-top:40px;border-top:1px solid #31404d;padding-top:20px;color:#8ca0b1;font-size:13px}
@media(max-width:700px){main{padding:18px 12px 55px}.hero{padding:20px}.arrow{display:none}.module{min-width:130px}.matrix{min-width:950px}}
"""


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'JSON 根节点必须是对象: {path}')
    return value


def require_file(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f'缺少非空媒体: {path}')
    return path


def source_media(run_dir: Path, diagnostic: Path) -> dict[str, Path]:
    """在复制前解码44张原 RGB、旧媒体与本次输出。"""
    from PIL import Image

    sources = {}
    for index in range(4):
        frame = f'{index:03d}'
        sources[f'assets/input/{frame}_0.jpg'] = require_file(
            run_dir / f'processed/validation/128/images/{frame}_0.jpg')
        for branch in BRANCHES:
            sources[f'assets/raw/{branch}/{frame}.png'] = require_file(
                run_dir / f'gaussian_edits/{branch}/{frame}.png')
            sources[f'assets/prior/{branch}/{frame}.png'] = require_file(
                run_dir / f'difix_refined/{branch}/{frame}.png')
            for condition in CONDITIONS:
                sources[f'assets/{condition}/{branch}/{frame}.png'] = require_file(
                    diagnostic / condition / branch / f'{frame}.png')
    decoded = 0
    for path in sources.values():
        try:
            with Image.open(path) as image:
                if image.width <= 0 or image.height <= 0 or not {'R', 'G', 'B'}.issubset(image.getbands()):
                    raise ValueError(f'图像必须含RGB且尺寸非零: {path}')
                image.load()
        except Exception as exc:
            raise ValueError(f'原图解码失败: {path}') from exc
        decoded += 1
    if len(sources) != 44 or decoded != 44:
        raise AssertionError(f'预期44张图，实际文件{len(sources)}、解码{decoded}')
    return sources


def validate_completed(run: dict, run_dir: Path, diagnostic: Path) -> None:
    if run.get('status') != 'completed':
        raise ValueError('只有 completed 诊断能构建结果矩阵')
    if run.get('output_frames') != 24:
        raise ValueError('本次新诊断必须实际记录24张输出')
    if run.get('baseline_replay_verified') is not True:
        raise ValueError('公开单图重放尚未逐像素验证一致')
    parameters = run.get('parameters', {})
    if (parameters.get('timestep') != 199 or parameters.get('frames_per_condition') != 8
            or parameters.get('max_output_frames') != 24):
        raise ValueError('本次诊断参数与固定四帧×两分支×三条件合同不符')
    if not isinstance(run.get('source', {}).get('checkpoint'), str):
        raise ValueError('缺少实际 checkpoint 路径')
    for field, expected in (
        ('edits', run_dir / 'gaussian_edits'),
        ('prior', run_dir / 'difix_refined'),
        ('legal_reference_rgb', run_dir / 'processed/validation/128/images/000_0.jpg'),
    ):
        actual = run.get('source', {}).get(field)
        if not isinstance(actual, str) or Path(actual).resolve() != expected.resolve():
            raise ValueError(f'诊断来源与固定run不符: {field}')
    if not isinstance(run.get('compatibility'), dict) or not isinstance(run.get('weight_checks'), dict):
        raise ValueError('缺少真实兼容或权重记录')
    for name in CONDITIONS:
        if name not in run.get('conditions', {}):
            raise ValueError(f'缺少真实条件记录: {name}')
    for branch in BRANCHES:
        for field in ('baseline_replay', 'paired_checks'):
            rows = run.get(field, {}).get(branch)
            if not isinstance(rows, list) or [row.get('index') for row in rows] != list(range(4)):
                raise ValueError(f'{field}/{branch} 缺少逐帧真实记录')
        if any(row.get('same_png_pixels') is not True for row in run['baseline_replay'][branch]):
            raise ValueError(f'{branch} 单图重放中有帧与先前结果不一致')
        for index, row in enumerate(run['paired_checks'][branch]):
            keys = ('same_batch_shape', 'view0_vae_latent_torch_equal',
                    'text_embeddings_torch_equal', 'timestep_torch_equal')
            if any(row.get(key) is not True for key in keys):
                raise ValueError(f'{branch}/{index} 双图输入配对未通过')
            trace = diagnostic / 'paired_inputs' / branch / f'{index:03d}.pt'
            recorded = row.get('paired_unet_inputs')
            if not isinstance(recorded, str) or Path(recorded).resolve() != trace.resolve():
                raise ValueError(f'{branch}/{index} trace来源路径不符')
            require_file(trace)


def copy_evidence(run_path: Path | None, review_path: Path | None,
                  sources: dict[str, Path], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if run_path is not None:
        shutil.copy2(run_path, output / 'run.json')
    if review_path is not None:
        shutil.copy2(review_path, output / 'assistant_review.json')
    for relative, source in sources.items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def flow() -> str:
    modules = ('原四帧 RGB / 原生渲染', '公开单图与双图参考修复', '逐帧 noop / delete 对照', '背景与重新引车审核')
    return '<div class="flow" aria-label="组件与数据流">' + '<span class="arrow" aria-hidden="true">→</span>'.join(
        f'<div class="module">{esc(label)}</div>' for label in modules) + '</div>'


def matrix(branch: str) -> str:
    rows = [
        ('input', '原始输入 RGB', '相同四张前视目标'),
        ('raw', '原生高斯渲染', '修复之前'),
        ('prior', '原先单图 Difix', '此前已保存的对照'),
        ('official_public_single', LABELS['official_public_single'], '本次公开单图重放'),
        ('official_mv_self', LABELS['official_mv_self'], '本次双图路径；参考为自身渲染'),
        ('official_mv_legal_ref', LABELS['official_mv_legal_ref'], '本次双图路径；参考为原 000 RGB'),
    ]
    head = '<tr><th scope="col">来源 / 条件</th>' + ''.join(
        f'<th scope="col">第 {i} 帧</th>' for i in range(4)) + '</tr>'
    body = []
    for key, title, sub in rows:
        cells = []
        for index in range(4):
            frame = f'{index:03d}'
            if key == 'input':
                link = f'assets/input/{frame}_0.jpg'
            else:
                link = f'assets/{key}/{branch}/{frame}.png'
            cells.append(f'<td><a href="{link}" aria-label="{esc(title)} 第{index}帧原图">'
                         f'<img src="{link}" loading="lazy" alt="{esc(title)} 第{index}帧"></a></td>')
        body.append(f'<tr><th scope="row" class="rowlabel">{esc(title)}<span>{esc(sub)}</span></th>'
                    + ''.join(cells) + '</tr>')
    return (f'<section><h2>{esc(LABELS[branch])}</h2>'
            '<p class="section-lead">四帧按相同顺序排列；点击图片可查看原尺寸 PNG。原始输入行只用于视觉参照。</p>'
            f'<div class="matrix-wrap"><table class="matrix">{head}{"".join(body)}</table></div></section>')


def metrics(run: dict) -> str:
    baseline_rows = [row for branch in BRANCHES for row in run['baseline_replay'][branch]]
    paired_rows = [(branch, row) for branch in BRANCHES for row in run['paired_checks'][branch]]
    equal_count = sum(row.get('same_png_pixels') is True for row in baseline_rows)
    max_abs = max(float(row['max_abs_u8']) for row in baseline_rows)
    paired_ok = sum(all(row.get(key) is True for key in (
        'same_batch_shape', 'view0_vae_latent_torch_equal',
        'text_embeddings_torch_equal', 'timestep_torch_equal')) for _, row in paired_rows)
    batch = paired_rows[0][1].get('batch_shape', [])
    batch_text = f'{batch[0]} 图' if isinstance(batch, list) and batch else '未记录'
    cards = [
        ('权重文件', Path(run['source']['checkpoint']).name),
        ('修复条件', f"t={run['parameters']['timestep']} · {run['parameters'].get('precision', '未记录')}"),
        ('双图入口批次', batch_text),
        ('单图重放与旧图完全相同', f'{equal_count}/8 · 最大灰阶差 {max_abs:g}'),
        ('双图配对输入核对', f'{paired_ok}/8 帧字段一致'),
    ]
    card_html = ''.join(f'<div class="card"><small>{esc(name)}</small><strong>{esc(value)}</strong></div>'
                        for name, value in cards)
    table_rows = []
    for branch, row in paired_rows:
        diff = row['self_vs_legal']
        mask = row['vehicle_mask_similarity']
        table_rows.append('<tr>'
                          f'<td>{esc(LABELS[branch])} / {row["index"]}</td>'
                          f'<td>{esc(diff["mean_abs_u8"])} / {esc(diff["max_abs_u8"])}</td>'
                          f'<td>{esc(mask["self_mask_mae_to_raw_noop_u8"])} / '
                          f'{esc(mask["legal_mask_mae_to_raw_noop_u8"])}</td>'
                          f'<td>{"是" if all(row.get(key) is True for key in ("same_batch_shape", "view0_vae_latent_torch_equal", "text_embeddings_torch_equal", "timestep_torch_equal")) else "否"}</td>'
                          '</tr>')
    table = ('<div class="data"><table><thead><tr><th>分支 / 帧</th><th>自参考 vs 原 RGB：平均 / 最大灰阶差</th>'
             '<th>目标车区域对原 noop：自参考 / 原 RGB 的 MAE</th><th>输入配对一致</th></tr></thead><tbody>'
             + ''.join(table_rows) + '</tbody></table></div>')
    raw_details = {'compatibility': run['compatibility'], 'weight_checks': run['weight_checks'],
                   'baseline_replay': run['baseline_replay'], 'paired_checks': run['paired_checks']}
    return ('<h2>实测记录</h2><div class="meta">' + card_html + '</div>'
            '<p class="notice">原始 000 参考图含目标车。目标区域 MAE 只说明像素差，不能据此判定车辆身份、是否重新引车或背景是否正确。</p>'
            '<p>双图自参考与原 RGB 参考使用同一双图入口；上表保留每帧的实际配对结果。'
            '单图重放用于核对既有结果，不把不同网络路径间的差异直接解释为参考图收益。</p>'
            + table
            + '<details><summary>兼容适配、权重和逐帧原始数值</summary><pre>'
            + esc(json.dumps(raw_details, ensure_ascii=False, indent=2)) + '</pre></details>'
            + '<details><summary>参数与来源路径</summary><pre>'
            + esc(json.dumps({'source': run['source'], 'parameters': run['parameters'],
                              'conditions': run['conditions']}, ensure_ascii=False, indent=2))
            + '</pre></details>')


def review_note(run: dict | None, review: dict | None) -> str:
    verdict = (review['human_verdict'] if review is not None and 'human_verdict' in review
               else None if run is None else run.get('human_verdict'))
    if review is None:
        status = '待助手审核；未发现本次诊断的 assistant_review.json。'
        attachment = ''
    else:
        status = '已附本次诊断的真实助手审核记录。'
        attachment = ('<p><a href="assistant_review.json">查看 assistant_review.json</a></p>'
                      '<details><summary>审核记录原文</summary><pre>'
                      + esc(json.dumps(review, ensure_ascii=False, indent=2)) + '</pre></details>')
    return (f'<h2>审核状态</h2><p>{status} 人类判定：<strong>{esc(json.dumps(verdict, ensure_ascii=False))}</strong>。</p>'
            + attachment)


def page(run: dict | None, review: dict | None) -> str:
    complete = run is not None and run.get('status') == 'completed'
    status = '已生成真实诊断对照' if complete else '待诊断完成 · pending'
    badge = 'badge' if complete else 'badge pending'
    source_status = run.get('status', 'run.json 尚未生成') if run else 'run.json 尚未生成'
    if complete:
        content = (metrics(run) + ''.join(matrix(branch) for branch in BRANCHES)
                   + review_note(run, review))
    else:
        evidence_link = '<a href="run.json">查看当前 run.json</a>' if run is not None else '尚无 run.json'
        content = ('<h2>尚无完整结果</h2><p class="notice">本页仅记录待运行状态；不会把已有高斯编辑或旧 Difix 当作本次参考诊断输出。</p>'
                   f'<p>实际状态：<strong>{esc(source_status)}</strong>。{evidence_link}</p>'
                   + review_note(run, review))
    footer_evidence = '<a href="run.json">run.json</a>' if run is not None else 'run.json 待生成'
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DGGT 参考修复逐帧审核</title><style>{STYLE}</style></head>
<body><main><header class="hero"><div class="eyebrow">DGGT · WAYMO SCENE 128</div>
<h1>参考修复逐帧审核</h1><span class="{badge}">{esc(status)}</span>
<p>固定原四张前视目标图，比较不编辑与删除分支中的公开单图、自参考双图及合法原 RGB 参考双图。所有判断应回到逐帧画面。</p>
{flow()}</header>{content}
<div class="footer">证据：{footer_evidence}；本页只整理已保存文件，不代填人工结论。</div>
</main></body></html>'''


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if name in ('src', 'href') and value:
                self.paths.append(value)


def check_links(output: Path) -> dict:
    parser = Links()
    parser.feed((output / 'index.html').read_text(encoding='utf-8'))
    checked, missing = [], []
    for link in parser.paths:
        parsed = urlsplit(link)
        if parsed.scheme or parsed.netloc or link.startswith('#'):
            continue
        relative = Path(unquote(parsed.path))
        if relative.is_absolute() or '..' in relative.parts:
            missing.append(link)
            continue
        checked.append(link)
        if not (output / relative).is_file():
            missing.append(link)
    return {'status': 'pass' if not missing else 'fail', 'links_checked': len(checked),
            'unique_local_files': len(set(checked)), 'missing_links': missing,
            'videos_encoded': 0}


def build(run_dir: Path, output: Path) -> dict:
    diagnostic = run_dir / DIAGNOSTIC
    run_path = diagnostic / 'run.json'
    review_path = diagnostic / 'assistant_review.json'
    run = load_json(run_path) if run_path.is_file() else None
    review = load_json(review_path) if review_path.is_file() else None
    complete = run is not None and run.get('status') == 'completed'
    sources: dict[str, Path] = {}
    if complete:
        validate_completed(run, run_dir, diagnostic)
        sources = source_media(run_dir, diagnostic)
    copy_evidence(run_path if run is not None else None,
                  review_path if review is not None else None, sources, output)
    (output / 'index.html').write_text(page(run, review), encoding='utf-8')
    audit = check_links(output)
    audit.update(page_status='completed' if complete else 'pending',
                 new_diagnostic_png_count=sum(key.startswith('assets/official_') for key in sources),
                 decoded_rgb_images=sum(key.startswith('assets/') for key in sources),
                 new_diagnostic_png_decoded=sum(key.startswith('assets/official_') for key in sources),
                 paired_trace_files_verified=8 if complete else 0,
                 source_run_json=str(run_path) if run is not None else None,
                 source_assistant_review_json=str(review_path) if review is not None else None)
    (output / 'delivery_validation.json').write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if audit['status'] != 'pass':
        raise RuntimeError(f'页面本地链接缺失: {audit["missing_links"]}')
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.run_dir, args.output_dir), ensure_ascii=False))


if __name__ == '__main__':
    main()
