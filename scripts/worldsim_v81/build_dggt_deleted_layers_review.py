#!/usr/bin/env python3
"""Build a read-only frame-000 DGGT layer review from saved PNG evidence.

The probe owns the renders. This builder decodes the 25 PNGs, optionally copies
already-existing context images, and writes only HTML plus delivery validation.
"""

from __future__ import annotations

import argparse
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
from urllib.parse import unquote, urlsplit


RUN_DEFAULT = Path('/root/autodl-tmp/runs/worldsim_v81/WS-V81-DGGT-WAYMO-INFERENCE-20261011/r1')
DIAGNOSTIC = Path('diagnostics/deleted_layers_frame000_r1')
LAYERS = ('static_before', 'static_delete', 'dynamic_before', 'dynamic_delete', 'joint_delete')
KINDS = ('black', 'white', 'alpha', 'expected_depth', 'depth_valid')
LABELS = {
    'static_before': '静态层 · 删除前',
    'static_delete': '静态层 · 删除后',
    'dynamic_before': '动态层 · 删除前',
    'dynamic_delete': '动态层 · 删除后',
    'joint_delete': '静态+动态 · 联合删除',
}
KIND_LABELS = {
    'black': '黑底累计颜色 G',
    'white': '白底合成 G+(1−A)',
    'alpha': '累积透明度 A',
    'expected_depth': '预测期望深度（显示图）',
    'depth_valid': '有效深度区域',
}


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def load_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'JSON 根节点必须为对象: {path}')
    return value


def image_path(layer: str, kind: str) -> str:
    return f'assets/{layer}/{kind}.png'


def decode_png(path: Path) -> tuple[int, int]:
    from PIL import Image

    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f'缺少非空PNG: {path}')
    try:
        with Image.open(path) as image:
            if image.format != 'PNG' or image.width <= 0 or image.height <= 0:
                raise ValueError(f'不是有效的非空PNG: {path}')
            image.load()
            return image.size
    except Exception as exc:
        raise ValueError(f'PNG实际解码失败: {path}') from exc


def validate_assets(output: Path) -> dict:
    sizes = {}
    for layer in LAYERS:
        for kind in KINDS:
            relative = image_path(layer, kind)
            sizes[relative] = decode_png(output / relative)
    if len(set(sizes.values())) != 1:
        raise ValueError(f'固定帧五层图像尺寸不一致: {sizes}')
    return {'decoded_png': len(sizes), 'image_size': list(next(iter(sizes.values())))}


def optional_references(run_root: Path, output: Path) -> dict[str, str]:
    """Only show images that exist; never synthesize a missing input or sky."""
    alpha_pair = run_root / 'diagnostics/alpha_compositing_pair_r1'
    candidates = {
        'input': (
            output / 'sources/input.png',
            run_root / 'gaussian_edits/input_rgb/000.png',
            alpha_pair / 'assets/input/000.png',
        ),
        'official_delete': (
            output / 'sources/official_delete.png',
            run_root / 'gaussian_edits/delete/000.png',
            alpha_pair / 'assets/official/delete/000.png',
        ),
        'sky': (
            output / 'sources/sky.png',
            alpha_pair / 'assets/sky/000.png',
        ),
    }
    references = {}
    for name, sources in candidates.items():
        source = next((path for path in sources if path.is_file()), None)
        if source is None:
            continue
        decode_png(source)
        target = output / 'assets/reference' / f'{name}.png'
        if source.resolve() != target.resolve():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        decode_png(target)
        references[name] = target.relative_to(output).as_posix()
    return references


def image_card(layer: str, kind: str, *, heading: str | None = None) -> str:
    label = heading or f'{LABELS[layer]} · {KIND_LABELS[kind]}'
    path = image_path(layer, kind)
    return (f'<figure><img src="{esc(path)}" alt="{esc(label)}" loading="lazy">'
            f'<figcaption>{esc(label)}</figcaption></figure>')


def reference_card(label: str, path: str) -> str:
    return (f'<figure><img src="{esc(path)}" alt="{esc(label)}" loading="lazy">'
            f'<figcaption>{esc(label)}</figcaption></figure>')


def light_run_stats(run: dict) -> dict:
    """Expose small recorded summaries, never tensors or entire run JSON in HTML."""
    result = {key: run[key] for key in ('status', 'task', 'scene', 'frame', 'frame_index',
                                       'render_calls', 'render_mode', 'seconds',
                                       'peak_cuda_allocated_bytes', 'peak_cpu_rss_kib') if key in run}
    for key in ('calibration', 'depth_display', 'layer_statistics', 'render_seconds',
                'interpretation', 'stats', 'statistics', 'metrics', 'resource'):
        if key in run:
            value = run[key]
            if len(json.dumps(value, ensure_ascii=False, default=str)) <= 18000:
                result[key] = value
    return result


def render_page(run: dict, review: dict | None, references: dict[str, str],
                alpha_pair_exists: bool) -> str:
    review_text = (review or {}).get('summary_zh') or (review or {}).get('summary')
    if not review_text:
        review_text = '独立助手审核尚未提供；本页只展示真实层输出，不代填人工结论。'
    reference_labels = {'input': '输入 RGB', 'official_delete': '既有官方删除图', 'sky': '既有模型 sky 参考'}
    reference_html = ''.join(reference_card(reference_labels[name], path)
                             for name, path in references.items())
    reference_section = (f'<section class="panel"><h2>已有来源参照</h2><div class="grid three">'
                         f'{reference_html}</div><p>仅展示找到并实际解码的原图；缺失来源不补造。</p></section>'
                         if references else '')
    pair_link = ('<a href="../alpha_compositing_pair_r1/index.html">查看既有 alpha 合成配对页</a>'
                 if alpha_pair_exists else '')
    run_stats = esc(json.dumps(light_run_stats(run), ensure_ascii=False, indent=2, default=str))
    review_link = '<a href="assistant_review.json">助手审核 JSON</a>' if review is not None else ''
    details = ''.join(
        f'<section class="panel"><h3>{esc(LABELS[layer])} · 其余通道</h3>'
        '<div class="grid four">'
        + ''.join(image_card(layer, kind) for kind in KINDS if kind != 'black')
        + '</div></section>'
        for layer in LAYERS if layer != 'joint_delete'
    )
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DGGT · 删除层拆分 · frame 000</title><style>
:root{{color-scheme:dark;font-family:system-ui,-apple-system,"Segoe UI",sans-serif;background:#0b1119;color:#edf3f7}}
*{{box-sizing:border-box}}body{{margin:0;line-height:1.55}}main{{max-width:1580px;margin:auto;padding:24px 24px 72px}}
h1,h2,h3{{line-height:1.2}}h1{{font-size:clamp(30px,3vw,46px);margin:5px 0 8px}}h2{{font-size:24px;margin:0 0 13px}}h3{{font-size:19px;margin:0 0 12px}}
p{{color:#b7c7d4;max-width:1150px;margin:9px 0 13px}}a{{color:#90d8f5}}.eyebrow{{color:#7ce2cf;font-size:12px;font-weight:750;letter-spacing:.13em;text-transform:uppercase}}
.hero,.panel{{border:1px solid #2c4051;background:#121e2b;border-radius:16px;padding:20px;margin:0 0 16px}}
.hero{{background:linear-gradient(125deg,#163144,#111c29 65%,#253244)}}.badge{{display:inline-block;background:#164337;color:#a1f0d8;border-radius:999px;padding:5px 11px;font-size:13px;font-weight:700}}
.flow{{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-top:15px}}.module{{padding:9px 12px;border:1px solid #3a6275;background:#152a3b;border-radius:9px;color:#dcecf4}}.arrow{{font-size:21px;color:#83cada}}
.notice{{border-left:4px solid #e9b36e;background:#352b21;border-radius:7px;padding:12px 15px;color:#f1d9bb;margin:14px 0}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:13px}}.grid.three{{grid-template-columns:repeat(3,minmax(0,1fr))}}.grid.four{{grid-template-columns:repeat(4,minmax(0,1fr))}}.grid.five{{grid-template-columns:repeat(5,minmax(0,1fr))}}
figure{{margin:0;min-width:0}}img{{display:block;width:100%;height:auto;background:#080d13;border-radius:8px;border:1px solid #263849}}figcaption{{font-size:13px;color:#bfd4e1;padding:7px 1px 2px}}
.subhead{{color:#99acbc;font-size:14px;margin:0 0 9px}}details{{border:1px solid #2d4254;background:#101a26;border-radius:11px;padding:12px 15px;margin-top:16px}}summary{{cursor:pointer;color:#d5e8f1;font-weight:700}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;color:#beced9}}
.links{{display:flex;gap:18px;flex-wrap:wrap;margin-top:10px}}@media(max-width:900px){{.grid.four,.grid.five{{grid-template-columns:repeat(2,minmax(0,1fr))}}.grid.three{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}@media(max-width:590px){{main{{padding:14px}}.grid,.grid.three,.grid.four,.grid.five{{grid-template-columns:1fr}}}}
</style></head><body><main>
<header class="hero"><div class="eyebrow">DGGT / 同一高斯、相机与 RGB 选择 / frame 000</div>
<h1>删除前后：静态层与动态层</h1><span class="badge">固定首帧 · 五层真实输出</span>
<p>把原有高斯按静态、动态和联合删除拆开显示，帮助定位黑洞在各层的可见形态。所有面板沿用同一份已保存高斯、预测相机与实例选择。</p>
<div class="flow"><span class="module">既有高斯 / 相机 / 选择</span><span class="arrow">→</span><span class="module">静态与动态层</span><span class="arrow">→</span><span class="module">五组保存 PNG</span><span class="arrow">→</span><span class="module">配对观察与审核</span></div>
<div class="notice"><strong>读图边界：</strong>black 是累计颜色 G；white = G+(1−A)。白底变亮不等于道路恢复；单独的静态或动态层不等于联合渲染中的真实贡献。深度来自 DGGT 预测，非 GT，也不能按显示灰度当作米尺度。</div></header>
<section class="panel"><h2>静态层 · 删除前 / 删除后</h2><p class="subhead">同一首帧、同一视角；两图均为黑底累计颜色 G。</p><div class="grid">{image_card('static_before', 'black')}{image_card('static_delete', 'black')}</div></section>
<section class="panel"><h2>动态层 · 删除前 / 删除后</h2><p class="subhead">单独层只用于观察，不能直接读成联合结果里的贡献。</p><div class="grid">{image_card('dynamic_before', 'black')}{image_card('dynamic_delete', 'black')}</div></section>
<section class="panel"><h2>联合删除 · 五种视图</h2><div class="grid five">{''.join(image_card('joint_delete', kind) for kind in KINDS)}</div><p>expected_depth 是预测期望深度的显示图；depth_valid 标出有效区域。它们不是道路标签或真实米制测距。</p></section>
{details}{reference_section}
<section class="panel"><h2>助手审核</h2><p>{esc(review_text)}</p><p>人工 verdict 仅由用户或指定评审填写。</p></section>
<details><summary>真实执行的轻量统计与证据入口</summary><pre>{run_stats}</pre><div class="links"><a href="run.json">诊断 run.json</a>{review_link}{pair_link}</div></details>
<p>页面经本地文件存在性与 PIL 解码核验；未做浏览器视觉 QA。未嵌入 float PT，未编视频。</p>
</main></body></html>'''


class Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.paths: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if name in ('src', 'href') and value:
                self.paths.append(value)


def validate_links(output: Path, page: str) -> dict:
    parser = Links()
    parser.feed(page)
    checked, missing = [], []
    for link in parser.paths:
        parsed = urlsplit(link)
        if parsed.scheme or parsed.netloc or link.startswith('#'):
            continue
        relative = Path(unquote(parsed.path))
        target = (output / relative).resolve()
        if not target.is_relative_to(output.parent.resolve()) or not target.is_file():
            missing.append(link)
        else:
            checked.append(link)
    return {'links_checked': len(checked), 'unique_local_files': len(set(checked)),
            'missing_links': missing}


def build(run_root: Path, output: Path) -> dict:
    run = load_object(output / 'run.json')
    if run.get('status') != 'complete':
        raise ValueError(f'诊断尚未完成，不能构建完成页: {run.get("status")}')
    if run.get('scene') != '128' or run.get('frame') != 0 or run.get('render_calls') != 5:
        raise ValueError('run 不是 scene128/frame000/五层真实渲染')
    expected_files = {image_path(layer, kind) for layer in LAYERS for kind in KINDS}
    if len(run.get('asset_files', [])) != 25 or set(run['asset_files']) != expected_files:
        raise ValueError('run 的25张图清单与固定五层合同不一致')
    calibration = run.get('calibration', {})
    tolerance = calibration.get('float_tolerance')
    if (not isinstance(tolerance, (int, float)) or
            any(not isinstance(calibration.get(key), (int, float)) or calibration[key] > tolerance
                for key in ('paired_joint_G_max_abs', 'paired_joint_A_max_abs')) or
            calibration.get('old_delete_rgb_mismatch_pixels') != 0 or
            calibration.get('old_delete_alpha_mismatch_pixels') != 0):
        raise ValueError('联合删除与既有配对/PNG的校准合同未通过')
    assets = validate_assets(output)
    review_path = output / 'assistant_review.json'
    review = load_object(review_path) if review_path.is_file() else None
    references = optional_references(run_root, output)
    alpha_pair_exists = (output / '../alpha_compositing_pair_r1/index.html').is_file()
    page = render_page(run, review, references, alpha_pair_exists)
    (output / 'index.html').write_text(page, encoding='utf-8')
    links = validate_links(output, page)
    result = {'status': 'pass' if not links['missing_links'] else 'fail', 'page_status': 'completed',
              **assets, **links, 'optional_reference_png_decoded': len(references),
              'reference_images': references, 'assistant_review_present': review is not None,
              'alpha_compositing_page_linked': alpha_pair_exists,
              'browser_visual_qa': False, 'videos_encoded': 0,
              'float_pt_embedded': False, 'source_run_json': str(output / 'run.json')}
    (output / 'delivery_validation.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if links['missing_links']:
        raise RuntimeError(f'页面引用缺失: {links["missing_links"]}')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=RUN_DEFAULT)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    output = (args.output_dir or run_root / DIAGNOSTIC).resolve()
    print(json.dumps(build(run_root, output), ensure_ascii=False))


if __name__ == '__main__':
    main()
