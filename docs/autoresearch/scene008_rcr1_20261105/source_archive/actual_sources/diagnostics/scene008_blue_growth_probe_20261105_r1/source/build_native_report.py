"""Build the paired native-10K review from saved metric numerators and counts."""
import csv
import html
import json
from pathlib import Path
import shutil
import sys

LABELS = {
    'a6_static_front': 'Static opacity ahead of measured a6',
    'a6_reveal_static_front': 'Static opacity ahead of reprojected a6',
    'a6_delete_w': 'Visible a6 contribution after deletion',
    'a6_reveal_w': 'a6 contribution on reprojected rays after deletion',
    'a6_wrong_w37': 'a37 contribution on visible a6',
    'a37_wrong_w6': 'a6 contribution on visible a37',
    'strict_a37_w': 'a37 contribution on strict occlusion rays',
    'a6_iso_rgb': 'Isolated a6 RGB MAE',
    'a37_iso_rgb': 'Isolated a37 RGB MAE',
    'a6_delete_rgb': 'a6 RGB MAE after deletion',
    'a6_keep_rgb': 'a6 RGB MAE before deletion',
    'a37_keep_rgb': 'a37 RGB MAE before deletion',
    'a6_depth_m': 'Measured a6 depth MAE after deletion (m)',
    'road_keep_rgb': 'Observed road RGB MAE before deletion',
    'protected_keep_rgb': 'Protected-region RGB MAE before deletion',
    'full_keep_rgb': 'Full-frame RGB MAE before deletion',
}
ARCH = '''
<svg viewBox="0 0 940 268" role="img" aria-label="Architecture and training data flow">
<defs><marker id="arr" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="#63717b"/></marker></defs>
<g fill="none" stroke="#63717b" stroke-width="2" marker-end="url(#arr)">
<path d="M196 127H217V57H253"/><path d="M196 127H253"/><path d="M196 127H217V197H253"/>
<path d="M484 57H512V127H543"/><path d="M484 127H543"/><path d="M484 197H512V127H543"/>
<path d="M711 127H758"/><path d="M632 161V207H758"/>
</g>
<g stroke="#bbcad3" rx="11">
<rect x="12" y="92" width="184" height="70" rx="10" fill="#eef3f6"/>
<rect x="253" y="27" width="231" height="60" rx="10" fill="#e4f3f0"/>
<rect x="253" y="97" width="231" height="60" rx="10" fill="#e4f3f0"/>
<rect x="253" y="167" width="231" height="60" rx="10" fill="#e8eef7"/>
<rect x="543" y="92" width="168" height="70" rx="10" fill="#f4f0e7"/>
<rect x="758" y="92" width="170" height="70" rx="10" fill="#eef3f6"/>
<rect x="758" y="177" width="170" height="60" rx="10" fill="#eef3f6"/>
</g>
<g fill="#172837" font-family="sans-serif" text-anchor="middle" font-size="16">
<text x="104" y="121">RGB、LiDAR、轨迹</text><text x="104" y="145">实例可见与遮挡证据</text>
<text x="368" y="53">a6 独立动态层</text><text x="368" y="74" font-size="13">a6 可见 RGB、覆盖、深度</text>
<text x="368" y="123">a37 独立动态层</text><text x="368" y="144" font-size="13">a37 可见 RGB、覆盖、深度</text>
<text x="368" y="193">原生静态层与环境</text><text x="368" y="214" font-size="13">背景 RGB 驱动普通增密</text>
<text x="627" y="121">原生联合深度排序</text><text x="627" y="145" font-size="13">实例贡献、实测深度</text>
<text x="843" y="120">重建与删除 a37</text><text x="843" y="144" font-size="13">共用一份场景</text>
<text x="843" y="204">训练内检查点</text><text x="843" y="225" font-size="13">固定射线与试看片</text>
<text x="410" y="251" font-size="13">静态层另测实测表面前的透射率，约束阻挡</text>
</g></svg>'''


def json_read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def render(run, destination):
    destination.mkdir(parents=True, exist_ok=True)
    data = destination / 'data'
    data.mkdir(exist_ok=True)
    sets = {}
    for arm in ('stopped_v1', 'stopped_v2', 'smoke', 'resume', 'control', 'formal'):
        sets[arm] = {}
        historical = {'stopped_v1': 'run_r1', 'stopped_v2': 'run_r2'}
        root = run.parent / historical[arm] / 'smoke' if arm in historical else run / arm
        for path in sorted(root.glob('probe_*/summary.json')):
            record = json_read(path)
            step = record['iteration']
            sets[arm][step] = record
            prefix = f'{arm}_{step:05}'
            shutil.copyfile(path, data / (prefix + '.json'))
            for png in sorted(path.parent.glob('raw*.png')):
                shutil.copyfile(png, data / (prefix + '_' + png.name))
        for pattern in ('returned.json', 'restore_audit.json', 'gradient_routing_*.json', 'progress.json', 'early_accept_*.json'):
            for path in root.glob(pattern):
                shutil.copyfile(path, data / f'{arm}_{path.name}')
    with (data / 'fixed_ray_metrics.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        out = csv.writer(handle)
        out.writerow(['arm', 'step', 'raw', 'metric', 'count', 'sum', 'mean'])
        for arm, records in sets.items():
            for step, rec in records.items():
                for row in rec['rows']:
                    for name, value in row['metrics'].items():
                        n, s = value['count'], value['sum']
                        out.writerow([arm, step, row['raw'], name, n, s, s/n if n else None])

    def table(columns):
        columns = [(arm, step) for arm, step in columns if step in sets[arm]]
        if not columns:
            return '<p>尚无匹配记录。</p>'
        headers = ''.join(f'<th>{arm} · {step}</th>' for arm, step in columns)
        rows = []
        for key, name in LABELS.items():
            cells = []
            for arm, step in columns:
                item = sets[arm][step]['pooled'][key]
                val = item['mean']
                cells.append('<td>—</td>' if val is None else f"<td>{val:.4f}<small>{item['count']:,}</small></td>")
            rows.append(f'<tr><th>{name}</th>{"".join(cells)}</tr>')
        return f'<table><tr><th>指标 · 有效像素</th>{headers}</tr>{"".join(rows)}</table>'

    galleries = []
    for step in (1000, 10000):
        for raw in (105, 113, 130):
            imgs = []
            for arm in ('control', 'formal'):
                path = data / f'{arm}_{step:05}_raw{raw}.png'
                if path.is_file():
                    imgs.append(f'<figure><figcaption>{arm} · {step} · raw{raw}</figcaption>'
                                f'<img src="data/{path.name}" alt="{arm} raw{raw}"></figure>')
            if imgs:
                galleries.append('<div class="gallery">' + ''.join(imgs) + '</div>')
    status = html.escape((run/'status').read_text().strip())
    head = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>Scene008 · 原生 10K 实例监督</title>
<style>
body{font:16px/1.6 system-ui,sans-serif;margin:0;background:#f5f8f9;color:#172837}
main{max-width:1150px;margin:auto;padding:36px 24px}
h1,h2{line-height:1.24} section{background:white;border:1px solid #dbe4e8;border-radius:12px;padding:20px;margin:24px 0;overflow:auto}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums} td,th{padding:6px 9px;text-align:left;border-bottom:1px solid #dbe4e8}small{display:block;color:#62717d}
svg,img{width:100%;display:block}.gallery{display:flex;gap:12px;flex-wrap:wrap;margin-top:12px}
figure{margin:0;min-width:330px;flex:1}figcaption{font-weight:600;padding:5px 0}a{color:#06636b}
</style><main><h1>Scene008 原生 10K · 实例监督与静态遮挡</h1>"""
    report = head + f'<p>执行状态 <strong>{status}</strong>。原生训练输入为 raw089–137。冻结的观测库来自同场景 199 帧。</p>'
    report += '<section><h2>组件与数据流</h2>' + ARCH + '</section>'
    report += '''<section><h2>训练与判定口径</h2>
<p>两车从初始点云分开，沿用原生动态参数化、相机采样、学习率时钟与 SGCM。a6 和 a37 的颜色分别从自身可见 RGB 更新，联合排序的贡献和实测深度约束可见所有权。静态普通增密读取背景分支的二维位置与时间梯度。实测静态遮挡项在独立渲染中读取标记透射率。</p>
<p>同一射线可在两车孤立图中出现。联合排序中的错误实例贡献才纳入串车指标。未直接显露的射线使用其他时刻测到的 a6 表面回投，供机制检查。训练沿用共享的实测参考射线，指标衡量同场景拟合。生成几何、道路修复和 10K 后的替换均未参与。</p></section>'''
    report += '<section><h2>原生训练内的早期读出 · raw105、113、130</h2>'
    report += table([('formal', 100), ('formal', 500), ('control', 1000), ('formal', 1000), ('formal', 3000), ('formal', 3300), ('formal', 6000), ('formal', 6300)]) + '</section>'
    report += '<section><h2>早期试验发现的道路退化</h2><p>第一版在 1,915 步停止，1K 的贡献图显示 a37 越界覆盖可见道路。第二版加背景负约束后，a37 在三帧可见道路上的权重由约 0.171 降至 0.093。分层重放显示第二版的道路误差主要来自静态层，静态单层 MAE 为 0.0834，旧双动态为 0.0393。第二版在 1,929 步停止。第三版保留背景负约束，并恢复原生背景 LiDAR 全图归一化。两个试跑现场均保存。</p>'
    report += table([('stopped_v1', 1000), ('stopped_v2', 1000), ('control', 1000), ('smoke', 1000)]) + '</section>'
    report += '<section><h2>10K 固定 49 帧配对</h2>' + table([('control', 10000), ('formal', 10000)]) + '</section>'
    report += '''<section><h2>试看片</h2><p>上排依次为原始 RGB、无编辑联合渲染、删除 a37。下排依次为 a6 孤立、a37 孤立、联合分组贡献。分组颜色为 a37 红、a6 绿、静态蓝。</p>'''
    report += ''.join(galleries) + '</section>'
    edit = {}
    for arm in ('union', 'native'):
        folder = run / f'edit_{arm}'
        path = folder / 'delete_domains.json'
        if path.is_file():
            edit[arm] = json_read(path)
            shutil.copyfile(path, data / f'edit_{arm}.json')
            for image in folder.glob('edit_raw*.png'):
                shutil.copyfile(image, data / f'edit_{arm}_{image.name}')
    if set(edit) == {'union', 'native'}:
        old, new = edit['union'], edit['native']
        a, b = old['road_reobserved_rgb'], new['road_reobserved_rgb']
        report += '<section><h2>删除 a37 后的显露区域</h2>'
        report += (f"<p>在 {b['count']:,} 个同时有跨时刻道路 RGB 与路面深度代理的像素上，"
                   f"删车后道路 RGB MAE 为旧双动态 {a['mean']:.4f}、本轮 {b['mean']:.4f}。"
                   "道路回投证据来自同一场景。固定 PVD 足迹仅用于诊断渲染支撑，隐藏几何与纹理精度缺独立真值。</p>")
        rows = []
        for key, label in (('measured_visible', '实测可见 a6'), ('measured_reprojected', '实测回投 a6'),
                           ('fixed_pvd_proxy', '固定 PVD 足迹'), ('road_depth_proxy', '路面深度代理')):
            cols = []
            for item in (old, new):
                domain = item['pooled'][key]
                cols.append(f"<td>{' / '.join(f'{v:.4f}' for v in domain['mean'])}<small>{domain['count']:,}</small></td>")
            rows.append(f'<tr><th>{label}</th>{"".join(cols)}</tr>')
        report += ('<table><tr><th>区域</th><th>旧双动态 · a6 / 静态 / 环境</th>'
                   '<th>本轮 · a6 / 静态 / 环境</th></tr>' + ''.join(rows) + '</table>')
        report += '<p>删车贡献图中红为静态、绿为 a6、蓝为环境。</p>'
        for raw in (105, 113, 130):
            report += '<div class="gallery">'
            for arm in ('union', 'native'):
                path = f'edit_{arm}_edit_raw{raw}.png'
                report += f'<figure><figcaption>{arm} · raw{raw}</figcaption><img src="data/{path}" alt="{arm} raw{raw}"></figure>'
            report += '</div>'
        report += '</section>'
    report += '<section><h2>可核对文件</h2><a href="data/fixed_ray_metrics.csv">逐帧计数、误差和与均值</a> · <a href="data/audit.json">恢复、步数与检查点审计</a> · <a href="data/formal_early_accept_01000.json">1K 继续条件</a></section></main></html>'
    (destination / 'index.html').write_text(report, encoding='utf-8')


if __name__ == '__main__':
    render(Path(sys.argv[1]), Path(sys.argv[2]))
