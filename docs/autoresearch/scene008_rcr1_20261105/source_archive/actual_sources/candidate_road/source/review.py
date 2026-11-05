"""Fixed-ray evaluation and visual report for the native 10K + R3 1K scene."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

RUN = Path(__file__).resolve().parent.parent
CAMPAIGN = RUN.parent
NATIVE = CAMPAIGN/'scene008_native_ownership_20261105'
LEGACY = CAMPAIGN/'scene008_a37_only_B10k_road_20261103_r3'
OBS = CAMPAIGN/'scene008_instances_20261105_r1/observations/frames'
REFERENCE = CAMPAIGN/'scene008_a6_a37_dynamic_20261104_r1/inputs/a6_probe_reference.npz'
PILOTS = (105, 113, 130)
LABELS = {
    'road_reobserved_after_delete': '删车后 · 有跨时刻 RGB 的显露道路 MAE',
    'a6_visible_rgb': '删车后 · 已观测 a6 RGB MAE',
    'a6_measured_depth': '删车后 · a6 实测深度 MAE (m)',
    'a6_visible_w': '删车后 · 已观测 a6 贡献',
    'a6_reprojected_w': '删车后 · 实测回投 a6 贡献',
    'a37_kept_w': '无编辑 · 可见 a37 贡献（恢复后）',
    'road_visible_rgb': '无编辑 · 可见道路 RGB MAE',
    'a6_noedit_rgb': '无编辑 · 已观测 a6 RGB MAE',
    'a37_noedit_rgb': '无编辑 · 已观测 a37 RGB MAE',
    'protect_noedit_rgb': '无编辑 · 保护区 RGB MAE',
    'full_noedit_rgb': '无编辑 · 全图 RGB MAE',
}
SVG = '''
<svg viewBox="0 0 920 240" role="img" aria-label="输入、道路恢复和联合渲染">
<defs><marker id="a" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0L7 3.5L0 7Z" fill="#536779"/></marker></defs>
<g fill="none" stroke="#536779" stroke-width="2" marker-end="url(#a)">
<path d="M188 60H222"/><path d="M188 169H222"/><path d="M432 60H470V115H499"/><path d="M432 169H470V115H499"/><path d="M686 115H724"/>
</g>
<g stroke="#bbcad3">
<rect x="10" y="33" width="178" height="54" rx="9" fill="#edf3f6"/>
<rect x="10" y="142" width="178" height="54" rx="9" fill="#edf3f6"/>
<rect x="222" y="25" width="210" height="70" rx="9" fill="#e4f2ed"/>
<rect x="222" y="134" width="210" height="70" rx="9" fill="#e7eef8"/>
<rect x="499" y="80" width="187" height="70" rx="9" fill="#faf0e0"/>
<rect x="724" y="80" width="186" height="70" rx="9" fill="#edf3f6"/>
</g>
<g fill="#172837" font-family="sans-serif" text-anchor="middle" font-size="15">
<text x="99" y="65">199 帧 RGB、LiDAR</text><text x="99" y="174">本轮原生 10K</text>
<text x="327" y="54">世界坐标道路</text><text x="327" y="77">重算证据与颜色，随后固定</text>
<text x="327" y="162">静态层、a6、a37、环境</text><text x="327" y="185">仅静态候选门控训练 1K</text>
<text x="592" y="109">统一深度排序</text><text x="592" y="132">同一份道路和静态门控</text>
<text x="817" y="109">无编辑 / 删除 a37</text><text x="817" y="132">同域 49 帧评价</text>
</g></svg>'''


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB'), np.float32)/255


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def metric(values, mask):
    sample = values[mask]
    if not np.isfinite(sample).all():
        raise ValueError('Nonfinite metric values')
    return dict(count=sample.size, sum=float(sample.sum(dtype=np.float64)))


def pool(rows):
    combined = {}
    for name, variants in rows[0]['metrics'].items():
        combined[name] = {}
        for key in variants:
            stats = [row['metrics'][name][key] for row in rows]
            count = sum(item['count'] for item in stats)
            combined[name][key] = dict(count=count, mean=sum(item['sum'] for item in stats)/count if count else None)
    return combined


def evaluate():
    dest = RUN/'delivery'
    dest.mkdir(exist_ok=False)
    images = dest/'images'
    images.mkdir()
    render = RUN/'evaluation/all49'
    assert load(RUN/'queue_status.json')['stage'] == 'COMPLETE'
    records = load(render/'metrics.json')
    assert records['parent_unchanged'] and records['parent_png_max_uint8_difference'] <= 1
    assert records['road_and_gates_active_in_noedit']
    formal = load(RUN/'formal/receipt.json')
    smoke = load(RUN/'smoke_summary.json')
    assert formal['updates'] == 1000 and smoke['full_state_restored']
    previous = load(NATIVE/'run_r3_review/data/audit.json')
    for item in previous['checkpoint']:
        path = Path(item['path'])
        assert path.stat().st_size == item['bytes'] and digest(path) == item['sha256'], path
    for name in ('restore_a', 'restore_b', 'evaluation'):
        assert load(RUN/name/'restore.json')['exact_full_state']
    with np.load(REFERENCE, allow_pickle=False) as source:
        fixed = source['depth_m'].copy()
    rows = []
    for frame in records['rows']:
        raw = frame['raw']
        before = rgb(render/f'{raw:03d}_baseline.png')
        after = rgb(render/f'{raw:03d}_repaired.png')
        kept0 = rgb(render/f'{raw:03d}_noedit_parent.png')
        kept1 = rgb(render/f'{raw:03d}_noedit_road.png')
        old = rgb(LEGACY/'evaluation/all49'/f'{raw:03d}_repaired.png')
        ref = fixed[raw-89]
        with np.load(OBS/f'{raw:03d}.npz',allow_pickle=False) as obs:
            target = obs['rgb'].astype(np.float32)
            road_target = obs['road_edit_rgb'].astype(np.float32)
            if target.max() > 1:
                target /= 255
            if road_target.max() > 1:
                road_target /= 255
            a6, a37 = obs['visible']
            domains = {
                'road_reobserved_after_delete': (road_target, obs['road_edit'] & (ref[3]>0), before, after, old),
                'a6_visible_rgb': (target, a6, before, after, old),
                'road_visible_rgb': (target, obs['road'], kept0, kept1, None),
                'a6_noedit_rgb': (target, a6, kept0, kept1, None),
                'a37_noedit_rgb': (target, a37, kept0, kept1, None),
                'protect_noedit_rgb': (target, obs['protect'], kept0, kept1, None),
                'full_noedit_rgb': (target, np.ones_like(a6), kept0, kept1, None),
            }
        result = {}
        for name, (goal, mask, b, a, legacy) in domains.items():
            result[name] = {key:metric(np.abs(image-goal).mean(2), mask)
                            for key, image in [('parent',b), ('new',a), ('old_r3',legacy)] if image is not None}
        depth = frame['a6_depth']
        result['a6_measured_depth'] = {key:dict(count=depth['count'],sum=depth[field])
                                       for key, field in [('parent','before'),('new','after')]}
        with np.load(render/f'{raw:03d}_layers.npz',allow_pickle=False) as features:
            for name, mask in [('a6_visible_w', a6), ('a6_reprojected_w', ref[1]>0)]:
                result[name] = {'parent':metric(features['before'][0],mask), 'new':metric(features['after'][1],mask)}
            result['a37_kept_w'] = {'new':metric(features['keep'][2],a37)}
        rows.append(dict(raw=raw, metrics=result))
        grid = np.vstack((np.hstack((before, after, old)), np.hstack((target, kept0, kept1))))
        Image.fromarray(np.uint8(np.round(grid*255))).save(images/f'{raw:03d}.jpg', quality=90)
    assert [item['raw'] for item in rows] == list(range(89,138))
    pooled = pool(rows)
    assert pooled['road_reobserved_after_delete']['new']['count'] == 287368
    assert pooled['a6_visible_rgb']['new']['count'] == 39308
    summary = dict(status='PASS_ENGINEERING', native_10k_files_sha256_verified=True, formal=formal, smoke=smoke,
                   parent_binding=load(RUN/'parent_binding.json'), selection=load(RUN/'selection.json'),
                   road_inputs=load(RUN/'inputs/receipt.json'), pooled=pooled, rows=rows,
                   reference='Same-scene re-observations; sRGB metrics decoded identically for all compared PNGs; depth and contribution from native raster',
                   quality_accepted=False,human_verdict=None)
    save(dest/'summary.json',summary)
    with (dest/'fixed_ray_metrics.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.writer(handle)
        writer.writerow(['raw','metric','arm','count','sum','mean'])
        for row in rows:
            for name, variants in row['metrics'].items():
                for key, item in variants.items():
                    count=item['count']; total=item['sum']
                    writer.writerow([row['raw'],name,key,count,total,total/count if count else None])
    table=''
    for name, title in LABELS.items():
        cells=[]
        for key in ('parent','new','old_r3'):
            item=pooled[name].get(key)
            cells.append('<td>—</td>' if item is None else f"<td>{item['mean']:.6f}<small>{item['count']:,}</small></td>")
        table+=f'<tr><th>{title}</th>{"".join(cells)}</tr>'
    contents=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>Scene008 · 原生 10K + 道路恢复 1K</title>
<style>
body{{background:#f5f8f9;color:#172837;font:16px/1.58 system-ui,sans-serif;margin:0}}
main{{max-width:1100px;margin:auto;padding:30px 22px}}
section{{background:white;border:1px solid #d7e0e7;border-radius:12px;padding:19px;margin:18px 0;overflow:auto}}
svg,img{{width:100%;display:block}} table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}
th,td{{text-align:left;padding:7px 10px;border-bottom:1px solid #dfe6ec}}small{{display:block;color:#667483}}
a{{color:#05616a}}input{{width:45%;margin:0 8px}}button{{padding:4px 11px}}
</style><main><h1>Scene008 原生 10K + R3 道路恢复 1K</h1>
<section><h2>组件与数据流</h2>{SVG}
<p>沿用 R3 的损失、步数与 199 帧证据流程，针对新父模型重算道路颜色校准和静态门控候选。删车训练保留 a6。正式场景中的道路与门控跨相机、时间和编辑状态共享。</p></section>
<section><h2>49 帧固定域配对</h2>
<table><tr><th>指标 / 像素数</th><th>本轮原生 10K</th><th>本轮 + 道路 1K</th><th>旧强编辑 R3</th></tr>{table}</table>
<p>显露道路参考来自同一场景的跨时刻观测。隐藏 a6 下部缺独立 RGB 真值。RGB 用一致的 PNG 量化后重算。</p></section>
<section><h2>49 帧试看片</h2>
<p>上排从左到右为本轮原生 10K 删车、本轮 +1K 删车、旧强 R3 删车。下排依次为原始 RGB、本轮原生无编辑、本轮 +1K 无编辑。</p>
<label for="raw">raw<span id="number">113</span></label><input id="raw" type="range" min="89" max="137" value="113">
<button data-frame="105">105</button><button data-frame="113">113</button><button data-frame="130">130</button>
<img id="frame" src="images/113.jpg" alt="raw113 六格对照">
<script>
const slider=document.getElementById('raw'),number=document.getElementById('number'),image=document.getElementById('frame');
function update(){{const key=slider.value.padStart(3,'0');number.textContent=key;image.src='images/'+key+'.jpg';image.alt='raw'+key+' 六格对照';}}
slider.addEventListener('input',update);document.querySelectorAll('button[data-frame]').forEach(b=>b.addEventListener('click',()=>{{slider.value=b.dataset.frame;update();}}));
</script></section>
<section><h2>可核对文件</h2><a href="fixed_ray_metrics.csv">49 帧逐项误差和及分母</a> · <a href="summary.json">指标与验收回执</a> · <a href="road_checkpoint001000.pt">1K 可恢复检查点</a></section></main></html>'''
    (dest/'index.html').write_text(contents,encoding='utf-8')
    shutil.copyfile(RUN/'formal/checkpoint001000.pt',dest/'road_checkpoint001000.pt')
    save(dest/'checkpoint_receipt.json',dict(sha256=digest(dest/'road_checkpoint001000.pt'),
        source=str(RUN/'formal/checkpoint001000.pt'), parent=summary['parent_binding']['parent']))
    print(json.dumps(pooled,indent=2))


if __name__ == '__main__':
    evaluate()
