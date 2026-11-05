"""Produce the paired 49-frame result and ordering diagnostics."""
import csv
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image

RUN = Path(__file__).resolve().parent.parent
FUSION = RUN.parent / 'scene008_a6_local_fusion_20261105'
sys.path.insert(0, str(RUN.parent / 'scene008_a6_residual_replay_20261105' / 'source'))
from report import image_grid, pooled


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    order = read(RUN / 'order_controls.json')
    black = read(RUN / 'black_order.json')
    selected = read(RUN / 'support_verified.json')
    assert selected['status'] == 'PASS' and selected['frames'] == 49
    assert order['center_noop_max'] == {'rgb':0., 'alpha':0., 'depth_sum':0., 'extra':0.}
    summary = dict(status='PASS', frames=49, selected='support_adjoint', native_center_parity=order['center_noop_max'],
                   adjusted_rows=selected['n_adjusted'], max_attenuation=selected['cap'],
                   black={'before':pooled([x['baseline'] for x in black['rows']]),
                          'after':pooled(selected['black'])}, street=selected['chosen']['street'],
                   fixed_hidden_pixels=sum(x['metrics']['lower_pvd']['count'] for x in selected['black']),
                   ordering=order['black'], zero_shift=order['zero_shift_ratio'])
    d = RUN / 'delivery'
    d.mkdir()
    frames = d / 'frames'
    frames.mkdir()
    for raw in range(89, 138):
        with Image.open(FUSION / 'render' / f'{raw:03d}_pvd_pvd.png') as im:
            y, x = np.nonzero(np.asarray(im)[:, :, 0] > 5)
            w, h = im.size
        box = (max(0,int(x.min())-11), max(0,int(y.min())-11), min(w,int(x.max())+12), min(h,int(y.max())+12))
        src = RUN / 'support_render'
        labels = ['Before RGB','After RGB','Before PVD contribution','After PVD contribution',
                  'Before blocked weight','After blocked weight']
        paths = [src/f'{raw:03d}_{side}{kind}.png' for kind in ('','_pvd','_blocked') for side in ('before','after')]
        image_grid(paths, labels, box=box, scale=3).save(frames/f'{raw:03d}.png')
        paths = [FUSION/'render'/f'{raw:03d}_{arm}.png' for arm in ('native','pvd','direct')] + [src/f'{raw:03d}_after.png']
        image_grid(paths, ['Native a6','PVD only','Direct overlay','Guarded fusion'],
                   box=box, scale=2, columns=4).save(frames/f'{raw:03d}_reference.png')
        paths = [src/f'{raw:03d}_{state}_{side}.png' for state in ('keep','delete') for side in ('before','after')]
        image_grid(paths, ['No edit | before','No edit | after','Delete a37 | before','Delete a37 | after']).save(
            frames/f'{raw:03d}_street.jpg', quality=91)
    for name in ('black_order.json','order_controls.json','support_search.json','support_verified.json','support_accepted.npz',
                 'replay_receipt.json'):
        shutil.copyfile(RUN/name, d/name)
    (d/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    with (d/'fixed_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        out=csv.writer(f,lineterminator='\n')
        out.writerow(['raw','state','arm','metric','count','sum','mean'])
        for raw in range(89,138):
            record=black['rows'][raw-89]
            arms={'parent':record['baseline'],'adopted':selected['black'][raw-89], **record['modes']}
            for arm,r in arms.items():
                for k,m in r['metrics'].items():
                    out.writerow([raw,'black',arm,k,m['count'],m['sum'],m['sum']/m['count'] if m['count'] else ''])
            for state in ('keep','delete'):
                for k,m in selected['street'][raw-89][state].items():
                    out.writerow([raw,state,'paired',k,m['count'],m['sum'],m['sum']/m['count'] if m['count'] else ''])

    before, after = (summary['black'][key] for key in ('before','after'))
    d37 = summary['street']['delete']
    keep = summary['street']['keep']
    rows = [
        ('黑底下部显露', f'{before["ratio"]:.2%}', f'{after["ratio"]:.2%}'),
        ('删 a37 下部显露', f'{d37["before_fraction"]:.2%}', f'{d37["after_fraction"]:.2%}'),
        ('黑底已观测 a6 RGB MAE', f'{before["known_rgb"]:.6f}', f'{after["known_rgb"]:.6f}'),
        ('黑底已测深度 MAE (m)', f'{before["measured_depth"]:.6f}', f'{after["measured_depth"]:.6f}'),
        ('无编辑道路 RGB MAE', f'{keep["road_before"]:.6f}', f'{keep["road_after"]:.6f}'),
        ('删车道路 RGB MAE', f'{d37["road_before"]:.6f}', f'{d37["road_after"]:.6f}'),
    ]
    table = ''.join(f'<tr><th>{name}</th><td>{a}</td><td>{b}</td></tr>' for name,a,b in rows)
    labels = {'clip_near':'裁剪行 · 中位数 · ±0.25 m', 'clip_wide':'裁剪行 · 中位数 · ±0.75 m',
              'all_near':'全 a6 与 PVD · 中位数 · ±0.25 m', 'all_wide':'全 a6 与 PVD · 中位数 · ±0.75 m',
              'clip_mass_near':'裁剪行 · 三片 · ±0.25 m', 'clip_mass_wide':'裁剪行 · 三片 · ±0.75 m',
              'all_mass_near':'全 a6 与 PVD · 三片 · ±0.25 m', 'all_mass_wide':'全 a6 与 PVD · 三片 · ±0.75 m'}
    controls = ''.join(f'<tr><th>{labels[name]}</th><td>{x["ratio"]:.5%}</td><td>{x["failures"]}</td></tr>'
                      for name,x in order['black'].items())
    html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Scene008 a6 射线遮挡与接管</title>
<style>
body{{background:#f3f7fa;color:#172938;font:16px/1.55 system-ui,sans-serif;margin:0}}
main{{max-width:1080px;margin:auto;padding:24px}}section{{background:#fff;border:1px solid #d6e2e8;border-radius:10px;padding:17px;margin:16px 0;overflow:auto}}
svg,img{{width:100%;display:block}}table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}
th,td{{text-align:left;padding:6px 9px;border-bottom:1px solid #dde7ec}}input{{width:46%}}a{{color:#086675}}
</style><main><h1>Scene008 · a6 射线遮挡与接管</h1>
<section><h2>组件与数据流</h2>
<svg viewBox="0 0 940 147" role="img" aria-label="射线归因与保护">
<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0 0L8 3L0 6Z" fill="#637788"/></marker></defs>
<g fill="none" stroke="#637788" stroke-width="2" marker-end="url(#arrow)">
<path d="M195 37H222V69H246"/><path d="M195 108H222V69H246"/><path d="M441 69H477"/><path d="M683 69H717"/>
</g><g stroke="#bac8d0">
<rect x="10" y="13" width="185" height="49" rx="8" fill="#edf4f8"/><rect x="10" y="83" width="185" height="49" rx="8" fill="#edf4f8"/>
<rect x="246" y="45" width="195" height="49" rx="8" fill="#e2f1eb"/><rect x="477" y="45" width="206" height="49" rx="8" fill="#f7f0e0"/>
<rect x="717" y="45" width="212" height="49" rx="8" fill="#edf4f8"/></g>
<g fill="#172938" font-family="sans-serif" font-size="15" text-anchor="middle">
<text x="102" y="43">冻结场景与 PVD</text><text x="102" y="114">RGB、LiDAR 与保护域</text>
<text x="344" y="75">原生射线及排序对照</text><text x="580" y="75">逐行影响与受限接管</text><text x="823" y="75">49 帧黑底与街景复核</text></g></svg>
<p>排序对照保留为诊断。正式方案沿用中心排序，按每行对下游 PVD 和保护像素的影响分配少量不透明度衰减。每个系数在两种编辑状态和所有帧共用，PVD 与道路场固定。</p></section>
<section><h2>固定域验收</h2><table><tr><th>指标</th><th>上一轮</th><th>本轮</th></tr>{table}</table>
<p>固定显露范围为 {summary['fixed_hidden_pixels']:,} 个像素。采用版本调整 {summary['adjusted_rows']:,} 个原生行身份。独立重放覆盖 49 帧黑底、无编辑和删 a37。隐藏表面缺少独立几何真值。</p></section>
<section><h2>49 帧黑底对照</h2><p>两列为上一轮和本轮。三行依次为 RGB、PVD 可见贡献、未显露权重。</p>
<label for="raw">raw<span id="no">113</span></label><input id="raw" type="range" min="89" max="137" value="113"><button data-raw="97">97</button><button data-raw="105">105</button><button data-raw="113">113</button><button data-raw="130">130</button><button data-raw="137">137</button>
<img id="black" src="frames/113.png" alt="raw113 黑底配对"></section>
<section><h2>固定资产四列</h2><img id="reference" src="frames/113_reference.png" alt="原生、PVD、直接叠加、采用版本"></section>
<section><h2>街景复核</h2><p>上排无编辑，下排仅删 a37。两列共用相同道路颜色场。</p><img id="street" src="frames/113_street.jpg" alt="街景对照"></section>
<section><h2>排序对照</h2><p>在同一份上一轮资产上比较条件中位数和三份光学厚度采样。中心顺序重放逐像素一致。基础黑底显露率为 {order['baseline']:.5%}。</p>
<table><tr><th>排序方法</th><th>下部显露</th><th>黑底保护失败帧</th></tr>{controls}</table></section>
<section><h2>逐帧与复现</h2><a href="fixed_metrics.csv">逐帧指标</a> · <a href="order_controls.json">排序诊断</a> · <a href="support_verified.json">采用结果</a> · <a href="support_accepted.npz">持久系数</a> · <a href="replay_receipt.json">独立重放</a></section>
<script>
const input=document.getElementById('raw'),no=document.getElementById('no'),black=document.getElementById('black'),street=document.getElementById('street'),reference=document.getElementById('reference');
function update(){{const raw=input.value.padStart(3,'0');no.textContent=raw;black.src='frames/'+raw+'.png';street.src='frames/'+raw+'_street.jpg';reference.src='frames/'+raw+'_reference.png';}}
input.addEventListener('input',update);document.querySelectorAll('button[data-raw]').forEach(b=>b.addEventListener('click',()=>{{input.value=b.dataset.raw;update();}}));
</script></main>'''
    (d/'index.html').write_text(html,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
