"""Assemble the black-background comparison and same-frame street replay."""
import csv
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image

RUN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RUN.parent / 'scene008_a6_residual_replay_20261105' / 'source'))
from report import image_grid, pooled

FUSION = RUN.parent / 'scene008_a6_local_fusion_20261105'
SWITCH = RUN.parent / 'scene008_a6_ray_switch_20261105'
PILOTS = (105, 113, 130)
ARCH = '''<svg viewBox="0 0 940 145" role="img" aria-label="a6 局部接管数据流">
<defs><marker id="arr" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0 0L8 3L0 6Z" fill="#637788"/></marker></defs>
<g fill="none" stroke="#637788" stroke-width="2" marker-end="url(#arr)">
<path d="M188 36H219V69H240"/><path d="M188 106H219V69H240"/><path d="M437 69H474"/><path d="M676 69H710"/>
</g><g stroke="#bac8d0">
<rect x="10" y="12" width="178" height="49" rx="8" fill="#edf4f8"/><rect x="10" y="82" width="178" height="49" rx="8" fill="#edf4f8"/>
<rect x="240" y="45" width="197" height="49" rx="8" fill="#e2f1eb"/><rect x="474" y="45" width="202" height="49" rx="8" fill="#f7f0e0"/>
<rect x="710" y="45" width="218" height="49" rx="8" fill="#edf4f8"/>
</g><g fill="#172938" font-family="sans-serif" font-size="15" text-anchor="middle">
<text x="99" y="43">原生 a6 与固定 PVD</text><text x="99" y="113">实测与非目标保护</text>
<text x="339" y="75">共享车体局部接管</text><text x="575" y="75">同一排序合成</text><text x="819" y="75">49 帧黑底及街景复核</text></g></svg>'''


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def mean(m):
    return m['sum'] / m['count'] if m['count'] else None


def main():
    choices = []
    for name, receipt, ledger, image_dir in (
            ('tradeoff', 'verify.json', 'accepted.npz', 'render'),
            ('geometry', 'geometry_verified.json', 'geometry_accepted.npz', 'geometry_render'),
            ('combined', 'combined_verified.json', 'combined_accepted.npz', 'combined_render'),
            ('joint_optimized', 'joint_optimized_verified.json', 'joint_optimized_accepted.npz', 'joint_optimized_render')):
        if (RUN / receipt).exists():
            item = read(RUN / receipt)
            if item['status'] == 'PASS':
                choices.append((name, receipt, ledger, image_dir, item))
    assert choices
    name, receipt, ledger, image_dir, verification = max(
        choices, key=lambda c: (c[-1]['chosen']['black'], c[-1]['chosen']['street']['delete']['after_fraction']))
    search = read(RUN / 'black_search.json')
    assert verification['status'] == 'PASS' and verification['frames'] == 49
    assert all(not row['violations'] for row in verification['black'])
    dest = RUN / 'delivery'
    dest.mkdir()
    frames = dest / 'frames'
    frames.mkdir()
    generated = RUN / image_dir
    for raw in range(89, 138):
        with Image.open(FUSION / 'render' / f'{raw:03d}_pvd_pvd.png') as im:
            y, x = np.nonzero(np.asarray(im)[:, :, 0] > 5)
            w, h = im.size
        box = max(0, int(x.min()) - 11), max(0, int(y.min()) - 11), min(w, int(x.max()) + 12), min(h, int(y.max()) + 12)
        variants = ('before', 'after')
        paths = [generated / f'{raw:03d}_{arm}{kind}.png' for kind in ('', '_pvd', '_blocked') for arm in variants]
        labels = ['Previous a6', 'New a6', 'Previous PVD contribution', 'New PVD contribution',
                  'Previous blocked contribution', 'New blocked contribution']
        image_grid(paths, labels, box=box, scale=3).save(frames / f'{raw:03d}.png')
        reference = [FUSION / 'render' / f'{raw:03d}_{arm}.png' for arm in ('native', 'pvd', 'direct')]
        reference.append(generated / f'{raw:03d}_after.png')
        image_grid(reference, ['Native a6', 'PVD only', 'Direct overlay', 'Guarded fusion'],
                   box=box, scale=2, columns=4).save(frames / f'{raw:03d}_reference.png')
        states = ('keep', 'delete')
        paths = [generated / f'{raw:03d}_{state}_{arm}.png' for state in states for arm in variants]
        image_grid(paths, ['No edit | previous', 'No edit | new', 'Delete a37 | previous', 'Delete a37 | new']).save(
            frames / f'{raw:03d}_street.jpg', quality=91)
    for filename in ('black_search.json', 'shader_validation.json', 'tradeoff.json',
                     'geometry_black_search.json', 'geometry_verified.json', 'combined_verified.json',
                     'joint_optimized_verified.json', 'joint_optimized_trials.json'):
        if (RUN / filename).exists():
            shutil.copyfile(RUN / filename, dest / filename)
    shutil.copyfile(RUN / receipt, dest / 'verify.json')
    shutil.copyfile(RUN / ledger, dest / 'accepted.npz')
    best = verification['chosen']
    keep = best['street']['keep']
    delete = best['street']['delete']
    original = pooled(search['prior'])
    accepted = pooled(verification['black'])
    stats = dict(status='PASS', frames=49, selected=name, black={'before': original, 'after': accepted},
                 street=best['street'], guards=verification['guards'], training_updates=0,
                 replay_98_images=True, fixed_hidden_pixels=sum(r['metrics']['lower_pvd']['count'] for r in search['prior']))
    (dest / 'summary.json').write_text(json.dumps(stats, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    with (dest / 'fixed_metrics.csv').open('w', encoding='utf-8-sig', newline='') as h:
        writer = csv.writer(h, lineterminator='\n')
        writer.writerow(['raw', 'state', 'variant', 'metric', 'count', 'sum', 'mean'])
        for variant, rows in (('previous', search['prior']), ('accepted', verification['black'])):
            for row in rows:
                for key, value in row['metrics'].items():
                    writer.writerow([row['raw'], 'black', variant, key, value['count'], value['sum'], mean(value)])
        for row in verification['street']:
            for state in ('keep', 'delete'):
                for key, value in row[state].items():
                    writer.writerow([row['raw'], state, 'paired', key, value['count'], value['sum'], mean(value)])
    prior_black, after_black = original['ratio'], accepted['ratio']
    table = ''.join(f'<tr><th>{label}</th><td>{before}</td><td>{after}</td></tr>' for label, before, after in (
        ('a6 黑底下部显露', f'{prior_black:.2%}', f'{after_black:.2%}'),
        ('删 a37 街景下部显露', f'{delete["before_fraction"]:.2%}', f'{delete["after_fraction"]:.2%}'),
        ('黑底已观测 a6 RGB MAE', f'{original["known_rgb"]:.6f}', f'{accepted["known_rgb"]:.6f}'),
        ('黑底实测深度 MAE (m)', f'{original["measured_depth"]:.6f}', f'{accepted["measured_depth"]:.6f}'),
        ('无编辑道路 MAE', f'{keep["road_before"]:.6f}', f'{keep["road_after"]:.6f}'),
        ('删 a37 道路 MAE', f'{delete["road_before"]:.6f}', f'{delete["road_after"]:.6f}')))
    explanation = {
        'tradeoff': '按保护余量重新分配原生行接管强度。',
        'geometry': '采用基于实测 LiDAR 下界的局部裁剪。',
        'combined': '把通过保护的接管强度与实测下界裁剪组合。',
        'joint_optimized': '在实测下界裁剪上重新分配接管强度。'
    }[name]
    html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Scene008 a6 局部接管</title>
<style>
body{{background:#f3f7fa;color:#172938;font:16px/1.6 system-ui,sans-serif;margin:0}}
main{{max-width:1080px;margin:auto;padding:24px}}section{{background:white;border:1px solid #d6e2e8;border-radius:10px;padding:17px;margin:16px 0;overflow:auto}}
img,svg{{display:block;width:100%}}table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}
th,td{{text-align:left;padding:6px 9px;border-bottom:1px solid #dde7ec}}input{{width:48%}}a{{color:#086675}}
</style><main><h1>Scene008 · a6 局部接管</h1>
<section><h2>组件与数据流</h2>{ARCH}<p>{explanation} 原生检查点和 PVD 冻结。共用一份车体局部账本和已采用的道路颜色场，两种状态都由同一次深度排序合成。新增重建训练 0 步。</p></section>
<section><h2>固定域验收</h2><table><tr><th>指标</th><th>上一轮</th><th>本轮</th></tr>{table}</table>
<p>显露率按 {stats['fixed_hidden_pixels']:,} 个固定候选像素汇总，分母为同一批下部 PVD 的独立透明度。49 帧均通过原有黑底保护及街景检查。隐藏部分缺少独立几何真值。</p></section>
<section><h2>49 帧黑底对照</h2><p>左右分别是上一轮和本轮。三行分别是 RGB、PVD 可见贡献、仍被挡住的固定下部权重。</p>
<label for="raw">raw<span id="no">113</span></label><input id="raw" type="range" min="89" max="137" value="113"><button data-raw="97">97</button><button data-raw="105">105</button><button data-raw="113">113</button><button data-raw="130">130</button><button data-raw="137">137</button>
<img id="black" src="frames/113.png" alt="raw113 黑底对照"></section>
<section><h2>黑底固定资产对照</h2><p>原生 a6、PVD 独立、直接叠加和本轮局部融合共用裁剪。</p>
<img id="reference" src="frames/113_reference.png" alt="raw113 四列黑底对照"></section>
<section><h2>街景复核</h2><p>上排无编辑，下排删 a37。两列共用上一轮校正过的道路。</p><img id="street" src="frames/113_street.jpg" alt="raw113 街景对照"></section>
<section><h2>逐帧与复现</h2><a href="fixed_metrics.csv">逐帧指标</a> · <a href="verify.json">全量验收</a> · <a href="black_search.json">分级接管对照</a> · <a href="accepted.npz">共享账本</a> · <a href="replay_receipt.json">独立重放回执</a></section>
<script>
const input=document.getElementById('raw'), no=document.getElementById('no'), black=document.getElementById('black'), street=document.getElementById('street'), reference=document.getElementById('reference');
function update(){{const raw=input.value.padStart(3,'0');no.textContent=raw;black.src='frames/'+raw+'.png';street.src='frames/'+raw+'_street.jpg';reference.src='frames/'+raw+'_reference.png';}}
input.addEventListener('input',update);document.querySelectorAll('button[data-raw]').forEach(b=>b.addEventListener('click',()=>{{input.value=b.dataset.raw;update();}}));
</script></main>'''
    (dest / 'index.html').write_text(html, encoding='utf-8')
    print(json.dumps(stats, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
