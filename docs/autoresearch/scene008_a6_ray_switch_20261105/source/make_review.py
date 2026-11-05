"""Publish the isolated CUDA support test and its street readout."""
import csv
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent.parent
RAY=ROOT.parent/'scene008_a6_ray_support_20261105'
PREV=ROOT.parent/'scene008_a6_truncated_support_20261105'
FUSION=ROOT.parent/'scene008_a6_local_fusion_20261105'
sys.path.insert(0,str(ROOT.parent/'scene008_a6_residual_replay_20261105/source'))
from report import image_grid, pooled

PILOTS=(105,113,130)
LABELS=dict(previous='上一轮矩匹配高斯',all_incremental='全部残余候选叠加裁剪',incremental_guarded='叠加裁剪 · 保护通过',
            all_switch='全部对应行改用射线裁剪',guarded_switch='通过保护的同位替换')
ARCH='''<svg viewBox="0 0 940 150" role="img" aria-label="车体半空间裁剪数据流">
<defs><marker id="arr" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="#657885"/></marker></defs>
<g fill="none" stroke="#657885" stroke-width="2" marker-end="url(#arr)">
<path d="M192 41H224V74H249"/><path d="M192 111H224V74H249"/><path d="M452 74H487"/><path d="M691 74H722"/>
</g><g stroke="#bdcbd5">
<rect x="10" y="17" width="182" height="49" rx="8" fill="#edf4f8"/><rect x="10" y="86" width="182" height="49" rx="8" fill="#edf4f8"/>
<rect x="249" y="49" width="203" height="50" rx="8" fill="#e5f0eb"/><rect x="487" y="49" width="204" height="50" rx="8" fill="#f8f0e0"/>
<rect x="722" y="49" width="207" height="50" rx="8" fill="#edf4f8"/></g>
<g fill="#172938" font-family="sans-serif" font-size="15" text-anchor="middle">
<text x="101" y="47">固定 PVD 与原生 a6</text><text x="101" y="116">车体坐标的保留侧平面</text>
<text x="350" y="80">积分求射线保留比例</text><text x="589" y="80">原生排序合成与保护</text><text x="825" y="80">黑底对照、街景复核</text>
</g></svg>'''


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    r=load(RAY/'result.json')
    selected=load(ROOT/'result.json')
    scene=load(ROOT/'street.json')
    previous=load(PREV/'result.json')
    validation=load(ROOT/'shader_validation.json')
    assert r['frames']==selected['frames']==len(scene['rows'])==49
    assert validation['noop_maxabs']==0 and scene['prior_street_replay_passed']
    assert all(not row['violations'] for row in selected['rows'])
    values=dict(previous=previous['outcomes'][previous['chosen']]['rows'],
                all_incremental=r['all_cut'],incremental_guarded=r['rows'],
                all_switch=r['same_rows'],guarded_switch=selected['rows'])
    scores={name:pooled(rows) for name,rows in values.items()}
    dest=ROOT/'delivery'
    dest.mkdir(exist_ok=False)
    frames=dest/'frames'
    frames.mkdir()
    for raw in range(89,138):
        with Image.open(FUSION/'render'/f'{raw:03d}_pvd_pvd.png') as alpha:
            y,x=np.nonzero(np.asarray(alpha)[:,:,0]>5)
        bbox=(max(0,int(x.min())-10),max(0,int(y.min())-11),min(480,int(x.max())+11),min(320,int(y.max())+12))
        paths=[ROOT/'render'/f'{raw:03d}_{arm}{kind}.png'
               for kind in ('','_pvd','_blocked') for arm in ('previous','new')]
        labels=['Previous fusion','Guarded ray clipping','Previous PVD alpha','New PVD alpha',
                'Previous hidden contribution','New hidden contribution']
        image_grid(paths,labels,box=bbox,scale=3).save(frames/f'{raw:03d}.png')
        if raw in PILOTS:
            paths=[ROOT/'street'/f'{raw:03d}_{state}_{arm}.png'
                   for state in ('keep','delete') for arm in ('previous','new')]
            image_grid(paths,['No edit | previous','No edit | new','Delete a37 | previous','Delete a37 | new']).save(frames/f'{raw:03d}_street.jpg',quality=91)
    for p in ('result.json','street.json','shader_validation.json','switch.npz'):
        shutil.copyfile(ROOT/p,dest/p)
    shutil.copyfile(RAY/'result.json',dest/'incremental_control.json')
    shutil.copyfile(RAY/'build_provenance.json',dest/'build_provenance.json')
    best=dict(status='PASS',frames=49,training_updates=0,switched=selected['switched'],
              scores=scores,street=scene['pooled'],validation=validation)
    (dest/'summary.json').write_text(json.dumps(best,indent=2,allow_nan=False)+'\n')
    with (dest/'fixed_metrics.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.writer(handle,lineterminator='\n')
        writer.writerow(['raw','variant','lower_reveal','known_rgb','depth_m','outside_rgb_delta','violations'])
        for name,rows in values.items():
            for row in rows:
                rec=pooled([row])
                writer.writerow([row['raw'],name,*[rec[k] for k in ('ratio','known_rgb','measured_depth','outside_rgb_delta')],
                                 '|'.join(row['violations'])])
    table=''
    for name,v in scores.items():
        table+=f"<tr><th>{LABELS[name]}</th><td>{v['ratio']:.2%}</td><td>{v['known_rgb']:.6f}</td><td>{v['measured_depth']:.5f}</td><td>{v['outside_rgb_delta']:.5f}</td><td>{v['failed_frames']}</td></tr>"
    d=scene['pooled']['delete'];k=scene['pooled']['keep']
    html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Scene008 a6 车体半空间裁剪</title>
<style>
body{{background:#f4f8fa;color:#172938;font:16px/1.6 system-ui,sans-serif;margin:0}}
main{{max-width:1090px;margin:auto;padding:26px 22px}}
section{{background:white;border:1px solid #d8e3e8;border-radius:11px;padding:18px;margin:17px 0;overflow:auto}}
svg,img{{width:100%;display:block}}table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}
td,th{{text-align:left;padding:6px 9px;border-bottom:1px solid #dbe4e9}}a{{color:#066471}}input{{width:48%}}
</style><main><h1>Scene008 · a6 车体半空间裁剪</h1>
<p>固定下部黑底显露率 {selected['previous']:.2%} → <strong>{selected['ratio']:.2%}</strong>，通过的同位替换 {selected['switched']:,} 行。训练更新为零。</p>
<section><h2>组件与数据流</h2>{ARCH}
<p>冻结原生 10K、道路 1K 和 PVD。按上一轮的车体坐标平面裁剪原生行，用沿实际相机射线落在保留侧的高斯质量比例缩放 alpha。排序与深度读出沿用原生高斯中心。私有 CUDA 路径关闭裁剪时，49 帧与原生完全一致；独立解析 alpha 最大差 {validation['analytic']['pixel_alpha_error']:.2g}。</p>
<p>先测试在上一轮结果上叠加裁剪；全量候选违反保护，受保护搜索保留了父版。再用同一批原生行和原有平面，逐组把矩匹配近似换成射线裁剪。每帧继续验证已观测 RGB、深度、覆盖、非目标变化和下部显露。</p></section>
<section><h2>49 帧黑底对照</h2><p>左为上一轮，右为本轮。三行依次是 RGB、PVD 贡献、固定下部仍未显露的红色权重。</p>
<label for="raw">raw<span id="no">113</span></label><input id="raw" type="range" min="89" max="137" value="113"><button data-raw="105">105</button><button data-raw="113">113</button><button data-raw="130">130</button>
<img id="frame" src="frames/113.png" alt="raw113 黑底配对"></section>
<section><h2>固定域对照</h2><table><tr><th>版本</th><th>下部显露</th><th>已观测 RGB MAE</th><th>深度 MAE (m)</th><th>非目标 RGB 变化</th><th>保护失败帧</th></tr>{table}</table>
<p>显露率相对于下部 PVD 独立透明度，固定评价范围仍是 26,229 个像素。非目标变化相对 38.3% 的祖先版本计量。隐藏部分缺独立几何真值。</p></section>
<section><h2>街景复核</h2>
<p>删 a37 后的同域显露率 {d['before_fraction']:.2%} → {d['after_fraction']:.2%}，显露道路 MAE {d['road_before']:.6f} → {d['road_after']:.6f}。无编辑的可见 a37 RGB 差为 {k['a37_delta']:.2g}。上排无编辑，下排删 a37，左旧右新。</p>
<img src="frames/113_street.jpg" alt="raw113 街景复核"></section>
<section><h2>逐帧与复现</h2><a href="fixed_metrics.csv">逐帧指标</a> · <a href="result.json">同位替换试跑</a> · <a href="incremental_control.json">叠加裁剪对照</a> · <a href="street.json">49 帧街景</a> · <a href="switch.npz">共享行账本</a></section>
<script>
const raw=document.getElementById('raw'),no=document.getElementById('no'),frame=document.getElementById('frame');
function update(){{const key=raw.value.padStart(3,'0');no.textContent=key;frame.src='frames/'+key+'.png';}}
raw.addEventListener('input',update);document.querySelectorAll('button[data-raw]').forEach(b=>b.addEventListener('click',()=>{{raw.value=b.dataset.raw;update();}}));
</script></main></html>'''
    (dest/'index.html').write_text(html,encoding='utf-8')
    print(json.dumps(best,indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
