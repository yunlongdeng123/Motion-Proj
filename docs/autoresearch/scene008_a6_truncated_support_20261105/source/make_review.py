"""Build the comparable a6 support-split review and preserve the accepted recipe."""
import csv
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image

RUN=Path(__file__).resolve().parent.parent
LATEST=RUN.parent/'scene008_a6_residual_replay_20261105'
FUSION=RUN.parent/'scene008_a6_local_fusion_20261105'
sys.path.insert(0,str(LATEST/'source'))
from report import image_grid, pooled

PILOTS=(105,113,130)
LABEL={'previous':'上一轮共享退役','one_child':'单个保留侧高斯','three_slabs':'三片保留侧高斯'}
ARCH='''<svg viewBox="0 0 930 156" role="img" aria-label="保留侧重表达数据流">
<defs><marker id="end" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="#607383"/></marker></defs>
<g fill="none" stroke="#607383" stroke-width="2" marker-end="url(#end)">
<path d="M175 42H208V77H240"/><path d="M175 113H208V77H240"/><path d="M450 77H486"/><path d="M676 77H710"/>
</g><g stroke="#bdcbd5">
<rect x="10" y="17" width="165" height="50" rx="9" fill="#edf4f8"/><rect x="10" y="88" width="165" height="50" rx="9" fill="#edf4f8"/>
<rect x="240" y="52" width="210" height="50" rx="9" fill="#e3f1ec"/>
<rect x="486" y="52" width="190" height="50" rx="9" fill="#f9f1e0"/><rect x="710" y="52" width="209" height="50" rx="9" fill="#edf4f8"/>
</g><g fill="#172938" font-family="sans-serif" font-size="15" text-anchor="middle">
<text x="92" y="48">固定 PVD 和 a6</text><text x="92" y="119">下部与保护射线</text>
<text x="345" y="83">车体坐标确定保留侧</text><text x="580" y="83">单高斯或三片重表达</text>
<text x="815" y="83">黑底与街景逐帧检查</text></g></svg>'''


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    result=read(RUN/'result.json')
    street=read(RUN/'street.json')
    prior=read(LATEST/'result.json')
    planes=read(RUN/'planes.json')
    chosen=result['chosen']
    assert result['status']=='PASS' and street['status']=='PASS'
    assert result['frames']==len(street['rows'])==49
    variants=dict(previous=prior['result'])
    variants.update({k:v['rows'] for k,v in result['outcomes'].items()})
    stats={k:pooled(rows) for k,rows in variants.items()}
    assert all(not row['violations'] for rows in variants.values() for row in rows)
    folder=RUN/'delivery'
    folder.mkdir(exist_ok=False)
    frames=folder/'frames'
    frames.mkdir()
    for raw in range(89,138):
        with Image.open(FUSION/'render'/f'{raw:03d}_pvd_pvd.png') as alpha:
            y,x=np.nonzero(np.asarray(alpha)[:,:,0]>5)
        box=(max(0,int(x.min())-10),max(0,int(y.min())-11),min(480,int(x.max())+11),min(320,int(y.max())+12))
        paths=[RUN/'render'/f'{raw:03d}_{arm}{kind}.png'
               for kind in ('','_pvd','_blocked') for arm in ('previous','new')]
        labels=['Previous fusion','Accepted re-expression','Previous PVD alpha','New PVD alpha',
                'Previous hidden contribution','New hidden contribution']
        image_grid(paths,labels,box=box,scale=3).save(frames/f'{raw:03d}.png')
        if raw in PILOTS:
            paths=[RUN/'street'/f'{raw:03d}_{state}_{arm}.png'
                   for state in ('keep','delete') for arm in ('previous','new')]
            image_grid(paths,['No edit | previous','No edit | new','Delete a37 | previous','Delete a37 | new']).save(frames/f'{raw:03d}_street.jpg',quality=91)
    for name in ('result.json','street.json','parent_replay.json','planes.json','planes.npz','accepted.npz'):
        shutil.copyfile(RUN/name,folder/name)
    summary=dict(status='PASS',winner=chosen,stats=stats,street=street['pooled'],planes=planes,
                 parent_replay=read(RUN/'parent_replay.json'),training_updates=0,frames=49,
                 frame_tolerance=dict(max_reveal_regression=result['per_frame_reveal_drop_limit'],
                                      outside_reference=result['outside_reference']))
    save(folder/'summary.json',summary)
    with (folder/'fixed_metrics.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.writer(handle,lineterminator='\n')
        writer.writerow(['raw','arm','lower_reveal','observed_rgb_mae','lidar_z_mae_m','outside_rgb_delta','outside_alpha_delta'])
        for name,rows in variants.items():
            for row in rows:
                one=pooled([row])
                writer.writerow([row['raw'],name,*[one[k] for k in ('ratio','known_rgb','measured_depth','outside_rgb_delta','outside_alpha_delta')]])
    tr=''
    for name,m in stats.items():
        count=result['outcomes'][name]['selected'] if name in result['outcomes'] else 0
        tr+=f"<tr><th>{LABEL[name]}</th><td>{m['ratio']:.2%}</td><td>{m['known_rgb']:.6f}</td><td>{m['measured_depth']:.5f}</td><td>{m['outside_rgb_delta']:.5f}</td><td>{count:,}</td></tr>"
    d=street['pooled']['delete'];k=street['pooled']['keep']
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Scene008 a6 保留侧高斯</title>
<style>
body{{background:#f3f7f9;color:#172938;font:16px/1.6 system-ui,sans-serif;margin:0}}
main{{max-width:1070px;margin:auto;padding:24px}} section{{padding:18px;border:1px solid #d9e3e9;border-radius:11px;background:white;margin:17px 0;overflow:auto}}
svg,img{{width:100%;display:block}} table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}
td,th{{padding:6px 9px;border-bottom:1px solid #dde6eb;text-align:left}}a{{color:#05616b}}input{{width:50%}}
</style><main><h1>Scene008 · a6 保留侧高斯</h1>
<p>固定 PVD、轨迹、道路和上一轮已通过的共享门控。黑底主实验中，采用 <strong>{LABEL[chosen]}</strong>，下部显露率 {stats['previous']['ratio']:.2%} → {stats[chosen]['ratio']:.2%}。训练更新为零。</p>
<section><h2>组件与数据流</h2>{ARCH}
<p>按下部与受保护足迹归因，从车体坐标确定同一批保留侧平面。比较单个矩匹配高斯与三个条件区间高斯。三片版用条件质量与轴向方差设定强度。两个版本使用同一候选池，各自在同一保护检查下选择通过的行。</p>
<p>沿用上一轮的 RGB、深度、覆盖和非目标保护。非目标变化继续相对 38.3% 版本计量，限制跨轮累积偏移。固定下部评价域继续使用原来的 26,229 个像素。另限制单帧显露率相对上一轮最多下降 1 个百分点。</p></section>
<section><h2>49 帧黑底对照</h2><p>左为上一轮，右为本轮通过保护的版本。依次为 RGB、PVD 可见贡献和仍未显露的红色权重。</p>
<label for="raw">raw<span id="no">113</span></label><input type="range" id="raw" min="89" max="137" value="113"><button data-raw="105">105</button><button data-raw="113">113</button><button data-raw="130">130</button>
<img id="frame" src="frames/113.png" alt="raw113 黑底对照"></section>
<section><h2>同域指标</h2><table><tr><th>版本</th><th>下部显露</th><th>已观测 RGB MAE</th><th>LiDAR 深度 MAE (m)</th><th>相对 38.3% 版的非目标 RGB 变化</th><th>登记行</th></tr>{tr}</table>
<p>各组选中版本均通过黑底逐帧保护。下部显露是联合贡献相对于固定 PVD 独立透明度的比例。未观测区域仍缺独立几何真值。</p></section>
<section><h2>49 帧街景复核</h2>
<p>删 a37 后，显露率 {d['before_fraction']:.2%} → {d['after_fraction']:.2%}，显露道路 RGB MAE {d['road_before']:.6f} → {d['road_after']:.6f}。无编辑时可见 a37 配对 RGB 差为 {k['a37_delta']:.2g}。两种状态共用同一套 a6 资产。</p>
<p>左为上一轮，右为本轮。上排无编辑，下排删 a37。</p>
<img src="frames/113_street.jpg" alt="raw113 街景复核"></section>
<section><h2>逐帧与复现</h2><a href="fixed_metrics.csv">逐帧指标</a> · <a href="result.json">搜索与保护回执</a> · <a href="street.json">街景逐帧</a> · <a href="planes.npz">共享车体平面</a> · <a href="accepted.npz">通过的重表达行</a></section>
<script>
const raw=document.getElementById('raw'),no=document.getElementById('no'),frame=document.getElementById('frame');
function update(){{const key=raw.value.padStart(3,'0');no.textContent=key;frame.src='frames/'+key+'.png';}}
raw.addEventListener('input',update);document.querySelectorAll('button[data-raw]').forEach(b=>b.addEventListener('click',()=>{{raw.value=b.dataset.raw;update();}}));
</script></main></html>'''
    (folder/'index.html').write_text(page,encoding='utf-8')
    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
