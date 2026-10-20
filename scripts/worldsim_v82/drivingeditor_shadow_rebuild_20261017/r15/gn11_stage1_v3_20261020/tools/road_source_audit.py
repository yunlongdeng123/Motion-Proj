"""把每个世界道路单元的真实来源/优化记录导出为离线 HTML；不运行训练。"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

GATES = ['in_view', 'surface_supported', 'mask_clear', 'road_support', 'measured_depth_match',
         'boxes_clear', 'angle_ok', 'resolution_ok', 'quality_ok']


HTML = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>道路世界单元来源审计</title>
<style>body{background:#101214;color:#eee;font:17px system-ui;margin:28px}h1{font-size:28px}small,.muted{color:#aab3bd}select{background:#222b33;color:white;padding:8px;border:1px solid #506070}table{border-collapse:collapse;width:100%;margin-top:20px}td,th{padding:10px;text-align:left;border-bottom:1px solid #343a40}.ok{color:#42d49b}.no{color:#fa7474}.bar{display:inline-block;background:#279d7c;height:10px;margin-right:10px}.crop{position:relative;width:84px;height:62px;overflow:hidden;border:1px solid #657080}.crop img{position:absolute;width:480px;height:320px}.cross{position:absolute;left:38px;top:26px;color:#ff4060;font:bold 14px monospace}.card{border:1px solid #46535e;border-radius:12px;padding:18px;margin:16px 0}svg text{fill:#ddd;font:14px system-ui}svg rect{fill:#1d3343;stroke:#5486a6}</style>
<h1>一个道路查询单元 X 的来源审计</h1><p>查询：raw__RAW__ / a__ACTOR__　世界单元 <select id="unit"></select></p>
<svg viewBox="0 0 1000 75" width="100%"><defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0 0L0 6L7 3" fill="#aaa"/></marker></defs><rect x="0" y="10" width="205" height="48" rx="8"/><text x="12" y="40">相机 / LiDAR / 五车掩码</text><rect x="250" y="10" width="210" height="48" rx="8"/><text x="262" y="40">永久世界单元 + 49帧原RGB</text><rect x="505" y="10" width="210" height="48" rx="8"/><text x="517" y="40">共享薄层道路高斯优化</text><rect x="760" y="10" width="235" height="48" rx="8"/><text x="772" y="40">正常场景渲染 / 来源与反事实</text><path d="M205 34H245 M460 34H500 M715 34H755" stroke="#aaa" marker-end="url(#a)"/></svg>
<div id="metadata" class="card"></div>
<p class="muted">硬几何通过 ≠ 已参与优化 ≠ 因果帮助。CPU 计划占比不是实测训练权重；梯度只说明连接和量级，需配对去来源实验才可讨论帮助。</p>
<table><thead><tr><th>来源帧 / 原图裁片</th><th>几何验证</th><th>CPU候选占比</th><th>训练可靠性 / 单元内有效占比</th><th>实际RGB损失访问</th><th>纹理DC采样梯度L2 / 次数</th></tr></thead><tbody id="rows"></tbody></table>
<p class="muted">裁片中心为该世界单元在来源帧中的投影，不是其他位置供体。道路支持为稀疏测量+曲面代理，不是密集语义真值。未确认曝光的先验不显示为恢复的真实纹理。</p>
<script>const D=__DATA__;const unit=document.getElementById('unit');for(const x of D.units){const o=document.createElement('option');o.value=x.id;o.textContent='X'+x.id+' · '+['UNKNOWN','真实多视角','连续先验'][x.provenance];unit.append(o)}
function show(){const x=D.units.find(v=>v.id===Number(unit.value));document.getElementById('metadata').innerHTML='世界坐标 [m]：'+x.world.map(v=>v.toFixed(3)).join(', ')+'<br>最近曲面支撑 XY 距离：'+x.distance.toFixed(3)+' m；通过来源数：'+x.source_count+'；支持：'+x.support+'<br>优化记录：'+D.optimization_status+'；完整优化更新：'+D.optimizer_updates+'<br>颜色：'+(x.color?x.color.map(v=>v.toFixed(4)).join(', '):'尚无优化结果')+'<br>注意：单来源候选保持 UNKNOWN；一个单元的参数也可能收到相邻单元观测经 splat 重叠传来的梯度。';
document.getElementById('rows').innerHTML=x.sources.map(s=>{const failed=D.gates.filter((g,i)=>!(s[6]&(1<<i)));const ok=failed.length===0;const crop='<div class="crop"><img src="frames/'+s[0]+'.jpg" style="left:'+(42-s[1])+'px;top:'+(31-s[2])+'px"><span class="cross">+</span></div>';return '<tr><td>raw'+s[0]+crop+'</td><td class="'+(ok?'ok':'no')+'">'+(ok?'几何通过':'拒绝：'+failed.join(' / '))+'<br><small>uv '+s[1].toFixed(2)+', '+s[2].toFixed(2)+'；原RGB '+s.slice(3,6).map(v=>v.toFixed(3)).join(', ')+'</small></td><td><span class="bar" style="width:'+(100*s[7])+'px"></span>'+s[7].toFixed(4)+'</td><td>'+(s[8]===null?'尚未执行':s[8].toFixed(4)+' / '+s[9].toFixed(4))+'</td><td>'+(s[10]===null?'尚未执行':s[10])+'</td><td>'+(s[11]===null?'尚未测量':s[11].toExponential(3)+' / '+s[12])+'</td></tr>'}).join('')}
const preferred=D.units.filter(x=>x.provenance===1).sort((a,b)=>b.source_count-a.source_count)[0];if(preferred)unit.value=String(preferred.id);unit.onchange=show;show();</script></html>'''


def build(bundle_path, output, raw=89, actor=37, optimization=None):
    if output.exists():
        raise FileExistsError('preserve prior audit; use a new output')
    meta = json.loads(bundle_path.read_text(encoding='utf-8'))
    with np.load(bundle_path.parent/meta['units'], allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    selected = np.unique(data['query_unit_id'][(data['query_frame'] == raw-89) & (data['query_actor'] == actor)])
    if not len(selected):
        raise ValueError('no query world units for this frame and actor')
    state, opt_meta, color_map = None, {}, {}
    if optimization is not None:
        opt_meta = json.loads((optimization/'metadata.json').read_text())
        if Path(opt_meta['input_bundle']).resolve() != bundle_path.resolve():
            raise ValueError('optimization belongs to another world-unit bundle')
        with np.load(optimization/'optimization.npz', allow_pickle=False) as archive:
            state = {key: archive[key].copy() for key in archive.files}
        color_map = dict(zip(state['unit_ids'], state['color']))
    rows = []
    for index in selected:
        base = data['base_weight'][:, index]
        possible = base/max(base.sum(), 1e-12)
        effective = None if state is None else state['effective_rgb_weight'][:, index].copy()
        if effective is not None:
            effective = effective/max(effective.sum(), 1e-12)
        sources = []
        for frame in range(49):
            bits = sum(int(data['gate_'+gate][frame, index]) << bit for bit, gate in enumerate(GATES))
            sources.append([89+frame, *np.round(data['source_uv'][frame, index].astype(float), 3).tolist(),
                            *np.round(data['source_rgb'][frame, index].astype(float), 5).tolist(), bits, float(possible[frame]),
                            None if state is None else float(state['reliability'][frame, index]),
                            None if state is None else float(effective[frame]),
                            None if state is None else int(state['objective_visits'][frame, index]),
                            None if state is None else float(state['gradient_l2_cumulative'][frame, index]),
                            None if state is None else int(state['gradient_samples'][frame, index]),
                            None if state is None else float(state['effective_rgb_weight'][frame, index])])
        rows.append(dict(id=int(index), world=data['world'][index].tolist(),
                         distance=float(data['surface_distance_m'][index]), support=bool(data['surface_supported'][index]),
                         source_count=int(data['source_count'][index]), provenance=int(data['provenance'][index]),
                         color=color_map[index].tolist() if index in color_map else None, sources=sources))
    result = dict(gates=GATES, units=rows, optimization_status='尚未训练 / 权重仅CPU候选' if state is None else '实际优化记录；不是因果证明',
                  optimizer_updates=int(opt_meta.get('optimizer_updates', 0)))
    output.mkdir(parents=True)
    (output/'frames').mkdir()
    for fid, frame in meta['frames'].items():
        Image.open(frame['original_rgb']).convert('RGB').resize((480, 320), Image.Resampling.LANCZOS).save(output/'frames'/f'{int(fid[-3:])}.jpg', quality=90)
    payload = json.dumps(result, ensure_ascii=False, separators=(',', ':'), allow_nan=False).replace('</', '<\\/')
    (output/'index.html').write_text(HTML.replace('__RAW__', str(raw)).replace('__ACTOR__', str(actor)).replace('__DATA__', payload), encoding='utf-8')
    summary = dict(query_raw=raw, actor=actor, world_units=len(rows), input_bundle=str(bundle_path),
                   optimizer_updates=result['optimizer_updates'], actual_optimization_loaded=state is not None,
                   observed=sum(row['provenance'] == 1 for row in rows), prior=sum(row['provenance'] == 2 for row in rows),
                   unknown=sum(row['provenance'] == 0 for row in rows), fabricated_weights=False,
                   causal_help_proven=False, quality_accepted=False)
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--query-raw', type=int, default=89)
    parser.add_argument('--actor', type=int, default=37)
    parser.add_argument('--optimization', type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.bundle, args.output, args.query_raw, args.actor, args.optimization), indent=2))
