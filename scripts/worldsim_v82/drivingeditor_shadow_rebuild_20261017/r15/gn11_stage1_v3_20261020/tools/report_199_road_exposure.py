"""把真实199审计导出为固定单元配对表、原图裁片和来源拒绝页；不训练。"""

import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np

from prepare_multiview_road import read
from road_source_audit import GATES


STYLE = '''body{background:#11171d;color:#e6edf3;font:17px system-ui;margin:28px;max-width:1450px}a{color:#75c5ff}h1{font-size:28px}p{line-height:1.6}table{border-collapse:collapse;width:100%}td,th{padding:10px;text-align:left;border-bottom:1px solid #34424d}small,.muted{color:#aab9c5}.ok{color:#4ed6a1}.no{color:#f39494}select{padding:7px;background:#25323d;color:#fff;border:1px solid #608095}.card{padding:16px;border:1px solid #466071;border-radius:10px;margin:18px 0}.crop{position:relative;width:104px;height:72px;overflow:hidden;border:1px solid #638396}.crop img{position:absolute;width:480px;height:320px}.cross{position:absolute;left:46px;top:27px;color:#ff4872;font:bold 18px monospace}svg{width:100%;height:auto}svg text{font:16px system-ui;fill:#e6edf3}svg rect{fill:#203748;stroke:#5688a8}'''

ARCH = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1160 105"><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0 0L0 6L7 3" fill="#aaa"/></marker></defs><g fill="#203748" stroke="#5688a8"><rect x="5" y="10" width="235" height="80" rx="8"/><rect x="300" y="10" width="245" height="80" rx="8"/><rect x="605" y="10" width="245" height="80" rx="8"/><rect x="910" y="10" width="245" height="80" rx="8"/></g><g fill="#e6edf3" font-family="sans-serif" font-size="17"><text x="20" y="42">原始同场景 raw0–198</text><text x="20" y="69">相机 / LiDAR / 标注 / 位姿</text><text x="315" y="42">固定49帧世界道路单元</text><text x="315" y="69">同口径几何与遮挡筛选</text><text x="620" y="42">49 与 199 配对曝光计数</text><text x="620" y="69">零 / 单来源 / 多视角</text><text x="925" y="42">来源裁片 / 拒绝原因 / 表格</text><text x="925" y="69">仅审计，不接训练输入</text></g><path d="M240 50H295 M545 50H600 M850 50H905" stroke="#aaa" marker-end="url(#arrow)"/></svg>'''

DETAIL = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>199帧固定道路单元审计</title><style>__STYLE__</style><h1>raw89 / a__ACTOR__：同一世界道路单元的 199 帧曝光</h1><p><a href="index.html">返回审计汇总</a>　筛选 <select id="filter"><option value="new">新增确认</option><option value="all">全部单元</option><option value="zero">199仍零来源</option><option value="single">199仅单来源</option></select>　单元 <select id="unit"></select></p><div class="card" id="meta"></div><p class="muted">硬通过仅表示当前几何审计口径；不是语义GT、纹理质量通过或训练贡献。所有训练权重、访问、梯度和因果帮助均未执行。可切换全类别标注包络，查看实际遮挡筛选范围。</p><p><label><input id="mask" type="checkbox">显示标注遮挡包络</label>　<label><input id="passed" type="checkbox" checked>只显示通过来源</label>　<span id="n"></span></p><table><thead><tr><th>来源 / 原图裁片</th><th>窗口 / 验证结果</th><th>投影 / 测量深度 / 原RGB</th><th>CPU几何候选权重</th></tr></thead><tbody id="rows"></tbody></table><script>
const D=__DATA__,f=document.getElementById('filter'),u=document.getElementById('unit');
function select(){const list=D.units.filter(x=>f.value==='all'||(f.value==='new'&&x.before<2&&x.after>=2)||(f.value==='zero'&&x.after===0)||(f.value==='single'&&x.after===1));u.replaceChildren();for(const x of list){const o=document.createElement('option');o.value=x.id;o.textContent='X'+x.id+' · '+x.before+' → '+x.after;u.append(o)}show()}
function show(){const x=D.units.find(x=>x.id===Number(u.value));if(!x){document.getElementById('meta').textContent='此筛选下没有单元';document.getElementById('rows').replaceChildren();return}document.getElementById('meta').textContent='世界坐标 [m] '+x.world.map(v=>v.toFixed(3)).join(', ')+'；原49通过 '+x.before+'，新增150通过 '+(x.after-x.before)+'，199合计 '+x.after+'；固定曲面支持 '+x.support+'；原V3/SAM49来源数 '+x.native+'（不同口径，仅供参考）';let n=0;document.getElementById('rows').innerHTML=x.sources.map((s,raw)=>{const ok=s[2]===511;if(document.getElementById('passed').checked&&!ok)return '';n++;const fail=D.gates.filter((g,i)=>!(s[2]&(1<<i)));const image=document.getElementById('mask').checked?raw+'_occluder.png':raw+'.jpg';return '<tr><td>raw'+raw+'<div class="crop"><img src="frames/'+image+'" style="left:'+(52-s[0])+'px;top:'+(36-s[1])+'px"><span class="cross">+</span></div></td><td class="'+(ok?'ok':'no')+'">'+(raw>=89&&raw<=137?'原49':'新增150')+'<br>'+(ok?'几何通过':fail.join(' / '))+'</td><td>uv '+s[0]+', '+s[1]+'<br>LiDAR '+s[3]+' m<br>RGB '+s.slice(4,7).join(', ')+'</td><td>'+s[7].toExponential(3)+'<br><small>未进入训练</small></td></tr>'}).join('');document.getElementById('n').textContent='显示 '+n+' / 199 来源'}f.onchange=select;u.onchange=show;document.getElementById('mask').onchange=show;document.getElementById('passed').onchange=show;if(!D.units.some(x=>x.before<2&&x.after>=2))f.value='all';select();</script></html>'''


def build(audit_path, output):
    if output.exists():
        raise FileExistsError('preserve earlier report')
    report = read(audit_path/'audit.json')
    meta = read(Path(report['input_bundle']))
    with np.load(Path(report['input_bundle']).parent/meta['units'], allow_pickle=False) as data:
        units = {k:data[k] for k in ['world', 'surface_supported', 'surface_distance_m', 'is_query', 'query_frame', 'query_actor', 'query_unit_id', 'source_count']}
    with np.load(audit_path/'exposure.npz', allow_pickle=False) as data:
        evidence = {k:data[k] for k in data.files}
    output.mkdir(parents=True)
    shutil.copytree(audit_path/'frames', output/'frames')
    shutil.copy2(audit_path/'audit.json', output/'audit.json')
    (output/'architecture.svg').write_text(ARCH, encoding='utf-8')
    before, extra = evidence['original49_count'], evidence['added150_count']
    after = before+extra
    with (output/'fixed_query_units.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['unit_id', 'world_x_m', 'world_y_m', 'world_z_m', 'fixed_surface_supported',
                         'native49_reference_sources', 'paired49_sources', 'added150_sources', 'paired199_sources', 'newly_confirmed'])
        for uid in np.flatnonzero(units['is_query']):
            writer.writerow([uid, *units['world'][uid], units['surface_supported'][uid], units['source_count'][uid],
                             before[uid], extra[uid], after[uid], before[uid] < 2 <= after[uid]])
    for actor in [6, 37]:
        ids = np.unique(units['query_unit_id'][(units['query_frame'] == 0) & (units['query_actor'] == actor)])
        rows = []
        for uid in ids:
            # 列顺序：uv, bits, depth, RGB, CPU base weight；不伪造优化值。
            sources = [[*[round(float(v), 2) for v in evidence['source_uv'][raw, uid]],
                        int(evidence['gate_bits'][raw, uid]), round(float(evidence['measured_depth_m'][raw, uid]), 3),
                        *[round(float(v), 4) for v in evidence['source_rgb'][raw, uid]],
                        float(format(float(evidence['base_weight'][raw, uid]), '.6g'))] for raw in range(199)]
            rows.append(dict(id=int(uid), world=units['world'][uid].tolist(), support=bool(units['surface_supported'][uid]),
                             distance=float(units['surface_distance_m'][uid]),
                             native=int(units['source_count'][uid]), before=int(before[uid]), after=int(after[uid]), sources=sources))
        payload = json.dumps(dict(gates=GATES, units=rows), separators=(',', ':'), allow_nan=False)
        detail = DETAIL.replace("+'；固定曲面支持 '+x.support", "+'；最近LiDAR曲面支撑XY距离 '+x.distance.toFixed(3)+' m；固定曲面支持 '+x.support")
        (output/f'raw89_a{actor}.html').write_text(detail.replace('__STYLE__', STYLE).replace('__ACTOR__', str(actor)).replace('__DATA__', payload), encoding='utf-8')
    table = ''
    for name, row in report['summaries'].items():
        a, b = row['original49'], row['all199']
        label = f'<a href="{name}.html">{name}</a>' if name in ['raw89_a6', 'raw89_a37'] else name
        table += f'<tr><td>{label}</td><td>{row["world_units"]}</td><td>{a["confirmed"]} → {b["confirmed"]}</td><td>{row["newly_confirmed"]}</td><td>{row["zero_to_confirmed"]} / {row["single_to_confirmed"]}</td><td>{b["zero_source_supported"]} / {b["single_source_unconfirmed"]} / {b["unsupported"]}</td><td>{row["native49_reference_confirmed"]}</td></tr>'
    source_rows = ''.join(f'<tr><td>raw{x["raw"]}</td><td>{"原49" if x["original49"] else "新增150"}</td><td>{x["source_unit_pairs"]}</td><td>{x["quality"]:.3f}</td></tr>' for x in report['frames'])
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>V3 199帧道路曝光审计</title><style>{STYLE}</style><h1>V3：199 帧道路曝光审计（无训练）</h1>{ARCH}<p>真实范围 raw0–198；原窗口 raw89–137。固定26,088个世界单元，查询单元13,720个；不新增曲面、法线或查询，不融合RGB。49帧原图、位姿、标定、LiDAR及时间戳全部与原准备数据核对通过。</p><div class="card">V1基线：49/无额外道路参考；V3主实验：49/49保持不变；本次：不训练/199仅审计；V3扩展49/199与完整199/199训练暂缓。GPU使用0，训练更新0，审计不接入主实验。</div><p>配对两列均使用原始全类别相机框＋投影标签框＋3D框投影包络，另加三维射线遮挡、稀疏LiDAR道路/深度、角度/分辨率/质量门限。与原V3的SAM掩码不同，因此原V3确认数仅单列参考，不拿不同筛选口径直接计算增益。确认需至少2帧；单来源、无曲面支持仍不能确认。</p><table><thead><tr><th>固定区域</th><th>单元数</th><th>同口径49 → 199确认</th><th>新增确认</th><th>零→确认 / 单→确认</th><th>199零来源 / 单来源 / 无曲面支持</th><th>原V3/SAM49参考</th></tr></thead><tbody>{table}</tbody></table><p><a href="raw89_a6.html">逐单元查看 a6</a> · <a href="raw89_a37.html">逐单元查看 a37</a> · <a href="fixed_query_units.csv">完整固定查询单元CSV</a> · <a href="audit.json">真实数据与口径记录</a></p><p class="muted">这是几何证据审计，不证明真实隐藏纹理已恢复；标注不全、稀疏测量和固定曲面误差仍可能影响筛选。零来源不等于永久不可见。尚无可靠性更新、训练访问、梯度或去来源因果结果。human_verdict=null；quality_accepted=false。</p><h2>逐帧候选流量（全部固定单元，含上下文）</h2><table><thead><tr><th>帧</th><th>窗口</th><th>通过的来源–单元对</th><th>固定49清晰度基准权重</th></tr></thead><tbody>{source_rows}</tbody></table></html>'''
    (output/'index.html').write_text(page, encoding='utf-8')
    return report['summaries']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.audit, args.output), indent=2))
