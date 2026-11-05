"""Publish the heuristic design, matched tests and evidence audit."""
import csv
import json
from pathlib import Path
import shutil
import sys

import numpy as np

RUN=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(RUN.parent/'scene008_a6_residual_replay_20261105/source'))
from report import image_grid


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def main():
    evidence=read(RUN/'evidence_audit.json')
    real=read(RUN/'rescue_rgb_evaluation.json')
    positive=read(RUN/'positive_rgb_evaluation.json')
    beams=read(RUN/'first_return/summary.json')
    proof=read(RUN/'verification.json')
    proximity=read(RUN/'beam_anchor_proximity.json')
    assert proof['status']=='EXECUTION_PASS' and not proof['method_accepted']
    assert real['frames']==positive['frames']==49
    target=evidence['pooled']['all']
    early=read(RUN/'build_evidence.json')
    frame_hits=sum(v['witness_rays'] for v in beams['rows'])
    summary=dict(status='EXECUTION_PASS',method_accepted=False,raw_lidar_frames=199,
          legacy_free=target['counts']['bank_free'],legacy_free_direct_through=target['counts']['bank_free_reviewed_through'],
          lower_pixels=target['counts']['fixed_lower'],lower_measured_occlusion=target['fixed_lower_overlap']['occluded'],
          lower_unknown=target['fixed_lower_overlap']['unknown'],first_return_witnesses=frame_hits,
          build_beam_witnesses=proximity['build_beams'],rescue_proxy_rows=real['proxy_rows'],
          positive_proxy_rows=positive['proxy_rows'],build_measured_matches=sum(v['measured_matches'] for v in early['views']),
          positive_held={arm:positive['stats']['held'][arm]['black'] for arm in positive['arms']},
          actual_held={arm:real['stats']['held'][arm]['black'] for arm in real['arms']})
    d=RUN/'delivery'
    d.mkdir()
    images=d/'frames'
    images.mkdir()
    labels=['Parent','Same cut','Proxy only','Cut + proxy']
    for raw in range(89,138):
        source=RUN/'rescue_rgb_render'
        paths=[source/f'{raw:03d}_black_{arm}.png' for arm in real['arms']]
        image_grid(paths,labels,box=(180,160,340,245),scale=2,columns=4).save(images/f'{raw:03d}_actual.png')
        paths=[source/f'{raw:03d}_{state}_{arm}.png' for state in ('keep','delete') for arm in real['arms']]
        image_grid(paths,[f'{state} | {arm}' for state in ('No edit','Delete a37') for arm in labels],columns=4).save(
            images/f'{raw:03d}_actual_street.jpg',quality=90)
    for raw in positive['held']:
        source=RUN/'positive_rgb_render'
        paths=[source/f'{raw:03d}_black_{arm}.png' for arm in positive['arms']]
        image_grid(paths,labels,box=(180,160,340,245),scale=2,columns=4).save(images/f'{raw:03d}_positive.png')
        paths=[source/f'{raw:03d}_{state}_{arm}.png' for state in ('keep','delete') for arm in positive['arms']]
        image_grid(paths,[f'{state} | {arm}' for state in ('No edit','Delete a37') for arm in labels],columns=4).save(
            images/f'{raw:03d}_positive_street.jpg',quality=90)
    for name in ('contract.json','evidence_audit.json','build_evidence.json','positive_build.json',
                 'positive_contract.json','beam_anchor_proximity.json','verification.json',
                 'rescue_rgb_evaluation.json','positive_rgb_evaluation.json','rescue_ledger.npz','positive_ledger.npz'):
        shutil.copyfile(RUN/name,d/name)
    shutil.copyfile(RUN/'first_return/summary.json',d/'first_return_summary.json')
    shutil.make_archive(str(d/'first_return_witnesses'),'zip',RUN/'first_return/witnesses')
    (d/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    with (d/'fixed_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f,lineterminator='\n')
        writer.writerow(['experiment','raw','held','state','arm','metric','n','sum','mean','failures'])
        for data in (real,positive):
            for r in data['rows']:
                for state,arms in r['states'].items():
                    for arm,v in arms.items():
                        for k,m in v['metrics'].items():
                            writer.writerow([data['mode'],r['raw'],r['held'],state,arm,k,m['count'],m['sum'],
                                             m['sum']/m['count'] if m['count'] else '', '|'.join(v['failures'])])
    def number(x):
        return '—' if x is None else f'{x:.5f}'
    table=''
    for arm,label in zip(positive['arms'],('原资产','同一裁剪','仅加代理面','裁剪与代理面')):
        m=positive['stats']['held'][arm]['black']
        table+=f'<tr><th>{label}</th><td>{number(m["panel_rgb"]["mean"])}</td><td>{number(m["panel_depth"]["mean"])}</td><td>{m["frame_failures"]} / 9</td></tr>'
    main_table=''
    for arm,label in zip(real['arms'],('原资产','同一裁剪','仅加代理面','裁剪与代理面')):
        m=real['stats']['all'][arm]['black']
        v=real['stats']['all'][arm]['delete']['reveal']
        main_table+=f'<tr><th>{label}</th><td>{v:.2%}</td><td>{number(m["known_rgb"]["mean"])}</td><td>{m["frame_failures"]} / 49</td></tr>'
    buttons=''.join(f'<button data-positive="{r}">{r}</button>' for r in positive['held'])
    html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Scene008 启发式接管验证</title>
<style>
body{{background:#f3f7fa;color:#172938;font:16px/1.6 system-ui,sans-serif;margin:0}}main{{max-width:1200px;margin:auto;padding:24px}}
section{{background:#fff;border:1px solid #d6e2e8;border-radius:10px;padding:18px;margin:16px 0;overflow:auto}}
svg,img{{width:100%;display:block}}table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}th,td{{text-align:left;padding:7px 10px;border-bottom:1px solid #dde7ec}}a{{color:#086675}}input{{width:44%}}.finding{{border-left:4px solid #b66532;padding-left:12px}}
</style><main><h1>Scene008 · 启发式接管验证</h1>
<p class="finding">首轮固定对照已经完成。当前的独立叠加式实测代理面未通过保护验收。场景资产采用前序已验证版本。</p>
<section><h2>组件与数据流</h2>
<svg viewBox="0 0 1100 156" role="img" aria-label="观测证据、规则、接管与评价">
<defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0 0L8 3L0 6Z" fill="#647787"/></marker></defs>
<g fill="none" stroke="#647787" stroke-width="2" marker-end="url(#a)"><path d="M186 40H212V76H239"/><path d="M186 114H212V76H239"/><path d="M438 76H473"/><path d="M674 76H710"/><path d="M882 76H914"/></g>
<g stroke="#bdcbd4"><rect x="10" y="15" width="176" height="49" rx="8" fill="#edf4f8"/><rect x="10" y="89" width="176" height="49" rx="8" fill="#edf4f8"/>
<rect x="239" y="51" width="199" height="49" rx="8" fill="#e1f0e9"/><rect x="473" y="51" width="201" height="49" rx="8" fill="#f6efdf"/><rect x="710" y="51" width="172" height="49" rx="8" fill="#edf4f8"/><rect x="914" y="51" width="176" height="49" rx="8" fill="#edf4f8"/></g>
<g fill="#172938" font-family="sans-serif" font-size="15" text-anchor="middle"><text x="98" y="46">原始 RGB、测距与轨迹</text><text x="98" y="120">固定 PVD 与旧场景</text><text x="338" y="82">实测、自由线段、未知</text><text x="573" y="82">保留、裁剪、代理面</text><text x="796" y="82">同一次深度合成</text><text x="1002" y="82">构建与留出验收</text></g></svg></section>
<section><h2>固定实验设计</h2>
<table><tr><th>实验</th><th>固定干预</th><th>检查</th></tr>
<tr><th>E0 证据分层</th><td>分离膨胀框外射线、测距穿越、遮挡与未知</td><td>规则能否获得独立的可退役证据</td></tr>
<tr><th>E1 实际下部</th><td>同一份车体体积统一裁剪 25%</td><td>原资产、裁剪、仅加代理面、裁剪加代理面</td></tr>
<tr><th>E2 实测车尾面正对照</th><td>按构建帧几何选定面板，统一裁剪 60%</td><td>固定同一裁剪，检验实测代理面能否修复损伤</td></tr>
<tr><th>E3 原始第一回波</th><td>raw0–198，实际传感器原点，到测量终点前 15 cm</td><td>保存自由线段及传感器、时刻和像素来源</td></tr></table>
<p>构建帧排除 9 个留出视角。代理面的锚点、颜色和强度只使用构建帧。父模型、PVD、固定下部范围和体积含历史全场景信息，因此该划分仅检验新增代理面构建。新增训练步数为零。</p></section>
<section><h2>证据审计</h2>
<p>原观测库的 {summary['legacy_free']:,} 个 free 像素全部来自膨胀车辆框外，测距穿越项为 {summary['legacy_free_direct_through']}。固定下部 {summary['lower_pixels']:,} 个像素里，{summary['lower_measured_occlusion']:,} 个有前方 a37 的实测遮挡，{summary['lower_unknown']:,} 个对 a6 为未知。</p>
<p>原始 199 帧第一回波得到 {frame_hits:,} 条候选体积内自由线段，其中 {summary['build_beam_witnesses']} 条来自构建集。线段命中覆盖了 19 个原体积分区，收窄到标注车体下部后覆盖 11 个分区。线段不证明整个分区为空。</p>
<p>在 3 cm 邻近探针下，431 个下部 PVD 中有 3 个点接近至少两个相隔三帧的构建自由线段。2,176 个构建来源实测 RGB 锚点没有同类冲突。该距离探针保留了物体扫描内运动和位姿误差的不确定性。</p></section>
<section><h2>E1 实际下部结果</h2>
<p>{summary['build_measured_matches']:,} 次构建投影通过实测核对，选出代理面 {summary['rescue_proxy_rows']} 个。直接裁剪没有在实测采样处触发覆盖或 RGB 损伤规则，损伤主要出现在未获得这种实测支持的其他区域。</p>
<table><tr><th>对照</th><th>删车街景显露</th><th>已观测 a6 黑底 RGB MAE</th><th>黑底保护失败</th></tr>{main_table}</table>
<p>显露继续按固定 PVD 的可见贡献计量，隐藏区缺独立真值。</p>
<label for="raw">raw<span id="rawno">113</span></label><input id="raw" type="range" min="89" max="137" value="113"><button data-raw="105">105</button><button data-raw="113">113</button><button data-raw="130">130</button>
<img id="actual" src="frames/113_actual.png" alt="实际下部四组黑底"><img id="actualstreet" src="frames/113_actual_street.jpg" alt="实际下部两种街景"></section>
<section><h2>E2 实测面正对照</h2>
<p>覆盖率损失触发未选出代理面。加入独立的真实 RGB 损伤触发后，得到 {summary['positive_proxy_rows']} 个构建来源代理面。每个代理面至少有两个构建视角作证，强度由单步颜色插入估计，最高 0.35。</p>
<table><tr><th>对照</th><th>留出面板 RGB MAE</th><th>留出面板深度 MAE (m)</th><th>黑底保护失败</th></tr>{table}</table>
<p class="finding">代理面加入后的留出 RGB 和深度均恶化，仅加代理面组也有损伤。独立叠加不能作为安全接管机制。</p>
<div>{buttons}</div><img id="positive" src="frames/113_positive.png" alt="实测面正对照黑底"><img id="positivestreet" src="frames/113_positive_street.jpg" alt="实测面正对照街景"></section>
<section><h2>下一步固定验证</h2>
<p>把实测表面分组与旧支撑退出绑定。先为同一组实测锚点登记对应的旧支撑，再共同分配原生与代理面的覆盖预算，检验同一退休量下的原始 RGB、深度及局部尾部损伤。未知区域继续单列。第一回波线段只为邻近支撑提供证据，跨视角一致性和实测保护仍需同时成立。</p>
<p>当这组耦合启发式通过无编辑与删车的留出检查后，再比较学习型动作预测。</p></section>
<section><h2>复现</h2><p>全 49 帧完成黑底和两种街景评价。独立打开保存的账本后，raw105、113、130 的四组及三种状态重放与记录精确一致。</p><a href="contract.json">实验合同</a> · <a href="fixed_metrics.csv">全部逐帧指标</a> · <a href="rescue_rgb_evaluation.json">实际下部评价</a> · <a href="positive_rgb_evaluation.json">正对照评价</a> · <a href="first_return_summary.json">第一回波核查</a> · <a href="first_return_witnesses.zip">自由线段来源</a> · <a href="verification.json">独立重放</a></section>
<script>
const slider=document.getElementById('raw'),no=document.getElementById('rawno'),actual=document.getElementById('actual'),street=document.getElementById('actualstreet');
function update(){{let k=slider.value.padStart(3,'0');no.textContent=k;actual.src='frames/'+k+'_actual.png';street.src='frames/'+k+'_actual_street.jpg';}}
slider.addEventListener('input',update);document.querySelectorAll('button[data-raw]').forEach(b=>b.addEventListener('click',()=>{{slider.value=b.dataset.raw;update();}}));
document.querySelectorAll('button[data-positive]').forEach(b=>b.addEventListener('click',()=>{{let k=b.dataset.positive.padStart(3,'0');document.getElementById('positive').src='frames/'+k+'_positive.png';document.getElementById('positivestreet').src='frames/'+k+'_positive_street.jpg';}}));
</script></main>'''
    (d/'index.html').write_text(html,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
