"""Publish full-domain consistency controls and the mixed-support evidence."""
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image,ImageDraw,ImageFont

RUN=Path(__file__).resolve().parent.parent
PREV=RUN.parent/'scene008_a6_local_fusion_20261105'
ARMS=('parent','corridor','center','dual','center_matched')
LABELS=('Parent','Corridor 0.5','Center 0.5','Dual 0.5','Center matched')
HELD=(89,95,101,105,113,121,125,130,137)
PROBES=(97,105,113,130)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def avg(m):return m['sum']/m['count'] if m and m['count'] else None

def grid(paths,box=None,scale=1,rows=1):
    panels=[]
    for p in paths:
        with Image.open(p) as im:
            im=im.convert('RGB')
            if box:im=im.crop(box)
            if scale!=1:im=im.resize((im.width*scale,im.height*scale),Image.Resampling.NEAREST)
            panels.append(im)
    w=max(panels[0].width,205);h=panels[0].height+25
    result=Image.new('RGB',(w*5,h*rows),'#03080d');draw=ImageDraw.Draw(result)
    font=ImageFont.truetype('DejaVuSans.ttf',14)
    for i,im in enumerate(panels):
        x=i%5*w;y=i//5*h
        draw.text((x+6,y+4),LABELS[i%5],font=font,fill='#e3ecf2')
        result.paste(im,(x+(w-im.width)//2,y+25))
    return result

ARCH='''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1140 180" aria-label="完整下部一致性规则组件图">
<defs><marker id="a" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0L7 3.5L0 7Z" fill="#587180"/></marker></defs>
<g fill="none" stroke="#587180" stroke-width="2" marker-end="url(#a)"><path d="M180 75H225"/><path d="M410 75H455"/><path d="M660 75H705"/><path d="M890 75H935"/></g>
<g stroke="#bacbd5"><rect x="5" y="24" width="175" height="102" rx="7" fill="#edf4f7"/><rect x="225" y="24" width="185" height="102" rx="7" fill="#e3f0ea"/><rect x="455" y="24" width="205" height="102" rx="7" fill="#fff0da"/><rect x="705" y="24" width="185" height="102" rx="7" fill="#e3f0ea"/><rect x="935" y="24" width="200" height="102" rx="7" fill="#edf4f7"/></g>
<g font-family="sans-serif" font-size="15" text-anchor="middle" fill="#193340"><text x="92" y="56">固定父场景与 PVD</text><text x="92" y="85">原始 RGB / LiDAR</text><text x="317" y="56">40 BUILD 帧 αT×f</text><text x="317" y="85">中心 / 条件支撑 / 保护</text><text x="557" y="56">持久 UID 资格与系数</text><text x="557" y="85">C / CM + 剂量匹配</text><text x="797" y="56">同一三维裁剪算子</text><text x="797" y="85">原生联合排序合成</text><text x="1035" y="56">49 帧 × 三状态验收</text><text x="1035" y="85">失败保存父资产</text><text x="570" y="160">完整 431 个下部 PVD 基元、26,229 条固定目标射线；没有修改 PVD 参数或训练策略</text></g></svg>'''

def main():
    evaluation=read(RUN/'evaluation.json');registration=read(RUN/'registration.json')
    verification=read(RUN/'verification.json');failure=read(RUN/'failure_verification.json')
    assert verification['status']==failure['status']=='PASS' and len(verification['checks'])==45 and len(failure['records'])==15
    assert evaluation['frames']==49 and not evaluation['adopted']
    with np.load(RUN/'ledger.npz',allow_pickle=False) as f:ledger={k:f[k] for k in f.files}
    with np.load(RUN/'selected_state.npz',allow_pickle=False) as f:assert not bool(f['adopted']) and not np.any(f['strength'])
    full,center,dual,protected=ledger['masses'];total=float(full.sum())
    ratio=np.divide(protected,full+protected,out=np.zeros_like(full),where=full+protected>0)
    protected_exclusion=float(full[ratio>.02].sum()/total)
    dual_coverage=float(full[ledger['eligible_dual']].sum()/total)
    additional_mode_exclusion=float(full[ledger['eligible_center']&~ledger['eligible_dual']].sum()/total)
    p=evaluation['pools']['all'];table=[];records=[]
    for arm,label in zip(ARMS,LABELS):
        b,d=p['black'][arm],p['delete'][arm]
        table.append(f'<tr><th>{label}</th><td>{b["reveal"]:.6%}</td><td>{d["reveal"]:.6%}</td>'
            f'<td>{b["failed_frames"]} / {p["keep"][arm]["failed_frames"]} / {d["failed_frames"]}</td>'
            f'<td>{b["scene"]["known_rgb"]["mean"]:.8f}</td><td>{b["measured_depth"]["mean"]:.6f}</td>'
            f'<td>{b["parent_depth_change"]["mean"]:.8f}</td></tr>')
    for row in evaluation['rows']:
        for state,arms in row['states'].items():
            for arm,value in arms.items():
                records.append([row['raw'],row['held'],state,arm,avg(value['lower'])/avg(value['isolated']),
                    avg(value['scene'].get('known_rgb',value['scene'].get('a6_after'))),
                    avg(value['scene'].get('outside_rgb_delta')),avg(value['local']['measured_depth']),
                    avg(value['local']['depth_change']),max(t['max'] for t in value['local']['tails'].values()),'|'.join(value['issues'])])
    assert len(records)==735
    ref97=evaluation['rows'][8]['states']['black']
    outside97={a:avg(ref97[a]['scene']['outside_rgb_delta']) for a in ARMS}
    gain=100*(p['black']['dual']['reveal']-p['black']['parent']['reveal'])
    mode_effect=100*(p['black']['dual']['reveal']-p['black']['center_matched']['reveal'])
    parent_path=RUN.parent/read(RUN/'contract.json')['source_parent']
    dump(RUN/'state_interface.json',dict(
        loader='source/consistency.py: Consistency.frame(raw,state).render(strength)',
        parent_asset=str(parent_path),parent_asset_sha256=sha(parent_path),
        coefficient='Additional fraction of native conditional physical-corridor mass retired relative to the loaded parent; zero preserves the parent',
        persistence='strength indexes native_row_id and is shared across every time and edit state',
        renderer='All retained components stay in the same native joint compositing pass',
        required_dependencies='The campaign parent native reconstruction, PVD, road and inherited helper modules referenced by consistency.py remain required'))
    out=RUN/'delivery';out.mkdir(exist_ok=False)
    for name in ('data','source','frames'):(out/name).mkdir()
    for pattern in ('*.json','*.npz'):
        for source in RUN.glob(pattern):shutil.copyfile(source,out/'data'/source.name)
    shutil.copytree(RUN/'registration_frames',out/'data/registration_frames')
    for source in (RUN/'source').glob('*.py'):shutil.copyfile(source,out/'source'/source.name)
    for raw in range(89,138):
        with Image.open(PREV/'render'/f'{raw:03d}_pvd_pvd.png') as im:
            y,x=np.nonzero(np.asarray(im)[:,:,0]>5)
            box=(max(0,int(x.min())-12),max(0,int(y.min())-16),min(im.width,int(x.max())+13),min(im.height,int(y.max())+14))
        paths=[RUN/'render'/f'{raw:03d}_black_{a}.png' for a in ARMS]
        paths += [RUN/'render'/f'{raw:03d}_{a}_{suffix}.png' for suffix in ('pvd','blocked') for a in ARMS]
        grid(paths,box,3,3).save(out/'frames'/f'{raw:03d}_black.jpg',quality=95)
        if raw in HELD or raw==97:
            for state in ('keep','delete'):
                paths=[RUN/'render'/f'{raw:03d}_{state}_{a}.png' for a in ARMS]
                grid(paths,box,3).save(out/'frames'/f'{raw:03d}_{state}.jpg',quality=95)
                if raw in PROBES:grid(paths).save(out/'frames'/f'{raw:03d}_{state}_full.jpg',quality=92)
    with (out/'fixed_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f,lineterminator='\n');writer.writerow(['raw','held','state','arm','lower_visibility',
            'known_a6_RGB_MAE','outside_RGB_change_from_original38','measured_depth_MAE_m','parent_measured_depth_change_m','protected_RGB_change_max','failures'])
        writer.writerows(records)
    preflight_section=''
    if (RUN/'preflight.json').is_file():
        pf=read(RUN/'preflight.json')
        preflight_section=f'<section><h2>混合支撑的下一轮预检</h2><p>完整原生前方αT贡献中，既有corridor可裁到的条件质量只占 <strong>{pf["all_roles"]["front_crop_fraction"]:.4%}</strong>。关联真实a6深度点与独立PVD提案后，双侧候选共有 <strong>{pf["eligible"]["rows"]:,}</strong> 个UID，覆盖前方贡献的 <strong>{pf["eligible_front_coverage"]:.4%}</strong>；这些候选中带继承clip的前方责任占 <strong>{pf["eligible"]["front_clipped_fraction"]:.4%}</strong>。这比继续微调corridor资格触及更广，但仍不能代表全部未显露支撑。</p><p>真实测量点和提案点按实际时刻映射到车体坐标，累计父传输责任形成持久分界。这里仅完成观测关联与覆盖预检，尚未运行children换基或退役；前景αT归因与PVD可见deficit是不同量。</p><p><a href="data/preflight.json">覆盖与几何关联预检</a> · <a href="data/preflight_ledger.npz">预检账本</a></p></section>'
    summary=dict(status='VERIFIED_EXPERIMENT_NOT_ADOPTED',full_target_pixels=26229,lower_PVD_rows=431,
        center_UIDs=int(ledger['eligible_center'].sum()),dual_UIDs=int(ledger['eligible_dual'].sum()),
        target_crop_mass=total,protected_exclusion_fraction=protected_exclusion,dual_target_crop_coverage=dual_coverage,
        extra_mode_exclusion_fraction=additional_mode_exclusion,black_gain_percentage_points=gain,
        mode_vs_matched_percentage_points=mode_effect,outside_raw97=outside97,pools=p,
        selected_asset=evaluation['selected_asset'],independent_scene_metric_replays=60,
        sparse_BUILD_frames_reconstructed=40,real_BUILD_mass_replays=verification['registration_rebuilds'],
        next_reexpression='design and coverage preflight only; children not tested',training_updates=0)
    dump(out/'summary.json',summary)
    (out/'architecture.svg').write_text(ARCH,encoding='utf-8')
    html_text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Scene008 · 完整下部深度一致性规则</title>
<style>body{{margin:0;background:#f3f6f8;color:#17313f;font:16px/1.65 system-ui,sans-serif}}main{{max-width:1400px;padding:28px 24px;margin:auto}}section{{background:white;border:1px solid #d5e1e8;border-radius:8px;padding:20px;margin:18px 0;overflow:auto}}h1{{font-size:28px}}h2{{font-size:21px}}svg,img{{display:block;width:100%;height:auto}}th,td{{padding:8px;border-bottom:1px solid #dbe5eb;text-align:left;white-space:nowrap}}table{{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}}a{{color:#006775}}select,button{{font:inherit;padding:5px;margin-right:8px}}input{{width:45%}}.notice{{border-left:5px solid #b66d37;padding-left:14px}}.small{{font-size:14px;color:#49616d}}</style>
<main><h1>Scene008 · 完整下部深度一致性规则验证</h1><p class="notice">固定 431 个 PVD 下部基元、26,229 条原有目标射线，完成五臂 × 49 帧 × 三状态。双一致性规则黑底显露增加约 <strong>{gain:.6f} 个百分点</strong>，与剂量匹配中心规则仅差 <strong>{mode_effect:.8f} 个百分点</strong>。三个资格分支均略超 raw97 的累计非目标 RGB 门槛，本轮保留父资产。</p>
<section><h2>组件与数据流</h2>{ARCH}<p>动作始终是原36格下部三维 corridor 的精确条件高斯质量裁剪，仅改持久 native UID 系数。中心 C 与双一致性 CM 共用父场景、射线、裁剪及保护定义。C要求至少80%的目标裁剪贡献位于独立PVD深度前12 cm；CM还要求该部分经继承半空间与物理裁剪后的条件mode也在前。每条规则分别要求两个不同BUILD帧支持；受保护裁剪贡献比例≤2%。运行时不会按当前像素的front标签临时选择。</p><p>五臂为零操作、整corridor 50%、C 50%、CM 50%、C剂量匹配。匹配系数为 0.5×M<sub>CM</sub>/M<sub>C</sub>={float(ledger['match_coefficient']):.12f}，M包含所选UID在目标域的完整裁剪贡献，包含off-class命中。它匹配父传输下的一阶量；退役后的实际传输和PVD显露仍需实测。主候选预先固定为CM。</p></section>
<section><h2>规则筛到了什么</h2><p>构建帧目标裁剪责任总量 {total:.6f}。保护比例门槛排除了其中 <strong>{protected_exclusion:.4%}</strong>；C保留32个UID，CM保留30个，仅覆盖全部可裁剪目标贡献的 <strong>{dual_coverage:.4%}</strong>。CM比C另外排除的完整目标质量仅占总量 <strong>{additional_mode_exclusion:.4%}</strong>。</p><p>这轮完整下部对照表明，额外mode资格只改变很小一部分已有资格；它没有解决占主要份额的混合支撑。上述比例的分母是本算子可裁剪的父αT×f，不是全部未显露PVD，也不是隐藏几何错误率。保护低贡献只是风险指标；资格允许最多20%目标质量不满足对应front条件，不能解释为几何纯度或身份真值。</p></section>
<section><h2>完整49帧配对结果</h2><table><tr><th>分支</th><th>黑底下部显露</th><th>删a37街景显露</th><th>失败帧 黑/keep/delete</th><th>可见a6 RGB MAE</th><th>a6测量深度MAE / m</th><th>相对父测量深度变化 / m</th></tr>{''.join(table)}</table>
<p>资格分支的新增深度和局部RGB tail门槛均通过；失败来自继承的累计非目标变化约束。raw97中，父资产相对旧38.3%融合的非目标RGB平均变化为 {outside97['parent']:.11f}，CM为 {outside97['dual']:.11f}，固定上限0.005；超出约 {(outside97['dual']-.005):.3g}（RGB单位0–1）。因此，这不是一次大幅新增色斑的证据，也不能据此放宽预注册门槛。</p><p>无编辑状态保留a37遮挡，不要求PVD显露提高；删a37只关闭a37，使用同一资格与裁剪系数。隐藏域没有独立RGB/深度真值，显露率表示固定PVD贡献，不表示补全准确率。</p></section>
<section><h2>49帧黑底a6与街景</h2><p>五列顺序与表格相同。黑底依次为RGB、PVD实际贡献、固定目标未显露贡献（红色）；各分支使用同一裁剪与最近邻放大。街景可查看9个内部留出帧及失败探针97。</p><label>raw<span id="number">113</span></label><input id="raw" type="range" min="89" max="137" value="113"><button data-raw="97">97</button><button data-raw="105">105</button><button data-raw="113">113</button><button data-raw="130">130</button><button data-raw="137">137</button><select id="state"><option value="black">黑底</option><option value="keep">无编辑街景</option><option value="delete">删a37街景</option></select><img id="frame" src="frames/113_black.jpg" alt="五臂黑底与PVD贡献"><p id="availability"></p><a id="full" href="frames/113_delete_full.jpg">当前探针街景全图</a></section>
{preflight_section}<section><h2>下一轮：先证明重表达保真，再退出混合支撑</h2><p>下一步设计为：父核分成同时保留两侧的children，先检验完整换基R相对父资产P的RGB、α、深度保真；通过后再比较仅退出提案侧S与剂量匹配的整核退出U。解析矩守恒不能代替投影透明度和联合成像守恒。需要真实观测约束保留侧，并正确处理继承clip；不会继续扫描本轮退役率或资格阈值。</p><p><a href="data/next_reexpression_contract.json">下一轮设计合同（children尚未实验）</a></p></section>
<section><h2>复核、来源与边界</h2><p>40帧BUILD登记，内部HELD为89、95、101、105、113、121、125、130、137；它们已重复用于机制诊断，不是未触碰测试集。当前系数只由BUILD生成，完整49帧保护参与最终veto。原生排序中心与辅助深度最大差约4.77×10⁻⁶ m。40帧稀疏质量账本精确重建，三BUILD帧重新计算通过；raw105/113/130与失败raw97共60项场景指标保存状态重放通过，零操作重新加载与父资产精确一致。</p><p>原模型、道路、PVD几何/颜色/透明度未修改，训练更新0。本轮使用原生中心排序透明合成，没有不透明表面首交保证。保存系数表示父资产上额外裁去的物理质量比例，0表示保持父资产，加载器与依赖见状态接口。AutoDL保持开机。</p><p><a href="summary.json">汇总</a> · <a href="fixed_metrics.csv">735行逐帧指标</a> · <a href="data/registration.json">资格登记</a> · <a href="data/independent_audit.json">独立审计</a> · <a href="data/verification.json">主重放</a> · <a href="data/failure_verification.json">raw97重放</a> · <a href="data/selected_state.npz">实际选择状态</a> · <a href="data/state_interface.json">状态接口</a> · <a href="architecture.svg">组件图</a> · <a href="MANIFEST.sha256">文件校验</a></p></section>
<script>const raw=document.getElementById('raw'),state=document.getElementById('state'),street=new Set({json.dumps(list(HELD)+( [97]))}),probes=new Set([97,105,113,130]);function update(){{const r=Number(raw.value),key=String(r).padStart(3,'0');document.getElementById('number').textContent=r;const ok=state.value==='black'||street.has(r);document.getElementById('frame').src='frames/'+key+'_'+(ok?state.value:'black')+'.jpg';document.getElementById('availability').textContent=ok?'':'该帧街景保留全量指标；此处展示黑底，街景图见9个内部留出帧及raw97。';const full=document.getElementById('full');full.hidden=!probes.has(r);full.href='frames/'+key+'_'+(state.value==='keep'?'keep':'delete')+'_full.jpg';}}raw.addEventListener('input',update);state.addEventListener('change',update);document.querySelectorAll('button[data-raw]').forEach(b=>b.addEventListener('click',()=>{{raw.value=b.dataset.raw;update();}}));</script></main></html>'''
    (out/'index.html').write_text(html_text,encoding='utf-8')
    manifest=[f'{sha(p)}  {p.relative_to(out).as_posix()}' for p in sorted(out.rglob('*')) if p.is_file()]
    (out/'MANIFEST.sha256').write_text('\n'.join(manifest)+'\n',encoding='utf-8')
    print(json.dumps(dict(delivery=str(out),files=len(manifest)+1,csv_rows=len(records),adopted=False)))

if __name__=='__main__':main()
