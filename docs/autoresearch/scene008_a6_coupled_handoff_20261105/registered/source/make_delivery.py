"""Publish measured-surface handoff experiments, including rejected controls."""
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
THIN = ROOT.parent / 'scene008_a6_coupled_handoff_20261105'
REGISTERED = ROOT.parent / 'scene008_a6_registered_handoff_20261105'
PREV = ROOT.parent / 'scene008_a6_local_fusion_20261105'
RUNS = {'thin': THIN, 'registered': REGISTERED}
HELD = [89,95,101,105,113,121,125,130,137]
ARMS = ['parent','retirement','proxy_only','coupled']
LABELS = ['Parent','Retire only','New only','Coupled']

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def dump(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def mean(value):
    return value.get('mean',value.get('sum',0)/value['count'] if value['count'] else None)

def grid(paths, labels, box=None, scale=1, columns=4):
    panels=[]
    for path in paths:
        with Image.open(path) as im:
            im=im.convert('RGB')
            if box is not None:
                im=im.crop(box)
            if scale != 1:
                im=im.resize((im.width*scale,im.height*scale), Image.Resampling.NEAREST)
            panels.append(im)
    w=max(panels[0].width,205);h=panels[0].height+27
    output=Image.new('RGB',(w*columns,h*((len(panels)+columns-1)//columns)), '#070b10')
    draw=ImageDraw.Draw(output)
    font=ImageFont.truetype('DejaVuSans.ttf',14)
    for i,(im,label) in enumerate(zip(panels,labels)):
        x=(i%columns)*w;y=(i//columns)*h
        output.paste(im,(x+(w-im.width)//2,y+27))
        draw.text((x+7,y+5),label,fill='#e5edf2',font=font)
    return output

ARCH='''<svg viewBox="0 0 1080 205" role="img" aria-label="观测表面、旧支撑登记、联合合成与验收">
<defs><marker id="end" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0L7 3.5L0 7Z" fill="#506c7c"/></marker></defs>
<g fill="none" stroke="#506c7c" stroke-width="2" marker-end="url(#end)"><path d="M172 88H210"/><path d="M385 88H424"/><path d="M617 88H655"/><path d="M845 88H884"/></g>
<g stroke="#b9cbd5"><rect x="8" y="42" width="164" height="92" rx="8" fill="#eef5f8"/><rect x="210" y="42" width="175" height="92" rx="8" fill="#e4f2ec"/><rect x="424" y="42" width="193" height="92" rx="8" fill="#fff0d8"/><rect x="655" y="42" width="190" height="92" rx="8" fill="#e4f2ec"/><rect x="884" y="42" width="188" height="92" rx="8" fill="#eef5f8"/></g>
<g font-family="sans-serif" font-size="16" text-anchor="middle" fill="#183240"><text x="90" y="72">原始 RGB + LiDAR</text><text x="90" y="100">固定车姿与父场景</text><text x="297" y="72">BUILD 实测表面组</text><text x="297" y="100">3 组 / 17 个代理面</text><text x="520" y="72">旧支撑归因</text><text x="520" y="100">薄壳 / 贡献登记</text><text x="750" y="72">成对退役与加入</text><text x="750" y="100">统一排序计算 αT</text><text x="978" y="72">49 帧 × 3 状态</text><text x="978" y="100">RGB / 深度 / 保护</text><text x="540" y="174">固定 PVD、道路与非登记原生行；本轮未产生可接受的新融合资产</text></g></svg>'''

def main():
    out=REGISTERED/'delivery'
    out.mkdir(exist_ok=False)
    (out/'frames').mkdir();(out/'data').mkdir();(out/'source').mkdir()
    scores={}; all_csv=[];table=[]
    for variant,run in RUNS.items():
        destination=out/'data'/variant;destination.mkdir()
        for pattern in ('*.json','*.npz'):
            for source in sorted(run.glob(pattern)):
                shutil.copyfile(source,destination/source.name)
        code=out/'source'/variant;code.mkdir()
        for source in sorted((run/'source').glob('*.py')):
            shutil.copyfile(source,code/source.name)
        verification=read(run/'verification.json')
        assert verification['saved_ledger_exact_replay'] and len(verification['checked'])==108
        scores[variant]=[]
        for rank in range(3):
            result=read(run/f'joint_group{rank}_evaluation.json')
            fit=read(run/f'joint_group{rank}_fit.json')
            assert not result['method_accepted']
            held=result['pools']['held']['black']
            item=dict(group=rank,build_status=result['build_choice']['status'],
                      held={arm:held[arm]['group'] for arm in ARMS},
                      scene_failed_frames={state:result['pools']['all'][state]['coupled']['failed_frames']
                                           for state in ('black','keep','delete')},
                      issues=result['issues'],selected_gamma=result['selected_gamma'],
                      accepted=result['method_accepted'])
            scores[variant].append(item)
            before=held['retirement']['group'];after=held['coupled']['group']
            table.append(f'<tr><td>{"薄壳" if variant=="thin" else "贡献登记"}</td><td>{rank+1}</td>'
              f'<td>{after["rgb"]["count"]}</td><td>{before["rgb"]["mean"]:.5f} → {after["rgb"]["mean"]:.5f}</td>'
              f'<td>{before["depth"]["mean"]:.4f} → {after["depth"]["mean"]:.4f}</td>'
              f'<td>{after["coverage_error"]["mean"]:.2%}</td><td>{"通过" if item["build_status"]=="BUILD_PASS" else "未通过"}</td>'
              f'<td>{item["scene_failed_frames"]["black"]} / {item["scene_failed_frames"]["keep"]} / {item["scene_failed_frames"]["delete"]}</td></tr>')
            for row in result['rows']:
                for state,states in row['states'].items():
                    for arm,values in states.items():
                        g=values['group']
                        all_csv.append([variant,rank+1,row['raw'],row['held'],state,arm,
                            g['rgb']['count'],mean(g['rgb']),mean(g['depth']),mean(g['coverage_error']),
                            mean(g['old_new_coverage']),mean(g['total_alpha']),mean(g['known_rgb']),
                            mean(g['known_change']),mean(g['outside_change']),'|'.join(values['failures'])])
            for raw in HELD:
                with Image.open(PREV/'render'/f'{raw:03d}_pvd_pvd.png') as mask:
                    y,x=np.nonzero(np.asarray(mask)[:,:,0]>5)
                box=(max(0,int(x.min())-12),max(0,int(y.min())-16),min(480,int(x.max())+13),min(320,int(y.max())+14))
                source=run/f'joint_group{rank}_render'
                for state in ('black','keep','delete'):
                    paths=[source/f'{raw:03d}_{state}_{arm}.png' for arm in ARMS]
                    target=out/'frames'/f'{variant}_{rank}_{raw:03d}_{state}.jpg'
                    grid(paths,LABELS,box=box,scale=3).save(target,quality=94)
                    if state!='black' and raw in (105,113,130):
                        grid(paths,LABELS).save(out/'frames'/f'{variant}_{rank}_{raw:03d}_{state}_full.jpg',quality=91)
                if raw in (105,113,130):
                    diag=run/'diagnostic'
                    paths=[diag/f'{rank}_{raw:03d}_{arm}_{suffix}.png'
                           for suffix in ('coverage','change') for arm in ARMS]
                    grid(paths,[s+' | '+a for s in ('AlphaT','Delta') for a in LABELS],box=box,scale=3).save(
                        out/'frames'/f'{variant}_{rank}_{raw:03d}_contribution.jpg',quality=96)
    dump(out/'summary.json',dict(status='EXECUTION_VERIFIED_METHOD_REJECTED',scores=scores,
        prior_asset_accepted=True,new_asset_accepted=False,retirement=.5,
        held=HELD,training_updates=0,learned_policy=False,
        independent_probe_replays=216,autodl='kept on',
        limits=['Only 3 small measured groups; held counts 26/9/6.',
          'Repeated mechanism held frames, not untouched validation or scene generalization.',
          'Same-surface group attribution is not semantic Gaussian identity.',
          'Native center-depth alpha compositing, no first opaque surface guarantee.']))
    with (out/'fixed_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f,lineterminator='\n')
        writer.writerow(['variant','group','raw','held','state','arm','group_pixels','rgb_mae','depth_mae_m',
           'group_coverage_abs_error','group_contribution','total_alpha','known_rgb_mae','known_rgb_change',
           'outside_rgb_change','guard_failures'])
        writer.writerows(all_csv)
    errata=dict(field='no_other_retirement_levels_run',scope='joint stage only',
        actual_execution='root fit evaluated 0.25,0.5,0.75,1.0; vector joint and all49 scene evaluation used 0.5',
        result_files_preserved=True)
    dump(out/'metadata_errata.json',errata)
    (out/'architecture.svg').write_text(ARCH,encoding='utf-8')
    body=''.join(table)
    report=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Scene008 · 实测表面成对接管验证</title>
<style>body{{margin:0;background:#f3f6f8;color:#172f3b;font:16px/1.65 system-ui,sans-serif}}main{{max-width:1380px;margin:auto;padding:28px 24px}}section{{background:#fff;border:1px solid #d9e3e9;border-radius:9px;padding:20px;margin:18px 0;overflow:auto}}h1{{font-size:28px}}h2{{font-size:21px}}p{{max-width:1100px}}svg,img{{display:block;width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}}th,td{{text-align:left;padding:9px;border-bottom:1px solid #dbe5eb;white-space:nowrap}}.warning{{border-left:5px solid #b86a2d;padding-left:14px}}a{{color:#006875}}select,button{{font:inherit;padding:5px;margin-right:6px}}label{{margin-right:12px}}.small{{font-size:14px;color:#4a6170}}figure{{margin:16px 0}}figcaption{{font-size:14px;color:#4a6170}}code{{background:#edf2f5;padding:1px 4px}}</style>
<main><h1>Scene008 · 实测表面成对接管验证</h1>
<p class="warning"><strong>本轮全部执行与重放通过；3 个小表面组在两种接管规则下均未通过完整验收。</strong>没有发布新的可接受融合资产。继续保留上一轮已验证的父资产；固定下部显露率仍按父资产的黑底 50.93%、删 a37 街景 48.86% 记录。</p>
<section><h2>本轮要验证的规则</h2>{ARCH}
<p>先在实测车尾小面上验证能否接管。原始测量融合、固定车姿与已有父场景作为输入；PVD 几何、颜色、透明度均保持固定。新代理面从 BUILD 实测点取得几何与颜色，随后只拟合低维加入系数，未训练神经策略。</p>
<p>以 αT 登记原生支撑，在车体坐标中设置固定接管区域，联合渲染中同时减少旧质量并加入新表面。所有对象一起进行原生中心深度排序和透明合成，无编辑与删除 a37 使用同一套参数。</p></section>
<section><h2>从上一轮损伤定位到本轮对照</h2>
<p>上一轮 399 个实测代理面，独立渲染在透明度 ≥ 0.02 的实测面像素上深度 MAE 为 0.0488 m。运行时投影与原始相机投影的最大偏差为 3.26×10<sup>−5</sup> 像素，中心深度误差约 10<sup>−6</sup> m。先验对齐或单位换算未发现明显接线错误。该独立分母与先前联合场景完整面分母不同，不能直接当成同域改善。</p>
<p>raw113 的同域联合诊断：加入旧 399 面后，测量面前方旧贡献均值约 0.383 → 0.365，后方旧贡献约 0.617 → 0.146；有符号深度误差约 +0.012 → −0.372 m。新表面进入后压低了远处旧贡献，近处旧贡献仍在。父场景较好的平均深度可能含有前后贡献抵消；仅增加正确位置的表面并不保证最终深度更准。</p>
<p>本轮进一步要求每个锚点至少有两个不同 BUILD 帧的真实 RGB；邻域 8 cm、法线差 ≤ 20°、组内平面残差 q95 ≤ 6 cm，切向尺度上限 3.5 cm。枚举全部合格组，得到 15/9/8 个锚点，产生 7/5/5 个代理面。独立场景内验证范围很小，不代表完整车身恢复。</p></section>
<section><h2>固定对照与分母</h2>
<p>40 个 BUILD 帧构建与拟合，9 个内部留出帧为 raw89、95、101、105、113、121、125、130、137。三组 BUILD/留出测量像素分别为 141/26、49/9、65/6。留出帧已用于多轮机制诊断，不能称为未触碰测试集；父场景、PVD 与继承域仍含同场景完整观测信息。</p>
<p>每组保留四个配对分支：父资产、只退出旧支撑、只加入新面、二者成对。主比较固定旧支撑退役 50%；标量预算另外测试 25%、50%、75%、100%。加入系数先按 BUILD 的组贡献预算二分，再用 7/5/5 维有界联合求解调整，最多两次更新，系数上限 0.35。几何、颜色、尺度均不自由调整。</p>
<p><strong>薄壳规则</strong>只退出实测组定向小盒内的条件高斯质量。<strong>贡献登记规则</strong>先要求原生行在 BUILD 目标测量射线上贡献总量 ≥ 0.05，且至少跨两个帧，再沿组法线追踪其条件质量模式的 q05–q95，切向范围固定。它登记了渲染责任，尚未要求模式与实测表面深度一致。</p>
<p>后一规则登记 755/175/310 行，全部同时承担非目标支撑。目标组占其目标与保护域贡献的中位比例仅 1.72%/1.04%/1.91%，法线范围厚 2.82/1.98/3.05 m。此结果说明需要进一步区分深度与混合支撑。</p></section>
<section><h2>固定留出像素结果</h2>
<p>箭头表示「只退出旧支撑 → 成对接管」。RGB 与深度都必须改善；组贡献相对父资产的平均绝对变化 ≤ 0.01，并通过 BUILD 与全 49 帧保护。这里的组贡献是同次联合排序中的旧指定质量加新面 αT，累计透明度接近 1 不能替代这项检查。</p>
<table><thead><tr><th>旧支撑规则</th><th>组</th><th>测量像素</th><th>RGB MAE</th><th>深度 MAE / m</th><th>组贡献绝对偏差</th><th>BUILD</th><th>保护失败帧 黑底/无编辑/删 a37</th></tr></thead><tbody>{body}</tbody></table>
<p>薄壳第 2 组的留出 RGB 与深度同时改善，但组贡献偏差及黑底保护未通过。贡献登记后的第 2 组相对仅退役时深度改善，相对父资产的 RGB 仍变差。没有任何一组达到完整接管条件；尚不执行多组同时启用。</p></section>
<section><h2>黑底 a6 与街景配对图</h2>
<p>四列依次：父资产、仅退役、仅新面、成对接管。每个 raw 的各分支采用同一裁剪和最近邻放大。街景裁剪同样由 a6 区域确定，原图可通过下方链接检查。</p>
<label>规则 <select id="variant"><option value="thin">薄壳</option><option value="registered" selected>贡献登记</option></select></label>
<label>组 <select id="group"><option value="0">1</option><option value="1" selected>2</option><option value="2">3</option></select></label>
<label>raw <select id="raw">{''.join(f'<option value="{r}"'+(' selected' if r==113 else '')+f'>{r}</option>' for r in HELD)}</select></label>
<label>状态 <select id="state"><option value="black">黑底 a6</option><option value="keep">无编辑街景</option><option value="delete">删 a37 街景</option></select></label>
<figure><img id="frame" src="frames/registered_1_113_black.jpg" alt="四分支同域比较"><figcaption>实测小面变化幅度有限；请结合下面的贡献差值与表格判定。</figcaption></figure>
<figure id="diag"><img id="contribution" src="frames/registered_1_113_contribution.jpg" alt="四分支组贡献及变化"><figcaption>上排：指定旧质量与新面 αT；下排：相对父资产变化，红增蓝减，显示范围 ±0.05。图仅在固定已观测 a6 像素读取。</figcaption></figure>
<a id="full" href="frames/registered_1_113_delete_full.jpg">该探针删 a37 街景全图</a></section>
<section><h2>边界、重放与下一步</h2>
<p>两套账本分别从保存状态独立重放 raw105/113/130 × 3 状态 × 4 分支 × 3 组，共 216 个精确指标比较；并对物理代理透明度与逐命中系数实现做一致性检查。原始模型、固定 PVD 与道路未改写。使用原生高斯中心排序，尚无 MOG-2 不透明表面首交保证。</p>
<p>下一轮先按实测表面的近侧、同面与远侧区分旧支撑，记录跨帧深度一致性，分别决定可退出部分与需要保留的部分。零操作作为显式可行分支；先检验能否同时改善 RGB 与深度、保护其他成像，再测试下部缺失区显露。上述为下一轮待验证设计。</p>
<p><a href="summary.json">汇总</a> · <a href="fixed_metrics.csv">49 帧逐项指标</a> · <a href="architecture.svg">组件图</a> · <a href="data/thin/verification.json">薄壳重放</a> · <a href="data/registered/verification.json">登记重放</a> · <a href="data/registered/independent_audit.json">独立审计</a> · <a href="metadata_errata.json">元数据解释</a> · <a href="MANIFEST.sha256">文件校验</a></p>
<p class="small">训练更新 0；全部内部数值选择仅使用 BUILD。失败方案保留为实验，不替换已验证资产。AutoDL 保持开机。</p></section>
<script>
const ids=['variant','group','raw','state'];const controls=Object.fromEntries(ids.map(x=>[x,document.getElementById(x)]));
function update(){{const v=controls.variant.value,g=controls.group.value,r=controls.raw.value.padStart(3,'0'),s=controls.state.value;const prefix='frames/'+v+'_'+g+'_'+r;
document.getElementById('frame').src=prefix+'_'+s+'.jpg';const probe=['105','113','130'].includes(r);document.getElementById('diag').hidden=!probe;document.getElementById('contribution').src=probe?prefix+'_contribution.jpg':'';const full=document.getElementById('full');full.hidden=!probe;full.href=prefix+'_'+(s==='keep'?'keep':'delete')+'_full.jpg';}}
ids.forEach(x=>controls[x].addEventListener('change',update));
</script></main></html>'''
    (out/'index.html').write_text(report,encoding='utf-8')
    manifest=[]
    for path in sorted(out.rglob('*')):
        if path.is_file():
            manifest.append(f'{sha(path)}  {path.relative_to(out).as_posix()}')
    (out/'MANIFEST.sha256').write_text('\n'.join(manifest)+'\n',encoding='utf-8')
    print(json.dumps(dict(delivery=str(out),files=len(manifest)+1,csv_rows=len(all_csv),method_accepted=False)))

if __name__=='__main__':
    main()
