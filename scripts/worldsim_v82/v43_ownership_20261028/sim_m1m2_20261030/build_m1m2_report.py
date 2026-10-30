"""汇总已执行的原生 M1/M2 对照，不生成或修改实验 RGB。"""
from pathlib import Path
import json, shutil, html

BASE = Path(__file__).resolve().parents[2]
SRC = BASE / 'work/v43_sim_mechanism_20261030'
OUT = BASE / 'outputs/V43_Ownership_20261027/gpu_v43/sim_m1m2_20261030'
M0 = BASE / 'outputs/V43_Ownership_20261027/gpu_v43/sim_mechanism_20261030/evidence'
M1 = SRC / 'm1_assignment/evidence'
M2 = SRC / 'm2_repair/evidence'
REF = SRC / 'heldout_edit_refs/heldout_edit_refs_r1'
OUT.mkdir(parents=True, exist_ok=True)

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path, data): Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
def copy(source, dest):
    path=OUT/dest
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source,path)
    return dest

# 所有 JSON 原样保存；PNG 只取交付画廊的精确同帧结果。NPZ/checkpoint继续留远端。
for label, root in [('m1', M1), ('m2', M2), ('heldout_reference', REF)]:
    for path in root.rglob('*.json'):
        if 'scene_loader' not in path.parts:
            copy(path, 'evidence/'+label+'/'+path.relative_to(root).as_posix())
copy(SRC/'M1_M2_CONTRACT.json','contract.json')
copy(SRC/'heldout_edit_refs/visual_review.json','visual_review.json')
copy(M0/'edit_r3_10k/reference_metrics.json','evidence/m0_metrics.json')
copy(M0/'edit_r3_10k/render_manifest.json','evidence/m0_render_manifest.json')

names={'m0':'M0 · 原动态分支编辑','m1':'M1 · 仅归属/路由（高召回）',
       'pos':'M2 · 位置修复','cap':'M2 · 位置＋尺度','split':'M2 · 位置＋尺度＋拆分',
       'fit_m1':'M1＋200步局部外观','fit_m2':'M2拆分＋200步局部外观'}
roots={'m0':M0/'edit_r3_10k','m1':M1/'recall_verified',
       'pos':M2/'recall_position','cap':M2/'recall_position_scale','split':M2/'recall_position_scale_split',
       'fit_m1':M2/'refit_assignment','fit_m2':M2/'refit_split'}
frames=['000000','000008','000015']; edits=['none','delete','shift','rotate']; channels=['final','white','alpha']
assets={}; train_rows=[]
for arm,root in roots.items():
    allowed=['000008'] if arm.startswith('fit_') else frames
    rows=read(root/'reference_metrics.json')['rows']
    for row in rows:
        train_rows.append(dict(arm=arm,frame=row['frame'],edit=row['edit'],regions=row['regions']))
    for frame in allowed:
        for edit in edits:
            for channel in channels:
                source=root/edit/(frame+'_'+channel+'.png')
                assert source.is_file(),source
                dest='images/'+arm+'/'+edit+'/'+frame+'_'+channel+'.png'
                assets['train/'+arm+'/'+frame+'/'+edit+'/'+channel]=copy(source,dest)
for frame in frames:
    for edit in edits:
        path=M0/'dataset_r2/privileged/references'/edit/'shadow_on/rgb'/(frame+'.png')
        assets['reference/'+frame+'/'+edit]=copy(path,'images/reference/'+edit+'/'+frame+'.png')

heldout_names={'m0':None,'m1':None,'pos':'heldout_recall_position','cap':'heldout_recall_position_scale',
               'split':'heldout_recall_position_scale_split','fit_m1':'heldout_refit_assignment','fit_m2':'heldout_refit_split'}
for edit in edits:
    assets['heldout_reference/'+edit]=copy(REF/edit/'000008.png','images/heldout_reference/'+edit+'.png')
    for arm,dirname in heldout_names.items():
        if arm in ('m0','m1'):
            dirname='heldout_recall' if (arm,edit)==('m1','delete') else 'heldout_'+arm+'_'+edit
            root=M1/dirname
            for channel in channels:
                source=root/(channel+'.png')
                assert source.is_file(),source
                assets['heldout/'+arm+'/'+edit+'/'+channel]=copy(source,'images/heldout/'+arm+'/'+edit+'_'+channel+'.png')
        else:
            for channel in channels:
                source=M2/dirname/edit/('000008_'+channel+'.png')
                assert source.is_file(),source
                assets['heldout/'+arm+'/'+edit+'/'+channel]=copy(source,'images/heldout/'+arm+'/'+edit+'_'+channel+'.png')

uniform=read(M1/'heldout_uniform_metrics.json')
urows=uniform['rows']
# Normalize arm names using exact recorded names, never mix raw/clamped.
mapping={'m0':'m0','m1':'m1','heldout_recall_position_depth_r2':'pos','heldout_recall_position_scale_depth_r2':'cap',
         'heldout_recall_position_scale_split_depth_r2':'split','heldout_refit_assignment':'fit_m1','heldout_refit_split':'fit_m2'}
heldout_rows=[]
for row in urows:
    if row['arm'] in mapping:
        heldout_rows.append(dict(arm=mapping[row['arm']],edit=row['edit'],full=row['full'],original_body_footprint=row['original_body_footprint']))
assert len(heldout_rows)==28,(len(heldout_rows),{r['arm'] for r in urows})
metrics=dict(training_camera=train_rows,heldout=heldout_rows,display_metric='clamped_mae; raw_mae preserved separately',
    scope='image errors against independent edits, not Gaussian identity accuracy')
write(OUT/'metrics.json',metrics)
write(OUT/'assets.json',assets)
causal=read(M1/'causal_receipt.json')
recall=read(M1/'recall_verified/responsibility_summary.json')
completion=read(M2/'final_completion_receipt.json')
assert completion['native_checkpoint_unchanged'] and completion['local_refit']['total_updates']==400
assert all(x==0 for x in completion['local_refit']['save_reload_parity'])
assert read(M2/'depth_replay_parity.json')

def metric(arm, edit, field='full', key='clamped_mae', frame='000008'):
    return next(r['regions'][field][key] for r in train_rows if (r['arm'],r['frame'],r['edit'])==(arm,frame,edit))
table=''
for arm in names:
    table+='<tr><td>'+names[arm]+'</td>'+''.join('<td>%.5f</td>'%metric(arm,e) for e in edits)+'</tr>'
coverage=''.join('<tr><td>'+r['frame']+'</td><td>'+str(r['selected'])+'</td><td>%.3f%%</td><td>%d</td></tr>'%(100*r['selected_body_contribution_pixel_sum']/r['body_contribution_pixel_sum'],r['selected_with_background_conflict']) for r in recall['frames'])
delta={e:dict(raw_relative_improvement=1-metric('fit_m2',e,key='raw_mae')/metric('fit_m1',e,key='raw_mae'),
              clamped_relative_improvement=1-metric('fit_m2',e)/metric('fit_m1',e)) for e in edits}
state=dict(task_id='WS-V43-SIM-M1-M2-20261030',artifact_kind='actual_run_report',
    base_native_updates=10000,new_local_refit_updates=400,reference_association_frames=3,temporal_consistency_passed=False,
    unique_gaussian_gt=False,truth_assisted_not_mathematical_upper_bound=True,new_transferable_model_trained=False,
    edited_or_heldout_rgb_used_for_optimization=False,automatic_autodl_shutdown=False,human_verdict=None,
    frozen_representation_train_views=60,main_geometry_heldout_views=20,local_refit_train_views=8,local_refit_heldout_views=8,
    same_budget_refit_relative_full_image_improvement=delta,
    conclusion='No robust whole-edit gain from hard reassignment or direct moment repair; split plus local appearance refit helps some deletion errors but fails none/shift preservation. Not evidence that all feasible assignments or repairs fail.',
    depth_evidence='use corrected *_depth_r2 NPZ; historical images unchanged with exact RGB parity',
    next_step='audit pure vs mixed footprint responsibility and enforce unedited-state preservation for learned/joint surface-aware repair; no bulk dataset generation yet')
write(OUT/'report_state.json',state)

svg='''<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="460" viewBox="0 0 1400 460"><defs><marker id="a" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><path d="M0,0 L9,3 L0,6" fill="#55778b"/></marker></defs><style>text{font-family:Microsoft YaHei,Arial;fill:#183a4c;font-size:19px}.h{font-size:22px;font-weight:bold}.b{fill:#e9f1f8;stroke:#9bb6c8;stroke-width:1.5}.p{fill:#fff3dc;stroke:#b9b2a1}.g{fill:#e0f1e8;stroke:#a7c9b8}.line{fill:none;stroke:#55778b;stroke-width:2;marker-end:url(#a)}</style>
<rect class="b" x="16" y="26" width="308" height="402" rx="10"/><text class="h" x="40" y="62">固定输入与独立真值</text>
<rect class="b" x="36" y="88" width="268" height="98" rx="6"/><text x="50" y="120">原生整场景重建 · 10K</text><text x="50" y="153">static / dynamic / E 冻结</text>
<path class="line" d="M169 187V210"/><rect class="g" x="36" y="216" width="268" height="84" rx="6"/><text x="50" y="247">原观测相机的表面与深度</text><text x="50" y="278">GT 在世界上，不在高斯上</text>
<rect class="p" x="36" y="325" width="268" height="80" rx="6"/><text x="50" y="355">独立编辑 / 留出参考</text><text x="50" y="386">仅评价，不用于调参或拟合</text>
<rect class="g" x="365" y="26" width="618" height="402" rx="10"/><text class="h" x="387" y="62">成对干预：同一候选，不同允许操作</text>
<rect class="b" x="387" y="88" width="574" height="65" rx="6"/><text x="406" y="119">M1：实际 αT 贡献 → 条件归属 → 原参数编辑</text><text x="406" y="144" style="font-size:16px">几何支持敏感性另存；高召回不是正确身份保证</text>
<path class="line" d="M674 154V180"/><rect class="p" x="387" y="187" width="574" height="91" rx="6"/><text x="406" y="218">M2：固定 M1 候选 → 位置 → 尺度 → 条件拆分</text><text x="406" y="253">继承外观；每步独立保留；0 优化更新</text>
<path class="line" d="M674 279V305"/><rect class="b" x="387" y="314" width="574" height="91" rx="6"/><text x="406" y="345">等预算控制：M1 / M2 各 200 步局部 DC＋opacity</text><text x="406" y="380">只用原观测 RGB；保存 / 恢复；容量另列</text>
<rect class="b" x="1024" y="26" width="360" height="402" rx="10"/><text class="h" x="1046" y="62">同请求渲染与分层验收</text>
<rect class="b" x="1044" y="88" width="320" height="91" rx="6"/><text x="1060" y="120">none / delete / shift / rotate</text><text x="1060" y="154">原环境与白底配对</text>
<path class="line" d="M1205 180V210"/><rect class="g" x="1044" y="216" width="320" height="91" rx="6"/><text x="1060" y="247">车形残留 / 背景 / 结构</text><text x="1060" y="282">C_G / α / 实际环境回填</text>
<path class="line" d="M1205 308V327"/><rect class="p" x="1044" y="335" width="320" height="70" rx="6"/><text x="1060" y="365">3 个锚点＋1 个留出相机</text><text x="1060" y="391" style="font-size:16px">不作完整时序或泛化结论</text>
<path class="line" d="M324 133H385"/><path class="line" d="M984 133H1043"/></svg>'''
(OUT/'architecture.svg').write_text(svg,encoding='utf8')

report='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V43 M1/M2 · 归属与表示修复对照</title><style>
body{max-width:1500px;margin:28px auto;padding:0 22px;background:#f4f8fb;color:#183a4b;font:16px/1.6 "Microsoft YaHei",sans-serif}section{background:#fff;border:1px solid #c7d9e4;padding:20px;margin:20px 0;border-radius:10px}h1{font-size:29px}h2{font-size:21px}img{width:100%;display:block}figure{margin:0}figcaption{font-size:14px;padding:7px 0}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.notice{background:#fff0d9;padding:14px}.controls{display:flex;gap:15px;flex-wrap:wrap;margin-bottom:14px}select{padding:8px;font:inherit}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:8px;border:1px solid #c9dbe7;text-align:left}th{background:#e9f1f7}code{overflow-wrap:anywhere}a{color:#176aaa}.small{font-size:14px;color:#536e7f}.scroll{overflow:auto}@media(max-width:850px){.grid{grid-template-columns:1fr}}</style>
<h1>V43：归属能带来多少收益，表示修复又增加什么？</h1>
<p>单个独立仿真机制场景 · 原生实际重建 10K · M1/M2 第一轮执行结果。<a href="../sim_mechanism_20261030/index.html">M0 数据与重建现场</a> · <a href="../../index.html">V43 设计</a></p>
<section><h2>当前答案</h2><p><b>仅重新归属没有得到稳健编辑收益；本次几何和拆分修复也尚未解决问题。</b> 高召回候选把大量背景贡献一并选入，直接整颗移除会损伤背景。位置、尺度、拆分后，外观与原几何不再匹配；加入等预算局部外观拟合后，部分删车误差降低，但未编辑画面与横移仍退化。</p>
<p class="notice">这是几种明确实现的真值辅助参考，不是“完美归属”的答案。99.9%是删除前静态层在车体像素中的<b>贡献权重覆盖</b>，不是车辆高斯召回率，也不包含全部阴影。负结果不能证明所有可能的归属／修复都无效，更不能据此宣称拆分已是必要或充分条件。</p>
<p><a href="contract.json">执行合同</a> · <a href="report_state.json">实际状态</a> · <a href="metrics.json">全部同口径 raw/clamped 指标</a> · <a href="visual_review.json">独立 subagent 视觉核验</a></p></section>
<section><h2>组件与控制变量</h2><img src="architecture.svg" alt="固定重建与世界证据，M1归属，M2同候选分步修复，成对环境和编辑评价">
<p>归属构建只读取原观测相机的未编辑表面／深度和真实光栅贡献。编辑参考不进入选择或拟合；留出相机复用中帧既定候选与修复参数。0/8/15 三个锚点有实际责任审计，未运行16帧完整责任或时序稳定性验收。原动态分支仍作为带噪的继承对象路由，并未被当成正确身份 GT。</p></section>
<section><h2>同帧图像：优先看残影、结构与误改</h2><div class="controls"><label>帧 <select id="frame"><option>000000</option><option selected>000008</option><option>000015</option></select></label><label>操作 <select id="edit"><option value="none">不编辑</option><option selected value="delete">删车</option><option value="shift">横移1.5m</option><option value="rotate">转向30°</option></select></label><label>输出 <select id="channel"><option value="final">原环境最终 RGB</option><option value="white">统一白底诊断</option><option value="alpha">高斯 alpha</option></select></label></div><div id="main" class="grid"></div>
<p class="small">参考列始终是独立仿真 RGB。其余列为相同编辑下的原生输出；白底表示透射，并非道路修复。深度辅助通道已在新目录重放纠正；C_G、alpha、E、final 与修复前逐值相同，画廊使用这些已核验原图。</p></section>
<section><h2>M1：为什么“选得更多”没有直接变好</h2><p>13个协方差支撑采样点靠近车表面的严格/宽松规则，对后两帧实际静态车像素贡献仅覆盖约1.4–7.6%。新增真实 alpha*T 高召回对照，取消深度门控与背景冲突否决：</p>
<table><tr><th>帧</th><th>选中静态行</th><th>原静态车像素贡献覆盖</th><th>同时参与非车表面像素的行</th></tr>'''+coverage+'''</table>
<p>这些双重贡献可能是宽高斯的投影足迹，也可能包含真实混合支撑；不能自动给出物理混合身份。删车后，剩余高斯因透射变化重新参与成像。与独立无车道路深度比较，原车区域的合成期望深度 MAE 约7.2–7.5m；也有少量像素与道路深度接近，因此不能把所有删后 alpha 都算残影。</p>
<p><a href="evidence/m1/causal_receipt.json">严格/宽松/贡献选择敏感性</a> · <a href="evidence/m1/recall_verified/responsibility_summary.json">贡献守恒核对</a> · <a href="evidence/m1/postdelete_depth_audit.json">删后显露深度审计</a></p></section>
<section><h2>M2：每一步究竟改变什么</h2><table><tr><th>对照</th><th>操作</th><th>冻结／限制</th></tr>
<tr><td>位置</td><td>同一候选的中心置于其 alpha*T 加权车体 first-hit XYZ</td><td>这是条件重心，不保证落在真实单一表面；0更新</td></tr>
<tr><td>＋尺度</td><td>最大轴固定截到0.08m，保留轴比</td><td>启发式尺度对照，未搜索最佳阈值；颜色与不透明度冻结</td></tr>
<tr><td>＋拆分</td><td>两类贡献均大于1e-4时，父点分为随车与静态子点</td><td>各自独立表面重心；继承父SH；按光学厚度比例分opacity；新增8443/10457/8927行</td></tr>
</table><p>位置和尺度不会改变“整颗删掉之后”的输出，这是预期的同源控制。拆分保留的背景子点才会改变delete。拆分比例仅在同位置同足迹时保透射；子点移位后不保证像素alpha守恒，也不能从父外观唯一恢复车与道路两套颜色。</p>
<p><a href="evidence/m2/main_method_receipt.json">方法与特权信息</a> · <a href="evidence/m2/final_completion_receipt.json">完成／容量／重放记录</a> · <a href="evidence/m2/receipt_errata.json">旧收据字段勘误</a> · <a href="evidence/m2/depth_replay_parity.json">深度通道修复与RGB不变核验</a></p></section>
<section><h2>局部外观适配：等步数、容量单列</h2><p>只对中帧追加一次200步控制：M1与M2拆分各优化选中行DC＋opacity，Adam学习率分别0.0025/0.05；原观测完整RGB L1，无编辑GT、无留出GT、无新训练数据。M1 50,712参数，M2 92,540参数，容量不同如实保留。各保存后恢复，渲染差均为0。</p>
<div class="controls"><label>操作 <select id="fit_edit"><option value="none">不编辑</option><option selected value="delete">删车</option><option value="shift">横移</option><option value="rotate">转向</option></select></label></div><div id="fit" class="grid"></div>
<p>原环境中帧全图 MAE（0–1，<b>clamped</b>；越小越好）：</p><div class="scroll"><table><tr><th>方案</th><th>none</th><th>delete</th><th>shift</th><th>rotate</th></tr>'''+table+'''</table></div>
<p>raw与clamped排名可能不同；完整分区值保留在metrics.json。以上不是归属精度或泛化成绩。即使部分删车误差下降，仍须结合原位置车形残留和保留道路检查。</p></section>
<section><h2>留出视角：不重新读取真值来选点</h2><p>中帧相机沿世界Y移动1m并重新瞄准；候选、位置、尺度、拆分与局部外观固定。独立4路参考另外生成，none与旧参考仅51/460800通道存在小差异（8bit MAE 0.000161，最大5），已记录。该单视角不是泛化结论。</p>
<div class="controls"><label>操作 <select id="hold_edit"><option value="none">不编辑</option><option selected value="delete">删车</option><option value="shift">横移</option><option value="rotate">转向</option></select></label><label>对照 <select id="hold_set"><option value="main">0更新主对照</option><option value="fit">200步局部适配</option></select></label></div><div id="heldout" class="grid"></div>
<p><a href="evidence/m1/heldout_uniform_metrics.json">同一FP32口径40行完整评分</a> · <a href="evidence/heldout_reference/manifest.json">独立参考生成与微小重放差异</a></p></section>
<section><h2>下一步与本轮边界</h2><ol><li>将条件贡献与物理归属分开：先区分纯车、投影足迹跨界、真实混合、错误深度；不把“对车像素有微小贡献”直接变成整颗车身份。</li><li>修复必须加入未编辑状态保持。位置／协方差／拆分与外观需共同约束，不能只把点压到first-hit重心；只根据RGB原图拟合仍可能补偿出错误分解。</li><li>单列阴影接收面与环境残留。当前body候选没有解决关联阴影；环境被冻结，但实际回填随alpha变化。下一轮先补世界表面对应和受控局部修复，再扩3–5机制场景与训练模块。</li></ol><p>本轮未训练可迁移联合模型，未对真实场景一提供完美GT，未证明“必须拆分”。保留失败运行和所有原始输出；完整FP32与selection留远端。AutoDL保持开机。</p></section>
<script>const A=ASSETS,N=NAMES;function card(src,title){return '<figure><img src="'+src+'" alt="'+title+'"><figcaption>'+title+'</figcaption></figure>'}function main(){const f=document.querySelector('#frame').value,e=document.querySelector('#edit').value,c=document.querySelector('#channel').value;document.querySelector('#main').innerHTML=card(A['reference/'+f+'/'+e],'独立仿真参考')+['m0','m1','pos','cap','split'].map(k=>card(A['train/'+k+'/'+f+'/'+e+'/'+c],N[k]+' · '+c)).join('')}function fit(){const e=document.querySelector('#fit_edit').value;document.querySelector('#fit').innerHTML=card(A['reference/000008/'+e],'独立仿真参考')+['fit_m1','fit_m2'].map(k=>card(A['train/'+k+'/000008/'+e+'/final'],N[k])).join('')}function hold(){const e=document.querySelector('#hold_edit').value,ks=document.querySelector('#hold_set').value==='fit'?['fit_m1','fit_m2']:['m0','m1','pos','cap','split'];document.querySelector('#heldout').innerHTML=card(A['heldout_reference/'+e],'留出 · 独立仿真参考')+ks.map(k=>card(A['heldout/'+k+'/'+e+'/final'],N[k])).join('')}for(const id of ['frame','edit','channel'])document.getElementById(id).onchange=main;document.getElementById('fit_edit').onchange=fit;document.getElementById('hold_edit').onchange=hold;document.getElementById('hold_set').onchange=hold;main();fit();hold();</script></html>'''
report=report.replace('ASSETS',json.dumps(assets,ensure_ascii=False)).replace('NAMES',json.dumps(names,ensure_ascii=False))
(OUT/'index.html').write_text(report,encoding='utf8')
print(json.dumps({'report':str(OUT/'index.html'),'assets':len(assets),'train_metrics':len(train_rows),'heldout_metrics':len(heldout_rows)},ensure_ascii=False))
