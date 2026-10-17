"""从实测 JSON 生成中文诊断报告、严重性表格与仓库报告。"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import html
import json
import statistics
import shutil
import os

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs'
E=OUT/'r15_diagnostic_evidence'
REPO=ROOT/'work/r15_audit/publish_repo'
if os.name=='nt':REPO=Path('\\\\?\\'+str(REPO))
DOC=REPO/'docs/autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/paused_diagnostic_20261017'
DOC.mkdir(parents=True,exist_ok=True)
p=json.loads((E/'performance.json').read_text())
a=json.loads((E/'actors/actors.json').read_text())
n=json.loads((E/'shadow_precision.json').read_text())
pause=json.loads((E/'pause_receipt.json').read_text())
r6=next(r for r in a['records'] if r['actor']==6 and r['raw']==105)
r37=next(r for r in a['records'] if r['actor']==37 and r['raw']==121)
g6_depth=r6['static_dynamic_depth_delta_inside_median'];g37_depth=r37['static_dynamic_depth_delta_inside_median']
g6_reject=100*r6['gate_rejected_fraction_inside'];g37_reject=100*r37['gate_rejected_fraction_inside']
g6_median=100*statistics.median(r['gate_rejected_fraction_inside'] for r in a['records'] if r['actor']==6 and r['gate_rejected_fraction_inside'] is not None)
g37_median=100*statistics.median(r['gate_rejected_fraction_inside'] for r in a['records'] if r['actor']==37 and r['gate_rejected_fraction_inside'] is not None)
now=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
def average(case):return statistics.mean(r['forward_backward_seconds'] for r in p['cases'] if r['case']==case and 'forward_backward_seconds' in r)
base=average('baseline');off=average('shadow_off');one=average('sun_samples_1');chunk=average('gaussian_chunk_2048');nocp=average('checkpoint_disabled')
delta=100*(base-off)/base
causes=[
 ('P0','数值正确性','FP32世界协方差及其显式求逆失稳',f"raw107的976,304个有效Actor高斯中，11个实际FP32协方差已非正定，8个逆矩阵对称部分非正定；最大物理条件数{n['conditioning']['physical_condition_max']:.2e}。仅chunk512→2048就使部分透射率从1变为约1.8e-35，FP64参考约1。",'先改为保留正定性的因子化/局部坐标积分，配FP64参考、分块不变性与梯度检查；仅增大chunk或追加clamp不合格。'),
 ('P0','性能主因','Python/Torch逐Actor、射线块、高斯块的小算子循环',f"同帧F+B {base:.2f}s，关阴影{off:.2f}s，差额{delta:.1f}%。单次无梯度完整前向记录380,987次cudaLaunchKernel，7,551次nonzero及13,640次cudaStreamSynchronize；阴影前向5.54/6.12s。",'批处理太阳方向与候选对，缓存每帧/每块边界，避免Python读取GPU条件；建立融合积分/反传内核。算子计数来自额外profiling，不拿profiling墙钟当训练吞吐。'),
 ('P1','反传开销','对许多小积分块使用activation checkpoint',f"1290次checkpoint使积分调用5715→7005，射线×高斯对3818万→6757万。禁用后F+B {nocp:.2f}s，峰值约11.94GiB；原配置约{base:.2f}s、8.85–8.98GiB。",'依据块大小和实际显存选择checkpoint，融合反传；当前只做副本诊断，未应用该变更。'),
 ('P1','重复工作','5个太阳方向串行重复调度和粗筛',f"5→1太阳样本时F+B约{one:.2f}s；同样5方向批处理的可见性前向4.67→1.77s，但最大透射率差0.00858。",'保留5方向覆盖，先修数值，再验证批处理等价；减到1方向改变了软阴影模型，不能当无损提速。'),
 ('P1','规模放大','增密使Actor数量和阴影候选对急增',f"Actor种子163,387→1,038,133（6.35倍）；总点771,821→2,068,305。a6由3453→123800（35.9倍）。训练相邻日志区间中位数从前1k约2.21s/步增至3k–4k约15.66s/步。",'检查增密收益、空间组织和真实可见支持；先使用有效加速结构，不以删车底、截断点数或隐藏坏例代替修复。'),
 ('P1','主相机工作量','76通道被容量10的ABI拆为8次整景光栅',"单次无梯度前向：整景8通道批次合计0.276s；加43路独立alpha和2路分支，共46函数调用、51底层通道pass。反传也要经过这些分支。",'复用几何预处理/排序/分箱，评估融合多通道输出；不可把config容量直接写大而突破现有CUDA ABI。'),
 ('P2','独立分支','每步43实例独立alpha和2个孤立分支',"无梯度独立alpha共0.125s，静/动态分支共0.039s。它们服务完整实例监督和竞争约束，有明确用途，但当前逐路提交有额外成本。",'批量渲染/共享预处理，保持完整内外域与未知标签语义。不能为提速删掉监督。'),
 ('P2','CPU与输入','逐步重新解码、验证、传输冻结监督；重复不变工作',f"本次frame构造/传输0.127s，旧诊断0.165s；每步还读legacy配置并把mask送CPU计算boundary_distance，即使本表示的NPS特征恒为0。",'缓存冻结Frame/有效域、距离图、配置与相机；profile后再做异步搬运。该量级不足以解释十几秒的主差距。'),
 ('P2','周期成本','增密/SGCM、归属审计、保存与GPU锁等待',"3k两轮全46帧归属审计约148.0+147.3s，split约6.6s、owner约45.5s；约5.8分钟，另有保存/锁等待。它们不会出现在每帧F+B中。",'单列周期与排队成本，按同worker相邻区间算吞吐。当前证据不支持把每步16s归因于这些间歇工作。'),
 ('P2','已排除主因','Adam、梯度裁剪及天空采样不是主要瓶颈',"零学习率副本Adam计时0.033s，梯度裁剪0.035s，屏幕梯度积累0.001s；场景几何求值约0.048s，天空采样0.0013s（无梯度）。",'这些局部计时用于排除主因；父子模块耗时不可相加。无证据需要先更换GPU或增加CPU线程。'),
]
json_data=dict(run_id='WS-V82-R15-PAUSED-DIAGNOSIS-20261017',updated_singapore=now,production_SGD_updates=0,
              checkpoint_step=4000,last_logged_step=4100,causes=[dict(severity=x[0],category=x[1],cause=x[2],evidence=x[3],next_step=x[4]) for x in causes],
              benchmark=dict(baseline_F_B_seconds=base,shadow_off_F_B_seconds=off,shadow_increment_fraction=delta/100),
              failure_ledger_refs=['V82-F01','V82-F05','V82-F06'],failure_ledger_delta=['V82-F05','V82-F06'],human_verdict=None,problem_solved=None)
(E/'severity_table.json').write_text(json.dumps(json_data,ensure_ascii=False,indent=2),encoding='utf-8')
svg='''<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="224" viewBox="0 0 1120 224"><defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="#425b70"/></marker></defs><g fill="#f3f7fa" stroke="#425b70" stroke-width="1.5"><rect x="8" y="76" width="180" height="72" rx="5"/><rect x="225" y="76" width="185" height="72" rx="5"/><rect x="453" y="12" width="225" height="76" rx="5"/><rect x="453" y="140" width="225" height="76" rx="5" stroke="#c13b32" fill="#fff1ef"/><rect x="725" y="76" width="160" height="72" rx="5"/><rect x="930" y="76" width="180" height="72" rx="5"/></g><g font-family="Microsoft YaHei,Arial" font-size="16" text-anchor="middle" fill="#172b3c"><text x="98" y="105">RGB / LiDAR</text><text x="98" y="132">mask / 固定位姿</text><text x="318" y="105">4k检查点</text><text x="318" y="132">静态 + 43 Actor</text><text x="566" y="42">主相机光栅</text><text x="566" y="68">76通道 / 8批次</text><text x="566" y="170">5方向高斯密度积分</text><text x="566" y="196">性能与数值主要故障</text><text x="805" y="105">RGB合成</text><text x="805" y="132">分层 / 删除</text><text x="1020" y="105">同帧计时 / FP64</text><text x="1020" y="132">49帧归属证据</text></g><g fill="none" stroke="#425b70" stroke-width="2" marker-end="url(#a)"><path d="M188 112H220"/><path d="M410 102H430V50H448"/><path d="M410 122H430V178H448"/><path d="M678 50H700V98H720"/><path d="M678 178H700V126H720"/><path d="M885 112H925"/></g></svg>'''
(E/'architecture.svg').write_text(svg,encoding='utf-8')
def esc(x):return html.escape(str(x))
def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(v)+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in r)+'</tr>' for r in rows)+'</tbody></table></div>'
case_rows=[]
for label,title in [('baseline','原配置'),('shadow_off','关阴影'),('sun_samples_1','太阳样本5→1'),('gaussian_chunk_2048','高斯块512→2048'),('checkpoint_disabled','禁用积分checkpoint')]:
    records=[r for r in p['cases'] if r['case']==label]
    case_rows.append([title,len(records),f"{statistics.mean(r['forward_seconds'] for r in records):.3f}",f"{statistics.mean(r['backward_seconds'] for r in records):.3f}",f"{average(label):.3f}",f"{max(r['image_max_abs_delta_from_baseline_linear'] for r in records):.6f}"])
num=n['selected_ray_references']
numerical_rows=[[i,f'{x:.8g}',f'{y:.8g}',f'{z:.8g}'] for i,x,y,z in zip(num['indices'],num['chunk512'],num['chunk2048'],num['dense_FP64_physical_covariance'])]
actor_rows=[]
for actor,rawframe in [(6,105),(37,121)]:
    r=next(x for x in a['records'] if x['actor']==actor and x['raw']==rawframe)
    actor_rows.append([f'a{actor} / raw{rawframe}',r['body_pixels'],f"{r['delete_static_visible_alpha_inside_mean']:.3f}",f"{r['delete_other_actor_visible_alpha_inside_mean']:.3f}",f"{r['static_dynamic_depth_delta_inside_median']:.2f} m",f"{100*r['gate_rejected_fraction_inside']:.2f}%",f"{r['delete_shadow_effect_MAE_inside_linear']:.6f}"])
doc_intro=f'''# R15暂停诊断：阴影性能、数值与a6/a37分层

更新时间：{now}。task/run：`WS-V82-R15-PAUSED-DIAGNOSIS-20261017`。完整[HTML报告](report.html)，[实测严重性表](r15_diagnostic_evidence/severity_table.json)。当前执行授权只见仓库 `docs/RESEARCH_STATUS.md`。

![输入、静态与43Actor、主光栅、阴影积分和诊断输出](r15_diagnostic_evidence/architecture.svg)

结论：当前实现存在明确的性能瓶颈和已复现的阴影数值错误。4k原配置同帧前向+反传{base:.2f}s，关阴影{off:.2f}s，差额约{delta:.1f}%。旧流程从头GN11 30k加实例30k的训练进程合计181.5分钟，不能仅引用最后27k的81.1分钟；两版初始化、目标、Actor及工作量不同。

本次暂停保存点是4000，最后日志4100。完整模型、13组非空Adam状态和RNG已读回，硬链接独立保留于远端 `backend_restore_r15/paused_checkpoint_004000_20261017/checkpoint.pt`。仅控制器增加暂停启动保护；生产训练表示、损失、参数、CUDA ABI和检查点均未用诊断候选覆盖。所有副本的参数更新为0；Adam单独计时使用lr=0。相关心跳原本PAUSED，未创建新任务或subagent。

## 按严重性排序

| 优先级 | 原因 | 证据与处理方向 |
|---|---|---|
'''
for sev,cat,cause,evidence,step in causes:doc_intro+=f'| {sev} | {cause} | {evidence} {step} |\n'
doc_intro+=f'''
各行存在父子关系，不能把其时间或加速比相加。完整计时在performance.json；profiling的算子计数与未插桩吞吐分别记录。当前没有应用“减太阳样本”“改大chunk”“关阴影”等改变输出的方案。

## 阴影数值错误

当前用FP32旋转/尺度构造世界协方差后显式求逆；高度各向异性点会产生非正定矩阵，后续把a/transverse强行clamp为正，仍可能得到错误但有限的遮挡。扩大高斯块使此前被粗筛排除的病态点参与积分，结果明显依赖chunk。五个最大差异网格点的同模型FP64稠密参考（全部有效Actor点、无AABB）支持接近无遮挡；原FP32逆矩阵即使用FP64后续算术也给出几乎全遮挡。仅把c-b²/a改成残差二次型仍有非有限值，未作为修复采用。

保留有限梯度不等于阴影数值正确。先用局部旋转/尺度因子避免显式病态求逆，复核FP64、分块与太阳批处理等价及梯度，然后再提速。这里只定位错误，没有证明该错误解释全部外观差距。

## a6 / a37

49帧无阴影分层、46个训练帧精确alpha*T特征梯度审计，以及6个锚帧的阴影/删除反事实均完成。a6 raw105及a37 raw121删除后关阴影仍有残留；静态层单独渲染保留相同车形外观，a6还受其他Actor贡献影响。删除确实移除了目标owner的全部高斯，残影并非单纯漏删目标点或剩下投射阴影。

![a6锚帧分层](r15_diagnostic_evidence/a06_raw105_components.png)

![a37锚帧分层](r15_diagnostic_evidence/a37_raw121_components.png)

现有竞争门控只在静动态深度差<0.45m时工作。a6锚帧中位深度差{g6_depth:.2f}m、门控拒绝{g6_reject:.2f}%；a37为{g37_depth:.2f}m、拒绝{g37_reject:.2f}%。道路/远处静态表面上的车形外观没有被“同表面占用竞争”覆盖。不能把正确静态道路点全部转成Actor来消除图像残影。

3k审计紧接opacity reset：最大单点opacity约0.01。a6/a37没有任何点达到既定重复道路正证据；实际split=0、Actor→static=0、static→Actor=0，仅从待定真实测量池加入459点（a6 188、a37 114）。4k时静态点中source_candidate_owner为a6/a37的数量都为0，当前静态转Actor规则没有这两个目标的候选入口。该限制不能靠多跑相同审计自动消除；9k审计也与3000步reset间隔重合。

a6种子3453→123800点，a37 10349→83130；4k中心越出软OBB的点分别29834、14573。OBB项约束中心且为软惩罚，不保证高斯支撑范围或独立资产边界。raw alpha车外支撑还可能包含合法被遮挡部分，不能全部当作道路误入。处理应分开：Actor碎片的实际多帧证据、静态外观残影的真实跨时刻表面观测、以及opacity reset后的有效审计时机；保留完整内外mask监督，未知不当背景，不按高度切车底。

## 范围与验证

- 同一保存点raw107做重复F+B计时；9次测量，0生产SGD。43实例/2,068,305点/480×320，PyTorch 2.2.2+cu118、RTX4080SUPER。
- 计时覆盖数据、几何、环境、光栅、阴影、反传、梯度裁剪、屏幕统计与lr=0 Adam；周期审计用原日志单列，不伪造每步占比。后续material/joint实际训练尚未发生，旧零更新探测不作为最终吞吐。
- 暂停控制器在启动任何worker前读PAUSE_REQUESTED，实际执行返回0且未启动训练；原控制器已备份。该旧控制器并非通用恢复入口（仍有初始step2断言），不能删除暂停标记后直接当作4k恢复命令。
- FP64数值核查只覆盖一个帧的5个最大分块差异射线；不是全49帧FP64验收。当前没有修复后训练、完整编辑GT或最终质量通过结论。
- BUILD/QUERY边界不变；这些QUERY已用于开发。静态残影来源由反事实可定位，但无遮挡道路真值仍缺失；alpha覆盖率不是补景质量分数。
- 源码、输入、GT、mask、历史失败与检查点保留，human_verdict=null，problem_solved=null。

复现脚本：仓库 `scripts/worldsim_v82/drivingeditor_shadow_rebuild_20261017/r15/diagnostics/`。依次用共享GPU锁运行audit_performance.py、audit_actors.py、audit_shadow_precision.py；它们只读4000检查点，不恢复长训。服务器结果为 `backend_restore_r15/paused_diagnostic_20261017/`，checkpoint保持在远端，不推Git。

failure_ledger_refs=[V82-F01,V82-F05,V82-F06]，failure_ledger_delta=[V82-F05,V82-F06]。新的两条记录分别为阴影实现的性能/数值故障和归属审计的时机/候选覆盖缺口，不是作者方法或高斯表征的普遍否定。

一手参考：[PyTorch CUDA同步与小算子调度](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#avoid-unnecessary-cpu-gpu-synchronization)、[activation checkpoint重计算](https://docs.pytorch.org/docs/stable/checkpoint.html)、[病态矩阵数值限制](https://docs.pytorch.org/docs/2.1/notes/numerical_accuracy.html#extremal-values-in-linalg)。本案主张以实际代码与测量为证据，文档只用于解释机制。
'''
(DOC/'README.md').write_text(doc_intro,encoding='utf-8')
severity_table=table(['严重性','类别 / 原因','实测证据','处理方向'],[[f'<b>{s}</b>',esc(c+'：'+cause),esc(ev),esc(action)] for s,c,cause,ev,action in causes])
html_report=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>R15暂停诊断：性能、数值与a6/a37</title><style>
*{{box-sizing:border-box}}body{{max-width:1280px;margin:30px auto;padding:0 24px;font:16px/1.75 system-ui,'Microsoft YaHei',sans-serif;color:#172b3c;background:#f6f8fa}}h1{{font-size:30px}}h2{{margin-top:36px;font-size:23px}}p{{margin:12px 0}}a{{color:#12619c}}.card{{padding:18px 22px;background:white;border:1px solid #c6d2dd;border-left:5px solid #287954;border-radius:6px}}.critical{{border-left-color:#bd3e34}}.muted{{color:#526778;font-size:14px}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{border:1px solid #b9c9d5;padding:10px 12px;text-align:left;vertical-align:top}}th{{background:#e7eef4}}td:first-child{{white-space:nowrap}}.scroll{{overflow-x:auto;margin:16px 0}}figure{{margin:22px 0;padding:12px;background:white;border:1px solid #c6d2dd}}img{{max-width:100%;height:auto}}figcaption{{font-size:13px;color:#526778}}code{{overflow-wrap:anywhere;font-size:13px}}details{{margin:14px 0;padding:12px;background:white}}summary{{cursor:pointer;font-weight:600}}.num td{{font-variant-numeric:tabular-nums}}@media(max-width:700px){{body{{padding:0 12px}}td,th{{min-width:130px}}h1{{font-size:25px}}}}
</style></head><body>
<h1>R15已暂停：性能、阴影数值与a6/a37归属诊断</h1><p class="muted">更新：{now} · 4k冻结检查点 · 49帧分层 / 46训练帧贡献 · 无subagent · 生产SGD新增0</p>
<div class="card critical"><b>结论：当前实现有明确问题，继续原样长训不合适。</b><p>阴影占同帧前向+反传的增量约{delta:.1f}%，同时存在FP32矩阵失稳和严重分块依赖。a6/a37删除后关阴影仍留有静态车形外观；3k归属修正实际没有执行任何split或owner转移。候选提速和质量修复尚未应用，训练保持暂停。</p></div>
<p>旧版从原始测量完成GN11 30k，再完成实例30k，三段训练进程合计约181.5分钟。两版初始化、目标和工作量不同，但本次同模型关阴影对照足以定位当前主要开销，不能把慢90倍全部解释为“从头训练”。</p>
<figure><img src="r15_diagnostic_evidence/architecture.svg" alt="输入、静态与43Actor、主相机光栅、阴影积分、RGB合成和诊断输出"><figcaption>实际组件与数据流；重点故障在阴影密度积分，分层通过独立反事实核查。</figcaption></figure>
<h2>暂停与检查点</h2><p>训练worker、flock与controller均已终止并释放GPU，相关心跳原本PAUSED；已写入禁止自动续训/关机的标记。保存点是<b>4000步</b>，最后日志4100，未落盘部分未冒称已恢复。检查点已读回，包含完整模型、13组非空Adam状态和各RNG；使用独立硬链接保留。</p><p><code>{esc(pause['checkpoint'])}</code></p><p>控制器增加启动暂停保护，实际执行返回0且没有启动worker；原控制器已备份。训练后端、损失、参数及CUDA ABI没有用诊断候选覆盖。旧控制器仍有初始step2断言，当前不能作为4k通用恢复入口。</p>
<h2>按严重性排序：变慢的原因与数值风险</h2>{severity_table}<p class="muted">行间有父子和重复关系，时间/加速比不可相加。整步对照含前向与反传；模块计时为同步无梯度前向；算子计数来自独立profiler。三种口径分别报告。</p>
<h2>同模型、同帧的诊断对照</h2>{table(['副本设置','重复数','前向 s','反传 s','合计 s','线性RGB最大差'],case_rows)}<p>raw107，4k、2,068,305点、43实例；PyTorch 2.2.2+cu118、RTX4080SUPER。完整训练日志3k–4k相邻区间中位数约15.66s/步，包含不同训练帧和周期事件，因此不能用本表单帧19.9s代替日志平均。lr=0 Adam状态更新在额外副本上单列0.033s，梯度裁剪0.035s；没有参数更新。</p>
<h2>阴影为什么不仅慢，而且算错</h2><p>物理协方差的最大条件数约2.38×10⁸。FP32构造的矩阵已有11个非正定，显式求逆后的对称部分有8个非正定。实现又把积分二次型中的a和横向距离clamp为正，错误可能以有限的巨大光学厚度出现，普通“loss和梯度有限”检查不会发现。</p><p>317个活动道路网格点中，分块变化造成的最大透射率差为<b>1.0</b>。下表只列最大差异的5个点，FP64参考重新从保存的旋转/尺度/位姿构造协方差，对全部有效Actor点做无AABB稠密积分：</p>{table(['网格索引','原chunk512','chunk2048','同模型FP64参考'],numerical_rows)}<p>即使后续运算改为FP64，继续使用原FP32逆矩阵仍把这些点算为几乎全遮挡；问题不只来自最后的求和。仅将c−b²/a改成残差二次型仍有58/66个接收点非有限，未作为修复采用。应先使用保留正定性的局部因子计算，再验分块、批次和梯度；当前没有以改大chunk、少太阳样本或关阴影掩盖错误。这一诊断不证明全部重建差距都由阴影数值错误造成。</p>
<h2>a6、a37：残影来源与归属约束缺口</h2><p>删除明确移除该owner全部高斯。关阴影后残影仍在，静态层单独渲染也保留对应车形；a6还包含其他Actor的贡献。a37锚帧的删除阴影开关在原车身区的线性RGB MAE仅0.000045，而残影很明显，主要来源已定位到静态层。覆盖率不等于道路恢复正确。</p>{table(['锚帧','原车身像素','删除后静态可见α均值','其他Actor可见α均值','静/动态深度差中位数','竞争门控拒绝率','删除阴影开关MAE'],actor_rows)}
<figure><img src="r15_diagnostic_evidence/a06_raw105_components.png" alt="a6同帧GT、全场景、独立资产、alpha、删除、关闭阴影、静态层与竞争门控"><figcaption>a6 raw105：独立资产的下部碎片明显；删除后的残留在静态及其他Actor分支中仍可见。</figcaption></figure>
<figure><img src="r15_diagnostic_evidence/a37_raw121_components.png" alt="a37同帧GT、全场景、独立资产、alpha、删除、关闭阴影、静态层与竞争门控"><figcaption>a37 raw121：静态层单独、关闭阴影仍有车形重影，不能当成真实投射阴影。</figcaption></figure>
<p>竞争约束的深度差上限是0.45m；a6锚帧的{g6_depth:.2f}m、a37的{g37_depth:.2f}m均超出范围。全49帧门控拒绝率中位数分别{g6_median:.2f}%和{g37_median:.2f}%。这类位于后方表面的外观重影没有被“同表面占用竞争”覆盖；把所有静态道路点直接转成Actor会引入新的错误。</p><p>3k审计紧接3000步opacity reset，最大单点α仅约0.01。两车没有任何点通过重复道路正证据门槛，真实split=0、Actor→static=0、static→Actor=0；仅新增459个真实测量池点（a6 188、a37 114）。4k的静态点中，source_candidate_owner为a6/a37的数量又都为0，现有静态转Actor规则对这两车没有候选入口。9k也与reset间隔重合。应调整有效审计时机并核查候选覆盖，而非把“审计脚本跑过”当成已经修正。</p><p>a6点数3453→123800，a37 10349→83130；中心越出软OBB分别29834、14573。软框约束只管中心，不保证高斯支撑干净。a6锚帧的已知车外raw alpha&gt;0.05支撑1479像素，a37为3049像素，其中也可能包含合理遮挡支撑，不能全部判道路。保留完整mask内外监督、未知域及真实测量，不按高度切轮胎/底盘。</p>
<details><summary>其他四个固定锚帧</summary>{''.join('<figure><img src="r15_diagnostic_evidence/'+x['folder']+'_components.png" alt="'+x['folder']+'分层反事实"><figcaption>'+x['folder']+'</figcaption></figure>' for x in a['anchors'] if x['folder'] not in ['a06_raw105','a37_raw121'])}</details>
<h2>下一步修复顺序与验证边界</h2><p>先修阴影数值并通过FP64/分块不变性/梯度测试；再把太阳与候选对批处理、缩减小checkpoint开销并复用光栅预处理；归属侧分别处理Actor碎片、静态外观残影和reset后的审计时机。每项先做固定检查点的可审查对照，再决定是否恢复训练。本次只诊断与发布代码，没有继续训练或宣称质量修复。</p><p>已覆盖逐步数据、几何、主光栅、阴影、反传、优化器、周期审计及锁等待的证据。后续material/joint实际训练未发生；此前当前点数的零更新探测约21.3/43.6s前后向，只能作风险提示。最终逐像素阴影成本未验收。FP64参考只验证一帧5个最大分块差异点；49帧分层包含已见开发QUERY，没有配对干净道路GT，不能据覆盖率证明补景正确。</p>
<h2>原始证据与一手参考</h2><p><a href="r15_diagnostic_evidence/performance.json">同步计时与算子计数</a> · <a href="r15_diagnostic_evidence/cpu_profile.txt">CPU调用剖析</a> · <a href="r15_diagnostic_evidence/shadow_precision.json">分块与FP64参考</a> · <a href="r15_diagnostic_evidence/actors/actors.json">98条逐车逐帧记录</a> · <a href="r15_diagnostic_evidence/pause_receipt.json">暂停/检查点凭据</a> · <a href="r15_diagnostic_evidence/old_baseline_training_timing.json">旧版从头训练时间</a> · <a href="r15_diagnostic_evidence/severity_table.json">严重性表JSON</a></p><p>实测代码：tracing.py的geometry_transmittance/_precision_integral、SunAmbient.visibility；raster.py的render_buffers/cuda_raster；model.py的train_image；competition.py；train.py中reset后保存再进入3k审计；owner_003000.py的来源候选与道路门槛。生产源保留，诊断源码随v82提交。</p><p class="muted">机制参考：<a href="https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#avoid-unnecessary-cpu-gpu-synchronization">PyTorch同步与小算子调度</a>、<a href="https://docs.pytorch.org/docs/stable/checkpoint.html">checkpoint重计算</a>、<a href="https://docs.pytorch.org/docs/2.1/notes/numerical_accuracy.html#extremal-values-in-linalg">病态矩阵数值限制</a>。本案结论来自实际代码与测量，引用不代替实测。failure refs：V82-F01 / F05 / F06；human_verdict=null，problem_solved=null。</p></body></html>'''
(OUT/'r15_training_diagnosis.html').write_text(html_report,encoding='utf-8')
(DOC/'report.html').write_text(html_report,encoding='utf-8')
target=DOC/'r15_diagnostic_evidence'
target.mkdir(exist_ok=True)
for filename in ['architecture.svg','performance.json','cpu_profile.txt','shadow_precision.json','pause_receipt.json','PAUSE_REQUESTED.json','execution_policy.json','old_baseline_training_timing.json','severity_table.json']:
    shutil.copy2(E/filename,target/filename)
for image in E.glob('*_components.png'):shutil.copy2(image,target/image.name)
(target/'actors').mkdir(exist_ok=True)
shutil.copy2(E/'actors/actors.json',target/'actors/actors.json')
print('Report:',OUT/'r15_training_diagnosis.html')
print('F+B:',base,off,'shadow increment',delta)
