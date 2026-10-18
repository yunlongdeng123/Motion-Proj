"""User-facing finite experiment plan and measured baseline engineering evidence."""
from pathlib import Path
import json,shutil,html,datetime,re
W=Path(__file__).parent;O=W.parents[1]/'outputs/baseline_engineering_20261018'
def read(n):
    p=W/n;return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None
def main():
    O.mkdir(parents=True,exist_ok=True)
    for n in ['channel_gate.json','cpu_policy_audit.json','stage_input_audit.json','matched5_input_receipt.json','experiment_contract.json','prior_trace_receipt.json','queue_status.json','full_queue_status.json','pilot_results.json','source_manifest.json','review_queue_status.json','isolated_control_contract.json','stage_binding_receipt.json','constraint_paths.json','live_GN11_status.json','review_CPU_binding_receipt.json','static_growth_receipt.json','attribution_queue_status.json']:
        if (W/n).exists():shutil.copy2(W/n,O/n)
    shutil.copytree(W/'prior_trace',O/'prior_trace',dirs_exist_ok=True,ignore=shutil.ignore_patterns('loader'))
    shutil.copy2(W/'static_provenance_raw128.png',O/'static_provenance_raw128.png')
    svg='''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1240 235"><defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#55728c"/></marker></defs><rect width="1240" height="235" rx="14" fill="#eff5fa"/><g font-family="Microsoft YaHei,Segoe UI,sans-serif" text-anchor="middle" fill="#18334c" font-size="17"><g fill="#fff" stroke="#9cb3c6" stroke-width="2"><rect x="20" y="45" width="205" height="105" rx="9"/><rect x="270" y="45" width="205" height="105" rx="9"/><rect x="520" y="45" width="205" height="105" rx="9"/><rect x="770" y="45" width="205" height="105" rx="9"/><rect x="1020" y="45" width="200" height="105" rx="9"/></g><text x="122" y="80">原始 49 帧测量</text><text x="122" y="112">两阶段一致的五身份</text><text x="372" y="80">工程与约束验收</text><text x="372" y="112">通道 / 梯度 / 状态</text><text x="622" y="80">重训 GN11 30K</text><text x="622" y="112">五身份前景与真实先验</text><text x="872" y="80">五实例 30K</text><text x="872" y="112">验明原约束再启动</text><text x="1120" y="80">49 帧同轴对照</text><text x="1120" y="112">动态 / 静态 / Delete</text><text x="620" y="201" font-size="15">旁路：旧三身份起点的四组 × 300 步，仅用于诊断阶段切换、原 body α 约束及身份归一化</text></g><g stroke="#55728c" stroke-width="2.5" marker-end="url(#a)"><path d="M229 97H261"/><path d="M479 97H511"/><path d="M729 97H761"/><path d="M979 97H1011"/></g></svg>'''
    svg=svg.replace('四组 × 300 步','原四组＋纠正孤立body一组，各300步').replace('原 body α 约束及身份归一化','body α 接线及身份归一化')
    (O/'architecture.svg').write_text(svg,encoding='utf-8')
    q=read('queue_status.json') or {};full=read('full_queue_status.json') or {}
    inp=read('stage_input_audit.json');g=read('channel_gate.json');trace=read('prior_trace_receipt.json')
    now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
    rows=[('优先修复','两阶段身份集合不一致','已确认：GN11只有旧3身份；a6/a37到实例阶段才加入，canonical前景覆盖0/45418和0/555950像素。','从原始测量重训五身份GN11，再接同一五身份实例阶段；不再直接沿用旧3身份GN11当正式五身份起点。'),
      ('继承性阻塞','原 raw-body α 约束被替换','原生 DrivingEditor 与 GN11有100×孤立动态body α整图MSE；刚体适配中权重设0，换为100×可见ID α均值，遮挡语义不同。','单独恢复原项做短实验；明确是恢复原约束，不能把添加新损失与修工程混写。'),
      ('已纠正短测定义','先前body控制臂接线仍是可见union','实际five_raw_body只恢复了权重100，state仍指向dynamic_visible_alpha。此前“恢复孤立body”的描述无效；前四臂记录的opacity项都是可见union MSE。','旧臂及检查点保留，改列为可见union控制；另排five_raw_body_isolated 300步，只换成renderer真实孤立dynamic_alpha并登记状态域。'),
      ('继承性阻塞','固定拓扑与原增密/剪枝不同','刚体适配固定点拓扑；配置里存在原增密和大点剪枝参数，不代表实例阶段实际调用。a6旧5最大σ约3.29米。','逐项列出原方法、GN11及刚体适配的实际调用/梯度/生效阶段；未闭合前不得称完整原生方法。'),
      ('继承性核对','原生剪枝也有阶段边界','原配置>500起每100步且<15000调用，点数上限也参与门控；大点判据在3000后启用。契约化世界尺度要用scene.cameras_extent及世界中心，不能用初始化为1的spatial_lr_scale或Actor局部中心代替。','恢复调用必须保留实际调度与坐标语义；不能把“原方法有剪枝”扩大成全程保证没有大点。'),
      ('显式适配差异','SH升阶与相机采样','GN11读取2000步SH升阶，实例写死1000；GN11每49帧无放回采样，实例每步有放回随机。旧3/旧5共享这些设置。','纳入正式配置取舍；这些差异本身不能解释3→5的新增退化，也不能混称GPU通道错误。'),
      ('比较混杂','身份均值从3变5','100×mean_K导致每身份系数100/3→100/5，降低40%。这不是通道索引算错。','单独测试五身份100×5/3，保持其余损失不变。'),
      ('本次排除','第二批身份通道串线或断梯度','实际CUDA：五身份2批/5批、换序、单ID参考及union通过；真实107/128/137前向一致，128逐ID梯度通过。','保留原始receipt；合成五身份均有非零opacity梯度，真实128的a5不可见零梯度明确保留。'),
      ('本次排除','静态渲染适配数值错误','49帧原生静态renderer与适配器最大RGB差约2.4e-7。raw128恶化确实出现在旧5参数中。','检查无额外GT mask乘图；保留native/GN11/旧3/旧5逐帧原尺寸图。'),
      ('撤回误判','背景学习率10倍错误','较早本地快照曾导致误判；实际旧5归档train在调度后补乘0.1，实际调用路径4节点验收通过。','不是本次已证实故障，不往新实验里混入一个虚构的学习率修复。')]
    table=''.join('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in row)+'</tr>' for row in rows)
    arms=[('three_replay','3','0','100','复核原三身份路径'),('five_replay','5','0','100','复核当前五身份路径'),('five_raw_body（旧命名）','5','100，可见union','100','实际仅加入可见union α约束，不能当孤立body恢复'),('five_ID_weight','5','0','166.6667','只恢复与旧3相同的每ID系数'),('five_raw_body_isolated','5','100，孤立body','100','纠正后的原始孤立body α控制；与five_replay相比只恢复此项')]
    armrows=''.join('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in row)+'</tr>' for row in arms)
    pilot=read('pilot_results.json')
    measured=''
    if pilot:
        measured='<h3>300步实际通路诊断</h3><p>以下都是step300、raw91。前四臂opacity日志名字虽含raw_body，实际张量是全场景可见union，不能据此判断孤立body约束。三身份与五身份监督域不同；不同权重的总loss不能排序为算法优劣。全49帧与恢复另有排队检查。</p><table><tr><th>臂</th><th>实际opacity域</th><th>该域 α MSE（未加权）</th><th>ID α MSE（均值）</th><th>静态RGB原始损失</th></tr>'
        for arm in pilot['arms']:
            m=arm.get('last_metric')
            if m:
                measured+='<tr><td>'+html.escape(arm['arm'])+'</td><td>'+('孤立body' if arm['arm']=='five_raw_body_isolated' else '全场景可见union')+'</td>'+''.join('<td>'+format(m[k],'.6f')+'</td>' for k in ['raw_body_opacity_loss','instance_mask_loss','static_rgb_loss'])+'</tr>'
        measured+='</table><p><a href="pilot_results.json">实际结果与域更正receipt</a>。撤回此前“已恢复原孤立body且MSE下降”的解释；原数据仍保留。任何一项MSE降低都不等于道路或资产干净。</p>'
    doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>R15：旧五身份基线工程验收与重跑</title><style>body{font:16px/1.65 Segoe UI,Microsoft YaHei,sans-serif;background:#f4f7fa;color:#172d41;margin:0}main{max-width:1320px;margin:30px auto;padding:0 22px}h1{font-size:29px}h2{font-size:22px;margin-top:32px}a{color:#086aab}section,.card{background:white;border:1px solid #cedce7;padding:20px;border-radius:10px;margin:18px 0}.warn{background:#fff1d9;border-left:5px solid #c07810;padding:15px}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}figure{margin:0}figure img{width:100%;display:block;background:#182535}figcaption{padding:7px;background:#e9f1f7}table{border-collapse:collapse;width:100%}th,td{border:1px solid #cbd9e4;padding:10px;vertical-align:top}th{background:#eaf1f6;text-align:left}.controls{display:flex;gap:14px;align-items:center;flex-wrap:wrap}input[type=range]{flex:1;min-width:220px}small{color:#527087}code{overflow-wrap:anywhere}.proof{width:100%}@media(max-width:900px){.grid{grid-template-columns:repeat(2,1fr)}td,th{font-size:14px}}@media(max-width:550px){.grid{grid-template-columns:1fr}}</style><main>
<h1>R15：旧五身份基线工程验收与重跑</h1>
<p class="warn">当前“旧 GN11 30K＋五实例30K”未通过工程与编辑质量验收，保留为历史诊断结果。不能据此断言原算法不行，也不能把旧三身份静态层里的完整车辆当作干净道路真值。</p>
<p>快照时间：TIME。<a href="../identity_comparison_20261018/index.html">已有三模型同步动态／静态／Delete视频</a> · <a href="../r15_repair_4k_20261018/tristate/index.html">修复R15 4K三维页</a>。</p>
<img class="proof" src="architecture.svg" alt="原始测量和一致身份经工程验收、GN11、实例训练，输出同步分支对照">
<section><h2>当前已确认什么</h2><table><thead><tr><th>级别/状态</th><th>项目</th><th>实际证据</th><th>处理</th></tr></thead><tbody>ROWS</tbody></table>
<p>原约束证据：<a href="https://github.com/WangXu-xxx/DrivingEditor/blob/560d57c3825ed2bdbc6a6602e0011ad51e8da60e/train.py#L441-L450">原生动态body α损失</a>；<a href="https://github.com/WangXu-xxx/DrivingEditor/blob/560d57c3825ed2bdbc6a6602e0011ad51e8da60e/train.py#L480-L531">原生增密/剪枝及优化路径</a>。当前系统应称“GN11＋刚体身份适配”，不能省略这些差异称为完整原生复现。</p>
<p><a href="stage_input_audit.json">阶段输入逐像素审计</a> · <a href="channel_gate.json">实际CUDA通道与梯度</a> · <a href="cpu_policy_audit.json">实际学习率调用路径</a> · <a href="prior_trace_receipt.json">49帧native静态渲染对照</a></p></section>
<section><h2>同帧追溯：观察图 → GN11静态起点 → 旧3静态 → 旧5静态</h2>
<p>这里显示实际完整帧PNG。GN11起点已包含新增车辆的静态外观；旧五身份阶段才令它们退出静态RGB监督。该链路确认了退化发生的阶段，尚不能单凭图像把全部原因归到某个损失。</p>
<div class="controls"><button id="prev">上一帧</button><input id="frame" type="range" min="89" max="137" value="128"><button id="next">下一帧</button><label>raw <input id="raw" type="number" min="89" max="137" value="128" style="width:70px"></label><b id="frameLabel">128 / 49帧</b></div>
<div class="grid">PANELS</div><p><small>统一raw帧号，全部图像解码完成后一起切换。原尺寸480×320；点击图像可查看文件。没有配对干净道路GT，没有把车辆投影区域直接清空或涂图。</small></p>
<details><summary>raw128完整溯源合图</summary><img class="proof" src="static_provenance_raw128.png"></details></section>
<section><h2>已安排的有限实验</h2><p>四组各300步，合计上限1200步；seed0、相同旧GN11起点、原30K学习率时间表、固定拓扑、完整49帧。它们用于因果诊断，不是新正式基线。原GN11/旧3/旧5/R15及修复4K检查点均保留。</p><table><tr><th>诊断臂</th><th>身份数</th><th>raw-body α权重</th><th>ID α均值权重</th><th>唯一比较目的</th></tr>ARMS</table>
<p>短实验状态：<code>PILOT</code>。新正式GN11队列：<code>FULL</code>。</p>
<p>正式基线重新从原始测量训练GN11 30K，两阶段使用相同五身份二值域。已完成CPU实际49相机、0.1米尺度、SGCM地面拟合及掩码一致性检查；五组真实LiDAR先验为21358／15193／2444／11625／98452点。后续五实例30K需先核对原约束继承、损失梯度和Adam/RNG恢复，不会直接复用当前有争议的约束接法。</p>
<p>掩码来源透明：a5/a26/a18沿用旧真实标签，a6/a37沿用此前核对过的原生SAM3完整提议；不是R15融合三值或已训练参数。480×320标签最近邻扩展给原1920×1280读取器，并严格验证回到训练分辨率后五身份逐像素一致，没有新增高分辨率测量。</p>
<p><a href="experiment_contract.json">四臂固定合同与源码hash</a> · <a href="matched5_input_receipt.json">五身份一致输入receipt</a> · <a href="queue_status.json">短实验状态</a> · <a href="full_queue_status.json">正式两阶段队列状态</a></p></section>
<section><h2>正式比较前的验收门槛</h2><ol><li>实际使用源码、二进制、身份顺序、坐标尺度、帧号、训练与显示图像域全部可追溯。</li><li>原生约束按实际调用、有效域、归一化、梯度和作用参数逐项对照；有意改动单独登记，不混入工程修复。</li><li>检查点恢复必须覆盖静态/实例/环境、Adam、采样器及Python/NumPy/Torch/CUDA RNG。中断先查已用步数，不盲目重跑。</li><li>同49帧五目标的动态、静态、逐身份独立资产与删除前后共同验收；报告真实米下的中心/支撑外溢、保体、静态车形和道路细节。</li><li>训练loss有限、框外率降低或脚本完成，都不等于资产干净。旧3是复现控制，不是干净道路GT；整体重建分数也不能替代删除检查。</li></ol>
<p>工程门槛是有限范围的证据，不宣称“所有工程问题已经排完”。同一训练窗口，无独立编辑GT；human_verdict=null。浏览器file://自动验收受到安全策略限制，本页只记录真实图像、媒体与JS检查，不冒称浏览器操作已通过。</p></section></main>
<script>let requested=128;const keys=['observed','GN11_static','old3_static','old5_static'];async function seek(n){n=Math.max(89,Math.min(137,Math.round(Number(n)||128)));requested=n;const fid='0008'+String(n).padStart(3,'0');let urls=keys.map(k=>'prior_trace/'+fid+'/'+k+'.png');try{await Promise.all(urls.map(url=>new Promise((ok,no)=>{let im=new Image();im.onload=ok;im.onerror=no;im.src=url})));if(requested!==n)return;document.querySelectorAll('[data-panel]').forEach((im,i)=>{im.src=urls[i];im.parentElement.href=urls[i]});document.getElementById('frame').value=n;document.getElementById('raw').value=n;document.getElementById('frameLabel').textContent=n+' · '+(n-88)+'/49';}catch(e){document.getElementById('frameLabel').textContent='图像读取失败 raw'+n}}document.getElementById('frame').oninput=e=>seek(e.target.value);document.getElementById('raw').onchange=e=>seek(e.target.value);document.getElementById('prev').onclick=()=>seek(requested-1);document.getElementById('next').onclick=()=>seek(requested+1);seek(128);</script></html>'''
    labels=['观察RGB','GN11 30K静态起点（旧3动态集合）','旧三实例30K静态层','当前旧五实例30K静态层']
    keys=['observed','GN11_static','old3_static','old5_static']
    panels=''.join(f'<figure><a href="prior_trace/0008128/{key}.png"><img data-panel src="prior_trace/0008128/{key}.png" alt="{label}"></a><figcaption>{label}</figcaption></figure>' for key,label in zip(keys,labels))
    doc=doc.replace('TIME',now).replace('ROWS',table).replace('ARMS',armrows).replace('PANELS',panels).replace('PILOT',html.escape(q.get('state','NOT_COLLECTED'))).replace('FULL',html.escape(full.get('state','NOT_COLLECTED')))
    doc=doc.replace('四组各300步，合计上限1200步；','原四组各300步已完成，合计1200步。另补纠正接线的孤立body控制300步，总诊断训练上限1500步；')
    doc=doc.replace('四臂固定合同与源码hash','原四臂历史合同与源码hash（body域更正见下）')
    review=read('review_queue_status.json') or {}
    live=read('live_GN11_status.json') or {}
    live_section='<section><h2>实际进度与更正后的验收队列</h2><p>五身份一致GN11：'+str(live.get('iteration','待取回'))+'/30000，状态 <code>'+html.escape(live.get('status','未知'))+'</code>。该数字来自原生训练状态快照，不是完成或质量通过。</p>'
    live_section+='<p>后续队列：<code>'+html.escape(review.get('state','未登记'))+'</code>。等当前GN11释放GPU后，顺序完成纠正孤立body 300步、两臂真实损失梯度及恢复检查、五臂全49帧分支/独立资产/逐身份删除。恢复检查只在独立审计副本执行共6次更新，不计入训练、不回写检查点。正式新实例更新仍为0。</p>'
    live_section+='<p>实际两阶段坐标变换最大差0、时间最大差1.11×10⁻¹⁶，49帧身份与二值union一致。框软损失在归一化坐标的平方域计算，.1尺度使数值为米平方的.01倍；这是损失尺度说明，不能单凭此认定坐标转换错误。</p>'
    live_section+='<p><a href="stage_binding_receipt.json">实际GN11与实例绑定</a> · <a href="constraint_paths.json">逐项原约束／适配调用矩阵</a> · <a href="isolated_control_contract.json">body控制更正合同</a> · <a href="review_queue_status.json">后续有限队列</a> · <a href="live_GN11_status.json">原生GN11状态快照</a> · <a href="review_CPU_binding_receipt.json">审计更新代码CPU绑定检查（未代表GPU通过）</a></p></section>'
    growth=read('static_growth_receipt.json')
    if growth:
        parameters=growth['static_parameters']
        growth_rows=''
        for actor in [6,37]:
            key='a%d_initial_OBB_region'%actor
            count=parameters['GN11'][key]['sigma_max_axis_m']['count']
            values=[f'a{actor} 初始空间区域',str(count)]
            values += [format(parameters[m][key]['sigma_max_axis_m']['median'],'.3f') for m in ['GN11','old3','old5']]
            values += [format(parameters[m]['same_row_changes_from_GN11'][key]['max_axis_sigma_ratio']['maximum'],'.2f') for m in ['old3','old5']]
            growth_rows+='<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in values)+'</tr>'
        attribution=read('attribution_queue_status.json') or {}
        live_section+='<section><h2>静态点同一参数行追踪：不是所有点一起变大</h2><p>三套静态点数均为915840，保留全部原点、无拓扑重排的实际合同支持逐行比较。以GN11 raw128时中心落在a6/a37真实框内的静态点固定选择空间区域，追踪同一批参数到旧3和旧5。此处不是Actor资产归属，也未覆盖全部跨框支撑。</p>'
        live_section+='<table><tr><th>固定诊断区域</th><th>静态点数</th><th>GN11 σ中位/m</th><th>旧3 σ中位/m</th><th>旧5 σ中位/m</th><th>旧3单点最大尺度比</th><th>旧5单点最大尺度比</th></tr>'+growth_rows+'</table>'
        live_section+='<p>σ为最大轴尺度；比值相对同一GN11原始参数行。多数区域点缩小与少数点极端放大可以同时发生，不能把蓝白条带简单归为整体尺度爆炸，也不能据中心位于车框就把道路点转成车。全场景尺度还受距离契约化影响，大σ不自动等于坏点。</p>'
        live_section+='<p>只读真实贡献追踪队列：<code>'+html.escape(attribution.get('state','待登记'))+'</code>。将在已登记工程复核之后，核对GN11/旧3/旧5在raw107、128、137的a6/a37车身与下部区域：用原生rasterizer的颜色梯度得到逐Gaussian αT，并与实际alpha和未截断RGB求和配平；另核对原生静态RGB参数梯度。全部零训练更新，结果尚未用于宣称原因或质量通过。</p>'
        live_section+='<p><a href="static_growth_receipt.json">完整CPU静态参数统计</a> · <a href="attribution_queue_status.json">零更新贡献追踪状态</a></p></section>'
    doc=doc.replace('<section><h2>正式比较前的验收门槛</h2>',live_section+'<section><h2>正式比较前的验收门槛</h2>')
    doc=doc.replace('<section><h2>正式比较前的验收门槛</h2>',measured+'<section><h2>正式比较前的验收门槛</h2>')
    (O/'index.html').write_text(doc,encoding='utf-8')
    status=dict(at=now,old5_accepted_as_clean_baseline=False,new_protocol='matched-five-ID GN11 30K then verified five-instance 30K',
        old5_engineering_cleared=False,channel_gate=g['status'],native_static_gate=trace['status'],stage_input_audit=inp['status'],pilot=q,full_queue=full,review_queue=review,attribution_queue=read('attribution_queue_status.json'),live_GN11=live.get('iteration'),isolated_body_restoration_claim_for_original_control_retracted=True,human_verdict=None)
    (O/'status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    banner='<p id="baseline-engineering-warning" style="background:#5d3314;color:#fff1db;padding:16px;border:1px solid #e6a45b;border-radius:8px">旧GN11＋五实例结果尚未通过工程与编辑质量验收。已发现GN11与实例阶段的身份集合不一致；本页保留历史真实输出。<a style="color:#fff" href="../baseline_engineering_20261018/index.html">查看重新安排的工程验收与两阶段重跑</a>。</p>'
    for p in [O.parent/'identity_comparison_20261018/index.html',O.parent/'identity_comparison_20261018/sync_player.html']:
        s=p.read_text(encoding='utf-8')
        if 'id="baseline-engineering-warning"' in s:s=re.sub(r'<p id="baseline-engineering-warning".*?</p>',banner,s,count=1,flags=re.S)
        else:s=s.replace('<body>','<body>'+banner,1)
        p.write_text(s,encoding='utf-8')
    print(str(O/'index.html'))
if __name__=='__main__':main()
