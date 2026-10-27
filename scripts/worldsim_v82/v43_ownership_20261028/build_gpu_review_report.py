"""Current first-scene GPU review, never mark diagnostics as accepted quality."""
from pathlib import Path
import json,re,html,subprocess,sys
B=Path(__file__).resolve().parents[2];O=B/'outputs/V43_Ownership_20261027/gpu_v43';S=B/'outputs/GS_Scene_Diagnostics_20261027/14705303724557273004_3105_000_3125_000';W=Path(__file__).parent
ab=json.loads((O/'first_review_ablation_delivery_v1/index.json').read_text());cal=O/'first_review_opacity_delivery_v1';done=(cal/'delivery_validation.json').exists()
stats=[]
if done:
    for arm in ['legacy','balanced','field']:
        d=json.loads((cal/arm/'summary.json').read_text());rows=[r for r in d['records'] if r['split']=='holdout'];stats.append(dict(arm=arm,holdout_frames=len(rows),inside_alpha=sum(r['inside_alpha'] for r in rows)/len(rows),outside_alpha=sum(r['outside_alpha'] for r in rows)/len(rows),visible_fraction=sum(r['visible_fraction'] for r in rows)/len(rows)))
boundary=O/'first_review_boundary_delivery_v1';boundary_done=(boundary/'delivery_validation.json').exists()
if boundary_done:
    rows=[r for r in json.loads((boundary/'boundary/summary.json').read_text())['records'] if r['split']=='holdout'];stats.append(dict(arm='boundary',holdout_frames=len(rows),inside_alpha=sum(r['inside_alpha'] for r in rows)/len(rows),outside_alpha=sum(r['outside_alpha'] for r in rows)/len(rows),visible_fraction=sum(r['visible_fraction'] for r in rows)/len(rows)))
report=dict(context='14705303724557273004_3105_000_3125_000',ablation_frames=31,ablation_images=310,ablation_all_control_byte_max=ab['all_control_byte_max'],opacity_status='formal10K complete' if done else 'running: three arms formal10K after real save/evaluate/restore smoke',opacity_stats=stats,formal_updates_per_arm=10000,extra_smoke_updates_per_arm=110,changed_geometry=False,changed_DC=False,identity_learning=False,completion=False,quality_accepted=False,human_verdict=None)
report['completed_opacity_arms']=4 if boundary_done else 3 if done else 0
(O/'first_gpu_review_status.json').write_text(json.dumps(report,indent=2),encoding='utf8')
text='''# V43 第一场景：原生剥离与通用opacity改进

![组件](gpu_review_architecture.svg)

本轮仅第一场景；第二未新增训练，第三继续暂缓。无编辑主通路继续保留原V1输出，模型和历史结果不覆盖。

## 原生剥离：候选集合尚未达到干净分离

已完成31帧×2种带宽×5种控制，共310张实际原生RGB；未抑制控制与旧Actor10K输出最大PNG字节差0。共享HTML新增6个剥离组：去unknown、去车辆14–19、两者去除，分别0/5px；原删除动态资产基线保留。车辆组包含其他车辆，属于全局语义剥离控制，不是仅删除目标。

raw98/101/109/113/114排查图中，两者去除后车身主体明显减少，底部/原位仍有暗色残留；同时背景出现大范围空缺与异常曝光/颜色，不能宣称完成干净车辆分离。原生合成中的环境图和此前被遮挡的表达可能在剥离后暴露，进一步的类别贡献诊断另列。没有把这些图当成完成的背景补全。

几何显示现在使用每个剥离组保存的固定0/5px与20%规则，和左侧原生RGB对应。右侧手动预览只会进一步隐藏，不会改写固定实验。RGB、alpha、原语义与原动静态分支仍分别保留。

## opacity：同起点、冻结结构与颜色的三组控制

所有组从相同Actor10K重新开始，各增加10K优化；不是从各自smoke继续正式训练。固定6241个高斯及其xyz、尺度、旋转、DC、static和环境。完整内外掩码监督均保留。33帧用于校准、16帧检查；基础重建与种子已见过49帧，因此只称校准留出，不称独立重建泛化。

- legacy：只优化原opacity，使用原整图alpha MAE×0.1。
- balanced：只优化原opacity，使用掩码内MAE×0.05＋掩码外MAE×0.05；这是显式重新分配权重，不声称与旧0.1同强度。
- field：冻结原opacity，用Actor局部共享MLP预测有界q；o'=1-(1-o)^q。q在0.25–4内，由仅训练帧的可见车身/语义支持可靠度门控，无支持时q=1。输入没有case、帧号、编辑角度；加入logq≈0与局部平滑正则各0.001。表示容量与正则改变单列，不能把field与balanced当单因素比较。

三组RGB训练目标、稀疏深度约束、采样序列与起点一致；检查覆盖、外泄漏、合成遮挡、转向与横移，不依赖RGB分数。原固定opacity增强仍作为旧控制保留。校准模型不回写原checkpoint或语义owner。

## 工程与执行

三组分别完成真实100更新smoke及恢复到110的下一阶段验证，覆盖前后向、finite loss/梯度、保存、49视角评价、原生编辑入口及完整Actor/MLP/Adam/采样与所有RNG恢复。每组额外110更新单列。正式方法和10K预算保持不变。本轮为分钟级短实验，不是小时级长跑。

一个父控制器持有全局GPU锁，按实测显存准入，最多两个已登记子进程并发。其他工作不能误入同一GPU租约。CUDA allocator与NVML采样分开记录；首轮实际并发NVML约3.6GB、利用率约93%。最初控制器因gpu_slot导入路径在训练前失败，日志保留，修复后重启控制器，没有丢失训练更新。

尚未学习unknown身份分类、拆分静态混合高斯或联合补全背景，也没有证明联合优化优于简单流水线。failure_ledger_refs=[V82-F01]，failure_ledger_delta=none。
'''
if stats:
    text+='\n## 正式10K校准留出结果\n\n|组|车身内部alpha均值|掩码外alpha均值|车身合成可见比例|\n|---|---:|---:|---:|\n'
    text+=''.join('|'+r['arm']+'|'+format(r['inside_alpha'],'.5f')+'|'+format(r['outside_alpha'],'.5f')+'|'+format(r['visible_fraction'],'.5f')+'|\n' for r in stats)
    text+='\n这是原始观测掩码上的覆盖/泄漏/遮挡统计，不是编辑RGB真值或正确身份率；转向侧面仍需逐帧看图。\n'
else:text+='\n当前：正式对照运行中，完成前不写改进结论。\n'
if (O/'residual_class_probe.json').exists():
    d=json.loads((O/'residual_class_probe.json').read_text());r=next(r for r in d['records'] if r['raw']==113 and r['radius']==5);w=r['class_alpha_mass_in_original_target_mask'];fraction=w['road']/sum(w.values());text+='\n## 剥离后残留的类别贡献\n\nraw113、5px、去车辆＋unknown后，原目标车身掩码内剩余Gaussian贡献约'+format(fraction*100,'.1f')+'%仍携带原road标签，另有terrain/sky；剩余合成alpha均值约'+format(r['remaining_composite_alpha_in_target_mask'],'.3f')+'。加权中心深度约'+format(r['remaining_alpha_weighted_center_depth_m'],'.1f')+'m，原目标实测LiDAR中位约'+format(r['original_target_LiDAR_median_m'],'.1f')+'m。这个统计覆盖原车身掩码，不能冒充对每个暗残影像素的精确分类，也不能把中心深度当表面交点。\n\n因此，仅删vehicle/unknown标签不足以完成干净分离。保留的背景类也在目标区域参与成像，支持静态混合支撑／车辆外观污染的解释。仍需区分几何归属与外观污染，不把这些road行直接整批迁入车辆。阴影没有重标到车身，车底问题继续暂缓。\n'
if (O/'opacity_view_audit.json').exists():
    records=json.loads((O/'opacity_view_audit.json').read_text())['records'];text+='\n## 同一姿态、固定诊断域的覆盖与边缘代价\n\n|组|31帧转向独立alpha均值|原视角车身外5px alpha均值|\n|---|---:|---:|\n'
    for name in ['base','fixed2','legacy','balanced','field']:
        tr=[r for r in records if r['case']=='turn_left15' and r['variant']==name];no=[r for r in records if r['case']=='no_edit' and r['variant']==name];text+='|'+name+'|'+format(sum(r['isolated_alpha_on_common_domain'] for r in tr)/len(tr),'.5f')+'|'+format(sum(r['outside5px_alpha'] for r in no)/len(no),'.5f')+'|\n'
    text+='\n转向域固定为同姿态原Actor10K独立alpha>.1的像素，不是编辑GT。固定增强覆盖最高但外圈代价更大；两域平衡取得部分覆盖改善，边缘也增加；共享场当前几乎保持原状，没有额外收益证据。generic渲染接口与已保存正式RGB逐字节一致，V1无编辑路径浮点差0。\n'
if boundary_done:
    text+='\n## 第四组：按对象投影大小划分完整外域\n\n观察到balanced边缘外泄漏后，追加一个配对boundary控制：内域权重.05；外域分为邻近带与其余外域，各.025，整个掩码外域均覆盖。邻近半径由clip(round(.1*sqrt(body像素数)),2,12)决定，不使用case/帧号/角度触发。也是同Actor10K起点、冻结几何/DC、额外10K，另100＋10恢复smoke；这是复用同一开发留出做的探索性跟进，不是新的独立测试。\n'
    if (O/'boundary_view_audit.json').exists():
        rr=json.loads((O/'boundary_view_audit.json').read_text())['records'];text+='\n|组|31帧转向同域alpha|原视角外5px alpha|\n|---|---:|---:|\n'
        for name in ['base','balanced','boundary']:
            t=[r for r in rr if r['case']=='turn_left15' and r['variant']==name];n=[r for r in rr if r['case']=='no_edit' and r['variant']==name];text+='|'+name+'|'+format(sum(r['isolated_alpha_on_common_domain'] for r in t)/len(t),'.5f')+'|'+format(sum(r['outside5px_alpha'] for r in n)/len(n),'.5f')+'|\n'
    text+='\n第四组在16帧校准留出上，内部alpha约0.930，外域alpha约0.00171；对比legacy的0.909 / 0.00220，同时改善了源观测域内的覆盖和外泄漏。编辑视角固定域包含边缘，整体alpha下降不自动等于核心车身变淡，需再分核心和边缘核对。\n'
if (O/'opacity_core_fringe_CPU.json').exists():
    rr=json.loads((O/'opacity_core_fringe_CPU.json').read_text())['summary'];text+='\n## 转向核心和边缘分别核对\n\n固定同姿态原Actor10K alpha>.8为核心，.1<alpha≤.8为边缘；取自已保存8bit地图，量化界限1/255，不是编辑GT。\n\n|组|31帧核心alpha|边缘alpha|\n|---|---:|---:|\n'
    for r in rr:
        if r['case']=='turn_left15':text+='|'+r['variant']+'|'+format(r['core_alpha'],'.5f')+'|'+format(r['fringe_alpha'],'.5f')+'|\n'
    text+='\n当前优先保留简单的分区监督改动做下一轮核验，共享MLP没有显示额外收益，不设为默认。静态车影、遮挡和未见侧面仍未解决；不声称跨场景泛化已证实。\n'
(W/'FIRST_GPU_REVIEW.md').write_text(text,encoding='utf8');(O/'FIRST_GPU_REVIEW.md').write_text(text,encoding='utf8')
# Reuse established hierarchical layout, update components instead of a flat list.
svg=(O/'opacity_unknown_architecture.svg').read_text(encoding='utf8').replace('先核对unknown，再验证覆盖与剥离','原生剥离诊断 + 共享opacity校准').replace('归属与unknown诊断（CPU已完成）','原生语义剥离（31帧已完成）').replace('通用opacity校准（GPU待验证）','通用opacity校准（'+('10K完成' if done else 'GPU运行中')+'）').replace('已完成中心点预览；精确RGB合成需要GPU','同标签精确RGB合成 + alpha；主体减少仍有残留').replace('当前新增内容仅CPU处理，左图RGB未重渲染；精确剥离与校准没有启动','原生RGB / alpha / 实际几何同步；原模型与V1默认通路保留').replace('第一场景校准留出 + 第二场景固定规则迁移','第一场景33帧校准 / 16帧检查；仅更新opacity').replace('车身内外alpha分别归一化，均保留监督','同起点：legacy / balanced / shared field').replace('先固定几何 / DC / 点数，检验opacity因素','几何 / DC / 点数冻结，内外掩码监督均保留')
if boundary_done:svg=svg.replace('通用opacity校准（10K完成）','覆盖与边缘改进（四组10K）').replace('同起点：legacy / balanced / shared field','同起点：整图 / 两域 / 分区 / 共享场').replace('证据约束的Actor局部共享校准场','按对象大小分区监督；共享场保留对照').replace('有界 / 平滑 / 跨时间共享；无支持时回到q=1','完整外域监督；几何 / 颜色 / 点数冻结')
(O/'gpu_review_architecture.svg').write_text(svg,encoding='utf8')
links=[('原生去车辆＋U / 0px','v43_remove_hide_both_0px'),('原生去车辆＋U / 5px','v43_remove_hide_both_5px'),('原生只去U / 5px','v43_remove_hide_unknown_5px')]
if done:links += [(arm+' 转向再10K','v43_opacity_'+arm+'_turn_left15') for arm in ['legacy','balanced','field']]
if boundary_done:links += [('内外分区 转向再10K','v43_opacity_boundary_turn_left15')]
nav=''.join('<a class="button" href="#workspace?group='+gid+'&amp;raw=113">'+title+'</a> ' for title,gid in links)
panel='<!-- gpu-review-start --><section class="panel" id="gpuReview"><h2>第一场景GPU诊断：'+('三组10K已完成' if done else '原生剥离已交付；opacity对照运行中')+'</h2><p>'+nav+'</p><p>剥离后仍有暗色残留，并损伤背景；不标为成功修复。固定实验规则与右侧几何同步。<a href="../../V43_Ownership_20261027/gpu_v43/FIRST_GPU_REVIEW.md">实验合同、结果与组件图 →</a> · <a href="../../V43_Ownership_20261027/gpu_v43/ablation_raw113.png">raw113原生剥离排查图 →</a></p><p>跳转后若当前页未更新，请刷新；也可直接从“V43身份与结构实验”选择对应组。</p></section><!-- gpu-review-end -->'
if boundary_done:panel=panel.replace('三组10K已完成','四组10K已完成').replace('跳转后若当前页未更新，请刷新；也可直接从“V43身份与结构实验”选择对应组。','当前结果优先看内外分区控制，并与固定opacity增强对照；共享MLP未显示额外收益。点击按钮即可切组，保留原默认无编辑通路。')
page=(S/'index.html').read_text(encoding='utf8');page=re.sub('<!-- gpu-review-start -->.*?<!-- gpu-review-end -->','',page,flags=re.S);page=page.replace('<section class="panel" id="workspace">',panel+'<section class="panel" id="workspace">',1) if '<section class="panel" id="workspace">' in page else page.replace('<section class="panel" id="training">',panel+'<section class="panel" id="training">',1)
if 'href="#gpuReview"' not in page:page=page.replace('<a href="#badcaseReview">','<a href="#gpuReview">GPU诊断</a><a href="#badcaseReview">',1)
(S/'index.html').write_text(page,encoding='utf8')
hub=B/'outputs/GS_Scene_Diagnostics_20261027/index.html';s=hub.read_text(encoding='utf8');marker='<!-- first-gpu-review-link -->'
if marker not in s:s=re.sub(r'(<main\b[^>]*>)',r'\1'+marker+'<div class="notice"><strong>第一场景GPU诊断已更新：</strong><a href="14705303724557273004_3105_000_3125_000/index.html#gpuReview">原生剥离、四组opacity控制与同帧几何 →</a></div>',s,count=1)
hub.write_text(s,encoding='utf8')
print(json.dumps(dict(opacity_complete=done,stats=stats,first_scene_only=True)))
