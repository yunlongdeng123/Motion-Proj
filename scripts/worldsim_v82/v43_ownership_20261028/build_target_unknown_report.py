"""Evidence, hierarchical component diagram, failure-preserving comparison UI."""
from pathlib import Path
import json,re,html,statistics
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parents[2];O=B/'outputs/V43_Ownership_20261027/gpu_v43';D=O/'target_unknown_delivery_v1';R=O/'target_unknown_v2';R.mkdir(exist_ok=True);S=B/'outputs/GS_Scene_Diagnostics_20261027/14705303724557273004_3105_000_3125_000';W=Path(__file__).parent

def diagram():
    s=['<svg xmlns="http://www.w3.org/2000/svg" width="1360" height="950" viewBox="0 0 1360 950"><defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#637f91"/></marker></defs><rect width="1360" height="950" fill="#f7fafc"/>']
    def text(x,y,lines,size=20,color='#244454'):
        for i,line in enumerate(lines):s.append(f'<text x="{x}" y="{y+i*(size+10)}" font-family="Microsoft YaHei, sans-serif" font-size="{size}" text-anchor="middle" fill="{color}">{html.escape(line)}</text>')
    def box(x,y,w,h,lines,fill='#e7eff6',size=20):
        s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="#abc2cf" stroke-width="1.5"/>');text(x+w/2,y+32,lines,size)
    def arrow(x,y,x2,y2):s.append(f'<path d="M{x} {y}L{x2} {y2}" fill="none" stroke="#637f91" stroke-width="2" marker-end="url(#a)"/>')
    text(680,43,['V43：目标专属 unknown 与背景保护'],31);text(680,81,['共同验收：原位无车辆残影；正确背景与其他车辆保持稳定'],22)
    box(30,110,1300,92,['输入：目标实例 / 原49帧重建与实测掩码 / 完整199帧RGB、LiDAR、相机','保留四种信息：语义类别、实例归属、运动状态、证据状态'],size=21)
    for x,title,fill in [(30,'① 两种归属证据','#fff1d9'),(475,'② 目标专属 unknown','#e1f2e9'),(920,'③ 原生分解与背景修复','#fff1d9')]:
        box(x,244,410,509,[],fill);text(x+205,283,[title],23)
    box(50,312,370,100,['原生 alpha×T 反向归因','删除动态资产后，追踪静态贡献']);arrow(235,413,235,438)
    box(50,439,370,100,['车身与真实背景分别积累','深度、跨帧支持、其他物体排除']);arrow(235,540,235,565)
    box(50,566,370,145,['逐表面重复可见性','≥3次清晰观测，跨度≥0.5秒','缺测与语义冲突保留'],size=19)
    box(495,312,370,100,['可靠车辆 / 可靠背景','区分空间身份与车辆外观污染']);arrow(680,413,680,438)
    box(495,439,370,100,['目标相关不确定集合','混合支撑单列；全局unknown不删除']);arrow(680,540,680,565)
    box(495,566,370,145,['学习保留权重与局部外观','权重不是已校准的身份概率','原语义与动静态分支分别保存'],size=19)
    box(940,312,370,100,['实测背景补层 / 受限表面插值','坐标来自真实世界表面']);arrow(1125,413,1125,438)
    box(940,439,370,100,['联合更新保留率、DC与补层','对照：固定剥离集合后补景']);arrow(1125,540,1125,565)
    box(940,566,370,145,['分区域保护与原生RGB检查','车旁邻域 / 其他车辆 / 远处背景','目前仍有残影，尚未通过验收'],size=19)
    arrow(235,202,235,244);arrow(440,500,475,500);arrow(885,500,920,500);arrow(1125,753,1125,796)
    box(30,797,1300,107,['输出：同一HTML帧轴上的原生RGB、真实高斯参数、背景补层与证据着色','默认无编辑仍走原V1；原模型不覆盖。下一关：混合高斯的空间支撑拆分。'],size=22)
    s.append('</svg>');(R/'architecture.svg').write_text(''.join(s),encoding='utf8')

def main():
    diagram();probe=json.loads((D/'probe_summary.json').read_text());memory=json.loads((D/'memory199_summary.json').read_text());dense=json.loads((D/'dense_memory_summary.json').read_text());stats=[]
    for name in ['split199','joint199','joint_dense199']:
        d=json.loads((D/name/'summary.json').read_text());m=[r for r in d['metrics'] if r['split']=='calibration_holdout'];r=next(r for r in d['metrics'] if r['raw']==113);stats.append(dict(name=name,step=d['step'],holdout_protected_changed_pixels_mean=statistics.mean(x['protected_pixels_changed_gt1byte'] for x in m),raw113_protected_changed_pixels=r['protected_pixels_changed_gt1byte'],raw113_sparse_background_pixels=r['sparse_background_pixels'],measured_background_seeds=d['measured_background_seeds']))
    report=dict(context=probe['context'],status='experiments_complete_not_quality_accepted',source_evidence_frames=199,reconstruction_native_frames=49,mask_counts=probe['mask_counts'],experiments=stats,formal_updates=30000,pilot_updates=2000,extra_smoke_updates=650,source_checkpoint_unchanged=True,unknown_is_target_scoped=True,learned_retention_not_identity_probability=True,RGB_vehicle_residual_fully_removed=False,background_preservation_passed=False,quality_accepted=False,human_verdict=None)
    if (R/'mixed_support_audit/summary.json').exists():report['mixed_support_audit']=json.loads((R/'mixed_support_audit/summary.json').read_text())
    (R/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    md='''# V43：目标专属unknown算法与原生剥离实验

![组件图](architecture.svg)

目标是删车后RGB中无车辆残留，同时保护正确背景。**本轮尚未通过这个联合验收**。第一场景已完成三个新10K对照，第二、第三没有追加实验。默认无编辑V1输出、原模型和旧结果继续保留。新增结果是实际原生高斯渲染，没有视频生成或贴图覆盖。

## 算法改动

1. 目标相关unknown单独建集合。删除目标动态资产后，使用原生alpha×T的精确辅助通道反向归因，找参与目标车身成像的静态行。全局低置信、缺测unknown不再作为删除集合。语义、实例、运动、证据四字段分别保存。
2. 车辆与背景证据分别积累。车身实测深度、车身掩码、尺度用于车辆假设；避开所有预测车辆/行人与标注物体、真实深度一致、跨帧重复的世界表面用于背景假设。二维语义和高斯中心深度都不是物理身份GT。
3. 混合支撑不强行归车。参与目标成像，同时得到清晰背景支持的行单列；保留几何归属与车辆外观污染的区别。原static分支不自动等于正确背景。
4. 联合调整目标相关静态行的保留率和DC，以及实证背景补层的DC/opacity；其余原高斯、所有原几何、原环境冻结。保留率只是优化控制量，还不是已校准的身份概率。对照组先固定整行剥离集合，再拟合同一背景证据。
5. 背景保护按目标邻域、其他车辆、整片外域分别归一化。49帧pilot的整图均值会掩盖局部损伤，正式199帧证据实验使用分区保护。监督参考是原删除动态资产后的RGB；也可能保留其已有错误，因此不能当干净背景GT。

## 输入证据与边界

完整第一片段实际读取199帧，保留原始CRC、相机、时刻和LiDAR关系；并非把原生重建偷偷改成199帧。原重建与查询仍为49帧，33帧优化、16帧校准检查；基础重建见过这些49帧，199帧背景记忆也使用整段，因此没有独立泛化测试。

追加帧的SegFormer-B0输入为480×320，原49帧沿用旧预测不变，其预处理来源1920×1280；这是显式记录的证据预处理差异。语义模型冻结，没有训练分割器。标注框只用于排除可能前景，不能证明剩下的高斯一定是背景。

发现全片段固定分段会随片段长度改变资格，正式背景表面规则改为：至少3个清晰帧、每个世界表面的观测跨度至少0.5秒。旧全局分段结果保留作审计，没有通过放大unknown强求减少残影。

三角形插值仅在重复实测点包围的近水平表面内使用，最长边1米、法线竖直分量大于0.95，不外推。插值色彩和位置不是新增实测；其查询遮挡检查只覆盖这些表面，不能冒称完整场景GT。这组增加几何密度，不代表未见背景已经有观测。

## 原生剥离结果

'''
    counts=probe['mask_counts'];md+=f"目标相关静态候选 {counts['target_candidate']:,} 行，捕获原车身域静态贡献约 {100*probe['target_candidate_total_target_mass_fraction']:.2f}%；背景重复支持 {counts['background_protected']:,} 行，其中与目标混合 {counts['mixed_target_background']:,} 行。严格纯车辆候选 {counts['vehicle_candidate']} 行。零不是‘静态里没有车’，而是当前条件没有证明整行纯车辆。\n\n"
    md+='整颗删除目标相关unknown仍产生明显背景破坏；加背景深度保护也不能解决混合支撑。联合优化减轻部分车身外观，同时比固定整行删除保住更多背景；原位暗色/结构残留仍可见，其他区域仍有变化，没有采纳为最终算法。需要进一步处理混合高斯的空间支撑，不能把同一行的所有像素贡献一起搬走。\n\n'
    md+='|组|新增步数|补层高斯|16帧检查：保护区改变像素数/帧|raw113保护区改变像素|\n|---|---:|---:|---:|---:|\n'
    for s in stats:md+=f"|{s['name']}|{s['step']}|{s['measured_background_seeds']}|{s['holdout_protected_changed_pixels_mean']:.0f}|{s['raw113_protected_changed_pixels']}|\n"
    md+='\n改变像素指三通道平均绝对变化超过1/255；保护域为原车身外扩8px之外。它衡量改动影响，不是背景正确率；这里也不能替代车身残影的逐帧目视检查。\n\n'
    if 'mixed_support_audit' in report:
        audit=report['mixed_support_audit'];er=next(r for r in audit['environment'] if r['raw']==113)
        md+=f"## 额外因果诊断：共享支撑与环境回填\n\n目标相关候选整体可见贡献只有 {100*audit['candidate_all_view_target_mass_fraction']:.2f}% 落在车身域内，其余在域外；这不是‘其余一定属于可靠背景’，但说明整行删除会同时作用于大量其他像素。当前优化只调整整行保留率和DC，尚未拆开混合高斯的空间支撑。\n\n原环境图查询中出现约−12到+11的颜色值。raw113整颗剥离上界图有 {er['saturated_pixels']} 个超亮饱和像素，其中 {er['saturated_pixels_environment_dominant']} 个由环境回填项的绝对贡献主导，约 {100*er['saturated_pixels_environment_dominant']/er['saturated_pixels']:.1f}%。环境项按 (1-alpha)×env 实际分离；去掉或限制环境回填的图保存作因果对照，未采纳为修复。不能把全部白色/彩色破洞直接当成道路几何缺失，也不能直接clamp后宣称背景正确。\n\n"
    md+='## 交付与验收\n\n共享工作台新增6组，原生RGB与真实静态保留率、DC变化和背景补层几何对应。V2证据着色：橙=目标相关未分清，紫=目标/背景混合，蓝=重复背景支持，灰=此目标之外，青=新增背景补层（语义仍unknown）。这些是目标相对的证据状态，不替代19类语义编码；旧语义与旧owner没有被学习权重覆盖。橙框标为原位参考，其他原始车辆蓝框保留。\n\n正式3×10K；49帧pilot为2×1K；各轮save/eval/完整状态恢复与下一阶段入口smoke额外650更新单列，合计32650更新。199帧正式组与插值组实际原生RGB恢复差0；早期pilot检查模型/优化器状态与RNG恢复，未保存RGB恢复对照，不冒称检查项相同。分钟级实验，未启动小时级长跑。并行最多两个已登记子进程持有全局GPU租约。权重和原数据保留远端，取回图像、参数与SHA清单，未下载checkpoint。\n\n未证明联合优化在普遍场景优于强流水线；本轮固定剥离基线较弱，不能代替最强分解方法比较。failure_ledger_refs=[V82-F01]，failure_ledger_delta=none。\n'
    for raw in [98,101,109,113,114]:
        names=['baseline','target_upper_bound','target_preserve_observed_bg','split199','joint199','joint_dense199'];sheet=Image.new('RGB',(1440,700),'#0b1625');draw=ImageDraw.Draw(sheet)
        for i,name in enumerate(names):
            im=Image.open(D/name/'native_rgb'/('%03d.png'%raw));x=(i%3)*480;y=(i//3)*350;sheet.paste(im,(x,y+28));draw.text((x+8,y+7),name+' / raw'+str(raw),fill='white')
        sheet.save(R/('comparison_raw%d.png'%raw))
    (R/'REPORT.md').write_text(md,encoding='utf8');(W/'TARGET_UNKNOWN_V2.md').write_text(md,encoding='utf8')
    links=[('原删除基线','baseline'),('整颗剥离上界','target_upper_bound'),('固定剥离后修复10K','split199'),('联合分解10K','joint199'),('表面插值联合10K','joint_dense199')]
    nav=' '.join('<a class="button" href="#workspace?group=v43_unknown_v2_'+key+'&amp;raw=113">'+title+'</a>' for title,key in links)
    panel='<!-- target-unknown-start --><section class="panel" id="targetUnknownV2"><h2>V43 目标专属unknown：三组10K已交付，尚未通过干净剥离验收</h2><p>'+nav+'</p><p>完整199帧用于背景证据，查询仍为49帧；原位仍有暗残留，背景保护也未完全通过。右侧可选“V2目标专属unknown / 背景证据”，保留率与实际补层几何已对应原生RGB。</p><p><a href="../../V43_Ownership_20261027/gpu_v43/target_unknown_v2/REPORT.md">算法、证据与限制 →</a> · <a href="../../V43_Ownership_20261027/gpu_v43/target_unknown_v2/architecture.svg">组件图 →</a> · <a href="../../V43_Ownership_20261027/gpu_v43/target_unknown_v2/comparison_raw113.png">raw113六组对照 →</a></p></section><!-- target-unknown-end -->'
    page=(S/'index.html').read_text(encoding='utf8');page=re.sub('<!-- target-unknown-start -->.*?<!-- target-unknown-end -->','',page,flags=re.S);page=page.replace('<!-- gpu-review-start -->',panel+'<!-- gpu-review-start -->',1)
    if 'href="#targetUnknownV2"' not in page:page=page.replace('href="#gpuReview"','href="#targetUnknownV2">V2 unknown</a><a href="#gpuReview"',1)
    (S/'index.html').write_text(page,encoding='utf8');hub=B/'outputs/GS_Scene_Diagnostics_20261027/index.html';s=hub.read_text(encoding='utf8');marker='<!-- target-unknown-link -->'
    if marker not in s:s=re.sub(r'(<main\b[^>]*>)',r'\1'+marker+'<div class="notice"><a href="14705303724557273004_3105_000_3125_000/index.html#targetUnknownV2">第一场景：目标专属unknown、199帧背景证据与三组新10K →</a>（仍有残影，未采纳）</div>',s,count=1)
    hub.write_text(s,encoding='utf8')
    p=B/'outputs/V43_Ownership_20261027/index.html';s=p.read_text(encoding='utf8');marker='<!-- target-unknown-overview -->'
    if marker not in s:s=re.sub(r'(<main\b[^>]*>)',r'\1'+marker+'<div class="notice"><a href="gpu_v43/target_unknown_v2/architecture.svg">V43目标专属unknown组件图 →</a> · <a href="gpu_v43/target_unknown_v2/REPORT.md">199帧证据与三组10K结果（尚未验收通过） →</a></div>',s,count=1)
    p.write_text(s,encoding='utf8');print(json.dumps({k:v for k,v in report.items() if k!='mixed_support_audit'}))

if __name__=='__main__':main()
