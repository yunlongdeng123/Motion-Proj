"""Assemble only completed experiments and verified local artifacts."""
from pathlib import Path
import hashlib, json

R=Path(__file__).resolve().parent
read=lambda p: json.loads((R/p).read_text(encoding='utf-8'))
ref=lambda p: {'path':p,'sha256':hashlib.sha256((R/p).read_bytes()).hexdigest(),'bytes':(R/p).stat().st_size}
g=read('student/geometry_receipt_object_r1.json')
c=read('student_native_support_control_r1/formal80_receipt_r1.json')
reload=read('student_native_support_control_r1/disk_reload_receipt_r1.json')
assert g['completed']==40 and not g['material_fixed_domain_coverage']['passed']
assert c['steps']==80 and c['geometry_bitwise_preserved'] and c['unknown_DC_bitwise_preserved']
assert reload['maximum_float_difference']==0 and reload['maximum_8bit_difference']==0
actor='student_native_support_control_r1/final_core_r1/core/runs/formal80_r1/actor.npz'
assert ref(actor)['sha256']=='0dc65c4aa9fefbf73a1c2d482c0015915818c5bf0ac34d5d318e0784693da53a'

media=[]
def add(src,dst):
    if not (R/src).is_file(): raise FileNotFoundError(src)
    media.append([src,dst]);return dst
def figure(title,src,dst,caption):
    return {'title':title,'file':add(src,dst),'caption':caption}

figures=[
    figure('实际 SD1.5 条件与唯一输出','sd15_seed101_occlusion_r2/preview_original_input_mask_output.png','sd15_input_output.png','原图、受保护输入、实际遮挡许可、输出。后保险杠圆灯、轮胎及底部碎片未通过二维门；未送入Omni。'),
    figure('用于 Omni 历史控制与 TRELLIS 的同一车型参考','../omni_hq_vehicle_teacher_20261122/input/rear_rgb.png','hq_rear_reference.png','模型实际使用对应RGBA。TRELLIS和Omni输入文件SHA相同；官方预处理与采样参数分别记录。'),
]
geometry_figures=[
    figure('Omni：历史高质量参考控制','../omni_hq_vehicle_teacher_20261122/audits/bbox_frame_fixed_registered_audit/grey_six_orthographic.png','omni_grey.png','固定框轴修正后的历史资产；本轮未重采样。'),
    figure('Amodal3R：真实遮挡输入','alternatives/retrieved/amodal_real_raw110_seed101_r2_cpu_audit/registered_audit/grey_six_views.png','amodal_grey.png','输入为真实partial RGB及显式遮挡掩码。轮胎缺失与锥状支撑未通过结构门。'),
    figure('TRELLIS-image-large：同高质量参考','alternatives/retrieved/trellis_hq_rear_seed101_r1_cpu_audit/registered_audit/grey_six_views.png','trellis_grey.png','四轮、双轴、驾驶舱与货厢关系可辨。保留原生成比例；有界配准的统一scale=1.10触上界。'),
]

view_labels={'novel_00':'正前','novel_01':'左前','novel_02':'左后','novel_03':'正后','novel_04':'右后','novel_05':'右前','build_086':'raw86实拍相机','build_110':'raw110实拍相机'}
views={}
methods=[{'label':'TRELLIS 原生教师','images':{}},{'label':'24k重采样学生 · 40步几何','images':{}},{'label':'410k支撑保留 · 80步观测校正','images':{}}]
for camera,label in view_labels.items():
    base='student/geometry_object_r1/eval/'+camera
    final='student_native_support_control_r1/final_core_r1/core/runs/formal80_r1/eval/'+camera
    for channel,name in [('rgb','RGB'),('alpha','单层Alpha')]:
        key=camera+'_'+channel;views[key]=label+' · '+name
        ending='rgb_premult.png' if channel=='rgb' else 'alpha.png'
        for i,src in enumerate([base+'/teacher_'+ending,base+'/student_'+ending,final+'/student_'+ending]):
            dst=f'comparison_{i}_{camera}_{channel}.png';add(src,dst);methods[i]['images'][key]='media/'+dst

scene_figures=[
    figure('raw110：最终对象单层 RGB','audit/T_native_scene_final80_r1/renders/110/isolated_T/rgb.png','scene110_isolated.png','实际原生高斯光栅化，410080行；包含观测校正与生成先验。'),
    figure('raw110：删除 a9 后联合 RGB','audit/T_native_scene_final80_r1/renders/110/T_context_delete/rgb.png','scene110_delete.png','同一对象接回冻结道路与背景。车底红灰平面残留仍可见，未晋升场景。'),
    figure('raw110：联合可见贡献 Alpha','audit/T_native_scene_final80_r1/renders/110/T_context_delete/target_alpha.png','scene110_alpha.png','此图表示联合深度排序后的a20贡献；不能代替RGB材质质量。'),
    figure('raw86：删除 a9 后联合 RGB','audit/T_native_scene_final80_r1/renders/086/T_context_delete/rgb.png','scene086_delete.png','固定保留车辆掩码内仍存在变化；未逐相机裁掉新高斯以掩盖问题。'),
]

def cvg(key,domain):
    return next(x['fraction_student_alpha_at_least_0p95'] for x in g['material_fixed_domain_coverage']['records'] if x['key']==key and x['domain']==domain)
geom_hf=[x['high_frequency'] for x in g['metrics'] if x['key'].startswith('novel')]
correlations=[x['correlation'] for x in geom_hf];energies=[x['energy_ratio'] for x in geom_hf]
metrics=[
    ['二维分支','基础SD1.5 inpainting：1候选／20步；道路LoRA关闭','真实V/R像素最大变化0；二维结构未过，新增Omni采样0'],
    ['独立学生','24,235行；正式40步几何','缺失域覆盖未过；该组80步颜色未执行'],
    ['学生高覆盖 · raw86／110',f'{cvg("build_086","teacher_fixed_high_confidence"):.2%}／{cvg("build_110","teacher_fixed_high_confidence"):.2%}','固定teacher高置信域中，student Alpha≥.95的比例；门为99%'],
    ['学生新视图细节',f'HF相关 {min(correlations):.3f}～{max(correlations):.3f}；能量比 {min(energies):.3f}～{max(energies):.3f}','与同相机512原生教师比较；颜色80尚未执行'],
    ['足迹：正前主轴σ中位数','教师1.084px；学生初始3.709px／40步3.698px','同相机、含相同0.3px²低通；初始化已发生低通与斑点化'],
    ['支撑保留控制','410,080行；真实可见DC 12,865行优化80步','全部几何/协方差/不透明度及397,215行未知DC逐位冻结'],
    ['未知细节监测','HF相关≈1，能量比≈1','原先验保真；不是本轮学出了未知纹理'],
    ['正式预算','40步几何＋80步独立观测校正＝120步','不同实验组；不把失败组的颜色阶段写成已跑'],
    ['短程与恢复','20步＋2重放＋1丢弃探针；控制5步＋2重放','完整Adam／Python／NumPy／Torch CPU与CUDA RNG恢复通过'],
    ['磁盘重载','最终控制8视图float／8bit最大差均0','最终actor与恢复状态均已取回验真'],
    ['场景接入','旧行退休的保留车辆RGB/Alpha差为0；新增对象仍有遮挡','raw86 native11 RGB变化239／245像素；raw110全部K为488像素'],
    ['道路与场景优化','0步','道路、轨迹、相机、环境与其它实例参数冻结；接地和场景未验收'],
]
geometry_metrics=[
    ['模型／输入','全部1030点 mean／P90','结构与数值结论'],
    ['Omni／同一高质量参考（历史）','9.93／38.02 cm','测距门未过；保留粗先验'],
    ['Amodal3R／真实partial＋遮挡掩码','8.85／36.01 cm','轮胎结构与测距未过'],
    ['TRELLIS-image-large／同一高质量参考','4.68／9.08 cm','对象本体门可用于受控试验；场景绑定另审'],
]
appearance_metrics=[['真实V阶段','raw86：RGB L1／HF相关','raw110：RGB L1／HF相关']]
for label,key in [('原生成教师','original_teacher_real_V'),('真实RGB初始化','clean_control_init_real_V'),('80步观测校正','formal80_real_V')]:
    items=reload[key];appearance_metrics.append([label,*[f'{x["RGB_L1"]:.5f}／{x["HF"]["correlation"]:.4f}' for x in items]])
findings=[
    '本轮选择TRELLIS-image-large作为更有用的粗车辆先验。它与Omni历史控制使用同一高质量后3/4参考；Amodal3R使用其真实遮挡接口，输入条件不同。SD1.5唯一结果止于二维门。',
    '24k重采样把位置、协方差、密度与取色共同改变：正前投影主轴σ中位数由1.084增至3.709像素，初始取色还混合约10～11个有效源。颜色训练之前就已有细节损失。清楚的原生高斯输出说明当前油画效果不能作为3DGS表示上限的证据。',
    '支撑保留控制保存了既有未知细节，只优化真实可见外观。真实V的初始化收益与80步收益已分开列出；raw110的HF相关仍只有0.2652，真实可见细节尚未全部通过。',
    '接地检查区分了固定表面与Gaussian覆盖：后右轮处缺少测距支持，前右处存在局部证据冲突。单次俯仰/横滚诊断保留独立版本，未烘入本对象。',
    '真实feature-VJP使旧行退出不再损伤固定保留车辆区域；加入新高斯后仍发生遮挡。后续应针对观测几何、可见性和原生残留处理，不以增加颜色步数代替交接验收。',
]
disclaimer=('参考图及生成隐藏车身属于车型先验，并非a20隐藏真值。1030点为对象框筛选的测距归属代理；部分后窗回波可能位于玻璃后，未认定为外壳。全部数字为BUILD86／110拟合集或合成教师保真；134／179／198未用于本轮模型、姿态或参数选择。不同模型输入协议及默认采样设置不同，单候选不能决定模型上限。Amodal一次兼容性工程重放及其事后重建的执行源码来源已透明记录。候选资产未通过完整车身真实性、场景接地、残影及保留车辆交接验收。')

metadata={
    'schema':'a20.gaussian_actor_candidate.v1','identity':{'alias':'a20','native_id':21,'track':'bzFbWRrSGOgEaXxnuGV0tw'},
    'delete_request':{'alias':'a9','native_id':10},'asset':dict(ref(actor),path='a20_actor_candidate.npz'),'canonical_source_asset':ref(actor),'rows':410080,
    'coordinates':'canonical actor metres; x forward / y left / z up; original bounded registered actor, no physical-pose diagnostic baked in',
    'field_semantics':{'xyz':'metres','scales':'activated standard deviations, not logs','rotation':'activated normalized quaternion wxyz','coefficients':'SH coefficients; DC RGB=C0*DC+0.5; higher SH unchanged','opacity':'already activated; preserve values including 1.0'},
    'source':'same saved TRELLIS-image-large SLAT; official Gaussian decoder; no stochastic resampling',
    'method':'native support initialization and true-visible appearance control, not completed compression distillation',
    'formal_updates':80,'trainable_observed_DC_rows':12865,'unknown_DC_rows_frozen':397215,
    'geometry_and_unknown_DC_bitwise_preserved':True,'scene_qualified':False,'simulation_qualified':False,
    'trajectory':'frozen source a20 track; use task restoration index and native scene adapter',
    'restore':'../../work/a20_sd15_sameview_20261122/audit/T_native_scene_final80_r1/restore_index.json',
    'quality_record':'result.json','disclaimer':disclaimer,
}
(R/'a20_actor_candidate.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
result={
    'schema':'a20.teacher_and_gaussian_asset_comparison.v1','date':'2026-11-22 Asia/Singapore',
    'headline':'TRELLIS提供了更有用的粗车辆先验；保留原生高斯支撑可避免明显细节损失，场景交接仍未验收。',
    'scene_qualified':False,'full_distillation_completed':False,'GPU_keep_on':True,
    'actual_stage_counts':{'object_cache_builds':1,'object_smoke_updates':20,'object_smoke_replay_updates':2,'object_discarded_material_probe_updates':1,'object_geometry_updates':40,'object_material_updates':0,'retained_support_control_smoke_updates':5,'retained_support_control_replay_updates':2,'retained_support_control_observed_color_updates':80,'joint_scene_updates':0,'road_updates':0},
    'metrics':metrics,'geometry_metrics':geometry_metrics,'appearance_metrics':appearance_metrics,
    'figures':figures,'geometry_figures':geometry_figures,'scene_figures':scene_figures,
    'viewer':{'views':views,'methods':methods},'media':media,'findings':findings,'disclaimer':disclaimer,'asset':{'source':actor,'sha256':ref(actor)['sha256']},
    'records':{'SD_2D_gate':ref('2D_gate.json'),'baseline40':ref('student/geometry_receipt_object_r1.json'),'native_support80':ref('student_native_support_control_r1/formal80_receipt_r1.json'),'disk_reload':ref('student_native_support_control_r1/disk_reload_receipt_r1.json'),'detail_footprint':ref('student/detail_footprint_cpu/detail_footprint.json'),'scene':ref('audit/T_native_scene_final80_r1/T_native_scene_receipt.json')},
}
(R/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

md='''# a20：车辆教师与高斯资产对照

2026-11-22。TRELLIS-image-large是本轮更有用的粗车辆先验。保留原生410080高斯支撑可以避免24k重采样造成的明显细节损失；完整a20与场景交接尚未验收。

![Architecture components](architecture_executed.svg)

## 实验组与输入

| 组 | 实际执行 | 结果 |
| --- | --- | --- |
| SD1.5同视角补全 | 基础9通道inpainting，DDIM20、CFG3、seed101，1候选，0网络训练 | 后保险杠圆灯、轮胎和碎片失败；未送Omni |
| Omni高质量参考控制 | 复用上一轮正确框轴资产，本轮0新采样 | mean/P90 9.93/38.02cm，测距未过 |
| Amodal3R | 真实partial+官方可见/遮挡接口，seed101、12+12步 | 轮胎/锥状支撑失败，mean/P90 8.85/36.01cm |
| TRELLIS-image-large | 与Omni同RGBA，seed101、25+25步；同SLAT官方Gaussian头0.188s解码，无重采样 | 四轮/舱斗可辨；mean/P90 4.68/9.08cm |
| 24k独立重采样学生 | 24000表面点+235不透明测量锚点；20步smoke，正式40几何 | 高覆盖门失败，80颜色未跑 |
| 410k支撑保留控制 | 5步smoke，正式80可见外观校正 | 原生几何/协方差/opacity与未知DC逐位冻结，重载结果一致 |

SD仅生成a9遮挡guard与a20潜在投影交集。V=28384及单列近似可见边界R=3504像素从原图硬拷回，误差0；U=43045生成，非遮挡潜在域N=30058锁灰。原生RGB只裁片/pad到512，没有预先降到480；480类别掩码仅最近邻放大。旧整框U方案只有CPU准备，未推理。二维门拒绝在Torch/CUDA与Omni准备前真实阻断。

Omni/TRELLIS共用HQ后3/4 RGBA，SHA ac5f39fbb07029ec4018c56269f94949a4d9471a4c31d539d2a2e0e49b951399。Amodal输入不同，不能把三者解释为严格同输入排名。官方源码、模型与DINO权重、实际tokens及哈希均有记录。Amodal首次SLAT遇到同class导出位置兼容问题；原seed重放的SS coords逐元素相同，41.14秒及额外12步SS单列。执行源码启动hash未捕获，归档明确标为事后精确逆补丁重建；不把后增hook伪称已执行。

## 坐标与对象灰模门

独立看灰模确定语义轴：Amodal前方+Y、TRELLIS前方−Y，两者+Z向上。分别用proper R=[[0,1,0],[-1,0,0],[0,0,1]]和[[0,-1,0],[1,0,0],[0,0,1]]，det=+1。统一以5m长初始化，保留宽高比，不分轴拉伸。各做一次原边界/原12迭代有界yaw、统一scale和平移配准。TRELLIS scale=1.10触上界，最终LWH=5.49930×2.42789×2.03343m；原始/正反矩阵全保存。全部1030代理测距点与原阈值保留。

TRELLIS本体准入仅用于受控独立对象试验。四轮/两轴、驾驶舱/货厢关系成立，轮底高差约1.13cm；格栅与尾管仍粗。真实道路接地与场景遮挡单独设门。

## 独立学生与保真控制

独立学生：6固定512²合成视图+BUILD86/110的1152×768原生相机；几何/覆盖接受实际高斯光栅化梯度、几何正则与alpha/depth监督。测量不透明锚点固定，玻璃回波不做外壳。正式40从clean init起跑，未将smoke20接成正式训练。原生xyz/scale梯度非零，teacher到受保护分支梯度为0。恢复覆盖模型、Adam、Python/NumPy/Torch CPU和CUDA RNG、下一步重放、下一阶段丢弃探针。

正式40后，在固定teacher高置信域内，BUILD86/110的Alpha≥.95覆盖为95.543%/95.219%，低于99%入口；合成视图HF相关仅.028–.206，能量比.254–.321。该组80颜色保持关闭，DC从初始至40逐元素未变。

CPU同相机足迹诊断：正前主轴σ中位数教师1.084px、初始学生3.709px、40后3.698px；后视为1.060/3.349/3.348。共同包含0.3px²低通。初始取色混合约10–11有效源，含教师Σ与15mm采样足迹的投影二阶矩主轴约2.59–2.83px。初始0次optimizer时已见宽斑点。位置、密度、协方差和取色同时改变，不能把原因只归节点数；未执行80也不能据此证明颜色优化绝对上限。

控制组保留全部410080行xyz/scales/wxyz/opacity和高阶SH。12865行通过实际深度/alpha与真实V资格检查的DC从原RGB初始化并优化，397215未知DC永远不进入优化器。80步只用真实V RGB+0.1edge+0.1HF，未知新视图仅保真监测。它是教师初始化与观测校正，不能冒充完整压缩蒸馏。两组正式总预算为40+80=120；短程、重放和丢弃探针单列。

| 真实V阶段 | raw86 RGB L1/HF相关 | raw110 RGB L1/HF相关 |
| --- | --- | --- |
| 原生成教师 | .09180/.0915 | .08855/.00349 |
| 真实RGB初始化 | .03129/.3987 | .04033/.1047 |
| 正式80后 | .01880/.5053 | .03632/.2652 |

初始化收益与优化收益分开；未知HF≈1是原先验逐位保留，不是学出隐藏真值。raw110真实细节仍有差距。RGB/edge/HF训练使用固定中性背景的premultiplied合成色，不在低覆盖处除alpha；固定teacher95%域和真实V分别要求99%像素student alpha≥.95，不能用student自选mask缩小域。

## 场景接地与归属交接

同一固定道路不仅有Gaussian：base_atlas固定网格及测量平面支持均检查。原T前轮相对局部路面偏低约14–19cm。单次固定边界pitch/roll/Z诊断得到−2.21819°/−2.09377°/+0.08809m，保持原yaw/scale/XY/轮形与道路；全1030门仍过。真实固定surface查询仍显示后右缺可信支持、前右局部证据冲突，不能宣称四轮贴地。物理变体未烘入独立对象。

原native21不是纯净车辆包。4 baseline、feature-VJP将V、a9与全部保留K贡献分开；K_other有贡献的1922旧行全部保护，KEEP/DELETE分别退出14994/24449行。纯退出旧行的4次因果对照在K内RGB/alpha float逐像素完全相同。加入新对象后仍会遮挡K，不能用旧行归属错误解释全部差异，也未逐相机裁新GS掩盖问题。

冻结最终T控制只做6次原生场景raster、0更新。raw86右侧保留皮卡native11 RGB变化239/245像素（KEEP/DELETE）；raw110该车0，但全部K仍488。DELETE的原生a9 alpha为0，车底仍有红灰平面残留；原a9区域legacy覆盖不能单凭区域重叠判语义。道路/其它车/环境/轨迹/相机参数全冻结，无联合优化或场景晋升。

## 持久化与资源

最终标准actor SHA0dc65c4aa9fefbf73a1c2d482c0015915818c5bf0ac34d5d318e0784693da53a，410080行。完整mutable checkpoint绑定冻结完整base；独立重载8视图float及PNG最大差0。baseline40重载只出现3个8bit通道值差1LSB，未写逐位相同。

正式几何40训练段6.238s、正式控制80段4.538s，均不含Runtime加载；baseline20smoke4.783s，控制5smoke2.908s。正式运行远少于一小时，短程真实覆盖全部状态与阶段入口。生成模型无网络微调；scene/road更新均0。各运行完整进程与依赖开销按原receipt记录，不将runner段当完整SSH耗时。GPU按用户要求保持开机。

core只取必要source、最终状态/actor、关键图与证据并逐SHA验真；581MB完整float/失败现场保留远端immutable恢复指针，未为其阻塞交付。HTML保留历史报告导航，原实验组未删除。

## 结果入口

- result.json：主数值、实际阶段计数与媒体索引。
- a20_actor_candidate.json：单位、激活、来源、身份与未验收范围。
- alternatives/availability.md：两替换模型、权重、接口及工程重放。
- student/detail_footprint_cpu/detail_footprint.md：双低通/密度与初始图诊断。
- student/SMOKE_GEOMETRY_RESULT.md：失败压缩学生与恢复证明。
- student_native_support_control_r1/RESULT.md：独立控制与状态恢复。
- audit/T_native_scene_final80_r1/report.md：场景RGB/alpha、K和残留。
- audit/restore_index.json：完整恢复根；最终场景restore_index与control core另有精确SHA。

免责声明：'''+disclaimer+'\n'
(R/'report.md').write_text(md,encoding='utf-8')
print('Assembled completed results, candidate metadata and report.')
