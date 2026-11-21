"""Freeze the bounded HQ-reference experiment result; never runs a model."""
from pathlib import Path
import json

R = Path(__file__).resolve().parent
read = lambda p: json.loads((R / p).read_text(encoding='utf-8'))
fixed = read('audits/bbox_frame_fixed_registered_audit/audit.json')
uniform = read('audits/bbox_frame_fixed_uniform_audit/audit.json')
execution = read('execution_summary.json')
execution['numeric_and_visual_verdict'] = 'GREY_GATE_FAILED_NO_MATERIAL_NO_STUDENT'
execution['final_verdict_source'] = 'audits/grey_gate_result.json'
metrics = [
    ['当前结论', '粗车型几何候选；未通过a20教师准入', '测距失败，部件细节仍需修复'],
    ['测距均值／P90 · 全部1030点', '9.93 cm／38.02 cm', '预登记门槛 ≤7.5 cm／≤15 cm：未通过'],
    ['raw86 可见核心召回／未知区排除IoU', '98.46%／98.46%', '≥95%／≥90%：通过'],
    ['raw110 可见核心召回／未知区排除IoU', '99.06%／99.06%', '≥95%／≥90%：通过'],
    ['可靠背景溢出 · raw86／110', '0%／0%', '≤0.5%：通过'],
    ['配准后长×宽×高', '4.861×2.550×1.981 m', '各轴相对已知框0.8～1.25：通过'],
    ['结构检查', '四轮、两轴、驾驶舱／货厢关系可辨', '轮胎、轮拱、底盘仍有熔连；未认定高质量'],
    ['实际模型输入', '后3/4单车RGBA；前3/4仅作结构参考', '官方图像列表为独立batch，未使用多图联合条件'],
    ['本轮计算', '两次形状推理；训练0步', '初次框轴诊断＋一次坐标修正重放；GPU保持开机'],
]
media = [
    ['input/reference_original.png', 'reference.png'],
    ['input/rear_rgb.png', 'rear_rgb.png'],
    ['input/rear_alpha.png', 'rear_alpha.png'],
    ['proposal_bbox_frame_fixed/raw_grey_six_views.png', 'corrected_raw_grey.png'],
    ['audits/bbox_frame_fixed_registered_audit/grey_six_orthographic.png', 'corrected_orthographic.png'],
    ['audits/bbox_frame_fixed_registered_audit/measured_points_six_orthographic.png', 'measurement_overlay.png'],
    ['audits/bbox_frame_fixed_registered_audit/projection_086.png', 'projection_086.png'],
    ['audits/bbox_frame_fixed_registered_audit/projection_110.png', 'projection_110.png'],
    ['proposal/raw_grey_six_views.png', 'initial_axis_diagnostic.png'],
]
figures = [
    {'title': '用户提供的高质量车型参考', 'file': 'reference.png', 'caption': '左侧后3/4视图作为Omni输入；右侧前3/4视图只作结构核对。'},
    {'title': '实际输入：后3/4原始RGB裁片', 'file': 'rear_rgb.png', 'caption': '模型使用此裁片的前景RGBA；原始RGB未重绘、未缩放。'},
    {'title': '实际输入的前景Alpha', 'file': 'rear_alpha.png', 'caption': '推理前修正近侧后轮被抠图截断的问题；保留轮胎和后视镜。'},
    {'title': '框轴修正后：原始Omni灰模', 'file': 'corrected_raw_grey.png', 'caption': '未用测距变形；车顶与后视镜较干净，四轮和货厢可辨。'},
    {'title': '固定候选：正交六视灰模', 'file': 'corrected_orthographic.png', 'caption': '前、后、左、右、顶、底；只作有界刚体及统一尺度配准，未分轴拉伸。轮拱、后轮和底盘仍偏粗糙。'},
    {'title': '实测点与表面残差', 'file': 'measurement_overlay.png', 'caption': '全部1030个实测代理点保留；较大残差集中驾驶舱后部，尾门区域贴合较好。'},
    {'title': 'raw86：可见轮廓投影', 'file': 'projection_086.png', 'caption': '固定可见核心和可靠背景分别检查；未知遮挡区不作为完整轮廓真值。'},
    {'title': 'raw110：可见轮廓投影', 'file': 'projection_110.png', 'caption': '轮廓指标通过；不能替代完整形状及实测点检查。'},
    {'title': '保留的初次结果：框轴接线错误诊断', 'file': 'initial_axis_diagnostic.png', 'caption': '初次把车辆LWH直接传为模型XYZ，使车体过宽。该结果单独归档，不作为有效Omni能力对照。'},
]
findings = [
    '这张参考能够驱动Omni生成较完整、部件关系可辨的皮卡灰模。但它目前适合作为车型级粗几何候选，尚不能作为已验收的a20教师。',
    '本轮发现并修正了BBox轴序接线错误：Omni接口直接使用原始XYZ，模型前方为−Z、上方为+Y。将车辆LWH改为模型的WHL后，统一按5米车长初始化，车宽从3.72米降至2.62米。固定同一图像、seed、步数和门槛，只重放一次。',
    '测距失败具有局部性：尾部649点均值3.64厘米、P90为6.72厘米；驾驶舱后部314点均值18.93厘米、P90为41.37厘米。raw86与raw110分别统计仍失败，不能通过继续整体下移解决。',
    '四轮／两轴、驾驶舱／货厢关系比早期占位先验清楚，车顶和后视镜没有明显漂块；轮胎轮拱和底盘仍有低频熔连，前格栅和细部结构较粗。网格完整与轮廓通过不构成教师准入。',
    '本轮按前置灰模协议停止：未生成材质、未训练学生，也未替换真实可见车身。下一步若继续，应保留可见a20表面，针对缺失部件修复并重新通过灰模门，再进入教师蒸馏。',
]
disclaimer = ('参考图是车型先验，并非已证实的a20隐藏部分真值；1030个点按对象框和遮挡框保护筛选，属于实测几何归属代理，尚非语义核验的a20专属真值。轮廓来自保守可见核心掩码，不能验证遮挡部分。86／110用于有界配准，指标为拟合集指标；134／179／198未参与生成、配准或选择。门槛为本轮生成前固定的工程标准，人工质量分数留空。本次单候选试验不能据此判断Omni的能力上限。')
result = {
    'schema': 'a20.omni_hq.experiment_result.v1',
    'date': '2026-11-22 Asia/Singapore',
    'headline': '高质量参考改善了粗车型结构；测距与局部部件关系仍未通过，暂不作为合格a20教师。',
    'verdict': 'GREY_GATE_FAILED_NO_MATERIAL_NO_STUDENT',
    'human_quality_verdict': None,
    'approved_as_a20_teacher': False,
    'whole_visible_actor_replacement_allowed': False,
    'automatic_teacher_promotion': False,
    'geometry_only': True,
    'teacher_student_training_steps': 0,
    'candidate': 'proposal_bbox_frame_fixed',
    'measurement_point_count': fixed['source_point_count'],
    'point_to_surface_m': fixed['point_to_surface_m'],
    'uniform_point_to_surface_m': uniform['point_to_surface_m'],
    'registered_extent_lwh_m': fixed['mesh_extent_xyz_m'],
    'registration_and_gate_source': 'audits/bbox_frame_fixed_registered_audit/audit.json',
    'spatial_residuals_source': 'audits/bbox_frame_fixed_registered_audit/spatial_residual_breakdown.json',
    'metrics': metrics, 'media': media, 'figures': figures, 'findings': findings,
    'disclaimer': disclaimer, 'execution': execution,
}
(R/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
report = f'''# Omni＋高质量参考：a20车辆灰模教师验收

2026-11-22。当前结论：**粗车型几何候选有改善，未通过a20教师准入。** 本轮止于灰模；材质生成0次，学生训练0步，GPU保持开机。

![Architecture components](../../outputs/omni_hq_vehicle_teacher_20261122/architecture.svg)

## 实际执行

输入为用户提供的1536×1024双视图参考。左侧后3/4原始RGB裁片754×516经CPU前景抠图后作为单张RGBA条件；模型推理前修正近侧后轮Alpha截断。原RGB不重绘、不缩放。右侧前3/4只作结构核对。官方图像列表接口展开为batch，未提供联合多视图条件。

冻结Hunyuan3D-Omni，seed101，50步，CFG4.5，形状解码256。初次错误传入raw XYZ框[1,.416,.5]；独立原始灰模检查确认前方−Z／上方+Y后，按车辆尺寸[5,2.5,2.08]的WHL传入[.5,.416,1]，仅作一次坐标修正重放。两次形状推理共100请求步、进程墙钟154.95秒、峰值分配显存7.68GiB。原始候选、修正候选及latent均保存。

模型原始XYZ跨度为[1.044659,.811535,1.991053]。采用det=+1的旋转矩阵[[0,0,-1],[-1,0,0],[0,1,0]]进入对象坐标，再仅按5米车长作统一尺度初始化，LWH=[5,2.623384,2.037955]米。没有分轴缩放。原始比例与全部变换可恢复。

## 灰模验收

一次固定12迭代有界配准：yaw±10°，统一尺度[.9,1.1]，平移边界±[.35,.35,.20]米。最终scale=.972044，yaw=−.201608°，平移[.046204,−.000947,−.136381]米。未作候选、角度或阈值扫描。

| 检查 | 结果 | 生成前固定门槛 | 状态 |
| --- | --- | --- | --- |
| 全部1030点至真实三角面的均值 | 9.93 cm | ≤7.5 cm | 未过 |
| P90 | 38.02 cm | ≤15 cm | 未过 |
| raw86可见核心召回／未知区排除IoU | 98.46%／98.46% | ≥95%／≥90% | 通过 |
| raw110可见核心召回／未知区排除IoU | 99.06%／99.06% | ≥95%／≥90% | 通过 |
| 两帧可靠背景溢出 | 0%／0% | ≤0.5% | 通过 |
| 配准后LWH | 4.861×2.550×1.981 m | 各轴比值0.8～1.25 | 通过 |

四轮和两轴存在，驾驶舱、货厢、保险杠可辨，车顶和双镜无早期明显漂块。后轮、轮拱与底盘局部熔连；细部部件关系仍未建立。额外约0.143平方米薄片位于前挡风下沿／引擎盖后缘，可能为雨刷／饰条，未仅按连通块数量判失败。人工质量评分保持null。

## 测距卡点

raw86为428点，均值9.78厘米、P90为37.55厘米；raw110为602点，均值10.03厘米、P90为39.67厘米。全部点均保留。尾部x∈[−2.5,−1.8]米的649点贴合较好，均值3.64厘米、P90为6.72厘米；驾驶舱后部x∈[−1,−.2]米的314点均值18.93厘米、P90为41.37厘米。后者147点残差超过15厘米，相对最近表面平均位移[+.0805,+.0051,−.2188]米。高度z∈[.26,.56]米的363点均值18.04厘米。不能把全部失败归为地面点，也不能靠全车整体下移修复。

保留未配准统一初始化、配准结果、精确正交六视、点叠加、逐点残差和最大100残差CSV。raw134／179／198未被用于本轮候选选择或配准。

## 教师准入结论

新候选能够提供车型级粗形状，但尚不具备整车替换或监督缺失a20表面的准入资格。下一步应保留真实可见车身，针对缺失部件建立受观测约束的修复，再重复灰模验收；通过后才进入材质和学生蒸馏。本轮没有把先验初始化或完整网格命名为完成蒸馏。

## 归档入口

- 输入与轴序：input_contract.json、input_contract_bbox_frame_fixed.json。
- 执行与取回：execution_summary.json、execution_takeback_receipt.json。
- 生成前门槛：audits/preregistered_gate.json。
- 主审计：audits/bbox_frame_fixed_registered_audit/audit.json。
- 完整变换与原始形状：proposal_bbox_frame_fixed/、audits/bbox_frame_fixed_uniform_initialization/、audits/bbox_frame_fixed_bounded_registration/。
- 代码与模型接口快照：code/、source/、evidence/。
- 远端现场：/root/autodl-tmp/drivingeditor_repro/campaign/omni_hq_vehicle_teacher_20261122_r1。

免责声明：{disclaimer}
'''
(R/'report.md').write_text(report, encoding='utf-8')
print('Frozen result and report written.')
