"""汇总实际诊断结果与用户纠正后的协议；不将接线完成视为资产合格。"""
from pathlib import Path
import json,math
R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    bench=read(R/'student/delivery/fixed_detail_metrics.json')
    qa=read(R/'student/independent_qa.json')
    rows=bench['rows'];historical=bench['historical_reference_rows']
    def row(stage):return next(x for x in rows+historical if x['stage']==stage and x['raw']==110 and x['role']=='vehicle')
    old,init,final=[row(s) for s in ['handoff_fixed_guarded','canonical_prior_init','joint_fit']]
    a10=next(x for x in qa['raw086_a10_native11_rows'] if x['stage']=='joint_fit')
    crops={}
    for raw in [86,110,134,179,198]:
        folder=R/'conditioning'/('train' if raw in [86,110] else 'diagnostic')/f'{raw:03d}'
        m=read(folder/'mapping.json');x0,y0,x1,y1=m['crop_native_xyxy_half_open']
        x0,y0,x1,y1=math.floor(x0*.6),math.floor(y0*.6),math.ceil(x1*.6),math.ceil(y1*.6)
        crops[f'{raw:03d}']=[x0,y0,x1-x0,y1-y0]
    metrics=[
        ['资产总判定','灰模未通过；不可晋升','可见车身被整车先验替换，结构与材质仍有明显问题'],
        ['raw110 可见 RGB MAE',f'上一轮 {old["RGB_MAE"]:.5f} → 本轮 {final["RGB_MAE"]:.5f}','固定旧评分域；本轮较差'],
        ['raw110 局部 SSIM',f'上一轮 {old["masked_SSIM"]:.4f} → 本轮 {final["masked_SSIM"]:.4f}','本轮较差'],
        ['raw110 高频对应度 σ1',f'上一轮 {old["HF_cosine_1"]:.4f} → 本轮 {final["HF_cosine_1"]:.4f}','纹理对应仍较差'],
        ['raw110 高频能量比 σ1',f'上一轮 {old["HF_energy_ratio_1"]:.4f} → 本轮 {final["HF_energy_ratio_1"]:.4f}','相对真实可见教师；不是越大越好'],
        ['raw110 本轮初值 → 拟合',f'MAE {init["RGB_MAE"]:.5f} → {final["RGB_MAE"]:.5f}','能拟合颜色，不说明车体和先验合格'],
        ['raw86 其他车辆 a10',f'{a10["pixels_gt_1_255"]}/{a10["pixels"]} 像素变化 >1/255','目标 RGB MAE 0.10102 → 0.14937；保护失败'],
        ['最终方向灰模测距','精确均值 8.83 cm，p90 29.03 cm','独立核查；未进行测距驱动几何优化'],
        ['实际学生更新','200 车辆＋200 联合；额外 smoke 9','先验初始化＋可见 RGB 拟合，非完整教师蒸馏'],
        ['前置灰模硬门','当前候选被拒绝；材质／学生入口均阻断','拒绝发生在 Torch／CUDA 初始化前']]
    figures=[
        dict(title='真实训练输入：可见车身与遮挡分开',file='train_inputs.jpg',caption='raw86／110；测距与图像来源保留，未把不可见车身当成真实观测。',wide=True),
        dict(title='纯点条件负例：扫描线栅架',file='point_only.jpg',caption='点面距离较小，仍未形成合格车壳与轮位。',wide=True),
        dict(title='单张完整条件图：生成部分不是观测',file='amodal_condition.png',caption='原可见 RGB 硬拷贝保留，下半部为生成。二维保护没有保证三维可见车身不被替换。',wide=True),
        dict(title='本轮灰模：未通过验收',file='grey_candidate.jpg',caption='含已知尺寸配准及前向坐标修正；浮镜、凸顶、轮形失真均保留。',wide=True),
        dict(title='已执行材质诊断：不能替代灰模验收',file='textured_candidate.jpg',caption='真实 UV／基础色渲染，未重光照；有错位字样、模糊反光与不合理轮胎材质。',wide=True)]
    result=dict(task='SCENE065-VEHICLE-COMPLETION-PRIOR-20261122',date='2026-11-22',status='diagnostic_rejected_geometry',
        quality_pass=False,promotion_allowed=False,human_verdict=None,
        headline='当前 a20 先验未通过灰模验收。本轮400步归档为诊断；已前移验收并阻断后续材质／学生入口。',
        stages=[['真实输入','已准备；BUILD 86／110'],['两种条件','纯点负例＋图像／粗框候选'],['灰模验收','未通过；可见车身未保真'],['已执行诊断','1次材质＋400学生更新'],['新版流程','灰模先过；仅缺失区蒸馏'],['资源状态','不追加实验；GPU保持开机']],
        figures=figures,
        media=[['conditioning/train_contact_sheet.jpg','train_inputs.jpg'],['shape_output/preview_three_views.jpg','point_only.jpg'],['semantic_output/preview_original_input_mask_output.png','amodal_condition.png'],['semantic_output/heading_fixed_preview_three_views.jpg','grey_candidate.jpg'],['semantic_output/texture_preview_three_views.jpg','textured_candidate.jpg']],
        metrics=metrics,findings=[
            '当前生成先验替换了整车几何，真实可见部分也被改变；模型产生的浮块、异常轮形与材质错位尚未解决。完整车壳、齐全贴图及较高 Alpha 只能说明资产可渲染。',
            '固定旧评分域的结果显示，本轮 raw110 的颜色、局部结构和细节对应均弱于上一轮；raw86 另一辆车 a10 也有明显退化。当前结果不晋升，不据此给三维生成模型或3DGS表示能力作全局结论。',
            '后续先建立并保护可信观测表面，只在缺失区使用通过灰模验收的教师补片。学生需增加几何、轮廓和新视角蒸馏，分别保留真实监督与生成监督；当前400步不构成这条完整闭环。',
            '现有道路资产及上一轮七阶段实验保留。新版灰模门已在材质和学生入口前执行，当前候选实际被拒绝，未初始化CUDA。'],
        student_viewer=dict(reference='../../work/vehicle_road_joint_20261121/takeback/vehicle_road_joint_20261121_r1/delivery/references',previous='../../work/vehicle_road_joint_20261121/takeback/vehicle_road_joint_20261121_r1/delivery/handoff_fixed_guarded',current='../../work/vehicle_completion_prior_20261122/student/delivery',crops=crops),
        disclaimer='隐藏下半车身没有完整真实监督；生成结构与材质不代表真实还原。掩码为自动估计；原生场景见过全序列，诊断帧不是独立盲测。新旧系统包含多项变化，比较不能分离单个生成模型的独立收益。')
    (R/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(R/'result.json')
if __name__=='__main__':main()
