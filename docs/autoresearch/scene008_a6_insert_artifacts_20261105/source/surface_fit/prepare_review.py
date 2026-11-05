"""Build a compact CPU review and figures from completed, fixed-run evidence."""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parent
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def architecture(path):
    im=Image.new('RGB',(1520,650),'white');d=ImageDraw.Draw(im)
    fontpath='C:/Windows/Fonts/arial.ttf'
    font=ImageFont.truetype(fontpath,22);small=ImageFont.truetype(fontpath,19);title=ImageFont.truetype(fontpath,29)
    d.text((28,22),'Observed-anchored continuous depth + fixed foreground RGBA',font=title,fill='#152c3d')
    def box(x,y,w,h,text,fill='#eaf2f8'):
        d.rounded_rectangle((x,y,x+w,y+h),radius=10,fill=fill,outline='#45617a',width=2)
        lines=text.split('\n');height=28*len(lines);start=y+(h-height)/2
        for i,line in enumerate(lines):
            tw=d.textbbox((0,0),line,font=font)[2];d.text((x+(w-tw)/2,start+i*28),line,font=font,fill='#152c3d')
    def arrow(x,y,xx,yy):
        d.line((x,y,xx,yy),fill='#45617a',width=3)
        if xx!=x:d.polygon([(xx,yy),(xx-10,yy-6),(xx-10,yy+6)],fill='#45617a')
        else:d.polygon([(xx,yy),(xx-6,yy-10),(xx+6,yy-10)],fill='#45617a')
    box(25,105,280,105,'Frozen PVD depth prior\nReference106 rays')
    box(25,235,280,105,'BUILD40 visible LiDAR\nFrontmost association')
    box(355,145,315,170,'Cubic B-spline depth\n5 fixed Huber IRLS solves\nPrior + real z + curvature','#e8f3eb')
    arrow(305,158,355,210);arrow(305,288,355,255)
    box(720,145,315,170,'Posterior XYZ along rays\nC2 continuous depth\n0.6 HRpx fixed kernels','#e8f3eb');arrow(670,230,720,230)
    box(25,400,280,105,'Fixed SAM2 matte + RGB\nReal106 always retained')
    box(355,390,315,130,'Foreground RGBA gate\nGenerated opacity .99 alpha\nNo alpha-zero canvas birth','#fff1d9');arrow(305,453,355,453)
    box(720,390,315,130,'Same BUILD40 RGB domain\nFrozen geometry\n1000 colour-fit steps\nUnknown columns keep matte','#fff1d9');arrow(670,453,720,453);arrow(878,315,878,390)
    box(1090,225,400,165,'One canonical persistent asset\n45,314 rows, no PVD training\nNo camera or free-opacity fit','#ebeffb');arrow(1035,230,1090,270);arrow(1035,453,1090,350)
    d.text((1090,420),'Runtime views of the same asset:',font=small,fill='#152c3d')
    d.text((1090,452),'native480 / analytic area control',font=small,fill='#152c3d')
    d.text((1090,484),'HR1920 black details (area identity)',font=small,fill='#152c3d')
    d.text((27,586),'Fixed49 internal comparisons; HELD9 excluded from colour fitting, previously inspected for development.',font=small,fill='#152c3d')
    im.save(path)

def depth_figure(path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with np.load(ROOT/'results/posterior.npz') as z:
        before=z['prior_depth_map_m'];after=z['posterior_depth_map_m'];valid=np.isfinite(after)
    y,x=np.where(valid);ys=slice(y.min()-4,y.max()+5);xs=slice(x.min()-4,x.max()+5)
    fig,axes=plt.subplots(1,3,figsize=(14,4.4),constrained_layout=True)
    lo,hi=np.nanpercentile(np.r_[before[valid],after[valid]],[1,99])
    for ax,data,title in zip(axes,[before,after,after-before],['Frozen expected-depth prior','C2 posterior (geometry changed)','Posterior minus prior']):
        delta='minus' in title
        image=ax.imshow(data[ys,xs],origin='upper',cmap='coolwarm' if delta else 'viridis',vmin=-2.71 if delta else lo,vmax=2.71 if delta else hi)
        ax.set_title(title);ax.set_xticks([]);ax.set_yticks([]);fig.colorbar(image,ax=ax,shrink=.8,label='camera-z metres')
    fig.suptitle('Same final matte support / reference106 pixels; |depth change| > 0.6m on 5.20% of chart\nDepth colour scale clips shared 1st-99th percentiles; delta scale includes the full tail',fontsize=11)
    fig.savefig(path,dpi=160);plt.close(fig)

def main():
    root=ROOT/'combined_v1/observed_fit';dest=root/'evaluation';summary=load(dest/'summary.json');hr=load(dest/'HR_summary.json')
    receipt=load(dest/'receipt.json');fit=load(root/'formal/receipt.json');tail=load(ROOT/'results/tail_audit.json');depth=load(ROOT/'results/receipt.json')
    review=dict(status='COMPLETED_FIXED_EXPERIMENT_REQUIRES_EXPLICIT_QUALITY_REVIEW',
        asset=fit['asset'],fit_steps=1000,smoke_steps=32,model_training_steps=0,
        fixed_BUILD40_RGB_pixels=394206,fit_RGB_MAE=fit['BUILD_RGB_MAE'],
        geometry_changed_from_old_prior=True,geometry_frozen_during_colour_fit=True,
        source_and_background_parameters_frozen=True,composited_images_not_claimed_unchanged=True,
        analytic_area_is_viewport_control_not_automatic_adoption=True,
        source_receipts=dict(depth_sha256=sha(ROOT/'results/receipt.json'),combined_sha256=sha(ROOT/'combined_v1/receipt.json'),
            fit_sha256=sha(root/'formal/receipt.json'),evaluation_sha256=sha(dest/'receipt.json'),tail_audit_sha256=sha(ROOT/'results/tail_audit.json')),
        fixed49_source_checks=receipt['checks'],
        tail_risk={k:tail[k] for k in ['delta_z_m','tail_abs_gt06','out_of_box','accepted_anchor_residual_m','largest_anchor_residuals']},
        tail_interpretation='Large shifts are mostly high-alpha interior points, not only low-alpha borders; measured depth and occlusion evaluation is needed.',
        neighbour_continuity=depth['neighbors'],metrics={},preservation={},
        HR_initial_preview_only_raws=[105,106,113,130],
        HR_initial_aggregate_must_not_be_read_as_all49=True,HELD_used_only_for_final_evaluation=False,
        HELD_excluded_from_recorded_colour_fit_objective=True,
        HELD_previously_used_for_representation_diagnosis_and_method_development=True,
        independent_final_confirmation=False,untouched_generalization_test=False,
        evaluation_roles=load(ROOT/'combined_v1/evaluation_roles.json'),
        limitations=['Single-reference chart and generated RGB are not independently measured hidden surfaces.',
            'Reduced average depth error does not remove 2.62m accepted-anchor residual or 2.70m forward tail.',
            'Native480 RGB worsens against old projective despite HR improvement; area control improves RGB but lowers alpha.',
            'Outside/road compositing changes include localized large deltas; unchanged scene parameters do not imply unchanged images.'])
    for c,values in summary.items():
        review['metrics'][c]={}
        for split,flat in values.items():
            review['metrics'][c][split]={k:v for k,v in flat.items() if k in (
                'solo.a6.rgb_error','solo.a6.baseline_rgb_error','solo.a6.measured_depth_error','solo.a6.baseline_measured_depth_error',
                'scene.keep.a6.rgb_error','scene.keep.a6.baseline_rgb_error','scene.keep.a6.visible_coverage',
                'scene.keep.occlusion_diagnostics.a37_coverage','scene.delete.preservation.road_reobserved.rgb_error',
                'scene.delete.preservation.road_reobserved.baseline_rgb_error')}
        review['preservation'][c]={k:v for k,v in values['all49'].items() if (
            ('preservation.' in k and ('parent_rgb_change' in k or 'rgb_error' in k)) or
            ('fixed_depth_preservation.' in k and ('parent_change_m' in k or 'depth_error_m' in k)))}
    review['HR_fixed_RGB']={s:{k:v for k,v in flat.items() if not k.startswith('initial.')} for s,flat in hr.items()}
    write(ROOT/'combined_v1/review.json',review)
    architecture(ROOT/'combined_v1/architecture.png');depth_figure(ROOT/'combined_v1/depth_posterior.png')
    text='''**连续深度与前景 RGBA 固定实验审查**

已完成 C2 深度后验、固定 matte RGBA 组合、相同 40BUILD 像素域的 32 步 smoke 与 1000 步颜色拟合，以及 49 帧原生渲染评价。此处的 PASS 表示保存、恢复、导出、重载和评价工程通过；不等于自动采用，也不表示达到了参考图质量。原 PVD 权重未训练，新的 chart 几何确实改变；颜色拟合阶段才冻结该几何。

![architecture](architecture.png)

新 canonical 资产为 `observed_fit/formal/complete_a6.npz`，45,314 行。生成 chart opacity 为 `.99 * sampled_alpha`，真 106 行保留 `.99`；alpha 零的非真实前景点不出生。初始颜色来自 prototype 与固定 matte 采样，没有继承旧粗糙深度的正式拟合色。394,206 个 BUILD RGB 目标与原 HR 拟合逐字一致；新资产所有这些像素 alpha > .9804，没有按新覆盖率缩减目标。25,095 个零贡献颜色列保留初值。

|固定指标|旧 projective|新 C2 + matte|
|---|---:|---:|
|HR all49 straight RGB MAE|0.042121|0.041305|
|HR HELD9 straight RGB MAE|0.040365|0.039359|
|native480 all49 black RGB MAE|0.065308|0.072399|
|native480 HELD9 black RGB MAE|0.065848|0.072624|
|native480 all49 LiDAR z MAE (m)|0.317099|0.201782|
|native480 HELD9 LiDAR z MAE (m)|0.302434|0.189463|
|area480 all49 black RGB MAE, 双方均补偿|0.048629|0.047182|
|area480 all49 LiDAR z MAE (m), 双方均补偿|0.270509|0.168410|

固定面积补偿是同一 canonical 资产的 viewport 对照，没有第二次颜色拟合。它在 HR 逐字恒等，在 480 降低 chart opacity；新资产 visible joint alpha 均值由 native 的 .998828 降至 .976360。原生 negative FoV、投影、轨迹和编译 rasterizer 保持原样。HR 和 area 指标有所改善，native480 RGB 则退化，必须分开报告。

深度连续性在相同 erode1 支持内的相邻边 p99 从 .237219m 降至 .027087m，>10cm 的边从 5684 降至零。但 2210/42536 (5.196%) chart 点移动超过 .6m，最大前移 2.70124m，锚点最大残差仍为 2.62045m。大移动点只有 270 个 prior alpha≤.9，中位 alpha=.99746、中位距 matte 边界40.38HRpx，不能解释成只修低 alpha 边缘。相同点集的 out-of-box 总数1231→1221，chart1075→1065；最大 chart 轴向越界 .15376→.15809m。没有为改善这些数字加硬边界或重调 sigma。

![depth](depth_posterior.png)

背景、a37、道路源参数冻结不等于最终成像不变。native 新旧对照中，a37 keep 的 RGB 变化最大 .015596，536128 固定像素中零个>.02；道路 keep 有6/1075281像素>.02；outside keep 有6043/6312448像素>.02，最大变化 .723011；outside delete 有9004像素>.02，最大 .738736。道路 reobserved delete 的 RGB 误差 .020423→.020120，但相对最早 thinM2 的 .018075 仍较差。所有具体 mean/max/阈值计数、固定分母和深度变化保留于 `review.json` / `observed_fit/evaluation/summary.json`；删除状态 a37 像素没有当成隐藏道路真值。

106/113/130 和105的实际黑底、keep/delete街景、同相机crop与RGB/alpha/depth replay保存在 `observed_fit/evaluation/raw*/`。HELD9 不进入本次颜色拟合目标，但 raw113 等此前已经用于薄片、球核、表面表示缺陷诊断与内部方法开发；这些数值属于内部比较，不能称独立未触碰的泛化测试。原评价 receipt 的历史 final-only 标记与全部原始指标保留，由 `evaluation_roles.json` 明确修正解释。`HR_summary` 中 initial 只覆盖四个预览视角，不能当成 all49 对照。
'''
    (ROOT/'combined_v1/REVIEW.md').write_text(text,encoding='utf-8')

if __name__=='__main__':main()
