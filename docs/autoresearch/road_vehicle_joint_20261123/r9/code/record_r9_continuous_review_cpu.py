"""Record scoped independent v2 acceptance; root scheduling remains separate."""
from pathlib import Path
import numpy as np
from audit_r9_shared_field_cpu import load,read,write,pin

WORK=Path(__file__).resolve().parents[2];R9=WORK/'old_raw_teacher_joint_20261123_r9';OUT=R9/'cpu_review/joint_shared_continuous_v2'
fp=R9/'joint_shared_continuous_v2/final_atlas.npz';m=read(OUT/'independent_metrics.json');f=load(fp);s=load(R9/'joint_shared_continuous_v2/solve_state.npz')
assert m['decision']=='LOCAL_NUMERIC_CHECKS_PASS' and all(m['checks'].values())
assert pin(fp)['sha256']==m['final_atlas']['sha256']=='2eaf1569a97fb8ee6907696dfa28db91bb5f3072f74d37491007c87012e2c647'
raw_weights=np.stack([load(R9/f'world_observations_v2/{int(raw):03d}/world_observation.npz')['weight'] for raw in s['frames']]).astype(np.float64)
scale=np.median(raw_weights[raw_weights>0]);raw_weights/=scale;saved=s['data_weights'].astype(np.float64);valid=s['valid']
weight_checks={'all_original_valid_samples_have_positive_weight':bool((saved[valid]>0).all()),
  'outside_valid_weights_zero':not bool(saved[~valid].any()),'weights_finite':bool(np.isfinite(saved).all()),
  'robust_weights_never_exceed_original_normalized_weights':bool((saved[valid]<=raw_weights[valid]*(1+1e-6)).all())}
assert all(weight_checks.values())
write(OUT/'continuous_weight_support_audit.json',{'schema':'R9.continuous_weight_support_audit.v1','complete':True,'decision':'PASS',
  'checks':weight_checks,'valid_samples_per_view':valid.sum((1,2)).tolist(),'positive_weight_samples_per_view':(saved>0).sum((1,2)).tolist(),
  'minimum_robust_retention_factor':float((saved[valid]/raw_weights[valid]).min()),'all_original_valid_RGB_channel_samples':int(valid.sum())*3,
  'source_observation_median_weight_normalization':float(scale),'fixed_pair_domain_not_filtered_by_robust_weight':True})
numeric={'schema':'R9.independent_continuous_field_numeric_acceptance.v2','complete':True,'decision':'PASS',
  'final_atlas_sha256':pin(fp)['sha256'],'contract':m['contract'],'independent_metrics':pin(OUT/'independent_metrics.json'),
  'source_sampling_qualification':pin(R9/'cpu_review/joint_shared_v1/source_sampling_qualification_audit.json'),
  'weight_support_audit':pin(OUT/'continuous_weight_support_audit.json'),'checks':m['checks'],
  'fixed_candidate_scores_unchanged_input_only':m['unchanged_input_candidate_scores'],
  'candidate_scores_are_not_current_field_quality':True,'source_grain':m['source_grain'],
  'boundary_LF':m['boundary_LF'],'PDE_channels':m['PDE_channels'],'influence_PDE':m['influence_PDE'],
  'lineage_threshold_tail':m['lineage_threshold_tail'],'fixed1792':m['fixed1792'],
  'scope':'Continuous world RGB residual, exact source/real-lock/permission invariants, solver/export precision, grain and seams; no whole-scene geometry claim.',
  'confidence_is_calibrated_probability':False,'conditional_influence_is_pointwise_RGB_blend_alpha':False,
  'CPU_only':True,'GPU_forward':0,'diffusion_calls':0,'native_F':0,'native_B':0,'Gaussian_updates':0}
assert not (OUT/'numeric_acceptance.json').exists();write(OUT/'numeric_acceptance.json',numeric)
frames=[86,110,134,179,198]
visual={'schema':'R9.independent_continuous_field_visual_review.v2','complete':True,'decision':'PASS',
  'final_atlas_sha256':pin(fp)['sha256'],'frames':frames,
  'reviewer':'Independent assistant inspection of all five original-resolution1152x768 material_only PNGs, seed/v2 and v1/v2 contacts, world atlas and source/influence layers',
  'root_concur':False,'root_review_status':'PENDING separate root review and GPU scheduling',
  'material_internal_decision':'PASS','numeric_decision':'PASS','whole_scene_restoration_accepted':False,
  'contactsheet':pin(OUT/'five_view_seed_final_material_contactsheet.png'),'failed_v1_comparison':pin(OUT/'five_view_v1_v2_material_contactsheet.png'),
  'world_atlas':pin(OUT/'world_atlas_seed_final.png'),'teacher_influence':pin(OUT/'world_teacher_influence.png'),
  'notes':[
    'The structured horizontal/fan-shaped bands and isolated dark stamp visible in v1 are absent at comparable magnitude in v2. V1 remains a failed control.',
    'All five original-resolution surface views show continuous material tone without newly visible hard seams; complete seed grain and existing fine pavement variation remain legible.',
    'The five evaluated material masks are bitexact to v1. No change of display support, old C/S/W masking or original-photo fill explains the improvement.',
    'The1792 fixed rim pixels remain1632 covered by actual new field RGB plus160 unsupported existing geometry;0 covered pixels meet the old dark threshold.',
    'This PASS concerns available shared-surface colour/texture. The160 unsupported pixels and whole-scene/native Gaussian coverage are separate, unchanged questions.',
    'Source kind2 is similar-photo-dominant, not guaranteed pure seed:40850 kind2 texels contain a subthreshold continuous teacher tail, max linear RGB change0.00020972. Source kind3 marks influence>.001, not all nonzero influence.'
  ],
  'per_view_notes':[{'raw':r,'decision':'PASS','notes':['Original-resolution material inspected; no new pronounced v1-like stripes or internal hard colour seams; unsupported surface explicitly separate.'],
                     'material_only':pin(OUT/f'{r:03d}_material_only.png')} for r in frames],
  'full_scene_Gaussian_run_performed_by_this_review':False,'field_RGB_modified_by_review':False,
  'numeric_acceptance':pin(OUT/'numeric_acceptance.json'),'CPU_only':True,'GPU_forward':0,'diffusion_calls':0,'Gaussian_updates':0}
assert not (OUT/'visual_review.json').exists();write(OUT/'visual_review.json',visual)
svg='''<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="275" viewBox="0 0 1080 275"><rect width="1080" height="275" fill="#111827"/><defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#93c5fd"/></marker></defs><g fill="#1f2937" stroke="#64748b"><rect x="20" y="35" width="210" height="102" rx="8"/><rect x="280" y="35" width="250" height="102" rx="8"/><rect x="580" y="35" width="230" height="102" rx="8"/><rect x="860" y="35" width="200" height="102" rx="8"/><rect x="280" y="184" width="530" height="60" rx="8"/></g><g fill="#e5e7eb" font-family="Arial,Microsoft YaHei,sans-serif" text-anchor="middle" font-size="16"><text x="125" y="65">已核验两视角原 raw</text><text x="125" y="91">固定联合校色候选</text><text x="125" y="117">不重新生成</text><text x="405" y="65">连续世界 RGB 残差场</text><text x="405" y="91">PDE + 实拍梯度参考</text><text x="405" y="117">6407 实拍 / 域外硬锁</text><text x="695" y="65">独立 PDE / grain / seam</text><text x="695" y="91">同域五曲面原像素目视</text><text x="695" y="117">numeric / visual PASS</text><text x="960" y="78">Root 目视与调度</text><text x="960" y="107">Gauss 阶段独立门控</text><text x="545" y="208">来源分层：实拍 / 相似实拍 / 教师观测与推断影响</text><text x="545" y="233">1632 新材质 +160 几何不支持 · 原分母保留</text></g><g stroke="#93c5fd" stroke-width="2.5" marker-end="url(#a)" fill="none"><path d="M230 85H276"/><path d="M530 85H576"/><path d="M810 85H856"/><path d="M545 184V141"/></g></svg>'''
(OUT/'architecture_components.svg').write_text(svg,encoding='utf-8')
text='''# R9 连续共享外观场 v2 独立验收

共享曲面材质的数值与独立目视均 PASS，等待 root 独立目视与后续阶段调度。当前 field SHA `2eaf1569a97fb8ee6907696dfa28db91bb5f3072f74d37491007c87012e2c647`；v1 失败对照保持不变。

![Architecture components](architecture_components.svg)

原 raw、合法观测、联合校色候选、非平面几何及6407实拍锁固定。v2求解相对于完整实拍seed的连续RGB残差，以实拍梯度保留颗粒。数据可见性进入变分目标，不再作为逐点RGB覆盖系数。

独立四邻差分模板重算PDE，保存float32状态的相对残差约1.98e-5至2.17e-5，处于严格传播的导出量化界约6.11e-5至6.67e-5内；原double CG阈值1e-6未改。真实锁与权限外逐字节不变，RGB=seed+δ独立重放通过。全部103469合法视角样本仍有正权重，未按robust权重缩减固定分母。

完整实际更新域212193 texels的2cm高频相关性0.99999125、HF变化/seed HF RMS为0.004198；4cm为0.99993436和0.011645。固定v1边界低频跳变p95=.0014863（seed=.0016208）；当前来源边界=.0031048（seed=.0032403），均符合原接缝合同。旧130872通道全RGB/低频改善只表示沿用候选校色质量，不冒称当前field的新指标。

五张1152×768纯材质图已逐一目视，与v1逐位相同显示域。v1明显条带及孤立深色印记已消退，未见同等级新增硬接缝；旧C/S/W与原图RGB没有混入默认材质。1792黑边仍完整拆分1632新材质、160现有几何不支持，已覆盖部分旧暗阈值计数0；几何和全场景修复不由此项PASS表示。

来源kind3阈值为条件teacher influence>.001，观测和推断另有mask。kind2是相似实拍占主导，包含40850个有微弱连续teacher尾部的texel（最大linear RGB变化.00020972），不能写成全部bitexact纯seed。influence和confidence均非校准概率；v2 influence也不是RGB逐点混合alpha。

独立入口numeric_acceptance.json、visual_review.json；CPU验收新增GPU、扩散、Gauss更新均0。
'''
(OUT/'README.md').write_text(text,encoding='utf-8')
print({k:pin(OUT/k) for k in ['numeric_acceptance.json','visual_review.json','continuous_weight_support_audit.json','README.md','architecture_components.svg']})
