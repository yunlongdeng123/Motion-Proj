"""Bind independent numeric PASS and visual FAIL; no producer/result mutation."""
from pathlib import Path
import json
from audit_r9_shared_field_cpu import read,pin,write

WORK=Path(__file__).resolve().parents[2];R9=WORK/'old_raw_teacher_joint_20261123_r9';OUT=R9/'cpu_review/joint_shared_v1'
field=R9/'joint_shared_v1/final_atlas.npz';contract=R9/'acceptance_contract_v1.json'
metrics_path=OUT/'independent_metrics.json';source_path=OUT/'source_sampling_qualification_audit.json'
metrics=read(metrics_path);sources=read(source_path)
assert metrics['decision']=='LOCAL_NUMERIC_CHECKS_PASS' and all(metrics['checks'].values()) and sources['decision']=='PASS'
assert pin(field)['sha256']==metrics['final_atlas']['sha256']=='8ad4b9594ef5e14f9f77ebf57a395ebf3da410bb435eb2170f6b2558de9886dd'
numeric={'schema':'R9.independent_shared_field_numeric_acceptance.v1','complete':True,'decision':'PASS',
  'final_atlas_sha256':pin(field)['sha256'],'contract':pin(contract),'independent_metrics':pin(metrics_path),
  'source_sampling_qualification':pin(source_path),'fixed_pairs':metrics['fixed_pairs'],
  'boundary_LF_jump':metrics['boundary_LF_jump'],'seed_boundary_LF_jump':metrics['seed_boundary_LF_jump'],
  'checks':metrics['checks'],'fixed1792':metrics['fixed1792'],
  'scope':'Original fixed overlap, exact source/camera/permission/real-lock checks, LF boundary jumps and legal unknown classification; actual-surface material only.',
  'confidence_metadata_is_uncalibrated_and_unused':True,'visual_gate_is_separate':True,
  'full_scene_restoration_accepted_by_this_numeric_review':False,
  'new_Gaussian_fit_authorized_by_numeric_alone':False,'CPU_only':True,'GPU_forward':0,'native_F':0,'native_B':0,'Gaussian_updates':0,'diffusion_calls':0}
assert not (OUT/'numeric_acceptance.json').exists();write(OUT/'numeric_acceptance.json',numeric)
frames=[86,110,134,179,198]
visual={'schema':'R9.independent_shared_field_visual_review.v1','complete':True,'decision':'FAIL',
  'final_atlas_sha256':pin(field)['sha256'],'frames':frames,
  'reviewer':'independent assistant inspection of seed/final contact, full-resolution actual-surface 110/134, world atlas, alpha and source-kind layers',
  'root_concur':False,'root_review_status':'PENDING separate root review',
  'material_internal_decision':'FAIL','numeric_decision':'PASS',
  'full_restoration_decision':'NOT_ACCEPTED','gaussian_fit_decision':'NOT_ACCEPTED',
  'contactsheet':pin(OUT/'five_view_seed_final_material_contactsheet.png'),
  'world_atlas':pin(OUT/'world_atlas_seed_final.png'),'teacher_alpha':pin(OUT/'world_teacher_alpha.png'),
  'source_kind_110':pin(OUT/'110_source_kind.png'),
  'notes':[
    'Compared with the same-domain photo seed, the shared final material shows new horizontal or fan-shaped tone bands and patch boundaries, repeated consistently in all five surface projections.',
    'The structures remain in material_only PNGs, so they are not the display hatching for missing geometry.',
    'The fragmented teacher-alpha/source3 support has spatially corresponding stripes. This is an observed association, not a uniquely established causal attribution to the solver.',
    'A localized dark feature appears in the right-central material region. Its semantic origin is not established by road-class proxy alone.',
    'All1792 historical rim pixels remain accounted for:1632 now show actual new field RGB and0 meet the old dark threshold;160 have no existing mesh support and remain hatched. The160 are not teacher-black evidence.',
    'No original-photo pixels, old C/S/W masks, hard-final composites or parent Gaussian RGB fill the default material view.'
  ],
  'per_view_notes':[
    {'raw':86,'decision':'FAIL','notes':['New structured bands are visible across the lower-left and central material relative to seed; source/geometry holes are separately hatched.']},
    {'raw':110,'decision':'FAIL','notes':['Bands are particularly visible around x200..550,y620..767; a right-central dark feature is also present. Covered historical rim pixels use actual field RGB.']},
    {'raw':134,'decision':'FAIL','notes':['The same world bands and transitions remain visible in this unfitted diagnostic camera, including the lower central material.']},
    {'raw':179,'decision':'FAIL','notes':['The same structured material variation persists at another viewpoint; no per-view image copying was used.']},
    {'raw':198,'decision':'FAIL','notes':['The shared bands persist in the fifth camera, so visual color/seam acceptance is not met.']}
  ],
  'material_view_scope':'All existing projected surface with all-four legal atlas material support, without old per-view C/S/W restriction. Optional photo context is a separate diagnostic.',
  'numeric_acceptance':pin(OUT/'numeric_acceptance.json'),'field_RGB_changed_by_review':False,
  'CPU_only':True,'GPU_forward':0,'native_F':0,'native_B':0,'Gaussian_updates':0,'diffusion_calls':0}
assert not (OUT/'visual_review.json').exists();write(OUT/'visual_review.json',visual)

svg='''<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="270" viewBox="0 0 1040 270"><rect width="1040" height="270" fill="#111827"/><defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#93c5fd"/></marker></defs><g fill="#1f2937" stroke="#64748b"><rect x="20" y="35" width="210" height="100" rx="8"/><rect x="282" y="35" width="215" height="100" rx="8"/><rect x="550" y="35" width="220" height="100" rx="8"/><rect x="822" y="35" width="198" height="100" rx="8"/><rect x="282" y="184" width="488" height="60" rx="8"/></g><g fill="#e5e7eb" font-family="Arial,Microsoft YaHei,sans-serif" text-anchor="middle" font-size="16"><text x="125" y="66">086 / 110 原 raw_model</text><text x="125" y="92">真实 XYZ / Brown / 新保护</text><text x="125" y="118">来源与资格独立重放</text><text x="389" y="67">共享世界全 RGB 联合融合</text><text x="389" y="94">6407 实拍锁定</text><text x="389" y="120">相似实拍边界回退</text><text x="660" y="68">固定域数值 + 五曲面目视</text><text x="660" y="96">numeric PASS</text><text x="660" y="122">visual FAIL</text><text x="920" y="79">Gauss 拟合门</text><text x="920" y="108">NOT_ACCEPTED</text><text x="525" y="208">1792 黑边：1632 新材质 +160 几何不支持</text><text x="525" y="233">原分母完整保留 · 不混入原图 RGB</text></g><g stroke="#93c5fd" stroke-width="2.5" marker-end="url(#a)" fill="none"><path d="M230 85H278"/><path d="M497 85H546"/><path d="M770 85H818"/><path d="M660 184V139"/></g></svg>'''
(OUT/'architecture_components.svg').write_text(svg,encoding='utf-8')
readme='''# R9 共享道路外观场独立验收

数值 PASS，目视 FAIL；当前不进入 Gaussian 拟合。绑定 field SHA `8ad4b9594ef5e14f9f77ebf57a395ebf3da410bb435eb2170f6b2558de9886dd`，原合同、融合结果和失败现场保持。

![Architecture components](architecture_components.svg)

来源采样、权重与完整资格 mask 独立重放误差为0。原每视角世界候选71344/78741保留；当前合法47731/55738，真实锚点4148/5228。固定重叠43624 texels、130872 channels未按robust结果缩减。全RGB残差改善25.8365%，σ=.12m低频改善58.6162%；融合边界低频跳变p95=.00207772（同边seed=.00162079），符合原合同。

6407真实观测与权限外RGB逐字节不变。五帧默认图直接显示完整可用共享曲面材质；不使用旧逐视角C/S/W，不回填原图。全分辨率1152×768 PNG、来源层与域mask均保留。contactsheet仅为显示裁片，不重定义数值分母。

目视可见新增横向/扇形条带与局部色块，在共享图和五个视角中重复，与碎片化teacher-alpha/source3支持空间对应。语义及采样资格通过未消除该共享图缺陷；具体成因不由本QA唯一确定。

历史1792黑边像素完整拆分：1632已由新field RGB覆盖，旧暗阈值下剩余0；160现有几何不支持，明确hatch，不算教师黑色，也未宣称几何/全场景修复。confidence是未校准元数据，未用于本独立验收。

独立入口为numeric_acceptance.json与visual_review.json；independent_metrics.json保留原计算快照，source_sampling_qualification_audit.json补齐来源重放。CPU-only，新增GPU/扩散/Gauss更新均0。
'''
(OUT/'README.md').write_text(readme,encoding='utf-8')
print(json.dumps({k:pin(OUT/k) for k in ['numeric_acceptance.json','visual_review.json','README.md','architecture_components.svg']}))
