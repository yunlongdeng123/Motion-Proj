"""Bind actual native-resolution scene observations; keep numerical gates separate."""
from pathlib import Path
from verify_joint_asset_cpu import read,write,pin,SOURCE_SHA

ROOT=Path(__file__).resolve().parents[1];AUDIT=ROOT/'joint_persistent_asset_audit_v1';OUT=AUDIT/'formal_visual_review'
inputs=read(OUT/'input_bindings.json');gate=read(AUDIT/'formal400_v1_joint_asset_gate.json')
assert gate['decision']=='PASS'
notes={
86:{'road':['Compared with A0, the central road tone is more even. Fine grain remains softer than the direct shared-surface readout.',
            'Jagged coloured edges near the lane boundary and the material perimeter remain visible.'],
    'actor':['A20 dark-blue body and cab/window colours change modestly; no obvious new silhouette deformation is visible.',
             'A conspicuous red/blue/grey patch beneath the vehicle remains in the shared render.'],
    'overall':'Colour control produces a limited local improvement; unresolved underbody and boundary artifacts prevent whole-scene acceptance.'},
110:{'road':['The near road interior is more uniform than A0; the road-only layer still has irregular support edges and streaked primitive boundaries.',
             'A lower-field wavy/fragmented coloured boundary remains visible in the actual shared scene.'],
     'actor':['Cab/window highlights are brighter than A0 while the main body shape is retained.',
              'The red slab and dark/cyan smear below the bumper remain; the isolated actor image does not contain the same ground smear.'],
     'overall':'The source input passed its material gate, but the native road/vehicle scene still contains conspicuous residuals.'},
134:{'road':['No obvious global colour collapse relative to A0 is visible; central tone is more even.',
             'The inherited fragmented lane edges and lower-road streaks remain in the road-only and shared readouts.'],
     'actor':['Held-out view shows a modest cab/window colour change without a new gross hue shift.',
              'The underbody red/blue residual persists in the shared scene.'],
     'overall':'Evaluation-only colour remains broadly stable; the same unresolved scene artifacts remain.'},
179:{'road':['The road colour is broadly stable relative to A0 and shows no obvious new global tint.',
             'Support-edge roughness and streaked patches remain; the clean shared field appearance does not transfer as a perfect full-scene restoration.'],
     'actor':['Body/cab changes are consistent with the other views; no obvious new large colour deterioration is seen.',
              'The coloured/dark underbody footprint is still visible.'],
     'overall':'No obvious new whole-view colour deterioration, while residual scene defects remain.'},
198:{'road':['The fifth view remains close in overall tone to179 and to its A0 baseline.',
             'The same boundary and underbody-region material/context defects remain.'],
     'actor':['The A20 appearance remains consistent with179; no new major hue change is apparent.',
              'Persistent red/blue/grey material below the vehicle prevents an artifact-free scene claim.'],
     'overall':'Colour-only control is reproducible across the diagnostic views; whole-scene repair is not accepted.'}
}
rows=[]
for source in inputs['rows']:
    raw=source['raw'];row={'raw':raw,'original_resolution_wh':[1152,768],
      'shared_road_actor_original_PNGs_independently_inspected':True,
      'baseline_original_PNGs_and_native_focus_crops_inspected':True,
      'files':source['files'],'notes':notes[raw],
      'training_visibility_supervision_available':raw in [86,110],
      'evaluation_only':raw in [134,179,198],
      'missing_strict_V_is_not_zero_ground_truth':True,
      'native_actor_focus_triptych':pin(OUT/f'{raw:03d}_photo_A0_formal_actor_focus.png'),
      'native_shared_underbody_pair':pin(OUT/f'{raw:03d}_shared_underbody_focus_native_pair.png')}
    rows.append(row)
review={'schema':'R9.actual_formal_joint_native_scene_visual_review.v1','complete':True,
  'decision':'FAIL','full_restoration_decision':'NOT_ACCEPTED','scope':'Whole actual native scene visual restoration, not the already accepted input surface material or persistent-array invariants.',
  'source_shared_input_material_decision':'PASS','persistent_asset_and_native_reload_decision':'PASS',
  'colour_only_result':'Limited local road/actor colour improvement with inherited scene artifacts unresolved',
  'root_concur':False,'root_review_status':'PENDING separate root final-scene review',
  'source_field_sha256':SOURCE_SHA,'formal_asset_gate':pin(AUDIT/'formal400_v1_joint_asset_gate.json'),
  'formal_model_assets':gate['formal_model_assets'],'input_bindings':pin(OUT/'input_bindings.json'),
  'frames':[86,110,134,179,198],'per_view_observations':rows,
  'summary':[
    'All five original1152x768 shared, road and actor PNGs were inspected against the SHA-verified A0 images; current output colours are not judged from numerical PASS alone.',
    'Road interior tone is more even; A20 body/cab colours change modestly. The three evaluation-only views show no obvious new global colour collapse.',
    'The shared scene retains conspicuous red/blue/grey underbody remnants, jagged/fragmented road boundaries and streaked primitive appearance. Whole-scene visual restoration is not accepted.',
    'The isolated actor image lacks the same ground smear. This observation does not uniquely attribute the residual to a specific Gaussian owner or prove a new alpha hole.',
    'Black outside the road-only/actor-only footprint is component background or missing component support, not a black road-teacher target.',
    'The old1792-pixel source-domain rim mask is not redefined as a new native-render quality denominator by this visual review.'
  ],
  'new_GG_dark_pixel_count_claimed':False,'historical1792_definition_changed':False,
  'Gaussian_centres_scales_opacity_rotation_or_tracks_changed_by_review':False,
  'formal_native_images_are_actual_Gaussian_renders':True,'review_modified_any_RGB':False,
  'shared_contactsheet':pin(OUT/'five_view_shared_A0_formal_contactsheet.png'),
  'CPU_only':True,'auditor_GPU_forward':0,'auditor_GPU_backward':0,'diffusion_calls':0,'updates':0}
write(OUT/'final_visual_review.json',review)
svg='''<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="320" viewBox="0 0 1120 320"><rect width="1120" height="320" fill="#111827"/><defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#93c5fd"/></marker></defs><g fill="#1f2937" stroke="#64748b"><rect x="20" y="40" width="205" height="100" rx="8"/><rect x="280" y="40" width="230" height="100" rx="8"/><rect x="565" y="40" width="240" height="100" rx="8"/><rect x="860" y="40" width="240" height="100" rx="8"/><rect x="565" y="205" width="535" height="80" rx="8"/></g><g fill="#e5e7eb" font-family="Arial,Microsoft YaHei,sans-serif" text-anchor="middle" font-size="16"><text x="122" y="70">共享场 + 原道路/A20</text><text x="122" y="96">冻结几何 / 轨迹 / K</text><text x="122" y="123">源场输入已验收</text><text x="395" y="70">实际联合颜色拟合400步</text><text x="395" y="96">道路RGB / 车辆SH DC</text><text x="395" y="123">其余字段固定</text><text x="685" y="70">持久资产 + 公共加载</text><text x="685" y="96">全部字段 / 浮点文件核验</text><text x="685" y="123">PASS</text><text x="980" y="70">五视角实际场景目视</text><text x="980" y="96">车下残影 / 路面碎边</text><text x="980" y="123">整场景 NOT_ACCEPTED</text><text x="832" y="236">smoke：恢复逐字节一致 / 下一步浮点重放按登记界核验</text><text x="832" y="263">v1严格失败保留 · 正式方法未改 · 核验器不启动GPU</text></g><g stroke="#93c5fd" stroke-width="2.5" marker-end="url(#a)" fill="none"><path d="M225 90H276"/><path d="M510 90H561"/><path d="M805 90H856"/><path d="M685 205V144"/></g></svg>'''
(AUDIT/'architecture_components.svg').write_text(svg,encoding='utf-8')
text='''# 联合颜色资产核验与最终目视

正式400步持久资产与新鲜公共加载核验PASS；实际场景目视为NOT_ACCEPTED。道路内域色调更均匀，车辆颜色变化有限，车下残影和道路碎边仍需保留为未解决结果。

![Architecture components](architecture_components.svg)

道路596100行只放开reference_srgb与新增derived reference_linear，当前effective_row_rgb与sRGB一致。历史capacity_effective_rgb保留原字节。车辆419005行只允许coefficients的DC项变化，SH1..15、XYZ、尺度、不透明度、旋转、active和全部原lineage冻结。轨迹、相机、上下文和源场SHA固定。

五视角真实VJP保护并集冻结道路3463行、A20 DC6570行。所有质量数组行数完整、finite且非负，并集由原mass重新核对；后三视角严格V未打开，仅用于评价，不当零真值。原生渲染保护区RGB对A0逐字节一致。

独立核验器verify_joint_asset_cpu.py提供freeze、check、render-check三个CPU入口。新鲜加载证据来自root显式GPU进程；CPU逐文件核对五帧全部joint/component键、所有像素、dtype/shape/bytes和finite，另核相机K、时间戳、模型时间、全局alpha。静态gate合法迁移路径时，对NPZ IDs和checkpoint原始tensor storage/元数据独立比较，不导入Torch。

smoke_v1严格下一步重放失败保留。v2完整恢复1290叶逐字节一致；下一步1291叶按预登记4eps(dtype)*(1+abs(ref))全量核对，参数最大差2.98e-8、optimizer状态最大差2.27e-13、loss差0。RNG、step、非float及optimizer超参严格一致。正式400步方法、采样、学习率、先验与source/guard合同未变。

正式资产门为formal400_v1_joint_asset_gate.json。最终五视角观察和原像素对照位于formal_visual_review/final_visual_review.json；输入共享场PASS、持久加载PASS与整场景NOT_ACCEPTED分别记录，未以数值通过替代目视判断。所有本核验器工作为CPU，未新增GPU、扩散或训练。
'''
(AUDIT/'README.md').write_text(text,encoding='utf-8')
print({name:pin(path) for name,path in {'visual':OUT/'final_visual_review.json','README':AUDIT/'README.md','architecture':AUDIT/'architecture_components.svg'}.items()})
