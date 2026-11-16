"""冻结原始86/110 completion C与实拍A；其余帧仅诊断，正式更新为0。"""
from pathlib import Path
import sys,json,hashlib,os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='2'
import numpy as np
from PIL import Image
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v82_gn11_v42/scripts/worldsim_v82')
from drivingeditor_dynamic_edit.road_asset_supervision import direct_teacher_arrays
OLD=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_mask_decoupling_20261116_r1')
PREV=OLD.parent/'road_asset_20261116_r1'
BASE=OLD.parent/'road_direct_teacher_20261116_r1'
OUT=BASE/'supervision_r3';OUT.mkdir(parents=True,exist_ok=False)
def read(p):return json.loads(Path(p).read_text())
def load(p):
 with np.load(p) as z:return {k:z[k].copy() for k in z.files}
def pin(p):return dict(path=str(Path(p).resolve()),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def rgb(p):return np.asarray(Image.open(p).convert('RGB')).copy()
def mask(p):return np.asarray(Image.open(p).convert('L'))>127
def image(p,a):Image.fromarray(np.uint8(np.clip(a.transpose(1,2,0),0,1)*255+.5)).save(p)
prepared={x['id']:x for x in read(OLD/'prepared/prepared.json')['cases']}
cases={x['id']:x for x in read(OLD/'output_guard_probe/control_freeze.json')['cases']}
cfg=read(PREV.parent/'scene065_material_loop_20261114_r1/config.json')
rows=[];pins=[]
for raw in [86,110]:
 sid=f'scene065_raw{raw:03d}';case=cases[sid];prep=prepared[sid]
 paths=dict(observation=case['observation']['path'],teacher_C=case['source_raw_B']['path'],
  guard=case['native_guard_roles']['path'],roles=prep['views']['B']['roles_native']['path'],common=case['original_common_native']['path'],
  road_observed=case['legal_input_artifacts']['road_observed']['path'],F_control=str(PREV/'phase0/supervision_triangle_r8'/f'{raw:03d}.npz'),
  initial=str(PREV/'phase2/formal_r1/rendered'/f'{raw:03d}'/'arrays.npz'),view=str(Path(cfg['road_views'])/f'{raw:03d}.npz'))
 guard=load(paths['guard']);role=load(paths['roles']);common=load(paths['common']);F=load(paths['F_control']);init=load(paths['initial']);view=load(paths['view'])
 role.update(W_guard=guard['W_guard'],H_B=guard['H_B'],K=guard['K'],physical_unknown=common['physical'])
 geom=dict(surface_valid=F['surface_intersection_valid']&F['geometry_support'],
   retained_guard=view['retained_reveal_guard']|view['retained_box_guard']|F['retained_world_guard'],K=F['K'])
 arrays=direct_teacher_arrays(rgb(paths['observation']),rgb(paths['teacher_C']),role,mask(paths['road_observed']),geom,F,init['road_only_alpha'])
 arrays.update(initial_road_premult_srgb=init['road_only_rgb'],ground_world_xyz=F['ground_world_xyz'],world_row_index=F['world_row_index'],
  triangle_world_rows=F['triangle_world_rows'],triangle_weights=F['triangle_weights'],F_control_confidence=F['confidence'])
 arrays['target_A']=arrays['target_A_srgb'];arrays['target_C']=arrays['target_C_srgb']
 np.savez_compressed(OUT/f'{raw:03d}.npz',**arrays)
 image(OUT/f'{raw:03d}_teacher_full.png',arrays['target_C_srgb']);image(OUT/f'{raw:03d}_observed_full.png',arrays['target_A_srgb'])
 overlay=arrays['target_A_srgb'].transpose(1,2,0).copy()
 for m,c in [(arrays['A_mask'],[.1,.9,.4]),(arrays['C_mask'],[1,.2,.55]),(arrays['K'],[.2,.55,1.])]:overlay[m]=.45*overlay[m]+.55*np.asarray(c)
 Image.fromarray(np.uint8(np.clip(overlay,0,1)*255+.5)).save(OUT/f'{raw:03d}_A_C_K_overlay.png')
 C=arrays['C_mask'];A=arrays['A_mask'];alpha=arrays['fixed_initial_alpha'];P=arrays['initial_road_premult_srgb']
 row=dict(raw=raw,training_allowed=True,native_physical_C_pixels=int((role['physical_unknown']&role['W_guard']&~role['K']).sum()),
  native_full_W_pixels=int(role['W_guard'].sum()),native_effect_only_pixels=int((role['W_guard']&~role['physical_unknown']&~role['K']).sum()),
  A_pixels=int(A.sum()),C_common_pixels=int(C.sum()),C_teacher_qualified_pixels=int((arrays['C_teacher_hole']&arrays['fixed_high_alpha']).sum()),
  coverage_A_pixels=int(arrays['coverage_A'].sum()),coverage_C_pixels=int(arrays['coverage_C'].sum()),
  C_teacher_only_pixels=int(arrays['C_teacher_only'].sum()),C_teacher_only_photo_pixels=int(arrays['C_teacher_only_photo'].sum()),C_full_footprint_pixels=int(arrays['C_full_hole'].sum()),
  C_full_core_pixels=int(arrays['C_full_core'].sum()),C_initial_alpha_excluded=int((arrays['C_teacher_hole']&~arrays['fixed_high_alpha']).sum()),
  A_C_overlap=int((A&C).sum()),q_A_sum=float(arrays['q_A'].sum()),q_C_sum=float(arrays['q_C'].sum()),
  q_A_coverage_sum=float(arrays['q_A_coverage'].sum()),q_C_coverage_sum=float(arrays['q_C_coverage'].sum()),
  coverage_outside_overlap=int((arrays['coverage_mask']&arrays['outside']).sum()),
  teacher_F_signal_sRGB_MAE=float(np.abs(arrays['target_C_srgb']-arrays['F_control_C_srgb'])[:,C].mean()) if C.any() else None,
  initial_A_premult_MAE=float(np.abs(P-alpha[None]*arrays['target_A_srgb'])[:,A].mean()) if A.any() else None,
  initial_C_premult_MAE=float(np.abs(P-alpha[None]*arrays['target_C_srgb'])[:,C].mean()) if C.any() else None,
  source_files={k:pin(p) for k,p in paths.items()})
 rows.append(row);pins.extend(row['source_files'].values())
manifest=dict(schema='road_direct_teacher_supervision.v1',teacher_frames=[86,110],diagnostic_only_frames=[134,179,198],
  teacher_role='frozen historical B raw_native is current C pseudo teacher; no hidden GT',
  colors='native sRGB premultiplied loss; linear arrays audit-only; no EOTF of premultiplied shader output',
  C_mask='all4x4 physical_unknown & frozen W_guard & same road surface & no retained guards/K & fixed initialalpha>.9 & F_control_valid',
  A_mask='all4x4 actual road_observed & C_mat & visible_support & ~H_B &~K, same road/guard/initialalpha qualification',
  q='binary full-native-footprint/geometry validity only, not calibrated SD correctness; same q/mask in actualC and Fcontrol arms',
  C_effect='W_guard outside physical hole; diagnostic only, no shadow teacher loss',
  coverage='A_visible | C_teacher_hole; physical geometry validity only, independent of F_control_color_valid and fixed_initial_alpha',
  outside='complement of full authorized coverage union K, not complement of high-alpha photo mask',
  B_protocol='no formal blind B this round; blind claim requires world-region selection and exclusion from ALL pre-cache RGB, atlas, retrieval, observed anchors and teachers before asset construction',
  blind_B_claimed=False,SD_frozen=True,theta_prior_frozen=True,z_or_eta_estimated=False,formal_updates=0,human_verdict=None,rows=rows)
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
freeze=dict(schema='immutable_direct_teacher_supervision.v1',source_files=pins,builder=pin(__file__),
 target_files=[pin(OUT/f'{raw:03d}.npz') for raw in [86,110]],manifest=pin(OUT/'manifest.json'),formal_updates=0)
(OUT/'freeze.json').write_text(json.dumps(freeze,indent=2));print(json.dumps(manifest,ensure_ascii=False),flush=True)
