"""Append R8 wiring UI and R7 rim evidence; never changes R7 run data/code."""
from pathlib import Path
from datetime import datetime
import hashlib,json,os,shutil,subprocess
import numpy as np
from PIL import Image

WS=Path(r'C:\Users\dengyunlong\Documents\Codex\2026-11-22\065-file-c-users-dengyunlong-documents-2')
ROOT=WS/'work/road_vehicle_joint_dggt_20261123_r8';OUT=WS/'outputs/road_vehicle_joint_dggt_20261123_r8';ASSETS=OUT/'report_assets';WORK=ROOT/'report'
HTML=Path(r'C:\Users\dengyunlong\Documents\Codex\2026-11-16\xian\outputs\road_material_three_rounds_20261121\index.html')
BEGIN=b'<!-- ROAD_VEHICLE_JOINT_DGGT_R8_UI_BEGIN -->';END=b'<!-- ROAD_VEHICLE_JOINT_DGGT_R8_UI_END -->';MOUNT='road_vehicle_joint_dggt_r8'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def asset(path,relative,expected=None,roi=None,focus_roi=None):
 if expected and sha(path)!=expected:raise ValueError('Actual evidence SHA mismatch '+str(path))
 target=ASSETS/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
 record=dict(path=MOUNT+'/'+relative,sha256=sha(path))
 if path.suffix.lower()=='.png':
  with Image.open(path) as im:
   record.update(width=im.width,height=im.height)
   for kind,bounds in [('roi',roi),('focus',focus_roi)]:
    if bounds:
     crop=im.crop(bounds);roi_relative=str(Path(relative).parent/kind/Path(relative).name).replace('\\','/');roi_target=ASSETS/roi_relative;roi_target.parent.mkdir(parents=True,exist_ok=True);crop.save(roi_target)
     assert np.array_equal(np.asarray(crop),np.asarray(im)[bounds[1]:bounds[3],bounds[0]:bounds[2]])
     record[kind]=dict(path=MOUNT+'/'+roi_relative,width=crop.width,height=crop.height,bbox=bounds,source_png_sha256=record['sha256'])
 return record
def main():
 ASSETS.mkdir(parents=True,exist_ok=True)
 contract_path=ROOT/'report_inputs/forward_smoke_five_views/contract.json';contract=read(contract_path)
 assert contract['complete'] and contract['frames']==[86,110,134,179,198] and contract['GPU_calls']==contract['training_updates']==0
 audit_path=ROOT/'audit/a20_baseline/baseline_audit.json';audit=read(audit_path)
 images={}
 for row in contract['rows']:
  raw=row['raw'];stem=f'{raw:03d}'
  pinhole=WS/f'work/road_shared_appearance_20261122_r7/projection_cache_v1/{stem}/photo_pinhole1152.png'
  images[str(raw)]=dict(pinhole=asset(pinhole,f'inputs/{stem}_real_pinhole.png',row['pinhole_image_sha256']),
   model=asset(ROOT/f'report_inputs/forward_smoke_five_views/{stem}_dggt_input.png',f'inputs/{stem}_dggt_input.png',row['model_input_sha256']))
 contract_asset=asset(contract_path,'inputs/contract.json');audit_asset=asset(audit_path,'audit/a20_baseline.json')
 mask_qa=ROOT/'audit/mask_coordinate_audit/mask_coordinate_QA.json';mask_asset=asset(mask_qa,'audit/mask_coordinate_QA.json')
 rim_dir=WS/'work/road_shared_appearance_20261122_r7/cpu_review/black_rim_domain_diagnostic_v1';receipt=read(rim_dir/'delivery_receipt.json');diagnosis=read(rim_dir/'fixed1792_enriched_diagnosis.json')
 ledger={f['name']:f for f in receipt['files']}
 rim=dict(receipt=asset(rim_dir/'delivery_receipt.json','r7_rim/delivery_receipt.json'),
  diagnosis=asset(rim_dir/'fixed1792_enriched_diagnosis.json','r7_rim/fixed1792_enriched_diagnosis.json',ledger['fixed1792_enriched_diagnosis.json']['sha256']),
  figure=asset(rim_dir/receipt['reason_figure'],'r7_rim/'+receipt['reason_figure'],ledger[receipt['reason_figure']]['sha256']),
  fixed_PNG=diagnosis['primary_fixed_rim_pixels'],fixed_float=diagnosis['float_threshold_rim_pixels'],geometry_supported=diagnosis['regions']['rim1792']['flags']['geometry_valid'],geometry_gap=diagnosis['geometry_query_gap_actual_Gaussian_alpha']['pixels'])
 architecture=asset(WORK/'architecture_components.svg','architecture_components.svg')
 baseline=None;first2_stage=ROOT/'takeback/baseline_A0_first2';stage=ROOT/'report_inputs/baseline_A0_five_views'
 if not (stage/'receipt.json').is_file():stage=first2_stage
 if (stage/'receipt.json').is_file():
  baseline_receipt=read(stage/'receipt.json');assert baseline_receipt['complete'] and baseline_receipt['optimizer_updates']==0 and baseline_receipt['actual_nativeB']==0
  source_receipt=read(stage/'source_receipt.json');assert source_receipt['actor_assets'][0]['sha256']==audit['actor']['sha256'] and source_receipt['road_SHA']==audit['renderer']['road']['sha256']
  views={}
  for raw in baseline_receipt['frames']:
   stem=f'{raw:03d}'
   with np.load(WS/f'work/road_shared_appearance_20261122_r7/projection_cache_v1/{stem}/view_cache.npz') as z:mask=z['requested_C']|z['requested_S']
   with np.load(ROOT/f'audit/mask_coordinate_audit/rectified_masks/{stem}/rectified_masks.npz') as z:mask|=z['actor_visible_proxy']
   y,x=np.nonzero(mask);roi=[max(int(x.min())-24,0),max(int(y.min())-24,0),min(int(x.max())+25,1152),min(int(y.max())+25,768)]
   baseline_images={key:asset(stage/stem/(key+'.png'),f'{stage.name}/{stem}/{key}.png',roi=roi) for key in ['rgb','alpha','road_only_rgb','road_only_alpha','actor_only_rgb','actor_only_alpha','road_alpha','actor_alpha','context_alpha','depth_owner_alphaT','expected_depth_m_0_60']}
   record=read(stage/stem/'receipt.json')
   for key in ['road_vs_R7_parent','actor_vs_selected_r3','shared_vs_selected_r3']:
    if key in record:assert record[key]['rgb']['max_abs']==record[key]['alpha']['max_abs']==0
   views[str(raw)]=dict(images=baseline_images,receipt=record,roi=roi)
  reference_receipt=read(first2_stage/'receipt.json')
  for row in reference_receipt['rows']:
   for key in ['road_vs_R7_parent','actor_vs_selected_r3','shared_vs_selected_r3']:assert row[key]['rgb']['max_abs']==row[key]['alpha']['max_abs']==0
  baseline=dict(id=stage.name,frames=baseline_receipt['frames'],receipt=baseline_receipt,source=source_receipt,views=views,
   receipt_file=asset(stage/'receipt.json',stage.name+'/receipt.json'),camera_audit_file=asset(stage/'camera_timestamp_audit.json',stage.name+'/camera_timestamp_audit.json'),
   first2_reference=reference_receipt,first2_reference_file=asset(first2_stage/'receipt.json','baseline_A0_first2/receipt.json'),
   cumulative_nativeF=baseline_receipt['actual_nativeF']+(reference_receipt['actual_nativeF'] if stage.name!='baseline_A0_first2' else 0),cumulative_nativeB=0)
 r9=dict(stage='source_audit_pending',source_report=r'C:\Users\dengyunlong\Documents\Codex\2026-11-19\autodl-worldsim-2d-file-c-users\outputs\road_2d_teacher_defaults_20261120\index.html',
  source_role='已认可原始raw_model道路候选；硬合成回填单列诊断',old_Brown_native_ROI=[512,751,1342,1280],old_Brown_dark_pixels=4173,old_dark_pixels_weight_zero=True,
  R7_pinhole_dark_pixels=1792,source_images=None,shared_surface_result=None,Gaussian_result=None,optimization_updates=0,
  architecture=asset(WORK/'r9_architecture_components.svg','r9/architecture_components.svg'))
 r9root=WS/'work/old_raw_teacher_joint_20261123_r9/audit';mapping_path=r9root/'brown_mapping_audit.json';source_path=r9root/'raw_teacher_sources.json'
 if mapping_path.is_file() and source_path.is_file():
  mapping=read(mapping_path);sources=read(source_path)
  assert mapping['complete'] and mapping['decision']=='PASS_COORDINATE_AUDIT' and mapping['road_surface_acceptance']=='NOT_EVALUATED_BY_THIS_COORDINATE_AUDIT'
  assert sources['complete'] and sources['teacher_view_count']==2 and sources['new_GPU_forward']==sources['new_diffusion_calls']==sources['training_updates']==0
  assert sha(source_path)==mapping['source_manifest']['sha256']
  source_views={}
  for row in mapping['rows']:
   raw=row['raw'];stem=f'{raw:03d}';source=next(s for s in sources['sources'] if s['raw']==raw)
   assert source['seed']==101 and source['prompt_id']=='legacy' and row['old_CS_W_K_used_for_qualification'] is False and row['hard_final_RGB_used_for_supervision'] is False
   with np.load(WS/f'work/road_shared_appearance_20261122_r7/projection_cache_v1/{stem}/view_cache.npz') as z:ui_mask=z['requested_C']|z['requested_S']
   y,x=np.nonzero(ui_mask);roi=[max(int(x.min())-24,0),max(int(y.min())-24,0),min(int(x.max())+25,1152),min(int(y.max())+25,768)]
   output={}
   for key,name in [('raw_pinhole','raw_model_pinhole_preview.png'),('hard_pinhole','hard_final_pinhole_audit_only.png'),('protection_overlay','raw_preview_R8_protected_overlay.png')]:
    output[key]=asset(r9root/f'mapping/{stem}/{name}',f'r9/mapping/{stem}/{name}',row['outputs'][name]['sha256'],roi=roi)
   for key,name,sourcekey in [('raw_Brown','raw_model.png','raw_model'),('hard_Brown','final_native.png','final_native')]:
    output[key]=asset(r9root/f'source_inputs/{stem}/{name}',f'r9/source_inputs/{stem}/{name}',source[sourcekey]['sha256'])
   source_views[str(raw)]=dict(images=output,roi=roi,roi_role='display-only bbox, not source permission',coordinate_domain='pinhole preview from Brown original',source_qualification='not_evaluated',mapping_receipt=asset(r9root/f'mapping/{stem}/receipt.json',f'r9/mapping/{stem}/receipt.json'))
  r9.update(stage='coordinate_audited_source_qualification_pending',frames=[86,110],source_images=source_views,
   coordinate_audit=asset(mapping_path,'r9/brown_mapping_audit.json'),source_manifest=asset(source_path,'r9/raw_teacher_sources.json',mapping['source_manifest']['sha256']),coordinate_status='PASS',surface_status='not_evaluated',new_diffusion_calls=0)
 shared_dir=r9root.parent/'cpu_review/joint_shared_v1'
 if (shared_dir/'numeric_acceptance.json').is_file() and (shared_dir/'visual_review.json').is_file():
  numeric=read(shared_dir/'numeric_acceptance.json');visual=read(shared_dir/'visual_review.json');field=r9root.parent/'joint_shared_v1/final_atlas.npz'
  assert numeric['complete'] and visual['complete'] and numeric['decision']=='PASS' and visual['decision']=='FAIL'
  assert sha(field)==numeric['final_atlas_sha256']==visual['final_atlas_sha256']
  assert numeric['Gaussian_updates']==visual['Gaussian_updates']==0 and numeric['diffusion_calls']==visual['diffusion_calls']==0
  metrics=read(shared_dir/'independent_metrics.json');assert sha(shared_dir/'independent_metrics.json')==numeric['independent_metrics']['sha256']
  alpha_receipt=read(shared_dir/'teacher_influence_display_receipt.json');assert alpha_receipt['complete'] and alpha_receipt['field']['sha256']==numeric['final_atlas_sha256'] and alpha_receipt['field_array']=='effective_teacher_blend'
  alpha_ledger={row['raw']:row for row in alpha_receipt['rows']}
  views={}
  for raw in visual['frames']:
   stem=f'{raw:03d}';scope=read(shared_dir/f'{stem}_scope.json')
   assert not scope['old_CS_W_used_for_scope'] and not scope['photo_pixels_used_as_material'] and scope['authorized_unknown']==0
   with np.load(shared_dir/f'{stem}_domain_masks.npz') as z:domain=z['geometry']
   y,x=np.nonzero(domain);roi=[max(int(x.min())-12,0),max(int(y.min())-12,0),min(int(x.max())+13,1152),min(int(y.max())+13,768)]
   with np.load(WS/f'work/road_shared_appearance_20261122_r7/projection_cache_v1/{stem}/view_cache.npz') as z:display_focus=z['requested_C']|z['requested_S']
   y,x=np.nonzero(display_focus);focus_roi=[max(int(x.min())-24,0),max(int(y.min())-24,0),min(int(x.max())+25,1152),min(int(y.max())+25,768)]
   image_records={key:asset(shared_dir/f'{stem}_{key}.png',f'r9/joint_shared_v1/{stem}/{key}.png',roi=roi,focus_roi=focus_roi) for key in ['material_hatched','material_only','seed_material_hatched','seed_material_only','source_kind','optional_photo_context']}
   for key,filename in [('raw_pinhole','raw_model_pinhole_preview.png'),('hard_pinhole','hard_final_pinhole_audit_only.png')]:
    path=r9root/f'mapping/{stem}/{filename}'
    if path.is_file():image_records[key]=asset(path,f'r9/joint_shared_v1/{stem}/{key}.png',roi=roi,focus_roi=focus_roi)
   alpha_path=shared_dir/f'{stem}_teacher_alpha.png'
   if alpha_path.is_file():image_records['teacher_alpha']=asset(alpha_path,f'r9/joint_shared_v1/{stem}/teacher_alpha.png',alpha_ledger[raw]['PNG']['sha256'],roi=roi,focus_roi=focus_roi)
   views[str(raw)]=dict(images=image_records,roi=roi,focus_roi=focus_roi,focus_role='display-only rectangle, never a mask or scientific denominator',scope=scope,scope_file=asset(shared_dir/f'{stem}_scope.json',f'r9/joint_shared_v1/{stem}/scope.json'))
  shared=dict(id='joint_shared_v1',field_sha256=numeric['final_atlas_sha256'],frames=visual['frames'],numeric_status=numeric['decision'],visual_status=visual['decision'],root_decision='不进入高斯拟合',Gaussian_result=None,
   fixed_pair=numeric['fixed_pairs'][0],boundary_LF_p95=numeric['boundary_LF_jump']['p95'],fixed1792=numeric['fixed1792'],views=views,
   numeric_file=asset(shared_dir/'numeric_acceptance.json','r9/joint_shared_v1/numeric_acceptance.json'),visual_file=asset(shared_dir/'visual_review.json','r9/joint_shared_v1/visual_review.json'),
   source_qualification_file=asset(shared_dir/'source_sampling_qualification_audit.json','r9/joint_shared_v1/source_sampling_qualification_audit.json',numeric['source_sampling_qualification']['sha256']),
   contract_file=asset(r9root.parent/'acceptance_contract_v1.json','r9/joint_shared_v1/acceptance_contract_v1.json',numeric['contract']['sha256']),alpha_receipt_file=asset(shared_dir/'teacher_influence_display_receipt.json','r9/joint_shared_v1/teacher_influence_display_receipt.json'))
  shared['world']={key:asset(shared_dir/(key+'.png'),f'r9/joint_shared_v1/{key}.png') for key in ['world_atlas_seed_final','world_source_kind','world_teacher_alpha','five_view_seed_final_material_contactsheet']}
  r9.update(stage='shared_v1_visual_failed_continuous_v2_pending',surface_status='visual_failed',shared_v1=shared,shared_surface_result=dict(id='joint_shared_v1',field_sha256=numeric['final_atlas_sha256'],visual_status='FAIL'),source_qualification_status='PASS')
  continuous_contract=r9root.parent/'continuous_field_contract_v2.json'
  if continuous_contract.is_file():r9['continuous_v2']=dict(stage='pending_independent_and_root_review',Gaussian_result=None,contract_file=asset(continuous_contract,'r9/continuous_field_contract_v2.json'))
 current_dir=r9root.parent/'cpu_review/joint_shared_continuous_v2';root_review_path=current_dir/'root_shared_input_review.json'
 if root_review_path.is_file():
  numeric=read(current_dir/'numeric_acceptance.json');visual=read(current_dir/'visual_review.json');root_review=read(root_review_path)
  assert numeric['complete'] and visual['complete'] and root_review['complete']
  assert numeric['decision']==visual['decision']==root_review['decision']=='PASS'
  assert root_review['root_concur'] and not root_review['whole_scene_restoration_accepted']
  assert sha(current_dir/'numeric_acceptance.json')==root_review['numeric_acceptance']['sha256'] and sha(current_dir/'visual_review.json')==root_review['independent_visual_review']['sha256']
  field_sha=sha(r9root.parent/'joint_shared_continuous_v2/final_atlas.npz');assert field_sha==numeric['final_atlas_sha256']==visual['final_atlas_sha256']==root_review['final_atlas_sha256']
  alpha=read(current_dir/'teacher_influence_display_receipt.json');assert alpha['field']['sha256']==field_sha
  assert not numeric['conditional_influence_is_pointwise_RGB_blend_alpha'] and numeric['Gaussian_updates']==0
  alpha_ledger={row['raw']:row for row in alpha['rows']};material_ledger={row['raw']:row['material_only'] for row in visual['per_view_notes']}
  views={}
  for raw in visual['frames']:
   stem=f'{raw:03d}';old=r9['shared_v1']['views'][str(raw)];roi=old['roi'];focus_roi=old['focus_roi'];scope=read(current_dir/f'{stem}_scope.json')
   assert scope['same_evaluated_domain_bitexact_v1'] and not scope['old_CS_W_used'] and not scope['photo_RGB_used_as_material']
   images_current={}
   for key in ['material_hatched','material_only','seed_material_hatched','seed_material_only','source_kind','optional_photo_context']:
    expected=material_ledger[raw]['sha256'] if key=='material_only' else None
    images_current[key]=asset(current_dir/f'{stem}_{key}.png',f'r9/continuous_v2/{stem}/{key}.png',expected,roi=roi,focus_roi=focus_roi)
   images_current['teacher_influence']=asset(current_dir/f'{stem}_teacher_alpha.png',f'r9/continuous_v2/{stem}/teacher_influence.png',alpha_ledger[raw]['PNG']['sha256'],roi=roi,focus_roi=focus_roi)
   for key in ['raw_pinhole','hard_pinhole']:
    if key in old['images']:images_current[key]=old['images'][key]
   views[str(raw)]=dict(images=images_current,scope=scope,scope_file=asset(current_dir/f'{stem}_scope.json',f'r9/continuous_v2/{stem}/scope.json'))
  grain=next(row for row in numeric['source_grain'] if row['domain']=='complete_actual_updated_domain' and row['HF_sigma_m']==.02)
  seam=next(row for row in numeric['boundary_LF'] if row['domain']=='fixed_v1_teacher_boundary')
  continuous=dict(id='joint_shared_continuous_v2',stage='shared_input_accepted',field_sha256=field_sha,frames=visual['frames'],numeric_status='PASS',visual_status='PASS',root_status='PASS',Gaussian_result=None,
   views=views,source_grain=grain,boundary_LF=seam,lineage_tail=numeric['lineage_threshold_tail'],fixed1792=numeric['fixed1792'],parameter_contract=root_review['constraints'],
   contract_file=asset(r9root.parent/'continuous_field_contract_v2.json','r9/continuous_v2/contract.json',numeric['contract']['sha256']),numeric_file=asset(current_dir/'numeric_acceptance.json','r9/continuous_v2/numeric_acceptance.json',root_review['numeric_acceptance']['sha256']),
   visual_file=asset(current_dir/'visual_review.json','r9/continuous_v2/visual_review.json',root_review['independent_visual_review']['sha256']),root_file=asset(root_review_path,'r9/continuous_v2/root_shared_input_review.json','fa1490aafd847c4e6bd4e9e2b1193ac6232d6cae29ed8672be4aa4be0652a25f'),alpha_file=asset(current_dir/'teacher_influence_display_receipt.json','r9/continuous_v2/teacher_influence_display_receipt.json'))
  continuous['world']={key:asset(current_dir/(key+'.png'),f'r9/continuous_v2/{key}.png') for key in ['world_atlas_seed_final','world_source_kind_observed_inferred','world_teacher_influence','five_view_seed_final_material_contactsheet','five_view_v1_v2_material_contactsheet']}
  r9.update(stage='shared_continuous_v2_input_accepted_fit_not_run',surface_status='shared_input_PASS',continuous_v2=continuous,shared_surface_result=dict(id='joint_shared_continuous_v2',field_sha256=field_sha,visual_status='PASS'),architecture=asset(WORK/'r9_current_architecture_components.svg','r9/continuous_v2/architecture_components.svg'),
   joint_fit=dict(status='not_run',result=None,manifest_schema='R9.joint_native_RGB_fit.report.v1',manifest_path=str(r9root.parent/'joint_fit_report_manifest.json'),adapter='load_joint_fit_manifest(manifest_path, *, shared_field_sha256, road_parent_sha256, actor_parent_sha256)'))
 coverage=None;coverage_dir=ROOT/'audit/baseline_A0_coverage_v1';context_path=ROOT/'audit/context_owner_CPU_v1/context_inventory_and_mixture.json'
 if (coverage_dir/'coverage_occlusion_audit.json').is_file() and context_path.is_file():
  cov=read(coverage_dir/'coverage_occlusion_audit.json');context=read(context_path)
  assert cov['complete'] and cov['CPU_only'] and cov['new_nativeF']==cov['new_nativeB']==cov['optimizer_updates']==cov['new_teacher_calls']==0
  assert context['complete'] and context['CPU_only'] and context['GPU_calls']==context['optimizer_updates']==0
  rows=[]
  for row in cov['frames']:
   scopes={s['scope']:s for s in row['scopes']};original=scopes['original_CS'];expanded=scopes['full_CS_dilation8'];visible=scopes.get('actor_visible_frozen_V')
   rows.append(dict(raw=row['raw'],original_pixels=original['pixels'],expanded_pixels=expanded['pixels'],original_scene_alpha95_low=original['thresholds']['0.95']['scene_total_alpha_low'],expanded_scene_alpha95_low=expanded['thresholds']['0.95']['scene_total_alpha_low'],frozen_visible=None if visible is None else dict(pixels=visible['pixels'],isolated_alpha95_low=visible['thresholds']['0.95']['actor_isolated_alpha_low'],isolated_sufficient_shared_owner_alpha95_low=visible['thresholds']['0.95']['actor_isolated_sufficient_joint_actor_low'])))
  coverage=dict(rows=rows,receipt=asset(coverage_dir/'coverage_occlusion_audit.json','coverage/coverage_occlusion_audit.json'),figure=asset(coverage_dir/'five_view_coverage_occlusion.png','coverage/five_view_coverage_occlusion.png'),context_receipt=asset(context_path,'coverage/context_inventory_and_mixture.json'),new_nativeF=0,new_nativeB=0,optimizer_updates=0)
 owner=None;owner_dir=ROOT/'audit/owner_unification_CPU_v1'
 if (owner_dir/'owner_review_delivery.json').is_file():
  delivery=read(owner_dir/'owner_review_delivery.json');own=read(owner_dir/'owner_unification_audit.json');ledger={Path(row['path']).name:row for row in delivery['files']}
  assert delivery['complete'] and delivery['R8_cumulative_F']==27 and delivery['R8_cumulative_B']==2 and not delivery['training_enabled']
  assert own['complete'] and own['parameter_updates']==0 and own['optimization_not_executed']
  owner=dict(rows=[dict(raw=int(raw),**row['exact_unified_owner_integral']) for raw,row in own['frame_results'].items()],cumulative_F=delivery['R8_cumulative_F'],cumulative_B=delivery['R8_cumulative_B'],measurement_F=delivery['prior_actual_measurement_F'],measurement_B=delivery['prior_actual_measurement_B'],updates=0,
   receipt=asset(owner_dir/'owner_unification_audit.json','owner/owner_unification_audit.json',ledger['owner_unification_audit.json']['sha256']),figure=asset(owner_dir/'same_instance_owner_review.png','owner/same_instance_owner_review.png',ledger['same_instance_owner_review.png']['sha256']),delivery=asset(owner_dir/'owner_review_delivery.json','owner/owner_review_delivery.json'))
 colour_guard=None;guard_dir=ROOT/'joint_colour_guard_v1'
 if (guard_dir/'receipt.json').is_file():
  guard=read(guard_dir/'receipt.json')
  assert guard['complete'] and guard['nativeF']==guard['nativeB']==5 and guard['updates']==0 and guard['source_RGBA_exact']
  assert sha(guard_dir/'colour_guard.npz')==guard['guard_sha256']=='8025f666a402c44442695fa019bc4e244eefdc036bda1caced1fe2e332625187'
  assert guard['source_sha256']==r9['continuous_v2']['field_sha256']
  assert all(row['source_RGB_max_abs']==row['source_alpha_max_abs']==0 for row in guard['rows'])
  colour_guard=dict(receipt=asset(guard_dir/'receipt.json','colour_guard/receipt.json'),rows=guard['rows'],guard_sha256=guard['guard_sha256'],nativeF=5,nativeB=5,updates=0,cumulative_F=owner['cumulative_F']+5,cumulative_B=owner['cumulative_B']+5)
 data=dict(schema='R8.road_vehicle_wiring_UI.v1',stage='aligned_baseline' if baseline else 'wiring',title='道路＋A20联合重建',frames=contract['frames'],images=images,input_contract=contract,
  contract=contract_asset,a20_audit=audit_asset,mask_audit=mask_asset,architecture=architecture,
  baseline_actor_sha256=audit['actor']['sha256'],road_sha256=audit['renderer']['road']['sha256'],
  model_result=None,native_baseline_result=baseline,optimization_updates=0,
  official_dggt=dict(status='waiting_actual_forward_and_render',result=None,manifest_schema='R8.official_DGGT_noDiff_forward.report.v1',manifest_path=str(ROOT/'official_dggt_report_manifest.json'),adapter='load_dggt_forward_manifest(manifest_path)',independent_of_native_joint_RGB_fit=True),
  official_branch='原版DGGT多帧Transformer无扩散前向对照',adaptation_branch='A20＋RoadRows初始化的native共享渲染适配',
  parameter_contract=dict(centres_frozen=True,rigid_trajectory_frozen=True,other_vehicles_protected=True,local_scale_opacity_only_after_coverage_holes_confirmed=True,RGB_joint_fit=True,optimization_currently_paused=True),coverage=coverage,owner=owner,colour_guard=colour_guard,rim=rim,r9=r9)
 mount=HTML.parent/MOUNT
 if not mount.exists():subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',f"New-Item -ItemType Junction -Path '{mount}' -Target '{ASSETS}' | Out-Null"],check=True)
 assert mount.resolve()==ASSETS.resolve()
 before=HTML.read_bytes()
 if BEGIN in before:
  start=before.index(BEGIN);end=before.index(END,start)+len(END);prefix=before[:start];suffix=before[end:]
 else:prefix=before;suffix=b''
 payload=json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c').encode('utf-8')
 block=BEGIN+b'\r\n<script id="road-vehicle-dggt-r8-data" type="application/json">'+payload+b'</script>\r\n<script>\r\n'+(WORK/'wiring_viewer.js').read_bytes()+b'\r\n</script>\r\n'+END
 backup=WORK/('index.before_r8_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.html.bak');backup.write_bytes(before)
 after=prefix+block+suffix;assert HTML.read_bytes()==before;HTML.write_bytes(after)
 (OUT/'report_data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
 publication=dict(report=str(HTML),anchor='road-vehicle-joint-dggt',before_sha256=hashlib.sha256(before).hexdigest(),after_sha256=hashlib.sha256(after).hexdigest(),backup=str(backup),outside_R8_UI_block_exact=True,R7_code_compact_and_run_data_modified=False,model_result_published=False,optimization_updates=0,HTML_frozen=False)
 (OUT/'report_publication_receipt.json').write_text(json.dumps(publication,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(publication,ensure_ascii=False))
if __name__=='__main__':main()
