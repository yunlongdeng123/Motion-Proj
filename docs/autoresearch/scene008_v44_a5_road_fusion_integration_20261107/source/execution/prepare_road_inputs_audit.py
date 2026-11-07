from pathlib import Path
import hashlib,json
import numpy as np
B=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
ROOT=B/'scene008_v44_a5_road_fusion_integration_20261107_r1'
AA=B/'scene008_road_seam_contact_20261107_r1/engineer/canonical_n2_mip_candidate_v2'
C=B/'scene008_road_fusion_bake_20261107_r1/engineer/canonical_edgeblend_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,value):
 p=ROOT/'contract'/name;assert not p.exists();p.write_text(json.dumps(value,indent=2)+'\n')
assets=[];records=[]
for name,d in [('AA',AA),('C',C)]:
 m=json.loads((d/'scene_manifest.json').read_text());p=d/m['new_road']['file'];assert sha(p)==m['new_road']['sha256']
 with np.load(p,allow_pickle=False) as a:values={k:a[k].copy() for k in m['new_road']['tuple_fields']}
 assert all(len(v)==40392 for v in values.values());assets.append(values)
 records.append({'variant':name,'manifest':str(d/'scene_manifest.json'),'manifest_sha256':sha(d/'scene_manifest.json'),
  'asset':str(p),'asset_sha256':sha(p),'rows':40392,'parent_rows':m['new_road']['parent_rows'],'renderer_policy':m['renderer_policy'],
  'frozen_v44_binding_sha256':m['frozen_v44_binding_sha256']})
checks={}
for k in assets[0]:
 equal=np.array_equal(assets[0][k],assets[1][k]);checks[k]={'array_equal':equal,'dtype':str(assets[0][k].dtype),'shape':list(assets[0][k].shape),'max_abs_difference':float(np.abs(assets[0][k]-assets[1][k]).max())}
 if k!='render_colours_srgb':assert equal,k
colours=assets[0]['render_colours_srgb']!=assets[1]['render_colours_srgb']
write('road_AA_C_asset_pair_audit.json',{'status':'PASS_CANONICAL_AA_C_GEOMETRY_OPACITY_EXACT_ONLY_RGB_DIFFERENT','assets':records,
 'tuple_comparison':checks,'changed_child_RGB_rows':int(colours.any(1).sum()),'new_training_updates':0,
 'C_role':'Read-only baked C material, no refit; query does not read original target RGB or target masks.',
 'main_baseline':'Latest SemanticActor5V44Integration; AA is only a geometry-support ablation, never the main baseline.'})
source128=B/'scene008_road_seam_contact_20261107_r1/red/raw128_contact_audit_v2.json'
source130=B/'scene008_road_r0_v44_20261107_r1/review/asset_ground_conflict/canonical_chart_35ray_first_hit_probe.json'
a=json.loads(source128.read_text());b=json.loads(source130.read_text())
rays128=[{'pixel_xy':r['pixel_xy'],'ordering':{1:'chart_first',2:'ground_first',3:'near_contact_005m',0:'uncertain'}[r['ordering_matte_001']],
          'legacy_V44_assetW':r['baseline_assetW']} for r in a['evidence_pixels'] if r['baseline_assetW']>.9]
rays130=[{'pixel_xy':r['pixel_xy'],'ordering':r['matte_thresholds']['0.01']['ordering'],'legacy_V44_assetW':r['baseline_assetW']} for r in b['pixels']]
assert len(rays128)==36 and len(rays130)==35
write('fixed_wheel_rays_v1.json',{'status':'FROZEN_FROM_ORIGINAL_ROAD_REVIEW_BEFORE_NEW_CANDIDATE_GPU','source_files':{str(p):sha(p) for p in [source128,source130]},
 'rays':{'128':rays128,'130':rays130},'total_rays':71,'official_pilot_raw128_rays':36,'separate_diagnostic_raw130_rays':35,
 'metric':'assetW=group_alpha_T[asset=0,y,x]; candidate-minus-latestSEM must be reported, then C-minus-AA separately.',
 'reference_role':'Same-scene inferred C2/ground ordering, not independent hidden wheel ground truth. R0 file supplies frozen diagnostic labels only, never the runtime baseline.',
 'selection_not_conditioned_on_new_predictions':True,'model_updates':0})
contract=ROOT/'contract/independent_evaluation_contract_v1.json';assert sha(contract)=='4560bb91f4cc3b27b89693acd0918f939d2ac07e9bcd7cbbc6e0574e885516cc'
print(json.dumps({'asset_pair_audit':str(ROOT/'contract/road_AA_C_asset_pair_audit.json'),'wheel_registry':str(ROOT/'contract/fixed_wheel_rays_v1.json'),
 'AA_asset_sha256':records[0]['asset_sha256'],'C_asset_sha256':records[1]['asset_sha256'],'changed_RGB_rows':int(colours.any(1).sum()),
 'wheel_rays':71,'independent_contract_sha256':sha(contract)}))
