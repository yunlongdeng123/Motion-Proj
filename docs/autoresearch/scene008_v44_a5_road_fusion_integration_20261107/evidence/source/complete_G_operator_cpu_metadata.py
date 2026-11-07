"""Close raw-versus-GT-residual operator metadata; no native/GPU calls."""
from pathlib import Path
import hashlib,json,os,tarfile,time
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
B=Path('/root/autodl-tmp/drivingeditor_repro/campaign');R=B/'scene008_v44_a5_road_fusion_integration_20261107_r1';O=R/'G4_joint_seam_edit_operator';WORLD=B/'scene008_road_fusion_bake_20261107_r1/red_review/world_masks49_v1'
import sys
sys.path.insert(0,str(R/'source'))
from registered_seam_pairs import seam_pairs
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest_path=O/'operator_manifest.json';m=json.loads(manifest_path.read_text());receipt=json.loads((O/'receipt.json').read_text())
assert receipt['actual_native_forwards']==receipt['actual_dummy_backwards']==22
gts=[];residual0=[];residualtarget=[]
targetroot=B/'scene008_road_fusion_bake_20261107_r1/targets_edgeblend_v1'
for row in m['seam_records']:
 raw=row['raw'];state=row['state'];pairs=seam_pairs(raw,WORLD)
 selected=[p for p in pairs if all(p[k]==row[k] for k in ['kind','edge','tangent_bin'])];assert len(selected)==1
 pair=selected[0];left=pair['minus_mask'];right=pair['plus_mask']
 with np.load(B/f'scene008_instances_20261105_r1/observations/frames/{raw:03d}.npz',allow_pickle=False) as z:gt=z['rgb'].astype(np.float32)
 if gt.shape==(320,480,3):gt=gt.transpose(2,0,1)
 assert gt.shape==(3,320,480)
 with np.load(R/f'baseline4/frames/{raw:03d}/{state}_float.npz',allow_pickle=False) as z:rgb=z['rgb'].copy()
 folder='projected_BUILD40' if raw in [94,128] else 'projected_HELD9'
 with np.load(targetroot/folder/f'raw{raw:03d}.npz',allow_pickle=False) as z:target=z['edgeblend_rgb'].transpose(2,0,1).copy()
 gts.append(gt[:,right].mean(1,dtype=np.float64)-gt[:,left].mean(1,dtype=np.float64))
 residual=rgb-gt;residual0.append(residual[:,right].mean(1,dtype=np.float64)-residual[:,left].mean(1,dtype=np.float64))
 residual=target-gt;residualtarget.append(residual[:,right].mean(1,dtype=np.float64)-residual[:,left].mean(1,dtype=np.float64))
out=O/'signed_residual_J_supplement.npz';assert not out.exists()
np.savez_compressed(out,GT_signed_J=np.stack(gts),baseline_signed_J=np.stack(residual0),source_C_target_signed_J=np.stack(residualtarget))
supplement={'schema':'JOINT_OPERATOR_RAW_RGB_AND_GT_RESIDUAL_SEMANTICS_V1','operator_manifest_sha256':sha(manifest_path),'operator_file_sha256':m['operator_sha256'],
 'original_operator_keys':{'baseline_J':'Float64 side means of baseline RGB signedjump; GT not subtracted','target_J':'Float64 side means of source C target RGB signedjump; GT not subtracted','signed_g':'Signed alphaT mean-plus minus mean-minus; same derivative for raw RGB and GT residual'},
 'residual_supplement':out.name,'residual_supplement_sha256':sha(out),'residual_definition':'RGB-GT in float32, then per-side float64 mean; operator derivative unchanged',
 'no_solver_or_material_fitting_executed':True,'synthetic8F_status':'CANCELLED_BEFORE_EXECUTION_BY_PARENT_SCOPE_CHANGE_TO_GEOMETRY','actual_native_forwards':22,'actual_dummy_backwards':22,'new_GPU_calls_in_this_supplement':0,'optimizer_updates':0}
(O/'signed_residual_J_supplement.json').write_text(json.dumps(supplement,indent=2)+'\n')
ledger={'schema':'ROAD_G_OPERATOR_ACTUAL_COMPUTE_APPENDIX_V1','previous_ledger':'road_execution_ledger_v1.json','prior_native_forwards':1747,'prior_backwards':1656,
 'G_actual_native_forwards':22,'G_actual_dummy_backwards':22,'new_native_forwards_total':1769,'new_native_or_dummy_backwards_total':1678,
 'formal_colour_residual_updates_total':800,'engineering_colour_residual_updates_total':16,'original_Gaussian_SH_geometry_opacity_identity_updates':0,
 'synthetic8F':'Not executed; explicitly cancelled','G_solver_render':'Not executed','all49':'Not executed','GPU_power':'KEEP_ON'}
(O/'execution_ledger_appendix.json').write_text(json.dumps(ledger,indent=2)+'\n')
files=[p for p in O.rglob('*') if p.is_file()]+[R/'source/diagnose_joint_seam_edit_G4.py',R/'source/registered_seam_pairs.py',R/'source/complete_G_operator_cpu_metadata.py',R/'contract/joint_seam_edit_safe_feasibility_v1.json',R/'contract/G4_operator_launch.json',R/'G4_joint_seam_edit_operator.log',R/'review/saved_raw130_component_takeover_cpu.json']
entries=[{'path':str(p.relative_to(R)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]
D=R/'delivery';fm=D/'G_operator_only_metadata_manifest.json';assert not fm.exists();fm.write_text(json.dumps({'files':entries,'actual_compute':ledger},indent=2)+'\n')
archive=D/'G_operator_only_evidence.tar.gz';assert not archive.exists()
with tarfile.open(archive,'w:gz') as t:
 for p in files:t.add(p,arcname='evidence/'+str(p.relative_to(R)),recursive=False)
 t.add(fm,arcname='evidence/G_operator_only_metadata_manifest.json',recursive=False)
record={'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'files':len(files)+1,'native_forwards':22,'dummy_backwards':22,'optimizer_updates':0}
(D/'G_operator_only_archive_receipt.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
