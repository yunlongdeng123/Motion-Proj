"""Reuse the four exact measured point masses and measure only the remaining 45."""
from pathlib import Path
import ast,hashlib,json,os,subprocess,sys,time
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_v44_full5_integration_20261107_r1')
old=ROOT/'source/calibrate_visible_actor5.py';source=old.read_text()
assert hashlib.sha256(old.read_bytes()).hexdigest()=='9570853aa7546863bd38f05f88e943e417e8cf65252c88d61b1bf6f2c809d631'
mask_manifest=ROOT/'inputs/semantic_actor5_masks49_v1/manifest.json'
assert hashlib.sha256(mask_manifest.read_bytes()).hexdigest()=='b92a2d43a85f2317181456cd335d6999fc609dc9661a43f9f831335152564f37'
changes={
 "MASKS=ROOT/'inputs/semantic_actor5_masks4_v1'":"MASKS=ROOT/'inputs/semantic_actor5_masks49_v1'",
 "OUT=ROOT/'semantic_actor5_calibration4'":"OUT=ROOT/'semantic_actor5_calibration49'\nPRIOR=ROOT/'semantic_actor5_calibration4'\nREUSED=[94,105,130,137]",
 'RAWS=[94,105,130,137]':'RAWS=list(range(89,138))',
 "  p=MASKS/f'{raw:03d}.png';assert sha(p)==MASK_SHA[raw]":"  p=MASKS/f'{raw:03d}.png'\n  if raw in REUSED:assert sha(p)==MASK_SHA[raw]",
 '  assert int(mask.sum())=={94:2052,105:10773,130:0,137:0}[raw]':'  if raw in REUSED:assert int(mask.sum())=={94:2052,105:10773,130:0,137:0}[raw]',
 ' source_files=[MODEL/x for x in':
 " assert sha(MASKS/'manifest.json')=='b92a2d43a85f2317181456cd335d6999fc609dc9661a43f9f831335152564f37'\n"
 " assert sum(int(x.sum()) for x in mask_values.values())==121128\n"
 " assert sum(int((np.nonzero(x)[0]<=np.nonzero(x)[0].min()+.25*(np.nonzero(x)[0].max()-np.nonzero(x)[0].min())).sum()) if x.any() else 0 for x in mask_values.values())==25906\n"
 " prior_protocol=json.loads((PRIOR/'calibration_protocol.json').read_text());prior_receipt=json.loads((PRIOR/'receipt.json').read_text())\n"
 " assert prior_receipt['joint_forwards']==4 and prior_receipt['dummy_feature_backwards']==4 and prior_receipt['model_parameters_environment_and_original_identity_unchanged']\n"
 " prior_mass_sources={str(raw):identity(PRIOR/f'point_mass/{raw:03d}.npz') for raw in REUSED}\n"
 ' source_files=[MODEL/x for x in',
 ' sources={str(p):identity(p) for p in source_files}':' sources={str(p):identity(p) for p in source_files}\n assert sources==prior_protocol[\'source_files\']',
 "  'planned_joint_forwards':4,'planned_dummy_feature_backwards':4,'optimization_updates':0,'optimizer_created':False,":
 "  'planned_joint_forwards':45,'planned_dummy_feature_backwards':45,'total_visibility_views':49,'reused_measured_raw_ids':REUSED,'prior_mass_sources':prior_mass_sources,'prior_protocol_sha256':sha(PRIOR/'calibration_protocol.json'),'optimization_updates':0,'optimizer_created':False,",
 " before=fingerprints();write(OUT/'model_tensor_fingerprints_before.json',before)":" before=fingerprints();assert before==json.loads((PRIOR/'model_tensor_fingerprints_before.json').read_text());write(OUT/'model_tensor_fingerprints_before.json',before)",
 ' from utils.sh_utils import eval_sh\n for raw in RAWS:':
 " from utils.sh_utils import eval_sh\n"
 " new_forward_count=0;new_backward_count=0;processed=set()\n"
 " for raw in REUSED:\n"
 "  p=PRIOR/f'point_mass/{raw:03d}.npz'\n"
 "  with np.load(p,allow_pickle=False) as z:\n"
 "   assert int(z['branch_boundary'])==ns;mass=np.column_stack([z['inside_mass'],z['outside_mass']])\n"
 "  assert mass.shape==(n,2) and np.isfinite(mass).all() and mass.min()>=0\n"
 "  accumulated+=mass.astype(np.float64);(OUT/f'point_mass/{raw:03d}.npz').write_bytes(p.read_bytes())\n"
 "  f=dict(next(x for x in prior_receipt['frames'] if x['raw']==raw));f.pop('selected_semantic5',None);f.update(joint_forwards=0,dummy_feature_backwards=0,measurement_reused=True,prior_point_mass_sha256=sha(p))\n"
 "  frames.append(f);cached.append((raw,mass,None));processed.add(raw)\n"
 "  (OUT/f'{raw:03d}_native_keep.png').write_bytes((PRIOR/f'{raw:03d}_native_keep.png').read_bytes())\n"
 " with np.load(PRIOR/'semantic_actor5_rows.npz',allow_pickle=False) as z:\n"
 "  assert np.array_equal(accumulated[:ns,0],z['static_inside_mass']) and np.array_equal(accumulated[ns:,0],z['dynamic_inside_mass'])\n"
 "  assert np.array_equal(accumulated[:ns].sum(1),z['static_total_mass']) and np.array_equal(accumulated[ns:].sum(1),z['dynamic_total_mass'])\n"
 " write(OUT/'reuse_audit.json',{'status':'PASS_EXACT_FOUR_VIEW_MASS_AGGREGATE_AND_MODEL_BINDING','reused_raw_ids':REUSED,'source_mass_files':prior_mass_sources,'new_gpu_calls_for_reused_views':0})\n"
 " for raw in [x for x in RAWS if x not in REUSED]:\n"
 "  assert raw not in processed",
 "  write(OUT/'kernel_execution_progress.json',{'current_raw':raw,'completed_joint_forwards':len(frames)+1,'completed_dummy_backwards':len(frames),'optimization_updates':0})":
 "  new_forward_count+=1;write(OUT/'kernel_execution_progress.json',{'current_raw':raw,'completed_joint_forwards':new_forward_count,'completed_dummy_backwards':new_backward_count,'reused_views':4,'optimization_updates':0})",
 "  loss.backward();write(OUT/'kernel_execution_progress.json',{'current_raw':raw,'completed_joint_forwards':len(frames)+1,'completed_dummy_backwards':len(frames)+1,'optimization_updates':0})":
 "  loss.backward();new_backward_count+=1;write(OUT/'kernel_execution_progress.json',{'current_raw':raw,'completed_joint_forwards':new_forward_count,'completed_dummy_backwards':new_backward_count,'reused_views':4,'optimization_updates':0})",
 "  frames.append(frame);cached.append((raw,mass,colours));write(OUT/'progress.json',{'completed_raw_ids':[x['raw'] for x in frames],'joint_forwards':len(frames),'dummy_feature_backwards':len(frames),'optimization_updates':0})":
 "  frames.append(frame);cached.append((raw,mass,colours));processed.add(raw);write(OUT/'progress.json',{'completed_raw_ids':sorted(processed),'joint_forwards':new_forward_count,'dummy_feature_backwards':new_backward_count,'reused_raw_ids':REUSED,'optimization_updates':0})",
 ' inside=accumulated[:,0];total=accumulated.sum(1);purity=np.divide(inside,total,out=np.zeros(n),where=total>0)':
 " assert sorted(processed)==RAWS and len(frames)==49 and new_forward_count==45 and new_backward_count==45\n frames.sort(key=lambda x:x['raw'])\n inside=accumulated[:,0];total=accumulated.sum(1);purity=np.divide(inside,total,out=np.zeros(n),where=total>0)",
 "'selected_inside_RGB_T_sum':(per_inside[selected,None]*colours[selected]).sum(0,dtype=np.float64).tolist(),":
 "'selected_inside_RGB_T_sum':(per_inside[selected,None]*colours[selected]).sum(0,dtype=np.float64).tolist() if colours is not None else None,",
 "'unselected_inside_RGB_T_sum':(per_inside[~selected,None]*colours[~selected]).sum(0,dtype=np.float64).tolist(),":
 "'unselected_inside_RGB_T_sum':(per_inside[~selected,None]*colours[~selected]).sum(0,dtype=np.float64).tolist() if colours is not None else None,",
 "'RGB_T_note':'Computed from visible point masses and original per-view SH colours; no fitting or pixel composite.'":
 "'RGB_T_note':'Computed from original per-view SH colours when measured now; reused four views have null RGB attribution for the new selection to avoid replay. No fitting or pixel composite.'",
 " for path,old in sources.items():assert identity(path)==old,path":
 " for path,old in sources.items():assert identity(path)==old,path\n for raw,old in prior_mass_sources.items():assert identity(old['path'])==old",
 "'joint_forwards':4,'dummy_feature_backwards':4,":"'joint_forwards':45,'dummy_feature_backwards':45,'reused_raw_ids':REUSED,'total_visibility_views':49,",
 "PASS_VISIBLE_MASS_IDENTITY_CALIBRATION_ENGINEERING_PENDING_ROOT_REVIEW":"PASS_INCREMENTAL_ALL49_VISIBLE_MASS_IDENTITY_CALIBRATION_PENDING_FOUR_VIEW_COMPOSITION",
}
for before,after in changes.items():
 assert source.count(before)==1,(before,source.count(before));source=source.replace(before,after)
ast.parse(source);worker=ROOT/'source/calibrate_visible_actor5_all49.py';assert not worker.exists();worker.write_text(source)
assert not (ROOT/'semantic_actor5_calibration49').exists()
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines()[0]);assert free>=12288
env=os.environ.copy()
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[k]='1'
cmd=[sys.executable,'-B',str(worker)]
with (ROOT/'semantic_actor5_calibration49.log').open('x') as log:p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,env=env,start_new_session=True)
receipt={'status':'STARTED_INCREMENTAL_45_BUILD_VISIBILITY_MEASUREMENTS','pid':p.pid,'command':cmd,'source_sha256':hashlib.sha256(worker.read_bytes()).hexdigest(),'reused_raw_ids':[94,105,130,137],'new_raw_ids':[x for x in range(89,138) if x not in [94,105,130,137]],'new_joint_forwards_planned':45,'new_dummy_backwards_planned':45,'optimization_updates':0,'free_MiB':free,'unix':time.time(),'no_49_candidate_image_release':True,'shutdown':False,'reset_card':False}
(ROOT/'contract/semantic_actor5_calibration49_launch.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
