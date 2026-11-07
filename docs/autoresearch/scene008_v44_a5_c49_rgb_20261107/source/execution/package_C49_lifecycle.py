from pathlib import Path
import hashlib,json,subprocess,tarfile,time
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_v44_a5_c49_rgb_20261107_r1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads((R/p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
smoke=read('smoke600s_r1/returned.json');gate=read('smoke_freshrestore_r1/returned.json');qual=read('parent_visibility49/receipt.json')
assert smoke['actual_training_seconds_before_disposable']>=600 and smoke['smoke_updates']==1992 and smoke['actual_optimizer_updates']==1994 and gate['actual_optimizer_updates']==2 and qual['eligible_parents']==2119
D=R/'delivery';D.mkdir(exist_ok=True);tag='C49_lifecycle_bootstrap_light_v1';files={}
def take(p,arc=None):
 p=Path(p)
 if p.is_file():files[arc or 'evidence/'+str(p.relative_to(R))]=p
for folder in ['parent_visibility49','C_warm0_pilot','smoke600s_r1','smoke_freshrestore_r1','review']:
 for p in (R/folder).rglob('*'):
  if p.is_file() and p.suffix=='.json':take(p)
for p in (R/'contract').glob('*.json'):take(p,'contract/'+p.name)
for p in (R/'source').glob('*.py'):take(p,'source/execution/'+p.name)
for p in (R/'parent_visibility49').rglob('*.npz'):take(p)
for p in (R/'smoke600s_r1').glob('*.pth'):take(p)
for p in (R/'smoke_freshrestore_r1').glob('*.pth'):take(p)
for root in ['C_warm0_pilot','smoke600s_r1/evaluation']:
 for p in (R/root/'frames').rglob('*.png'):take(p)
for p in (R/'smoke600s_r1').rglob('parent_RGB.npz'):take(p)
for p in R.glob('*.log'):
 if p.name in ['parent_visibility49.log','smoke600s_r1.log','smoke_freshrestore_r1.log']:take(p,'evidence/logs/'+p.name)
processes=subprocess.run(['ps','-eo','pid,ppid,etimes,args'],capture_output=True,text=True,check=True).stdout
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
health={'status':'C49_REAL600S_LIFECYCLE_PASSED_FRESH_FORMAL_ACTIVE','created_unix':time.time(),'campaign_processes':[x for x in processes.splitlines() if str(R) in x and 'package_C49_lifecycle.py' not in x],
 'GPU_compute_inventory':gpu.strip(),'automatic_controller_stop_after_formal':2000,'further5K10K_requires_new_parent_release':True,'original_scene_and_geometry_updates':0,'no_process_was_stopped':True,'GPU_power':'KEEP_ON'}
hp=D/'C49_lifecycle_health_v1.json';write(hp,health);take(hp,'evidence/C49_lifecycle_health_v1.json')
ledger={'schema':'C49_BACKGROUND_CONTROL_LIFECYCLE_ACTUAL_LEDGER_V1','scope':'Only this C49 RGB-only campaign; prior road/Full5/H branches remain separate',
 'qualification':{'native_forwards':98,'dummy_backwards':98,'optimizer_updates':0},'smoke':{'native_forwards':4012,'backwards':3988,'optimizer_updates':1994,'smoke_training_seconds':smoke['actual_training_seconds_before_disposable'],'training_updates':1992,'disposable_next2':2},
 'fresh_restore_gate':{'native_forwards':6,'backwards':4,'optimizer_updates':2,'public_parity_forwards':2,'next2_updates':2},
 'engineering_totals_before_formal':{'native_forwards':4116,'backwards':4090,'new_C_parent_RGB_updates':1996},'formal_updates_in_this_bootstrap_packet':0,'formal_scope':'Running independently freshCwarm/Adam0, actual formal receipts to be appended at nodes; no prebooking',
 'initial_public_Cwarm0_forwards':8,'early49_public_forwards':8,'final_smoke_public_forwards':8,'public_forwards_already_in_smoke_total':True,
 'warm0_qualification_pilot8_reused_from_qualification_no_extra_forward':True,'eligible_parents':2119,'frozen_zero_mass_parents':7979,'parent_RGB_scalars':6357,'full_parent_rows':10098,'full_child_rows':40392,
 'initializer':'existing source C400 warm material, asset3afc6e25, not zero-trained material','all49_current_BUILD':True,'original_Gaussian_SH_geometry_opacity_identity_updates':0,'quality_accepted':False,'GPU_power':'KEEP_ON'}
lp=D/'C49_lifecycle_execution_ledger_v1.json';write(lp,ledger);take(lp,'evidence/C49_lifecycle_execution_ledger_v1.json')
entries=[{'path':arc,'remote_path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for arc,p in sorted(files.items())]
mf=D/(tag+'_manifest.json');assert not mf.exists();write(mf,{'files':entries,'actual_compute':ledger,'large_21K_models_and_frame_float_arrays_retained_remote':True})
archive=D/(tag+'.tar.gz');assert not archive.exists()
with tarfile.open(archive,'w:gz') as t:
 for arc,p in sorted(files.items()):t.add(p,arcname=arc,recursive=False)
 t.add(mf,arcname='evidence/'+mf.name,recursive=False)
for row in entries:assert sha(row['remote_path'])==row['sha256'],row['path']
record={'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'files':len(files)+1,'engineering_new_C_RGB_updates':1996,'formal_updates_prebooked':0};write(D/(tag+'_archive_receipt.json'),record);print(json.dumps(record),flush=True)
