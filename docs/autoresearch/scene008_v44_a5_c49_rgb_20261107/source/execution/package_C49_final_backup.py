from pathlib import Path
import hashlib,json,subprocess,tarfile,time
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_v44_a5_c49_rgb_20261107_r1');D=R/'delivery';D.mkdir(exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads((R/p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
formal=read('formal2000_r1/returned.json');full=read('control_final49_2000/receipt.json');h500=read('H2_material500_retry1/receipt.json');h2k=read('H2_material2000_retry1/receipt.json')
assert formal['actual_optimizer_updates']==2000 and formal['actual_native_forwards']==4016 and formal['actual_native_backwards']==4000
assert full['actual_native_forwards']==196 and h500['actual_native_forwards']==h2k['actual_native_forwards']==8
decision={'schema':'C49_FORMAL_STOP2000_FINAL_SCOPE_DECISION_V1','decision':'STOP_NEW_C49_MATERIAL_FORMAL_AT2000_NO5K_NO10K','authority':'Root explicit decision after actual epoch-window MSE/geometry review',
 'actual_formal_updates':2000,'planned5000_and10000':'NOT_EXECUTED_NOT_QUEUED','root_MSE_last5_vs_first5_relative_percent':-.0009545,'root_RGB_change_diagnostic':{'mean_abs':.0364,'max_abs':.4198,'is_parameter_convergence_claim':False},
 'geometry_and_original_scene_updates':0,'known_control_quality_accepted':False,'next_world_boundary_handover_GPU':'NOT_AUTHORIZED_BY_THIS_DECISION','GPU_power':'KEEP_ON','shutdown':False}
dp=R/'contract/C49_formal_stop2000_decision.json';assert not dp.exists();write(dp,decision)
processes=subprocess.run(['ps','-eo','pid,ppid,etimes,args'],capture_output=True,text=True,check=True).stdout
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
owned={}
for rel in ['contract/smoke600s_r1_launch.json','contract/lifecycle_queue_launch.json','contract/smoke_freshrestore_r1_launch.json','contract/formal2000_r1_launch.json','contract/control49_queue_launch.json','contract/control_final49_2000_launch.json','contract/H2_material500_retry1_launch.json','contract/H2_material2000_retry1_launch.json']:
 pid=read(rel)['pid'];proc=Path('/proc')/str(pid);cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace') if proc.exists() else None
 assert cmd is None or str(R) not in cmd,(rel,pid,cmd)
 owned[rel]={'pid':pid,'currently_present':proc.exists(),'current_command_if_reused':cmd}
health={'status':'COMPLETE_C49_FORMAL2000_NO_FUTURE_TRAINING_CONTROLLER','unix':time.time(),'known_owned_processes':owned,'GPU_compute_inventory':gpu.strip(),
 'other_campaign_related_process_lines':[x for x in processes.splitlines() if str(R) in x and 'package_C49_final_backup.py' not in x],
 'no_process_or_service_stopped':True,'no_automatic5K_or10K':True,'source_and_target_after_guards_passed':formal['source_and_target_guards_unchanged'],'original_Gaussian_SH_geometry_opacity_identity_updates':0,'GPU_power':'KEEP_ON'}
hp=R/'final_execution_health.json';assert not hp.exists();write(hp,health)
ledger={'schema':'C49_COMPLETE_ACTUAL_LEDGER_V1','scope':'C49 background RGB campaign only; historical E/F/G/H/Full5 diagnostics are separate',
 'qualification':{'F':98,'B':98,'optimizer_updates':0},'smoke':{'F':4012,'B':3988,'optimizer_updates':1994,'training_smoke_seconds':600.2084789276123,'continuous_smoke_updates':1992,'disposable_next2_updates':2},
 'freshrestore_and_public_parity':{'F':6,'B':4,'optimizer_updates':2},'fresh_formal2000':{'F':4016,'B':4000,'optimizer_updates':2000},
 'new_H2_material500':{'F':8,'B':0,'optimizer_updates':0},'new_H2_material2000':{'F':8,'B':0,'optimizer_updates':0},'control49_Cwarm0_vs2000':{'F':196,'B':0,'optimizer_updates':0},
 'total_native_forwards':8344,'total_native_or_dummy_backwards':8090,'new_formal_C_parent_RGB_updates':2000,'new_engineering_C_parent_RGB_updates':1996,'original_scene_Gaussian_SH_geometry_opacity_identity_updates':0,
 'H2_material500_first_CPU_bytecheck_failure':{'F':0,'B':0,'optimizer_updates':0,'reason':'NaN diagnostic metadata must use dtype/shape/bytes equality; original failed source/log retained'},
 'old_H2_113F100B_not_counted_again':True,'C400_warm_initializer_not_new_formal_training':True,'all49_are_BUILD':True,'no_heldout_claim':True,'no5K_or10K_training':True,'no_H49':True,
 'warm0_complete49_float_saved':False,'warm0_4pilot_float_saved':True,'warm0_complete49_PNG_saved':True,'material2000_complete49_float_and_PNG_saved':True,'quality_accepted':False,'GPU_power':'KEEP_ON'}
lp=R/'C49_final_execution_ledger.json';assert not lp.exists();write(lp,ledger)
files={}
def take(p,arc=None):
 p=Path(p)
 if p.is_file():files[arc or 'evidence/'+str(p.relative_to(R))]=p
for p in R.glob('*.json'):take(p)
for p in (R/'contract').glob('*.json'):take(p,'contract/'+p.name)
for p in (R/'source').glob('*.py'):take(p,'source/execution/'+p.name)
for root in ['review','formal2000_r1','H2_material500_retry1','H2_material2000_retry1']:
 for p in (R/root).rglob('*'):
  if p.is_file() and (p.suffix=='.json' or (p.suffix in ['.png','.pth','.jsonl'] and 'point_mass' not in p.parts)):take(p)
for root in ['formal_export500_r1','formal_export2000_r1']:
 for p in (R/root).rglob('*'):
  if p.is_file():take(p,'model/'+str(p.relative_to(R)))
for p in (R/'control_final49_2000').rglob('*.json'):take(p)
for p in R.glob('*.log'):take(p,'evidence/logs/'+p.name)
entries=[{'path':arc,'remote_path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for arc,p in sorted(files.items())]
def pack(stem,mapping):
 mf=D/(stem+'_manifest.json');assert not mf.exists();rows=[{'path':arc,'remote_path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for arc,p in sorted(mapping.items())];write(mf,{'files':rows,'final_actual_ledger':ledger})
 archive=D/(stem+'.tar.gz');assert not archive.exists()
 with tarfile.open(archive,'w:gz',compresslevel=4) as t:
  for arc,p in sorted(mapping.items()):t.add(p,arcname=arc,recursive=False)
  t.add(mf,arcname='evidence/'+mf.name,recursive=False)
 record={'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'files':len(rows)+1};write(D/(stem+'_archive_receipt.json'),record);print(json.dumps(record),flush=True)
 return record
pack('C49_final_nodes_assets_v1',files)
imagefiles={'evidence/'+str(p.relative_to(R)):p for p in (R/'control_final49_2000').rglob('*.png')};assert len(imagefiles)==196;pack('C49_control49_all_PNG_v1',imagefiles)
for part in range(8):
 raw_ids=list(range(89,138))[part*7:(part+1)*7];group={}
 for raw in raw_ids:
  for p in (R/f'control_final49_2000/material2000/frames/{raw:03d}').glob('*_float.npz'):group['evidence/'+str(p.relative_to(R))]=p
 if group:pack(f'C49_control49_float_part{part+1}',group)
