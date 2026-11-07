"""CPU-only completed-stage evidence; no Gaussian weights or pixel arrays."""
from pathlib import Path
import hashlib,json,tarfile,time

R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_v44_a5_road_fusion_integration_20261107_r1')
REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
TAG='road_execution_through_F400_v1'
D=R/'delivery';D.mkdir(exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(4194304),b''):h.update(block)
 return h.hexdigest()
def read(p):return json.loads((R/p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

phases=[]
def add(name,receipt,forward,backward=0,formal=0,engineering=0,reason=''):
 p=R/receipt;assert p.is_file(),receipt
 value=read(receipt)
 for key,expected in [('actual_native_forwards',forward),('joint_forwards',forward),('actual_native_backwards',backward),('dummy_feature_backwards',backward)]:
  if key in value:assert value[key]==expected,(name,key,value[key],expected)
 if 'actual_optimizer_updates' in value:assert value['actual_optimizer_updates']==formal+engineering
 phases.append(dict(phase=name,receipt=receipt,receipt_sha256=sha(p),actual_native_forwards=forward,
  actual_native_or_dummy_backwards=backward,new_colour_residual_formal_updates=formal,
  new_colour_residual_engineering_updates=engineering,original_Gaussian_SH_geometry_opacity_identity_updates=0,
  status=value.get('status'),note=reason))
def rendered(name,folder,count):
 r=read(folder+'/receipt.json');frames=r['frames']
 actual=sum(len(f['states']) for f in frames)
 assert actual==count and all(set(f['states'])=={'keep','delete'} for f in frames)
 progress=R/folder/'kernel_execution_progress.json'
 if progress.exists():assert json.loads(progress.read_text())['completed_native_forwards']==count
 add(name,folder+'/receipt.json',count)
rendered('latest_semantic_baseline4','baseline4',8)
add('AA_first_forward_assertion_failure','contract/first_forward_engineering_failure.json',1,reason='One actual native call occurred before the diagnostic unused-channel assertion. Preserved; not zero compute.')
rendered('AA4_retry1','AA4_retry1',8);rendered('C4_retry1','C4_retry1',8)
for variant in ['baseline','AA','C']:rendered('tire130_'+variant,'diagnostic130/'+variant,2)
add('D_support_qualification_v2','support_qualification4_v2/receipt.json',8,8)
rendered('D_fixed_support4_v2','D4_v2',8)
add('E_existing_surface_qualification','E4_appearance_qualification/receipt.json',8,8)
add('E_public_zero_residual_identity4','E_zero_identity4/receipt.json',8,reason='Fresh real public-API eight forwards, counted separately from baseline and smoke.')
add('E_real_smoke_and_disposable_next2','E_smoke_r1/returned.json',20,12,engineering=6)
add('E_fresh_restore_next2_and_train_public_parity','E_restore_parity_r1/returned.json',6,4,engineering=2,reason='Two same-delta direct128 K/D parity forwards plus four training forwards; fresh restoration begins bit exact, subsequent numeric differences are retained.')
add('E_fresh_formal400','E_formal400_r1/returned.json',808,800,formal=400)
add('F_bounded_mass_qualification','F4_bounded_appearance_qualification/receipt.json',8,8)
add('F_public_zero_residual_identity4','F_zero_identity4/receipt.json',8)
add('F_real_smoke_and_disposable_next2','F_smoke_r1/returned.json',20,12,engineering=6)
add('F_fresh_restore_next2_and_train_public_parity','F_restore_parity_r1/returned.json',6,4,engineering=2)
add('F_fresh_formal400','F_formal400_r1/returned.json',808,800,formal=400)
totals={key:sum(p[key] for p in phases) for key in ['actual_native_forwards','actual_native_or_dummy_backwards','new_colour_residual_formal_updates','new_colour_residual_engineering_updates','original_Gaussian_SH_geometry_opacity_identity_updates']}
assert totals==dict(actual_native_forwards=1747,actual_native_or_dummy_backwards=1656,new_colour_residual_formal_updates=800,new_colour_residual_engineering_updates=16,original_Gaussian_SH_geometry_opacity_identity_updates=0),totals
ledger=dict(schema='SCENE008_ROAD_INTEGRATION_ACTUAL_EXECUTION_LEDGER_V1',scope='Only this road integration campaign; previous identity/reconstruction/training campaigns have separate ledgers.',
 status='COMPLETED_FIXED_F400_PENDING_OR_RECORDED_QUALITY_REVIEW_NO49',actual_totals=totals,phases=phases,
 not_executed=['D qualification v1 GPU: superseded before execution','D130 extra two forward diagnostic: cancelled before execution','Any road-integration49 render','Any original Gaussian or SH optimization'],
 original_source_Gaussians_optimized=False,all_current_four_views_are_BUILD=True,
 old_HELD105_137_is_historical_role_only=True,source_C_targets_include_observation_anchoring=True,
 future_or_unseen_view_protection_guarantee=False,automatic_quality_adoption=False,automatic49=False,
 GPU_power='KEEP_ON',shutdown=False,reset_card=False,created_unix=time.time())
ledger_path=D/(TAG+'_ledger.json');assert not ledger_path.exists();write(ledger_path,ledger)

files={}
def take(p,arc=None):
 p=Path(p)
 if not p.is_file():return
 arc=arc or 'evidence/'+str(p.relative_to(R)).replace('\\','/')
 assert arc not in files or files[arc]==p,(arc,p)
 files[arc]=p
for p in R.rglob('*.json'):
 relative=p.relative_to(R)
 if any(x in relative.parts for x in ['frames','point_mass','delivery']):continue
 if relative.parts[0]=='contract':take(p,'contract/'+p.name)
 else:take(p)
for root,arc in [(R/'source','source/execution'),(R/'source_snapshots','source/api_snapshots')]:
 if root.exists():
  for p in root.rglob('*'):
   if p.is_file() and p.suffix in ['.py','.json','.txt']:take(p,arc+'/'+str(p.relative_to(root)))
for p in (R/'contract').glob('*.txt'):take(p,'contract/'+p.name)
api_dir=REPO/'scripts/worldsim_v82/drivingeditor_dynamic_edit'
for name in ['v44_integration.py','road_fusion_integration.py','fixed_road_support.py','baseline_road_appearance.py','bounded_road_appearance.py']:
 take(api_dir/name,'source/api_snapshots/current_'+name)
for pattern in ['support_qualification4_v2/safe_road_children.npz','E4_appearance_qualification/safe_baseline_road_rows.npz','F4_bounded_appearance_qualification/bounded_appearance_mass.npz']:
 p=R/pattern;assert p.is_file();take(p)
for folder in ['E_zero_material','E_smoke_r1','E_restore_parity_r1','E_formal400_r1','F_zero_material','F_smoke_r1','F_restore_parity_r1','F_formal400_r1']:
 for p in (R/folder).rglob('*'):
  if p.is_file() and 'frames' not in p.relative_to(R/folder).parts and p.suffix in ['.pth','.npz','.json']:take(p)
for p in R.glob('*.log'):take(p,'evidence/logs/'+p.name)
take(ledger_path,'evidence/road_execution_ledger_v1.json')

retained=[]
for folder in ['support_qualification4_v2','E4_appearance_qualification','F4_bounded_appearance_qualification']:
 for p in (R/folder/'point_mass').glob('*.npz'):
  retained.append(dict(remote_path=str(p),bytes=p.stat().st_size,sha256=sha(p),reason='Raw per-view feature-gradient masses retained remotely; aggregate matrix/sidecar is included.'))
entries=[dict(path=arc,remote_path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,sha256=sha(p)) for arc,p in sorted(files.items())]
manifest=dict(schema='SCENE008_ROAD_EXECUTION_LIGHT_MANIFEST_V1',files=entries,remote_point_mass_files=retained,
 excluded='Original Gaussian weights and full-resolution frame float arrays; PNGs separately delivered with SHA manifests.',
 actual_totals=totals,quality_adoption=False,all49_executed=False)
manifest_path=D/(TAG+'_manifest.json');assert not manifest_path.exists();write(manifest_path,manifest)
files['evidence/road_execution_files_manifest_v1.json']=manifest_path
archive=D/(TAG+'.tar.gz');assert not archive.exists()
with tarfile.open(archive,'w:gz') as tar:
 for arc,p in sorted(files.items()):tar.add(p,arcname=arc,recursive=False)
for entry in entries:
 p=Path(entry['remote_path']);assert p.stat().st_size==entry['bytes'] and p.stat().st_mtime_ns==entry['mtime_ns'] and sha(p)==entry['sha256'],p
receipt=dict(status='PASS_COMPLETED_ROAD_PHASES_LIGHT_ARCHIVE_AND_SOURCE_STABILITY',archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),files=len(files),actual_totals=totals)
write(D/(TAG+'_archive_receipt.json'),receipt);print(json.dumps(receipt),flush=True)
