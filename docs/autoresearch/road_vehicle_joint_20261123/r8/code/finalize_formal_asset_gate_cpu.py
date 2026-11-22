"""Small formal persistence gate, retaining smoke and image-quality decisions."""
import argparse,copy
from pathlib import Path
from verify_joint_asset_cpu import read,write,pin,SOURCE_SHA

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
folder=a.root/'joint_persistent_asset_audit_v1';run=a.root/'joint_rgb_formal400_v1'
asset_path=folder/'formal400_v1_CPU_assets.json';native_path=folder/'formal400_v1_native_all_float_readback.json';smoke_path=folder/'smoke6_v2_joint_lifecycle_gate.json'
asset=read(asset_path);native=read(native_path);smoke=read(smoke_path);receipt=read(run/'receipt.json');resources=read(run/'resources.json')
original=read(a.root/'joint_rgb_smoke6_v1/fit_contract.json');formal=read(run/'fit_contract.json')
for cfg in [original,formal]:cfg.pop('schema',None);cfg.pop('replay_validation',None)
checks={'formal_method_configuration_exact':formal==original,
  'formal_complete':receipt['complete'] is True and resources['complete'] is True,
  'formal400_fresh':receipt['mode']=='fit' and receipt['logical_steps']==receipt['actual_optimizer_updates']==400 and receipt['formal_run_fresh_initialization'] is True and receipt['restore_replay'] is None,
  'source_field_exact':receipt['source_sha256']==SOURCE_SHA,
  'asset_independent_PASS':asset['complete'] and asset['decision']=='PASS' and not asset['failed_checks'],
  'fresh_native_all_float_independent_PASS':native['complete'] and native['decision']=='PASS' and not native['failed_checks'],
  'smoke_lifecycle_independent_PASS':smoke['complete'] and smoke['decision']=='PASS',
  'no_new_diffusion':receipt['new_diffusion_calls']==resources['new_diffusion_calls']==0}
result={'schema':'R9.formal400_independent_joint_persistent_asset_gate.v1','complete':True,
  'decision':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'source_field_sha256':SOURCE_SHA,
  'formal_run_receipt':pin(run/'receipt.json'),'formal_resources':pin(run/'resources.json'),
  'native_fresh_run_receipt':native['native_execution_receipt'],'frozen_invariants':pin(folder/'frozen_invariants.json'),
  'independent_asset_check':pin(asset_path),'independent_all_float_native_check':pin(native_path),'smoke_lifecycle_gate':pin(smoke_path),
  'formal_model_assets':[asset['road_asset'],*asset['actor_assets']],
  'scope':'Persistent colour-only asset invariants and fresh public-loader parity; final scene visual quality is evaluated separately.',
  'whole_scene_restoration_accepted_by_this_gate':False,'auditor_added_GPU_forward':0,'auditor_added_GPU_backward':0,'diffusion_calls':0,'updates':0}
write(a.output,result);print({'decision':result['decision'],'checks':checks,'output':pin(a.output)})
