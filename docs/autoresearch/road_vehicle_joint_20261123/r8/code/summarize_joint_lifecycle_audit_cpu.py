"""Small combined independent lifecycle gate; no execution of native actions."""
import argparse
from pathlib import Path
from verify_joint_asset_cpu import read,write,pin,SOURCE_SHA

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prefix',default='smoke6_v2');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
folder=a.root/'joint_persistent_asset_audit_v1'
files={'persisted_asset':folder/f'{a.prefix}_CPU_assets.json','fresh_native_float_parity':folder/f'{a.prefix}_native_float_readback.json',
       'replay':folder/f'{a.prefix}_independent_replay.json'}
data={k:read(v) for k,v in files.items()}
checks={k+'_independent_PASS':x.get('complete') is True and x.get('decision')=='PASS' and not x.get('failed_checks') for k,x in data.items()}
asset=data['persisted_asset'];native=data['fresh_native_float_parity'];replay=data['replay']
checks['all_five_native_frames_read_back']=sorted(x['raw'] for x in native['rows'])==[86,110,134,179,198]
checks['source_field_exact']=asset['source_field']['sha256']==SOURCE_SHA
result={'schema':'R9.independent_joint_persistent_lifecycle_gate.v1','complete':True,
  'decision':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'source_field_sha256':SOURCE_SHA,
  'frozen_invariants':pin(folder/'frozen_invariants.json'),'independent_evidence':{k:pin(v) for k,v in files.items()},
  'strict_restore_leaf_count':replay['strict_restore']['leaf_count'],
  'strict_restore_all_bytes_exact':replay['strict_restore']['all_state_leaf_bytes_exact'],
  'next_step_replay_leaf_count':replay['bounded_next_step_replay']['leaf_count'],
  'next_step_replay_bitexact':replay['bounded_next_step_replay']['all_state_leaf_bytes_exact'],
  'next_step_replay_parameter_max_abs':replay['bounded_next_step_replay']['parameter_max_abs'],
  'next_step_replay_optimizer_state_max_abs':replay['bounded_next_step_replay']['optimizer_state_max_abs'],
  'next_step_replay_maximum_error_over_registered_bound':replay['bounded_next_step_replay']['maximum_error_over_registered_bound'],
  'formal400_method_configuration_unchanged':replay['checks']['formal_and_all_method_configuration_exact'],
  'original_strict_replay_failure_preserved':replay['original_v1_strict_replay_failure'],
  'scope':'Independent persisted-array, measured K-freeze, exact fresh native float parity, and state restore/bounded replay checks; final fitted image quality remains a separate review.',
  'root_GPU_scheduling_is_external':True,'auditor_GPU_forward':0,'auditor_GPU_backward':0,'diffusion_calls':0,'updates':0}
write(a.output,result);print({'decision':result['decision'],'output':pin(a.output)})
