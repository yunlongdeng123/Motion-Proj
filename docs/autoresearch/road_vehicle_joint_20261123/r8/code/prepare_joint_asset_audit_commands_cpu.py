"""Prepare exact CPU/native-check argv; never execute any child command."""
import argparse,json
from pathlib import Path
from verify_joint_asset_cpu import pin,write

ap=argparse.ArgumentParser();ap.add_argument('--campaign-root',type=Path,default=Path('/root/autodl-tmp/drivingeditor_repro/campaign'))
ap.add_argument('--fit-run',type=Path,required=True);ap.add_argument('--native-verify-output',type=Path,required=True)
ap.add_argument('--fit-contract',type=Path);ap.add_argument('--frozen',type=Path);ap.add_argument('--output',type=Path,required=True)
a=ap.parse_args();root=a.campaign_root/'road_vehicle_joint_dggt_20261123_r8';python='/root/autodl-tmp/envs/motionproj/bin/python'
frozen=a.frozen or root/'joint_persistent_asset_audit_v1/frozen_invariants.json';fit_contract=a.fit_contract or root/'joint_rgb_contract_v1.json'
asset=a.fit_run/'asset';out=a.fit_run/'independent_persistent_audit';verifier=root/'code/verify_joint_asset_cpu.py';fitter=root/'code/fit_joint_rgb.py'
common=[python,str(verifier)]
commands=[
 {'name':'fresh_CPU_persisted_asset_check','GPU':False,'prerequisites':['asset actually exported'],
  'argv':common+['check','--frozen',str(frozen),'--asset',str(asset),'--output',str(out/'CPU_assets.json')]},
 {'name':'root_scheduled_fresh_native_public_reload','GPU':True,'owner':'root GPU scheduler only',
  'prerequisites':['CPU asset check PASS','fit expected final_readout saved','root permits native verification'],
  'argv':[python,str(fitter),'--mode','verify','--execute','--contract',str(fit_contract),'--asset',str(asset),
     '--expected',str(a.fit_run/'final_readout'),'--output',str(a.native_verify_output)]},
 {'name':'fresh_CPU_native_float_file_parity','GPU':False,
  'prerequisites':['native fresh verify actually complete with preserved float expected/reloaded and camera audit'],
  'argv':common+['render-check','--frozen',str(frozen),'--asset',str(asset),
    '--native-expected',str(a.native_verify_output/'expected'),'--native-reloaded',str(a.native_verify_output/'reloaded'),
    '--native-run-receipt',str(a.native_verify_output/'receipt.json'),'--native-camera-audit',str(a.native_verify_output/'camera_timestamp_audit.json'),
    '--output',str(out/'native_float_readback.json')]}
]
result={'schema':'R9.joint_asset_fresh_process_audit_command_plan.v1','complete':True,'prepared_only':True,
  'any_command_executed_by_this_preparer':False,'fit_run':str(a.fit_run),'frozen_invariants':pin(frozen),
  'checker_source':pin(verifier),'native_fitter_source_at_preparation':pin(fitter),'fit_contract':pin(fit_contract),
  'commands_in_order':commands,'GPU_forward_performed':0,'updates':0,'diffusion_calls':0}
write(a.output,result);print(json.dumps({'prepared_only':True,'output':pin(a.output)}))
