"""Bind the completed lifecycle evidence to a fresh formal400 launch."""
from pathlib import Path
import hashlib,json

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_vehicle_joint_dggt_20261123_r8')


def read(p):return json.loads(p.read_text())
def pin(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())


cfg=read(ROOT/'joint_rgb_contract_v1.json');smoke=ROOT/'joint_rgb_smoke6_v2'
smoke_cfg=read(ROOT/'joint_rgb_smoke_contract_v2.json')
assert all(smoke_cfg[k]==v for k,v in cfg.items() if k!='schema'),'Formal method changed'
files={
    'smoke_resources':smoke/'resources.json',
    'smoke_fit':smoke/'receipt.json',
    'exact_restore':smoke/'checkpoint_restore/state_comparison.json',
    'bounded_replay':smoke/'cuda_replay/state_comparison.json',
    'native_fresh_reload':smoke/'fresh_reload/receipt.json',
    'independent_CPU_assets':ROOT/'joint_persistent_asset_audit_v1/smoke6_v2_CPU_assets.json',
    'independent_native_float_readback':ROOT/'joint_persistent_asset_audit_v1/smoke6_v2_native_float_readback.json'
}
for role,p in files.items():
    r=read(p);assert r['complete'] is True,(role,r)
    assert r.get('decision','PASS')=='PASS' and not r.get('failed_checks'),role
assert read(files['smoke_resources'])['nativeF']==44 and read(files['smoke_resources'])['nativeB']==14
assert read(files['native_fresh_reload'])['nativeF']==15
assert read(files['exact_restore'])['strict_restore'] is True
assert all(x['expected_bytes_sha256']==x['replay_bytes_sha256'] for x in read(files['exact_restore'])['leaves'])
assert cfg['formal_steps']==400 and cfg['train_frames']==[86,110]
out=ROOT/'formal400_launch_approval.json';assert not out.exists()
value=dict(schema='R8.root_formal400_launch_approval.v1',complete=True,decision='PASS',
    contract=pin(ROOT/'joint_rgb_contract_v1.json'),source_sha256=cfg['source']['sha256'],
    evidence={role:pin(p) for role,p in files.items()},
    method_code={name:pin(ROOT/'code'/name) for name in ['fit_joint_rgb.py','joint_rgb_runtime.py','joint_identity_adapter.py','joint_native_adapter.py','joint_rgb_common.py']},
    source_and_all_geometry_fixed=True,new_diffusion_calls=0,
    formal400_fresh_initialization=True,smoke_checkpoint_not_used_for_formal=True,
    smoke_v1_strict_next_update_failure_preserved=True,
    root_review='Checkpoint restore bytes and counters exact. Root read complete leaf comparator and accepted its registered float replay bound; independent CPU leaf audit also continues before final delivery.',
    whole_scene_restoration_accepted=False,GPU_stays_on=True)
out.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(pin(out)))
