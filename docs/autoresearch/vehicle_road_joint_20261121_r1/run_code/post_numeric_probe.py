from pathlib import Path
import json,numpy as np,torch
import run_joint as m
import lower_prior
original=m.load(m.ROOT/'initial_vehicle_prior.npz')
e=m.Experiment();lower_prior.install(e,m)
same={k:bool(np.array_equal(original[k],e.initial[k])) for k in original}
if not all(same.values()):raise RuntimeError('fixed initializer/VJP replay changed original rows '+str(same))
opt=e.optimizer('vehicle');s=torch.load(m.ROOT/'smoke/checkpoint.pt',map_location='cpu',weights_only=False)
states=[];images=[];losses=[];restores=[]
for repeat in range(2):
    e.restore(s,opt);diff=m.differences(s,e.snapshot('vehicle',4,opt,'smoke'));restores.append(diff)
    if diff:raise RuntimeError('full state mismatch '+str(diff))
    r=e.step('vehicle',4,opt,'independent_numeric_probe');states.append(m.cpu(e.vehicle.state_dict()));losses.append(r['loss'])
    with torch.no_grad():images.append(m.cpu(e.render(134,'delete')['rgb']).numpy())
delta={k:float((states[0][k]-states[1][k]).abs().max()) for k in states[0]};image_diff=float(np.abs(images[0]-images[1]).max())
proof=dict(passed=max(delta.values())<2e-5 and image_diff<1/255 and abs(losses[0]-losses[1])<1e-6,saved_state_restore_bit_exact=not any(restores),repeated_same_checkpoint_independent_steps=2,next_step_parameter_difference=delta,next_step_loss_difference=abs(losses[0]-losses[1]),next_step_render_max_difference=image_diff,one_display_gray_code=1/255,initialization_reproduced_bit_exact=same,extra_optimizer_updates=2,formal_updates=0,interpretation='checkpoint state exact; subsequent FP32 native CUDA atomic backward steps numerically approximate')
m.write(m.ROOT/'post_numeric_probe.json',proof)
v=torch.load(m.ROOT/'checkpoints/vehicle_step00200.pt',map_location='cpu',weights_only=False)['vehicle'];j=torch.load(m.ROOT/'checkpoints/joint_step00200.pt',map_location='cpu',weights_only=False)['vehicle'];init=e.original_state;visible=np.asarray(m.load(m.ROOT/'visible_support.npz')['visible_optimizer_rows'],bool);hidden=~visible
frozen=dict(reproduced_initial_vehicle_bit_exact=all(same.values()),hidden_rows=int(hidden.sum()),visible_rows=int(visible.sum()),hidden_opacity_logit_max_change=float((v['opacity_logits']-init['opacity_logits'])[hidden].abs().max()),hidden_scale_log_delta_max_change=float(v['log_scale_delta'][hidden].abs().max()),joint_opacity_logits_equal_vehicle_end=bool(torch.equal(v['opacity_logits'],j['opacity_logits'])),joint_scale_equal_vehicle_end=bool(torch.equal(v['log_scale_delta'],j['log_scale_delta'])),canonical_xyz_equal=bool(torch.equal(init['base_xyz'],j['base_xyz'])),canonical_rotation_equal=bool(torch.equal(init['base_rotation'],j['base_rotation'])))
parent=m.load(m.ROAD/'road.npz');asset=m.load(m.ROOT/'assets/joint_fit/road.npz');allowed=e.road_train_ids;keep=np.ones(len(parent['active']),bool);keep[allowed]=False
frozen['road_noncolor_fields_equal']={k:bool(np.array_equal(parent[k],asset[k])) for k in parent if k!='reference_srgb'};frozen['road_RGB_outside_1191_equal']=bool(np.array_equal(parent['reference_srgb'][keep],asset['reference_srgb'][keep]))
frozen['passed']=all(frozen['road_noncolor_fields_equal'].values()) and frozen['road_RGB_outside_1191_equal'] and frozen['hidden_opacity_logit_max_change']==0 and frozen['hidden_scale_log_delta_max_change']==0 and frozen['joint_opacity_logits_equal_vehicle_end'] and frozen['joint_scale_equal_vehicle_end'] and frozen['canonical_xyz_equal']
m.write(m.ROOT/'asset_cpu_freeze_proof.json',frozen)
ledger=m.read(m.ROOT/'resource_ledger.json');ledger.update(formal_vehicle_updates=200,formal_joint_updates=200,failed_smoke_optimizer_updates=6,successful_smoke_base_updates=7,successful_smoke_replay_updates=2,successful_smoke_total_updates=9,independent_numeric_probe_updates=2,total_optimizer_updates_all_attempts=417,formal_method_budget_unchanged=True,failed_smoke_path=str(m.ROOT/'smoke_failed_r1'))
for k in ['smoke_updates','extra_replay_updates']:ledger.pop(k,None)
m.write(m.ROOT/'resource_ledger.json',ledger)
print(json.dumps(dict(numeric=proof,frozen=frozen),indent=2))
if not proof['passed'] or not frozen['passed']:raise RuntimeError('numeric/freeze post acceptance failed')
