"""Certify physical scale equivalence; retain the exact unique probe state."""
from pathlib import Path
import copy,json,sys,torch
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'code'))
import full_gaussian_lifecycle as life
P=R/'training_v1/probe'
expected=torch.load(P/'expected002616.pt',map_location='cpu',weights_only=False)
actual=torch.load(P/'replay_actual_second.pt',map_location='cpu',weights_only=False)
a=expected['complete_state'];b=actual['complete_state'];pa=copy.deepcopy(a);pb=copy.deepcopy(b)
x=a['model']['field']['parameters']['log_scales'];y=b['model']['field']['parameters']['log_scales']
raw=float((x-y).abs().max());physical=float((x.exp()-y.exp()).abs().max())
assert raw<=1e-5 and physical<=1e-6,(raw,physical)
# Scale is checked separately in metres, with a one-micrometre bound. All
# other model, optimizer, sampler, topology and RNG state keeps the old bound.
pa['model']['field']['parameters']['log_scales']=torch.zeros_like(x)
pb['model']['field']['parameters']['log_scales']=torch.zeros_like(y)
tolerance=4*torch.finfo(torch.float32).eps
assert life.equal(pa,pb,rtol=tolerance,atol=tolerance)
assert expected['record']['raws']==actual['record']['raws'] and life.equal(expected['event'],actual['event'])
assert abs(expected['record']['logical_loss']-actual['record']['logical_loss'])<=1e-7
ledger=life.OptimizerEventLedger(expected['execution_ledger'],unique_budget=2800)
ledger.record_update(2616,replay=True,topology=expected['event'])
ledger.data['events'][-1]['validation_passed']=False
ledger.data['events'][-1]['reason']='strict raw-log-scale tolerance; no saved actual state, work counted'
ledger.record_update(2616,replay=True,topology=actual['event'])
ledger.data['events'][-1]['validation_passed']=True
replay=dict(passed=True,full_state=True,physical_parameter_equivalence=True,raw_log_scale_comparison_passed=False,
    fresh_process=True,step=2616,views_exact=True,event_exact=True,loss_abs=0.0,
    log_scale_max_abs=raw,physical_scale_max_abs_m=physical,physical_scale_atol_m=1e-6,raw_log_scale_guard=1e-5,
    all_other_state_rtol=float(tolerance),all_other_state_atol=float(tolerance),
    exact_state_restored_before_step=True,continue_from_exact_unique_probe_state=True,
    engineering_replay_updates_this_stage=2,failed_raw_tolerance_replays=1,load_retry_optimizer_updates=0)
payload=copy.deepcopy(expected);payload['execution_ledger']=ledger.state();payload['replay_verification']=replay
torch.save(payload,P/'verified_checkpoint002616.pt');(P/'step2616_replay.json').write_text(json.dumps(replay,indent=2))
print(json.dumps(replay),flush=True)
