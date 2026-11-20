"""根代理核对真实 smoke 后创建一次性正式执行合同，不启动 CUDA。"""
from pathlib import Path
import hashlib,json,subprocess

B=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1/gpu_20261119/native_capacity_20261120_r1')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

def main():
    out=B/'formal_r1';launch=B/'launch_r1'
    if out.exists()or launch.exists():raise FileExistsError('preserve old launch/run')
    smoke=B/'smoke_r1';prefix=read(smoke/'receipt.json');readonly=read(smoke/'readonly_completion_r2/receipt.json')
    audit=read(smoke/'RGB_replay_audit_r1/receipt.json');ideal=read(smoke/'I_completion_r1/receipt.json')
    chain=read(smoke/'phase_chain.json')
    assert prefix['all5view_K_eligible_mass_zero'] and prefix['all_smoke_parameters_discard']
    assert all(r['RGB_max']==r['alpha_max']==r['RGB8_max']==0 for r in prefix['public1920_reload'])
    assert prefix['lifecycle']['shared_B_restore_max']==prefix['lifecycle']['shared_fresh_C_restore_max']==prefix['lifecycle']['oracle_restore_max']==0
    assert audit['restore_first_exact'] and audit['restore_second_exact'] and audit['RNG_exact'] and audit['groups_exact'] and audit['cursor_exact'] and audit['frozen_buffers_exact']
    assert audit['ci_RGB']['max_abs']<=1e-6 and all(r['RGB']['max_abs']<=1e-6 and r['alpha_exact']for r in audit['native_road'].values())
    assert ideal['complete'] and ideal['I_full_restore_max']==0 and ideal['I_logical_cursor']==100 and ideal['all_parameters_discard_for_formal']
    assert all(r['RGB_max']==r['alpha_max']==0 for r in ideal['public1920_reload']+ideal['joint_K'])
    assert readonly['complete'] and readonly['unique_checkpoint_count']==15
    assert all(r['field_params_exact']and r['readonly_RGB_exact']and r['alpha_exact']and r['objective_relative_difference']<=1e-6 for r in readonly['checkpoint_results'])
    assert len(readonly['public1920_reload'])==10 and len(readonly['joint_K'])==10
    assert all(r['RGB_max']==r['alpha_max']==r['RGB8_max']==0 for r in readonly['public1920_reload'])
    assert all(r['RGB_max']==r['alpha_max']==0 for r in readonly['joint_K'])
    phases=[]
    for phase in chain['phases']:
        path=Path(phase['path']);r=read(path);key=phase['elapsed_timekey'];seconds=float(r[key])
        assert seconds>0 and sha(path)==phase['SHA256']and abs(seconds-phase['elapsed_seconds'])<1e-6
        phases.append(dict(path=str(path),sha256=sha(path),elapsed_seconds_key=key,phase_role=phase['name']))
    wrapper=dict(schema='root_verified_native_smoke_qualification.v1',completed=True,
        elapsed_seconds=sum(float(read(p['path'])[p['elapsed_seconds_key']])for p in phases),phases=phases,
        checks=dict(shared_restore=True,RGB600_boundary=True,idealI_restore=True,public1920_fiveview=True,K=True),
        raw_time_gate_failures_preserved=True,raw_mixed_unit_assertion_preserved=True,formal_updates=0,formal_warm_start=False,
        fiveview_readonly_receipt=str(smoke/'readonly_completion_r2/receipt.json'),
        scope='actual native lifecycle qualification only; does not imply road image quality PASS',GPU_keep_on=True)
    # A deliberately pessimistic 3x observed update time plus5min IO remains below1hour.
    update_seconds=prefix['measured_computational_seconds']/prefix['physical_ledger']['engineering_accepted_updates']
    estimate=7400*update_seconds*3+300
    assert estimate<3600
    wrapper['queue_forecast']=dict(maximum_registered_updates=7400,observed_seconds_per_update=update_seconds,
        slowdown_factor=3,extra_IO_seconds=300,upper_estimate_seconds=estimate,
        reason='registered below-one-hour lifecycle branch; do not fill time with unrelated/repeated work')
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'another GPU process remains'
    launch.mkdir()
    qualification=launch/'smoke_qualification.json';write(qualification,wrapper)
    owner='native_capacity_formal_r1'
    auth=dict(GPU_authorized=True,formal_authorized=True,output=str(out),protocol_sha256=sha(B/'protocol.json'),
        smoke_receipt=str(qualification),smoke_receipt_sha256=sha(qualification),formal_warm_start=False,
        engineering_rule='measured_queue_below_one_hour_complete_lifecycle',queue_upper_estimate_seconds=estimate,
        throughput_receipt=str(smoke/'receipt.json'),throughput_receipt_sha256=sha(smoke/'receipt.json'),GPU_keep_on=True)
    write(launch/'authorization.json',auth)
    lease=dict(gpu_authorized=True,active=True,owner=owner,authorization=str(launch/'authorization.json'),phase='registered_formal',formal_allowed=True)
    write(launch/'gpu_lease.json',lease)
    print(json.dumps(dict(ready=True,output=str(out),owner=owner,seconds=wrapper['elapsed_seconds'],queue_forecast_seconds=estimate,
        authorization=str(launch/'authorization.json'),lease=str(launch/'gpu_lease.json')),ensure_ascii=False))

if __name__=='__main__':main()
