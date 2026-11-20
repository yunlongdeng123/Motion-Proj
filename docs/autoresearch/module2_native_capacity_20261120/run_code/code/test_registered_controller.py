import json,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
from run_registered_experiments import (validate_launch,canonical_loss_snapshot,oracle_resume_phase,
    oracle_relative_gap,ideal_frame_from_arrays,sha)


def dump(p,x):p.write_text(json.dumps(x));return p


def test_default_inspect_cannot_load_torch_or_native():
    script=Path(__file__).with_name('run_registered_experiments.py')
    source="import runpy,sys,pathlib;sys.path.insert(0,str(pathlib.Path(sys.argv[1]).parent));sys.argv=[sys.argv[1]];runpy.run_path(sys.argv[0],run_name='__main__');assert 'torch'not in sys.modules"
    r=subprocess.run([sys.executable,'-B','-c',source,str(script)],capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    x=json.loads(r.stdout);assert x['execute_default']is False and x['GPU_context_loaded']is False


def test_launch_requires_hash_bound_original_actual_smoke(tmp_path):
    protocol=dump(tmp_path/'protocol.json',{'frozen':'test'})
    original=dump(tmp_path/'original.json',dict(complete=False,total_seconds=601,formal_updates=0,all5view_K_eligible_mass_zero=True))
    checks=dict(shared_restore=True,RGB600_boundary=True,idealI_restore=True,public1920_fiveview=True,K=True)
    wrapper=dump(tmp_path/'wrapper.json',dict(completed=True,elapsed_seconds=601,checks=checks,
        phases=[dict(path=str(original),sha256=sha(original),elapsed_seconds_key='total_seconds',phase_role='actual lifecycle')]))
    output=tmp_path/'run'
    auth=dict(GPU_authorized=True,formal_authorized=True,protocol_sha256=sha(protocol),output=str(output),
        smoke_receipt=str(wrapper),smoke_receipt_sha256=sha(wrapper),minimum_smoke_seconds=600)
    validate_launch(auth,protocol,output)
    bad_wrapper=read_wrapper=json.loads(wrapper.read_text())
    dump(original,dict(complete=True,total_seconds=601,formal_updates=1,all5view_K_eligible_mass_zero=True))
    bad_wrapper['phases'][0]['sha256']=sha(original);dump(wrapper,bad_wrapper);auth['smoke_receipt_sha256']=sha(wrapper)
    with pytest.raises(RuntimeError):validate_launch(auth,protocol,output)
    auth['GPU_authorized']=False
    with pytest.raises(RuntimeError):validate_launch(auth,protocol,output)


def test_JSON_list_to_tuple_loss_snapshot_canonicalization():
    x=dict(schema='test',spec=dict(scales=(1.,2.,4.),source_weights=(('A',1.),('C',.5),('S',1.))),normalizers={'test':.123})
    encoded=json.loads(json.dumps(x));result=canonical_loss_snapshot(encoded)
    assert result==x and isinstance(encoded['spec']['scales'],list)


def test_short_queue_rule_retains_real_phase_duration_without_claiming600(tmp_path):
    protocol=dump(tmp_path/'protocol.json',{})
    original=dump(tmp_path/'original.json',dict(complete=False,total_seconds=401,formal_updates=0))
    checks=dict(shared_restore=True,RGB600_boundary=True,idealI_restore=True,public1920_fiveview=True,K=True)
    wrapper=dump(tmp_path/'wrapper.json',dict(completed=True,elapsed_seconds=401,checks=checks,
        phases=[dict(path=str(original),sha256=sha(original),elapsed_seconds_key='total_seconds')]))
    auth=dict(GPU_authorized=True,formal_authorized=True,protocol_sha256=sha(protocol),output=str(tmp_path/'run'),
        smoke_receipt=str(wrapper),smoke_receipt_sha256=sha(wrapper),engineering_rule='measured_queue_below_one_hour_complete_lifecycle',
        queue_upper_estimate_seconds=1800,throughput_receipt=str(original),throughput_receipt_sha256=sha(original))
    validate_launch(auth,protocol,tmp_path/'run')
    auth['queue_upper_estimate_seconds']=3600
    with pytest.raises(RuntimeError):validate_launch(auth,protocol,tmp_path/'run')


def test_ideal_is_I_only_preserves_full_score_and_unpremultiplies_once():
    alpha=np.full((3,4),.96,np.float32);weight=np.ones_like(alpha);available=np.ones_like(alpha,bool);K=np.zeros_like(available)
    weight[0,0]=.9;available[0,1]=False;K[0,2]=True
    target=np.zeros((3,3,3,4),np.float32);target[0]=.3*alpha;target[1]=.7*alpha;target[2]=.9*alpha
    z=dict(road_fixed_alpha=alpha,road_eligible_weight=weight,ideal_available=available,road_K=K,road_target_P=target)
    f=ideal_frame_from_arrays(z,np.ones_like(available))
    assert set(f['targets'])=={'I'}and f['scores']['I'].sum()==12 and f['contexts']['I'].sum()==9
    m=f['contexts']['I'];assert np.max(abs(f['targets']['I'][:,m]-.7))<1e-7
    assert np.all(f['targets']['I'][:,~m]==0)
    assert np.max(abs((f['targets']['I']*alpha[None])[:,m]-target[1][:,m]))<1e-7


def test_registered_continuation_resume_after2000_and_at_cap(tmp_path):
    recipe={'L0':1.};identity={'source':'fixture'}
    cp=tmp_path/'checkpoint_2000.pt';cp.write_bytes(b'opaque snapshot fixture')
    dump(tmp_path/'continuation.json',dict(max_total=4000,fixed_recipe=recipe,source_identity=identity,checkpoint_sha256=sha(cp)))
    for cursor in (2000,2100,4000):
        row=dict(iteration=cursor,objective=2.,measured_box_FW_gap=4.)
        maximum,continued,converged,last=oracle_resume_phase('G_ORACLE',cursor,[row],2000,recipe,identity,tmp_path)
        assert maximum==4000 and continued and not converged and last['iteration']==cursor
    with pytest.raises(RuntimeError):oracle_resume_phase('G_ORACLE',1900,[dict(iteration=1900,objective=2.,measured_box_FW_gap=4.)],2000,recipe,identity,tmp_path)


def test_train_only_gap_stop_and_missing_continuation_reject(tmp_path):
    row=dict(iteration=2000,objective=1.,measured_box_FW_gap=.005,held_loss=1000)
    maximum,continued,converged,last=oracle_resume_phase('G_ORACLE',2000,[row],2000,{}, {},tmp_path)
    assert maximum==2000 and not continued and converged and last is not None
    assert oracle_relative_gap(dict(objective=0,measured_box_FW_gap=.001))==1
    with pytest.raises(RuntimeError):oracle_resume_phase('G_ORACLE',2100,[dict(row,iteration=2100)],2000,{}, {},tmp_path)
    with pytest.raises(ValueError):oracle_resume_phase('IDEAL_ORACLE',1000,[],1000,{}, {},tmp_path)
