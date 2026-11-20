import json
from pathlib import Path
import numpy as np
import pytest
import launch_build_public_when_ready as w
import export_build_public_1920 as e


def test_F0_reuse_requires_exact_new_actual_ci_and_binds_old_packet(tmp_path,monkeypatch):
    b=tmp_path/'B';f=tmp_path/'F0';out=tmp_path/'output';(b/'formal_r1').mkdir(parents=True);f.mkdir();out.mkdir()
    monkeypatch.setattr(w,'B',b);monkeypatch.setattr(w,'F0ROOT',f)
    arrays=dict(ci0=np.full((2,3),.2,np.float32),active_world_rows=np.array([4,6],np.int64),eligible_active=np.array([True,False]))
    np.savez(b/'formal_r1/common_initial_ci.npz',**arrays);initial=tmp_path/'new.npz';np.savez(initial,**arrays)
    packet=f/'086.npz';np.savez(packet,RGB=np.zeros((3,1,1),np.float32))
    (f/'packets_manifest.json').write_text(json.dumps({'rows':[{'arm':'F0','raw':86,'path':str(packet),'SHA256':w.file_sha(packet)}]}))
    proof,rows=w.bind_F0(initial,out)
    assert w.read(proof)['arrays_exact']=={'ci0':True,'active_world_rows':True,'eligible_active':True}
    assert rows[0]['reused_old_F0']and not rows[0]['held_generalization_claim']and rows[0]['evaluation_role']=='BUILD_fit_all_qualified'
    arrays['ci0'][0,0]+=.001;np.savez(initial,**arrays)
    with pytest.raises(RuntimeError,match='differs'):w.bind_F0(initial,out)


def test_BUILD_roles_and_caller_owned_80F_gate_are_required(tmp_path):
    with pytest.raises(ValueError,match='BUILD arm'):e.inspect(tmp_path,tmp_path/'c',tmp_path/'r','G_NUMERIC')
    auth=tmp_path/'auth.json';auth.write_text(json.dumps({'GPU_authorized':True,'optimizer_updates':0,'max_native_forwards':40}))
    lease=tmp_path/'lease.json';lease.write_text(json.dumps({'gpu_authorized':True,'active':True,'owner':'build'}))
    with pytest.raises(RuntimeError,match='80F'):e.execute({},tmp_path/'out',lease,'build',auth)
    assert not(tmp_path/'out').exists()
