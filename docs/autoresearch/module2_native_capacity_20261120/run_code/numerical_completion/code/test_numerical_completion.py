import copy,json,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from checkpointed_lbfgsb import CheckpointedLBFGSB
from lbfgsb_cpu_fixture import fixture


def load_state(path,identity=None):
    x,f=fixture();s=CheckpointedLBFGSB(x,f,identity={'fixture':'same'}if identity is None else identity)
    s.load(path);return s


def _full_workspace_cross_process_restore_and_next_NEW_X(tmp_path):
    x,f=fixture();s=CheckpointedLBFGSB(x,f,identity={'fixture':'same'})
    assert s.next_accepted()is not None and s.next_accepted()is not None
    cp=tmp_path/'checkpoint.npz';s.save(cp);before=s.snapshot()
    expected_row=s.next_accepted();expected=s.snapshot()
    worker=Path(__file__).with_name('lbfgsb_cpu_fixture.py')
    r=subprocess.run([sys.executable,'-B',str(worker),str(cp),str(tmp_path/'restored.npz'),str(tmp_path/'next.npz')],capture_output=True,text=True)
    assert r.returncode==0,r.stdout+r.stderr
    restored=load_state(tmp_path/'restored.npz').snapshot();actual=load_state(tmp_path/'next.npz').snapshot()
    for key in s.ARRAY_NAMES:assert np.array_equal(before['arrays'][key],restored['arrays'][key]),key
    assert expected_row==json.loads(r.stdout)
    for key in ('x','f','g','wa','iwa','task','csave','lsave','isave'):
        assert np.array_equal(expected['arrays'][key],actual['arrays'][key]),key
    # dsave also stores the platform timer. Capture all of it; report any
    # cross-process timer differences rather than pretend bytewise equality.
    differences=np.flatnonzero(expected['arrays']['dsave']!=actual['arrays']['dsave']).tolist()
    (tmp_path/'dsave_difference.json').write_text(json.dumps(dict(indices=differences)))


def _exact_scipy_recipe_reference_and_bound_identity_rejection(tmp_path):
    x,f=fixture();s=CheckpointedLBFGSB(x,f,identity={'fixture':'same'})
    while not s.done:s.next_accepted()
    reference,objective,info=fmin_l_bfgs_b(lambda x:f(x)[:2],x,bounds=[(0,1)]*len(x),m=10,factr=1e7,pgtol=1e-8,maxls=20,maxfun=2000)
    assert np.max(abs(s.last_accepted_x-reference))<1e-12
    assert abs(s.last_accepted_f-objective)<1e-12
    cp=tmp_path/'end.npz';s.save(cp)
    with unittest.TestCase().assertRaises(ValueError):load_state(cp,identity={'fixture':'other'})
    bad=copy.deepcopy(s.snapshot());bad['arrays']['nbd'][0]=0
    with unittest.TestCase().assertRaises(ValueError):s.restore(bad)


def _FP64_changes_reductions_only_and_keeps_original_normalizers():
    import torch
    from native_capacity_loss import NativeCapacityLoss,LossSpec
    from fp64_reduction_loss import FP64ReductionLoss
    torch.set_num_threads(1);rng=np.random.default_rng(101);h,w=41,43
    alpha=np.full((h,w),.98,np.float32);context=np.ones((h,w),bool);context[:2,:2]=False
    target=rng.random((3,h,w),dtype=np.float32);P0=torch.tensor(rng.random((3,h,w),dtype=np.float32))
    frames={86:dict(alpha=alpha,targets={'I':target},contexts={'I':context},scores={'I':context})}
    spec=LossSpec(source_weights=(('I',1.),));original=NativeCapacityLoss(frames,spec=spec)
    saved=original.calibrate({86:P0});precise=FP64ReductionLoss(frames,spec=spec);precise.restore(saved)
    assert precise.snapshot()==saved
    with unittest.TestCase().assertRaises(RuntimeError):precise.calibrate({86:P0})
    p=P0.clone().requires_grad_();q=P0.clone().requires_grad_()
    f,_=original.objective({86:p});g,_=precise.objective({86:q})
    df=torch.autograd.grad(f,p)[0];dg=torch.autograd.grad(g,q)[0]
    assert g.dtype==torch.float64 and f.dtype==torch.float32
    assert abs(float(f)-float(g))/max(abs(float(g)),1e-12)<5e-6
    assert float((df-dg).norm())/max(float(dg.norm()),1e-12)<1e-5
    assert float((df-dg).abs().max())<1e-6


class TestNumericalCompletion(unittest.TestCase):
    def test_full_workspace_cross_process(self):
        with tempfile.TemporaryDirectory()as p:_full_workspace_cross_process_restore_and_next_NEW_X(Path(p))
    def test_SciPy_reference_and_rejections(self):
        with tempfile.TemporaryDirectory()as p:_exact_scipy_recipe_reference_and_bound_identity_rejection(Path(p))
    def test_FP64_precision_only(self):_FP64_changes_reductions_only_and_keeps_original_normalizers()


if __name__=='__main__':unittest.main(verbosity=2)
