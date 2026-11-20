"""CPU state-machine fixtures; affine context is not a native/GPU qualification."""
import copy,random
import numpy as np
import pytest
import torch
from scipy import sparse
from shared_field_gs_read import SharedAverageGSRead
from shared_training_state import SharedTrainingState

torch.set_num_threads(1)


class Field(torch.nn.Module):
    def __init__(self):
        super().__init__();self.low=torch.nn.Parameter(torch.full((1,1,3),.1));self.high=torch.nn.Parameter(torch.zeros((1,1,3)))
        self.register_buffer('fixed_gate',torch.ones(1,1,1));self.register_buffer('base',torch.full((1,1,3),.2))
    def fields(self):return .3*torch.tanh(self.low),.34*torch.tanh(self.high)
    def set_stage(self,stage):self.low.requires_grad_(stage=='C');self.high.requires_grad_(True)


class Context:
    frames=(110,86);active_world_rows=np.array([10,20,30]);native_calls=0
    def render_road(self,raw,colors):
        self.native_calls+=1
        # Two true graph nodes stand in for two native raster invocations.
        return {'rgb':colors.T.reshape(3,1,3)*(1 if raw==86 else .97)}


class Objective:
    def __init__(self):self.normalizers={'fixture_RGB':1.}
    def objective(self,predictions,mode):
        values=[((p-.5)**2).mean()for p in predictions.values()]
        value=sum(values)/len(values)
        if mode=='detail':value=value+sum(((p[:,:,1:]-p[:,:,:-1])**2).mean()for p in predictions.values())/len(values)
        return value,{'fixture_fullbatch':value.detach()}
    def snapshot(self):return {'schema':'CPU_fixture_only','normalizers':copy.deepcopy(self.normalizers)}
    def restore(self,s):
        if s['schema']!='CPU_fixture_only':raise ValueError('objective identity')
        self.normalizers=copy.deepcopy(s['normalizers'])


def fixture(*,mode='rgb',purpose='fixture',events=None):
    model=Field();active=np.array([10,20,30]);eligible=np.array([True,False,True])
    reader=SharedAverageGSRead(np.full((2,3),.2),sparse.csr_matrix([[1.],[.6]]),sparse.csr_matrix([[.9],[1.]]),
        [30,10],active,eligible,np.full((3,3),.11,np.float32),low_shape=(1,1),high_shape=(1,1))
    return SharedTrainingState(model,reader,Context(),Objective(),mode=mode,purpose=purpose,
        source_identity={'fixture':'not_native'},source_metadata={'original_field':'avg_common_A200','H0':True},
        ledger_hook=None if events is None else events.append)


def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys()and all(equal(a[k],b[k])for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b)and all(equal(x,y)for x,y in zip(a,b))
    return a==b


def test_full_state_save_restore_next_step_and_monotone_physical(tmp_path):
    random.seed(101);np.random.seed(101);torch.manual_seed(101);a=fixture();a.step();path=a.save(tmp_path/'step1.pt');saved=a.snapshot()
    expected_rng=(random.random(),np.random.random(),torch.rand(3));result_a=a.step();expected=a.snapshot()
    b=fixture();b.load(path)
    actual_rng=(random.random(),np.random.random(),torch.rand(3));assert equal(expected_rng,actual_rng)
    result_b=b.step();actual=b.snapshot()
    for key in('params','optimizer','RNG','objective','cursor','fullbatch_cursor','frozen_buffers','source_metadata'):assert equal(expected[key],actual[key])
    assert result_a['loss']==result_b['loss']
    before=dict(a.physical);a.restore(saved);assert a.cursor==1 and a.physical==before
    a.step();assert a.physical['parameter_updates']==3 and a.cursor==2
    assert a.physical['native_forward_calls']==6 and a.physical['native_backward_calls']==6 and a.physical['backward_calls']==3
    with pytest.raises(FileExistsError):a.save(path)


def test_actual_600_boundary_freezes_low_then_joint_and_stops_1200():
    a=fixture();low=a.model.low.detach().clone();groups=a.optimizer.param_groups
    assert [(g['name'],g['lr'])for g in groups]==[('low',.01),('high',.003)]
    for _ in range(600):r=a.step()
    assert r['stage']=='B'and a.stage=='C'and a.model.low.requires_grad
    assert torch.equal(low,a.model.low)and a.model.low not in a.optimizer.state
    assert int(a.optimizer.state[a.model.high]['step'])==600
    before=copy.deepcopy(a.snapshot());a.step()
    assert not torch.equal(low,a.model.low)and int(a.optimizer.state[a.model.low]['step'])==1
    a.restore(before);assert a.model.low not in a.optimizer.state
    for _ in range(600):a.step()
    assert a.cursor==1200 and int(a.optimizer.state[a.model.low]['step'])==600 and int(a.optimizer.state[a.model.high]['step'])==1200
    with pytest.raises(RuntimeError,match='budget'):a.step()


def test_restore_fail_closed_for_source_mode_purpose_phase_and_moments():
    a=fixture();a.step();saved=a.snapshot()
    for mutate in(lambda s:s.update(cursor=1.5),lambda s:s.update(fullbatch_cursor=0),lambda s:s.update(purpose='formal'),
                  lambda s:s.update(mode='detail'),lambda s:s['source_metadata'].update(original_field='pixel'),
                  lambda s:s['optimizer']['state'][1].update(step=torch.tensor(4.))):
        bad=copy.deepcopy(saved);mutate(bad)
        with pytest.raises(ValueError):a.restore(bad)
    with pytest.raises(ValueError):fixture(mode='detail').restore(saved)


def test_frozen_buffers_and_failed_gradient_require_restore():
    a=fixture();saved=a.snapshot()
    with torch.no_grad():a.model.fixed_gate.add_(.01)
    with pytest.raises(RuntimeError,match='frozen'):a.step()
    b=fixture();saved=b.snapshot();hook=b.model.high.register_hook(lambda x:x*float('nan'))
    with pytest.raises(FloatingPointError):b.step()
    assert b.cursor==0 and b.physical['parameter_updates']==0 and b.physical['failed_step_attempts']==1
    with pytest.raises(RuntimeError,match='restore'):b.step()
    with pytest.raises(RuntimeError):b.snapshot()
    hook.remove();b.restore(saved);b.step()
    assert b.cursor==1 and b.physical['step_attempts']==2 and b.physical['parameter_updates']==1


def test_only_two_parameters_and_explicit_source_frames_are_allowed():
    a=fixture();a.model.extra=torch.nn.Parameter(torch.zeros(1))
    with pytest.raises(ValueError,match='only named'):SharedTrainingState(a.model,a.reader,a.context,a.objective,mode='rgb',source_identity={'a':1},source_metadata={'b':2})
    a=fixture();a.context.frames=(86,)
    with pytest.raises(ValueError,match='fullbatch'):SharedTrainingState(a.model,a.reader,a.context,a.objective,mode='rgb',source_identity={'a':1},source_metadata={'b':2})


def test_short_stage_counts_are_smoke_only_and_checkpoint_bound():
    a=fixture()
    with pytest.raises(ValueError,match='smoke-only'):
        SharedTrainingState(a.model,a.reader,a.context,a.objective,mode='rgb',source_identity={'a':1},source_metadata={'b':2},stage_counts=(1,2))
    b=SharedTrainingState(a.model,a.reader,a.context,a.objective,mode='rgb',source_identity={'a':1},source_metadata={'b':2},purpose='smoke',stage_counts=(1,2))
    b.step();assert b.stage=='C';b.step();b.step()
    assert b.cursor==3 and int(b.optimizer.state[b.model.low]['step'])==2
    with pytest.raises(RuntimeError,match='budget'):b.step()
