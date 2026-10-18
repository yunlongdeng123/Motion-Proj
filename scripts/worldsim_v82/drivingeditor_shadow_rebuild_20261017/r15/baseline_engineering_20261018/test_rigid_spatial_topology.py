"""CPU engineering gates against pinned native spatial methods, not training.

Reference method bodies are AST-extracted; only CUDA device literals become
CPU. A storage shim supplies append/prune and zero-velocity temporal storage.
This proves spatial rules in the explicit rigid setting, not full PVG identity.
"""
from pathlib import Path
import ast, copy, hashlib, json, math, os, sys, unittest
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
W = Path(__file__).parent
sys.path.insert(0,str(W/'cpu_reference'))
import torch
from rigid_model import RigidInstanceGaussianModel, quaternion_to_matrix
from restore_contract import assert_exact_state
from rigid_spatial_topology import SpatialPolicy, spatial_mutation, after_backward, world_rows


class NativeStorage:
    """Geometry storage only; decision/sampling/pruning methods come from source."""
    def __init__(self, model, policy):
        world = model.evaluate(0.)
        self._xyz = world.xyz.detach().clone()
        self._rotation = world.rotation.detach().clone()
        for name in ['_features_dc','_features_rest','_scaling','_opacity']:
            setattr(self,name,getattr(model,name).detach().clone())
        n = model.num_points
        self._t = torch.zeros((n,1)); self._scaling_t = torch.zeros((n,1)); self._velocity = torch.zeros((n,3))
        self.point_ids = model.point_ids.clone()
        self.gaussian_ids = model.gaussian_ids.clone()
        self.parent_gaussian_ids = model.parent_gaussian_ids.clone()
        self.next_id = int(model.next_gaussian_id)
        self._sgcm_lineage_ids = torch.arange(n)
        self.xyz_gradient_accum = model.xyz_gradient_accum.clone()
        self.denom = model.denom.clone()
        self.t_gradient_accum = torch.zeros_like(self.denom)
        self.max_radii2D = model.max_radii2D.clone()
        self.contract = policy.contracted; self.percent_dense = policy.percent_dense
        self.big_point_threshold = policy.big_point_threshold
        self.t_grad = False; self.no_time_split = True
        self.T = 1.
        self.scaling_inverse_activation = torch.log
        self.scaling_t_inverse_activation = torch.log

    get_xyz = property(lambda s:s._xyz)
    get_scaling = property(lambda s:s._scaling.exp())
    get_opacity = property(lambda s:s._opacity.sigmoid())
    get_scaling_t = property(lambda s:s._scaling_t.exp())
    get_t = property(lambda s:s._t)
    get_inst_velocity = property(lambda s:s._velocity)

    def densification_postfix(self, xyz, dc, rest, opacity, scaling, rotation, t, scaling_t, velocity, sgcm_parent_indices=None):
        assert sgcm_parent_indices is not None
        parent = self.gaussian_ids[sgcm_parent_indices].clone()
        ids = self.point_ids[sgcm_parent_indices].clone()
        for name, new in zip(['_xyz','_features_dc','_features_rest','_opacity','_scaling','_rotation','_t','_scaling_t','_velocity'],
                             [xyz,dc,rest,opacity,scaling,rotation,t,scaling_t,velocity]):
            setattr(self,name,torch.cat([getattr(self,name),new]))
        self.point_ids = torch.cat([self.point_ids,ids])
        self.gaussian_ids = torch.cat([self.gaussian_ids,torch.arange(self.next_id,self.next_id+len(xyz))])
        self.parent_gaussian_ids = torch.cat([self.parent_gaussian_ids,parent])
        self.next_id += len(xyz)
        n = len(self._xyz)
        self.xyz_gradient_accum = torch.zeros((n,1)); self.t_gradient_accum = torch.zeros((n,1))
        self.denom = torch.zeros((n,1)); self.max_radii2D = torch.zeros(n)

    def prune_points(self, remove):
        for name in ['_xyz','_features_dc','_features_rest','_opacity','_scaling','_rotation','_t','_scaling_t','_velocity',
                     'point_ids','gaussian_ids','parent_gaussian_ids','xyz_gradient_accum','t_gradient_accum','denom','max_radii2D']:
            setattr(self,name,getattr(self,name)[~remove])


def install_native_reference():
    source = (W/'cpu_reference/gaussian_model.py').read_text(encoding='utf-8')
    native = next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='GaussianModel')
    wanted = ['densify_and_clone','densify_and_split','densify_and_prune']
    class CPUDevice(ast.NodeTransformer):
        def visit_Constant(self,node):
            return ast.copy_location(ast.Constant(value='cpu'),node) if node.value=='cuda' else node
    ns = dict(torch=torch,np=__import__('numpy'),build_rotation=quaternion_to_matrix)
    selected = [CPUDevice().visit(copy.deepcopy(n)) for n in native.body if isinstance(n,ast.FunctionDef) and n.name in wanted]
    assert len(selected)==3
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[])),'pinned_native_spatial_cpu','exec'),ns)
    for name in wanted:
        setattr(NativeStorage,name,ns[name])
    return hashlib.sha256((W/'cpu_reference/gaussian_model.py').read_bytes()).hexdigest()


def fixture(transformed=False):
    n=6
    xyz=torch.tensor([[0.,0.,0.],[.03,.04,0.],[.08,0.,0.],[.1,.02,0.],[.02,.1,0.],[.05,.05,0.]])
    sigma=torch.tensor([[.003,.004,.002],[.04,.03,.02],[.8,.7,.6],[.004,.005,.003],[.04,.06,.03],[.006,.007,.005]])
    poses=torch.eye(4).repeat(1,2,1,1)
    if transformed:
        a=.63;R=torch.tensor([[math.cos(a),-math.sin(a),0.],[math.sin(a),math.cos(a),0.],[0.,0.,1.]])
        poses[0,:,:3,:3]=R;poses[0,0,:3,3]=torch.tensor([2.,.5,0.]);poses[0,1,:3,3]=torch.tensor([-1.5,.2,0.])
    m=RigidInstanceGaussianModel(xyz,torch.full((n,3),.4),sigma,torch.tensor([0,0,0,1,1,1]),
        ['real_a','real_b'],[0.],poses,torch.ones((1,2),dtype=torch.bool),box_size=torch.ones((1,2,3))*10,
        device='cpu',sh_degree=3)
    opt=m.make_optimizer(dict(xyz=.001,f_dc=.001,f_rest=.001,scaling=.001,rotation=.001,opacity=.001),eps=1e-15)
    loss=sum((p*p).sum() for g in opt.param_groups for p in g['params'])
    loss.backward();opt.step();opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        m._opacity[3]=torch.logit(torch.tensor(.001))
        m.xyz_gradient_accum[:,0]=torch.tensor([.0002,.0002,0.,.0002,.0002,0.])
        m.denom.fill_(1);m.max_radii2D.fill_(100)
    return m,opt


NATIVE_HASH=install_native_reference()


class Gates(unittest.TestCase):
    def compare_reference(self, transformed, contracted, screen):
        model,opt=fixture(transformed)
        policy=SpatialPolicy(scene_extent=2.,contracted=contracted)
        ref=NativeStorage(model,policy)
        torch.manual_seed(31)
        with torch.no_grad():
            ref.densify_and_prune(policy.gradient_threshold*.5,policy.min_opacity,policy.scene_extent,screen,0.)
        torch.manual_seed(31)
        event=spatial_mutation(model,opt,policy,0.,max_screen_radius=screen)
        world=model.evaluate(0.)
        for name, a, b in [('world_xyz',world.xyz,ref._xyz),('world_rotation',world.rotation,ref._rotation),
                           ('scaling',model._scaling,ref._scaling),('opacity',model._opacity,ref._opacity),
                           ('DC',model._features_dc,ref._features_dc),('rest',model._features_rest,ref._features_rest)]:
            self.assertTrue(torch.allclose(a,b,atol=2e-6,rtol=2e-5),name)
        for name in ['point_ids','gaussian_ids','parent_gaussian_ids','max_radii2D','xyz_gradient_accum','denom']:
            self.assertTrue(torch.equal(getattr(model,name),getattr(ref,name)),name)
        self.assertEqual(int(model.next_gaussian_id),ref.next_id)
        self.assertTrue(event['native_append_resets_screen_radius_history'])
        self.assertFalse(event['temporal_PVG_parameters_or_time_splits'])
        self.assertGreater(event['split_parents'],0)
        self.assertGreater(event['cloned'],0)

    def test_native_spatial_rules_identity(self):self.compare_reference(False,True,None)
    def test_native_spatial_rules_rotated_translated(self):self.compare_reference(True,True,10)
    def test_native_spatial_rules_uncontracted(self):self.compare_reference(True,False,10)

    def test_moments_owners_and_roundtrip(self):
        model,opt=fixture(True);policy=SpatialPolicy(scene_extent=2.)
        before=model.capture(opt);rng=torch.get_rng_state().clone()
        spatial_mutation(model,opt,policy,0.,max_screen_radius=10)
        final=model.capture(opt);final_rng=torch.get_rng_state().clone()
        old_groups={tuple(g['parameter_names']):g for g in before['optimizer']['groups']}
        for g in opt.param_groups:
            p=g['params'][0];name=next(n for n,v in model.named_parameters() if v is p)
            old_group=old_groups[(name,)];old_idx=old_group['params'][0] if 'params' in old_group else None
            old_state_dict=before['optimizer']['state_dict']
            group_idx=next(i for i,x in enumerate(before['optimizer']['groups']) if tuple(x['parameter_names'])==(name,))
            state_key=old_state_dict['param_groups'][group_idx]['params'][0]
            old_state=old_state_dict['state'][state_key]
            for i,identity in enumerate(model.gaussian_ids.tolist()):
                if identity < 6:
                    self.assertTrue(torch.equal(opt.state[p]['exp_avg'][i],old_state['exp_avg'][identity]))
                    self.assertTrue(torch.equal(opt.state[p]['exp_avg_sq'][i],old_state['exp_avg_sq'][identity]))
                else:
                    self.assertEqual(int(opt.state[p]['exp_avg'][i].count_nonzero()),0)
                    parent=int(model.parent_gaussian_ids[i])
                    self.assertEqual(int(model.point_ids[i]),int(before['state_dict']['point_ids'][parent]))
            self.assertTrue(torch.equal(opt.state[p]['step'],old_state['step']))
            self.assertIsNone(p.grad)
        restored=RigidInstanceGaussianModel.from_capture(before,device='cpu')
        torch.set_rng_state(rng)
        spatial_mutation(restored,restored.optimizer,policy,0.,max_screen_radius=10)
        assert_exact_state(final,restored.capture(restored.optimizer))
        self.assertTrue(torch.equal(final_rng,torch.get_rng_state()))

    def test_schedule_boundaries(self):
        p=SpatialPolicy(scene_extent=2.)
        for step,expected in [(499,False),(500,False),(501,False),(600,True),(3000,True),(3100,True),(14900,True),(15000,False),(30000,False)]:
            self.assertEqual(p.mutation_due(step,6),expected)
        self.assertFalse(p.accumulation_active(600,1000000))

    def test_missing_actual_gradient_rejected(self):
        m,opt=fixture()
        from types import SimpleNamespace
        with self.assertRaisesRegex(ValueError,'missing actual'):
            after_backward(m,opt,dict(viewspace_points=SimpleNamespace(grad=None)),SpatialPolicy(scene_extent=2.),0.,600)

    def test_world_not_local_controls_threshold(self):
        m,opt=fixture(True)
        p=SpatialPolicy(scene_extent=2.)
        from rigid_spatial_topology import contracted_factor
        self.assertFalse(torch.allclose(contracted_factor(world_rows(m,0.),p),contracted_factor(m.local_xyz,p)))

    def test_edited_training_state_rejected(self):
        m,opt=fixture()
        m.translate('real_a',[.1,0,0])
        with self.assertRaisesRegex(ValueError,'rejects edited'):
            spatial_mutation(m,opt,SpatialPolicy(scene_extent=2.),0.)

    def test_actual_joint_gradient_rows_and_repeated_sources(self):
        m,opt=fixture()
        m.xyz_gradient_accum.zero_();m.denom.zero_();m.max_radii2D.zero_()
        from types import SimpleNamespace
        package=dict(viewspace_points=SimpleNamespace(grad=torch.tensor([[999.,999.,0.],[999.,999.,0.],[3.,4.,0.],[0.,2.,0.],[0.,2.,0.]])),
            static_point_count=2,instance_row_ids=torch.tensor([4,1,4]),radii=torch.tensor([100.,100.,1.,2.,3.]))
        event=after_backward(m,opt,package,SpatialPolicy(scene_extent=2.),0.,499)
        self.assertFalse(event['mutated'])
        self.assertEqual(float(m.xyz_gradient_accum[4]),7.)
        self.assertEqual(float(m.xyz_gradient_accum[1]),2.)
        self.assertEqual(float(m.denom[4]),2.)
        self.assertEqual(float(m.max_radii2D[4]),3.)
        self.assertEqual(float(m.xyz_gradient_accum.sum()),9.)


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Gates)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    receipt=dict(status='PASSED_CPU_SPATIAL_CANDIDATE_GATES' if result.wasSuccessful() else 'FAILED_CPU_SPATIAL_CANDIDATE_GATES',
        tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        native_geometry_source_sha256=NATIVE_HASH,reference='pinned native clone/split/prune AST, only device literals changed to CPU; storage shim with zero velocity and no time split',
        exact_full_PVG_equivalence_claimed=False,native_unused_temporal_RNG_draws_not_ported=True,
        native_append_clears_screen_radius_history_limitation_preserved=True,
        CUDA_initialized=torch.cuda.is_initialized(),GPU_execution_passed=False,
        formal_training_updates=0,implemented_in_any_running_training=False)
    (W/'spatial_topology_CPU_receipt.json').write_text(json.dumps(receipt,indent=2))
    if not result.wasSuccessful():sys.exit(1)
    assert not torch.cuda.is_initialized()
