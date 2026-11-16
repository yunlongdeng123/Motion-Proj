"""CPU 接口测试及真实原生方法 AST 拓扑/Adam/保存恢复测试。

原生方法只将 device='cuda' 适配为 CPU；不替换 parent lineage helpers。
光栅画质另由真实 GPU pilot 验收，这里的小深度例只测查询接线。
"""
import ast
import copy
import io
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import types
import unittest
import tempfile

import numpy as np
import torch
from torch import nn

from scripts.worldsim_v82.drivingeditor_dynamic_edit import (
    ActorEditQuery, actor_ids, bind_initial_ownership, install_identity_serialization, render_scene)
from scripts.worldsim_v82.drivingeditor_dynamic_edit.ownership import OWNERSHIP_KEY

ENGINE = Path(os.environ.get('DRIVINGEDITOR_NATIVE_ENGINE', '/root/autodl-tmp/drivingeditor_repro/campaign/r15_baseline_owner5_20261018/training_source/scene0008_089_137'))
FIELDS = {'xyz':'_xyz','f_dc':'_features_dc','f_rest':'_features_rest','opacity':'_opacity',
          'scaling':'_scaling','rotation':'_rotation','t':'_t','scaling_t':'_scaling_t','velocity':'_velocity'}


class CPUDevice(ast.NodeTransformer):
    def visit_keyword(self, node):
        node = self.generic_visit(node)
        if node.arg == 'device' and isinstance(node.value, ast.Constant) and node.value.value == 'cuda':
            node.value = ast.Constant(value='cpu')
        return node


def native_classes():
    helpers = ast.parse((ENGINE/'scene/hybrid_sgcm_helpers.py').read_text())
    utilities = ast.parse((ENGINE/'utils/general_utils.py').read_text())
    source = ast.parse((ENGINE/'scene/gaussian_model.py').read_text())
    nodes = [n for n in helpers.body if isinstance(n, (ast.Assign, ast.FunctionDef))]
    nodes += [n for n in utilities.body if isinstance(n,ast.FunctionDef) and n.name in ['build_rotation','get_expon_lr_func','get_step_lr_func']]
    nodes += [n for n in source.body if isinstance(n,ast.ClassDef)]
    module = types.ModuleType('native_actor_topology_cpu')
    namespace = module.__dict__
    namespace.update(torch=torch,np=np,nn=nn,math=math,BasicPointCloud=object,
                     inverse_sigmoid=lambda value:torch.log(value/(1-value)))
    adapted = CPUDevice().visit(ast.Module(body=copy.deepcopy(nodes),type_ignores=[]))
    exec(compile(ast.fix_missing_locations(adapted),'native:device_only_cpu','exec'),namespace)
    return namespace['GaussianModel'], namespace['Dynamic_GaussianModel']


def make_model(cls, n=6):
    model = cls.__new__(cls)
    model.setup_functions()
    model.active_sh_degree=0;model.max_sh_degree=3;model.spatial_lr_scale=1.
    model.contract=False;model.t_grad=True;model.no_time_split=True;model.big_point_threshold=.1
    model.T=.2;model.velocity_decay=1.
    values=dict(xyz=torch.arange(n*3,dtype=torch.float32).reshape(n,3)/100,
        f_dc=torch.arange(n*3,dtype=torch.float32).reshape(n,1,3)/10,f_rest=torch.zeros(n,15,3),
        opacity=torch.zeros(n,1),scaling=torch.full((n,3),math.log(.001)),
        rotation=torch.tensor([[1.,0.,0.,0.]]).repeat(n,1),t=torch.zeros(n,1),
        scaling_t=torch.full((n,1),math.log(.3)),velocity=torch.zeros(n,3))
    for name,value in values.items():setattr(model,FIELDS[name],nn.Parameter(value))
    args=SimpleNamespace(percent_dense=.01,position_lr_init=.00016,position_lr_final=.0000016,
        position_lr_delay_mult=.01,iterations=30000,feature_lr=.0025,opacity_lr=.005,
        scaling_lr=.005,rotation_lr=.001,t_lr_init=.0008,scaling_t_lr=.002,velocity_lr=.001)
    model.training_setup(args)
    model.max_radii2D=torch.arange(n,dtype=torch.float32)
    for index,group in enumerate(model.optimizer.param_groups):
        parameter=group['params'][0]
        moment=torch.arange(parameter.numel(),dtype=torch.float32).reshape(parameter.shape)+1+index*100
        model.optimizer.state[parameter]=dict(step=torch.tensor(7.),exp_avg=moment,exp_avg_sq=moment.square())
    return model,args


def snapshot(model):
    return {name:getattr(model,name).detach().clone() for name in FIELDS.values()}


class QueryTests(unittest.TestCase):
    def test_inference_export_only_removes_adam(self):
        from scripts.worldsim_v82.drivingeditor_dynamic_edit.export_inference import strip_adam,validate_equal
        from scripts.worldsim_v82.drivingeditor_dynamic_edit.ownership import LINEAGE_KEY
        data=[float(i) for i in range(18)];data[1]=torch.arange(9,dtype=torch.float32).reshape(3,3)
        source=torch.tensor([37,5,6],dtype=torch.int64)
        data[14]=dict(state={0:dict(exp_avg=torch.ones(3,3))},param_groups=[dict(params=[0])])
        data[14][LINEAGE_KEY]=dict(schema_version=1,lineage_ids=torch.arange(3),clone_blocked=torch.zeros(3,dtype=torch.bool))
        data[14][OWNERSHIP_KEY]=dict(schema_version=1,dynamic_actor_ids=[5,6,37],source_actor_ids=source,actor_ids=source.clone())
        slim=strip_adam(tuple(data));validate_equal(tuple(data),slim)
        self.assertNotIn('state',slim[14]);self.assertNotIn('param_groups',slim[14]);self.assertTrue(slim[14]['inference_only'])
        buffer=io.BytesIO();torch.save(slim,buffer);buffer.seek(0)
        validate_equal(tuple(data),torch.load(buffer,map_location='cpu',weights_only=False))
        with torch.no_grad():slim[1][0,0]=99
        self.assertEqual(data[1][0,0],0)
        with self.assertRaises(ValueError):validate_equal(tuple(data),slim)

    def test_checkpoint_description_is_readonly_and_does_not_rebind_order(self):
        from scripts.worldsim_v82.drivingeditor_dynamic_edit.scene_io import checkpoint_actors
        from scripts.worldsim_v82.drivingeditor_dynamic_edit.ownership import LINEAGE_KEY
        with tempfile.TemporaryDirectory() as temp:
            model=Path(temp);data=[None]*18;data[1]=torch.zeros(3,3)
            source=torch.tensor([37,5,6],dtype=torch.int64)
            data[14]={LINEAGE_KEY:dict(schema_version=1,lineage_ids=torch.arange(3),clone_blocked=torch.zeros(3,dtype=torch.bool)),
                OWNERSHIP_KEY:dict(schema_version=1,dynamic_actor_ids=[37,6,5],source_actor_ids=source,actor_ids=source.clone())}
            torch.save((tuple(data),7),model/'dynamic_chkpnt7.pth')
            info=checkpoint_actors(model,7)
            self.assertEqual(info['dynamic_actor_ids'],[5,6,37]);self.assertEqual(info['dynamic_rows'],3)
            data[14][OWNERSHIP_KEY]['schema_version']=True
            torch.save((tuple(data),7),model/'dynamic_chkpnt7.pth')
            with self.assertRaises(ValueError):checkpoint_actors(model,7)

    def test_non_contiguous_order_and_subset(self):
        a=ActorEditQuery([37,5,26,18,6],[37,6])
        self.assertEqual(a.dynamic_actor_ids,(5,6,18,26,37))
        self.assertEqual(a.delete_actor_ids,(6,37))
        for dynamic,delete in [([5,5],[]),([5],[6]),([True],[]),([5],[37]),([] ,[])]:
            with self.assertRaises(ValueError):ActorEditQuery(dynamic,delete)

    def test_native_routing_and_no_parameter_mutation(self):
        class Model:
            def __init__(self,xyz,colors):
                self._xyz=nn.Parameter(torch.tensor(xyz,dtype=torch.float32))
                self.colors=torch.tensor(colors,dtype=torch.float32)
                self._opacity=nn.Parameter(torch.ones(len(xyz),1)*.5)
            @property
            def get_xyz(self):return self._xyz
        static=Model([[0,0,4]],[[0,0,1]])
        dynamic=Model([[0,0,3],[0,0,2],[0,0,5]],[[0,1,0],[1,0,0],[1,1,0]])
        bind_initial_ownership(dynamic,[(5,1),(37,1),(6,1)],[5,6,37])
        ns={'torch':torch}
        exec("def prepare_rasterization(camera,model,pipe,*pos,**kw):\n return {'mask':torch.ones(len(model.get_xyz),dtype=torch.bool)}\n"
             "def render(camera,static,dynamic,args,bg,**kw):\n"
             " s=prepare_rasterization(camera,static,args)['mask'];d=prepare_rasterization(camera,dynamic,args,dynamic=True)['mask']\n"
             " xyz=torch.cat([static.get_xyz[s],dynamic.get_xyz[d]]);colors=torch.cat([static.colors[s],dynamic.colors[d]])\n"
             " return {'render':colors[xyz[:,2].argmin()], 'dynamic_mask':d, 'render_function':render}\n",ns)
        renderer=SimpleNamespace(render=ns['render'])
        original_prepare=ns['prepare_rasterization'];before=[p.detach().clone() for p in [static._xyz,static._opacity,dynamic._xyz,dynamic._opacity]]
        keep=render_scene(None,static,dynamic,None,None,dynamic_actor_ids=[37,5,6],renderer=renderer)
        self.assertIs(keep['render_function'],renderer.render)
        self.assertTrue(keep['dynamic_mask'].all())
        self.assertTrue(torch.equal(keep['render'],torch.tensor([1.,0.,0.])))
        deleted=render_scene(None,static,dynamic,None,None,dynamic_actor_ids=[5,6,37],delete_actor_ids=[37],renderer=renderer)
        self.assertEqual(deleted['dynamic_mask'].tolist(),[True,False,True])
        self.assertTrue(torch.equal(deleted['render'],torch.tensor([0.,1.,0.])))
        both=render_scene(None,static,dynamic,None,None,dynamic_actor_ids=[5,6,37],delete_actor_ids=[5,37],renderer=renderer)
        self.assertTrue(torch.equal(both['render'],torch.tensor([0.,0.,1.])))
        again=render_scene(None,static,dynamic,None,None,dynamic_actor_ids=[6,37,5],renderer=renderer)
        self.assertTrue(torch.equal(again['render'],keep['render']))
        self.assertIs(ns['prepare_rasterization'],original_prepare)
        for value,old in zip([static._xyz,static._opacity,dynamic._xyz,dynamic._opacity],before):self.assertTrue(torch.equal(value,old))
        with self.assertRaises(ValueError):render_scene(None,static,dynamic,None,None,dynamic_actor_ids=[5,6,37],delete_actor_ids=[26],renderer=renderer)
        exec("def static_only(camera,static,dynamic,args,bg,**kw):\n return prepare_rasterization(camera,static,args)\n",ns)
        with self.assertRaises(RuntimeError):
            render_scene(None,static,dynamic,None,None,dynamic_actor_ids=[5,6,37],delete_actor_ids=[37],
                         renderer=SimpleNamespace(render=ns['static_only']))


@unittest.skipUnless((ENGINE/'scene/gaussian_model.py').is_file(),'Requires frozen native source for actual AST methods')
class NativeTopologyTests(unittest.TestCase):
    def setUp(self):
        self.base,self.dynamic=native_classes()
        install_identity_serialization(self.base,self.dynamic)
        self.model,self.args=make_model(self.dynamic)
        bind_initial_ownership(self.model,[(5,1),(6,2),(18,1),(26,1),(37,1)],[37,5,26,6,18])

    def test_real_clone_split_prune_and_optimizer_alignment(self):
        model=self.model
        grad=torch.tensor([[0.],[2.],[0.],[0.],[0.],[2.]])
        model.densify_and_clone(grad,1.,1.,grad,1.)
        self.assertEqual(actor_ids(model).tolist(),[5,6,6,18,26,37,6,37])
        # Two distinct parents: native repeat(N) produces [6,26,6,26], not interleave.
        with torch.no_grad():model._scaling[1]=math.log(.1);model._scaling[4]=math.log(.1)
        n=len(model.get_xyz);before=actor_ids(model).clone();grad=torch.zeros(n,1);grad[[1,4]]=2
        selected=torch.zeros(n,dtype=torch.bool);selected[[1,4]]=True
        torch.manual_seed(17)
        model.densify_and_split(grad,1.,1.,grad,1.)
        expected=torch.cat([before[~selected],before[selected].repeat(2)])
        self.assertTrue(torch.equal(actor_ids(model),expected))
        prune=torch.arange(len(expected))%3==1
        model.prune_points(prune)
        self.assertTrue(torch.equal(actor_ids(model),expected[~prune]))
        for group in model.optimizer.param_groups:
            parameter=group['params'][0]
            self.assertEqual(len(parameter),len(model.get_xyz))
            self.assertEqual(parameter.shape,model.optimizer.state[parameter]['exp_avg'].shape)
            self.assertEqual(parameter.shape,model.optimizer.state[parameter]['exp_avg_sq'].shape)
            self.assertEqual(model.optimizer.state[parameter]['step'].item(),7)

    def test_real_capture_serialized_full_restore(self):
        capture=self.model.capture();self.assertEqual(len(capture),18)
        payload=capture[14][OWNERSHIP_KEY]
        self.assertTrue(torch.equal(payload['actor_ids'],actor_ids(self.model)))
        buffer=io.BytesIO();torch.save(capture,buffer);buffer.seek(0)
        restored=self.dynamic.__new__(self.dynamic)
        restored.max_sh_degree=3;restored.contract=False;restored.t_grad=True;restored.no_time_split=True
        restored.restore(torch.load(buffer,map_location='cpu',weights_only=False),training_args=self.args)
        self.assertTrue(torch.equal(actor_ids(restored),actor_ids(self.model)))
        for name,value in snapshot(self.model).items():self.assertTrue(torch.equal(value,getattr(restored,name)))
        for old,new in zip(self.model.optimizer.param_groups,restored.optimizer.param_groups):
            for name in ['step','exp_avg','exp_avg_sq']:
                self.assertTrue(torch.equal(self.model.optimizer.state[old['params'][0]][name],restored.optimizer.state[new['params'][0]][name]))
        from scripts.worldsim_v82.drivingeditor_dynamic_edit.export_inference import strip_adam
        inference=self.dynamic.__new__(self.dynamic);inference.max_sh_degree=3;inference.optimizer=None
        inference.restore(strip_adam(capture),training_args=None)
        self.assertTrue(torch.equal(actor_ids(inference),actor_ids(self.model)))
        for name,value in snapshot(self.model).items():self.assertTrue(torch.equal(value,getattr(inference,name)))
        self.assertIsNone(inference.optimizer)
        legacy=list(capture);legacy[14]=dict(legacy[14]);legacy[14].pop(OWNERSHIP_KEY)
        with self.assertRaises(ValueError):restored.restore(tuple(legacy),training_args=self.args)
        corrupt=list(capture);corrupt[14]=copy.deepcopy(corrupt[14]);corrupt[14][OWNERSHIP_KEY]['actor_ids'][0]=37
        with self.assertRaises(ValueError):restored.restore(tuple(corrupt),training_args=self.args)

    def test_empty_append_and_split_three_children(self):
        model=self.model;before=actor_ids(model).clone();grad=torch.zeros(len(before),1)
        model.densify_and_clone(grad,1.,1.,grad,1.)
        self.assertTrue(torch.equal(actor_ids(model),before))
        model.no_time_split=False
        with torch.no_grad():model._scaling[[0,3]]=math.log(.1)
        grad[[0,3]]=2;selected=torch.zeros(len(before),dtype=torch.bool);selected[[0,3]]=True
        model.densify_and_split(grad,1.,1.,grad,1.)
        self.assertTrue(torch.equal(actor_ids(model),torch.cat([before[~selected],before[selected].repeat(3)])))

    def test_metadata_does_not_change_native_topology_or_adam(self):
        _,original_class=native_classes()
        original,_=make_model(original_class)
        labeled=self.model
        def equal_models():
            for name in list(FIELDS.values())+['max_radii2D','xyz_gradient_accum','t_gradient_accum','denom']:
                self.assertTrue(torch.equal(getattr(original,name),getattr(labeled,name)),name)
            for old,new in zip(original.optimizer.param_groups,labeled.optimizer.param_groups):
                self.assertEqual(old['name'],new['name'])
                for field in ['step','exp_avg','exp_avg_sq']:
                    self.assertTrue(torch.equal(original.optimizer.state[old['params'][0]][field],
                                                labeled.optimizer.state[new['params'][0]][field]),field)
        def both(method,*args,**kwargs):
            before=torch.get_rng_state().clone()
            getattr(original,method)(*args,**kwargs)
            after=torch.get_rng_state().clone()
            torch.set_rng_state(before)
            getattr(labeled,method)(*args,**kwargs)
            self.assertTrue(torch.equal(torch.get_rng_state(),after))
            equal_models()
        equal_models()
        grad=torch.tensor([[2.],[0.],[0.],[0.],[0.],[2.]])
        both('densify_and_clone',grad,1.,1.,grad,1.)
        for model in [original,labeled]:
            model.no_time_split=False
            with torch.no_grad():model._scaling[[1,4]]=math.log(.1)
        grad=torch.zeros(len(original.get_xyz),1);grad[[1,4]]=2
        both('densify_and_split',grad,1.,1.,grad,1.,time_split=True)
        prune=torch.arange(len(original.get_xyz))%4==1
        both('prune_points',prune)
        self.assertFalse(bool(labeled._sgcm_clone_blocked.any()))
        self.assertIsNone(getattr(original,'_sgcm_lineage_ids',None))


if __name__=='__main__':unittest.main()
