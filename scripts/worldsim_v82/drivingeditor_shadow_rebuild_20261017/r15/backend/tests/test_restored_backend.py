import unittest,torch,numpy as np
from shadow_rebuild.config import Config
from shadow_rebuild.field import GaussianField
from shadow_rebuild.model import SceneModel
from shadow_rebuild.raster import Camera,render_buffers
from shadow_rebuild.math3d import srgb_to_linear
from shadow_rebuild.topology import densify_prune

def seed():
    return dict(xyz=np.array([[0,0,5],[1,0,6]],np.float32),rgb=np.array([[.4,.5,.6],[.2,.3,.4]],np.float32),observed_rgb=np.full((2,3),.5,np.float32),scales=np.full((2,3),.3,np.float32),quaternion=np.array([[1,0,0,0]]*2,np.float32),owner=np.array([-1,0]),road=np.array([True,False]),reference_rgb=np.zeros((2,3),np.float32),reference_confidence=np.zeros(2,np.float32),neighbor=np.array([0,1]))
def cfg():return Config(env_map_res=4,fixture_small_environment=True,max_gaussians=100,continuous_ownership=True,spatial_lr_scale=10)
class RestoredBackend(unittest.TestCase):
    def test_original_generic_schedule_and_SGCM_never_overlap(self):
        from shadow_rebuild.topology import generic_due
        c=cfg();c.densify_interval=100;c.densify_until=15000
        ticks=[i for i in range(16000) if generic_due(c,i)]
        self.assertEqual(ticks[0],600);self.assertEqual(ticks[-1],14900)
        self.assertFalse(set(ticks)&set(range(550,15000,100)))
    def test_capacity_guard_and_shapes(self):
        with self.assertRaises(ValueError):Config(env_map_res=32).validate()
        with self.assertRaises(ValueError):Config(max_gaussians=450000).validate()
        m=SceneModel(seed(),cfg());self.assertEqual(tuple(m.field.features_rest.shape),(2,15,3));self.assertEqual(tuple(m.environment.base.shape),(6,4,4,3));self.assertFalse(any('sky_logit' in x for x in m.state_dict()))
    def test_native_SH_dc_and_direction(self):
        f=GaussianField(seed(),cfg());poses=torch.eye(4)[None];valid=torch.ones(1,dtype=torch.bool)
        a=f.evaluate(poses,valid,torch.zeros(3));torch.testing.assert_close(a['appearance'],srgb_to_linear(torch.tensor(seed()['rgb'])),atol=1e-6,rtol=1e-6)
        f.active_sh_degree.fill_(3)
        with torch.no_grad():f.features_rest[1,0]=.2
        a=f.evaluate(poses,valid,torch.zeros(3));b=f.evaluate(poses,valid,torch.tensor([2.,1,0]));self.assertGreater(float((a['appearance'][1]-b['appearance'][1]).abs().max()),.001)
    def test_road_without_clean_reference_has_RGB_gradient(self):
        m=SceneModel(seed(),cfg());m.set_stage('geometry');cam=Camera([[15,0,7.5],[0,15,7.5],[0,0,1]],np.eye(4),16,16)
        poses=torch.eye(4)[None];g=m.field.evaluate(poses,torch.ones(1,dtype=torch.bool),cam.center);p=render_buffers(cam,g,1,'reference',competition=True);p['sky']=torch.zeros(16,16,3);m.config.shadows=False
        im=m.train_image(p,g);loss=(im-.1).square().mean();loss.backward();self.assertGreater(float(m.field.albedo_logit.grad[0].abs().sum()),0);self.assertEqual(float(m.field.reference_confidence.sum()),0)
    def test_optimizer_schedule_and_checkpoint(self):
        m=SceneModel(seed(),cfg());opt=m.optimizer('geometry');m.schedule(opt,10000)
        rates={g['name']:g['lr'] for g in opt.param_groups};self.assertAlmostEqual(rates['field.features_rest']/rates['field.features_dc'],1/20)
        self.assertLess(rates['field.xyz'],m.config.position_lr*m.config.spatial_lr_scale);self.assertEqual(int(m.field.active_sh_degree),3)
        other=SceneModel(seed(),cfg());other.load_state_dict(m.state_dict())
        for k,x in m.state_dict().items():torch.testing.assert_close(x,other.state_dict()[k],rtol=0,atol=0)
    def test_topology_preserves_identity_and_all_SH_rows(self):
        m=SceneModel(seed(),cfg());opt=m.optimizer('geometry')
        sum(p.sum() for p in m.parameters()).backward();opt.step();m.field.screen_gradient_accum.fill_(.01);m.field.screen_gradient_count.fill_(1)
        rec=densify_prune(m.field,opt,500);self.assertGreater(rec['after'],rec['before']);self.assertEqual(len(m.field.features_rest),len(m.field.owner));self.assertTrue(torch.equal(m.field.owner[m.field.neighbor],m.field.owner))
        self.assertEqual(opt.state[m.field.features_rest]['exp_avg'].shape,m.field.features_rest.shape)
    def test_inherited_guarded_SGCM_preserves_actor_and_lineage_budget(self):
        from shadow_rebuild.sgcm_adapter import StaticAdapter
        from shadow_rebuild.legacy.paper_sgcm import SGCMController,SGCMConfig
        from shadow_rebuild.legacy.hybrid_sgcm_helpers import bind_sgcm_guard
        from types import SimpleNamespace
        s=seed();s={k:np.concatenate((x[:1],x),axis=0) for k,x in s.items()};s['xyz'][:2]=[[0,0,.1],[0,0,.35]];s['owner']=np.array([-1,-1,0]);s['neighbor']=np.arange(3);s['scales'][:]=.03
        m=SceneModel(s,cfg());opt=m.optimizer('geometry');before=m.field.xyz[m.field.owner>=0].clone()
        c=SGCMConfig();ctl=SGCMController(c,1.,np.array([0,0,1,0]),np.eye(4),[np.array([[-2,-2,0],[2,2,1]])],{})
        camera=SimpleNamespace(world_view_transform=np.eye(4),full_proj_transform=np.eye(4),timestamp=0.,image_width=16,image_height=16,image_name='fixture')
        p=StaticAdapter(m.field,opt);bind_sgcm_guard(ctl,[camera],p);r=ctl.refine(p,torch.ones(2),550)
        self.assertEqual(r['cloned'],1);torch.testing.assert_close(m.field.xyz[m.field.owner>=0],before,rtol=0,atol=0)
        restored=m.optimizer('geometry');restored.load_state_dict(opt.state_dict());self.assertFalse(restored.state)
        p=StaticAdapter(m.field,opt);bind_sgcm_guard(ctl,[camera],p,resuming=True);r=ctl.refine(p,torch.ones(len(p.rows)),650);self.assertEqual(r['cloned'],0)
if __name__=='__main__':unittest.main()
