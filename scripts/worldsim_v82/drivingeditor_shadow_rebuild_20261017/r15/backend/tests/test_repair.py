import tempfile,json,unittest
from pathlib import Path
import numpy as np
import torch
from shadow_rebuild.config import Config
from shadow_rebuild.field import GaussianField
from shadow_rebuild.topology import densify_prune
from shadow_rebuild.ownership import split_boundary
from shadow_rebuild.identity import reassign_rigid_owner
from shadow_rebuild.repair_geometry import project_actor_constraints,select_actor_growth
from shadow_rebuild.evidence_audit import classify_current,repeated_near,audit_due
from shadow_rebuild.factor_shadow import Factors,transmittance,pair_sum,whitening


def make_field(root,repair=True):
    p=Path(root)/'boxes.json';p.write_text(json.dumps({'half_sizes_m':[[1.,1.,1.],[2.,2.,2.]]}))
    c=Config(fixture_small_environment=True,env_map_res=4,sgcm_enabled=False,repair_actor_geometry=repair,
        box_half_sizes_path=str(p),densify_fraction=.5,topology_grad_threshold=.1,prune_opacity_threshold=0.,max_gaussians=100)
    xyz=np.array([[3.,0.,0.],[.98,0.,0.],[.2,.1,0.],[0,.2,0.],[1.4,0.,0.],[0,.1,0.]],np.float32)
    n=len(xyz);owner=np.array([-1,0,0,0,0,1])
    s=dict(xyz=xyz,rgb=np.full((n,3),.5,np.float32),observed_rgb=np.full((n,3),.5,np.float32),scales=np.full((n,3),.1,np.float32),
        quaternion=np.tile(np.array([1,0,0,0],np.float32),(n,1)),owner=owner,road=owner<0,reference_rgb=np.zeros((n,3)),
        reference_confidence=np.zeros(n),neighbor=np.arange(n),identity_evidence_status=np.array([0,1,1,1,1,1]),identity_probability=np.tile([.1,.8,.1],(n,1)))
    f=GaussianField(s,c);opt=torch.optim.Adam(f.parameters(),lr=.01)
    return f,opt


class GeometryRepair(unittest.TestCase):
    def test_clip_keeps_initial_outlier_and_static_and_resets_only_changed_adam_components(self):
        with tempfile.TemporaryDirectory() as d:
            f,opt=make_field(d);opt.state[f.xyz]={'step':torch.tensor(1.),'exp_avg':torch.ones_like(f.xyz),'exp_avg_sq':torch.ones_like(f.xyz)}
            with torch.no_grad():f.xyz[1]=torch.tensor([2.,.02,0.]);f.xyz[4,0]=2.;f.xyz[0,0]=4.;f.log_scales[1]=2.
            project_actor_constraints(f,opt)
            self.assertLessEqual(float(f.xyz[1,0]),1.100001)
            self.assertAlmostEqual(float(f.xyz[4,0]),1.4,places=5)
            self.assertEqual(float(f.xyz[0,0]),4.)
            self.assertEqual(float(opt.state[f.xyz]['exp_avg'][1,0]),0.)
            self.assertEqual(float(opt.state[f.xyz]['exp_avg'][2,0]),1.)
            self.assertLessEqual(float(f.scales[1].max()),.150001)
            self.assertLessEqual(float((f.xyz[1]-f.geometry_origin[1]).norm()),.250001)

    def test_multiple_topology_generations_keep_original_anchor_and_budget(self):
        with tempfile.TemporaryDirectory() as d:
            f,opt=make_field(d);initial=f.geometry_origin.clone();initial_status=f.identity_initial_status.clone()
            for step in [600,700,800,900,1000,1100,1200]:
                f.screen_gradient_accum.fill_(1);f.screen_gradient_count.fill_(1)
                densify_prune(f,opt,step)
                torch.testing.assert_close(f.geometry_origin,initial[f.sgcm_lineage])
                torch.testing.assert_close(f.identity_initial_status,initial_status[f.sgcm_lineage])
                self.assertLessEqual(int((f.owner==0).sum()),16)
                self.assertLessEqual(float((f.xyz[f.owner>=0]-f.geometry_origin[f.owner>=0]).norm(dim=-1).max()),.250001)

    def test_boundary_split_cannot_reset_cumulative_budget(self):
        with tempfile.TemporaryDirectory() as d:
            f,opt=make_field(d);origin=f.geometry_origin[1].clone();n=len(f.xyz)
            split_boundary(f,torch.tensor([1]),torch.tensor([0]),torch.tensor([.2]),opt)
            torch.testing.assert_close(f.geometry_origin[[1,n]],origin.expand(2,3))
            project_actor_constraints(f,opt)
            self.assertEqual(int(f.owner[n]),0)

    def test_certified_owner_transfer_transforms_anchor_and_does_not_assign_unknown_owner(self):
        with tempfile.TemporaryDirectory() as d:
            f,opt=make_field(d);origin=f.geometry_origin[2].clone();target=np.eye(4);target[0,3]=1.
            reassign_rigid_owner(f,[2],1,np.eye(4),target,opt)
            torch.testing.assert_close(f.geometry_origin[2],origin-torch.tensor([1.,0.,0.]))
            self.assertEqual(int(f.owner[2]),1)
            with self.assertRaises(ValueError):reassign_rigid_owner(f,[2],-2,np.eye(4),target,opt)

    def test_unknown_does_not_densify_and_original_path_has_no_new_buffers(self):
        with tempfile.TemporaryDirectory() as d:
            f,opt=make_field(d);f.config.repair_evidence_audit=True;f.identity_evidence_status[1:5]=0
            chosen=select_actor_growth(f,torch.ones(len(f.xyz),dtype=torch.bool),torch.ones(len(f.xyz)))
            self.assertFalse(bool(chosen[1:5].any()))
            legacy,_=make_field(d,False);self.assertFalse(hasattr(legacy,'geometry_origin'))
            f2,_=make_field(d);f2.load_state_dict(f.state_dict(),strict=True)
            torch.testing.assert_close(f2.geometry_origin,f.geometry_origin)


class EvidenceRepair(unittest.TestCase):
    def test_actual_audit_updates_unknown_and_owner_without_source_tag(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from shadow_rebuild.evidence_audit import run_evidence_audit
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);f,opt=make_field(root);f.config.ownership_measurement_root=d
            with torch.no_grad():f.xyz[0]=0;f.geometry_origin[0]=0
            n=len(f.xyz)
            for key,value in [('source_frame',torch.zeros(n,dtype=torch.long)),('source_raw_row',torch.ones(n,dtype=torch.long)),('source_candidate_owner',torch.full((n,),-1,dtype=torch.long))]:f.register_buffer(key,value)
            b=np.repeat(np.array([[0,0,0],[.2,.1,0]],np.float32),3,axis=0)
            np.savez(root/'body_bank_a00.npz',xyz=b,source_world=b,source_frame=np.tile([0,1,2],2),source_raw_row=np.tile([1,2,3],2))
            np.savez(root/'actual_world_road_support.npz',xyz=np.full((3,3),10.),source_frame=np.arange(3),source_raw_row=np.array([1,2,3]),support_frames=np.full(3,3))
            poses=np.broadcast_to(np.eye(4,dtype=np.float32),(3,2,4,4)).copy()
            masks=torch.zeros(1,3,2,dtype=torch.bool);masks[0,:2,0]=True
            frame=SimpleNamespace(poses=torch.from_numpy(poses[0]),valid=torch.ones(2,dtype=torch.bool),camera=SimpleNamespace(center=torch.tensor([0.,0.,5.])),
                masks=masks,mask_known=torch.ones(2,dtype=torch.bool),mask_valid=torch.ones_like(masks),road_certified=torch.zeros(1,3,dtype=torch.bool),depth=torch.ones(1,3))
            data=SimpleNamespace(actor_ids=['a','b'],train_indices=[0,1,2],frame=lambda i:frame,poses=poses,valid=np.ones((3,2),bool),
                seed=dict(xyz=f.xyz.detach().numpy().copy(),scales=f.scales.detach().numpy(),owner=f.owner.numpy().copy(),road=f.road.numpy().copy(),quaternion=f.quaternion.detach().numpy()))
            def raster(camera,xyz,cov,opacity,feature,capacity):
                weights=feature.new_zeros(3,len(feature));weights[0,0]=1;weights[1,2]=1;weights[2,1]=1
                return (weights@feature).reshape(1,3,-1)
            with patch('shadow_rebuild.raster.cuda_raster',side_effect=raster):
                r=run_evidence_audit(SimpleNamespace(field=f,config=f.config),data,opt,1500,root/'audit')
            self.assertEqual(int(f.identity_evidence_status[1]),0)
            self.assertEqual(int(f.owner[1]),0)  # Negative identity is not static.
            self.assertEqual(int(f.identity_evidence_status[0]),1)
            self.assertEqual(int(f.owner[0]),0)  # Previously unknown static row entered.
            self.assertEqual(int(f.source_candidate_owner[0]),-1)  # Original provenance kept.
            self.assertEqual(r['actual_owner_changes'],1)
            self.assertEqual(r['funnel'][0]['static_to_actor'],1)

    def test_unknown_is_not_background_and_repeated_positive_is_required(self):
        c=classify_current(np.array([0,0,1,0,1.]),np.array([0,2,0,0,1.]),np.array([0,0,0,2,1.]),
            np.array([0,0,3,0,3]),np.array([0,3,0,0,3]),np.array([0,0,0,3,3]))
        self.assertEqual(c['contradicted'].tolist(),[False,True,False,False,False])
        self.assertEqual(c['road'].tolist(),[False,False,False,True,False])
        self.assertEqual(c['mixed'].tolist(),[False,False,False,False,True])

    def test_duplicate_returns_and_world_inconsistent_road_do_not_certify(self):
        b={'xyz':np.zeros((4,3)), 'source_frame':np.array([1,1,2,3]),'source_world':np.array([[0,0,0],[0,0,0],[1,0,0],[2,0,0.]])}
        n,_=repeated_near([[0,0,0]],b);self.assertEqual(n.tolist(),[3])
        n,_=repeated_near([[0,0,0]],b,world_consistency=True);self.assertEqual(n.tolist(),[1])
        n,_=repeated_near([[10,0,0]],b);self.assertEqual(n.tolist(),[0])

    def test_audit_schedule_avoids_reset_and_rejects_collision(self):
        c=Config(repair_actor_geometry=True,repair_evidence_audit=True,ownership_measurement_root='fixture')
        c.validate();self.assertEqual([s for s in range(1,4001) if audit_due(c,s)],[1500,2500,3500])
        c.ownership_audit_first=3000
        with self.assertRaises(ValueError):c.validate()


class FactorNumericRepair(unittest.TestCase):
    def test_thin_rotated_gaussians_chunk_invariance_and_finite_gradients(self):
        torch.manual_seed(3);dtype=torch.float64;n=11;r=17
        means=torch.randn(n,3,dtype=dtype,requires_grad=True)
        rotation=torch.eye(3,dtype=dtype)[None].repeat(n,1,1)
        angle=.31;rotation[:,:2,:2]=torch.tensor([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]],dtype=dtype)
        scales=torch.tensor([[.2,.1,1e-5]],dtype=dtype).repeat(n,1).requires_grad_()
        opacity=torch.full((n,),.2,dtype=dtype,requires_grad=True)
        origins=means.detach().repeat(2,1)[:r]+torch.tensor([0.,0.,-.5],dtype=dtype)
        directions=torch.tensor([0.,0.,1.],dtype=dtype).expand(r,3)
        factors=Factors(means,rotation,scales,opacity,torch.zeros(n,dtype=torch.long))
        m,w,a=whitening(factors,dtype)
        ids=torch.arange(n)[None].expand(r,n);ref=torch.exp(-pair_sum(origins,directions,m,w,a,torch.arange(r),ids,torch.ones(r,n,dtype=torch.bool)).clamp_max(80))
        for block in [2,4,16]:
            value=transmittance(origins,directions,factors,block_size=block,pair_batch=3,recompute=False)
            torch.testing.assert_close(value,ref,atol=1e-10,rtol=1e-10)
        grads=torch.autograd.grad(value.sum(),[means,scales,opacity])
        for grad in grads:self.assertTrue(bool(torch.isfinite(grad).all()))


if __name__=='__main__':unittest.main()
