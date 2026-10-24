import unittest
import numpy as np
from road_teacher_cpu import Teacher

class TeacherTests(unittest.TestCase):
    def atlas(self):
        x,y=np.meshgrid(np.arange(-10,11,.25),np.arange(-3,3,.25));xy=np.c_[x.ravel(),y.ravel()]
        return dict(xy=xy,z=.03*xy[:,0]+.02*xy[:,1],rgb=np.tile([.3,.35,.4],(len(xy),1)),
                    rgb_mad=np.zeros((len(xy),3)),source_frames=np.full(len(xy),3))
    def test_withdrawn_hole_has_no_label_or_geometry_leak(self):
        a=self.atlas();keep=(np.abs(a['xy'][:,0])>=4)|(np.abs(a['xy'][:,1])>=1)
        before=Teacher(a,keep).predict(np.array([[0.,0.],[1.,.1]]))
        a['rgb'][~keep]=100.;a['z'][~keep]=-999
        after=Teacher(a,keep).predict(np.array([[0.,0.],[1.,.1]]))
        np.testing.assert_array_equal(before['rgb'],after['rgb']);np.testing.assert_array_equal(before['z'],after['z'])
        self.assertGreater(before['nearest_donor_m'].min(),.8)
        np.testing.assert_allclose(before['rgb'],[[.3,.35,.4]]*2,atol=1e-10)
        self.assertTrue(np.all(before['loss_weight']>0))
    def test_plane_and_normals_finite(self):
        p=Teacher(self.atlas()).predict(np.array([[0.,0.],[2.,1.]]))
        self.assertTrue(np.isfinite(p['z']).all());self.assertLess(abs(p['z'][1]-.08),.02)
        np.testing.assert_allclose(np.linalg.norm(p['normal'],axis=1),1)
    def test_blocked_road_still_gets_direct_gradient(self):
        import torch
        from road_teacher_bridge import edit_teacher_loss,assert_independent
        master=torch.nn.Parameter(torch.tensor([.1,.1,.1]));edit=torch.nn.Parameter(master.detach().clone())
        assert_independent([master],[edit])
        with self.assertRaises(ValueError):assert_independent([master],[master])
        road_color=torch.nn.Parameter(torch.tensor([.1,.1,.1]));opacity=torch.nn.Parameter(torch.tensor(.2))
        # Fully opaque static ghost makes deleted RGB independent of the road.
        target=dict(pseudo_rgb=torch.full((1,1,3),.4),pseudo_depth=torch.full((1,1),13.),
                    source_kind=torch.full((1,1),2),loss_weight=torch.full((1,1),.3),observed_reference_valid=torch.zeros((1,1),dtype=torch.bool))
        mask=torch.ones((1,1),dtype=torch.bool)
        loss=edit_teacher_loss((road_color*opacity).reshape(1,1,3),opacity.reshape(1,1),
            torch.full((1,1),13.),edit.reshape(1,1,3),torch.zeros((1,1)),master.reshape(1,1,3),
            edit.reshape(1,1,3),target,mask,mask,~mask)
        loss['total'].backward()
        self.assertIsNone(master.grad)
        self.assertTrue((road_color.grad<0).all());self.assertLess(opacity.grad.item(),0)
        self.assertTrue((edit.grad<0).all())
    def test_zero_weight_has_zero_teacher_force(self):
        import torch
        from road_teacher_bridge import edit_teacher_loss
        rgb=torch.ones((1,1,3),requires_grad=True);scalar=torch.ones((1,1));mask=scalar.bool()
        target=dict(pseudo_rgb=rgb.detach()*0,pseudo_depth=scalar,source_kind=scalar*2,loss_weight=scalar*0,observed_reference_valid=mask&False)
        result=edit_teacher_loss(rgb,scalar,scalar,rgb,scalar,rgb.detach(),rgb,target,mask,mask,~mask)
        result['total'].backward();self.assertEqual(rgb.grad.abs().sum().item(),0)
    def test_observed_labels_cannot_be_overridden(self):
        import torch
        from road_teacher_bridge import edit_teacher_loss
        rgb=torch.ones((1,1,3),requires_grad=True);scalar=torch.ones((1,1));mask=scalar.bool()
        target=dict(pseudo_rgb=rgb.detach()*0,pseudo_depth=scalar,source_kind=scalar*2,loss_weight=scalar,observed_reference_valid=mask)
        result=edit_teacher_loss(rgb,scalar,scalar,rgb,scalar,rgb.detach(),rgb,target,mask,mask,~mask)
        result['total'].backward();self.assertEqual(rgb.grad.abs().sum().item(),0)
if __name__=='__main__':unittest.main()
