"""主运输核的独立距离/协方差及带异常值路面测试。"""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from rigid_patch import RoadFrames, transport_rigid


class RigidPatchTests(unittest.TestCase):
    def test_anisotropic_patch_preserves_pair_distances_and_mahalanobis(self):
        rng=np.random.default_rng(712)
        source=rng.normal(size=(80,3))*[.2,.2,.01]+[2300,1300,-20]
        sigma=np.exp(rng.uniform(-7,-3,size=(80,3)))
        orientation=Rotation.random(80,random_state=rng)
        q=orientation.as_quat()[:,[3,0,1,2]]
        colors=rng.random((80,3)).astype(np.float32);opacity=rng.random(80).astype(np.float32)
        sf=Rotation.from_euler('xyz',[.1,.2,.4]).as_matrix()
        tf=Rotation.from_euler('xyz',[-.15,.3,1.3]).as_matrix()
        target=dict(frame=tf,origin=np.array([4000.,2000.,-8.]))
        m=transport_rigid(source,sigma,q,colors,opacity,np.arange(80),source_origin=source.mean(0),
                          source_frame=sf,target=target,patch_id=9)
        # 距离约束独立于实现内部的assert；曲面法向残差也不能被压平。
        pair=rng.integers(0,80,size=(150,2));a,b=pair.T
        np.testing.assert_allclose(np.linalg.norm(m['world_xyz'][a]-m['world_xyz'][b],axis=1),
                                   np.linalg.norm(source[a]-source[b],axis=1),atol=1e-11,rtol=1e-10)
        np.testing.assert_array_equal(m['reference_srgb'],colors)
        np.testing.assert_array_equal(m['road_opacity'],opacity)
        np.testing.assert_array_equal(m['gaussian_sigma_m'],sigma)
        raw=orientation.as_matrix()
        cov=raw@np.array([np.diag(s*s) for s in sigma])@raw.transpose(0,2,1)
        samples=rng.normal(size=(80,3))*sigma
        sample_world=np.einsum('nij,nj->ni',raw,samples)
        expected=np.sum(samples**2/sigma**2,axis=1)
        R=tf@sf.T;transported=sample_world@R.T
        actual=np.einsum('ni,ni->n',transported,np.linalg.solve(m['covariance_world'],transported[...,None])[...,0])
        np.testing.assert_allclose(actual,expected,atol=2e-10,rtol=2e-10)
        np.testing.assert_allclose(np.linalg.eigvalsh(m['covariance_world']),np.linalg.eigvalsh(cov),atol=1e-15)

    def test_road_frame_rejects_sparse_support_and_ignores_height_outliers(self):
        u,v=np.meshgrid(np.linspace(-.3,.3,9),np.linspace(-.3,.3,9))
        x=u.ravel()+2400;y=v.ravel()+1300
        n=np.array([-.03,.02,1.]);n/=np.linalg.norm(n)
        z=(-20-n[0]*(x-2400)-n[1]*(y-1300))/n[2]
        xyz=np.c_[x,y,z];xyz[:7,2]+=2
        normal=np.tile(n,(len(x),1))
        frames=RoadFrames(xyz,normal,np.ones(len(x),bool),[1.,0.,0.])
        result=frames.at([2400,1300],.6)
        self.assertIsNotNone(result);self.assertLess(result['inlier_height_residual_p95'],1e-10)
        self.assertAlmostEqual(result['origin'][2],-20/n[2],places=9)
        np.testing.assert_allclose(result['frame'].T@result['frame'],np.eye(3),atol=1e-12)
        self.assertIsNone(frames.at([2410,1310],.1))


if __name__=='__main__':unittest.main(verbosity=2)
