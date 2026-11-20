"""Small CPU fixtures for world splitting, label isolation and ideal identity."""
import hashlib
import numpy as np
from scipy import ndimage as ndi
from source_world_split import world_partition, strict_core, prepare_frame
from prepare_ideal_world_control import triangle_world, compose_conditional_target

def test_world_split_is_camera_order_and_sign_invariant():
    xyz=np.array([[-.001,0.,0.],[0.,0.,0.],[2.,4.,0.],[2.5,5.5,2.],[-2.,-2.,0.]])
    block,held,_=world_partition(xyz)
    _,reverse,_=world_partition(xyz[::-1])
    assert np.array_equal(held,reverse[::-1])
    assert np.array_equal(block[0],[-1,0]) and np.array_equal(block[1],[0,0])
    for (bx,by),h in zip(block,held):
        d=hashlib.sha256(f'scene065|101|{bx}|{by}'.encode()).digest()
        assert h==(int.from_bytes(d[:8],'big')%5==0)
    assert held[2]==held[3]

def test_sparse_pixel_identity_not_reshape():
    shape=(9,12);ids=np.array([70,15,28,81]);xyz=np.array([[2.5,5.5,0],[0,0,0],[4.5,9.,0],[2.1,4.1,0]])
    hit=np.zeros(np.prod(shape),bool);hit[ids]=True
    maps=dict(pixel_ids=ids,surface_world_xyz=xyz,center_uv=np.array([[1.2,1.3]]*4),surface_hit=hit.reshape(shape))
    masks={r+'_pre_alpha':np.ones(shape,bool) for r in 'ACS'}
    masks.update(K_vehicle=np.zeros(shape,bool),source_material_valid=np.ones(shape,bool))
    ledger={k:np.ones((4,4),bool) for k in ['original_known','original_unknown','base_seed_ledger_mask','teacher_H_allow','old_A0_valid','common_material_valid']}
    ledger['original_source']=np.ones((4,4),np.uint8)
    out,receipt=prepare_frame(maps,masks,np.ones(shape),ledger)
    _,held,_=world_partition(xyz)
    assert np.array_equal(out['world_code'].ravel()[ids],np.where(held,2,1))
    assert out['A_unmapped'].sum()==np.prod(shape)-len(ids)
    assert receipt['domains']['A']['seen']+receipt['domains']['A']['held']==4

def test_held_teacher_perturbation_does_not_change_seen_targets_or_gradient():
    rng=np.random.default_rng(101);shape=(11,19);context=np.ones(shape,bool);context[:,9:]=False
    teacher=rng.random(shape);changed=teacher.copy();changed[~context]=rng.random((~context).sum())*100
    pred=rng.random(shape);core=strict_core(context,3)
    # Linear masked HF matrix, used to check the complete prediction gradient.
    def read(x):
        mass=ndi.gaussian_filter(context.astype(float),1.,truncate=3.,mode='constant')
        clean=np.where(context,x,0.)
        low=ndi.gaussian_filter(clean,1.,truncate=3.,mode='constant')/np.maximum(mass,1e-12)
        return (clean-low)[core]
    A=np.column_stack([read(e.reshape(shape)) for e in np.eye(np.prod(shape))])
    np.testing.assert_array_equal(read(teacher),read(changed))
    np.testing.assert_array_equal(A.T@(read(pred)-read(teacher)),A.T@(read(pred)-read(changed)))
    # Seen source Sobel context must mask first, then operate, not use held RGB.
    for axis in [0,1]:
        a=ndi.sobel(np.where(context,teacher,0.),axis=axis,mode='constant')[strict_core(context,1)]
        b=ndi.sobel(np.where(context,changed,0.),axis=axis,mode='constant')[strict_core(context,1)]
        np.testing.assert_array_equal(a,b)

def test_tiny_world_domain_strict_core_is_na_not_repaired():
    context=np.zeros((7,7),bool);context[3,3]=True
    assert not strict_core(context,1).any()

def test_canonical_triangle_forward_and_local_jacobian():
    y,x=np.indices((4,5));surface=np.stack([2*x+.2*y,3*y+.1*x,.03*x+.02*y],-1)
    uv=np.array([[.8,.2],[.2,.8],[4.,3.],[0.,0.],[-.1,.2]])
    xyz,valid,J=triangle_world(uv,surface,np.ones((4,5),bool))
    expected=np.stack([2*uv[:,0]+.2*uv[:,1],3*uv[:,1]+.1*uv[:,0],.03*uv[:,0]+.02*uv[:,1]],-1)
    np.testing.assert_allclose(xyz[:4],expected[:4],atol=1e-14,rtol=0)
    assert valid.tolist()==[True,True,True,True,False]
    np.testing.assert_allclose(J[0],[[2.,.2],[.1,3.],[.03,.02]],atol=1e-14,rtol=0)

def test_conditional_target_removes_readonly_floor_without_touching_fallback():
    readonly=np.full((3,2,3),.2,np.float32);weight=np.array([[1.,.3,0.],[.6,1.,0.]],np.float32)
    asphalt=np.full_like(readonly,.3);paint=np.full_like(readonly,.8);base=np.full_like(readonly,.47)
    valid=np.array([[1,1,1],[1,0,0]],bool)
    a=compose_conditional_target(readonly,weight,asphalt,valid,base)
    p=compose_conditional_target(readonly,weight,paint,valid,base)
    np.testing.assert_allclose((p-a)[:,valid],(weight[None]*(paint-asphalt))[:,valid],atol=5e-8,rtol=0)
    np.testing.assert_array_equal(p[:,~valid],base[:,~valid])
    np.testing.assert_array_equal(p[:,0,2],readonly[:,0,2])
