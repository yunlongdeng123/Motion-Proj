import numpy as np
from .material_evaluation import masked_hf,detail_stats,packet_metrics
from .crossview import sample

def test_mask_normalization_ignores_arbitrary_outside_K_color():
    mask=np.zeros((32,40),bool);mask[4:28,6:34]=True
    p=np.full((3,32,40),.37);q=p.copy();q[:,~mask]=1000.
    a,v,_=masked_hf(p,mask,2.);b,w,_=masked_hf(q,mask,2.)
    assert np.array_equal(v,w)and np.max(np.abs(a[:,v]))<1e-14
    assert np.max(np.abs(a[:,v]-b[:,v]))==0

def test_true_teacher_edge_matches_and_noise_is_not_fake_recall():
    mask=np.ones((48,48),bool);T=np.full((3,48,48),.3);T[:,:,24:]=.7
    good=detail_stats(T,T,mask)
    y,x=np.indices(mask.shape);bad=T+.03*((x+y)%2*2-1)[None]
    noisy=detail_stats(bad,T,mask)
    assert good['edges']['F1']==1 and good['scales']['1px']['HF_MAE']==0
    assert noisy['edges']['spurious_HF_RMS_on_teacher_flat']>good['edges']['spurious_HF_RMS_on_teacher_flat']+.01

def test_subpixel_actual_image_bilinear_has_known_affine_answer():
    y,x=np.indices((12,17));im=np.stack([x+2*y,3*x-y,x-y]).astype(float)
    uv=np.array([[2.25,3.75],[11.7,8.2]])
    expected=np.stack([uv[:,0]+2*uv[:,1],3*uv[:,0]-uv[:,1],uv[:,0]-uv[:,1]])
    assert np.max(np.abs(sample(im,uv)-expected))<1e-14

def test_joint_K_and_both_alpha_are_independent_hard_checks():
    h,w=320,480;D=np.ones((h,w),bool);K=np.zeros((h,w),bool);K[1,2]=True
    P=np.full((3,h,w),.4,np.float32);alpha=np.ones((h,w),np.float32)
    f=dict(baseline_road_rgb=P,baseline_joint_rgb=P,baseline_road_alpha=alpha,baseline_joint_alpha=alpha,K=K,D=D,
        stripe_domain=D,stripe_vertical=np.ones((h-2,w),bool),stripe_horizontal=np.ones((h,w-2),bool))
    packet=dict(road_rgb=P.copy(),joint_rgb=P.copy(),road_alpha=alpha.copy(),joint_alpha=alpha.copy(),K=K)
    assert all(packet_metrics(packet,f)['checks'].values())
    packet['road_rgb'][:,1,2]+=.2
    assert packet_metrics(packet,f)['checks']['joint_K_exact']
    packet['joint_rgb'][:,1,2]+=.01
    assert not packet_metrics(packet,f)['checks']['joint_K_exact']
    packet['joint_alpha'][0,0]=.9
    assert not packet_metrics(packet,f)['checks']['joint_alpha_exact']
