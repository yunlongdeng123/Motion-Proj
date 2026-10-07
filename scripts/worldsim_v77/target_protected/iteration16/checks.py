"""空间/信息合同的必要检查；不以CPU激活替代真实生成验收。"""
import numpy as np
import torch
from routing import observation, first_owner, letterbox_coordinates, attention_bias, pooled_owner
from multi_prior import MultiPriorBranch


def checks(checkpoint):
    from safetensors.torch import load_file
    from iteration14.multi_prior import MultiPriorBranch as Original
    torch.set_num_threads(1)
    torch.manual_seed(42)
    branch = MultiPriorBranch();original = Original()
    state = load_file(str(checkpoint));branch.load_state_dict(state,strict=True);original.load_state_dict(state,strict=True)
    assert branch.state_dict().keys()==original.state_dict().keys()
    assert sum(p.numel() for p in branch.parameters())==389856
    t = 2
    base = dict(geometry=torch.rand(t,12,16,24),bev=torch.rand(t,10,24,24),bev_rays=torch.rand(t,4,8,12,2)*2-1,
        reference_latents=torch.rand(6,4,32,32),reference_valid=torch.ones(6,1,32,32),
        reference_pose=torch.rand(6,13),query_time=torch.tensor([[0.],[.9]]),
        parameters=torch.ones(t,4),hole=torch.ones(t,1,16,24))
    route = dict(routing_query_owner=torch.ones(t,1,16,24,dtype=torch.long),
        routing_query_q=torch.ones(t,1,16,24)*.5,routing_reference_owner=torch.ones(6,8,8,dtype=torch.long),
        routing_reference_q=torch.ones(6,8,8)*.5,routing_projected_uv=torch.ones(t,384,2)*.5,
        routing_projected_radius=torch.zeros(t,384,2),routing_projected_q=torch.ones(t,384)*.5)
    route['routing_reference_owner'][3:] = 2
    activation = torch.randn(t,320,16,24)
    original.set_condition(**base);branch.set_condition(**(base|route))
    branch.routing_enabled = False
    with torch.no_grad():
        expected = original.residual(0,activation);disabled = branch.residual(0,activation)
    assert torch.equal(expected,disabled), '关闭绑定必须逐元素重现r47原路径'
    for index,channels in enumerate(branch.channels[1:],1):
        x=torch.randn(t,channels,8,12)
        with torch.no_grad():assert torch.equal(original.residual(index,x),branch.residual(index,x))
    branch.routing_enabled = True
    zero = {k:torch.zeros_like(v) for k,v in route.items()}
    branch.set_condition(**(base|zero))
    with torch.no_grad():assert torch.equal(expected,branch.residual(0,activation))
    branch.set_condition(**(base|route))
    changed = branch.residual(0,activation)
    assert not torch.equal(expected,changed), '非零绑定没有到达实际RGB分支'
    branch.zero_grad();changed.square().mean().backward()
    for family in ('reference_encoder.0.weight','attention.0.in_proj_weight','queries.0.weight'):
        p = dict(branch.named_parameters())[family]
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
    for size in ((18,32),(10,18),(5,9)):
        bias,owner,q = attention_bias(route,size)
        assert bias.shape==(t,size[0]*size[1],385) and torch.isfinite(bias).all()
        assert torch.all(bias[...,192:384]<0) and torch.all(bias[...,-1]==0)
        neutral = route|{'routing_query_owner':torch.zeros_like(route['routing_query_owner'])}
        assert torch.count_nonzero(attention_bias(neutral,size)[0])==0
        unknown_refs = route|{'routing_reference_owner':torch.zeros_like(route['routing_reference_owner'])}
        assert torch.count_nonzero(attention_bias(unknown_refs,size)[0])==0
        relabel=route.copy();relabel['routing_query_owner']=route['routing_query_owner']*999
        relabel['routing_reference_owner']=torch.where(route['routing_reference_owner']==1,999,7)
        assert torch.equal(bias,attention_bias(relabel,size)[0]), '身份ID只应是匹配索引'
    # 错配不能改变query或参考pose，只改指定外观token；null/其他token不受影响。
    delta=torch.zeros(384,branch.width);delta[0]=.5
    branch.set_condition(**(base|route|{'routing_reference_feature_delta':delta}))
    branch._query_size=(16,24)
    query=torch.randn(t,384,branch.width);tokens=torch.randn(t,385,branch.width)
    routed,kwargs=branch._route_attention(0,(query,tokens,tokens),
        {'key_padding_mask':torch.zeros(t,385,dtype=torch.bool),'need_weights':False})
    assert torch.equal(routed[0],query) and torch.equal(routed[1][:,1:],tokens[:,1:])
    assert torch.allclose(routed[1][:,0],tokens[:,0]+.5)
    assert torch.equal(branch._condition['reference_pose'],base['reference_pose'])
    mixed = torch.ones(1,1,4,4,dtype=torch.long);mixed[:,:,:,2:] = 2
    assert not torch.count_nonzero(pooled_owner(mixed,torch.ones_like(mixed).float(),(1,1))[0])
    frame = {'timestamp':0,'intrinsics_1024':[[128,0,512],[0,128,288],[0,0,1]],'camera_to_world':np.eye(4).tolist()}
    obs = observation(frame)
    actors = [{'instance_token':'A','category':'vehicle.car','translation':[0,0,4],
               'rotation':[1,0,0,0],'size':[4,4,4]},
              {'instance_token':'B','category':'vehicle.car','translation':[0,0,8],
               'rotation':[1,0,0,0],'size':[2,4,2]}]
    source,_,_ = first_owner(obs,actors,{'B':1},'A')
    deleted,_,_ = first_owner(obs,actors,{'B':1},'A',True)
    assert source[72,128]==0 and deleted[72,128]==1, 'A未删除/已删除的first-return顺序错误'
    info = {'letterbox':{'roi_xyxy':[103,17,842,466],'padding_xy':[0,50],'scale':256/739}}
    uv,inside = letterbox_coordinates(info)
    height = round(449*256/739)
    # 任意有效参考像素→原图→参考像素中心，使用真实round尺寸。
    yy,xx = np.where(inside);p = uv[yy,xx]*4
    back = np.c_[(p[:,0]-103)*256/739,(p[:,1]-17)*height/449+50]
    assert np.max(abs(back-np.c_[xx+.5,yy+.5]))<1e-5
    return {'r47_checkpoint_strict_load':True,'parameters':389856,'new_trainable_parameters':0,
        'routing_off_exact_original':True,'all_unknown_exact_original':True,
        'four_injection_blocks_off_exact':True,'identity_ID_relabel_invariant':True,
        'appearance_mismatch_changes_selected_tokens_only':True,
        'binding_reaches_RGB_and_gradients_finite':True,'three_query_scales_verified':True,
        'conflicting_identity_stays_unknown':True,'unknown_and_null_logit_bias_zero':True,
        'first_return_after_target_delete':True,'rounded_letterbox_inverse_verified':True,
        'scope':'CPU输入与受控激活；没有官方模型生成/反向传播或视觉收益结论'}
