import argparse
import numpy as np
import pytest
import torch
from drivingeditor_dynamic_edit.road_direct_teacher_training import (
    DirectRoadParameters,rgb_tensor,prepare_targets,masked_rgb_l1,stage_for_step,train)


def model():
    return DirectRoadParameters(torch.tensor([[.2,.3,.4],[.6,.7,.8],[.1,.2,.3]]),
        torch.tensor([.99,.99999,.8]),torch.tensor([[.01,.02,.001],[.03,.04,.002],[.05,.06,.003]]),
        [0],[0],[0],[1.15])


def test_dark_integer_rgb_uses_dtype_and_float_range():
    assert torch.equal(rgb_tensor(np.ones((2,4,3),np.uint8),'cpu'),torch.full((3,2,4),1/255))
    assert torch.equal(rgb_tensor(np.ones((2,4,3),np.float32),'cpu'),torch.ones(3,2,4))
    with pytest.raises(ValueError):rgb_tensor(np.full((2,4,3),1.1,np.float32),'cpu')


def test_initial_effective_opacity_exact_and_projection_preserves_frozen_rows():
    m=model()
    assert torch.equal(m.full_opacity()[:,0],m.base_opacity)
    with torch.no_grad():m.colors.fill_(3);m.opacity_logits.fill_(100);m.scale_factor.fill_(10)
    m.project()
    assert torch.all(m.colors==1)
    assert float(m.full_opacity()[0])<=1-1e-5+1e-8
    assert torch.equal(m.full_opacity()[1:,0],m.base_opacity[1:])
    assert torch.equal(m.full_scales()[1:],m.base_scales[1:])
    assert torch.equal(m.full_scales()[:,2],m.base_scales[:,2])
    assert float(m.scale_factor[0])<=1.15


def test_packet_detaches_entire_photo_coverage_parameter_paths():
    m=model();parameters=tuple(m.parameters())
    p=m.packet({},'photo');loss=(p['colors_precomp']*p['opacities']*p['scales']).sum()
    g=torch.autograd.grad(loss,parameters,allow_unused=True)
    assert g[0].abs().max()>0 and g[1] is None and g[2] is None
    p=m.packet({},'coverage');loss=(p['colors_precomp']*p['opacities']*p['scales']).sum()
    g=torch.autograd.grad(loss,parameters,allow_unused=True)
    assert g[0] is None and g[1].abs().max()>0 and g[2].abs().max()>0


def test_low_alpha_teacher_only_pixel_is_coverage_and_never_outside():
    zeros=np.zeros((2,4),bool);coverage=zeros.copy();coverage[0,0]=True
    data=dict(A_mask=zeros,C_mask=zeros,coverage_A=zeros,coverage_C=coverage,K=zeros,
        q_A=zeros.astype(np.float32),q_C=zeros.astype(np.float32),
        q_A_coverage=zeros.astype(np.float32),q_C_coverage=coverage.astype(np.float32),
        q_A_native_footprint=zeros.astype(np.float32),q_C_native_footprint=coverage.astype(np.float32),
        target_A_srgb=np.zeros((2,4,3),np.float32),target_C_srgb=np.ones((2,4,3),np.float32),
        F_control_C_srgb=np.zeros((2,4,3),np.float32),F_control_valid=zeros,
        fixed_initial_alpha=np.full((2,4),.2,np.float32))
    a=prepare_targets(data,'cpu','direct_C');b=prepare_targets(data,'cpu','matched_F_control')
    assert a['coverage_C'][0,0] and not a['outside'][0,0] and not a['photo_C'][0,0]
    assert a['coverage_q_C'][0,0]==1 and a['q'][0,0]==0
    for key in ('coverage_C','coverage_q_C','outside','photo_C','q'):assert torch.equal(a[key],b[key])
    assert not torch.equal(a['selected_C_srgb'],b['selected_C_srgb'])


def test_masks_normalize_each_domain_independently():
    p=torch.ones(3,2,4);target=torch.zeros_like(p)
    one=torch.zeros(2,4,dtype=torch.bool);one[0,0]=True
    assert masked_rgb_l1(p,target,one)==1
    assert masked_rgb_l1(p,target,torch.ones_like(one))==1


def test_stage_budget_and_formal_authorization(tmp_path):
    assert [stage_for_step(x,(2,2,2)) for x in (1,2,3,4,5,6)]==['color','color','coverage','coverage','joint','joint']
    with pytest.raises(ValueError):stage_for_step(7,(2,2,2))
    protocol=tmp_path/'protocol.json';protocol.write_text('{"formal_authorized": false}')
    with pytest.raises(ValueError,match='尚未授权'):
        train(argparse.Namespace(schedule='200,300,300',mode='formal',protocol=protocol))
