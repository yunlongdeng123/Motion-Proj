"""Reusable actor-local opacity interface. No scene, frame or edit-angle branches.

Optional experimental calibration; callers keep the native no-edit route.
Geometry, DC and instance ownership are not changed by this module.
"""
import math
import torch

def alpha_supervision(alpha,body,mode='balanced',inside_weight=.05,outside_weight=.05,near_outside=None):
    body=body.to(dtype=alpha.dtype);outside=1-body
    if mode=='legacy':return .1*(alpha-body).abs().mean()
    if mode=='boundary':
        if near_outside is None:raise ValueError('Explicit near-outside mask required')
        near=near_outside.to(dtype=alpha.dtype)*outside;far=outside-near
        return inside_weight*((1-alpha).abs()*body).sum()/body.sum().clamp_min(1)+.5*outside_weight*(alpha.abs()*near).sum()/near.sum().clamp_min(1)+.5*outside_weight*(alpha.abs()*far).sum()/far.sum().clamp_min(1)
    if mode!='balanced':raise ValueError(mode)
    return inside_weight*((1-alpha).abs()*body).sum()/body.sum().clamp_min(1)+outside_weight*(alpha.abs()*outside).sum()/outside.sum().clamp_min(1)

def adaptive_outer_mask(body):
    """Object image-area scale, independent of scene ID, frame ID and edit angle."""
    import numpy as np
    from scipy.ndimage import binary_dilation
    body=np.asarray(body,dtype=bool);radius=int(np.clip(round(.1*np.sqrt(body.sum())),2,12))
    return binary_dilation(body,iterations=radius)&~body,radius

class LocalCalibration(torch.nn.Module):
    def __init__(self,actor,confidence):
        super().__init__()
        self.net=torch.nn.Sequential(torch.nn.Linear(8,32),torch.nn.SiLU(),torch.nn.Linear(32,32),torch.nn.SiLU(),torch.nn.Linear(32,1))
        torch.nn.init.zeros_(self.net[-1].weight);torch.nn.init.zeros_(self.net[-1].bias)
        self.register_buffer('confidence',confidence[:,None]);self.register_buffer('inputs',torch.cat((actor.xyz.detach()/actor.box_half.clamp_min(.1),actor.log_scale.detach(),actor.opacity.detach().sigmoid(),self.confidence),1))
    def factors(self):return torch.exp(math.log(4)*self.confidence*torch.tanh(self.net(self.inputs)))

class QueryActor:
    def __init__(self,actor,calibration=None):self.actor=actor;self.calibration=calibration
    def kwargs(self,*args,**kw):
        d=self.actor.kwargs(*args,**kw)
        if self.calibration is not None:
            o=d['opacities'].clamp(1e-6,1-1e-6);d['opacities']=-torch.expm1(torch.log1p(-o)*self.calibration.factors())
        return d

def attach_saved_field(actor,state):
    """Attach only to matching row space. Cross-asset transfer must rebuild evidence."""
    confidence=state['confidence'].flatten();assert len(confidence)==len(actor.xyz)
    field=LocalCalibration(actor,confidence).to(actor.xyz.device);field.load_state_dict(state)
    expected=torch.cat((actor.xyz.detach()/actor.box_half.clamp_min(.1),actor.log_scale.detach(),actor.opacity.detach().sigmoid(),confidence[:,None]),1)
    assert torch.allclose(field.inputs,expected,atol=1e-7,rtol=1e-7),'Calibration belongs to another actor state'
    return QueryActor(actor,field)
