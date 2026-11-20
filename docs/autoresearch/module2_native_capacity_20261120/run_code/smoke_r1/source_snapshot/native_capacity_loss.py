"""Common fixed-mask quadratic native-image objective; no native import or optimization.

The objective is convex in the actual native Gaussian RGB vector if its renderer is
affine. World-field parameterizations need not be convex. All normalizers come from
the SAME frozen initial render and the current training split, never held labels.
"""
from dataclasses import dataclass
import copy, hashlib, json
import numpy as np
from scipy import ndimage as ndi
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossSpec:
    scales: tuple = (1., 2., 4.)
    truncate: float = 3.
    quantization_floor_squared: float = (1/255)**2
    source_weights: tuple = (('A',1.),('C',.5),('S',1.))
    stable_alpha_min: float = .95
    hf_weight: float = 1.
    sobel_weight: float = 1.


def tensor_sha(x):
    a=x.detach().cpu().contiguous().numpy()
    return hashlib.sha256(str((a.shape,a.dtype)).encode()+a.tobytes()).hexdigest()


def strict_core(mask,radius):
    a=np.asarray(mask,bool)
    return ndi.minimum_filter(a.astype(np.uint8),size=2*radius+1,mode='constant',cval=0)>0


def masked_gaussian(x,mask,sigma,truncate=3.,mass=None):
    """CHW signed signal; field is zero outside mask BEFORE either convolution."""
    radius=int(truncate*sigma+.5)
    t=torch.arange(-radius,radius+1,device=x.device,dtype=x.dtype)
    k=torch.exp(-.5*(t/sigma)**2);k=k/k.sum();c=x.shape[0]
    clean=torch.where(mask[None],x,torch.zeros_like(x))[None]
    a=F.conv2d(clean,k[None,None,None,:].expand(c,1,1,-1),padding=(0,radius),groups=c)
    a=F.conv2d(a,k[None,None,:,None].expand(c,1,-1,1),padding=(radius,0),groups=c)[0]
    if mass is None:
        m=mask.to(x.dtype)[None,None]
        m=F.conv2d(m,k[None,None,None,:],padding=(0,radius))
        mass=F.conv2d(m,k[None,None,:,None],padding=(radius,0))[0,0]
    return a/mass.clamp_min(1e-12)[None],mass


def signed_sobel(x,mask):
    k=x.new_tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]])/8.
    clean=torch.where(mask[None],x,torch.zeros_like(x))[None]
    c=x.shape[0]
    gx=F.conv2d(clean,k[None,None].expand(c,1,3,3),padding=1,groups=c)[0]
    gy=F.conv2d(clean,k.T[None,None].expand(c,1,3,3),padding=1,groups=c)[0]
    return torch.stack((gx,gy),0)


class RegionQuadratic:
    def __init__(self,target,alpha,context,score,spec,device):
        self.spec=spec;dtype=torch.float32
        context=np.asarray(context,bool);score=np.asarray(score,bool)
        self.full_shape=(3,*context.shape)
        if np.asarray(target).shape!=self.full_shape or np.asarray(alpha).shape!=context.shape or score.shape!=context.shape:
            raise ValueError('native objective CHW/HW shape mismatch')
        if np.any(context&(np.asarray(alpha)<spec.stable_alpha_min)):
            raise ValueError('context contains unstable alpha')
        if not np.isfinite(np.asarray(target)[:,context]).all():raise ValueError('nonfinite legal teacher')
        full_counts=dict(score=int(score.sum()),context=int(context.sum()),score_outside_context=int((score&~context).sum()))
        yy,xx=np.nonzero(context)
        self.bounds=(int(yy.min()),int(yy.max())+1,int(xx.min()),int(xx.max())+1) if len(yy) else (0,1,0,1)
        y0,y1,x0,x1=self.bounds;sl=np.s_[y0:y1,x0:x1]
        self.target=torch.as_tensor(np.asarray(target)[:,sl[0],sl[1]].copy(),device=device,dtype=dtype)
        a=torch.as_tensor(np.asarray(alpha)[sl].copy(),device=device,dtype=dtype)
        context=context[sl].copy();score=score[sl].copy()
        self.context=torch.as_tensor(context.copy(),device=device)
        self.score=torch.as_tensor((context&score).copy(),device=device)
        self.alpha=a.detach()
        # Fixed weights, <=1/.95; never divide an opacity optimization variable.
        self.inv_alpha=(a.clamp_min(spec.stable_alpha_min).reciprocal()*self.context).detach()
        self.teacher_P=torch.where(self.context[None],self.target*a[None],torch.zeros_like(self.target)).detach()
        self.core={};self.mass={}
        for sigma in spec.scales:
            key='HF_'+str(float(sigma))
            self.core[key]=torch.as_tensor(strict_core(context,int(spec.truncate*sigma+.5))&score,device=device)
            with torch.no_grad():
                _,mass=masked_gaussian(self.teacher_P,self.context,sigma,spec.truncate)
            self.mass[key]=mass.detach()
        self.sobel_core=torch.as_tensor(strict_core(context,1)&score,device=device)
        self.counts=dict(**full_counts,rgb=int(self.score.sum()),
            HF={key:int(value.sum())for key,value in self.core.items()},sobel=int(self.sobel_core.sum()),
            evaluation_bbox_y0_y1_x0_x1=list(self.bounds),full_image_shape=list(self.full_shape),
            crop_is_exact_zero_context_acceleration=True)

    @staticmethod
    def square_mean(x,mask):
        n=int(mask.sum())
        if not n:return x.sum()*0.
        return (x[...,mask]**2).mean()

    def terms(self,P,mode='detail'):
        if tuple(P.shape)!=self.full_shape:raise ValueError('native prediction shape mismatch')
        y0,y1,x0,x1=self.bounds;P=P[:,y0:y1,x0:x1]
        E=torch.where(self.context[None],(P-self.teacher_P)*self.inv_alpha[None],torch.zeros_like(P))
        result=dict(RGB=self.square_mean(E,self.score))
        if mode=='rgb':return result
        if mode!='detail':raise ValueError('mode must be rgb/detail')
        for sigma in self.spec.scales:
            key='HF_'+str(float(sigma))
            low,_=masked_gaussian(E,self.context,sigma,self.spec.truncate,mass=self.mass[key])
            result[key]=self.square_mean(E-low,self.core[key])
        result['SOBEL']=self.square_mean(signed_sobel(E,self.context),self.sobel_core)
        return result


class NativeCapacityLoss:
    """frames[raw]={alpha, targets{A/C/S}, contexts{A/C/S}, scores{A/C/S}}."""
    def __init__(self,frames,*,spec=LossSpec(),device='cpu'):
        self.spec=spec;self.device=torch.device(device);self.regions={}
        for raw,frame in frames.items():
            self.regions[int(raw)]={name:RegionQuadratic(frame['targets'][name],frame['alpha'],
                frame['contexts'][name],frame['scores'][name],spec,self.device)for name,_ in spec.source_weights}
        if not self.regions:raise ValueError('empty objective frames')
        self.normalizers=None
        self.signature={str(raw)+':'+name:dict(context=tensor_sha(r.context),score=tensor_sha(r.score),
            teacherP=tensor_sha(r.teacher_P),inv_alpha=tensor_sha(r.inv_alpha),counts=r.counts)
            for raw,regions in self.regions.items()for name,r in regions.items()}

    def calibrate(self,initial_predictions):
        if self.normalizers is not None:raise RuntimeError('normalizers are one-shot frozen')
        d={}
        with torch.no_grad():
            for raw,regions in self.regions.items():
                for name,r in regions.items():
                    values=r.terms(initial_predictions[raw],'detail')
                    d[str(raw)+':'+name]={key:max(float(value),self.spec.quantization_floor_squared)
                                        for key,value in values.items()}
        self.normalizers=d
        return self.snapshot()

    def objective(self,predictions,mode='detail'):
        if self.normalizers is None:raise RuntimeError('calibrate SAME initial predictions before use')
        terms={};total=None
        for raw,regions in self.regions.items():
            for name,weight in self.spec.source_weights:
                values=regions[name].terms(predictions[raw],mode)
                norms=self.normalizers[str(raw)+':'+name]
                value=values['RGB']/norms['RGB']
                if mode=='detail':
                    h=sum(values['HF_'+str(float(s))]/norms['HF_'+str(float(s))]for s in self.spec.scales)/len(self.spec.scales)
                    value=value+self.spec.hf_weight*h+self.spec.sobel_weight*values['SOBEL']/norms['SOBEL']
                weighted=weight*value/len(self.regions)
                total=weighted if total is None else total+weighted
                terms[str(raw)+':'+name]=dict(raw_terms=values,normalized_total=weighted)
        return total,terms

    def snapshot(self):
        return dict(schema='native_capacity_quadratic_loss.v1',spec=dict(self.spec.__dict__),
            signature=copy.deepcopy(self.signature),normalizers=copy.deepcopy(self.normalizers),
            convexity='affine native preclamp P in bounded ci, fixed masks/weights; field parameterization itself may be nonconvex',
            teacher_label_holdout_policy='only masked seen-context labels enter targets, normalizers or loss; original prior exposure is outside this claim')

    def restore(self,state):
        if state['schema']!='native_capacity_quadratic_loss.v1'or state['spec']!=dict(self.spec.__dict__)or state['signature']!=self.signature:
            raise ValueError('objective frozen identity changed')
        d=state['normalizers']
        if d is None or any(not np.isfinite(v)or v<self.spec.quantization_floor_squared for row in d.values()for v in row.values()):
            raise ValueError('invalid frozen normalization')
        self.normalizers=copy.deepcopy(d)
