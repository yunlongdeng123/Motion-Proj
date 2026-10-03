"""新的deletion请求合同；编辑mask与参考/BEV独立，不再复用foreground生成字段。"""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import torch
from torch.nn import functional as F


@dataclass
class DeletionRequest:
    target_rgb: np.ndarray
    edit_mask: np.ndarray
    alpha: np.ndarray
    priors: dict
    reference_manifest: dict

    def validate(self):
        if self.target_rgb.dtype!=np.uint8 or self.target_rgb.ndim!=4: raise ValueError('目标RGB应为T,H,W,3 uint8')
        if self.target_rgb.shape[:3]!=self.edit_mask.shape or self.edit_mask.dtype!=bool: raise ValueError('必须单独提供二值编辑mask')
        if self.alpha.shape!=self.edit_mask.shape or np.any((self.alpha>0)&~self.edit_mask): raise ValueError('写回范围越出编辑mask')
        if len(self.target_rgb)!=self.priors['geometry'].shape[0]: raise ValueError('帧条件错位')
        if not all(np.isfinite(x).all() for x in self.priors.values()): raise ValueError('条件非有限')
        refs=self.priors['references'];valid=self.priors['reference_valid'].astype(bool)
        if refs.dtype!=np.uint8 or refs.shape[:3]!=valid.shape or refs.shape[-1]!=3: raise ValueError('参考图与有效范围错位')
        if not np.all(refs[~valid]==127): raise ValueError('无效参考像素必须在编码前擦除')

    def masked_target_tensor(self):
        self.validate()
        value=torch.from_numpy(self.target_rgb.copy()).permute(0,3,1,2).float()/127.5-1
        return value.masked_fill(torch.from_numpy(self.edit_mask[:,None]),0)

    def branch_condition(self, reference_latents, device):
        valid=torch.from_numpy(self.priors['reference_valid'][:,None].astype('float32'))
        valid=F.adaptive_avg_pool2d(valid,reference_latents.shape[-2:])>.999
        names=('geometry','bev','bev_rays','reference_pose','query_time','parameters')
        result={k:torch.from_numpy(self.priors[k].copy()).float().to(device) for k in names}
        result.update(reference_latents=reference_latents.detach().float().to(device),reference_valid=valid.float().to(device),
                      hole=torch.from_numpy(self.priors['hole'][:,None].astype('float32')).to(device))
        return result

    def compose(self, generated):
        result=np.rint(generated*self.alpha[...,None]+self.target_rgb*(1-self.alpha[...,None])).clip(0,255).astype('uint8')
        assert np.array_equal(result[~self.edit_mask],self.target_rgb[~self.edit_mask])
        return result


def encode_references(model, request):
    """直接复用官方cond_frames的冻结RGB VAE，不新下载图像backbone。"""
    request.validate()
    embed=next(e for e in model.conditioner.embedders if getattr(e,'input_key',None)=='cond_frames' and hasattr(e,'encoder'))
    if any(p.requires_grad for p in embed.encoder.parameters()): raise RuntimeError('参考VAE必须冻结')
    refs=torch.from_numpy(request.priors['references'].copy()).permute(0,3,1,2).float()/127.5-1
    refs.masked_fill_(torch.from_numpy(~request.priors['reference_valid'].astype(bool))[:,None],0)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        latent=torch.cat([embed.encoder.encode(refs[i:i+1].cuda()) for i in range(len(refs))])*embed.scale_factor
    if latent.shape[:2]!=(len(refs),4): raise ValueError('官方参考VAE未输出4通道latent')
    return latent.detach().float()


def drop_priors(condition, drop_rgb=False, drop_geometry=False):
    """训练与消融使用同一种未知码；保留编辑M和视频相对时间，不关闭学习后的head。"""
    result={k:v.clone() for k,v in condition.items()}
    if drop_rgb:
        for key in ['reference_latents','reference_valid','reference_pose']:result[key].zero_()
    if drop_geometry:
        for key in ['geometry','bev','bev_rays']:result[key].zero_()
        result['geometry'][:,2]=1;result['bev'][:,6]=1
    if drop_rgb and drop_geometry:result['parameters'].zero_()
    return result


def check_request(request):
    request.validate();original=request.masked_target_tensor()
    alternative=request.target_rgb.copy();alternative[request.edit_mask]=255-alternative[request.edit_mask]
    alternate=DeletionRequest(alternative,request.edit_mask,request.alpha,request.priors,request.reference_manifest)
    assert torch.equal(original,alternate.masked_target_tensor()),'隐藏目标RGB漏进条件'
    composite=request.compose(np.full_like(request.target_rgb,173))
    assert np.array_equal(composite[~request.edit_mask],request.target_rgb[~request.edit_mask])
    pending=[r['reference_slot'] for r in request.reference_manifest['references'] if r['GPU_source_mask_validation_pending'] and not r['padding']]
    return {'target_hidden_RGB_mutation_invariant':True,'reference_invalid_pixels_erased':True,
        'outside_edit_mask_pixel_exact':True,'independent_edit_mask':True,'pending_GPU_reference_slots':pending}
