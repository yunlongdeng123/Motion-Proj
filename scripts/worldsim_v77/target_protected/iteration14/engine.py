"""重构后的公开deletion接口；先验不再寄存在旧foreground生成参数里。"""
from repair_drive import Engine
from interface import DeletionRequest, encode_references, drop_priors
import torch


class MultiPriorDeletionEngine(Engine):
    def __init__(self, branch, **kwargs):
        super().__init__(**kwargs)
        self.model.requires_grad_(False)
        self.branch=branch.cuda().eval()
        self.branch.attach(self.model.model.diffusion_model)
        self._request=None
        self._conditioned_priors=None
        self._unconditional_priors=None
        self.prior_cfg_calls={'unconditional':0,'conditional':0}
        self.model.denoiser.register_forward_pre_hook(self._route_prior_cfg,with_kwargs=True)

    def _route_prior_cfg(self, module, args, kwargs):
        if not self.branch.enabled:return
        # 官方get_unconditional_conditioning强制UC crossattn为0；顺序CFG每次调用一支。
        cond=kwargs.get('cond',args[5] if len(args)>5 else None)
        if cond is None or 'crossattn' not in cond:raise RuntimeError('CFG接口缺少官方条件字典')
        context=cond['crossattn']
        if context.shape[0]!=len(self._request.target_rgb):raise RuntimeError('新条件仅支持已验证的顺序CFG')
        unconditional=not bool(torch.count_nonzero(context))
        self.branch._condition=self._unconditional_priors if unconditional else self._conditioned_priors
        self.prior_cfg_calls['unconditional' if unconditional else 'conditional']+=1

    def get_deletion(self, request=None, *, arm='RGB_and_geometry', reference_latents=None):
        """兼容官方predict内部无参数调用，外部统一接收独立mask和多先验request。"""
        if request is not None:
            if not isinstance(request,DeletionRequest):raise TypeError('必须提供DeletionRequest')
            request.validate()
            if request.target_rgb.shape[1:3]!=self.out_size:raise ValueError('查询图像与相机几何分辨率不一致')
            if arm not in ['baseline','null_priors','RGB_only','geometry_only','RGB_and_geometry']:
                raise ValueError('未登记条件消融')
            if arm!='baseline' and any(r['GPU_source_mask_validation_pending'] and not r['padding']
                for r in request.reference_manifest['references']):raise RuntimeError('额外参考目标剔除尚未验证')
            self._request=request
            self.im=list(request.target_rgb);self.masks=list(request.edit_mask)
            self.masked_condition=request.masked_target_tensor()
            self.previous_segment_last_frame=None;self.im_result=[]
            self.prior_cfg_calls={'unconditional':0,'conditional':0}
            self.branch.enabled=arm!='baseline'
            if arm!='baseline':
                if reference_latents is None:reference_latents=encode_references(self.model,request)
                condition=request.branch_condition(reference_latents,'cuda')
                self.branch.use_rgb=True;self.branch.use_geometry=True
                condition=drop_priors(condition,drop_rgb=arm in ['null_priors','geometry_only'],
                    drop_geometry=arm in ['null_priors','RGB_only'])
                self.branch.set_condition(**condition)
                self._conditioned_priors=self.branch._condition
                self._unconditional_priors=drop_priors(self._conditioned_priors,True,True)
        if self._request is None:raise RuntimeError('先提交明确的DeletionRequest，禁止复用旧GUI隐式状态')
        return super().get_deletion()
