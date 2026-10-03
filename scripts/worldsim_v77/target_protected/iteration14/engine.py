"""重构后的公开deletion接口；先验不再寄存在旧foreground生成参数里。"""
from repair_drive import Engine
from interface import DeletionRequest, encode_references, drop_priors


class MultiPriorDeletionEngine(Engine):
    def __init__(self, branch, **kwargs):
        super().__init__(**kwargs)
        self.model.requires_grad_(False)
        self.branch=branch.cuda().eval()
        self.branch.attach(self.model.model.diffusion_model)
        self._request=None

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
            self.branch.enabled=arm!='baseline'
            if arm!='baseline':
                if reference_latents is None:reference_latents=encode_references(self.model,request)
                condition=request.branch_condition(reference_latents,'cuda')
                self.branch.use_rgb=True;self.branch.use_geometry=True
                condition=drop_priors(condition,drop_rgb=arm in ['null_priors','geometry_only'],
                    drop_geometry=arm in ['null_priors','RGB_only'])
                self.branch.set_condition(**condition)
        if self._request is None:raise RuntimeError('先提交明确的DeletionRequest，禁止复用旧GUI隐式状态')
        return super().get_deletion()
