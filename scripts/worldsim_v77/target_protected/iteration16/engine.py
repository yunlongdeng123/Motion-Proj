"""复用顺序CFG与官方deletion；新增可选reference patch空间绑定。"""
from iteration15.engine import MultiPriorDeletionEngine as ExistingEngine
from interface import DeletionRequest, encode_references, drop_priors


class MultiPriorDeletionEngine(ExistingEngine):
    def get_deletion(self,request=None,*,arm='RGB_and_geometry',reference_latents=None,appearance_delta=None,routing=True):
        if request is None:return super().get_deletion()
        if not isinstance(request,DeletionRequest):raise TypeError('需要明确DeletionRequest')
        request.validate()
        if request.target_rgb.shape[1:3]!=self.out_size:raise ValueError('query相机分辨率不匹配')
        if arm not in ['baseline','null_priors','RGB_only','geometry_only','RGB_and_geometry']:raise ValueError('未登记消融')
        if any(r['GPU_source_mask_validation_pending'] and not r['padding'] for r in request.reference_manifest['references']):
            raise RuntimeError('额外参考目标mask没有完成验证')
        self._request=request;self.im=list(request.target_rgb);self.masks=list(request.edit_mask)
        self.masked_condition=request.masked_target_tensor();self.previous_segment_last_frame=None;self.im_result=[]
        self.prior_cfg_calls={'unconditional':0,'conditional':0};self.branch.start_response()
        self.branch.enabled=arm!='baseline';self.branch.routing_enabled=routing
        if self.branch.enabled:
            if reference_latents is None:reference_latents=encode_references(self.model,request)
            condition=request.branch_condition(reference_latents,'cuda')
            if appearance_delta is not None:condition['routing_reference_feature_delta']=appearance_delta
            condition=drop_priors(condition,drop_rgb=arm in ['null_priors','geometry_only'],
                drop_geometry=arm in ['null_priors','RGB_only'])
            self.branch.use_rgb=True;self.branch.use_geometry=True
            self.branch.set_condition(**condition)
            self._conditioned_priors=self.branch._condition
            self._unconditional_priors=drop_priors(self._conditioned_priors,True,True,drop_parameters=True)
        # 不再次走旧版本的条件初始化。
        from repair_drive import Engine
        return Engine.get_deletion(self)
