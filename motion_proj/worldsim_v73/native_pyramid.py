"""可训练原生DPT多尺度特征；仅缓存完全冻结的aggregator前缀。"""
from pathlib import Path
import sys

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class NativeGeometryPyramid(nn.Module):
    def __init__(self,native_run,scene,view_indices,head=None,token_device='cuda',prefix_cache=None):
        super().__init__()
        sys.path.insert(0,'/root/autodl-tmp/external/worldsim_v72/vggt')
        from vggt.heads.dpt_head import DPTHead
        self.head=head
        if self.head is None:
            self.head=DPTHead(dim_in=2048,output_dim=2,activation='exp',conf_activation='expp1').cuda()
            state=torch.load(Path(native_run)/'latest.pt',map_location='cpu',weights_only=True)
            self.head.load_state_dict(state['depth_head'])
        self.token_inputs=[]
        self.view_keys=[]
        for i in view_indices:
            key=(str(native_run),scene['scene_id'],i,str(token_device))
            packed=prefix_cache.get(key) if prefix_cache is not None else None
            if packed is None:
                data=torch.load(Path(native_run)/'frozen_prefix'/f'{scene["scene_id"]}_{i:02}.pt',map_location='cpu',weights_only=True,mmap=True)
                packed=(tuple(data['tokens'][j].to(token_device) for j in [4,11,17,23]),
                        scene['views'][i]['image'][None,None].to(token_device),data['patch_start'])
                # 同一窗口多个Actor共享不可变前缀；绝不缓存跨优化步的可训练DPT输出。
                if prefix_cache is not None: prefix_cache[key]=packed
            self.token_inputs.append(packed)
            self.view_keys.append(key)

    def one_view(self,*inputs):
        tokens=[None]*24
        for j,t in zip([4,11,17,23],inputs[:4]): tokens[j]=t
        captured={}
        handles=[]
        for level,name in enumerate(['refinenet4','refinenet3','refinenet2','refinenet1']):
            def hook(module,values,result,key=level): captured[key]=result
            handles.append(getattr(self.head.scratch,name).register_forward_hook(hook))
        try:
            with torch.autocast('cuda',dtype=torch.bfloat16):
                depth,_=self.head(tokens,inputs[4],inputs[5])
        finally:
            for handle in handles: handle.remove()
        return (*tuple(captured[j] for j in range(4)),depth[0,...,0].float())

    def forward(self,include_depth=False):
        features=[[],[],[],[]]
        depths=[]
        for tokens,image,patch_start in self.token_inputs:
            # 共享训练的冻结前缀存CPU，逐样本移到GPU；保留完整视图和反向重算输入。
            device=next(self.head.parameters()).device
            maps=checkpoint(self.one_view,*(t.to(device) for t in tokens),image.to(device),patch_start,use_reentrant=False)
            for level,feature in enumerate(maps[:4]): features[level].append(feature)
            if include_depth: depths.append(maps[4])
        features=[torch.cat(level) for level in features]
        return (features,torch.cat(depths)) if include_depth else features

    def one_depth(self,*inputs):
        tokens=[None]*24
        for j,t in zip([4,11,17,23],inputs[:4]): tokens[j]=t
        with torch.autocast('cuda',dtype=torch.bfloat16):
            depth,_=self.head(tokens,inputs[4],inputs[5])
        return depth[0,...,0].float()

    def depth_only(self,output_cache=None):
        # 原生保守控制不消费query特征，避免保存/拼接无下游用途的多尺度副输出。
        # 原生DPT内部多层解码照常；每步重算，不缓存可训练输出。
        device=next(self.head.parameters()).device
        if output_cache is not None:
            # 调用者仅在一次固定权重的no-grad评价中共享，训练入口从不传此缓存。
            output=[]
            for key,(tokens,image,patch_start) in zip(self.view_keys,self.token_inputs):
                if key not in output_cache:
                    output_cache[key]=self.one_depth(*(t.to(device) for t in tokens),image.to(device),patch_start).detach().cpu()
                output.append(output_cache[key].to(device))
            return torch.cat(output)
        return torch.cat([checkpoint(self.one_depth,*(t.to(device) for t in tokens),image.to(device),
                                     patch_start,use_reentrant=False)
                          for tokens,image,patch_start in self.token_inputs])
