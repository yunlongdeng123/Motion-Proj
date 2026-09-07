"""可训练原生DPT多尺度特征；仅缓存完全冻结的aggregator前缀。"""
from pathlib import Path
import sys

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class NativeGeometryPyramid(nn.Module):
    def __init__(self,native_run,scene,view_indices,head=None,token_device='cuda'):
        super().__init__()
        sys.path.insert(0,'/root/autodl-tmp/external/worldsim_v72/vggt')
        from vggt.heads.dpt_head import DPTHead
        self.head=head
        if self.head is None:
            self.head=DPTHead(dim_in=2048,output_dim=2,activation='exp',conf_activation='expp1').cuda()
            state=torch.load(Path(native_run)/'latest.pt',map_location='cpu',weights_only=True)
            self.head.load_state_dict(state['depth_head'])
        self.token_inputs=[]
        for i in view_indices:
            data=torch.load(Path(native_run)/'frozen_prefix'/f'{scene["scene_id"]}_{i:02}.pt',weights_only=True)
            self.token_inputs.append((tuple(data['tokens'][j].to(token_device) for j in [4,11,17,23]),
                        scene['views'][i]['image'][None,None].to(token_device),data['patch_start']))

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
