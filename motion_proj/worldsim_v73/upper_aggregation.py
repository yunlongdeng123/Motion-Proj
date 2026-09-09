"""从第17组冻结状态重算VGGT最后6组；不缓存跨优化步的适配结果。"""
from contextlib import nullcontext
import math
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .native_pyramid import NativeGeometryPyramid


class LoRAProjection(nn.Module):
    def __init__(self,base,rank=8,alpha=8):
        super().__init__()
        self.base=base.requires_grad_(False)
        self.lora_A=nn.Parameter(base.weight.new_empty(rank,base.in_features))
        self.lora_B=nn.Parameter(base.weight.new_zeros(base.out_features,rank))
        self.scaling=alpha/rank
        nn.init.kaiming_uniform_(self.lora_A,a=math.sqrt(5))

    def forward(self,x):
        # 冻结权重仍需传递输入梯度；不能把base(x)包进no_grad。
        return self.base(x)+F.linear(F.linear(x,self.lora_A),self.lora_B)*self.scaling


class VGGTUpperTail(nn.Module):
    """固定18–23组接口；图像编码与0–17组不在此模块中。"""
    def __init__(self,weights_file=None,rank=8,alpha=8,adapt_projection=False,
                 embed_dim=1024,num_heads=16,vggt_root='/root/autodl-tmp/external/worldsim_v72/vggt'):
        super().__init__()
        sys.path.insert(0,str(vggt_root))
        from vggt.layers.block import Block
        from vggt.layers.rope import RotaryPositionEmbedding2D
        from safetensors import safe_open
        self.embed_dim=embed_dim
        self.config={'start_layer':18,'end_layer':23,'rank':rank,'alpha':alpha,
                     'embed_dim':embed_dim,'num_heads':num_heads,
                     'targets':['qkv','proj'] if adapt_projection else ['qkv'],
                     'weights_file':str(weights_file) if weights_file else None}
        rope=RotaryPositionEmbedding2D(frequency=100)
        self.frame_blocks=nn.ModuleList(); self.global_blocks=nn.ModuleList()
        context=safe_open(str(weights_file),framework='pt',device='cpu') if weights_file else nullcontext()
        with context as reader:
            keys=list(reader.keys()) if reader is not None else []
            for kind in ['frame','global']:
                for layer in range(18,24):
                    block=Block(dim=embed_dim,num_heads=num_heads,mlp_ratio=4,qkv_bias=True,
                                proj_bias=True,ffn_bias=True,init_values=.01,qk_norm=True,rope=rope)
                    if reader is not None:
                        prefix=f'aggregator.{kind}_blocks.{layer}.'
                        block.load_state_dict({key[len(prefix):]:reader.get_tensor(key)
                                               for key in keys if key.startswith(prefix)})
                    block.requires_grad_(False)
                    block.attn.qkv=LoRAProjection(block.attn.qkv,rank,alpha)
                    if adapt_projection: block.attn.proj=LoRAProjection(block.attn.proj,rank,alpha)
                    getattr(self,kind+'_blocks').append(block)

    @staticmethod
    def positions(batch_views,image_hw,patch_start,device):
        height,width=image_hw
        grid=torch.cartesian_prod(torch.arange(height//14,device=device),
                                 torch.arange(width//14,device=device))+1
        special=torch.zeros(patch_start,2,dtype=grid.dtype,device=device)
        return torch.cat([special,grid])[None].expand(batch_views,-1,-1).clone()

    def forward(self,layer17,image_hw,patch_start=5):
        batch,views,count,_=layer17.shape
        width=self.embed_dim
        # 第17组拼接后半是global状态；全窗口顺序与特殊token已经包含在缓存中。
        x=layer17[...,width:]
        pos=self.positions(batch*views,image_hw,patch_start,x.device)
        global_pos=pos.reshape(batch,views*count,2)
        for frame_block,global_block in zip(self.frame_blocks,self.global_blocks):
            x=x.reshape(batch*views,count,width)
            if self.training and torch.is_grad_enabled():
                frame=checkpoint(frame_block,x,pos,use_reentrant=False)
                x=checkpoint(global_block,frame.reshape(batch,views*count,width),global_pos,use_reentrant=False)
            else:
                frame=frame_block(x,pos=pos)
                x=global_block(frame.reshape(batch,views*count,width),pos=global_pos)
        return torch.cat([frame.reshape(batch,views,count,width),x.reshape(batch,views,count,width)],dim=-1)

    def adapter_state_dict(self):
        return {name:p.detach().cpu().clone() for name,p in self.named_parameters() if p.requires_grad}

    def load_adapter_state_dict(self,state):
        with torch.no_grad():
            for name,p in self.named_parameters():
                if p.requires_grad: p.copy_(state[name])

    @classmethod
    def from_checkpoint(cls,state):
        config=state['upper_config']
        model=cls(config['weights_file'],rank=config['rank'],alpha=config['alpha'],
                  adapt_projection='proj' in config['targets'],
                  embed_dim=config.get('embed_dim',1024),num_heads=config.get('num_heads',16))
        model.load_adapter_state_dict(state['upper_adapter'])
        return model


class FrozenUpperPrefix:
    """每窗口创建一次，可跨步复用；只持有完全冻结的4/11/17层CPU输入。"""
    def __init__(self,native_run,scene):
        self.scene_id=scene['scene_id']; self.layers={layer:[] for layer in [4,11,17]}
        self.images=[view['image'][None,None] for view in scene['views']]
        self.image_hw=tuple(self.images[0].shape[-2:])
        for i in range(len(scene['views'])):
            packed=torch.load(Path(native_run)/'frozen_prefix'/f'{self.scene_id}_{i:02}.pt',
                              map_location='cpu',weights_only=True,mmap=True)
            self.patch_start=packed['patch_start']
            for layer in self.layers: self.layers[layer].append(packed['tokens'][layer])
        self.layers={layer:torch.cat(values,dim=1) for layer,values in self.layers.items()}


class UpperAdaptedGeometryPyramid(NativeGeometryPyramid):
    """全视图上层前向后再选Actor视图；每次调用重建DPT输入计算图。"""
    def __init__(self,prefix,view_indices,head,upper_tail):
        nn.Module.__init__(self)
        self.prefix=prefix; self.view_indices=list(view_indices)
        self.head=head; self.upper_tail=upper_tail
        self.token_inputs=[]

    def _refresh_inputs(self):
        device=next(self.head.parameters()).device
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            current=self.upper_tail(self.prefix.layers[17].to(device),self.prefix.image_hw,self.prefix.patch_start)
        self.token_inputs=[]
        for i in self.view_indices:
            frozen=tuple(self.prefix.layers[layer][:,i:i+1] for layer in [4,11,17])
            self.token_inputs.append(((*frozen,current[:,i:i+1]),self.prefix.images[i],self.prefix.patch_start))

    def forward(self,include_depth=False):
        self._refresh_inputs()
        try:
            return super().forward(include_depth=include_depth)
        finally:
            # 输出持有所需梯度链；模块自身不保留上一优化步的适配token。
            self.token_inputs=[]

    def depth_only(self,output_cache=None):
        # 与原生接口保持调用兼容，但此实现不接受适配深度的持久缓存。
        self._refresh_inputs()
        try:
            return super().depth_only(output_cache=None)
        finally:
            self.token_inputs=[]


def make_upper_pyramid(native_run,scene,view_indices,head,upper_tail,prefix_cache):
    """训练与固定推理共用构造入口，加载4/11/17而不是旧最终token。"""
    if head is None:
        from vggt.heads.dpt_head import DPTHead
        head=DPTHead(dim_in=2048,output_dim=2,activation='exp',conf_activation='expp1')
        state=torch.load(Path(native_run)/'latest.pt',map_location='cpu',weights_only=True)
        head.load_state_dict(state['depth_head'])
        head.to(next(upper_tail.parameters()).device)
    key=scene['scene_id']
    if key not in prefix_cache: prefix_cache[key]=FrozenUpperPrefix(native_run,scene)
    return UpperAdaptedGeometryPyramid(prefix_cache[key],view_indices,head,upper_tail)
