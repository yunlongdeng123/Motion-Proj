"""只在原RGB cross-attention logits中增加无参数空间偏置。"""
import torch
from iteration15.multi_prior import MultiPriorBranch as ExistingBranch
from routing import attention_bias


class MultiPriorBranch(ExistingBranch):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.routing_enabled = True
        self._query_size = None
        self._bias_cache = {}
        self.routing_rows = []
        for index,module in enumerate(self.attention):
            module.register_forward_pre_hook(lambda module,args,kwargs,i=index:
                self._route_attention(i,args,kwargs),with_kwargs=True)

    def set_condition(self,*,routing_metadata=None,**condition):
        route = routing_metadata or {k:v for k,v in condition.items() if k.startswith('routing_')}
        base = {k:v for k,v in condition.items() if not k.startswith('routing_')}
        super().set_condition(**base)
        for k,v in route.items():
            if not torch.isfinite(v).all():raise ValueError('非有限空间对应')
            self._condition[k] = v.detach()
        self._bias_cache = {}

    def residual(self,index,activation):
        self._query_size = (min(18,activation.shape[-2]),min(32,activation.shape[-1]))
        return super().residual(index,activation)

    def _route_attention(self,index,args,kwargs):
        c = self._condition
        if not self.routing_enabled or c is None or 'routing_query_owner' not in c:return None
        # 错配只换已登记的主体patch外观，slot pose/身份/几何/valid保持原样。
        # delta在同一checkpoint的reference_encoder池化后计算，避免替换latent时串到邻patch。
        modified_args = args
        has_override = 'routing_reference_feature_delta' in c and bool(torch.count_nonzero(c['routing_reference_feature_delta']))
        if has_override:
            delta = c['routing_reference_feature_delta'].reshape(1,-1,self.width)
            delta = torch.cat([delta,torch.zeros((1,1,self.width),device=delta.device)],1)
            tokens = args[1]+delta
            modified_args = (args[0],tokens,tokens)+args[3:]
        if not torch.count_nonzero(c['routing_query_q']):return (modified_args,kwargs) if has_override else None
        key = (id(c),self._query_size)
        if key not in self._bias_cache:self._bias_cache[key] = attention_bias(c,self._query_size)
        bias,owner,q = self._bias_cache[key]
        if not torch.count_nonzero(bias):return (modified_args,kwargs) if has_override else None  # 全U必须走原始MHA路径。
        query = args[0]
        if bias.shape[:2]!=query.shape[:2]:raise ValueError('查询坐标/批次与attention不匹配')
        # 对每个head施加相同几何偏置；不改变Q/K/V编码器、池化或head参数。
        new = dict(kwargs)
        heads = self.attention[index].num_heads
        new['attn_mask'] = bias[:,None].expand(-1,heads,-1,-1).reshape(-1,*bias.shape[1:])
        invalid = kwargs['key_padding_mask']
        new['key_padding_mask'] = torch.zeros_like(invalid,dtype=bias.dtype).masked_fill(invalid,-torch.inf)
        if self.capture is not None:
            self.routing_rows.append(dict(self.capture,block=self.blocks[index],query_size=list(self._query_size),
                known_actor_queries=int((owner>0).sum()),known_background_queries=int((owner==-1).sum()),
                unknown_queries=int((owner==0).sum()),biased_pairs=int((bias<0).sum()),
                min_logit_bias=float(bias.min()),scope='预先计算的logit偏置；不是效果分数或注意力成功证明'))
        return modified_args,new

    def start_response(self):
        super().start_response();self.routing_rows = [];self._bias_cache = {}
