"""一次CPU接口检查：真实权重可加载、官方交替顺序复原与冻结输入的LoRA梯度。"""
import copy,json
from pathlib import Path
import resource,sys,time
from types import SimpleNamespace
import torch

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from motion_proj.worldsim_v73.upper_aggregation import VGGTUpperTail

torch.set_num_threads(2); torch.manual_seed(7305); started=time.monotonic()
actual=VGGTUpperTail('/root/autodl-tmp/models/eas_vggt/vggt/model.safetensors')
counts={'frozen_parameters':sum(p.numel() for p in actual.parameters() if not p.requires_grad),
        'lora_parameters':sum(p.numel() for p in actual.parameters() if p.requires_grad)}
del actual
from vggt.models.aggregator import Aggregator
tail=VGGTUpperTail(embed_dim=32,num_heads=4,rank=2,alpha=2).train()
frames=copy.deepcopy(tail.frame_blocks); globals_=copy.deepcopy(tail.global_blocks)
for block in [*frames,*globals_]: block.attn.qkv=block.attn.qkv.base
reference=SimpleNamespace(frame_blocks=frames,global_blocks=globals_,training=False,aa_block_size=1)
cached=torch.randn(1,3,11,64)
position=tail.positions(3,(28,42),5,cached.device)
with torch.no_grad():
    x=cached[...,32:]; fi=gi=0
    for _ in range(6):
        x,fi,frame=Aggregator._process_frame_attention(reference,x,1,3,11,32,fi,position)
        x,gi,global_=Aggregator._process_global_attention(reference,x,1,3,11,32,gi,position)
    expected=torch.cat([frame[0],global_[0]],dim=-1)
result=tail(cached,(28,42))
torch.testing.assert_close(result,expected,rtol=1e-5,atol=1e-6)
target=torch.randn_like(result[:,0]); loss=(result[:,0]-target).square().mean(); loss.backward()
gradients={name:float(p.grad.norm()) if p.grad is not None else None
           for name,p in tail.named_parameters() if p.requires_grad}
assert gradients['frame_blocks.0.attn.qkv.lora_B']>0
assert all(p.grad is None for p in tail.parameters() if not p.requires_grad)
assert not cached.requires_grad and cached.grad is None
optimizer=torch.optim.SGD([p for p in tail.parameters() if p.requires_grad],lr=.1); optimizer.step()
with torch.no_grad():
    after=tail(cached,(28,42))
    changed=cached.clone(); changed[:,2]+=torch.randn_like(changed[:,2])*.3
    cross_view=float((tail(changed,(28,42))[:,0]-after[:,0]).abs().max())
    adaptation=float((after-result.detach()).abs().max())
assert cross_view>0 and adaptation>0
print(json.dumps({'status':'done','seed':7305,**counts,'official_alternation_max_error':float((result.detach()-expected).abs().max()),
    'first_frame_lora_B_gradient':gradients['frame_blocks.0.attn.qkv.lora_B'],
    'first_frame_lora_A_gradient':gradients['frame_blocks.0.attn.qkv.lora_A'],
    'last_global_lora_B_gradient':gradients['global_blocks.5.attn.qkv.lora_B'],
    'synthetic_optimizer_steps':1,'adapted_output_max_change':adaptation,'other_view_effect_on_first_view':cross_view,
    'cached_input_requires_grad':cached.requires_grad,'frozen_weight_gradients':False,
    'wall_s':time.monotonic()-started,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
    'boundary':'真实最后6组权重仅CPU加载；3视图11token/32维随机模型验证官方顺序与梯度。不是24视图真实训练/数值复原、不是DPT/Query传感器端到端证据或GPU峰值。'},ensure_ascii=False))
