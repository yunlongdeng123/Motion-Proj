"""一次CPU保存/恢复与真实前缀构造检查，不运行聚合器或传感器前向。"""
import argparse,json,resource,sys,time
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from motion_proj.worldsim_v73.upper_aggregation import VGGTUpperTail,make_upper_pyramid

parser=argparse.ArgumentParser()
parser.add_argument('--native-run',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
torch.set_num_threads(2); torch.manual_seed(7305); started=time.monotonic()
tail=VGGTUpperTail('/root/autodl-tmp/models/eas_vggt/vggt/model.safetensors')
name='frame_blocks.0.attn.qkv.lora_B'
with torch.no_grad(): dict(tail.named_parameters())[name].fill_(1e-4)
args.output.parent.mkdir(parents=True,exist_ok=True)
payload={'upper_config':tail.config,'upper_adapter':tail.adapter_state_dict()}
checkpoint_path=args.output.with_suffix('.pt'); torch.save(payload,checkpoint_path)
restored=VGGTUpperTail.from_checkpoint(torch.load(checkpoint_path,map_location='cpu',weights_only=True))
max_error=max((p-dict(restored.named_parameters())[key]).abs().max().item()
              for key,p in tail.named_parameters() if p.requires_grad)
assert max_error==0
assert all(not p.requires_grad for key,p in restored.named_parameters() if 'lora_' not in key)
del tail,restored

# mmap读取既有窗口，不读取heldout测量或预测质量；只构造CPU输入。
scenes=torch.load(args.native_run/'build_observations.pt',map_location='cpu',weights_only=False,mmap=True)
scene=scenes[0]; cache={}
tail=VGGTUpperTail.from_checkpoint(payload)
pyramid=make_upper_pyramid(args.native_run,scene,[0,len(scene['views'])-1],None,tail,cache)
other=make_upper_pyramid(args.native_run,scene,[0],pyramid.head,tail,cache)
assert other.prefix is pyramid.prefix and other.head is pyramid.head
assert set(pyramid.prefix.layers)=={4,11,17}
assert pyramid.prefix.layers[17].shape[1]==len(scene['views'])
result={'status':'done','seed':7305,'adapter_max_restore_error':max_error,
    'adapter_checkpoint_bytes':checkpoint_path.stat().st_size,'changed_adapter_parameter':name,
    'changed_parameter_value':1e-4,'optimizer_updates':0,'sensor_forward_calls':0,
    'scene':scene['scene_id'],'window_views':len(scene['views']),'actor_views':pyramid.view_indices,
    'prefix_shapes':{str(k):list(v.shape) for k,v in pyramid.prefix.layers.items()},
    'shared_cpu_prefix':True,'shared_dpt_head':True,'old_layer23_used':False,
    'wall_s':time.monotonic()-started,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
    'boundary':'真实预训练权重/前缀/DPT仅CPU构造；参数人为赋值用于保存恢复，未优化、未聚合前向、未检查DPT/Query传感器梯度或GPU峰值。'}
args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False))
