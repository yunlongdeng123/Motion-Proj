import json
from pathlib import Path
import torch
from safetensors import safe_open

path=Path('/root/autodl-tmp/runs/worldsim_v73/WS-V73-M1-NATIVE-GEOMETRY-ADAPT-01/20260907T161500Z__native-dpt-surround25-dev6-s7301-r3/frozen_prefix/scene-0450_03.pt')
data=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
out={'kind':'CPU metadata and arithmetic only; no model forward/backward',
     'cache_file':str(path),'patch_start':data['patch_start'],
     'layers':{str(k):{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in data['tokens'].items()}}
with safe_open('/root/autodl-tmp/models/eas_vggt/vggt/model.safetensors',framework='pt',device='cpu') as weights:
    keys=['aggregator.'+group+'.18.attn.'+layer+'.weight' for group in ['frame_blocks','global_blocks'] for layer in ['qkv','proj']]
    out['upper_projection_shapes']={key:list(weights.get_slice(key).get_shape()) for key in keys}
S,P,C,heads,bytes_per=24,1301,1024,16,2
N=S*P
out['arithmetic']={'views':S,'image_hw':[378,672],'tokens_per_view':P,'global_tokens':N,
                   'one_bf16_state_mib':N*C*bytes_per/2**20,
                   'one_fp32_state_mib':N*C*4/2**20,
                   'four_concat_states_bf16_mib':4*N*(2*C)*bytes_per/2**20,
                   'four_concat_states_fp32_mib':4*N*(2*C)*4/2**20,
                   'dense_scores_bf16_gib':heads*N*N*2/2**30,
                   'dense_scores_fp32_gib':heads*N*N*4/2**30,
                   'rank8_qkv_last6_pairs_parameters':6*2*8*(C+3*C),
                   'rank8_qkv_proj_last6_pairs_parameters':6*2*8*((C+3*C)+(C+C))}
out['boundary']='score sizes are hypothetical dense allocations, not measured peak memory; proposed tail has not been implemented or benchmarked'
target=Path('/root/autodl-tmp/controller_logs/v73_upper_tail_metadata.json')
target.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
