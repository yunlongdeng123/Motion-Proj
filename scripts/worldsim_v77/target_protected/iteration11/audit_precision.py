"""CPU分解原始权重、BF16舍入及实际梯度更新；范数不是画面质量。"""
from pathlib import Path
import json,math
import torch
from safetensors import safe_open

def main():
    task=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929')
    base=Path('/root/autodl-tmp/external/worldsim_v75_downstream_bench/DriveEditor/checkpoints/model.safetensors')
    torch.set_num_threads(4);rows={}
    with safe_open(str(base),framework='pt',device='cpu') as f:
        for name,folder in [('r7',task/'r7/encoder_fixed_lowres/training'),('r14',task/'r14/training'),('r18',task/'r18/training')]:
            cfg=json.loads((folder/'config.json').read_text());sums=dict.fromkeys(['base','quantization','learned_after_rounding','net_difference'],0.);mx=dict.fromkeys(['quantization','learned_after_rounding','net_difference'],0.);rounded=0;dtypes={}
            with safe_open(str(folder/'attention_step_0160.safetensors'),framework='pt',device='cpu') as patch:
                for k in cfg['trainable_tensors']:
                    a=f.get_tensor(k);dtypes[str(a.dtype)]=dtypes.get(str(a.dtype),0)+1;a=a.float();q=a.bfloat16().float();b=patch.get_tensor(k).float();rounded+=int(torch.any(a!=q));parts={'base':a,'quantization':q-a,'learned_after_rounding':b-q,'net_difference':b-a}
                    for n,v in parts.items():sums[n]+=float(v.double().square().sum())
                    for n in mx:mx[n]=max(mx[n],float(parts[n].abs().max()))
            rows[name]={'base_checkpoint_dtypes':dtypes,'trainable_tensors_rounded_before_training':rounded,'l2_norms':{k:math.sqrt(v) for k,v in sums.items()},'relative_to_base_l2':{k:math.sqrt(v/sums['base']) for k,v in sums.items() if k!='base'},'max_abs_change':mx,'quantization_norm_over_net_update_norm':math.sqrt(sums['quantization']/sums['net_difference'])}
    result={'scope':'CPU逐张量复现旧model_init: strict原权重→全模型bf16→训练参数float32','confirmed_full_precision_not_preserved':True,'cases':rows,'image_quality_causality_proven':False,'new_GPU_forwards':0,'new_training_steps':0,'limits':'范数比不是效果损伤百分比；画面因果只见独立r20零步控制'}
    (task/'r19/precision_initialization_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('PRECISION_AUDIT',list(rows))
if __name__=='__main__':main()
