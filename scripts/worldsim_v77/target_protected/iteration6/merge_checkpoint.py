"""合并已有80个参数更新为原格式完整权重；原权重不改写。"""
from pathlib import Path
import argparse,json
from safetensors.torch import load_file,save_file
from train_pilot import selected,save_json,OFFICIAL

def main(root):
    output=root/'training/model_step_0160.safetensors'
    if output.exists():raise FileExistsError(output)
    base=load_file(str(OFFICIAL/'checkpoints/model.safetensors'))
    patch=load_file(str(root/'training/attention_step_0160.safetensors'))
    assert len(patch)==80 and all(selected(k) and k in base and patch[k].shape==base[k].shape for k in patch)
    changed=[]
    for k,v in patch.items():
        if (v!=base[k].float()).any():changed.append(k)
        base[k]=v
    assert len(changed)==80
    save_file(base,str(output),metadata={'source':'DriveEditor original trained checkpoint','fine_tune_task':'WS-V77-TARGET-PROTECTED-20260929/r6','steps':'160','architecture':'unchanged','update_scope':'80 existing spatial attention tensors'})
    save_json(root/'training/checkpoint_manifest.json',{'base':str(OFFICIAL/'checkpoints/model.safetensors'),'patch':str(root/'training/attention_step_0160.safetensors'),'merged':str(output),'keys':len(base),'changed_tensors':len(changed),'bytes':output.stat().st_size,'base_preserved':True,'architecture_unchanged':True})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
