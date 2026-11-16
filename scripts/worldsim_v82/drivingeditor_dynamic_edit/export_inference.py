"""CPU 只读导出推理包；原始 Adam/RNG 训练现场完整保留。"""
from pathlib import Path
import argparse
import copy
import json
import shutil

import torch

from .ownership import OWNERSHIP_KEY,LINEAGE_KEY
from .scene_io import checkpoint_actors


def strip_adam(payload):
    if not isinstance(payload,(tuple,list)) or len(payload)!=18:raise ValueError('只支持原生Gaussian18tuple')
    result=[]
    for index,value in enumerate(payload):
        if index==14:
            if not isinstance(value,dict):raise ValueError('未知optimizer payload')
            metadata={key:copy.deepcopy(item) for key,item in value.items() if key not in ['state','param_groups']}
            metadata['inference_only']=True
            result.append(metadata)
        elif torch.is_tensor(value):result.append(value.detach().cpu().clone())
        else:result.append(copy.deepcopy(value))
    return tuple(result)


def validate_equal(original,exported):
    for index,(left,right) in enumerate(zip(original,exported)):
        if index==14:
            if 'state' in right or 'param_groups' in right:raise ValueError('Adam未移除')
            for key in [LINEAGE_KEY,OWNERSHIP_KEY]:
                if key not in left:continue
                if set(left[key])!=set(right[key]):raise ValueError('身份metadata字段改变')
                for name,a in left[key].items():
                    b=right[key][name]
                    if torch.is_tensor(a):
                        if a.dtype!=b.dtype or a.shape!=b.shape or not torch.equal(a.cpu(),b.cpu()):raise ValueError('身份Tensor改变')
                    elif a!=b:raise ValueError('身份字段改变')
        elif torch.is_tensor(left):
            if left.dtype!=right.dtype or left.shape!=right.shape or not torch.equal(left.cpu(),right.cpu()):raise ValueError(f'参数{index}改变')
        elif left!=right:raise ValueError(f'字段{index}改变')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True);parser.add_argument('--iteration',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    opts=parser.parse_args();model=opts.model.resolve();output=opts.output.resolve()
    if output.exists() or output.is_relative_to(model):raise ValueError('推理包必须位于新的独立目录')
    info=checkpoint_actors(model,opts.iteration);output.mkdir(parents=True)
    records={}
    for prefix in ['chkpnt','dynamic_chkpnt']:
        name=f'{prefix}{opts.iteration}.pth';path=model/name
        payload,step=torch.load(str(path),map_location='cpu',weights_only=False,mmap=True)
        if int(step)!=opts.iteration:raise ValueError('源checkpoint step不一致')
        slim=strip_adam(payload);validate_equal(payload,slim)
        torch.save((slim,int(step)),output/name)
        roundtrip,saved_step=torch.load(str(output/name),map_location='cpu',weights_only=False,mmap=True)
        if int(saved_step)!=int(step):raise ValueError('导出step改变')
        validate_equal(payload,roundtrip)
        records[name]=dict(original_bytes=path.stat().st_size,inference_bytes=(output/name).stat().st_size,
            all_non_optimizer_fields_exact=True,identity_metadata_exact=True,Adam_state_removed=True)
        del slim,payload,roundtrip
    name=f'env_light_chkpnt{opts.iteration}.pth';shutil.copy2(model/name,output/name)
    records[name]=dict(original_bytes=(model/name).stat().st_size,inference_bytes=(output/name).stat().st_size,
        original_environment_format_copied_unchanged=True)
    for name in ['cameras.json','actual_scene_binding.json']:
        shutil.copy2(model/name,output/name)
    config=json.loads((model/'effective_config.json').read_text())
    config.update(inference_only=True,training_resume_supported=False,start_checkpoint=None,model_path=str(output))
    (output/'effective_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
    exported_info=checkpoint_actors(output,opts.iteration)
    if exported_info!=info:raise ValueError('导出包身份概要改变')
    manifest=dict(status='PASS_CPU_INFERENCE_EXPORT_FIELDS_AND_IDENTITY_EXACT',inference_only=True,
        training_resume_supported=False,source_training_directory=str(model),source_training_directory_untouched=True,
        iteration=opts.iteration,**info,files=records,total_bytes=sum(p.stat().st_size for p in output.iterdir() if p.is_file()),
        optimizer_created=False,optimization_updates=0,
        note='Gaussian Adam removed; environment format copied unchanged. Full original Adam/RNG checkpoints remain in source directory.',
        gpu_equivalence='PENDING_NATIVE_THREE_FRAME_COMPARISON')
    (output/'inference_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(manifest,ensure_ascii=False))


if __name__=='__main__':main()
