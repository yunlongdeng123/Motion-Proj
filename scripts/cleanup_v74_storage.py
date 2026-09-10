"""按明确类别清理可重建缓存；保留关键模型、表面与历史指标。"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path('/root/autodl-tmp')
REPO=ROOT/'motion_proj'
OUT=REPO/'docs/autoresearch/worldsim_v74/p0'

def save(name,value):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def size(path):
    return int(subprocess.check_output(['du','-s','-B1',str(path)],text=True).split()[0])

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if not args.execute:
        targets=[]
        for p in sorted((ROOT/'runs/worldsim_v73').glob('*/*/frozen_prefix')):
            targets.append({'path':str(p),'kind':'directory','allocated_bytes':size(p),
                'reason':'V73 冻结视觉聚合特征，V74 首轮不使用视觉',
                'recovery':'保留各 run 的 build_observations/config/metric_scales、VGGT 权重及 prepare_worldsim_v73_native_prefix.py；重新 GPU 前向生成'})
        retained=[]
        for folder in sorted((ROOT/'runs/worldsim_v6').glob('*/*/sensor_worker/sensors')):
            files=sorted(folder.glob('frame*.npz'))
            keep={files[i] for i in [0,len(files)//2,len(files)-1]} if files else set()
            retained.extend(str(p) for p in sorted(keep))
            for p in files:
                if p not in keep:
                    targets.append({'path':str(p),'kind':'file','allocated_bytes':p.stat().st_blocks*512,
                        'reason':'退役 V6 传感器重渲染帧；每 run 保留首/中/末三帧及所有 summary/config/指标',
                        'recovery':'按同 run 的 worker manifest、配置、原始数据和模型重建；可能需要恢复历史依赖环境'})
        for version in ['worldsim_v63','worldsim_v64','worldsim_v65','worldsim_v67']:
            groups={}
            for p in sorted((ROOT/'runs'/version).rglob('NATIVE_LOGITS.npy')):
                run=Path(*p.parts[:8])
                groups.setdefault(run,[]).append(p)
            for run,files in groups.items():
                keep={files[i] for i in [0,len(files)//2,len(files)-1]}
                retained.extend(str(p) for p in sorted(keep))
                for p in files:
                    if p not in keep:
                        targets.append({'path':str(p),'kind':'file','allocated_bytes':p.stat().st_blocks*512,
                            'reason':'退役侧路密集语义 logits；保留每组首/中/末样本及几何/指标/配置',
                            'recovery':'用原 run 的 source/model/config 重新推理；可能需要恢复历史依赖环境'})
        result={'task':'WS-V74-P0-STORAGE-01','status':'planned','code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
            'targets':targets,'retained_representative_payloads':retained,
            'planned_allocated_bytes':sum(t['allocated_bytes'] for t in targets),
            'preserved':'raw data, V73 checkpoints/fixed surfaces/rays/results, all run configs/manifests/metrics, environments and model sources',
            'boundary':'可重建不意味着逐字节可复原；原始帧级再分析可能需要昂贵的历史复现。无新校验和或指纹。'}
        save('storage_plan.json',result)
        print(json.dumps({'targets':len(targets),'gib':result['planned_allocated_bytes']/2**30,'retained_examples':len(retained)}))
        return
    plan=json.loads((OUT/'storage_plan.json').read_text())
    before=shutil.disk_usage(ROOT)
    result={'task':plan['task'],'status':'running','started_unix_s':time.time(),
            'before':dict(zip(['total','used','free'],before)),'deleted':[]}
    save('storage_result.json',result)
    for target in plan['targets']:
        p=Path(target['path'])
        resolved=p.resolve()
        if not resolved.is_relative_to(ROOT/'runs') or p.is_symlink():
            raise ValueError('清理目标越界或是符号链接: '+str(p))
        if target['kind']=='directory':
            if p.name!='frozen_prefix' or not p.is_dir():
                raise ValueError('目录类型不符: '+str(p))
            shutil.rmtree(p)
        else:
            p.unlink()
        result['deleted'].append(target['path'])
    after=shutil.disk_usage(ROOT)
    result.update(status='done',finished_unix_s=time.time(),after=dict(zip(['total','used','free'],after)),
        actual_free_increase_bytes=after.free-before.free,deleted_count=len(result['deleted']),
        note='文件系统空闲差值；与按路径累计占用分开报告；目标及恢复说明见 storage_plan.json。')
    save('storage_result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='deleted'}))

if __name__=='__main__': main()
