"""执行原训练入口与原配置，唯一训练预算为五实例30K；不续训R15。"""
import argparse,subprocess,sys
from identity_common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');p.add_argument('--resume',type=Path)
    a=p.parse_args();ready=read(TASK/'prepared.json')
    if ready['status']!='CPU_READY_GPU_NOT_STARTED':raise ValueError('先完成真实输入准备')
    old=read(TASK/'old_effective_config.json');args=old['arguments']
    assert args['iterations']==30000 and args['joint_background'] and args['seed']==0
    assert old['spec']['background_repair']['freeze_environment'] is True
    assert old['spec']['background_repair']['initialization']=='preserve_all_GN11_static_Gaussians'
    spec=read(TASK/'spec5.json')
    assert spec['background_checkpoint']==old['spec']['background_checkpoint']
    # 原入口的策略、原LR horizon、原loss与checkpoint/probe schedule完全沿用。
    cmd=[sys.executable,str(TASK/'infra/train.py'),'--spec',str(TASK/'spec5.json'),
         '--output',str(TASK/('train5_resume' if a.resume else 'train5')),
         '--iterations','30000','--joint-background']
    for key in ['seed','refine_poses_after','pose_translation_limit_m','box_boundary_weight',
                'pose_prior_weight','max_features','gpu_lock']:
        cmd+=['--'+key.replace('_','-'),str(args[key])]
    for key in ['checkpoint_steps','probe_steps']:
        cmd+=['--'+key.replace('_','-')]+[str(x) for x in args[key]]
    if a.resume:
        if TASK.resolve() not in a.resume.resolve().parents:raise ValueError('只恢复本次五实例checkpoint')
        cmd+=['--resume',str(a.resume)]
    if a.run:cmd+=['--run']
    write(TASK/('training_launch_resume.json' if a.resume else 'training_launch.json'),
        dict(command=cmd,run=a.run,old_actual_policy=old.get('training_policy'),
             intended_total_instance_steps=30000,R15_training_steps=0,
             historical_3_ID_checkpoint_read_only=True))
    subprocess.run(cmd,check=True)

if __name__=='__main__':main()
