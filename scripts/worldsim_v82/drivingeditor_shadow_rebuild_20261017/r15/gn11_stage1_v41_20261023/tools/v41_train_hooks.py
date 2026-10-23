"""只加初始化和步末道路调用；删除两块后逐字恢复原 V1。"""
import re

INITIALIZE = '''    # V41_BEGIN：道路密集监督不进入原 V1 训练前向。
    from utils.v41_road import create_road_training, after_step
    road_training = create_road_training(args, scene, first_iter)
    # V41_END
'''
UPDATE = '''            # V41_BEGIN：V1 本步完成，额外联合前向仅更新道路。
            road_training.step(iteration, gaussians, dynamic_gaussians, background, env_map)
            after_step(args, iteration)
            # V41_END
'''


def strip_hooks(source):
    return re.sub(r'^ *# V41_BEGIN[^\n]*\n.*?^ *# V41_END\n', '', source, flags=re.M | re.S)


def install_hooks(source):
    anchors = ('    progress_bar = tqdm(range(first_iter + 1, args.iterations + 1), desc="Training progress")\n',
               '# def mkdir_p(folder_path):\n')
    if 'V4_BEGIN' in source or 'V41_BEGIN' in source or any(source.count(x) != 1 for x in anchors):
        raise ValueError('必须从原 V1 安装，拒绝重复或模糊插入')
    result = source.replace(anchors[0], INITIALIZE+anchors[0], 1).replace(anchors[1], UPDATE+anchors[1], 1)
    if strip_hooks(result) != source:
        raise AssertionError('V1 原流程发生变化')
    return result
