"""只向原 V1 添加两处独立道路调用，不复制或重写原训练实现。"""

import re

INITIALIZE = '''    # V4_BEGIN：独立道路不进入 V1 模型或训练渲染。
    from utils.v4_road import create_road_training
    road_training = create_road_training(args, scene, first_iter)
    # V4_END
'''
UPDATE = '''            # V4_BEGIN：V1 本步全部完成后，道路独立更新；内部恢复全局 RNG。
            if road_training is not None:
                road_training.step(iteration)
            # V4_END
'''


def strip_hooks(source):
    return re.sub(r'^ *# V4_BEGIN[^\n]*\n.*?^ *# V4_END\n', '', source, flags=re.M | re.S)


def install_hooks(source):
    anchors = ('    progress_bar = tqdm(range(first_iter + 1, args.iterations + 1), desc="Training progress")\n',
               '# def mkdir_p(folder_path):\n')
    if 'V4_BEGIN' in source or any(source.count(anchor) != 1 for anchor in anchors):
        raise ValueError('V1 源码结构不匹配，拒绝模糊插入或重复安装')
    modified = source.replace(anchors[0], INITIALIZE+anchors[0], 1).replace(anchors[1], UPDATE+anchors[1], 1)
    if strip_hooks(modified) != source:
        raise AssertionError('道路调用以外不得修改原 V1 代码')
    return modified
