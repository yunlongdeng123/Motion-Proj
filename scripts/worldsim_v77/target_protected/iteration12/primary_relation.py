"""在实例分割前检查明确指定的保护车关系；不读取真实RGB或SAM。

这是已有严格深度准入规则的必要条件，不证明可见轮廓、显露或道路合法。
H与3D框投影相交只表示可能遮到B，不把矩形框当实例mask。
"""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).parents[1]))
from geometry_factory import projection


def check_primary_relation(frames, trajectory, holes, primary_token, min_frames=3):
    if not primary_token or len(frames) != len(trajectory) or len(frames) != len(holes):
        raise ValueError('主B身份及帧数必须明确一致')
    if min_frames != 3:
        raise ValueError('本前置检查沿用既有至少3帧遮挡合同，不接受逐例改阈值')
    rows = []
    for index, (frame, pose, hole) in enumerate(zip(frames, trajectory, holes)):
        if frame['timestamp'] != pose['timestamp']:
            raise ValueError('A与RGB时间戳不一致')
        hole = np.asarray(hole)
        if hole.dtype != np.bool_ or hole.shape != (576, 1024):
            raise ValueError('必须传入最终分辨率的二值H')
        actors = [a for a in frame['actors'] if a['instance_token'] == primary_token]
        if len(actors) != 1 or actors[0].get('interpolation_uncertain', False):
            raise ValueError('主B缺失、重复或位姿不确定，不能当成无车')
        fr = dict(frame, _w2c=np.linalg.inv(frame['camera_to_world']))
        pa, pb = projection(pose['actor'], fr), projection(actors[0], fr)
        if pa is None or pb is None:
            rows.append(dict(frame=index, contact_pixels_upper_bound=0,
                             whole_A_before_B=False, depth_unavailable=True))
            continue
        lo = np.maximum(np.floor(pb['box'][:2]).astype(int), 0)
        hi = np.minimum(np.ceil(pb['box'][2:]).astype(int), [1024, 576])
        contact = int(hole[lo[1]:hi[1], lo[0]:hi[0]].sum()) if np.all(hi > lo) else 0
        ordered = bool(pa['far_depth'] + .3 < pb['near_depth'])
        rows.append(dict(frame=index, contact_pixels_upper_bound=contact,
                         whole_A_before_B=ordered, depth_unavailable=False,
                         A_depth_m=[pa['near_depth'], pa['far_depth']],
                         B_depth_m=[pb['near_depth'], pb['far_depth']]))
    contact_frames = sum(r['contact_pixels_upper_bound'] > 0 for r in rows)
    possible_frames = sum(r['contact_pixels_upper_bound'] > 0 and r['whole_A_before_B'] for r in rows)
    reason = None if possible_frames >= min_frames else (
        'no_possible_primary_contact' if contact_frames == 0 else 'insufficient_ordered_primary_contact')
    return dict(primary_instance_token=primary_token, necessary_relation_pass=reason is None,
                reason=reason, possible_contact_frames=contact_frames,
                ordered_possible_contact_frames=possible_frames, frames=rows,
                boundary='GT-envelope possibility only; all existing SAM, evidence, scope and independent QA remain required',
                reads_RGB_or_SAM=False, training_admission=False)


def partition_primary_reveal(cases, hole_reader):
    """生成器先声明本case的主B，再建立昂贵的实例标签队列；拒绝隐式换B。"""
    ready, rejected, checks = [], [], []
    for case in cases:
        if case.get('requested_family') != 'protected_reveal' or not case.get('primary_instance_token'):
            raise ValueError('本入口只处理已显式声明主B的reveal任务')
        result = check_primary_relation(case['frames'], case['trajectory']['frames'],
                                        hole_reader(case), case['primary_instance_token'])
        checks.append(dict(case_id=case['case_id'], **result))
        (ready if result['necessary_relation_pass'] else rejected).append(case)
    return ready, rejected, checks
