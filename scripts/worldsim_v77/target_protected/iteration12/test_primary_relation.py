"""语义反例：同样二维交叠不代表合法A→B；错身份不能借邻车通过。"""
import copy
import numpy as np
import pytest
from primary_relation import check_primary_relation, partition_primary_reveal


def fixture(depth=15., x=0.):
    # 相机z朝前，A/B全部相同朝向；A深度位置是唯一改变项。
    actor=lambda z,xx=0.:dict(translation=[xx,0.,z],size=[2.,4.,2.],rotation=[1.,0.,0.,0.])
    b=dict(actor(25.),instance_token='B')
    f=dict(timestamp=100,camera_to_world=np.eye(4).tolist(),
           intrinsics_1024=[[600,0,512],[0,600,288],[0,0,1]],actors=[b])
    a=dict(timestamp=100,actor=actor(depth,x))
    h=np.zeros((576,1024),bool);h[250:327,450:575]=True
    return [copy.deepcopy(f) for _ in range(3)], [copy.deepcopy(a) for _ in range(3)], [h.copy() for _ in range(3)]


def test_overlapping_projection_requires_correct_depth():
    front=check_primary_relation(*fixture(15.),'B')
    behind=check_primary_relation(*fixture(35.),'B')
    assert front['necessary_relation_pass']
    assert behind['possible_contact_frames']==3 and not behind['necessary_relation_pass']


def test_faraway_hole_does_not_hide_primary():
    f,a,h=fixture()
    for m in h:m[:]=False;m[10:30,10:30]=True
    assert check_primary_relation(f,a,h,'B')['reason']=='no_possible_primary_contact'


def test_unknown_identity_and_wrong_time_are_errors():
    f,a,h=fixture()
    with pytest.raises(ValueError,match='主B缺失'):check_primary_relation(f,a,h,'neighbor')
    a[0]['timestamp']+=1
    with pytest.raises(ValueError,match='时间戳'):check_primary_relation(f,a,h,'B')


def test_queue_requires_explicit_target_and_keeps_rejects():
    f,a,h=fixture(35.)
    c=dict(case_id='bad',frames=f,trajectory=dict(frames=a),requested_family='protected_reveal',primary_instance_token='B')
    ready,rejected,checks=partition_primary_reveal([c],lambda _:h)
    assert not ready and rejected[0]['case_id']=='bad' and checks[0]['reason']
    del c['primary_instance_token']
    with pytest.raises(ValueError,match='显式声明'):partition_primary_reveal([c],lambda _:h)
