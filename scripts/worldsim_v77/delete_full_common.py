"""v77 DELETE工程的固定输入合同与纯指令操作。"""
import copy,json,pathlib
ROOT=pathlib.Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-DELETE-FULL-20260927/r1')
REPO=pathlib.Path('/root/autodl-tmp/motion_proj_v77')
OLD=pathlib.Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-EXPLICIT-POC-20260926/r1')
DE=pathlib.Path('/root/autodl-tmp/external/worldsim_v75_downstream_bench/DriveEditor')
SCENES={'scene_0230':{'actor':'22','count':50,'yaw':180},'scene_0255':{'actor':'25','count':100,'yaw':0}}

def dump(path,obj):
    path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(path)

def apply_delete(scene,actor_id):
    """QUERY只切换指定显式资产可见性，背景与其他对象保持相同引用。"""
    matches=[a for a in scene['actors'] if a['actor_id']==str(actor_id)]
    if len(matches)!=1:raise ValueError('目标actor必须唯一存在')
    if not matches[0]['visible']:raise ValueError('目标已删除，不重复提交')
    edited=copy.deepcopy(scene)
    next(a for a in edited['actors'] if a['actor_id']==str(actor_id))['visible']=False
    assert edited['background']==scene['background']
    return edited
