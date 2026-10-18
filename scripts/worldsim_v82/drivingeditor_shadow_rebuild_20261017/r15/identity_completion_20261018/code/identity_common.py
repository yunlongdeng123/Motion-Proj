"""五身份对照的输入合同；本模块不导入Torch。"""
from pathlib import Path
import json
import numpy as np

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1')
TASK=ROOT/'identity_completion_20261018'
R15=ROOT/'shadow_rebuild_20261017_v3/backend_restore_r15'
ACTORS=[5,26,18,6,37]
OLD_ACTORS=[5,26,18]
NEW_ACTORS=[6,37]
TASK_ID='WS-V82-R15-IDENTITY-COMPARISON-20261018'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def stat(path):
    p=Path(path);s=p.stat()
    return dict(path=str(p),size=s.st_size,mtime_ns=s.st_mtime_ns)

def compose_masks(old_masks,new_masks,new_known,new_valid):
    """原3身份不变；新2身份未知或重叠时拒绝完整mask训练。"""
    old=np.asarray(old_masks);new=np.asarray(new_masks)
    known=np.asarray(new_known);valid=np.asarray(new_valid)
    if old.dtype!=bool or new.dtype!=bool or known.dtype!=bool or valid.dtype!=bool:
        raise ValueError('mask/known/valid需要明确bool')
    if old.ndim!=3 or old.shape[-1]!=3 or new.shape!=(*old.shape[:2],2):
        raise ValueError('需要HW3旧mask与HW2新mask')
    if known.shape!=(2,) or valid.shape!=new.shape or (new&~valid).any():
        raise ValueError('新身份有效域无效')
    if (old.sum(-1)>1).any():raise ValueError('原旧版mask已存在重叠')
    combined=np.concatenate((old,new),-1)
    overlap=combined.sum(-1)>1
    missing=[dict(new_index=int(a),known=bool(known[a]),unknown_pixels=int((~valid[...,a]).sum()))
             for a in range(2) if not known[a] or not valid[...,a].all()]
    ready=not bool(overlap.any()) and not missing
    # 不把未知填0、不把重叠裁给任意ID。失败时仅返回诊断，不输出可训练标签。
    labels=None
    if ready:
        labels=np.zeros(old.shape[:2],np.uint8)
        for k in range(5):labels[combined[...,k]]=k+1
    return labels,dict(ready=ready,overlap_pixels=int(overlap.sum()),missing=missing,
                       old_positive_pixels=[int(old[...,a].sum()) for a in range(3)],
                       new_positive_pixels=[int(new[...,a].sum()) for a in range(2)])
