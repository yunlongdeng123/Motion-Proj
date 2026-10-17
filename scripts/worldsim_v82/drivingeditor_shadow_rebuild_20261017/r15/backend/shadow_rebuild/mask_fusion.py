"""逐区域对称融合。未知是独立监督状态，绝不编码为有效背景零标签。"""
import numpy as np
from scipy.ndimage import label, distance_transform_edt


def supported_components(mask, positive, minimum_hits=3):
    components,count=label(mask,structure=np.ones((3,3),bool))
    hits=np.bincount(components[positive],minlength=count+1)
    good=hits>=minimum_hits;good[0]=False
    return good[components]


def fuse_regions(old,old_known,sam,sam_known,body_evidence,road_evidence,radius=3):
    """两种来源共享相同实测点/连通域核验，无整实例回退或旧标签优先。

    重叠冲突像素对全部实例标为无效；同ID形状分歧只在实际车体证据邻域
    接受一侧，其余分歧保持未知。非重叠已可靠区域不因别处冲突而丢失。
    body_evidence是实际非道路LiDAR投影；本函数不创造mask或LiDAR点。
    """
    old=np.asarray(old,bool);sam=np.asarray(sam,bool)
    if old.shape!=sam.shape or body_evidence.shape!=old.shape:raise ValueError('HWK mismatch')
    h,w,k=old.shape;out=np.zeros_like(old);valid=np.broadcast_to(old_known|sam_known,old.shape).copy();source=np.zeros(old.shape,np.uint8)
    union_proposals=(old&old_known[None,None])|(sam&sam_known[None,None])
    ambiguous=np.zeros((h,w),bool)
    for a in range(k):
        x=old[...,a]&bool(old_known[a]);y=sam[...,a]&bool(sam_known[a]);e=body_evidence[...,a]&~road_evidence
        tx=supported_components(x,e);ty=supported_components(y,e)
        near=distance_transform_edt(~e)<=radius if e.any() else np.zeros((h,w),bool)
        if old_known[a] and sam_known[a]:
            agree=x&y;different=x^y
            good_agree=agree&(tx|ty)
            supported_unique=different&near&(tx|ty)
            accepted=good_agree|supported_unique
            unknown=(x|y)&~accepted
            source[...,a][good_agree]=3
            source[...,a][supported_unique&x]=1;source[...,a][supported_unique&y]=2
        else:
            accepted=tx|ty;unknown=(x|y)&~accepted
            source[...,a][accepted&x]=1;source[...,a][accepted&y]=2
        # A real world-road hit contradicts either prediction equally. Leave it
        # unresolved here; road/owner inference is a separate 3D evidence step.
        unknown|=(x|y)&road_evidence
        accepted&=~road_evidence
        out[...,a]=accepted;valid[...,a][unknown]=False
        ambiguous|=unknown
    overlap=out.sum(-1)>1
    # All claimants are unknown only at the contested pixels. No reinstatement
    # of an entire DEVA mask, and no arbitrary argmax winner.
    ambiguous|=overlap
    out[ambiguous]=False;valid[ambiguous]=False;source[ambiguous]=0
    assert not (out.sum(-1)>1).any() and not (out&~valid).any()
    known=(old_known|sam_known).astype(bool)
    return dict(masks=out,mask_known=known,mask_valid=valid,source=source,
                uncertainty=ambiguous,proposal_union=union_proposals.any(-1))
