"""Fixed SAM ensemble qualification. Scores are evidence, not certification."""
import numpy as np

POLICY=dict(score_floor=.50,score_window=.12,min_positive_coverage=.60,
            max_negative_coverage=.40,max_protected_fraction=.10)


def qualify_ensemble(masks,scores,*,positive_uv=(),negative_uv=(),protected_mask=None,policy=None):
    cfg=dict(POLICY if policy is None else policy)
    arrays=[np.asarray(m,bool) for m in masks]
    if not arrays:raise ValueError('at least one native pixel candidate required')
    shape=arrays[0].shape
    if any(a.shape!=shape for a in arrays):raise ValueError('candidate grids differ')
    scores=np.asarray(scores,float)
    if len(scores)!=len(arrays) or not np.isfinite(scores).all():raise ValueError('finite score per candidate required')
    protected=np.zeros(shape,bool) if protected_mask is None else np.asarray(protected_mask,bool)
    if protected.shape!=shape:raise ValueError('protected grid differs')
    def coverage(mask,points):
        p=np.asarray(points,float)
        if p.size==0:return None
        p=p.reshape(-1,2);valid=np.isfinite(p).all(1)
        p=np.floor(p[valid]).astype(int)
        if not len(p):return None
        inside=(p[:,0]>=0)&(p[:,1]>=0)&(p[:,0]<shape[1])&(p[:,1]<shape[0])
        hits=np.zeros(len(p),bool);hits[inside]=mask[p[inside,1],p[inside,0]]
        return float(hits.mean())
    records=[];qualified=[];unqualified=[]
    threshold=max(cfg['score_floor'],float(scores.max())-cfg['score_window'])
    for index,(mask,score) in enumerate(zip(arrays,scores)):
        positive=coverage(mask,positive_uv);negative=coverage(mask,negative_uv)
        conflict=float((mask&protected).sum()/max(int(mask.sum()),1))
        reasons=[]
        if score<threshold:reasons.append('outside_fixed_score_window')
        if positive is not None and positive<cfg['min_positive_coverage']:reasons.append('positive_support_conflict')
        if negative is not None and negative>cfg['max_negative_coverage']:reasons.append('negative_support_conflict')
        if conflict>cfg['max_protected_fraction']:reasons.append('protected_object_conflict')
        if not mask.any():reasons.append('empty_candidate')
        (unqualified if reasons else qualified).append(mask)
        records.append(dict(index=index,score=float(score),positive_coverage=positive,negative_coverage=negative,
                            protected_fraction=conflict,qualified=not reasons,reasons=reasons))
    union=np.logical_or.reduce(qualified) if qualified else np.zeros(shape,bool)
    intersection=np.logical_and.reduce(qualified) if qualified else np.zeros(shape,bool)
    possible=np.logical_or.reduce(unqualified) if unqualified else np.zeros(shape,bool)
    return dict(qualified_union=union,qualified_intersection=intersection,
                qualified_disagreement=union&~intersection,unqualified_possible=possible,
                receipt=dict(policy=cfg,threshold=threshold,candidates=records,qualified_count=len(qualified),
                             score_is_certification=False,missing_points_invalidates_accepted_mask=False))
