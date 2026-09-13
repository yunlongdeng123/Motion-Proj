"""V81 只读几何审计。相机深度为 optical-axis z，射线距离单独计算。"""
import numpy as np
from scipy.spatial import cKDTree

def transform(row):
    q=np.asarray(row['rotation'],dtype=float); w,x,y,z=q/np.linalg.norm(q)
    R=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    T=np.eye(4); T[:3,:3]=R; T[:3,3]=row['translation']; return T

def apply(points,T): return points@T[:3,:3].T+T[:3,3]

def project(world,T,K):
    xyz=apply(world,np.linalg.inv(T)); z=xyz[:,2]; h=xyz@K.T
    uv=h[:,:2]/np.maximum(z[:,None],1e-8); return uv,z,xyz

def remove_boxes(points,boxes,margin=.35):
    keep=np.ones(len(points),bool)
    for b in boxes:
        # nuScenes size 是 width,length,height；局部 x 是 length。
        p=apply(points,np.linalg.inv(transform(b)))
        half=np.asarray(b['size'])[[1,0,2]]/2+margin
        keep &= ~np.all(np.abs(p)<=half,axis=1)
    return points[keep]

def plane_fit(points,seed=8101):
    if len(points)<12: return None
    rng=np.random.default_rng(seed); best=None
    for _ in range(48):
        a,b,c=points[rng.choice(len(points),3,replace=False)]
        normal=np.cross(b-a,c-a); norm=np.linalg.norm(normal)
        if norm<1e-9: continue
        normal/=norm; inlier=np.abs((points-a)@normal)<.15
        if best is None or inlier.sum()>best.sum(): best=inlier
    if best is None or best.sum()<12:return None
    center=points[best].mean(0); _,_,vh=np.linalg.svd(points[best]-center,full_matrices=False)
    n=vh[-1]; residual=np.abs((points-center)@n)
    return {'center':center,'normal':n,'residual':residual,'inlier':best,
            'rms':float(np.sqrt(np.mean(residual[best]**2))), 'inlier_fraction':float(best.mean())}

def coverage(uv,box,bins=6):
    if not len(uv):return 0.
    x0,y0,x1,y1=box
    ids=np.floor((uv-[x0,y0])/[x1-x0,y1-y0]*bins).astype(int)
    valid=np.all((ids>=0)&(ids<bins),axis=1); ids=ids[valid]
    return len(np.unique(ids[:,0]+bins*ids[:,1]))/(bins*bins)

def roi_mask(uv,z,box):
    x0,y0,x1,y1=box
    return (z>1)&(uv[:,0]>=x0)&(uv[:,0]<x1)&(uv[:,1]>=y0)&(uv[:,1]<y1)

def reference_filter(uv,z,source,width,height,cell=8):
    """同像素格跨至少两个独立扫描一致；保留最近支撑，深度边界拒绝。"""
    valid=roi_mask(uv,z,[0,0,width,height])&(z<80)
    ids=np.flatnonzero(valid); bins=np.floor(uv[ids]/cell).astype(int)
    key=bins[:,0]+((width+cell-1)//cell)*bins[:,1]
    order=np.argsort(key); ids=ids[order]; key=key[order]
    out=[]
    for group in np.split(ids,np.flatnonzero(np.diff(key))+1):
        if not len(group) or len(np.unique(source[group]))<2:continue
        zs=z[group]; median=np.median(zs)
        if np.ptp(zs)>.5+.01*median:continue
        out.extend(group.tolist())
    return np.asarray(out,dtype=int)

def depth_metrics(pred,ref):
    pred=np.asarray(pred); ref=np.asarray(ref); total=len(ref)
    ok=np.isfinite(pred)&(pred>0)&np.isfinite(ref)&(ref>0)
    if not ok.any(): return {'support':total,'valid':0,'coverage':0.,'absrel':None,'rmse_m':None,'mae_m':None,'delta125':None}
    p=pred[ok]; g=ref[ok]; e=p-g
    return {'support':total,'valid':int(ok.sum()),'coverage':float(ok.mean()),'absrel':float(np.mean(abs(e)/g)),
            'rmse_m':float(np.sqrt(np.mean(e*e))),'mae_m':float(np.mean(abs(e))), 'delta125':float(np.mean(np.maximum(p/g,g/p)<1.25))}

def interaction_interval(rows,seed=8101,draws=2000):
    """log 均值同权；只使用四格都有观测的独立 log。"""
    from collections import defaultdict
    logs=defaultdict(lambda:defaultdict(list))
    for r in rows:
        if r.get('error') is not None: logs[r['log']][r['cohort']].append(r['error'])
    effects=[]
    for cells in logs.values():
        if all(c in cells for c in ['C00','C10','C01','C11']):
            e={c:np.mean(v) for c,v in cells.items()}
            effects.append(e['C11']-e['C10']-e['C01']+e['C00'])
    if len(effects)<3:return {'n_logs':len(effects),'interaction':None,'ci95':None,'status':'INSUFFICIENT_MATCHED_LOGS'}
    effects=np.asarray(effects); rng=np.random.default_rng(seed)
    boot=effects[rng.integers(0,len(effects),(draws,len(effects)))].mean(1)
    return {'n_logs':len(effects),'interaction':float(effects.mean()),'positive_fraction':float((effects>0).mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),'status':'DISCOVERY_ASSOCIATION'}
