"""只读初始化验收：固定世界选区、共同网格和投影分母。"""
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image, ImageDraw


def in_region(xy, region, halo=0.):
    delta=np.asarray(xy)-np.asarray(region['center_xy'])
    axes=np.asarray(region['axes_xy'])
    # 矩形两条轴由真实三维框的水平投影正交化，不随查看相机改变。
    local=delta@axes
    return (abs(local)<=np.asarray(region['half_xy'])+halo+1e-9).all(1)


def footprint_regions(boxes, model_to_world, anchor, raw=102):
    A=np.linalg.inv(anchor)@model_to_world;out={}
    for box in boxes:
        if int(box['actor']) not in (5,37,6):continue
        center=A[:3,:3]@np.asarray(box['center'])+A[:3,3]
        axes=A[:3,:3]@np.asarray(box['axes']);u=axes[:2,0];u=u/np.linalg.norm(u)
        v=np.array([-u[1],u[0]])
        if v@axes[:2,1]<0:v=-v
        # 真实3D框的8个角投影到水平面，保守取包围矩形，含垂向倾斜外延。
        corners=np.array([[box['half'][j]*(-1 if i&(1<<j) else 1) for j in range(3)] for i in range(8)])@axes.T
        horizontal=np.c_[corners[:,:2]@u,corners[:,:2]@v]
        half=np.abs(horizontal).max(0)
        out[str(box['actor'])]=dict(actor=int(box['actor']),anchor_raw=raw,center_xy=center[:2].tolist(),axes_xy=np.stack([u,v],axis=1).tolist(),half_xy=half.tolist(),context_halo_m=1.2,
            definition='fixed world horizontal envelope of the raw102 measured vehicle OBB; core, plus separate1.2m context; not image car mask or hidden-road GT')
    return out


def grid_membership(xy, keys, cell=.2):
    ij=np.floor(np.asarray(xy)/cell+1e-8).astype(np.int64)
    lookup={tuple(v):i for i,v in enumerate(keys)}
    return np.array([lookup.get(tuple(v),-1) for v in ij],np.int32)


def region_grid_keys(region,halo=0.,cell=.2):
    c=np.asarray(region['center_xy']);r=np.linalg.norm(np.asarray(region['half_xy'])+halo)+cell*2
    keys=np.array([[i,j] for i in range(int(np.floor((c[0]-r)/cell)),int(np.ceil((c[0]+r)/cell))) for j in range(int(np.floor((c[1]-r)/cell)),int(np.ceil((c[1]+r)/cell)))],np.int64)
    return keys[in_region((keys+.5)*cell,region,halo)]


def projected_mask(model_corners, W, K, width=480,height=320):
    """所有模型共用的固定世界格子投影并集，不用预测alpha挑分母。"""
    cp=np.asarray(model_corners)@W[:3,:3].T+W[:3,3]
    # 横穿近裁面的格子保留为不可评，不拉成整屏；记录在调用方。
    valid=(cp[:,:,2]>.02).all(1)
    uv=cp[:,:,:2]/np.maximum(cp[:,:,2:],1e-20)*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]
    image=Image.new('1',(width,height));draw=ImageDraw.Draw(image)
    for polygon in uv[valid]:
        if polygon[:,0].max()<0 or polygon[:,0].min()>=width or polygon[:,1].max()<0 or polygon[:,1].min()>=height:continue
        draw.polygon([tuple(p) for p in polygon],fill=1)
    return np.asarray(image,bool),int((~valid).sum())


def pixel_metrics(alpha,depth,mask,capacity=None):
    n=int(mask.sum())
    if n==0:return dict(pixels=0,alpha_mean=None,alpha_p10=None,alpha_p50=None,coverage50=None,coverage95=None,capacity95=None,depth_mean_m=None)
    values=alpha[mask];ok=mask&(alpha>1e-6)&(depth>0)
    return dict(pixels=n,alpha_mean=float(values.mean()),alpha_p10=float(np.quantile(values,.1)),alpha_p50=float(np.median(values)),coverage50=float(np.mean(values>=.5)),coverage95=float(np.mean(values>=.95)),
        capacity95=float(np.mean(capacity[mask]>=.95)) if capacity is not None else None,depth_mean_m=float(depth[ok].mean()) if ok.any() else None)


def horizontal_footprints(probes,centers,covariances,chunk=2000):
    """真实协方差的水平投影3σ椭圆覆盖；几何代理，不冒充渲染alpha。"""
    if not len(centers):return np.zeros(len(probes),bool)
    cov=np.asarray(covariances);eig=np.linalg.eigvalsh(cov);r=3*np.sqrt(np.maximum(eig[:,-1],0));tree=cKDTree(probes);covered=np.zeros(len(probes),bool)
    inverse=np.linalg.inv(cov)
    for start in range(0,len(centers),chunk):
        matches=tree.query_ball_point(centers[start:start+chunk],r[start:start+chunk],workers=1)
        for j,ids in enumerate(matches):
            if not len(ids):continue
            i=start+j;ids=np.asarray(ids);delta=probes[ids]-centers[i]
            inside=np.einsum('ni,ij,nj->n',delta,inverse[i],delta)<=9+1e-9
            covered[ids[inside]]=True
    return covered
