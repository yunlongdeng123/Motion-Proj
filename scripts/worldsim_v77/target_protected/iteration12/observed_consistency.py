"""有界可见轮廓反证：不接Y，不将隐藏或他车遮挡视为反证。

只剔除在合法可见帧中落到本实例轮廓外的cuboid代理点。
这是条件质量诊断，不增加表面、不填U、不代替补景网络。
"""
import numpy as np
import cv2
from pyquaternion import Quaternion
from actor_state import project_world


def consistent_actor_samples(observations,tracks,masks,samples,front):
    filtered={};stats={}
    for token,original in samples.items():
        n=len(original['local_xyz']);checked=np.zeros(n,np.int16);bad=np.zeros(n,np.int16)
        for t,(obs,a) in enumerate(zip(observations,tracks[token])):
            obs.validate()
            xyz=original['local_xyz']@Quaternion(a['rotation']).rotation_matrix.T+np.array(a['translation'])
            uv,z,good=project_world(xyz,obs);ids=np.flatnonzero(good)
            u,v=uv[ids].T
            # H边缘2px也不当负证据，避免灰洞/取整污染SAM判断。
            hole_halo=cv2.dilate(obs.hole.astype('uint8'),np.ones((5,5),np.uint8))>0
            visible=(~hole_halo[v,u])&(z[ids]<=front[t][v,u]+.15)
            ids=ids[visible];u,v=uv[ids].T
            # 2px只容纳采样stride/像素取整，不按case或Y调阈值。
            silhouette_halo=cv2.dilate(np.asarray(masks[token][t],np.uint8),np.ones((5,5),np.uint8))>0
            checked[ids]+=1;bad[ids]+=~silhouette_halo[v,u]
        keep=bad==0
        filtered[token]={k:(v[keep] if isinstance(v,np.ndarray) and v.ndim and len(v)==n else v) for k,v in original.items()}
        filtered[token]['observed_consistency_filter']='no_unoccluded_outside_silhouette_evidence'
        stats[token]={'input_samples':n,'retained_samples':int(keep.sum()),'contradicted_samples':int((~keep).sum()),
                      'no_testable_frame':int((checked==0).sum()),'testable_frame_mean':float(checked.mean()) if n else 0.,
                      'contradiction_frame_mean_removed':float(bad[~keep].mean()) if (~keep).any() else 0.,
                      'silhouette_halo_px':2,'hole_exclusion_halo_px':2,'depth_tolerance_m':.15}
    return filtered,stats


def checks():
    from actor_state import Observation
    h,w=24,32;K=np.array([[20,0,16],[0,20,12],[0,0,1.]])
    hole=np.zeros((h,w),bool);rgb=np.full((h,w,3),90,np.uint8)
    obs=Observation(rgb,hole,np.eye(4),K,0)
    actor={'rotation':[1,0,0,0],'translation':[0,0,0]}
    points=np.array([[0,0,10],[4,0,10]],np.float32)
    sample={'local_xyz':points,'rgb':np.full((2,3),90,np.uint8),'source_frame':np.zeros(2,np.int16),'source_pixel':np.array([[16,12],[24,12]])}
    mask=np.zeros((h,w),bool);mask[11:14,15:18]=True
    f=np.full((h,w),10.,np.float32)
    out,s=consistent_actor_samples([obs],{'B':[actor]},{'B':[mask]},{'B':sample},[f])
    assert len(out['B']['local_xyz'])==1 and s['B']['contradicted_samples']==1
    hole[:,22:]=True;rgb[hole]=127
    out,_=consistent_actor_samples([obs],{'B':[actor]},{'B':[mask]},{'B':sample},[f])
    assert len(out['B']['local_xyz'])==2,'洞中未知不得当无车反证'
    obs.hole=np.zeros_like(hole);obs.rgb[:]=90;f[:,22:]=4
    out,_=consistent_actor_samples([obs],{'B':[actor]},{'B':[mask]},{'B':sample},[f])
    assert len(out['B']['local_xyz'])==2,'他车遮挡不得当轮廓反证'
    assert np.array_equal(sample['local_xyz'],points),'必须保留原样本'
    print('OBSERVED_CONSISTENCY_CONTRACT_PASS: visible contradiction / H unknown / retained occlusion / immutable original')


if __name__=='__main__':checks()
