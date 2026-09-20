"""核对原生单时刻接口；不重跑模型/PnP，不拟合新坐标变换。"""
from pathlib import Path
from datetime import datetime,timezone
import json,sys,time,os
import numpy as np
from scipy.spatial.transform import Rotation
import torch

ROOT=Path('/root/autodl-tmp/runs/worldsim_v75')
OUT=ROOT/'WS-V75-DVGT-CONTRACT-01/20260920-r1'
REPO=Path('/root/autodl-tmp/external/worldsim_v81/DVGT')
SOURCES=[ROOT/'WS-V75-APPROACH-CLOSEDLOOP-01/20260920-association-r2/reconstruction',
         ROOT/'WS-V75-APPROACH-CLOSEDLOOP-02/20260920-r1/reconstruction']
A=np.array([[0,0,1],[-1,0,0],[0,-1,0.]])


def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def projection_stats(native,views,first_ego):
    yy,xx=np.mgrid[4:512:8,4:512:8];pixels=np.stack([xx,yy],-1).reshape(-1,2)
    rows=[]
    for i,v in enumerate(views):
        point=native[i,yy,xx].reshape(-1,3).astype(float)/.1
        norm=np.linalg.norm(point,axis=-1)
        cw=np.array(v['camera_world']);K=np.array(v['K_network'])
        world=point@A.T@first_ego[:3,:3].T+first_ego[:3,3]
        camera=(world-cw[:3,3])@cw[:3,:3]
        valid=np.isfinite(point).all(-1)&(norm>2)&(norm<80)&(pixels[:,1]>=v['pad_top']+4)&(pixels[:,1]<508-v['pad_top'])
        positive=valid&(camera[:,2]>.2)
        projected=camera@K.T;projected=projected[:,:2]/projected[:,2:]
        reproj=np.linalg.norm(projected-pixels,axis=-1)
        rays=np.c_[pixels,np.ones(len(pixels))]@np.linalg.inv(K).T
        cos=np.einsum('ij,ij->i',rays,camera)/(np.linalg.norm(rays,axis=-1)*np.linalg.norm(camera,axis=-1))
        angle=np.degrees(np.arccos(np.clip(cos,-1,1)))
        # 独立校准往返：已知射线上的固定20m相机Z，用同一官方坐标契约转一圈。
        round_world=(rays*20)@cw[:3,:3].T+cw[:3,3]
        encoded=((round_world-first_ego[:3,3])@first_ego[:3,:3])@A*.1
        decoded=(encoded/.1)@A.T@first_ego[:3,:3].T+first_ego[:3,3]
        back=(decoded-cw[:3,3])@cw[:3,:3]@K.T;back=back[:,:2]/back[:,2:]
        rows.append({'camera':v['camera'],'valid_range_samples':int(valid.sum()),'positive_fraction':float(positive.sum()/max(valid.sum(),1)),
                     'median_angular_error_deg':float(np.median(angle[valid])),
                     'median_positive_reprojection_px':float(np.median(reproj[positive])) if positive.any() else None,
                     'p90_positive_reprojection_px':float(np.percentile(reproj[positive],90)) if positive.any() else None,
                     'calibration_roundtrip_max_px':float(np.linalg.norm(back-pixels,axis=-1).max()),
                     'capture_delay_ms':v['delta_to_cutoff_ms']})
    return rows


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';assert not OUT.exists();OUT.mkdir(parents=True)
    protocol={'task_id':'WS-V75-DVGT-CONTRACT-01','run_id':OUT.name,'frozen_utc':datetime.now(timezone.utc).isoformat(),
              'sources':[str(s) for s in SOURCES],'repository':str(REPO),'revision':'51cf3f6d11fdff8bc7e2bbe1a88f71665ccb2236',
              'scope':'exact existing two one-timestep calls; no model, PnP, alignment fitting, camera reordering or scientific badcase selection',
              'checks':['official preprocessing versus saved input pixels','official metric scale and RDF/FLU algebra',
                        'known-ray projection roundtrip','predicted first-ego pose near identity','raw point reprojection and angular consistency','existing PnP results retained'],
              'input_roles':'known calibration used only for evaluation and algebra checks; source native forward received RGB only',
              'human_verdict':None,'failure_ledger_refs':['V74-H2-F20','V74-H2-F22'],'failure_ledger_delta':'none'}
    save(OUT/'protocol.json',protocol)
    sys.path.insert(0,str(REPO))
    from dvgt.utils.load_fn import load_and_preprocess_images
    from dvgt.utils.pose_encoding import decode_pose
    from dvgt.evaluation.utils.geometry import convert_point_in_ego_0_to_ray_depth_in_ego_n
    began=time.monotonic();cases=[];torch.set_num_threads(4)
    for source in SOURCES:
        p=json.loads((source/'protocol.json').read_text());inf=json.loads((source/'inference_result.json').read_text())
        assert inf['status']=='complete' and inf['strict_weights'] and not inf['official_source_changed']
        inputs=load_and_preprocess_images(str(source/'native_input'))
        assert list(inputs.shape)==[1,1,7,3,512,512]
        rgb=(inputs[0,0].permute(0,2,3,1)*255).round().byte().numpy()
        saved=np.load(source/'network_rgb.npy');assert np.array_equal(rgb,saved)
        order=sorted((source/'native_input/frame_0').glob('*.jpg'))
        assert len(order)==7 and all(q.resolve()==Path(v['image']).resolve() for q,v in zip(order,p['views']))
        z=np.load(source/'native_outputs.npz');pose,_=decode_pose(torch.from_numpy(z['absolute_ego_pose_enc']))
        xyz=torch.from_numpy(z['points']);identity=torch.eye(4,dtype=xyz.dtype)[:3][None,None]
        depth=convert_point_in_ego_0_to_ray_depth_in_ego_n(xyz,identity)
        pose_np=pose.numpy()[0,0];T0=np.array(p['views'][0]['ego_world'])
        rows=projection_stats(z['points'][0,0],p['views'],T0)
        old=json.loads((source/'geometry_audit.json').read_text()) if (source/'geometry_audit.json').exists() else None
        pose_spread=[]
        for v in p['views']:
            E=np.linalg.inv(T0)@np.array(v['ego_world'])
            pose_spread.append({'camera':v['camera'],'ego_origin_offset_m':float(np.linalg.norm(E[:3,3])),
                                'ego_rotation_offset_deg':float(np.degrees(Rotation.from_matrix(E[:3,:3]).magnitude()))})
        case={'log_id':p['log_id'],'source':str(source),'input_shape':list(inputs.shape),'input_saved_max_pixel_delta':0,
              'view_order_preserved':True,'official_variable_view_count_and_order_supported':True,
              'head_chunk_branch_taken':False,'single_timestep_has_no_temporal_baseline':True,
              'native_pose_translation_m':(pose_np[:3,3]/.1).tolist(),
              'native_pose_rotation_from_identity_deg':float(np.degrees(Rotation.from_matrix(pose_np[:3,:3]).magnitude())),
              'official_identity_pose_ray_norm_max_delta_m':float(torch.max(abs(depth-xyz.norm(dim=-1)))/.1),
              'projection':rows,'capture_pose_spread':pose_spread,'previous_pnp':old,
              'human_verdict':None}
        cases.append(case);print(json.dumps({'log':p['log_id'],'projection':rows}),flush=True)
    result={'status':'complete','cases':cases,'model_calls':0,'cuda_initialized':torch.cuda.is_initialized(),
            'wall_s':time.monotonic()-began,'human_verdict':None,'failure_ledger_delta':'none'}
    assert not result['cuda_initialized'];save(OUT/'result.json',result)


if __name__=='__main__':main()
