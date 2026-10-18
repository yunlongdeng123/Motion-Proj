"""Read-only CPU ownership/geometry supplement for the completed old five-ID baseline."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='2'
import sys,base64
from identity_common import *
import torch
torch.set_num_threads(2)

def quant(x):return dict(zip(['min','p50','p95','p99','max'],map(float,np.quantile(x,[0,.5,.95,.99,1]))))

def main():
    ck=TASK/'train5/checkpoint_030000.pt';before=stat(ck)
    saved=torch.load(ck,map_location='cpu',weights_only=False);assert saved['step']==30000
    old=saved['instances'];state=old['state_dict'];seed=dict(np.load(TASK/'prepared5/instances.npz'))
    scale=float(seed['world_scale']);xyz=state['_xyz'].numpy()/scale;initial=seed['local_xyz']/scale
    sigma=state['_scaling'].numpy();sigma=np.exp(sigma)/scale
    owner=state['point_ids'].numpy();assert np.array_equal(owner,seed['point_ids'])
    half=np.asarray(read(R15/'box_half_sizes.json')['half_sizes_m']);rng=np.random.default_rng(20261018)
    assert list(old['instance_ids'])==seed['instance_ids'].tolist()
    record=[];panels=[];tolerance=1e-5
    for idx,actor in enumerate(ACTORS):
        rows=np.flatnonzero(owner==idx);x=xyz[rows];x0=initial[rows]
        source_frames=seed['source_frame_indices'][rows];lidar=seed['source_lidar_indices'][rows]
        ownhalf=np.median(seed['box_size'][seed['valid'][:,idx],idx],axis=0)/(2*scale)
        item=dict(actor=actor,old_owner_index=idx,track_id=old['instance_ids'][idx],initial_points=len(rows),final_points=len(rows),
            fixed_topology=True,centers_outside_common_R15_box=int((np.abs(x)>half[actor]).any(-1).sum()),
            initial_centers_outside_common_R15_box=int((np.abs(x0)>half[actor]).any(-1).sum()),
            centers_outside_native_median_box=int((np.abs(x)>ownhalf).any(-1).sum()),
            centers_outside_common_box_with_tolerance=int((np.abs(x)>half[actor]+tolerance).any(-1).sum()),
            initial_centers_outside_common_box_with_tolerance=int((np.abs(x0)>half[actor]+tolerance).any(-1).sum()),
            native_half_box_m=ownhalf.tolist(),common_half_box_m=half[actor].tolist(),native_vs_common_half_max_difference_m=float(np.max(np.abs(ownhalf-half[actor]))),
            displacement_from_initial_m=quant(np.linalg.norm(x-x0,axis=-1)),max_sigma_m=quant(sigma[rows].max(-1)),
            evidence_status_available=False,unknown_state_available=False)
        record.append(item)
        ids=np.sort(rng.choice(rows,8000,replace=False)) if len(rows)>8000 else rows
        data=np.column_stack((xyz[ids],np.ones(len(ids)),np.full(len(ids),actor),sigma[ids].max(-1),seed['source_frame_indices'][ids],
            seed['source_lidar_indices'][ids],np.full(len(ids),actor),np.full((len(ids),3),-1))).astype('<f4')
        panels.append(dict(stage='old5',scope='a'+str(actor),coordinates='Actor-local meters; old normalized coordinates / actual world_scale',
            total=len(rows),counts={'owner1':len(rows)},half_box_m=half[actor].tolist(),
            evidence_status_is_synthetic_display=True,probability_available=False,
            points=dict(rows=len(ids),stride=12,data=base64.b64encode(data.tobytes()).decode())))
    assert before==stat(ck)
    rck=R15/'paused_checkpoint_004000_20261017/checkpoint.pt';rbefore=stat(rck)
    rstate=torch.load(rck,map_location='cpu',weights_only=False)['state'];r15records=[]
    for actor in ACTORS:
        take=rstate['field.owner']==actor;x=rstate['field.xyz'][take].numpy()
        r15records.append(dict(actor=actor,points=len(x),centers_outside_common_box_with_tolerance=int((np.abs(x)>half[actor]+tolerance).any(-1).sum()),
            centers_outside_common_box_exact=int((np.abs(x)>half[actor]).any(-1).sum())))
    assert rbefore==stat(rck)
    out=TASK/'old5_geometry';out.mkdir(exist_ok=True)
    report=dict(status='COMPLETE_READ_ONLY',checkpoint=before,checkpoint_step=30000,parameter_updates=0,
        actual_normalization_world_scale=scale,actors=record,R15_frozen4k_geometry=r15records,
        common_box_tolerance_m=tolerance,tolerance_reason='10 micrometers avoids treating FP32 normalized-to-meter boundary roundoff as physical crossing; exact and native counts retained',
        semantics='Old points have actual hard owner but no ternary evidence buffer. Viewer green means selected owner only; -1 means unavailable probability.',
        coordinate_comparison='same physical Actor-local length/width/height axes; normalized old coordinates divided by actual measured similarity scale',
        common_box='R15 fixed train46 median genuine box dimensions; native old49 median differences separately recorded',
        caveat='Box crossing and center density do not establish erroneous road ownership; point counts reflect different initialization/topology.',human_verdict=None)
    write(out/'geometry.json',report);write(out/'viewer_panels.json',dict(panels=panels))
    print(json.dumps(report))

if __name__=='__main__':main()
