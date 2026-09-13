"""普通正比例尺度校准诊断；只访问INPUT LiDAR，目标ROI外拟合，不改原模型。"""
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates
from motion_proj.worldsim_v81.geometry import transform,apply,project,remove_boxes,depth_metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
RUN=Path('/root/autodl-tmp/runs/worldsim_v81/WS-V81-GPU-P2-01');ATLAS=Path('/root/autodl-tmp/runs/worldsim_v81/WS-V81-CPU-01/20260913-cpu-r2');OUT=RUN/'analysis'
IDX=json.loads((ATLAS/'index.json').read_text());REG={r['roi_id']:r for r in map(json.loads,(ATLAS/'v81_roi_registry.jsonl').read_text().splitlines())};rows=list(map(json.loads,(RUN/'evaluation/metrics.jsonl').read_text().splitlines()))
ids=set(json.loads((OUT/'exploratory_review_index.json').read_text()))|{r['roi_id'] for r in json.loads((OUT/'anchor_diagnostics.json').read_text())}
world_cache={};output=[]
for row in rows:
    if row['method']!='vggt' or row['roi_id'] not in ids or row['texture_case']:continue
    if row['output_key']!='full6' and not row['anchor_camera']:continue
    roi=REG[row['roi_id']];manifest=json.loads((RUN/'input_manifests'/f'{row["window"]}.json').read_text());s=manifest['target_sample']
    if s not in world_cache:
        lidar=IDX['sample_data'][s]['LIDAR_TOP'];xyz=np.fromfile(Path(IDX['root'])/lidar['filename'],dtype=np.float32).reshape(-1,5)[:,:3]
        T=transform(IDX['poses'][lidar['ego_pose_token']])@transform(IDX['calibrated'][lidar['calibrated_sensor_token']]);xyz=apply(xyz,T);xyz=remove_boxes(xyz,IDX['annotations'].get(s,[]));world_cache[s]=xyz
    xyz=world_cache[s];target=next(v for v in manifest['views'] if v['camera']==roi['camera']);uv,z,_=project(xyz,np.array(target['world_from_camera']),np.array(target['K']));x0,y0,x1,y1=roi['box']
    outside=~((z>0)&(uv[:,0]>=x0-16)&(uv[:,0]<x1+16)&(uv[:,1]>=y0-16)&(uv[:,1]<y1+16));xyz=xyz[outside]
    rp=Path(row['result_path']);result=json.loads(rp.read_text());ratios=[];n_camera=0
    for view in result['views']:
        # 只用target时刻的输入相机；temporal18也不触碰HELDOUT时刻LiDAR。
        camera=next(v for v in manifest['views'] if v['camera']==view['camera']);uv,z,_=project(xyz,np.array(camera['world_from_camera']),np.array(camera['K']));ok=(z>1)&(z<80)&(uv[:,0]>=0)&(uv[:,0]<1600)&(uv[:,1]>=0)&(uv[:,1]<900)
        uv=uv[ok];z=z[ok]
        if not len(z):continue
        # 8px bin仅取当前帧最近回波，减少重复远层的尺度污染。
        cell=(uv[:,1]//8).astype(int)*200+(uv[:,0]//8).astype(int);order=np.argsort(z);_,first=np.unique(cell[order],return_index=True);keep=order[first];uv=uv[keep];z=z[keep]
        A=np.array(view['original_to_network_pixel_center']);q=np.c_[uv,np.ones(len(uv))]@A.T;depth=np.load(rp.parent/(view['camera']+'_depth_z_m.npy'));pred=map_coordinates(depth,[q[:,1],q[:,0]],order=1,mode='constant',cval=np.nan);ok=np.isfinite(pred)&(pred>0)
        ratios.extend((z[ok]/pred[ok]).tolist());n_camera+=bool(ok.sum())
    record={'roi_id':row['roi_id'],'window':row['window'],'log':row['log'],'output_key':row['output_key'],'variant':row['variant'],'role':'INPUT_LIDAR_SCALE_DIAGNOSTIC','method':'VGGT_PLUS_OUTSIDE_ROI_INPUT_LIDAR_GLOBAL_SCALE','native_model_unchanged':True,'fit_source':'current LIDAR_TOP; inflated annotations removed; target ROI plus 16px projected volume excluded before all camera projections','fit_points':len(ratios),'fit_cameras':n_camera,'target_input_sample':s,'heldout_used_for_fit':False,'baseline_mae_m':row['mae_m'],'baseline_absrel':row['absrel']}
    if len(ratios)<100:record['status']='INSUFFICIENT_OUTSIDE_INPUT_SUPPORT';output.append(record);continue
    scale=float(np.median(ratios));ref=np.load(ATLAS/'reference_geometry'/f'{row["roi_id"]}.npz');view=next(v for v in result['views'] if v['camera']==roi['camera']);A=np.array(view['original_to_network_pixel_center']);q=np.c_[ref['uv'],np.ones(len(ref['uv']))]@A.T;depth=np.load(rp.parent/(roi['camera']+'_depth_z_m.npy'));pred=map_coordinates(depth,[q[:,1],q[:,0]],order=1,mode='constant',cval=np.nan)*scale
    record.update(status='DONE',global_scale_correction=scale,**depth_metrics(pred,ref['depth_z']));output.append(record)
(OUT/'outside_roi_scale_control.json').write_text(json.dumps(output,indent=2))
chosen=['scene-0626_9a9c05fe_CAM_BACK_LEFT_13','scene-0632_fd5b6a5c_CAM_FRONT_RIGHT_02','scene-0139_7e27d5c0_CAM_FRONT_LEFT_10'];variants=['sparse2','sparse3','full6','temporal18']
fig,axs=plt.subplots(1,3,figsize=(13,4))
for ax,rid in zip(axs,chosen):
    rr={r['variant']:r for r in output if r['roi_id']==rid and r['output_key'].startswith('anchor_')}
    ax.plot(range(4),[rr[v]['baseline_mae_m'] for v in variants],'o-',label='Camera-baseline scale',color='#c66849');ax.plot(range(4),[rr[v]['mae_m'] for v in variants],'s-',label='Outside-ROI input LiDAR scale',color='#2978a3');ax.set_xticks(range(4),['2 cams','3 cams','6 cams','18 temporal'],rotation=15);ax.set_ylabel('Held-out ROI depth MAE (m)');ax.set_title(rid.split('_CAM')[0]);ax.grid(alpha=.2);ax.legend(fontsize=7)
fig.suptitle('Simple global-scale control | target-region input points excluded | no held-out fitting');fig.tight_layout(rect=(0,0,1,.92));fig.savefig(OUT/'figures/outside_roi_scale_control.png',dpi=150,bbox_inches='tight');plt.close(fig)
print(json.dumps([r for r in output if r['roi_id'] in chosen and r['output_key'].startswith('anchor_')],indent=2))
