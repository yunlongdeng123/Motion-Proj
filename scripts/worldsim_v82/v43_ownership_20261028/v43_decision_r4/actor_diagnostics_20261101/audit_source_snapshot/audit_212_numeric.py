"""Retain strict failure and measure Linux-cache/Windows FP32 vs FP64 control."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,math,importlib.util,struct
P=Path(__file__).resolve().parent;B=P.parent/'media';sys.path.insert(0,str(B/'vendor'))
import numpy as np
sp=importlib.util.spec_from_file_location('media_numeric_source',B/'build_media.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
C='14705303724557273004_3105_000_3125_000';OUT=B/'generated/scenes'/C
meta=json.loads((OUT/'media.json').read_text(encoding='utf-8'))
tm={x['timestamp_micros']:x['raw'] for x in meta['frames']}
remote=json.loads((B/'cache/waymo-train-212/lidar_decode_receipts.json').read_text(encoding='utf-8'))
original_receipts=json.loads((B/'cache/waymo-train-212/source_receipts.json').read_text(encoding='utf-8'))
original_links=[]
with (B/'cache/waymo-train-212/audit_original_records.tfrecord').open('rb') as stream:
    while header:=stream.read(12):
        length,crc=struct.unpack('<QI',header);payload=stream.read(length);tail=stream.read(4)
        f=m.pb.Frame.FromString(payload);raw=tm[f.timestamp_micros];r=original_receipts[raw]
        assert m.masked(header[:8])==crc and m.masked(payload)==struct.unpack('<I',tail)[0]==r['original_payload_masked_crc32c']
        assert length==r['payload_length'] and f.context.name==r['context_id']==C
        original_links.append(dict(raw=raw,original_header_offset=r['header_offset'],payload_bytes=length,original_record_payload_CRC_matches=True))

def reference64(frame):
    """Same point ordering/valid mask, geometric arithmetic independently in FP64.

    Production metadata quantization to FP32 is retained before promoting to64;
    inverse/trig/transform operations then run64 to isolate arithmetic loss.
    """
    lasers={x.name:x for x in frame.lasers};top=m.matrix(lasers[1].ri_return1.range_image_pose_compressed).astype(np.float64)
    roll,pitch,yaw=top[...,0],top[...,1],top[...,2]
    cr,sr,cp,ss,cy,sy=np.cos(roll),np.sin(roll),np.cos(pitch),np.sin(pitch),np.cos(yaw),np.sin(yaw)
    R=np.empty(top.shape[:2]+(3,3));R[...,0,0]=cy*cp;R[...,0,1]=cy*ss*sr-sy*cr;R[...,0,2]=cy*ss*cr+sy*sr
    R[...,1,0]=sy*cp;R[...,1,1]=sy*ss*sr+cy*cr;R[...,1,2]=sy*ss*cr-cy*sr
    R[...,2,0]=-ss;R[...,2,1]=cp*sr;R[...,2,2]=cp*cr
    pose32=np.asarray(frame.pose.transform,np.float32).reshape(4,4);world_inv=np.linalg.inv(pose32.astype(np.float64))
    clouds=[];sources=[];blocks=[]
    for ret in [0,1]:
        for cal in sorted(frame.context.laser_calibrations,key=lambda x:x.name):
            ri=getattr(lasers[cal.name],'ri_return'+str(ret+1));a=m.matrix(ri.range_image_compressed).astype(np.float64);h,w,_=a.shape
            if cal.beam_inclinations:incl=np.asarray(cal.beam_inclinations,np.float32).astype(np.float64)
            else:
                lo,hi=float(np.float32(cal.beam_inclination_min)),float(np.float32(cal.beam_inclination_max))
                incl=(.5+np.arange(h,dtype=np.float64))/h*(hi-lo)+lo
            incl=incl[::-1];ext=np.asarray(cal.extrinsic.transform,np.float32).reshape(4,4).astype(np.float64)
            az=((np.arange(w,0,-1,dtype=np.float64)-.5)/w*2-1)*math.pi-math.atan2(ext[1,0],ext[0,0])
            d=a[...,0];ci,si=np.cos(incl)[:,None],np.sin(incl)[:,None]
            xyz=np.stack((np.cos(az)[None]*ci*d,np.sin(az)[None]*ci*d,si*np.ones((1,w))*d),axis=-1)
            xyz=np.einsum('kr,ijr->ijk',ext[:3,:3],xyz)+ext[:3,3]
            if cal.name==1:
                xyz=np.einsum('hwij,hwj->hwi',R,xyz)+top[...,3:]
                xyz=np.einsum('ij,hwj->hwi',world_inv[:3,:3],xyz)+world_inv[:3,3]
            valid=(d>0)&(a[...,3]!=1);points=xyz[valid]
            clouds.append(points);sources.append(np.full(len(points),cal.name,np.int8));blocks.append(dict(laser_name=cal.name,return_index=ret,retained_points=len(points)))
    return np.concatenate(clouds),np.concatenate(sources),blocks

data={k:[] for k in ['cache_minus_windows32','cache_minus_reference64','windows32_minus_reference64']}
sensor={s:{k:[] for k in data} for s in ['TOP','side_LiDAR']};perframe=[];selected_actor_rows=0
for _,frame,_ in m.records(B/'cache/waymo-train-212/audit_original_records.tfrecord'):
    raw=tm[frame.timestamp_micros];cloud,details=m.decoder.lidar(frame,m.pb);ref,lasers,blocks=reference64(frame)
    assert len(ref)==len(cloud)==remote[raw]['retained_points']
    for a,b,c in zip(details,blocks,remote[raw]['sensors']):
        assert (a['laser_name'],a['return_index'],a['retained_points'])==(b['laser_name'],b['return_index'],b['retained_points'])==(c['laser_name'],c['return_index'],c['retained_points'])
    labels={l.id:l for l in frame.laser_labels if l.type==1};frame_errors=[]
    for track,actor in meta['actors'].items():
        with np.load(B/'generated'/actor['lidar']['npz']) as z:
            q=z['raw_frame']==raw
            if not q.any():continue
            src=z['source_cloud_row'][q];saved=z['actor_local_xyz'][q].astype(np.float64);box=labels[track].box
            co,si=math.cos(box.heading),math.sin(box.heading);rot=np.array([[co,-si,0],[si,co,0],[0,0,1.]])
            center=np.array([box.center_x,box.center_y,box.center_z]);win=(cloud[src,:3]-center)@rot;high=(ref[src]-center)@rot
            values=dict(cache_minus_windows32=saved-win,cache_minus_reference64=saved-high,windows32_minus_reference64=win-high)
            for key,delta in values.items():
                data[key].append(delta)
                for name,mask in [('TOP',lasers[src]==1),('side_LiDAR',lasers[src]!=1)]:
                    if mask.any():sensor[name][key].append(delta[mask])
            frame_errors.append(values['cache_minus_windows32']);selected_actor_rows+=len(saved)
    d=np.concatenate(frame_errors);translation=np.array(frame.pose.transform).reshape(4,4)[:3,3]
    perframe.append(dict(raw=raw,actor_rows=len(d),max_abs_cache_windows_m=float(np.abs(d).max()),
        world_ego_translation_m=translation.tolist(),FP32_translation_spacing_m=np.spacing(np.abs(translation).astype(np.float32)).astype(float).tolist()))

def describe(arrays):
    if not arrays:return None
    a=np.concatenate(arrays);v=np.max(np.abs(a),axis=1)
    return dict(actor_measurement_rows=len(a),max_abs_component_m=float(v.max()),median_max_abs_component_m=float(np.median(v)),
        p99_max_abs_component_m=float(np.quantile(v,.99)),max_abs_by_actor_axis_m=np.abs(a).max(0).tolist(),
        mean_signed_by_actor_axis_m=a.mean(0).tolist(),rows_over_strict_2e_5=int((v>2e-5).sum()))

result=dict(schema='cross_runtime_original_LiDAR_numeric_control_v1',context_id=C,status='STRICT_2E_5_PARITY_FAIL_RETAINED',
    strict_original_tolerance_m=2e-5,original_control_frames=[0,99,198],
    actor_measurement_rows=selected_actor_rows,metrics={k:describe(v) for k,v in data.items()},
    by_sensor={s:{k:describe(v) for k,v in a.items()} for s,a in sensor.items()},perframe=perframe,
    source_row_order_and_all10_sensor_return_counts_exactly_match=True,
    original_audit_record_payload_links=original_links,
    precision_reference='FP64 arithmetic with original production metadata FP32 quantization retained; independent Euler and transform evaluation',
    explanation='Millimetre-scale cross-runtime differences are measured separately from exact raw/source-index identity. TOP world-space transforms contain km-scale translations and cancellation. No nearest-neighbor reassignment or point replacement.',
    float32_budget='FP32 translation ULP scales are recorded per axis. Several dot/inverse/trig roundings may accumulate a small multiple of that scale; this is an explanatory scale budget, not a substitute parity guarantee.',
    new_acceptance_threshold=None,formal_geometry_gate=None,causal_badcase_label=None,GPU_queries=0,remote_queries=0)
(P/'media212_cross_runtime_numeric_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result['metrics']))
print(json.dumps({'sensor_strict_rows':{k:v['cache_minus_windows32']['rows_over_strict_2e_5'] for k,v in result['by_sensor'].items()},'perframe':perframe}))
