"""半核远端单worker顺序实测点裁剪。省去全部环境点/无关JPEG的跨网传输。"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import io,json,pathlib,sys,time,math,struct,tarfile
import numpy as np
from scipy.spatial import cKDTree
BASE=pathlib.Path('/root/autodl-tmp/codex_tmp/v43_actor_diagnostics_20261101')
sys.path.insert(0,str(BASE))
import waymo_numpy_cpu as decoder
from fast_range_matrix import matrix
decoder.matrix=matrix;pb=decoder.protobuf()

def run(key):
    scene=next(x for x in json.loads((BASE/'scenes.json').read_text())['scenes'] if x['scene_key']==key)
    inv=next(x for x in json.loads((BASE/'raw_inventory.json').read_text()) if x['scene_key']==key)
    out=BASE/'cropped'/key;out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():return
    start=time.monotonic();actors={};receipts=[];details=[];total=0;before=pathlib.Path(inv['path']).stat()
    samples={0,scene['expected_frames']//2,scene['expected_frames']-1};crc=decoder.crc_function()
    with (out/'audit_original_records.tfrecord').open('xb') as audit:
        for raw,payload,receipt in decoder.records(inv['path'],scene['expected_frames']-1):
            frame=pb.Frame.FromString(payload)
            if frame.context.name!=scene['context_id']:raise ValueError('context')
            if raw in samples:
                header=struct.pack('<Q',len(payload));audit.write(header+struct.pack('<I',crc(header))+payload+struct.pack('<I',crc(payload)))
            cloud,sensors=decoder.lidar(frame,pb);xyz=cloud[:,:3];tree=cKDTree(xyz);total+=len(xyz)
            frame_rows=0
            for label in frame.laser_labels:
                if label.type!=1:continue
                if label.id not in scene['actors']:raise ValueError('unknown track')
                b=label.box;c=np.array([b.center_x,b.center_y,b.center_z]);dim=np.array([b.length,b.width,b.height])
                co,si=math.cos(b.heading),math.sin(b.heading);R=np.array([[co,-si,0],[si,co,0],[0,0,1.]])
                idx=np.asarray(tree.query_ball_point(c,float(np.linalg.norm(dim)/2+1e-4)),dtype=np.int32)
                local=(xyz[idx]-c)@R if len(idx) else np.empty((0,3));inside=np.all(np.abs(local)<=dim/2+1e-5,axis=1)
                local=local[inside].astype(np.float32);idx=idx[inside];frame_rows+=len(idx)
                a=actors.setdefault(label.id,dict(alias=scene['actors'][label.id]['alias'],xyz=[],raw=[],source=[],intensity=[],elongation=[],labelled=0,counts=[]))
                a['labelled']+=1;a['counts'].append(dict(raw=raw,points=len(idx)))
                if len(idx):
                    a['xyz'].append(local);a['raw'].append(np.full(len(idx),raw,dtype=np.uint16));a['source'].append(idx)
                    a['intensity'].append(cloud[idx,3]);a['elongation'].append(cloud[idx,4])
            receipts.append(dict(raw=raw,timestamp_micros=int(frame.timestamp_micros),context_id=frame.context.name,
                                 original_payload_masked_crc32c=crc(payload),**receipt))
            details.append(dict(raw=raw,retained_points=len(xyz),box_rows=frame_rows,sensors=sensors))
            if raw%40==0:print(json.dumps(dict(scene=key,raw=raw,elapsed=time.monotonic()-start,box_rows=frame_rows)),flush=True)
    summaries={}
    for track,a in actors.items():
        arrays=dict(actor_local_xyz=np.concatenate(a['xyz']) if a['xyz'] else np.empty((0,3),np.float32),
            raw_frame=np.concatenate(a['raw']) if a['raw'] else np.empty(0,np.uint16),
            source_cloud_row=np.concatenate(a['source']) if a['source'] else np.empty(0,np.int32),
            intensity=np.concatenate(a['intensity']) if a['intensity'] else np.empty(0,np.float32),
            elongation=np.concatenate(a['elongation']) if a['elongation'] else np.empty(0,np.float32))
        name='actor_'+a['alias']+'.npz';np.savez_compressed(out/name,**arrays)
        summaries[track]=dict(alias=a['alias'],file=name,points=len(arrays['raw_frame']),labelled_frames=a['labelled'],per_frame_counts=a['counts'])
    assert set(actors)==set(scene['actors'])
    after=pathlib.Path(inv['path']).stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    for name,obj in [('source_receipts.json',receipts),('lidar_decode_receipts.json',details),('actors.json',summaries)]:
        (out/name).write_text(json.dumps(obj,separators=(',',':')))
    summary=dict(schema='actual_actor_box_lidar_cache_v1',scene_key=key,context_id=scene['context_id'],source_path=inv['path'],source_bytes=before.st_size,
        frame_count=len(receipts),actors=len(actors),measured_sensor_points=total,measured_box_rows=sum(x['points'] for x in summaries.values()),
        wall_seconds=time.monotonic()-start,source_size_mtime_unchanged=True,CRC_all_records='passed',audit_original_raw_frames=sorted(samples),
        CPU_threads=1,GPU_queries=0,source_indices='concatenated decoder.lidar row: return0 lasers1..5 then return1 lasers1..5; valid non-NLZ row order',
        coordinates='actor_local_x_forward_y_left_z_up_per_frame_raw_box_center',
        limitation='OBB points may contain road/occluders; TOP per-pixel ego compensation only; within-scan actor motion uncompensated; no completion')
    (out/'manifest.json').write_text(json.dumps(summary,indent=2))
    # uncompressed tar avoids wasting half-core CPU compressing already deflated NPZ.
    with tarfile.open(BASE/'cropped'/(key+'.tar.tmp'),'w') as tar:
        for p in sorted(out.iterdir()):tar.add(p,arcname=p.name)
    (BASE/'cropped'/(key+'.tar.tmp')).rename(BASE/'cropped'/(key+'.tar'))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    cfg=json.loads((BASE/'scenes.json').read_text());order=['waymo-train-212','waymo-train-537','waymo-train-564','waymo-train-008']
    order +=[s['scene_key'] for s in cfg['scenes'] if s['scene_key'] not in order and s['scene_key']!='waymo-val-035']
    for key in order:run(key)
