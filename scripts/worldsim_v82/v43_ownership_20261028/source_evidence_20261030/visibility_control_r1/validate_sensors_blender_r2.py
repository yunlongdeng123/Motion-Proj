exec(open('/tmp/v43_visibility_precheck.py',encoding='utf-8-sig').read().split("trajectories=")[0].replace("out.mkdir(exist_ok=False)","out.mkdir(exist_ok=True)"))
points=[];frameids=[];semantics=[];validation=[]
for f in range(16):
 g.set_frame(f,cfg,cam,car,'none',True);geom.build_bvh();cal=g.camera_metadata(f,cfg,cam);c2w=np.asarray(cal['camera_to_world']);l=np.load(ROOT/f'dataset_r2/observations/lidar/{f:06d}.npz');delta=l['xyz_world']-c2w[:3,3];ds=np.linalg.norm(delta,axis=1);hit=geom.cast(c2w[:3,3],delta/ds[:,None],150)
 err=np.linalg.norm(hit['xyz_world']-l['xyz_world'],axis=1)
 assert np.max(abs(l['sensor_to_world']-c2w))==0
 a=np.load(ROOT/f'dataset_r2/privileged/original/{f:06d}.npz');yx=np.argwhere(a['valid'])[::701];p=a['xyz_world'][yx[:,0],yx[:,1]];d=p-c2w[:3,3];h=geom.cast(c2w[:3,3],d/np.linalg.norm(d,axis=1)[:,None],150);e=np.linalg.norm(h['xyz_world']-p,axis=1)
 validation.append({'frame':f,'lidar_points':len(ds),'lidar_world_replay_max_error_m':float(err.max()),'lidar_replay_above_1cm_count':int((err>.01).sum()),'lidar_replay_p99_m':float(np.quantile(err,.99)),'sensor_camera_max_abs':0,'dense_replay_samples':len(p),'dense_replay_max_error_m':float(e.max()),'road_lane_hits':int(np.isin(hit['semantic_id'],[1,2]).sum())})
 points.append(l['xyz_world']);semantics.append(np.where(err<=.01,hit['semantic_id'],-1));frameids.append(np.full(len(ds),f))
np.savez_compressed(out/'lidar_replayed.npz',xyz=np.concatenate(points),semantic=np.concatenate(semantics),frame=np.concatenate(frameids))
g.dump_json(out/'sensor_validation.json',{'validation':validation,'lidar_support_definition':'actual sparse sensor returns, same static road/lane firsthit from original world replay; world Euclidean point proximity reported downstream','camera_coordinate_parity':'original 16 c2w/K replay checked against original observations metadata','note':'surface geometry visibility is not appearance/shadow-free evidence'})
