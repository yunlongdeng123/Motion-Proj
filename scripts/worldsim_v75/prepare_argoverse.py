"""Argoverse 2 开发输入桥接：真实RGB、标定、地图和GT轨迹，尚非重建输出。"""
import argparse
import json
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path('/root/autodl-tmp/data/av2/sensor/val')
LOG = '02678d04-cc9f-3148-9f95-1ba66347dff9'
OUT = Path('/root/autodl-tmp/runs/worldsim_v75/WS-V75-AV2-BRIDGE-01/20260920-r1')
CAMERA = 'ring_front_center'
N, W, H = 237, 1280, 704
QUAT = ['qx', 'qy', 'qz', 'qw']
POS = ['tx_m', 'ty_m', 'tz_m']
CORNERS = np.asarray(list(itertools.product([-0.5, 0.5], repeat=3)))
EDGES = [(i,j) for i in range(8) for j in range(i+1,8)
         if np.count_nonzero(CORNERS[i] != CORNERS[j]) == 1]

def poses(df, query_ns):
    df = df.sort_values('timestamp_ns').drop_duplicates('timestamp_ns')
    times = df.timestamp_ns.to_numpy(np.int64)
    query_ns = np.asarray(query_ns, dtype=np.int64)
    assert query_ns.min() >= times[0] and query_ns.max() <= times[-1]
    src, dst = (times-times[0])/1e9, (query_ns-times[0])/1e9
    result = np.repeat(np.eye(4)[None], len(query_ns), axis=0)
    result[:,:3,:3] = Slerp(src, Rotation.from_quat(df[QUAT].to_numpy()))(dst).as_matrix()
    for axis, name in enumerate(POS):
        result[:,axis,3] = np.interp(dst, src, df[name])
    return result

def xyz(points):
    return np.asarray([[p['x'],p['y'],p['z']] for p in points], dtype=np.float64)

def crop_image(path, crop):
    with Image.open(path) as im:
        return im.convert('RGB').resize((W,H), Image.Resampling.LANCZOS, box=tuple(crop))

def project_track(track, f, camera_world, K):
    if f not in track['frames']:
        return None
    i = track['frames'].index(f)
    rotation = Rotation.from_quat(track['quaternions'][i]).as_matrix()
    world = (CORNERS*np.asarray(track['dimensions'])) @ rotation.T + track['centers'][i]
    camera = (world-camera_world[f,:3,3]) @ camera_world[f,:3,:3]
    if camera[:,2].min() <= 0.1:
        return None
    uv = camera[:,:2]/camera[:,2,None]*K[:2] + K[2:]
    b = np.r_[uv.min(0), uv.max(0)]
    return {'corners':uv.tolist(), 'bounds':b.tolist(),
            'fully_inside':bool(b[0]>=0 and b[1]>=0 and b[2]<W and b[3]<H),
            'large_enough':bool(b[2]-b[0]>=16 and b[3]-b[1]>=12),
            'depth_m':float(camera[:,2].mean())}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT)
    a = parser.parse_args()
    out, raw = a.output, ROOT/LOG
    assert not (out/'input_manifest.json').exists(), '拒绝覆盖已冻结输入'
    out.mkdir(parents=True, exist_ok=True)
    images = sorted((raw/'sensors/cameras'/CAMERA).glob('*.jpg'))
    assert len(images)>=159
    image_ns = np.asarray([int(p.stem) for p in images], np.int64)
    time_ns = image_ns[0]+np.rint(np.arange(N)*1e9/30).astype(np.int64)
    nearest = np.abs(image_ns[:,None]-time_ns[None,:]).argmin(0)
    residual = image_ns[nearest]-time_ns
    assert np.abs(residual).max() < 26_000_000
    ego_df = pd.read_feather(raw/'city_SE3_egovehicle.feather')
    ego = poses(ego_df,time_ns)
    origin = ego[0,:3,3].copy()
    ego[:,:3,3] -= origin
    cal = pd.read_feather(raw/'calibration/egovehicle_SE3_sensor.feather').set_index('sensor_name').loc[CAMERA]
    extrinsic = np.eye(4)
    extrinsic[:3,:3] = Rotation.from_quat(cal[QUAT].to_numpy(float)).as_matrix()
    extrinsic[:3,3] = cal[POS].to_numpy(float)
    camera_world = ego @ extrinsic
    intr = pd.read_feather(raw/'calibration/intrinsics.feather').set_index('sensor_name').loc[CAMERA]
    width,height = int(intr.width_px),int(intr.height_px)
    assert Image.open(images[0]).size == (width,height)
    # 保留宽度、等比中央裁剪；RGB与投影使用完全相同的变换，不拉伸。
    crop_h = width*H/W
    top = (height-crop_h)/2
    crop = [0,top,width,top+crop_h]
    scale = W/width
    # resize的像素中心约定：(u+0.5)*s-0.5。
    K = np.array([intr.fx_px*scale,intr.fy_px*scale,
                  (intr.cx_px+0.5)*scale-0.5,(intr.cy_px-top+0.5)*scale-0.5])
    crop_image(images[0],crop).save(out/'initial_rgb.png')
    ann = pd.read_feather(raw/'annotations.feather')
    tracks, rejected = [], []
    for track_id, df in ann.groupby('track_uuid',sort=True):
        df = df.sort_values('timestamp_ns').drop_duplicates('timestamp_ns')
        # 缺失区间不插值成持续存在的actor。
        cuts = np.r_[0,np.flatnonzero(np.diff(df.timestamp_ns)>150_000_000)+1,len(df)]
        for segment,(lo,hi) in enumerate(zip(cuts[:-1],cuts[1:])):
            part = df.iloc[lo:hi]
            if len(part)<2:
                rejected.append({'track':track_id,'segment':segment,'reason':'single_annotation'})
                continue
            stamps = part.timestamp_ns.to_numpy(np.int64)
            frames = np.flatnonzero((time_ns>=stamps[0]) & (time_ns<=stamps[-1]))
            if len(frames)<2:
                continue
            source_ego = poses(ego_df, stamps)
            source_local = np.repeat(np.eye(4)[None],len(part),0)
            source_local[:,:3,:3] = Rotation.from_quat(part[QUAT].to_numpy()).as_matrix()
            source_local[:,:3,3] = part[POS].to_numpy()
            world = source_ego @ source_local
            world[:,:3,3] -= origin
            packed = pd.DataFrame({'timestamp_ns':stamps})
            packed[POS] = world[:,:3,3]
            packed[QUAT] = Rotation.from_matrix(world[:,:3,:3]).as_quat()
            interpolated = poses(packed,time_ns[frames])
            dims = part[['length_m','width_m','height_m']].to_numpy()
            if np.max(np.ptp(dims,axis=0)) > 1e-4:
                raise ValueError('轨迹尺寸变化，需显式处理后再渲染')
            tracks.append({'id':track_id,'segment':int(segment),'category':str(part.category.iloc[0]),
                           'frames':frames.tolist(),'centers':interpolated[:,:3,3].tolist(),
                           'quaternions':Rotation.from_matrix(interpolated[:,:3,:3]).as_quat().tolist(),
                           'dimensions':dims[0].tolist(),'source_annotations':len(part),
                           'first_ns':int(stamps[0]),'last_ns':int(stamps[-1])})
    map_path = next((raw/'map').glob('log_map_archive*.json'))
    map_raw = json.loads(map_path.read_text())
    lines = []
    type_counts = {}
    for lane in map_raw['lane_segments'].values():
        for side in ['left','right']:
            mark = lane[f'{side}_lane_mark_type']
            type_counts[mark] = type_counts.get(mark,0)+1
            if mark not in ['NONE','UNKNOWN']:
                lines.append({'kind':'lane','mark':mark,'xyz':(xyz(lane[f'{side}_lane_boundary'])-origin).tolist()})
    for area in map_raw['drivable_areas'].values():
        points = xyz(area['area_boundary'])-origin
        lines.append({'kind':'road_boundary','mark':None,'xyz':np.r_[points,points[:1]].tolist()})
    crossings = []
    for cross in map_raw['pedestrian_crossings'].values():
        e1,e2 = xyz(cross['edge1']),xyz(cross['edge2'])
        if np.linalg.norm(e1[0]-e2[0]) > np.linalg.norm(e1[0]-e2[-1]):
            e2 = e2[::-1]
        crossings.append((np.r_[e1,e2[::-1]]-origin).tolist())
    scene = {'tracks':tracks,'lines':lines,'crossings':crossings,'excluded_track_segments':rejected}
    (out/'scene.json').write_text(json.dumps(scene,indent=2)+'\n')
    np.savez(out/'trajectory.npz', timestamps_ns=time_ns,
             timestamps_us=np.rint((time_ns-time_ns[0])/1e3).astype(np.int64)+1_000_000,
             camera_world=camera_world.astype(np.float32),ego_world=ego.astype(np.float32),K=K)
    projections = []
    for track in tracks:
        ps = [project_track(track,f,camera_world,K) for f in range(N)]
        valid = [bool(p and p['fully_inside'] and p['large_enough']) for p in ps]
        prefix = next((i for i,v in enumerate(valid) if not v),N)
        projections.append({'id':track['id'],'segment':track['segment'],'category':track['category'],
                            'valid_prefix_frames':prefix,'projections':ps})
    (out/'projections.json').write_text(json.dumps(projections,indent=2)+'\n')
    frames = [0,30,60,90,120,150,180,234]
    sheet = Image.new('RGB',(1280,8*(352+26)),'#151c29')
    references = []
    for row,f in enumerate(frames):
        im = crop_image(images[nearest[f]],crop)
        im.save(out/f'reference-{f:03d}.png')
        overlay = im.copy()
        draw = ImageDraw.Draw(overlay)
        for item in projections:
            p = item['projections'][f]
            if p is None or not p['fully_inside'] or not p['large_enough']:
                continue
            xy = p['corners']
            for i,j in EDGES:
                draw.line([tuple(xy[i]),tuple(xy[j])],fill='yellow',width=2)
            b=p['bounds'];draw.text((b[0],b[1]-12),item['id'][:6],fill='yellow')
        overlay.save(out/f'overlay-{f:03d}.png')
        y=row*378
        ImageDraw.Draw(sheet).text((8,y+5),f'RGB / GT projection | t={f/30:.2f}s | source delta={residual[f]/1e6:.3f}ms',fill='white')
        sheet.paste(im.resize((640,352)),(0,y+26))
        sheet.paste(overlay.resize((640,352)),(640,y+26))
        references.append({'frame':f,'path':str(images[nearest[f]]),'source_delta_ns':int(residual[f])})
    sheet.save(out/'projection-review.jpg',quality=92)
    manifest = {'task_id':'WS-V75-AV2-BRIDGE-01','run_id':'20260920-r1',
                'role':'engineering_development_previously_exposed_log','log_id':LOG,'dataset':'Argoverse 2 Sensor val',
                'raw_path':str(raw),'camera':CAMERA,'source_images':len(images),'source_image_size':[width,height],
                'crop_xyxy':crop,'output_wh':[W,H],'K_fx_fy_cx_cy':K.tolist(),'city_origin':origin.tolist(),
                'first_timestamp_ns':int(time_ns[0]),'frames':N,'fps':30,'source_fps':20,
                'gt_exact_match_frames':np.flatnonzero(np.abs(residual)<=1000).tolist(),
                'nearest_gt_delta_max_ms':float(np.abs(residual).max()/1e6),'references':references,
                'interpolation':'world-frame translation linear + SO(3) Slerp; no extrapolation; split gaps >150ms',
                'map_mark_counts':type_counts,'track_segments':len(tracks),
                'candidate_rule':'geometric visibility only: full bbox >=16x12px, uninterrupted from frame0 through frame150; actual occlusion not yet inspected',
                'long_candidates':[{'id':p['id'],'category':p['category'],'prefix_frames':p['valid_prefix_frames']}
                                   for p in projections if p['valid_prefix_frames']>=151],
                'official_native_argoverse_loader':False,'world_model_generation_calls':0,
                'condition_source':'GT map, GT actor annotations and recorded ego trajectory; oracle engineering baseline',
                'missing_semantics':['traffic lights/signs/poles absent from AV2 map annotations',
                                     'double lane marks mapped to single renderer primitive',
                                     'drivable-area boundaries used as road-boundary proxy'],
                'claim_boundary':'适配与域外开发输入，不是自然重建结果或独立确认；不把重复的20Hz参考帧当30Hz真值',
                'human_verdict':None}
    (out/'input_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ['log_id','track_segments','gt_exact_match_frames','long_candidates']},ensure_ascii=False),flush=True)

if __name__ == '__main__':
    main()
