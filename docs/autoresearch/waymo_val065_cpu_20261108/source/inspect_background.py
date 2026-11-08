"""从 BUILD 实测回波提出同一世界表面的显露候选；框不代替实例轮廓。"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import json
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(path.read_text())


def transform(points, matrix):
    return points@matrix[:3, :3].T + matrix[:3, 3]


def project(world, obs):
    pose = np.array(obs['camera_pose_world']).reshape(4,4)@np.array(obs['camera_extrinsic']).reshape(4,4)
    cam = transform(world, np.linalg.inv(pose))
    depth = cam[:, 0]
    x, y = -cam[:, 1]/np.maximum(depth, .01), -cam[:, 2]/np.maximum(depth, .01)
    fx, fy, cx, cy, k1, k2, p1, p2, k3 = obs['camera_intrinsic_Brown']
    r = x*x+y*y
    radial = 1+k1*r+k2*r*r+k3*r*r*r
    uv = np.column_stack((fx*(x*radial+2*p1*x*y+p2*(r+2*x*x))+cx,
                          fy*(y*radial+p1*(r+2*y*y)+2*p2*x*y)+cy))
    valid = ((depth > .1) & np.isfinite(uv).all(1) & (uv[:,0] >= 0) & (uv[:,0] < 1920)
             & (uv[:,1] >= 0) & (uv[:,1] < 1280) & (fx*x+cx >= 0) & (fx*x+cx < 1920)
             & (fy*y+cy >= 0) & (fy*y+cy < 1280))
    return uv, depth, valid, pose[:3,3]


def occludes(origin, world, box):
    inverse = np.linalg.inv(np.array(box['world_from_actor']).reshape(4,4))
    o = transform(origin[None], inverse)[0]
    d = transform(world, inverse)-o
    safe = np.where(abs(d) < 1e-10, 1e-10, d)
    half = np.array(box['dimensions'])/2
    t1, t2 = (-half-o)/safe, (half-o)/safe
    near, far = np.minimum(t1,t2).max(1), np.maximum(t1,t2).min(1)
    return (far >= np.maximum(near,0)) & (near > 0) & (near < .985)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bank', type=Path, required=True)
    ap.add_argument('--selection', type=Path, required=True)
    args = ap.parse_args()
    selection = read(args.selection)
    build = set(selection['build'])
    refs = [r for r in [0,10,20,30,40,50,60,70] if r in build]
    queries = [r for r in [60,70,120,150,198] if r in build]
    target = selection['target']['track_id']
    observations = {r: read(args.bank/'observations'/f'{r:03d}.json') for r in set(refs+queries)}
    pairs = []
    for ref in refs:
        obs = observations[ref]
        cloud = np.fromfile(args.bank/obs['lidar_path'], dtype='<f4').reshape(-1,6)
        world = transform(cloud[:,:3], np.array(obs['ego_world']).reshape(4,4))
        uv, depth, valid, origin = project(world, obs)
        source_rows = np.flatnonzero(valid)
        w, p, z = world[valid], uv[valid], depth[valid]
        bins = (p[:,1].astype(int)//4)*480+p[:,0].astype(int)//4
        nearest = np.full(480*320, np.inf)
        np.minimum.at(nearest,bins,z)
        visible = z <= nearest[bins]+.15
        for box in obs['boxes']:
            local = transform(w,np.linalg.inv(np.array(box['world_from_actor']).reshape(4,4)))
            visible &= ~(abs(local) <= np.array(box['dimensions'])/2+.03).all(1)
            visible &= ~occludes(origin,w,box)
        w, p, source_rows = w[visible], p[visible], source_rows[visible]
        for query in queries:
            if abs(query-ref) < 20:
                continue
            qobs = observations[query]
            box = next(b for b in qobs['boxes'] if b['track']==target)
            local = transform(w,np.linalg.inv(np.array(box['world_from_actor']).reshape(4,4)))
            half = np.array(box['dimensions'])/2
            # 高度仅提出近地候选；尚无 road 语义或白车真实轮廓，不称道路真值。
            near_ground = ((abs(local[:,2]+half[2]) < .35)
                           & (abs(local[:,0]) < half[0]+8) & (abs(local[:,1]) < half[1]+3))
            qpx, qdepth, qvalid, qorigin = project(w,qobs)
            hidden = near_ground & qvalid & occludes(qorigin,w,box)
            ids = np.flatnonzero(hidden)
            if not len(ids):
                pairs.append(dict(query_raw=query, reference_raw=ref, returns=0, cells_2px_render=0, points=[]))
                continue
            cells = np.floor(qpx[ids]/8).astype(int)
            _, chosen = np.unique(cells,axis=0,return_index=True)
            picked = ids[chosen]
            pairs.append(dict(query_raw=query, reference_raw=ref, returns=len(ids),
                cells_2px_render=len(chosen), points=[dict(source_cloud_row=int(source_rows[i]),
                world_xyz_m=w[i].tolist(), reference_uv_native=p[i].tolist(),
                query_uv_native=qpx[i].tolist(), query_depth_m=float(qdepth[i])) for i in picked]))
    pairs.sort(key=lambda v:(-v['cells_2px_render'],-v['returns'],v['query_raw'],v['reference_raw']))
    result = dict(schema='waymo.cpu.background_nominations.v1', context_id=selection['context_id'],
        status='NOMINATIONS_REQUIRE_MASK_AND_SEMANTIC_QA',
        evidence='original full LiDAR rows in reference world coordinates; reference camera z-buffer and cuboid-ray checks; query target cuboid in front',
        limits=['cuboid occlusion is not the white vehicle silhouette',
                'near-bottom height is not road semantic truth',
                'sparse measured points are not dense delete RGB ground truth',
                'image-pose Brown projection omits rolling shutter and within-scan actor motion'],
        build_references=refs, build_queries=queries, pairs=pairs,
        GPU_queries=0, training_updates=0, human_verdict=None)
    (args.bank/'background_nominations.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(pairs=len(pairs), top=[{k:v for k,v in p.items() if k!='points'} for p in pairs[:5]])),flush=True)


if __name__ == '__main__':
    main()
