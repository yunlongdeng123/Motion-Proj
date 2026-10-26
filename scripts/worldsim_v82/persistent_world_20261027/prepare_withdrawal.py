"""真实可见道路的严格原始输入撤源组件案例，不把已训练B当撤源模型。

GT侧保留实测地面点；输入侧在编码前移除世界洞内全部点和所有源图对应区域。
现阶段只准备数据，不训练/发明补全算法；粗地面分类不称精确道路语义GT。
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[k] = '1'
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

ROOT = Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027')
RUNS = {
    'scene0017': Path('/root/autodl-tmp/drivingeditor_repro/campaign/hybrid30k_20261012/runs/scene0017_000_048_hybrid_30000_seed0_GN11'),
    'scene0029': Path('/root/autodl-tmp/drivingeditor_repro/campaign/followup_scene_prep_20261013/runs/scene0029_069_117_hybrid_30000_seed0_GN11'),
}


def transform(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def main():
    for name, run in RUNS.items():
        out = ROOT / 'evaluation_withdrawal_v2' / name
        out.mkdir(parents=True, exist_ok=True)
        if (out / 'audit.json').exists():
            continue
        config = json.loads((run / 'effective_config.json').read_text())
        source = Path(config['source_path'])
        cameras = json.loads((run / 'cameras.json').read_text())
        first = cameras[0]['img_name']
        anchor = np.loadtxt(source / 'pose' / (first + '.txt'))
        inv_anchor = np.linalg.inv(anchor)
        world_to_model = np.loadtxt(source / 'transform_matrix.txt')
        if name == 'scene0017':
            ground = json.loads((source / 'sgcm_input_validation.json').read_text())
            plane = np.array(ground['ground_fit']['plane_anchor'])
        else:
            ground = json.loads((run / 'actual_sgcm_binding.json').read_text())
            plane = np.array(ground['metadata']['ground_fit']['plane_anchor'])
        records, pool = [], []
        for row in cameras:
            frame = row['img_name']
            pose = np.loadtxt(source / 'pose' / (frame + '.txt'))
            cloud = np.fromfile(source / 'velodyne' / (frame + '.bin'), dtype=np.float32).reshape(-1, 6)
            local = transform(cloud[:, :3], inv_anchor @ pose)
            is_ground = (np.abs(local @ plane[:3] + plane[3]) <= .15) & (local[:, 0] > 3) & (local[:, 0] < 45) & (np.abs(local[:, 1]) < 8)
            records.append((row, cloud, local, is_ground))
            pool.append(local[is_ground])
        points = np.concatenate(pool)
        candidates = [(int(((np.abs(points[:, 0]-x)<2)&(np.abs(points[:, 1]-y)<1.2)).sum()), float(x), float(y))
                      for x in [8, 12, 16, 20, 24, 28] for y in [-3, 0, 3]]
        count, x, y = max(candidates)
        assert count >= 100, '没有足够实测地面支撑，不伪造正例'
        center = np.array([x, y]); half = np.array([2., 1.2]); margin = .3
        z = -(plane[0]*x+plane[1]*y+plane[3])/plane[2]
        rows = []
        for row, cloud, local, is_ground in records:
            frame = row['img_name']; raw = int(frame[-3:])
            inside = (np.abs(local[:, :2]-center) <= half+margin).all(1)
            c2w = np.eye(4);c2w[:3,:3]=row['rotation'];c2w[:3,3]=row['position']
            w2c = np.linalg.inv(c2w)
            K = np.array([[row['fx']/4,0,row['cx']/4],[0,row['fy']/4,row['cy']/4],[0,0,1.]])
            # v1按角点近裁剪遇到过整帧屏蔽；保留旧包，v2用真实射线/道路带相交。
            # 粗高度只用于定义撤源查询盒，目标点不由这个盒生成。
            gy,gx=np.mgrid[:320,:480]
            ray=np.stack([(gx+.5-K[0,2])/K[0,0],(gy+.5-K[1,2])/K[1,1],np.ones_like(gx)],-1)
            camera_to_anchor=inv_anchor@np.linalg.inv(world_to_model)@c2w
            origin=camera_to_anchor[:3,3]
            direction=ray@camera_to_anchor[:3,:3].T
            lo=np.array([x-half[0]-margin,y-half[1]-margin,z-.6])
            hi=np.array([x+half[0]+margin,y+half[1]+margin,z+.6])
            safe=np.where(np.abs(direction)>1e-12,direction,1e-12)
            ta=(lo-origin)/safe;tb=(hi-origin)/safe
            near=np.minimum(ta,tb).max(-1);far=np.maximum(ta,tb).min(-1)
            mask=((far>=np.maximum(near,0))&(far>0)).astype(np.uint8)*255
            mask=cv2.dilate(mask,np.ones((5,5),np.uint8))
            # 检查并覆盖隐藏实测点的投影，包含0.3m撤源保护带。
            hidden_cam = transform(transform(transform(local[inside], anchor), world_to_model), w2c)
            uv = (hidden_cam @ K.T)[:,:2] / np.maximum(hidden_cam[:,2:], 1e-8)
            ij=np.round(uv).astype(int)
            visible=(hidden_cam[:,2]>.1)&(ij[:,0]>=0)&(ij[:,0]<480)&(ij[:,1]>=0)&(ij[:,1]<320)
            for u,v in ij[visible]:cv2.circle(mask,(int(u),int(v)),2,255,-1)
            assert all(mask[v,u] for u,v in ij[visible])
            rgb=np.array(Image.open(source/'image/image_0'/(frame+'.png')).convert('RGB').resize((480,320),Image.Resampling.LANCZOS))
            # 目标只保留实测点，粗平面只用于目标分类/选洞，不提供给输入编码器。
            for sub in ['input_rgb','input_mask','input_points','target_rgb','target_points']:(out/sub).mkdir(exist_ok=True)
            Image.fromarray(rgb).save(out/'target_rgb'/f'{raw:03d}.png')
            rgb[mask>0]=0;Image.fromarray(rgb).save(out/'input_rgb'/f'{raw:03d}.png')
            Image.fromarray(mask).save(out/'input_mask'/f'{raw:03d}.png')
            np.savez_compressed(out/'input_points'/f'{raw:03d}.npz',local_xyz=local[~inside].astype(np.float32),remaining_attributes=cloud[~inside,3:])
            true_target=inside&is_ground
            np.savez_compressed(out/'target_points'/f'{raw:03d}.npz',measured_ground_xyz=local[true_target].astype(np.float32),
                                within_unpadded_hole=(np.abs(local[true_target,:2]-center)<=half).all(1))
            assert not (np.abs(local[~inside,:2]-center)<=half+margin).all(1).any()
            rows.append(dict(raw=raw,removed_all_returns=int(inside.sum()),measured_target_ground_points=int(true_target.sum()),
                             masked_rgb_pixels=int((mask>0).sum()),K=K.tolist(),world_to_camera=w2c.tolist()))
        audit=dict(scene=name,status='raw_input_withdrawal_component_prepared',source_frames=len(rows),rows=rows,
                   anchor_local_to_world=anchor.tolist(),world_to_model=world_to_model.tolist(),
                   hole_center_anchor_xy=center.tolist(),hole_half_extent_m=half.tolist(),withdrawal_margin_m=margin,
                   minimum_core_radius_m=1.2,source_images_masked_before_encoding=True,
                   source_points_removed_before_initialization=True,previous_gaussians_or_latents_in_input=False,
                   target_directory_must_not_be_conditioning=True,teacher_plane_not_provided_as_input=True,
                   scope='strict raw-input road-completion component, NOT full-scene retraining with B unseen',
                   limitations=['ground selected by historical coarse plane; not dense semantic-road GT',
                                'input cloud contains remaining raw returns; no road classifier answers provided',
                                'target RGB is original observed image; visibility validation and dense target mask still required',
                                'ray/road-band masking is conservative; lost valid context must be quantified; v1 retained'],
                   optimizer_updates=0,GPU_used=False,quality_accepted=False,human_verdict=None)
        (out/'audit.json').write_text(json.dumps(audit,indent=2));print(name,'PREPARED',sum(r['measured_target_ground_points'] for r in rows),flush=True)


if __name__=='__main__':main()
