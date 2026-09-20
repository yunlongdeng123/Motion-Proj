"""无UI的逐段反馈接口：复用官方动力学、Ludus及生成cache，不内置GT策略。"""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import torch

FLASH = Path('/root/autodl-tmp/external/worldsim_v75/flashdreams')
sys.path.insert(0, str(FLASH/'apps/interactive_drive'))
from interactive_drive.config import VehicleConfig
from interactive_drive.math3d import rig_pose_from_state
from interactive_drive.simulation.ego_vehicle_kinematics import integrate_vehicle
from interactive_drive.types import DriverCommand, VehicleState
from ludus_renderer import TimestampedScene
from ludus_renderer._ops.primitives import (
    PRIM_ROAD_BOUNDARY, PRIM_CROSSWALK, PRIM_LANE_LINE_WHITE_SOLID,
    PRIM_LANE_LINE_WHITE_DASHED, PRIM_LANE_LINE_YELLOW_SOLID,
    PRIM_LANE_LINE_YELLOW_DASHED)
from ludus_renderer.clipgt import _polylines_to_pool, _polygons_to_pool
from render_argoverse import tensor, make_camera, context, pool_for_tracks, render


def initial_state(base, trajectory):
    """只使用截至初帧的记录ego姿态估计初速度，不读取未来速度。"""
    manifest = json.loads((base/'input_manifest.json').read_text())
    table = pd.read_feather(Path(manifest['raw_path'])/'city_SE3_egovehicle.feather')
    past = table[table.timestamp_ns <= manifest['first_timestamp_ns']].sort_values('timestamp_ns').tail(2)
    assert len(past) == 2
    times = past.timestamp_ns.to_numpy(np.int64)
    dt = (times[1]-times[0])/1e9
    assert 0 < dt <= .15
    velocity = np.diff(past[['tx_m', 'ty_m', 'tz_m']].to_numpy(), axis=0)[0]/dt
    pose = trajectory['ego_world'][0]
    roll, pitch, yaw = Rotation.from_matrix(pose[:3, :3]).as_euler('xyz')
    speed = float(velocity @ pose[:3, 0])
    state = VehicleState(*map(float, pose[:3, 3]), float(yaw), speed, 0.,
                         pitch_rad=float(pitch), roll_rad=float(roll),
                         velocity_x_mps=float(velocity[0]), velocity_y_mps=float(velocity[1]))
    return state, {'speed_mps': speed, 'source_timestamps_ns': times.tolist(),
                   'cutoff_ns': manifest['first_timestamp_ns'], 'uses_future_velocity': False}


def upload_scene(data, times, K):
    camera, fit = make_camera(K)
    buckets = {}
    for line in data['lines']:
        if line['kind'] == 'road_boundary':
            kind = PRIM_ROAD_BOUNDARY
        else:
            yellow = 'YELLOW' in line['mark']; dashed = 'DASH' in line['mark']
            kind = ([PRIM_LANE_LINE_YELLOW_SOLID, PRIM_LANE_LINE_YELLOW_DASHED] if yellow else
                    [PRIM_LANE_LINE_WHITE_SOLID, PRIM_LANE_LINE_WHITE_DASHED])[int(dashed)]
        buckets.setdefault(kind, []).append(tensor(line['xyz']))
    lines = [_polylines_to_pool(items, kind, torch.device('cuda')) for kind, items in buckets.items()]
    polygon = _polygons_to_pool([tensor(x) for x in data['crossings']], PRIM_CROSSWALK, torch.device('cuda'))
    scene = TimestampedScene(lines, [polygon] if polygon else [], [pool_for_tracks(data['tracks'], times)])
    ctx = context(camera)
    return ctx, ctx.upload_scene(scene), fit


class FeedbackBridge:
    """真值文件只初始化ego/标定；条件场景单独传入，参考评价由调用方持有。"""
    def __init__(self, base, condition_scene, ground_snapper=None):
        self.base = Path(base)
        self.trajectory = np.load(self.base/'trajectory.npz')
        self.state, self.speed_source = initial_state(self.base, self.trajectory)
        self.extrinsic = np.linalg.inv(self.trajectory['ego_world'][0]) @ self.trajectory['camera_world'][0]
        self.config = VehicleConfig()
        self.ground_snapper = ground_snapper
        if ground_snapper is not None:
            # 初始化官方地面高度锚点；初始RGB对应的姿态保持原样。
            ground_snapper.snap(self.state, self.config)
        self.cursor = 0
        self.ctx, self.sid, self.camera_fit = upload_scene(copy.deepcopy(condition_scene), self.trajectory['timestamps_us'], self.trajectory['K'])

    def next_chunk(self, command, count):
        assert isinstance(command, DriverCommand)
        assert 0 <= command.throttle <= 1 and 0 <= command.brake <= 1 and -1 <= command.steer <= 1
        assert count == (5 if self.cursor == 0 else 8)
        end = self.cursor+count
        assert end <= len(self.trajectory['timestamps_us']), '不外推参考时间范围'
        indices = np.arange(self.cursor, end)
        poses = []
        for f in indices:
            # 初帧严格位于t0；初始RGB之前不能偷偷推进一次动力学。
            if f > 0:
                dt = float(self.trajectory['timestamps_us'][f]-self.trajectory['timestamps_us'][f-1])/1e6
                self.state = integrate_vehicle(self.state, command, dt, self.config)
                if self.ground_snapper is not None:
                    self.state = self.ground_snapper.snap(self.state, self.config)
            s = self.state
            rig = rig_pose_from_state(s.x_m, s.y_m, s.z_m, s.yaw_rad,
                                      s.pitch_rad+s.suspension_pitch_rad,
                                      s.roll_rad+s.suspension_roll_rad)
            poses.append(rig @ self.extrinsic)
        poses = np.stack(poses)
        times = self.trajectory['timestamps_us'][indices]
        conditions = render(self.ctx, self.sid, times, poses)
        self.cursor = end
        return {'indices': indices, 'timestamps_us': times, 'camera_world': poses,
                'conditions': conditions, 'ego_state': asdict(self.state)}


def run_feedback(pipeline, cache, bridge, policy, initial_rgb, blocks, consume):
    """policy.step只收到真实初帧/最新生成帧与当前ego，不收到参考actor或未来图像。

    consume逐段保存输入、输出和动作。异常直接向上传播；本函数不重试、不清cache。
    策略真实输入基线须由调用方先验证，本接口自身不证明策略或科学结论成立。
    """
    assert blocks > 0 and 5+8*(blocks-1) <= len(bridge.trajectory['timestamps_us'])
    observation = np.asarray(initial_rgb)
    observed_frame = 0
    rows = []
    for block in range(blocks):
        command = policy.step(rgb=observation.copy(), frame_index=observed_frame,
                              timestamp_us=int(bridge.trajectory['timestamps_us'][observed_frame]),
                              ego_state=asdict(bridge.state))
        record = bridge.next_chunk(command, 5 if block == 0 else 8)
        batch = torch.from_numpy(record['conditions'].copy()).permute(0, 3, 1, 2)[None, None]
        batch = batch.to('cuda', dtype=torch.bfloat16)/127.5-1
        with torch.inference_mode():
            output = pipeline.generate(autoregressive_index=block, cache=cache, input=batch)
            pipeline.finalize(autoregressive_index=block, cache=cache)
        assert bool(torch.isfinite(output).all())
        frames = ((output[0, 0].float().clamp(-1, 1)+1)*127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        assert len(frames) == len(record['indices'])
        row = {'block': block, 'observation_frame': observed_frame,
               'observation_source': 'initial_real_rgb' if block == 0 else 'previous_generated_chunk',
               'command': asdict(command), 'first_condition_frame': int(record['indices'][0]),
               'last_condition_frame': int(record['indices'][-1]), 'ego_state': record['ego_state']}
        consume(row, record, frames)
        rows.append(row)
        observed_frame = int(record['indices'][-1])
        observation = frames[-1]
    return rows
