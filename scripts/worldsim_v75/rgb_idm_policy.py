"""普通单前视角地面接触点感知＋官方nuPlan IDM；不读取参考车辆。"""
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from closed_loop_bridge import DriverCommand, rig_pose_from_state
from evaluate_localization import model
from following_geometry import Route

sys.path.insert(0, '/root/autodl-tmp/external/worldsim_simimpact/nuplan-devkit')
from nuplan.planning.simulation.observation.idm.idm_policy import IDMPolicy


class MotionTracks:
    """固定参数常速度Kalman跟踪；对象匹配只用当前/历史感知位置。"""
    def __init__(self):
        self.tracks = []; self.previous_time = None; self.serial = 0

    def update(self, detections, time_s, ego_velocity):
        dt = 0. if self.previous_time is None else time_s-self.previous_time
        assert dt >= 0
        F = np.eye(4); F[:2, 2:] = np.eye(2)*dt
        G = np.r_[np.eye(2)*dt*dt/2, np.eye(2)*dt]
        for track in self.tracks:
            track['state'] = F@track['state']; track['cov'] = F@track['cov']@F.T+4*(G@G.T)
        self.tracks = [x for x in self.tracks if time_s-x['seen'] <= .6]
        assigned = {}
        if self.tracks and detections:
            costs = np.array([[np.linalg.norm(t['state'][:2]-d['world_center']) for d in detections] for t in self.tracks])
            ti, di = linear_sum_assignment(costs)
            assigned = {int(d): self.tracks[int(t)] for t, d in zip(ti, di) if costs[t, d] <= 10}
        for i, detection in enumerate(detections):
            track = assigned.get(i)
            z = np.asarray(detection['world_center'])
            if track is None:
                track = {'id': self.serial, 'state': np.r_[z, ego_velocity], 'cov': np.diag([2.25, 2.25, 36., 36.]), 'seen': time_s}
                self.serial += 1; self.tracks.append(track)
            else:
                H = np.c_[np.eye(2), np.zeros((2, 2))]
                gain = track['cov']@H.T@np.linalg.inv(H@track['cov']@H.T+np.eye(2)*2.25)
                track['state'] += gain@(z-H@track['state'])
                track['cov'] = (np.eye(4)-gain@H)@track['cov']; track['seen'] = time_s
            detection['track_id'] = track['id']; detection['world_velocity'] = track['state'][2:].tolist()
        self.previous_time = time_s
        return detections


class RGBIDMPolicy:
    def __init__(self, route_xyz, K, extrinsic, ground_plane, detector=None):
        self.route = Route(route_xyz); self.K = np.asarray(K); self.extrinsic = np.array(extrinsic)
        self.plane = np.asarray(ground_plane)
        self.detector = model().to('cuda') if detector is None else detector
        self.idm = IDMPolicy(target_velocity=10., min_gap_to_lead_agent=1., headway_time=1.5, accel_max=1., decel_max=3.)
        self.tracker = MotionTracks(); self.last = None

    def camera_pose(self, state):
        return rig_pose_from_state(state['x_m'], state['y_m'], state['z_m'], state['yaw_rad'],
                                   state['pitch_rad']+state['suspension_pitch_rad'],
                                   state['roll_rad']+state['suspension_roll_rad']) @ self.extrinsic

    def contact(self, u, v, camera):
        ray = np.array([(u-self.K[2])/self.K[0], (v-self.K[3])/self.K[1], 1.]) @ camera[:3, :3].T
        origin = camera[:3, 3]
        normal = np.r_[-self.plane[:2], 1.]
        denominator = float(normal@ray)
        if denominator >= -1e-6: return None
        distance = float((self.plane[2]-normal@origin)/denominator)
        if distance <= 0: return None
        xyz = origin+distance*ray
        return xyz if np.linalg.norm(xyz[:2]-origin[:2]) <= 80 else None

    def perceive(self, rgb, state, camera_override=None):
        image = torch.from_numpy(np.array(rgb, copy=True)).permute(2, 0, 1).float().to('cuda')/255
        with torch.inference_mode(): pred = self.detector([image])[0]
        pred = {k: v.detach().cpu().numpy() for k, v in pred.items()}
        camera = self.camera_pose(state) if camera_override is None else camera_override
        candidates = []
        for box, score, label in zip(pred['boxes'], pred['scores'], pred['labels']):
            if score < .5 or label not in [3, 6, 8]: continue
            left, right = self.contact(box[0], box[3], camera), self.contact(box[2], box[3], camera)
            if left is None or right is None: continue
            center = (left+right)/2
            candidates.append({'box': box.tolist(), 'score': float(score), 'class_id': int(label),
                               'world_center': center[:2].tolist(), 'contact_edge_world': [left.tolist(), right.tolist()]})
        return candidates

    def choose_lead(self, detections, state):
        ego_s, _ = self.route.project([state['x_m'], state['y_m']]); candidates = []
        tangent = np.array([np.cos(state['yaw_rad']), np.sin(state['yaw_rad'])])
        for detection in detections:
            coordinates = np.array([self.route.project(x) for x in detection['contact_edge_world']])
            if coordinates[:, 1].min() > 1 or coordinates[:, 1].max() < -1: continue
            gap = float(coordinates[:, 0].min()-ego_s-2.4)
            if not (0 < gap <= 40): continue
            candidates.append({**detection, 'gap_m': gap,
                               'lead_speed_mps': float(np.clip(np.array(detection['world_velocity'])@tangent, 0, 35))})
        return min(candidates, key=lambda x: x['gap_m']) if candidates else None

    def acceleration(self, speed, lead):
        progress = 1e6 if lead is None else lead['gap_m']
        lead_speed = speed if lead is None else lead['lead_speed_mps']
        acceleration = self.idm.idm_model([], [0., max(0., speed)], [progress, lead_speed, 0.], self.idm.idm_params)[1]
        return float(np.clip(acceleration, -3, 1))

    def step(self, *, rgb, frame_index, timestamp_us, ego_state, camera_override=None):
        state = ego_state
        speed = state['speed_mps']; yaw = state['yaw_rad']
        detections = self.perceive(rgb, state, camera_override)
        detections = self.tracker.update(detections, timestamp_us/1e6, speed*np.array([np.cos(yaw), np.sin(yaw)]))
        lead = self.choose_lead(detections, state); acceleration = self.acceleration(speed, lead)
        progress, _ = self.route.project([state['x_m'], state['y_m']])
        target = self.route.at(progress+max(5., speed*1.5))
        delta = target-np.array([state['x_m'], state['y_m']])
        local_left = -np.sin(yaw)*delta[0]+np.cos(yaw)*delta[1]
        steer = np.arctan2(2*2.8*local_left, max(float(delta@delta), 1.))
        command = DriverCommand(throttle=max(0., acceleration)/3.5, brake=max(0., -acceleration)/6.,
                                steer=float(np.clip(steer/.5, -1, 1)), steer_is_direct=True)
        self.last = {'frame': frame_index, 'timestamp_us': timestamp_us, 'lead': lead,
                     'detections': detections, 'acceleration_mps2': acceleration, 'steer_rad': float(steer),
                     'ego_state': dict(ego_state),
                     'uses_gt_actor': False, 'provided_route': True, 'provided_metric_ground_plane': True}
        return command
