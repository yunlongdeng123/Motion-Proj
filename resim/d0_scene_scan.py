#!/usr/bin/env python
"""D0 scene selection scan (read-only) over nuScenes v1.0-mini.

Selection criteria are FROZEN here before any training/reconstruction result exists
(V7 plan section 6.3). Candidate pool = the 10 v1.0-mini scenes, because the G0 audit
showed only these have complete front-3-camera + LiDAR sweep coverage on this machine.

Per-scene statistics (keyframe annotations only, 2 Hz; no interpolation):
  - ego_total_dist_m, ego_total_yaw_deg : ego motion / rotation
  - n_vehicles, n_moving_vehicles       : vehicle instances (moving = lifetime displacement > 5 m)
  - n_persistent_moving                 : moving vehicles visible >= 8 consecutive keyframes (>= 4 s)
  - best_cutin_score                    : cut-in/merge proxy, see below
  - night                               : "night" in scene description (lower reconstruction priority)

Cut-in proxy for a moving vehicle: while located in front of ego (0 < x_long < 40 m in ego
frame), lateral offset |y| changes by > 1.5 m across its visible span and it stays visible
>= 8 consecutive keyframes. Score = lateral sweep (m) achieved during that span.

Frozen picks (v2, aligned with plan section 6.3 wording; fixed BEFORE any training result):
  S0 static-heavy   : day scene with ego_total_yaw < 30 deg (small ego rotation) and
                      ego_total_dist >= 50 m (enough parallax for background 3DGS),
                      minimizing n_moving_vehicles.
  S1 vehicle-dynamic: day scene with n_persistent_moving >= 2, maximizing n_persistent_moving,
                      tie-break smaller ego_total_yaw; != S0.
  S2 cut-in         : day preferred, n_persistent_moving >= 2 and ego_total_dist >= 10 m
                      (a static ego gives no parallax), maximizing best_cutin_score; != S0/S1.

v1 selector (S0 by min moving only) picked scene-0916 whose 218 deg ego yaw violates the
plan's "small ego rotation" requirement for S0, and would have allowed zero-parallax
scene-0553 as S2; v2 is the frozen version used for D0.
"""
import json
import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes

ROOT = "/root/autodl-tmp/data/nuscenes"
OUT = "/root/autodl-tmp/data/occgs/scene_specs/d0_scene_scan_v1.json"


def yaw_of(rot):
    return Quaternion(rot).yaw_pitch_roll[0]


def main():
    nusc = NuScenes(version="v1.0-mini", dataroot=ROOT, verbose=False)
    report = {}
    for scene in nusc.scene:
        name = scene["name"]
        samples = []
        tok = scene["first_sample_token"]
        while tok:
            s = nusc.get("sample", tok)
            samples.append(s)
            tok = s["next"]
        # ego trajectory from LIDAR_TOP keyframe ego poses
        ego_xy, ego_yaw = [], []
        for s in samples:
            sd = nusc.get("sample_data", s["data"]["LIDAR_TOP"])
            ep = nusc.get("ego_pose", sd["ego_pose_token"])
            ego_xy.append(ep["translation"][:2])
            ego_yaw.append(yaw_of(ep["rotation"]))
        ego_xy = np.array(ego_xy)
        d = np.linalg.norm(np.diff(ego_xy, axis=0), axis=1).sum()
        yaw_un = np.unwrap(np.array(ego_yaw))
        total_yaw = float(np.abs(np.diff(yaw_un)).sum() * 180 / np.pi)

        # vehicle instances
        inst = {}
        for si, s in enumerate(samples):
            sd = nusc.get("sample_data", s["data"]["LIDAR_TOP"])
            ep = nusc.get("ego_pose", sd["ego_pose_token"])
            e_xy = np.array(ep["translation"][:2])
            e_yaw = yaw_of(ep["rotation"])
            c, sn = np.cos(-e_yaw), np.sin(-e_yaw)
            for atok in s["anns"]:
                a = nusc.get("sample_annotation", atok)
                if not a["category_name"].startswith("vehicle."):
                    continue
                itok = a["instance_token"]
                w_xy = np.array(a["translation"][:2])
                r = w_xy - e_xy
                # ego frame: x forward, y left  (nuScenes ego: x forward)
                x_l = c * r[0] - sn * r[1]
                y_l = sn * r[0] + c * r[1]
                inst.setdefault(itok, []).append((si, w_xy[0], w_xy[1], x_l, y_l))
        n_veh = len(inst)
        n_moving = 0
        n_persistent = 0
        best_cutin = 0.0
        best_cutin_inst = None
        for itok, obs in inst.items():
            obs.sort()
            xy = np.array([(o[1], o[2]) for o in obs])
            disp = np.linalg.norm(xy[-1] - xy[0])
            if disp <= 5.0:
                continue
            n_moving += 1
            idx = [o[0] for o in obs]
            # longest consecutive keyframe run
            best_run, run = 1, 1
            for k in range(1, len(idx)):
                run = run + 1 if idx[k] == idx[k - 1] + 1 else 1
                best_run = max(best_run, run)
            if best_run >= 8:
                n_persistent += 1
                # cut-in proxy within front window
                front = [(o[3], o[4]) for o in obs if 0.0 < o[3] < 40.0]
                if len(front) >= 8:
                    ys = np.array([f[1] for f in front])
                    sweep = float(ys.max() - ys.min())
                    if sweep > 1.5 and sweep > best_cutin:
                        best_cutin = sweep
                        best_cutin_inst = itok
        report[name] = dict(
            n_samples=len(samples),
            ego_total_dist_m=round(float(d), 1),
            ego_total_yaw_deg=round(total_yaw, 1),
            n_vehicles=n_veh,
            n_moving_vehicles=n_moving,
            n_persistent_moving=n_persistent,
            best_cutin_score=round(best_cutin, 2),
            best_cutin_instance=best_cutin_inst,
            night="night" in scene["description"].lower(),
            description=scene["description"],
        )

    day = {k: v for k, v in report.items() if not v["night"]}
    s0_pool = {k: v for k, v in day.items() if v["ego_total_yaw_deg"] < 30.0 and v["ego_total_dist_m"] >= 50.0}
    s0 = min(s0_pool, key=lambda k: (s0_pool[k]["n_moving_vehicles"], s0_pool[k]["ego_total_yaw_deg"]))
    s1_pool = {k: v for k, v in day.items() if v["n_persistent_moving"] >= 2 and k != s0}
    s1 = max(s1_pool, key=lambda k: (s1_pool[k]["n_persistent_moving"], -s1_pool[k]["ego_total_yaw_deg"]))
    s2_pool = {k: v for k, v in report.items()
               if v["n_persistent_moving"] >= 2 and v["ego_total_dist_m"] >= 10.0 and k not in (s0, s1)}
    s2 = max(s2_pool, key=lambda k: ((not s2_pool[k]["night"]), s2_pool[k]["best_cutin_score"]))
    result = dict(scan=report, picks=dict(S0=s0, S1=s1, S2=s2))
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result["picks"], indent=2))
    for k in (s0, s1, s2):
        print(k, {kk: vv for kk, vv in report[k].items() if kk != "description"})
        print("   desc:", report[k]["description"])


if __name__ == "__main__":
    main()
