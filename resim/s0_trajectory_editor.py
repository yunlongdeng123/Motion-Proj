#!/usr/bin/env python
"""S0 cut-in trajectory editor (v2): smooth raised-cosine lateral offset + relative kinematics.

Fix vs v1:
  - actor must have majority of observations in front window 0<x<40 and mean |y|<15
  - edit applies a single smooth lateral offset envelope (C2-ish raised cosine), not
    per-frame independent sinusoids that explode acceleration
  - kinematic gate is on *edit deltas* (Δv/Δa) plus soft absolute caps; original motion
    is allowed to keep its own speed
  - V4 deliberately violates ego distance for validator testing
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PROC = Path("/root/autodl-tmp/data/occgs/processed_10Hz/mini")
OUT = Path("/root/autodl-tmp/data/occgs/scene_specs/s0_edits")
OUT.mkdir(parents=True, exist_ok=True)

DT = 0.1
DV_MAX = 3.0       # m/s change vs original
DA_MAX = 2.5       # m/s^2 change vs original
YAW_RATE_MAX = 0.8
MIN_ACTOR_DIST = 1.5
MIN_EGO_DIST = 2.0
Y_MAX_ROAD = 15.0
PEAK_DY = {"V1": 0.8, "V2": 0.8, "V3": 1.6, "V4": None}


def load_scene(sid, start, end):
    d = PROC / f"{sid:03d}"
    with open(d / "instances" / "instances_info.json") as f:
        inst = json.load(f)
    poses = {t: np.loadtxt(d / "lidar_pose" / f"{t:03d}.txt").reshape(4, 4)
             for t in range(start, end + 1)}
    return inst, poses


def world_to_ego(T_w_obj, T_w_ego):
    return np.linalg.inv(T_w_ego) @ T_w_obj


def ego_xy_yaw(T_w_obj, T_w_ego):
    Te = world_to_ego(T_w_obj, T_w_ego)
    return float(Te[0, 3]), float(Te[1, 3]), float(np.arctan2(Te[1, 0], Te[0, 0]))


def traj_of(inst, iid, start, end):
    fa = inst[iid]["frame_annotations"]
    out = []
    for i, fi in enumerate(fa["frame_idx"]):
        if start <= fi <= end:
            out.append((fi, np.array(fa["obj_to_world"][i], float), np.array(fa["box_size"][i], float)))
    return out


def pick_actor(inst, poses, start, end):
    best, best_score = None, -1.0
    for iid, info in inst.items():
        if not info.get("class_name", "").startswith("vehicle"):
            continue
        traj = traj_of(inst, iid, start, end)
        if len(traj) < 30:
            continue
        front = []
        for t, T, _ in traj:
            x, y, _ = ego_xy_yaw(T, poses[t])
            front.append((x, y))
        xs = np.array([p[0] for p in front])
        ys = np.array([p[1] for p in front])
        in_front = (xs > 0) & (xs < 40)
        if in_front.mean() < 0.35:
            continue
        if np.mean(np.abs(ys[in_front])) > 20:
            continue
        sweep = float(ys[in_front].max() - ys[in_front].min()) if in_front.any() else 0.0
        if sweep < 1.0:
            continue
        # prefer actors that move toward ego laterally (cut-in): |y| decreases
        y_abs = np.abs(ys[in_front])
        closing = float(y_abs[0] - y_abs[-1]) if len(y_abs) > 1 else 0.0
        score = sweep + max(0.0, closing)
        if score > best_score:
            best_score, best = score, iid
    return best, best_score


def raised_cosine(n, peak, shift=0.0):
    """Smooth envelope in [0,1] peaking at 0.5+shift, values in [0, peak]."""
    t = np.linspace(0, 1, n)
    center = 0.5 + shift
    # full-period raised cosine covering [0,1]
    w = 0.5 * (1 - np.cos(2 * np.pi * np.clip(t, 0, 1)))
    # shift by resampling
    t2 = np.clip(t - shift, 0, 1)
    w = 0.5 * (1 - np.cos(2 * np.pi * t2))
    return peak * w


def edit_traj(traj, poses, mode):
    n = len(traj)
    xs0, ys0, yaws0 = [], [], []
    for t, T, _ in traj:
        x, y, yaw = ego_xy_yaw(T, poses[t])
        xs0.append(x); ys0.append(y); yaws0.append(yaw)
    xs0, ys0, yaws0 = map(np.array, (xs0, ys0, yaws0))
    sign = np.sign(np.mean(ys0) + 1e-6)  # push toward centerline
    if mode == "V0":
        dys = np.zeros(n)
    elif mode == "V1":
        dys = -sign * raised_cosine(n, PEAK_DY["V1"], shift=-0.12)
    elif mode == "V2":
        dys = -sign * raised_cosine(n, PEAK_DY["V2"], shift=+0.12)
    elif mode == "V3":
        dys = -sign * raised_cosine(n, PEAK_DY["V3"], shift=0.0)
    elif mode == "V4":
        # slam to ego centerline (unsafe)
        dys = -ys0 + (-0.3 * sign)
    else:
        raise ValueError(mode)
    ys1 = ys0 + dys
    edited = []
    for k, (t, T, size) in enumerate(traj):
        Te = world_to_ego(T, poses[t]).copy()
        Te[0, 3] = xs0[k]
        Te[1, 3] = ys1[k]
        T_new = poses[t] @ Te
        edited.append((t, T_new, size, dict(x=float(xs0[k]), y=float(ys0[k]),
                                            y_new=float(ys1[k]), dy=float(dys[k]),
                                            yaw=float(yaws0[k]))))
    return edited, xs0, ys0, ys1, yaws0


def relative_kin_ok(ys0, ys1):
    reasons = []
    v0 = np.diff(ys0) / DT
    v1 = np.diff(ys1) / DT
    if len(v0) and np.max(np.abs(v1 - v0)) > DV_MAX:
        reasons.append(f"delta_vy {np.max(np.abs(v1-v0)):.2f}>{DV_MAX}")
    if len(v0) >= 2:
        a0 = np.diff(v0) / DT
        a1 = np.diff(v1) / DT
        if np.max(np.abs(a1 - a0)) > DA_MAX:
            reasons.append(f"delta_ay {np.max(np.abs(a1-a0)):.2f}>{DA_MAX}")
    return len(reasons) == 0, reasons


def validate(edited, ys0, ys1, yaws, others, poses, mode):
    reasons = []
    ok_k, r = relative_kin_ok(ys0, ys1)
    reasons += r
    yaw_u = np.unwrap(yaws)
    if len(yaw_u) > 1 and np.max(np.abs(np.diff(yaw_u) / DT)) > YAW_RATE_MAX:
        reasons.append("yaw_rate")
        ok_k = False
    if np.max(np.abs(ys1)) > Y_MAX_ROAD and mode != "V4":
        # only fail if edit made it worse beyond road
        if np.max(np.abs(ys1)) > np.max(np.abs(ys0)) + 0.5:
            reasons.append(f"|y|_worsened>{Y_MAX_ROAD}")
            ok_k = False
    for t, T, size, meta in edited:
        dist = np.hypot(meta["x"], meta["y_new"])
        if dist < MIN_EGO_DIST:
            reasons.append(f"ego_dist {dist:.2f}@{t}")
            ok_k = False
            break
    for t, T, size, meta in edited:
        for oid, oT, _ in others.get(t, []):
            To = world_to_ego(oT, poses[t])
            d = np.hypot(meta["x"] - To[0, 3], meta["y_new"] - To[1, 3])
            if d < MIN_ACTOR_DIST:
                reasons.append(f"actor_dist {d:.2f} vs {oid}@{t}")
                ok_k = False
                break
        if not ok_k and any("actor_dist" in x for x in reasons):
            break
    if mode == "V4":
        return False, reasons + ["forced_V4_reject"]
    return ok_k, reasons


def pick_actor_from_allowlist(inst, poses, start, end, allow):
    """Prefer cut-in among allowlisted JSON actor ids (RigidNodes true_ids)."""
    allow = {int(x) for x in allow}
    best, best_score = None, -1.0
    for iid in allow:
        info = inst.get(str(iid)) or inst.get(iid)
        if info is None:
            continue
        if not str(info.get("class_name", "")).startswith("vehicle"):
            continue
        traj = traj_of(inst, str(iid) if str(iid) in inst else iid, start, end)
        if len(traj) < 20:
            continue
        front = []
        for t, T, _ in traj:
            x, y, _ = ego_xy_yaw(T, poses[t])
            front.append((x, y))
        xs = np.array([p[0] for p in front])
        ys = np.array([p[1] for p in front])
        in_front = (xs > 0) & (xs < 50)
        if in_front.mean() < 0.2:
            continue
        sweep = float(ys[in_front].max() - ys[in_front].min()) if in_front.any() else 0.0
        y_abs = np.abs(ys[in_front]) if in_front.any() else np.array([0.0])
        closing = float(y_abs[0] - y_abs[-1]) if len(y_abs) > 1 else 0.0
        score = sweep + max(0.0, closing) + 0.1 * in_front.mean()
        if score > best_score:
            best_score, best = score, (str(iid) if str(iid) in inst else iid)
    return best, best_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_ids", type=int, nargs="+", default=[4, 5])
    ap.add_argument("--actor_id", type=str, default=None, help="force JSON actor id")
    ap.add_argument("--allow_actors", type=int, nargs="*", default=None,
                    help="restrict pick to these JSON true_ids (e.g. RigidNodes set)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=79)
    args = ap.parse_args()
    summary = {}
    for sid in args.scene_ids:
        inst, poses = load_scene(sid, args.start, args.end)
        if args.actor_id is not None:
            actor = args.actor_id if args.actor_id in inst else (
                int(args.actor_id) if int(args.actor_id) in inst else args.actor_id
            )
            # normalize to key type in inst
            if str(actor) in inst:
                actor = str(actor)
            score = 0.0
        elif args.allow_actors:
            actor, score = pick_actor_from_allowlist(
                inst, poses, args.start, args.end, args.allow_actors
            )
        else:
            actor, score = pick_actor(inst, poses, args.start, args.end)
        if actor is None:
            summary[sid] = dict(status="no_actor")
            print(sid, "no_actor")
            continue
        traj0 = traj_of(inst, actor, args.start, args.end)
        others = {}
        for iid, info in inst.items():
            if iid == actor or not str(info.get("class_name", "")).startswith("vehicle"):
                continue
            for t, T, size in traj_of(inst, iid, args.start, args.end):
                others.setdefault(t, []).append((iid, T, size))
        variants = {}
        for mode in ("V0", "V1", "V2", "V3", "V4"):
            edited, xs0, ys0, ys1, yaws = edit_traj(traj0, poses, mode)
            ok, reasons = validate(edited, ys0, ys1, yaws, others, poses, mode)
            frames = [dict(frame=t, obj_to_world=T.tolist(), box_size=size.tolist(), meta=meta)
                      for t, T, size, meta in edited]
            variants[mode] = dict(accepted=ok, reasons=reasons, frames=frames,
                                  peak_abs_dy=float(np.max(np.abs(ys1 - ys0))))
        out = dict(scene_idx=sid, actor_id=actor, cutin_score=score,
                   start=args.start, end=args.end, variants=variants, version="v2")
        path = OUT / f"scene_{sid:03d}_actor_{actor}_edits.json"
        json.dump(out, open(path, "w"), indent=2)
        summary[sid] = dict(actor=actor, score=score, path=str(path),
                            accept={k: variants[k]["accepted"] for k in variants},
                            peaks={k: variants[k]["peak_abs_dy"] for k in variants})
        print(sid, summary[sid])
    json.dump(summary, open(OUT / "s0_edit_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
