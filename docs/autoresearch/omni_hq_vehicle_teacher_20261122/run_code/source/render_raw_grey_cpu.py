"""原始 Omni 灰模六视图，所有标签均为原始坐标，不预先指定车头。"""
from pathlib import Path
import argparse
import json
import os
import time
os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
import cv2
import numpy as np
import trimesh
cv2.setNumThreads(2)


def render(mesh, eye, label, path):
    width, height = 960, 760
    canvas = np.full((height, width, 3), (26, 23, 20), np.uint8)
    eye = np.asarray(eye, float); eye /= np.linalg.norm(eye)
    up_hint = np.array([0., 1., 0.])
    if abs(eye @ up_hint) > .98:
        up_hint = np.array([0., 0., -1.])
    right = np.cross(up_hint, eye); right /= np.linalg.norm(right)
    up = np.cross(eye, right); basis = np.column_stack((right, up, eye))
    projected = np.asarray(mesh.vertices) @ basis
    lo, hi = projected[:, :2].min(0), projected[:, :2].max(0); center = (lo + hi)/2
    scale = min((width-100)/max(hi[0]-lo[0], 1e-6), (height-150)/max(hi[1]-lo[1], 1e-6))
    xy = np.column_stack(((projected[:, 0]-center[0])*scale+width/2,
                         height/2+20-(projected[:, 1]-center[1])*scale))
    faces = np.asarray(mesh.faces); polygons = np.rint(xy[faces]).astype(np.int32)
    normals = np.asarray(mesh.face_normals); visible = normals @ eye > 0
    depth = projected[faces, 2].mean(1)
    order = np.flatnonzero(visible); order = order[np.argsort(depth[order], kind="stable")]
    light = .25*right + .75*up + .6*eye; light /= np.linalg.norm(light)
    shade = .30+.70*np.maximum(normals @ light, 0)
    colors = np.rint(shade[:, None]*np.array([181, 181, 181])).clip(0,255).astype(np.uint8)
    for face_id in order:
        cv2.fillConvexPoly(canvas, polygons[face_id], tuple(map(int, colors[face_id])), lineType=cv2.LINE_8)
    cv2.putText(canvas, label, (24, 38), cv2.FONT_HERSHEY_SIMPLEX, .7, (237,237,237), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Raw Omni coordinates | geometry only | no mesh edits", (24, height-22),
                cv2.FONT_HERSHEY_SIMPLEX, .48, (170,170,170), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError("Cannot save grey preview")
    return dict(path=str(path), raw_eye_xyz=eye.tolist(), projection="orthographic",
                heading_label_assumed=False, visible_triangles=int(len(order)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--proposal", type=Path, required=True)
    ns = ap.parse_args(); started = time.monotonic(); root = ns.proposal
    mesh = trimesh.load(root/"candidate_raw_omni.ply", process=False, force="mesh")
    views = [("01_posx_posz", [1,.5,1], "Raw view +X / +Z"),
             ("02_negx_posz", [-1,.5,1], "Raw view -X / +Z"),
             ("03_posx_negz", [1,.5,-1], "Raw view +X / -Z"),
             ("04_negx_negz", [-1,.5,-1], "Raw view -X / -Z"),
             ("05_posz", [0,.05,1], "Raw view +Z"),
             ("06_posy", [0,1,0], "Raw view +Y (top)")]
    records = [render(mesh, eye, label, root/("raw_grey_"+name+".png")) for name, eye, label in views]
    images = [cv2.imread(x["path"]) for x in records]
    six = np.vstack([np.hstack(images[:3]), np.hstack(images[3:])])
    cv2.imwrite(str(root/"raw_grey_six_views.png"), six)
    cv2.imwrite(str(root/"raw_grey_three_views.png"), np.hstack(images[:3]))
    report = dict(schema="omni_hq.raw_grey_previews.v1", coordinate_system="raw Omni normalized; +Y used as display up",
                  geometry_modified=False, material_used=False, gpu_used=False, views=records,
                  renderer="CPU OpenCV flat-shaded far-to-near triangles", wall_seconds=time.monotonic()-started)
    (root/"raw_grey_previews.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
