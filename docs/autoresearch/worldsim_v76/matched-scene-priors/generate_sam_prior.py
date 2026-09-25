"""同场景 SAM 适配先验：官方自动区域 + 投影动态框提示，保留原 track ID。"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

SAM_ROOT = Path('/root/autodl-tmp/third_party/grounded-segment-anything-v51-stage-f')
DEFAULT_CKPT = Path('/root/autodl-tmp/models/gaussian_grouping_v51_stage_f/sam_vit_h_4b8939.pth')
SIGNS = np.array(list(itertools.product([-1, 1], repeat=3)))
EDGES = [(i, j) for i in range(8) for j in range(i+1, 8) if np.count_nonzero(SIGNS[i] != SIGNS[j]) == 1]


def dynamic_objects(scene):
    tracks = json.loads((scene/'instances/instances_info.json').read_text())
    local = np.linalg.inv(np.loadtxt(scene/'lidar_pose/000.txt'))
    result = {}
    for track_id, data in tracks.items():
        ann = data['frame_annotations']
        selected = [(f, np.array(p), np.array(size)) for f, p, size in zip(ann['frame_idx'], ann['obj_to_world'], ann['box_size']) if 0 <= f <= 60]
        if not selected:
            continue
        positions = np.stack([(local @ pose)[:3, 3] for _, pose, _ in selected])
        # 完全复用 VAD-GS 的选定时间段 dynamic 判定。
        if not (np.any(positions.std(0) > .5) or np.linalg.norm(positions[-1]-positions[0]) > 2):
            continue
        key = int(track_id)
        if not 0 <= key < 255:
            raise ValueError(f'track ID {key} exceeds VAD-GS uint8 contract')
        result[key] = {f: (pose, size) for f, pose, size in selected}
    return result


def project_box(pose, size, w2c, intrinsics, width, height):
    world = (SIGNS * size[None]/2) @ pose[:3, :3].T + pose[:3, 3]
    camera = world @ w2c[:3, :3].T + w2c[:3, 3]
    # 近裁面裁切边，不使用相机背面的角点投影。
    near = .1
    clipped = [p for p in camera if p[2] >= near]
    for i, j in EDGES:
        a, b = camera[i], camera[j]
        if (a[2] < near) != (b[2] < near):
            clipped.append(a + (near-a[2])/(b[2]-a[2])*(b-a))
    if not clipped:
        return None
    xyz = np.array(clipped)
    fx, fy, cx, cy = intrinsics[:4]
    uv = xyz[:, :2] / xyz[:, 2:] * [fx, fy] + [cx, cy]
    xyxy = np.concatenate([uv.min(0), uv.max(0)])
    xyxy[[0, 2]] = xyxy[[0, 2]].clip(0, width-1)
    xyxy[[1, 3]] = xyxy[[1, 3]].clip(0, height-1)
    if np.any(xyxy[2:]-xyxy[:2] < 2):
        return None
    return xyxy, float(camera[:, 2].mean())


def colored_overlay(image, labels):
    result = image.astype(np.float32).copy()
    for key in np.unique(labels):
        if key in (0, 1, 255):
            continue
        color = np.array([(int(key)*67+40)%256, (int(key)*131+90)%256, (int(key)*193+150)%256])
        selected = labels == key
        result[selected] = .55*result[selected] + .45*color
    return result.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CKPT)
    parser.add_argument('--names')
    parser.add_argument('--projection-only', action='store_true')
    parser.add_argument('--background-only', action='store_true')
    parser.add_argument('--stop-on-training', action='store_true')
    args = parser.parse_args()
    names = args.names.split(',') if args.names else [f'{f:03d}_{c}' for f in range(61) for c in range(6)]
    evidence = args.scene/'sam_prior_evidence'
    evidence.mkdir(exist_ok=True)
    if (evidence/'DYNAMIC_IDENTITY_BLOCKED.json').exists() and not (args.projection_only or args.background_only):
        raise RuntimeError('V76-F02: box-only actor labels failed identity gate; preserve evidence and use a visibility-qualified adapter')
    actors = {} if args.background_only else dynamic_objects(args.scene)
    if not args.projection_only:
        import torch
        torch.set_num_threads(2)
        torch.manual_seed(0)
        sys.path.insert(0, str(SAM_ROOT/'segment_anything'))
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        sam = sam_model_registry['vit_h'](checkpoint=str(args.checkpoint)).cuda().eval()
        generator = SamAutomaticMaskGenerator(sam, points_per_side=32, points_per_batch=32,
            pred_iou_thresh=.88, stability_score_thresh=.95, crop_n_layers=0)
        predictor = generator.predictor
        if not args.background_only:
            (args.scene/'sam_masks').mkdir(exist_ok=True)
        (args.scene/'sam_bkgd_masks').mkdir(exist_ok=True)
    rows = []
    report_path = evidence/('projection_report.json' if args.projection_only else 'background_report.json' if args.background_only else 'generation_report.json')
    previous_rows = {row['name']: row for row in json.loads(report_path.read_text()).get('views', [])} if report_path.exists() else {}
    report = {'scene': str(args.scene), 'seed': 0, 'task_id': f'{args.scene.name}-SAM-ADAPTER',
        'sam_commit': subprocess.check_output(['git','-C',str(SAM_ROOT),'rev-parse','HEAD'], text=True).strip(),
        'checkpoint': str(args.checkpoint), 'dynamic_ids': sorted(actors), 'views': rows,
        'input_role': 'adapted initialization prior; includes candidate temporal test frames',
        'method': 'SAM ViT-H AMG 32x32; box-prompt masks clipped to projected cuboid AABB; closest center depth wins overlap',
        'encoding': 'dynamic BGR channel0=original ID, others255; background grayscale labels2..254, residual0',
        'failure_ledger_refs': ['V76-F01', 'V76-F02'], 'failure_ledger_delta': 'none', 'status': 'running',
        'background_only': args.background_only}
    if args.background_only:
        report['method'] = 'SAM ViT-H AMG 32x32; no actor labels generated'
    def save():
        temporary = report_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2)+'\n')
        temporary.replace(report_path)
    start = time.monotonic()
    for name in names:
        if args.stop_on_training:
            state = json.loads(Path('/root/autodl-tmp/runs/v76_ego_view/VADGS-P0R1-000/pipeline_state.json').read_text())
            if state['stage'] not in ('exhaustive_matching', 'triangulation'):
                report['status'] = 'paused_for_main_pipeline'
                save()
                return
        frame, camera = map(int, name.split('_'))
        image = cv2.imread(str(args.scene/'images'/f'{name}.jpg'))
        if image is None:
            raise FileNotFoundError(name)
        h, w = image.shape[:2]
        intrinsics = np.loadtxt(args.scene/'intrinsics'/f'{camera}.txt')
        w2c = np.linalg.inv(np.loadtxt(args.scene/'extrinsics'/f'{name}.txt'))
        boxes = []
        for key, annotations in actors.items():
            if frame not in annotations:
                continue
            projected = project_box(*annotations[frame], w2c, intrinsics, w, h)
            if projected is not None:
                box, depth = projected
                boxes.append({'track_id': key, 'box_xyxy': box.tolist(), 'center_depth': depth})
        boxes.sort(key=lambda x: x['center_depth'])
        row = {'name': name, 'projected_objects': boxes}
        rows.append(row)
        if not args.projection_only:
            dynamic_path = args.scene/'sam_masks'/f'{name}.png'
            bkgd_path = args.scene/'sam_bkgd_masks'/f'{name}.png'
            if bkgd_path.exists() and (args.background_only or dynamic_path.exists()):
                row.update(previous_rows.get(name, {'status': 'existing'}))
                row['reused_existing_files'] = True
                continue
            rgb = image[..., ::-1].copy()
            segments = generator.generate(rgb)
            segments.sort(key=lambda x: x['area'], reverse=True)
            if len(segments) > 253:
                raise ValueError(f'{name}: {len(segments)} segments exceed uint8 label space')
            bkgd = np.zeros((h, w), dtype=np.uint8)
            for label, segment in enumerate(segments, start=2):
                bkgd[segment['segmentation']] = label
            dynamic = np.full((h, w, 3), 255, dtype=np.uint8)
            if boxes:
                predictor.set_image(rgb)
            for box in boxes:
                xyxy = np.array(box['box_xyxy'])
                masks, scores, _ = predictor.predict(box=xyxy, multimask_output=False)
                mask = masks[0].copy()
                x0,y0,x1,y1 = np.r_[np.floor(xyxy[:2]), np.ceil(xyxy[2:])].astype(int)
                clip = np.zeros((h,w), dtype=bool)
                clip[y0:y1+1, x0:x1+1] = True
                mask &= clip & (dynamic[...,0] == 255)
                dynamic[mask,0] = box['track_id']
                box['sam_score'] = float(scores[0])
                box['visible_pixels'] = int(mask.sum())
            predictor.reset_image()
            outputs = [(bkgd_path,bkgd)] if args.background_only else [(dynamic_path,dynamic), (bkgd_path,bkgd)]
            for path, array in outputs:
                temporary = path.with_suffix('.tmp.png')
                if not cv2.imwrite(str(temporary), array):
                    raise OSError(path)
                temporary.replace(path)
            row.update(status='generated', region_count=len(segments), dynamic_pixels=int((dynamic[...,0]!=255).sum()))
        if args.projection_only or frame in (0,20,40,60):
            overlay = image.copy()
            if not args.projection_only:
                # ID0/1也是合法actor，不沿用背景区域的排除语义。
                actor_vis = dynamic[...,0].astype(np.int32)+2
                actor_vis[dynamic[...,0]==255] = 0
                overlay = colored_overlay(overlay, actor_vis)
            for box in boxes:
                x0,y0,x1,y1 = np.array(box['box_xyxy']).astype(int)
                cv2.rectangle(overlay,(x0,y0),(x1,y1),(0,220,255),2)
                cv2.putText(overlay,str(box['track_id']),(x0,max(18,y0)),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,255),2)
            tiles = [image, overlay] if args.projection_only else [image, overlay, colored_overlay(image,bkgd)]
            montage = np.concatenate([cv2.resize(tile,(480,270)) for tile in tiles],axis=1)
            suffix = 'projection' if args.projection_only else 'background' if args.background_only else 'sam'
            cv2.imwrite(str(evidence/f'{name}_{suffix}.jpg'), montage)
        report['seconds'] = time.monotonic()-start
        save()
        print(f'{args.scene.name} {name}: {len(boxes)} dynamic boxes, {row.get("region_count")} SAM regions', flush=True)
    report['status'] = 'complete'
    save()


if __name__ == '__main__':
    main()
