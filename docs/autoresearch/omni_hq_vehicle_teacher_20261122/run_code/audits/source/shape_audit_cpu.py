"""Read-only CPU audit of the unique a20 prior against BUILD 86/110 evidence."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import trimesh


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def record(path):
    return dict(path=str(path), sha256=sha(path), bytes=Path(path).stat().st_size)


def stats(values):
    a = np.asarray(values, float)
    return dict(count=int(a.size), mean=float(a.mean()), p50=float(np.quantile(a, .5)),
                p90=float(np.quantile(a, .9)), p95=float(np.quantile(a, .95)),
                p99=float(np.quantile(a, .99)), maximum=float(a.max()))


def exact_point_mesh(points, mesh):
    """Exact triangle distances with conservative centroid-sphere/AABB pruning."""
    triangles = np.asarray(mesh.triangles)
    area = np.asarray(mesh.area_faces)
    valid = np.isfinite(triangles).all(axis=(1, 2)) & (area > 1e-14)
    face_ids = np.flatnonzero(valid)
    triangles = triangles[valid]
    if not len(triangles):
        raise RuntimeError('No finite nondegenerate triangles')
    centers = triangles.mean(1)
    radius = np.linalg.norm(triangles - centers[:, None], axis=2).max(1)
    max_radius = radius.max()
    lo, hi = triangles.min(1), triangles.max(1)
    tree = cKDTree(centers)
    vtree = cKDTree(triangles.reshape(-1, 3))
    upper, _ = vtree.query(points, workers=2)
    near = np.empty_like(points, dtype=float)
    dist = np.empty(len(points), float)
    faces = np.empty(len(points), np.int64)
    counts = []
    for i, p in enumerate(points):
        ids = np.asarray(tree.query_ball_point(p, upper[i] + max_radius + 1e-9), dtype=np.int64)
        bound = np.maximum(np.maximum(lo[ids] - p, p - hi[ids]), 0)
        ids = ids[(bound * bound).sum(1) <= (upper[i] + 1e-9) ** 2]
        if not len(ids):
            raise RuntimeError('Conservative exact-distance pruning removed every triangle')
        q = trimesh.triangles.closest_point(triangles[ids], np.broadcast_to(p, (len(ids), 3)))
        d2 = ((q - p) ** 2).sum(1)
        j = np.argmin(d2)
        near[i], dist[i], faces[i] = q[j], np.sqrt(d2[j]), face_ids[ids[j]]
        counts.append(len(ids))
    return dist, near, faces, dict(method='exact point-to-triangle; conservative bounding-sphere and AABB pruning',
                                  excluded_zero_area_faces=int((~valid).sum()),
                                  candidate_triangles=stats(counts))


def project(points, mapping):
    t = np.asarray(mapping['camera_from_actor'])
    pc = points @ t[:3, :3].T + t[:3, 3]
    x, y = -pc[:, 1] / pc[:, 0], -pc[:, 2] / pc[:, 0]
    fx, fy, cx, cy, k1, k2, p1, p2, k3 = mapping['camera_intrinsic_Brown_crop']
    r2 = x*x + y*y
    rad = 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
    uv = np.column_stack((fx*(x*rad + 2*p1*x*y + p2*(r2+2*x*x)) + cx,
                          fy*(y*rad + p1*(r2+2*y*y) + 2*p2*x*y) + cy))
    return uv, pc[:, 0]


def color_boundary(image, mask, color):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, color, 1, cv2.LINE_AA)


def camera_audit(mesh, points, distances, frame_dir, out):
    mapping_path = frame_dir / 'mapping.json'
    mapping = json.loads(mapping_path.read_text())
    assert mapping['raw'] in [86, 110] and mapping['role'] == 'train'
    rgb = np.asarray(Image.open(frame_dir/'context_rgb.png').convert('RGB')).copy()
    visible = np.asarray(Image.open(frame_dir/'visible_mask.png')) > 0
    background = np.asarray(Image.open(frame_dir/'known_background_mask.png')) > 0
    unknown = np.asarray(Image.open(frame_dir/'unknown_potential_mask.png')) > 0
    uv, depth = project(np.asarray(mesh.vertices), mapping)
    face = np.asarray(mesh.faces)
    height, width = visible.shape
    mask = np.zeros((height, width), np.uint8)
    polygons = uv[face]
    good = np.isfinite(polygons).all(axis=(1, 2)) & (depth[face] > .1).all(1)
    good &= (polygons[:, :, 0].max(1) >= 0) & (polygons[:, :, 0].min(1) < width)
    good &= (polygons[:, :, 1].max(1) >= 0) & (polygons[:, :, 1].min(1) < height)
    for p in np.rint(polygons[good]).astype(np.int32):
        cv2.fillConvexPoly(mask, p, 1, lineType=cv2.LINE_8)
    pred = mask > 0
    tp = int((pred & visible).sum()); fn = int((~pred & visible).sum())
    fp = int((pred & background).sum())
    overlay = rgb.copy()
    overlay[pred] = (.65*overlay[pred] + .35*np.array([25, 200, 230])).astype(np.uint8)
    overlay[~pred & visible] = [245, 50, 65]
    color_boundary(overlay, pred, (20, 220, 240))
    color_boundary(overlay, visible, (235, 30, 220))
    color_boundary(overlay, unknown, (180, 145, 40))
    lidar_overlay = rgb.copy()
    puv, pdepth = project(points, mapping)
    valid = (pdepth > 0) & (puv[:, 0] >= 0) & (puv[:, 0] < width) & (puv[:, 1] >= 0) & (puv[:, 1] < height)
    color = cv2.applyColorMap(np.clip(distances/.5*255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)[:, 0, ::-1]
    for i in np.flatnonzero(valid):
        cv2.circle(lidar_overlay, tuple(np.rint(puv[i]).astype(int)), 2, tuple(map(int, color[i])), -1)
    tile = np.concatenate([rgb, overlay, lidar_overlay], axis=1)
    header = np.full((56, tile.shape[1], 3), [17, 22, 30], np.uint8)
    cv2.putText(header, f'BUILD raw{mapping["raw"]:03d} | RGB / mesh silhouette / measured LiDAR distance',
                (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .50, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(header, 'Cyan=predicted  magenta=visible core  red=missed core | LiDAR: blue~0m, red>=0.5m',
                (8, 43), cv2.FONT_HERSHEY_SIMPLEX, .43, (200, 205, 215), 1, cv2.LINE_AA)
    image_path = out/f'projection_{mapping["raw"]:03d}.png'
    Image.fromarray(np.concatenate([header, tile])).save(image_path)
    Image.fromarray(mask*255).save(out/f'projected_silhouette_{mapping["raw"]:03d}.png')
    # Unknown/occluded pixels are excluded; the OBB potential is not an amodal GT mask.
    return dict(raw=mapping['raw'], visible_core_recall=tp/max(tp+fn, 1),
                reliable_domain_iou=tp/max(tp+fn+fp, 1), true_positive_pixels=tp,
                missed_visible_core_pixels=fn, predicted_known_background_pixels=fp,
                projected_area_pixels=int(pred.sum()),
                known_background_spill_fraction=fp/max(int(pred.sum()), 1),
                unknown_predicted_pixels=int((pred & unknown).sum()),
                mesh_vertices_behind_camera=int((depth <= 0).sum()),
                provenance={name:record(frame_dir/name) for name in
                            ['mapping.json','context_rgb.png','visible_mask.png','known_background_mask.png',
                             'unknown_potential_mask.png']},
                image=record(image_path), metric_domain='V union reliable background; U ignored')


def ortho(mesh, points, view, title, size=600):
    eye = np.asarray(view, float); eye /= np.linalg.norm(eye)
    world_up = np.array([0., 0., 1.]) if abs(eye[2]) < .95 else np.array([1., 0., 0.])
    right = np.cross(world_up, eye); right /= np.linalg.norm(right)
    up = np.cross(eye, right)
    basis = np.column_stack([right, up, eye])
    xyz = np.asarray(mesh.vertices) @ basis
    p = points @ basis
    all_xy = np.vstack([xyz[:, :2], p[:, :2]])
    lo, hi = all_xy.min(0), all_xy.max(0)
    center = (lo+hi)/2
    scale = (size-90)/max((hi-lo).max(), 1e-6)
    def screen(value):
        return np.column_stack([(value[:, 0]-center[0])*scale+size/2,
                                size/2+15-(value[:, 1]-center[1])*scale])
    xy = screen(xyz); face = np.asarray(mesh.faces)
    normals = np.asarray(mesh.face_normals)
    order = np.flatnonzero(normals@eye > 0)
    order = order[np.argsort(xyz[face[order], 2].mean(1))]
    light = np.asarray([.45, -.35, .82]); light /= np.linalg.norm(light)
    illum = .25+.75*np.maximum(normals@light, 0)
    colors = np.clip(illum[:, None]*[175, 194, 210], 0, 255).astype(np.uint8)
    canvas = np.full((size, size, 3), [17, 22, 30], np.uint8)
    polygons = np.rint(xy[face]).astype(np.int32)
    for idx in order:
        cv2.fillConvexPoly(canvas, polygons[idx], tuple(map(int, colors[idx])), lineType=cv2.LINE_8)
    # A small data panel conveys axes and the metre scale without decorating the asset.
    cv2.putText(canvas, title, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, .65, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'x forward / y left / z up | geometry only', (18, size-15),
                cv2.FONT_HERSHEY_SIMPLEX, .40, (180, 190, 205), 1, cv2.LINE_AA)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--conditioning', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--skip-preview', action='store_true')
    args = parser.parse_args(); started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    candidate = args.candidate
    mesh_path = candidate/'candidate_actor_native.ply'
    point_path = args.conditioning/'lidar'/'aggregated_lidar_train_086_110.npz'
    provenance_path = args.conditioning/'lidar'/'provenance.json'
    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    bank = np.load(point_path, allow_pickle=False)
    points = np.asarray(bank['xyz_canonical_m'], float)
    source_raw = np.asarray(bank['source_raw'])
    size = np.asarray(bank['size_lwh_m'], float)
    assert set(source_raw.tolist()) == {86, 110}
    assert bool(bank['measured_lidar_only']) and not bool(bank['native_gaussian_control_used'])
    vertices = np.asarray(mesh.vertices)
    assert len(vertices) and len(mesh.faces) and np.isfinite(vertices).all()
    original_files = {p.name:record(p) for p in candidate.iterdir() if p.is_file() and
                      p.name.startswith(('candidate_', 'input_points_', 'proposal_report', 'actual_point_'))}
    distances, closest, closest_face, method = exact_point_mesh(points, mesh)
    spans = np.ptp(vertices, axis=0); ratio = spans/size
    outlier = np.max(np.maximum(np.abs(vertices)-size/2, 0), axis=1)
    source_stats = {str(raw):stats(distances[source_raw == raw]) for raw in [86, 110]}
    axis_residual = {axis:stats(np.abs(points-closest)[:, i]) for i, axis in enumerate('xyz')}
    transform_npz = candidate/'input_points_and_transform.npz'
    transform = np.load(transform_npz, allow_pickle=False)
    half = float(transform['omni_half_span_m'])
    native = np.asarray(transform['points_native_m'], float)
    control = np.asarray(transform['conditioning_points_omni'], float)
    ids = np.asarray(transform['sampled_indices'], int)
    roundtrip = float(np.abs(control[:, [0, 2, 1]]*half-native[ids]).max())
    omni_path = candidate/'candidate_omni.glb'
    omni = trimesh.load(omni_path, force='mesh', process=False)
    omni_to_metric = np.asarray(omni.vertices)[:, [0, 2, 1]]*half
    conversion_error = cKDTree(vertices).query(omni_to_metric, workers=2)[0]
    six_path = None
    if not args.skip_preview:
        views = []
        for title, direction in [('Front +X',[1,0,.15]),('Rear -X',[-1,0,.15]),
                                 ('Left +Y',[0,1,.12]),('Right -Y',[0,-1,.12]),
                                 ('Top +Z',[0,0,1]),('Bottom -Z',[0,0,-1])]:
            views.append(ortho(mesh, points, direction, title))
        six_path = args.output/'shape_six_views.png'
        Image.fromarray(np.concatenate([np.concatenate(views[:3],axis=1),np.concatenate(views[3:],axis=1)])).save(six_path)
    projections = [camera_audit(mesh, points, distances, args.conditioning/'train'/f'{raw:03d}', args.output)
                   for raw in [86, 110]]
    failure = []
    if roundtrip > 1e-6*size.max(): failure.append('control_metric_roundtrip_error')
    if conversion_error.max() > 1e-5*size.max(): failure.append('normalized_to_metric_mesh_conversion_error')
    if (ratio < .5).any() or (ratio > 1.35).any(): failure.append('dimension_ratio_outside_0.5_to_1.35')
    if distances.mean() > .25: failure.append('measured_point_to_surface_mean_above_0.25m')
    if np.quantile(distances,.9) > .5: failure.append('measured_point_to_surface_p90_above_0.50m')
    for row in projections:
        if row['visible_core_recall'] < .8: failure.append(f'raw{row["raw"]:03d}_visible_core_recall_below_0.8')
        if row['known_background_spill_fraction'] > .05: failure.append(f'raw{row["raw"]:03d}_reliable_background_spill_above_5percent')
    # Never write a human pass/verdict here. Visual vehicle structure is reviewed separately.
    result = dict(schema='a20.independent_cpu_shape_audit.v1',
        numeric_status='FIXED_NUMERIC_FAILURE' if failure else 'NO_FIXED_NUMERIC_FAILURE_VISUAL_REVIEW_REQUIRED',
        failures=failure, human_verdict=None, geometry_modified=False, gpu_used=False,
        evaluated_raws=[86,110], heldout_read=False, source_point_count=len(points),
        source_raw_counts={str(raw):int((source_raw==raw).sum()) for raw in [86,110]},
        evidence_selection='same frozen BUILD measurement packet; geometrically filtered points, not semantic GT',
        physical_size_xyz_m=size.tolist(), mesh_bounds_m=mesh.bounds.tolist(),
        mesh_extent_xyz_m=spans.tolist(), extent_to_physical_size=ratio.tolist(),
        vertices=len(vertices), faces=len(mesh.faces), area_m2=float(mesh.area),
        watertight=bool(mesh.is_watertight), consistent_winding=bool(mesh.is_winding_consistent),
        fraction_vertices_outside_box_plus_2cm=float((outlier>.02).mean()),
        maximum_box_excess_m=float(outlier.max()),
        point_to_surface_m=stats(distances), by_source_raw=source_stats,
        absolute_closest_point_residual_xyz_m=axis_residual, exact_distance_method=method,
        control_roundtrip_max_abs_m=roundtrip, normalized_mesh_to_native_error_m=stats(conversion_error),
        projection_metrics=projections, six_views=record(six_path) if six_path else None,
        source_provenance={p.name:record(p) for p in [point_path,provenance_path,transform_npz]},
        candidate_artifacts=original_files, script=record(Path(__file__)),
        wall_seconds=time.monotonic()-started,
        fixed_thresholds=dict(extent_ratio=[.5,1.35], mean_distance_max_m=.25,p90_distance_max_m=.5,
                              visible_core_recall_min=.8,known_background_spill_max=.05),
        metric_notes=['Point-to-mesh distance is geometric distance, not LiDAR ray-depth residual.',
                      'OBB potential is not an amodal silhouette GT; unknown pixels are excluded.',
                      'Exact-distance computation does not mutate or rescale the candidate.'])
    np.savez_compressed(args.output/'measured_point_surface_residuals.npz',
        xyz_canonical_m=points, source_raw=source_raw, point_to_surface_m=distances,
        closest_surface_xyz_m=closest, closest_face=closest_face,
        representative_source_row=bank['representative_source_row'])
    (args.output/'audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    # Candidate file digests must still be identical after audit.
    for name, item in original_files.items():
        if sha(candidate/name) != item['sha256']:
            raise RuntimeError('Candidate mutated during audit: '+name)
    print(json.dumps({key:result[key] for key in ['numeric_status','failures','point_to_surface_m',
          'extent_to_physical_size','projection_metrics','wall_seconds']},indent=2),flush=True)


if __name__ == '__main__':
    main()
