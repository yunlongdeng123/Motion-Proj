"""Frozen grey-geometry audit; no GPU, heldout reads, textures, or student changes."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
local_runtime = Path(__file__).parents[1] / 'audit_runtime'
if local_runtime.exists():
    sys.path.insert(0, str(local_runtime))
import cv2
import numpy as np
from PIL import Image
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
import trimesh


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def digest(path):
    p = Path(path)
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return {'path': str(p.resolve()), 'bytes': p.stat().st_size, 'sha256': h.hexdigest()}


def load_legacy(path):
    spec = importlib.util.spec_from_file_location('frozen_legacy_shape_audit', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_mesh(path):
    m = trimesh.load(path, force='mesh', process=False)
    if not isinstance(m, trimesh.Trimesh) or not len(m.faces) or not np.isfinite(m.vertices).all():
        raise ValueError('Expected a finite triangular mesh')
    return m


def transform_matrix(yaw, scale, translation):
    c, s = np.cos(yaw), np.sin(yaw)
    t = np.eye(4)
    t[:3, :3] = scale*np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    t[:3, 3] = translation
    return t


def transformed(mesh, matrix):
    out = mesh.copy()
    out.apply_transform(matrix)
    return out


def ortho(mesh, direction, title, footer, size=620, points=None, distances=None):
    eye = np.asarray(direction, float)
    eye /= np.linalg.norm(eye)
    world_up = np.array([0., 0., 1.]) if abs(eye[2]) < .95 else np.array([1., 0., 0.])
    right = np.cross(world_up, eye)
    right /= np.linalg.norm(right)
    up = np.cross(eye, right)
    basis = np.column_stack([right, up, eye])
    xyz = np.asarray(mesh.vertices) @ basis
    lo, hi = xyz[:, :2].min(0), xyz[:, :2].max(0)
    center = (lo+hi)/2
    scale = (size-100)/max((hi-lo).max(), 1e-6)
    def screen(p):
        return np.column_stack([(p[:, 0]-center[0])*scale+size/2,
                                size/2+10-(p[:, 1]-center[1])*scale])
    xy = screen(xyz)
    faces = np.asarray(mesh.faces)
    normals = np.asarray(mesh.face_normals)
    order = np.flatnonzero(normals@eye > 0)
    order = order[np.argsort(xyz[faces[order], 2].mean(1))]
    light = .55*eye+.65*up-.2*right
    light /= np.linalg.norm(light)
    illum = .30+.70*np.maximum(normals@light, 0)
    colors = np.clip(illum[:, None]*[185, 194, 205], 0, 255).astype(np.uint8)
    canvas = np.full((size, size, 3), [17, 22, 30], np.uint8)
    polygons = np.rint(xy[faces]).astype(np.int32)
    for i in order:
        cv2.fillConvexPoly(canvas, polygons[i], tuple(map(int, colors[i])), lineType=cv2.LINE_8)
    if points is not None:
        puv = screen(np.asarray(points)@basis)
        col = cv2.applyColorMap(np.clip(np.asarray(distances)/.3*255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)[:, 0, ::-1]
        for j, uv in enumerate(np.rint(puv).astype(int)):
            cv2.circle(canvas, tuple(uv), 2, tuple(map(int, col[j])), -1, cv2.LINE_AA)
    cv2.putText(canvas, title, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, .60, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(canvas, footer, (16, size-16), cv2.FONT_HERSHEY_SIMPLEX, .38, (175, 190, 205), 1, cv2.LINE_AA)
    return canvas


def six_views(mesh, output, raw=False, points=None, distances=None):
    titles = ['+X', '-X', '+Y', '-Y', '+Z', '-Z'] if raw else ['Front +X', 'Rear -X', 'Left +Y', 'Right -Y', 'Top +Z', 'Bottom -Z']
    vectors = [[1,0,.1],[-1,0,.1],[0,1,.1],[0,-1,.1],[0,0,1],[0,0,-1]]
    footer = 'Raw generator axes; dimensionless; no forward assumption' if raw else 'Actor x forward / y left / z up; uniform size only'
    if points is not None:
        footer = 'All measured points projected; colors: blue 0m to red >=0.30m'
    tiles = [ortho(mesh,d,t,footer,points=points,distances=distances) for t,d in zip(titles,vectors)]
    Image.fromarray(np.concatenate([np.concatenate(tiles[:3],1),np.concatenate(tiles[3:],1)],0)).save(output)


def component_metrics(mesh):
    # Weld duplicated coordinates in a topology-only copy; the evaluated/exported mesh remains unchanged.
    topo = mesh.copy()
    topo.merge_vertices(digits_vertex=6)
    comps = trimesh.graph.connected_components(topo.face_adjacency, nodes=np.arange(len(topo.faces)), min_len=1)
    rows = []
    for ids in comps:
        area = float(topo.area_faces[ids].sum())
        vs = np.unique(topo.faces[ids].reshape(-1))
        b = np.asarray(topo.vertices)[vs]
        rows.append({'faces':int(len(ids)), 'vertices':int(len(vs)), 'area_m2':area,
                     'area_fraction':area/max(float(topo.area),1e-12),
                     'bounds_m':[b.min(0).tolist(),b.max(0).tolist()]})
    rows.sort(key=lambda x:x['area_m2'], reverse=True)
    return {'analysis_only_weld_decimal_digits':6, 'weld_did_not_change_evaluated_mesh':True,
            'count':len(rows), 'largest_component_area_fraction':rows[0]['area_fraction'] if rows else 0,
            'components':rows[:100], 'small_components_not_listed':max(0,len(rows)-100),
            'note':'Legitimate disconnected wheels or mirrors require semantic review; counts are not acceptance.'}


def low_support_regions(mesh):
    v=np.asarray(mesh.vertices)
    width=float(np.ptp(v[:,1]))
    bottom=float(v[:,2].min())
    rows=[]
    wheelbase={}
    for side,sign in [('left',1),('right',-1)]:
        ids=np.flatnonzero((v[:,2]<=bottom+.16)&(sign*v[:,1]>width/4))
        ids=ids[np.argsort(v[ids,0])]
        cuts=np.flatnonzero(np.diff(v[ids,0])>.30)+1 if len(ids) else []
        side_rows=[]
        for piece in np.split(ids,cuts):
            if len(piece)<20:
                continue
            p=v[piece]
            bounds=[p.min(0).tolist(),p.max(0).tolist()]
            row={'side':side,'vertices':len(piece),'bounds_m':bounds,
                 'contact_interval_midpoint_x_m':float((p[:,0].min()+p[:,0].max())/2),
                 'median_y_m':float(np.median(p[:,1])),'minimum_z_m':float(p[:,2].min()),
                 'offset_from_obb_bottom_proxy_m':float(p[:,2].min()+1.04)}
            rows.append(row); side_rows.append(row)
        if len(side_rows)==2:
            wheelbase[side]=side_rows[1]['contact_interval_midpoint_x_m']-side_rows[0]['contact_interval_midpoint_x_m']
    return {'method':'Low-surface diagnostic only: global min z + .16m, |y| > width/4, split along x at >.30m gaps, at least20 vertices per region.',
            'semantic_wheel_identity_automatic':False,'actual_ground_plane_fitted':False,
            'regions':rows,'candidate_side_front_rear_spacing_m':wheelbase,
            'region_bottom_height_spread_m':float(max(r['minimum_z_m'] for r in rows)-min(r['minimum_z_m'] for r in rows)) if rows else None,
            'note':'Manual views must establish whether these regions are the four tires. OBB lower face is only a plane proxy, not measured ground GT.'}


def raw_preview(args):
    args.output.mkdir(parents=True, exist_ok=False)
    mesh = load_mesh(args.mesh)
    v = np.asarray(mesh.vertices)
    ext = np.ptp(v,axis=0)
    six_views(mesh,args.output/'raw_axes_six_views.png',raw=True)
    # Standard generator Y-up views expose the nose and tail without choosing a longitudinal sign.
    sw = mesh.copy()
    sw.vertices = np.column_stack([v[:,0],-v[:,2],v[:,1]])
    tiles = [ortho(sw,d,t,'Generator Y up; raw longitudinal direction not selected',size=800)
             for d,t in [([1,1,.4],'Raw +X -Z'),([-1,-1,.4],'Raw -X +Z'),([0,1,.1],'Raw -Z'),([0,-1,.1],'Raw +Z')]]
    Image.fromarray(np.concatenate([np.concatenate(tiles[:2],1),np.concatenate(tiles[2:],1)],0)).save(args.output/'raw_y_up_four_views.png')
    r={'schema':'a20.hq.raw_mesh_inventory.v1','human_verdict':None,'mesh':digest(args.mesh),
       'raw_bounds':mesh.bounds.tolist(),'raw_extents_xyz':ext.tolist(),
       'raw_extent_ratios_to_largest':(ext/ext.max()).tolist(),'raw_units':'generator normalized coordinates',
       'vertices':len(mesh.vertices),'faces':len(mesh.faces),'heading_assumed':False,
       'gpu_used':False,'heldout_used':False}
    write_json(args.output/'raw_inventory.json',r)
    print(json.dumps(r,indent=2))


def axis_vector(text):
    if text not in ['+x','-x','+y','-y','+z','-z']:
        raise ValueError('Signed axis must be one of +x,-x,+y,-y,+z,-z')
    out=np.zeros(3)
    out['xyz'.index(text[1])]=1 if text[0]=='+' else -1
    return out


def initialize(args):
    args.output.mkdir(parents=True, exist_ok=False)
    mesh=load_mesh(args.mesh)
    forward,up=axis_vector(args.forward),axis_vector(args.up)
    if abs(forward@up)>1e-8:
        raise ValueError('Forward and up must be orthogonal')
    left=np.cross(up,forward)
    rotation=np.vstack([forward,left,up])
    assert np.linalg.det(rotation)>.999
    v=np.asarray(mesh.vertices)
    oriented=v@rotation.T
    spans=np.ptp(oriented,axis=0)
    uniform=5.0/spans[0]
    center=(oriented.max(0)+oriented.min(0))/2
    m=np.eye(4)
    m[:3,:3]=uniform*rotation
    m[:3,3]=-uniform*center
    initial=transformed(mesh,m)
    path=args.output/'candidate_actor_uniform_init.ply'
    initial.export(path)
    six_views(initial,args.output/'uniform_init_six_views.png')
    turns=[('Front right',[1,-1,.42]),('Rear right',[-1,-1,.42]),('Left side',[0,1,.12])]
    tiles=[ortho(initial,d,t,'Semantic heading + uniform scale; no XYZ forced fit',size=760) for t,d in turns]
    Image.fromarray(np.concatenate(tiles,1)).save(args.output/'uniform_init_three_views.png')
    r={'schema':'a20.hq.semantic_uniform_mapping.v1','human_verdict':None,
       'raw_mesh':digest(args.mesh),'mapped_mesh':digest(path),
       'semantic_source':args.semantic_reason,'raw_axis_forward':args.forward,'raw_axis_up':args.up,
       'raw_bounds':mesh.bounds.tolist(),'raw_extents_xyz':np.ptp(v,axis=0).tolist(),
       'semantically_oriented_raw_extents_lwh':spans.tolist(),
       'raw_aspect_lwh_relative_length':(spans/spans[0]).tolist(),
       'proper_rotation_3x3':rotation.tolist(),'rotation_determinant':float(np.linalg.det(rotation)),
       'uniform_scale_m_per_raw_unit':uniform,'raw_to_actor_uniform_4x4':m.tolist(),
       'actor_uniform_to_raw_4x4':np.linalg.inv(m).tolist(),
       'initialized_extent_xyz_m':np.ptp(initial.vertices,axis=0).tolist(),
       'known_actor_size_xyz_m':[5.0,2.5,2.08],
       'extent_to_known_size':(np.ptp(initial.vertices,axis=0)/[5.,2.5,2.08]).tolist(),
       'initialization_note':'5m length is imposed by a single uniform scale; width and height are not forced. No physical accuracy claim from the imposed length.',
       'max_abs_roundtrip_raw':float(np.abs(trimesh.transform_points(initial.vertices,np.linalg.inv(m))-v).max()),
       'topology_unchanged':True,'anisotropic_scaling':False,'lidar_used':False,'heldout_used':False,'gpu_used':False}
    write_json(args.output/'semantic_uniform_transform.json',r)
    print(json.dumps(r,indent=2))


def bounded_register(args):
    args.output.mkdir(parents=True,exist_ok=False)
    legacy=load_legacy(args.legacy)
    mesh=load_mesh(args.mesh)
    bank=np.load(args.conditioning/'lidar'/'aggregated_lidar_train_086_110.npz',allow_pickle=False)
    points=np.asarray(bank['xyz_canonical_m'],float)
    assert set(bank['source_raw'].tolist())=={86,110}
    q=np.array([0.,1.,0.,0.,0.])
    lower=np.array([-np.deg2rad(10),.90,-.35,-.35,-.20])
    upper=np.array([np.deg2rad(10),1.10,.35,.35,.20])
    history=[]
    start=time.monotonic()
    for iteration in range(12):
        mat=transform_matrix(q[0],q[1],q[2:])
        current=transformed(mesh,mat)
        dist,near,_,_=legacy.exact_point_mesh(points,current)
        near_base=trimesh.transform_points(near,np.linalg.inv(mat))
        def residual(p):
            moved=trimesh.transform_points(near_base,transform_matrix(p[0],p[1],p[2:]))
            return (moved-points).reshape(-1)
        solved=least_squares(residual,q,bounds=(lower,upper),loss='huber',f_scale=.075,
                             max_nfev=100,xtol=1e-8,ftol=1e-8,gtol=1e-8)
        delta=float(np.max(np.abs(solved.x-q)))
        history.append({'iteration':iteration,'parameters_before':q.tolist(),
                        'exact_mean_before_m':float(dist.mean()),'exact_p90_before_m':float(np.quantile(dist,.9)),
                        'parameters_after':solved.x.tolist(),'parameter_max_delta':delta,'optimizer_success':bool(solved.success)})
        q=solved.x
        if delta<1e-5:
            break
    mat=transform_matrix(q[0],q[1],q[2:])
    registered=transformed(mesh,mat)
    path=args.output/'candidate_actor_bounded_registered.ply'
    registered.export(path)
    r={'schema':'a20.hq.bounded_build_registration.v1','human_verdict':None,
       'input_mesh':digest(args.mesh),'registered_mesh':digest(path),
       'relative_initial_to_registered_4x4':mat.tolist(),'relative_registered_to_initial_4x4':np.linalg.inv(mat).tolist(),
       'yaw_degrees':float(np.rad2deg(q[0])),'uniform_scale':float(q[1]),'translation_xyz_m':q[2:].tolist(),
       'bounds_lower':lower.tolist(),'bounds_upper':upper.tolist(),
       'parameters_at_or_near_bound':((q-lower<1e-4)|(upper-q<1e-4)).tolist(),
       'method':'Deterministic 12-iteration maximum closest-surface ICP, 5 DOF (yaw, uniform scale, xyz translation). Huber loss on fixed-correspondence XYZ residuals with f_scale=.075m. All 1030 points retained.',
       'history':history,'source_raws':[86,110],'point_count':len(points),
       'selection_or_threshold_sweep':False,'anisotropic_scaling':False,'heldout_used':False,'gpu_used':False,
       'fit_set_warning':'This registration uses BUILD measurements; report post-fit scores as fitting-set metrics, separately from uniform initialization.',
       'wall_seconds':time.monotonic()-start}
    write_json(args.output/'bounded_registration.json',r)
    print(json.dumps(r,indent=2))


def audit(args):
    args.output.mkdir(parents=True,exist_ok=False)
    legacy=load_legacy(args.legacy)
    thresholds=json.loads(args.gate.read_text(encoding='utf-8'))
    limits=thresholds['layer_2_a20_adaptation']
    mesh=load_mesh(args.mesh)
    before=digest(args.mesh)
    pp=args.conditioning/'lidar'/'aggregated_lidar_train_086_110.npz'
    bank=np.load(pp,allow_pickle=False)
    points=np.asarray(bank['xyz_canonical_m'],float)
    source=np.asarray(bank['source_raw'])
    assert len(points)==1030 and set(source.tolist())=={86,110}
    assert bool(bank['measured_lidar_only']) and not bool(bank['native_gaussian_control_used'])
    start=time.monotonic()
    distances,closest,faces,method=legacy.exact_point_mesh(points,mesh)
    projections=[legacy.camera_audit(mesh,points,distances,args.conditioning/'train'/f'{raw:03d}',args.output) for raw in [86,110]]
    extent=np.ptp(mesh.vertices,axis=0)
    ratio=extent/np.asarray(bank['size_lwh_m'])
    fails=[]
    if distances.mean()>limits['mean_exact_point_to_surface_max_m']: fails.append('all_measured_proxy_point_mean')
    if np.quantile(distances,.9)>limits['p90_exact_point_to_surface_max_m']: fails.append('all_measured_proxy_point_p90')
    low,high=limits['each_axis_extent_to_known_size_range']
    if ((ratio<low)|(ratio>high)).any(): fails.append('extent_ratio')
    for p in projections:
        raw=p['raw']
        if p['visible_core_recall']<limits['each_build_visible_core_recall_min']: fails.append(f'raw{raw}_visible_core_recall')
        if p['reliable_domain_iou']<limits['each_build_unknown_aware_iou_min']: fails.append(f'raw{raw}_unknown_aware_iou')
        if p['known_background_spill_fraction']>limits['each_build_known_background_spill_fraction_max']: fails.append(f'raw{raw}_known_background_spill')
    six_views(mesh,args.output/'grey_six_views.png')
    six_views(mesh,args.output/'measured_points_six_views.png',points=points,distances=distances)
    triples=[('Front right',[1,-1,.42]),('Rear right',[-1,-1,.42]),('Left side',[0,1,.08])]
    tiles=[ortho(mesh,d,t,'Geometry only; x forward / y left / z up',size=760) for t,d in triples]
    Image.fromarray(np.concatenate(tiles,1)).save(args.output/'grey_three_views.png')
    # Spatial groups describe where residuals occur, not semantic point ownership or hidden filtering.
    groups={'lower_30cm_obb_band':points[:,2]<-2.08/2+.30,
            'upper_50cm_obb_band':points[:,2]>2.08/2-.50,
            'left_outer_40cm_obb_band':points[:,1]>2.5/2-.40,
            'right_outer_40cm_obb_band':points[:,1]<-2.5/2+.40,
            'rear_70cm_obb_band':points[:,0]<-2.5+.70,
            'front_70cm_obb_band':points[:,0]>2.5-.70}
    by_group={k:{'count':int(mask.sum()),'distance_m':legacy.stats(distances[mask]) if mask.any() else None} for k,mask in groups.items()}
    result={'schema':'a20.hq.independent_cpu_audit.v1','asset_label':args.label,
            'human_verdict':None,'layer_1_structure_status':'requires_separate_actual_view_findings',
            'layer_2_numeric_status':'FAIL_FIXED_ENGINEERING_GATE' if fails else 'NO_FIXED_NUMERIC_FAILURE',
            'numeric_failures':fails,'automatic_teacher_promotion':False,'whole_visible_actor_replacement_allowed':False,
            'gpu_used':False,'heldout_used':False,'geometry_modified_by_audit':False,
            'evaluated_raws':[86,110],'source_point_count':len(points),
            'source_raw_counts':{str(x):int((source==x).sum()) for x in [86,110]},
            'point_ownership':'OBB-selected actual measured points, not semantically verified a20-only GT. All 1030 retained in the main metric.',
            'physical_size_xyz_m':bank['size_lwh_m'].tolist(),'mesh_bounds_m':mesh.bounds.tolist(),
            'mesh_extent_xyz_m':extent.tolist(),'extent_to_known_size':ratio.tolist(),
            'extent_note':'Length can be imposed by initialization; no three-axis dimension forcing is permitted.',
            'point_to_surface_m':legacy.stats(distances),
            'by_source_raw':{str(x):legacy.stats(distances[source==x]) for x in [86,110]},
            'spatial_proxy_groups':by_group,'exact_distance_method':method,
            'vertices':len(mesh.vertices),'faces':len(mesh.faces),'watertight':bool(mesh.is_watertight),
            'winding_consistent':bool(mesh.is_winding_consistent),'surface_area_m2':float(mesh.area),
            'connected_components':component_metrics(mesh),
            'low_support_region_diagnostics':low_support_regions(mesh),
            'projection_metrics':projections,'preregistered_gate':digest(args.gate),
            'provenance':{'mesh':before,'points':digest(pp),'script':digest(__file__),'legacy_script':digest(args.legacy)},
            'metric_limits':limits,'wall_seconds':time.monotonic()-start,
            'notes':['Unknown image pixels excluded; projected OBB potential is not an amodal GT mask.',
                     'Point-to-surface is not LiDAR ray-depth residual.',
                     'Reliable-background spill is evaluated only within the frozen crop.',
                     'Numerical fit cannot override a wheel, semantic-heading, floating-part or component defect.',
                     'Complete raw teacher geometry does not protect observed a20 surfaces or RGB.']}
    np.savez_compressed(args.output/'point_surface_residuals.npz',xyz_canonical_m=points,source_raw=source,
                        point_to_surface_m=distances,closest_surface_xyz_m=closest,closest_face=faces,
                        representative_source_row=bank['representative_source_row'])
    write_json(args.output/'audit.json',result)
    assert digest(args.mesh)['sha256']==before['sha256']
    print(json.dumps({k:result[k] for k in ['asset_label','layer_2_numeric_status','numeric_failures','point_to_surface_m','extent_to_known_size','wall_seconds']},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('raw-preview'); a.set_defaults(func=raw_preview)
    a.add_argument('--mesh',type=Path,required=True); a.add_argument('--output',type=Path,required=True)
    a=s.add_parser('initialize'); a.set_defaults(func=initialize)
    a.add_argument('--mesh',type=Path,required=True); a.add_argument('--output',type=Path,required=True)
    a.add_argument('--forward',required=True); a.add_argument('--up',required=True)
    a.add_argument('--semantic-reason',required=True)
    a=s.add_parser('register'); a.set_defaults(func=bounded_register)
    a.add_argument('--mesh',type=Path,required=True); a.add_argument('--output',type=Path,required=True)
    a.add_argument('--conditioning',type=Path,required=True); a.add_argument('--legacy',type=Path,required=True)
    a=s.add_parser('audit'); a.set_defaults(func=audit)
    a.add_argument('--mesh',type=Path,required=True); a.add_argument('--output',type=Path,required=True)
    a.add_argument('--conditioning',type=Path,required=True); a.add_argument('--legacy',type=Path,required=True)
    a.add_argument('--gate',type=Path,required=True); a.add_argument('--label',required=True)
    args=p.parse_args(); args.func(args)


if __name__=='__main__':
    main()
