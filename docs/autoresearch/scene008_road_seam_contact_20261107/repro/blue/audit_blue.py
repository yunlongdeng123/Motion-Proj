"""CPU-only fixed raw128 seam audit. Never reads candidate control outputs.

Masks are frozen from original RGB eligibility and measured raw128 ground geometry;
render groups are used only for diagnostics after the masks are written.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
ASSET = ROOT / 'outputs/scene008_road_r0_v44/delivery/assets/road_candidate/new_road_canonical_candidate.npz'
LAYERS = BASE / 'layers_128'
BBOX = (165, 188, 208, 234)  # Approximate circle bounding box supplied by parent.
PATCH = (19.0, 32.6, 0.5, 3.5)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def finite_mean(x):
    return float(np.mean(x)) if np.size(x) else None


def group_stats(d, mask):
    out = {'pixels': int(mask.sum())}
    if not mask.any():
        return out
    w = d['group_alpha_T']
    rw = {'asset': w[0], 'a37': w[1], 'oldroad': d['oldroad_alpha_T'],
          'newroad': d['newroad_alpha_T'], 'static': w[3]}
    rgbt = {'asset': d['group_rgb_T'][0], 'a37': d['group_rgb_T'][1],
            'oldroad': d['oldroad_rgb_T'], 'newroad': d['newroad_rgb_T'],
            'static': d['group_rgb_T'][3]}
    for k in rw:
        mass = float(rw[k][mask].sum())
        out[k] = {'alphaT_mean': finite_mean(rw[k][mask]), 'alphaT_sum': mass,
                  'rgbT_mean': rgbt[k][:, mask].mean(1).astype(float).tolist(),
                  'mass_normalized_rgb': (rgbt[k][:, mask].sum(1) / mass).astype(float).tolist() if mass > 1e-8 else None}
    out['env_rgbT_mean'] = d['env_rgb_T'][:, mask].mean(1).astype(float).tolist()
    out['render_rgb_mean'] = d['rgb'][:, mask].mean(1).astype(float).tolist()
    return out


def make_masks():
    obs = np.load(LAYERS/'observations.npz')
    frame = np.load(BASE/'fullclip_frame128.npz')
    asset = np.load(ASSET)
    h, w = obs['road'].shape
    yy, xx = np.mgrid[:h, :w]
    x0, y0, x1, y1 = BBOX
    roi = (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
    road = roi & obs['road']
    guard = frame['guard']
    clean = road & ~guard
    # Verify that the frame's actual measured-ground coordinates use the frozen anchor.
    transform_error = float(np.max(np.abs(obs['world_to_camera'] @ asset['coordinate_anchor'] - frame['w2l'])))
    if transform_error > 1e-5:
        raise RuntimeError('ground coordinate transform mismatch')
    ground = frame['ground_local'].astype(np.float64)
    patch_ground = ground[(ground[:, 0] >= PATCH[0]-1) & (ground[:, 0] <= PATCH[1]+1)
                          & (ground[:, 1] >= PATCH[2]-1) & (ground[:, 1] <= PATCH[3]+1)]
    tree = cKDTree(ground[:, :2])
    rays = np.stack([xx+.5, yy+.5, np.ones_like(xx)], -1) @ np.linalg.inv(obs['K']).T
    c2l = np.linalg.inv(frame['w2l'])
    ray = rays @ c2l[:3, :3].T
    origin = c2l[:3, 3]
    initial_z = float(np.median(patch_ground[:, 2]))
    t = np.full((h, w), np.nan)
    valid_ray = np.abs(ray[..., 2]) > 1e-8
    t[valid_ray] = (initial_z-origin[2])/ray[..., 2][valid_ray]
    t = np.where(np.isfinite(t) & (t > 0) & (t < 100), t, 1.)
    converged = np.zeros((h, w), bool)
    support_n = np.zeros((h, w), np.int8)
    zspread = np.full((h, w), np.nan)
    for _ in range(12):
        xyz = origin + t[..., None]*ray
        dist, ids = tree.query(xyz[..., :2], k=8)
        z = ground[ids, 2]
        # Fixed raw-ground neighbourhood; no rendered/candidate height enters eligibility.
        good = (dist <= .8)
        zmed = np.median(z, axis=-1)
        good &= np.abs(z-zmed[..., None]) <= .15
        weight = np.where(good, 1/np.maximum(dist, .04)**2, 0)
        den = weight.sum(-1)
        target_z = (weight*z).sum(-1)/np.maximum(den, 1e-20)
        next_t = (target_z-origin[2])/np.where(valid_ray, ray[..., 2], 1)
        converged = np.abs(next_t-t)*np.linalg.norm(ray, axis=-1) < .02
        t = next_t
        support_n = good.sum(-1).astype(np.int8)
        zspread = np.sqrt((weight*(z-target_z[..., None])**2).sum(-1)/np.maximum(den, 1e-20))
    xyz = origin + t[..., None]*ray
    mapped = valid_ray & (t>0) & (t<100) & (support_n>=3) & (zspread<=.08) & converged
    inpatch = (xyz[..., 0]>=PATCH[0]) & (xyz[..., 0]<=PATCH[1]) & (xyz[..., 1]>=PATCH[2]) & (xyz[..., 1]<=PATCH[3])
    measured = obs['scene_depth_m'] > 0
    residual = t-obs['scene_depth_m']  # t is camera z because camera rays have z=1.
    depth_agree = mapped & measured & (np.abs(residual)<=.3)
    front_conflict = mapped & measured & (residual>.3)
    behind_conflict = mapped & measured & (residual<-.3)
    masks = {'roi': roi, 'original_road': road, 'guard': roi & guard,
             'clean_original_road': clean, 'nonroad': roi & ~obs['road'],
             'mapped_original_road': road & mapped & inpatch,
             'mapped_clean_original_road': clean & mapped & inpatch,
             'depth_qualified_original_road': road & depth_agree & inpatch,
             'depth_qualified_clean_original_road': clean & depth_agree & inpatch,
             'front_conflict_original_road': road & front_conflict & inpatch,
             'behind_conflict_original_road': road & behind_conflict & inpatch,
             'no_scene_return_original_road': road & ~measured}
    edges = [('near_x', 0, 19.8, 1), ('far_x', 0, 31.8, 1), ('lateral_y', 1, 2.71, 0)]
    bands = {}
    for name, axis, value, tangent in edges:
        signed = xyz[..., axis]-value
        for side, lo, hi in [('minus', -.4, -.08), ('plus', .08, .4)]:
            mask = masks['mapped_clean_original_road'] & (signed>=lo) & (signed<=hi)
            masks[f'band_{name}_{side}'] = mask
            bands[f'{name}_{side}'] = {'pixels': int(mask.sum()),
                'scene_depth_qualified_pixels': int((mask & depth_agree).sum()),
                'tangent_axis': tangent, 'tangent_bin_m': .2,
                'signed_range_m': [lo, hi]}
    registry = {
        'status': 'frozen_before_opening_n2_control', 'raw': 128,
        'blue_circle_provenance': 'Parent supplied approximate bbox from user screenshot; not an exact circle segmentation.',
        'bbox_xyxy_halfopen': list(BBOX), 'historical_approx_bbox_xyxy_halfopen': [180,190,204,225],
        'primary_denominator': 'original observed road inside fixed bbox; conservative cached vehicle guard separately excluded',
        'world_patch_anchor_local_m': {'x': list(PATCH[:2]), 'y': list(PATCH[2:])},
        'world_mapping': 'Raw128 measured ground only; iterative ray intersection with IDW nearest8 XY heights; radius .8m, >=3, local |z-median|<=.15m, weighted z spread<=.08m, final position convergence<.02m. Not dense ground truth.',
        'initial_height_m': initial_z, 'depth_threshold_m': .3,
        'depth_qualification': 'Measured scene return agrees with raw-ground-interpolated camera z within .3m. Front conflict / behind conflict / no-return separate. Conservative eligibility, not physical surface GT.',
        'bands': bands,
        'edge_centres': {n: {'axis':a,'anchor_local_m':v,'tangent_axis':ta} for n,a,v,ta in edges},
        'edge_interpretation': 'Frozen newroad centre-domain / strip limits; representation-support boundaries, not asserted physical road discontinuities.',
        'selection_exclusions': ['No n2 output read', 'No RGB value or fitted residual used for mask selection', 'No candidate alpha used for primary masks', 'No road_edit_rgb/depth used', 'No ROI-specific fit freedom'],
        'counts': {k:int(v.sum()) for k,v in masks.items()}, 'transform_max_abs_error': transform_error,
        'sources': {str(p):sha(p) for p in [LAYERS/'observations.npz', BASE/'fullclip_frame128.npz', ASSET]},
        'limits': ['Raw128 FRONT-only diagnostic', 'ROI contains foreground/nonroad; only original road is primary', 'Sparse scene depth and interpolated measured support, not hidden-road GT', 'Blue and red acceptance remain separate', 'No line/paint/shadow GT annotation'],
    }
    np.savez_compressed(BASE/'fixed_masks_raw128.npz', **masks, local_xyz=xyz.astype(np.float32),
                        support_count=support_n, measured_ground_height_spread_m=zspread.astype(np.float32),
                        camera_z_m=t.astype(np.float32), scene_depth_residual_m=residual.astype(np.float32))
    dump(BASE/'roi_registration.json', registry)
    return registry, masks


def main():
    registry, masks = make_masks()
    obs = np.load(LAYERS/'observations.npz')
    d = np.load(LAYERS/'r0_delete.npz')
    v = np.load(LAYERS/'v44_delete.npz')
    solo = np.load(LAYERS/'newroad_solo.npz')
    a = np.load(ASSET)
    frame = np.load(BASE/'fullclip_frame128.npz')
    original = obs['rgb'].transpose(2,0,1)
    audit = {'registry_sha256':sha(BASE/'roi_registration.json'), 'frozen_masks_sha256':sha(BASE/'fixed_masks_raw128.npz'),
             'group_order':['asset','a37','totalroad','static'],
             'colour_definition':'Each group colour is sum(RGBT)/sum(alphaT), not mean of low-mass per-pixel ratios.',
             'depth_caveat':'Newroad depth moment/alphaT is the weighted Gaussian-centre depth, not physical first-hit depth.',
             'conservation_max_abs':float(np.max(np.abs(d['rgb']-d['group_rgb_T'].sum(0)-d['env_rgb_T']-d['constant_background_rgb_T']))),
             'mask_stats':{}}
    allmasks = dict(masks)
    allmasks['diagnostic_road_newW_gt_half'] = masks['original_road'] & (d['newroad_alpha_T']>.5)
    for name,m in allmasks.items():
        if name=='guard':
            continue
        out = group_stats(d,m)
        if m.any():
            out['observed_rgb_mean'] = original[:,m].mean(1).astype(float).tolist()
            out['r0_rgb_mae'] = finite_mean(np.abs(d['rgb'][:,m]-original[:,m]))
            out['v44_rgb_mae'] = finite_mean(np.abs(v['rgb'][:,m]-original[:,m]))
            out['solo_alpha_mean'] = finite_mean(solo['alpha'][m])
            out['joint_over_solo_mean'] = finite_mean(d['newroad_alpha_T'][m]/np.maximum(solo['alpha'][m],1e-6))
        audit['mask_stats'][name] = out
    # Centre projections are source/mass provenance diagnostics, not per-pixel splat attribution.
    q = a['world_xyz'] @ obs['world_to_camera'][:3,:3].T + obs['world_to_camera'][:3,3]
    uv = (q @ obs['K'].T)[:,:2]/q[:,2:] - .5
    ui = np.rint(uv[:,0]).astype(int); vi = np.rint(uv[:,1]).astype(int)
    x0,y0,x1,y1=BBOX
    inside=(uv[:,0]>=x0)&(uv[:,0]<x1)&(uv[:,1]>=y0)&(uv[:,1]<y1)&(q[:,2]>0)
    inimage=(ui>=0)&(ui<480)&(vi>=0)&(vi<320)
    road_at=np.zeros(len(q),bool); guard_at=road_at.copy(); dep=np.zeros(len(q));
    road_at[inimage]=obs['road'][vi[inimage],ui[inimage]]
    guard_at[inimage]=frame['guard'][vi[inimage],ui[inimage]]
    dep[inimage]=obs['scene_depth_m'][vi[inimage],ui[inimage]]
    agree=(dep>0)&(np.abs(q[:,2]-dep)<=.3)
    conflict=(dep>0)&(q[:,2]>dep+.3)
    raw_idx=int(np.flatnonzero(a['source_raw']==128)[0])
    build49=(a['source_raw']>=89)&(a['source_raw']<=137)
    sv=a['source_valid'][raw_idx]
    counts={}
    strata={'all_bbox_centres':inside,'originalroad_centres':inside&road_at,
            'originalroad_unguarded_centres':inside&road_at&~guard_at,
            'originalroad_depth_agree_centres':inside&road_at&agree,
            'originalroad_unguarded_depth_agree_centres':inside&road_at&~guard_at&agree,
            'front_depth_conflict_centres':inside&conflict,'no_scene_depth_centres':inside&(dep<=0)}
    for name,m in strata.items():
        counts[name]={'rows':int(m.sum()),'material_eligible':int((m&a['material_eligible']).sum()),
                     'source_valid_raw128':int((m&sv).sum()),
                     'source_valid_any49':int((m&a['source_valid'][build49].any(0)).sum()),
                     'source_valid_ge3_49':int((m&(a['source_valid'][build49].sum(0)>=3)).sum()),
                     'eligible_raw128_source_valid':int((m&a['material_eligible']&sv).sum()),
                     'max_fit_mass':float(a['material_observed_mass_by_state'][:,m].max()) if m.any() else None}
    audit['centre_projection_provenance']={'caveat':'Centre projection only; Gaussian footprint may contribute beyond centre. source_valid is a source qualification cache, not a dense visibility/absence label.', 'counts':counts}
    dump(BASE/'blue_parent_audit.json',audit)
    # Real-image crop and fixed eligibility; no generated pixels.
    x0,y0,x1,y1=BBOX
    fig,ax=plt.subplots(2,4,figsize=(12,7),layout='constrained')
    images=[('Original observation',obs['rgb']),('Frozen V44 delete',v['rgb'].transpose(1,2,0)),
            ('Frozen R0 delete',d['rgb'].transpose(1,2,0)),('Original road / guard',np.stack([masks['original_road'],masks['guard'],np.zeros_like(masks['roi'])],-1).astype(float)),
            ('newroad alphaT',d['newroad_alpha_T']),('oldroad alphaT',d['oldroad_alpha_T']),
            ('static alphaT',d['group_alpha_T'][3]),('Measured world bands',np.zeros((320,480,3)))]
    for i,(name,img) in enumerate(images):
        if i==7:
            img=obs['rgb'].copy()*.35
            for j,e in enumerate(['near_x','far_x','lateral_y']):
                for side,col in [('minus',[1,.4,0]),('plus',[0,.8,1])]:
                    img[masks[f'band_{e}_{side}']]=col
        axis=ax.flat[i]
        axis.imshow(img[y0:y1,x0:x1],vmin=0,vmax=1,cmap='viridis' if img.ndim==2 else None,
                    extent=[x0,x1,y1,y0])
        axis.set_title(name);axis.set_xlabel('original x px');axis.set_ylabel('original y px')
    fig.suptitle('raw128 blue seam: fixed approximate bbox; road eligibility kept separate from foreground')
    fig.savefig(BASE/'blue_frozen_groups_and_bands.png',dpi=170);plt.close(fig)
    # Required simple architecture components figure.
    fig,ax=plt.subplots(figsize=(12,3));ax.axis('off')
    boxes=[(.02,.45,.20,.35,'Frozen original inputs\nRGB + road + guard\nmeasured ground / depth'),
           (.29,.45,.20,.35,'Fixed blue registry\nworld bands / eligibility\nno result-based selection'),
           (.56,.45,.19,.35,'Group decomposition\nnew / old / static\nRGBT + alphaT + depth'),
           (.81,.45,.17,.35,'Separate blue test\nfixed-material n2\nresidual + boundary jump')]
    from matplotlib.patches import FancyBboxPatch
    for x,y,w,h,t in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.01',fc='#edf3f8',ec='#34495e'))
        ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=10)
    for x in [.24,.51,.77]:ax.annotate('',xy=(x+.04,.625),xytext=(x,.625),arrowprops={'arrowstyle':'->','lw':1.5})
    ax.text(.5,.15,'Output: auditable masks, signed colour contributions, source eligibility, falsifiable support control\nSparse observed geometry is not hidden road GT; blue and red acceptance remain separate.',ha='center',va='center',fontsize=11)
    fig.savefig(BASE/'blue_components.png',dpi=170,bbox_inches='tight');plt.close(fig)
    print(json.dumps({'registry':registry['counts'],'bands':registry['bands'],'centre_provenance':counts,'audit_path':str(BASE/'blue_parent_audit.json')},ensure_ascii=False))


if __name__=='__main__':
    main()
