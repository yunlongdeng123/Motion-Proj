"""Independent CPU arithmetic and source-wiring audit; no asset or GPU writes."""
from pathlib import Path
import argparse, ast, hashlib, json, sys
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def pin(path):
    return dict(path=str(path), sha256=sha(path), bytes=Path(path).stat().st_size)


def bank(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def rotation(q):
    w, x, y, z = np.asarray(q, np.float64).T
    return np.stack([
        1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)
    ], 1).reshape(-1, 3, 3)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists(), 'Preserve previous audit'
    root = a.root
    code = root/'code'
    sys.path.insert(0, str(code))
    from local_gaussian_asset import C0, normal_quaternions, sample_linear, tangent_sigma
    gpath = root/'audit/local_surface_candidate_v2/candidate_geometry.npz'
    fpath = root/'local_shared_continuous_v1/final_atlas.npz'
    g, field = bank(gpath), bank(fpath)
    mask = g['cell_asset_mask'].astype(bool)
    assert mask.shape == (150, 333) and int(mask.sum()) == 34793
    assert np.array_equal(mask, g['asset_mask'])
    assert np.array_equal(g['reconstruction_mask'], g['cell_reconstruction_mask'])
    assert np.array_equal(g['blend_ring_mask'], g['cell_blend_ring_mask'])
    ids = np.flatnonzero(mask)
    xyz = g['cell_world_xyz'].reshape(-1, 3)[ids]
    normal = g['cell_normal_world'].reshape(-1, 3)[ids].astype(np.float64)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    uv = g['cell_uv_global'].reshape(-1, 2)[ids]
    vertex_uv = g['uv_global']
    average_uv = (vertex_uv[:-1,:-1]+vertex_uv[:-1,1:]+vertex_uv[1:,:-1]+vertex_uv[1:,1:])*.25
    assert np.array_equal(average_uv[mask], uv)
    assert np.allclose(uv-np.floor(uv), .5, atol=1e-5)
    pose = g['world_from_patch'].astype(np.float64)
    R = pose[:3,:3]
    assert np.allclose(R.T@R, np.eye(3), atol=1e-10) and np.linalg.det(R)>0
    local = ((xyz-pose[:3,3])@R).astype(np.float32)
    world_reconstructed = local.astype(np.float64)@R.T+pose[:3,3]
    local_normal = normal@R
    q = normal_quaternions(local_normal)
    matrices = rotation(q)
    normal_error = float(np.max(np.abs(matrices[:,:,2]-local_normal)))
    orthogonal_error = float(np.max(np.abs(matrices.transpose(0,2,1)@matrices-np.eye(3))))
    rgb, legal = sample_linear(field['atlas_linear'], uv, field['permission_mask']&(field['source_kind']>0))
    assert legal.all() and np.isfinite(rgb).all() and (rgb>=0).all() and (rgb<=1).all()
    srgb = np.where(rgb<=.0031308, 12.92*rgb, 1.055*rgb**(1/2.4)-.055).astype(np.float32)
    coeff = np.zeros((len(ids),16,3), np.float32)
    coeff[:,0]=(srgb-.5)/C0
    sh_rgb = coeff[:,0]*C0+.5
    inverse = np.where(sh_rgb<=.04045,sh_rgb/12.92,((sh_rgb+.055)/1.055)**2.4)
    sigma = tangent_sigma(.02, .9, .95)
    source_files = [code/n for n in ['local_gaussian_asset.py','local_joint_adapter.py','render_new_local_zero.py','fit_new_local_rgb.py']]
    for path in source_files:
        ast.parse(path.read_text(encoding='utf-8-sig'))
    adapter = source_files[1].read_text(encoding='utf-8-sig')
    zero = source_files[2].read_text(encoding='utf-8-sig')
    fitter = source_files[3].read_text(encoding='utf-8-sig')
    assert 'self.local_channel=5+len(self.base.actors)' in adapter
    assert "packet['features'][:,self.local_channel]=1" in adapter
    assert "local=out['extra'][self.local_channel-2]" in adapter
    assert "columns=[rt.local_channel+1,rt.local_channel+2]" in zero
    assert "out['extra'][c-2]" in zero
    assert "linear(rgb)*gain-target.transpose(2,0,1)" in fitter
    assert "self.base.transform@self.asset.world_from_patch" in adapter
    assert "self.asset.manifest['near_m']*self.base.meter_to_model" in adapter
    assert "if not len(self.gate_ids):" in adapter
    assert "merge_rows(packet,self.local_packet(raw))" in adapter
    assert adapter.count(".raster(") == 1 and zero.count(".raster(") == 1
    report = dict(
        schema='R10.new_local_wiring_CPU_audit.v1', complete=True, decision='PASS',
        scope='CPU arithmetic and reviewed wiring only; actual native zero-step guard and imagery required separately',
        CPU_only=True, nativeF=0, nativeB=0, optimizer_updates=0, asset_written=False,
        pins=dict(geometry=pin(gpath),source_field=pin(fpath),code=[pin(x) for x in source_files]),
        asset_cells=int(mask.sum()), cell_shape=list(mask.shape), SH_layout=[len(ids),16,3],
        world_local_roundtrip_max_m=float(abs(world_reconstructed-xyz).max()),
        quaternion_normal_max_abs=normal_error, quaternion_orthogonality_max_abs=orthogonal_error,
        cell_uv_vs_vertex_average_max_texel=float(abs(average_uv[mask]-uv).max()),
        all_four_source_support=True, source_linear_to_DC_to_linear_max_abs=float(abs(inverse-rgb).max()),
        tangent_sigma_m=float(sigma), normal_sigma_m=.0015,
        channel_mapping=dict(depth_feature=0,alpha_feature=1,road_feature=2,canonical_feature=3,context_feature=4,unified_actor_feature=5,local_feature=6,local_extra=4,K_guard_feature=7,body_guard_feature=8),
        single_shared_depth_sort=True,native_identity_gate_preserves_original_packet=True,
        exposure_contract='F is canonical linear; zero asset DC encodes OETF(F) once; RGB fit uses EOTF(native RGB)*gain(raw) versus F once, with no global output exposure change',
        accepted_actor_sha256='ab4fa110440d579bcfd57e4bd0b8a7f61c3f7821189450b58bd027c2f28f67a7',
        old_native_near_preserved=True,new_local_near_m=.2,
        deferred_findings=[
            'Nonempty retirement branch expects native_global_row_count; current accepted identity gate has no IDs and does not enter that branch.',
            'Blend ring labels define cell scope; initialization does not implement an opacity taper. Rendered boundary quality remains an actual zero-step check.',
            'Theoretical four-kernel cell-midpoint coverage is not actual scene coverage certification.'
        ],
        actual_render_acceptance='PENDING separate root-scheduled zero-step render and all-view K/body measurements',
        production_or_fit_authorized_by_this_audit=False)
    assert report['world_local_roundtrip_max_m']<1e-5
    assert normal_error<1e-6 and orthogonal_error<1e-6
    assert report['source_linear_to_DC_to_linear_max_abs']<3e-7
    a.output.mkdir(parents=True)
    (a.output/'receipt.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(complete=True,decision='PASS',receipt=pin(a.output/'receipt.json')),ensure_ascii=False))


if __name__=='__main__':
    main()
