"""实际49帧：复算框外见证、双源RGB、监督分母和未知区域；零训练。"""
import argparse
from pathlib import Path
import json
import numpy as np
from prepare_v41_revision import read, native_rgb, witness_at, sample_rgb, VERSION, REVISION_POLICY
from v41_dense_geometry import FixedSurface


def main():
    p = argparse.ArgumentParser(); p.add_argument('--dense', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); args = p.parse_args()
    m = read(args.dense/'manifest.json'); meta = read(m['source_bundle'])
    assert m['version'] == VERSION and len(m['frame_order']) == 49
    with np.load(Path(m['source_bundle']).parent/meta['units'], allow_pickle=False) as z:
        units = {k:z[k] for k in z.files}
    surface = FixedSurface(meta, units)
    data, evidence, rgb = [], [], []
    for fid in m['frame_order']:
        with np.load(args.dense/'frames'/f'{fid}.npz', allow_pickle=False) as z: data.append({k:z[k] for k in z.files})
        with np.load(args.dense/'frames'/f'{fid}_evidence.npz', allow_pickle=False) as z: evidence.append({k:z[k] for k in z.files})
        rgb.append(native_rgb(meta['frames'][fid]['original_rgb']))
    restored, deletion, source_checks = 0, 0, 0
    for i, fid in enumerate(m['frame_order']):
        f, e = data[i], evidence[i]; frame = meta['frames'][fid]
        domain = surface.camera(np.asarray(frame['intrinsic']), np.asarray(frame['world_to_camera']))
        v = f['visible']; vehicle = f['vehicle_body'].any(0) | f['vehicle_edge'].any(0)
        assert not (v & vehicle).any() and np.array_equal(f['protect'], ~v)
        assert not (v.ravel() & ~domain['observed']).any()
        assert np.array_equal(f['box_restored'], v & e['box_intersection'])
        assert not (v & ~e['single_view_valid']).any()
        assert np.all(e['source_lidar_triangle_indices'][v] >= 0)
        assert np.array_equal(e['box_clear_witness'], e['single_view_valid'] & ~e['box_intersection'])
        assert not (e['box_clear_witness'] & vehicle).any()
        selected = np.flatnonzero(v.ravel())
        source = f['witness_frame_index'].ravel()[selected]
        assert np.all((source >= 0) & (source < 49) & (source != i))
        for j in np.unique(source):
            ids = selected[source == j]; ef = evidence[j]
            valid, uv = witness_at(domain['world'][ids], meta['frames'][m['frame_order'][j]],
                ef['box_clear_witness'], ef['measured_patch_depth_m'])
            assert valid.all(); np.testing.assert_allclose(uv, f['witness_uv'].reshape(-1, 2)[ids], atol=2e-5)
            source_checks += len(ids)
        sel = np.flatnonzero(f['deletion_valid'].ravel()); deletion += len(sel)
        assert np.array_equal(f['deletion_domains'].any(0), f['deletion_valid'])
        assert not (f['deletion_valid'] & (f['actor_labels'] == 0)).any()
        assert not (f['deletion_valid'].ravel() & ~domain['observed']).any()
        ids = f['deletion_source_ids'].reshape(-1, 2)[sel]
        assert np.all((ids >= 0) & (ids < 49) & (ids != i)) and np.all(ids[:, 0] != ids[:, 1])
        colors = np.zeros((len(sel), 2, 3), np.float32)
        for slot in (0, 1):
            for j in np.unique(ids[:, slot]):
                chosen = np.flatnonzero(ids[:, slot] == j); pixels = sel[chosen]; ef = evidence[j]
                valid, uv = witness_at(domain['world'][pixels], meta['frames'][m['frame_order'][j]],
                    ef['box_clear_witness'], ef['measured_patch_depth_m'])
                assert valid.all()
                np.testing.assert_allclose(uv, f['deletion_source_uv'].reshape(-1, 2, 2)[pixels, slot], atol=2e-5)
                colors[chosen, slot] = sample_rgb(rgb[j], uv); source_checks += len(chosen)
        assert np.all(np.abs(colors[:, 0]-colors[:, 1]).mean(1) <= REVISION_POLICY['deletion_source_color_tolerance']+1e-7)
        np.testing.assert_allclose(colors.mean(1), f['deletion_target_rgb'].transpose(1, 2, 0).reshape(-1, 3)[sel], atol=1e-6)
        assert not f['deletion_target_rgb'][:, ~f['deletion_valid']].any()
        restored += int(f['box_restored'].sum())
    result = dict(status='PASS_ACTUAL49_EVIDENCE', frames=49, independently_reprojected_source_checks=source_checks,
        box_restored_pixels=restored, crossview_deletion_pixels=deletion,
        current_vehicle_RGB_used_for_deletion=False, vehicle_visible_road_overlap=0,
        unknown_or_prior_color_labels=0, GPU_used=False, optimizer_updates=0,
        semantic_or_hidden_GT_verified=False, quality_accepted=False)
    args.output.write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
