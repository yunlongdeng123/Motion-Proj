"""Persistent 3D road-row visibility veto from frozen retained-object conflicts.

GPU execution is explicit: this script is prepared without running it.
It performs one shared-scene raster per frame and a feature-only backward on
conflicting retained pixels. No optimizer, image-space crop, paste, or row move.
"""
from pathlib import Path
import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image

import closedloop as loop


C = loop.C
TARGET = 10
RGB_CONFLICT = 1 / 255
ROW_CONTRIBUTION = 1e-4
PICKUPS = {21: 'black_pickup', 8: 'right_white_pickup'}


def pin(path):
    return dict(path=str(Path(path).resolve()), sha256=loop.sha(path))


def legacy_rgb(path, shape):
    with np.load(path, allow_pickle=False) as data:
        rgb = data['rgb'].astype(np.float32)
    if rgb.shape == (*shape, 3):
        rgb = rgb.transpose(2, 0, 1)
    if rgb.shape != (3, *shape) or not np.isfinite(rgb).all():
        raise ValueError(f'legacy RGB is not on the exact camera grid: {path}')
    return rgb


def retained_stats(error, labels, selector, road_alpha):
    n = int(selector.sum())
    return dict(pixels=n, mean_max_channel_difference=float(error[selector].mean()) if n else None,
        max_channel_difference=float(error[selector].max()) if n else None,
        pixels_over_1byte=int((selector & (error > RGB_CONFLICT)).sum()),
        conflict_pixels_with_nonzero_same_sort_road_alpha=int((selector & (error > RGB_CONFLICT) & (road_alpha > 0)).sum()))


def execute(args):
    import torch

    source_manifest = Path(args.manifest).resolve()
    source = loop.read(source_manifest)
    source_road = Path(source['road'])
    source_road = source_road if source_road.is_absolute() else source_manifest.parent / source_road
    source_road = source_road.resolve()
    output_road = Path(args.output_road).resolve()
    output_manifest = Path(args.output_manifest).resolve()
    out = Path(args.output).resolve()
    if output_road == source_road or output_manifest == source_manifest:
        raise ValueError('veto must never overwrite the source road or manifest')
    if output_road.exists() or output_manifest.exists() or (out.exists() and any(out.iterdir())):
        raise FileExistsError('preserve the existing veto; use fresh output paths')
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    scene, manifest = loop.load_scene(source_manifest, metric=True)
    keys = [frame.key for frame in manifest.frames]
    if keys != list(range(199)):
        raise ValueError('this fixed closed-loop control requires all 199 registered frames')
    road_indices = [i for i, provider in enumerate(scene._providers) if provider.kind == 'road']
    if len(road_indices) != 1 or road_indices[0] != len(scene._providers) - 1:
        raise ValueError('the single persistent road provider must be the last provider')
    road_provider_index = road_indices[0]
    with np.load(source_road, allow_pickle=False) as data:
        asset = {key: data[key].copy() for key in data.files}
    original_active = asset['active'].astype(bool)
    world_row_ids = np.flatnonzero(original_active)
    count = len(world_row_ids)
    if not count:
        raise ValueError('there are no active road rows to inspect')
    provenance_path = Path(args.provenance)
    with np.load(provenance_path, allow_pickle=False) as data:
        provenance = {key: data[key].copy() for key in data.files}
    if provenance['world_keys'].shape[0] != len(original_active):
        raise ValueError('world row provenance and road asset lengths differ')
    provider_road = scene._providers[road_provider_index].state_objects[0]
    expected_xyz = torch.as_tensor(asset['model_xyz'][original_active], dtype=provider_road.xyz.dtype,
                                   device=provider_road.xyz.device)
    if not torch.equal(provider_road.xyz, expected_xyz):
        raise ValueError('road provider row order does not match source active world rows')
    del expected_xyz
    n_static = len(scene.static.get_xyz)
    n_dynamic = len(scene.dynamic.get_xyz) if scene._include_dynamic else 0
    maximum = np.zeros(count, np.float32)
    accumulated = np.zeros(count, np.float64)
    frame_count = np.zeros(count, np.int32)
    veto = np.zeros(count, bool)
    rows = []
    bindings = []
    registered_actors = set(manifest.actor_to_track)
    frame_data = {row['key']: row for row in manifest.data['frames']}
    start = time.monotonic()
    for key in keys:
        data = frame_data[key]
        label_path = Path(data['instance_mask'])
        label_path = label_path if label_path.is_absolute() else manifest.base / label_path
        old_path = Path(args.legacy_frames) / f'{key:06d}' / 'arrays.npz'
        with torch.no_grad():
            values, owners, camera, coverage = scene._rows(key, (TARGET,), (), False)
        road_start = n_static + n_dynamic + sum(item['rows'] for item in coverage[:road_provider_index])
        if coverage[road_provider_index]['rows'] != count or road_start + count != len(values['means3D']):
            raise RuntimeError('persistent road tail topology changed')
        shape = (camera.image_height, camera.image_width)
        with Image.open(label_path) as image:
            labels = np.asarray(image.resize((shape[1], shape[0]), Image.Resampling.NEAREST)).copy()
        if not set(np.unique(labels).tolist()).difference({0}).issubset(registered_actors):
            raise ValueError('unregistered positive identity in retained labels')
        retained = (labels > 0) & (labels != TARGET)
        old = legacy_rgb(old_path, shape)
        bindings.append(dict(key=key, instance_mask=pin(label_path), legacy_RGB=pin(old_path)))
        values = {name: value.detach() if hasattr(value, 'detach') else value for name, value in values.items()}
        values['mask'] = values['mask'] & ~scene._contains(owners, (TARGET,))
        features = values['features'].new_zeros(values['features'].shape)
        features[:, :2] = values['features'][:, :2]
        features[road_start:, 8] = 1
        features.requires_grad_(True)
        values = dict(values, features=features)
        pack = scene._bindings().raster(camera, scene.pipe, 0, values, scene._background)
        with torch.no_grad():
            current = pack['rgb']
            if scene.environment is not None:
                environment = scene.environment(camera.get_world_directions(False).permute(1, 2, 0)).permute(2, 0, 1)
                current = current + (1 - pack['alpha']) * environment
            current_np = current.detach().cpu().numpy()
            if not np.isfinite(current_np).all():
                raise FloatingPointError('nonfinite shared-scene RGB')
            difference = np.max(np.abs(current_np - old), axis=0)
            conflict = retained & (difference > RGB_CONFLICT)
            road_alpha = pack['extra'][6].detach().cpu().numpy()
        picked = np.empty(0, np.int64)
        if conflict.any():
            objective = pack['extra'][6][torch.as_tensor(conflict, device=features.device)].sum()
            contribution = torch.autograd.grad(objective, features)[0][road_start:, 8]
            if not bool(torch.isfinite(contribution).all()) or bool((contribution < -1e-6).any()):
                raise FloatingPointError('invalid same-sort alpha*T contribution')
            support = contribution.detach().cpu().numpy()
            picked = np.flatnonzero(support > ROW_CONTRIBUTION)
            maximum = np.maximum(maximum, support)
            accumulated += support
            frame_count[picked] += 1
            veto[picked] = True
            del objective, contribution
        row = dict(raw=key, retained=retained_stats(difference, labels, retained, road_alpha),
            pickups={str(actor): dict(name=name, **retained_stats(difference, labels, labels == actor, road_alpha)) for actor, name in PICKUPS.items()},
            responsible_road_rows_this_frame=int(len(picked)), cumulative_veto_rows=int(veto.sum()))
        rows.append(row)
        if key % 20 == 0 or key == keys[-1]:
            print(json.dumps(row), flush=True)
            loop.write(out / 'progress.json', dict(completed_frames=len(rows), total_frames=199, veto_rows=int(veto.sum())))
        del pack, values, features, current
    scene.assert_persistent_unchanged()
    removed = world_row_ids[veto]
    selected_model_xyz = asset['model_xyz'][removed]
    inverse = np.linalg.inv(asset['model_from_world'])
    world_xyz = selected_model_xyz @ inverse[:3, :3].T + inverse[:3, 3]
    evidence = dict(world_row_ids=removed, native_active_row_ids=np.flatnonzero(veto),
        world_keys=provenance['world_keys'][removed], world_xyz=world_xyz,
        conflict_alpha_T_max=maximum[veto], conflict_alpha_T_sum=accumulated[veto],
        conflict_frame_count=frame_count[veto])
    categories = {}
    for name in ['new_completion', 'legacy_completion', 'legacy_generated', 'protected_real_or_known', 'update_domain']:
        if name not in provenance:
            continue
        domain = np.asarray(provenance[name], bool)
        if domain.shape != original_active.shape:
            raise ValueError(f'provenance row coverage has wrong shape: {name}')
        before = int((domain & original_active).sum())
        lost = int(domain[removed].sum())
        categories[name] = dict(active_world_cells_before=before, active_world_cells_vetoed=lost,
            active_world_cells_after=before - lost, fraction_lost=lost / before if before else None)
        evidence[name] = domain[removed]
    if 'source_observation_count' in provenance:
        counts = provenance['source_observation_count']
        evidence['source_observation_count'] = counts[removed]
        known = (counts > 0) & original_active
        before = int(known.sum()); lost = int(known[removed].sum())
        categories['has_projected_source_observation'] = dict(active_world_cells_before=before,
            active_world_cells_vetoed=lost, active_world_cells_after=before-lost,
            fraction_lost=lost/before if before else None)
    np.savez_compressed(out / 'vetoed_world_rows.npz', **evidence)
    new_active = original_active.copy()
    new_active[removed] = False
    output_asset = dict(asset, active=new_active.astype(asset['active'].dtype))
    assert all(np.array_equal(output_asset[key], value) for key, value in asset.items() if key != 'active')
    output_road.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_road.with_name(output_road.name + '.writing')
    with temporary.open('xb') as stream:
        np.savez_compressed(stream, **output_asset)
    temporary.replace(output_road)
    output_source = dict(source, road=str(output_road))
    assert {key: value for key, value in output_source.items() if key != 'road'} == {key: value for key, value in source.items() if key != 'road'}
    loop.write(output_manifest, output_source)
    receipt = dict(schema='persistent_road_visibility_veto.v1', completed=True, frames=199,
        source_manifest=pin(source_manifest), source_road=pin(source_road), closedloop_source=pin(loop.__file__),
        veto_source=pin(__file__), frozen_labels_and_legacy_RGB=bindings,
        output_road=pin(output_road), output_manifest=pin(output_manifest),
        rgb_conflict='max_channel_abs_float_RGB_difference > 1/255 on frozen K',
        row_selection='same-sort alpha*T feature gradient summed over conflicting K pixels > 1e-4; union over 199 frames',
        row_contribution_threshold=ROW_CONTRIBUTION, rgb_conflict_threshold=RGB_CONFLICT,
        retained_actor_rule='all registered positive labels except target10', excluded_target_actor=TARGET,
        pickups=PICKUPS, active_rows_before=count, vetoed_world_rows=int(len(removed)),
        active_rows_after=int(new_active.sum()), active_row_fraction_lost=len(removed)/count,
        world_cell_coverage_loss=categories,
        coverage_definition='world-cell counts, not projected image coverage or exact surface area',
        road_geometry_and_RGB_payload_unchanged=True, source_road_not_overwritten=True,
        manifest_changes=['road'], persistent_veto_identical_for_every_frame=True,
        image_space_crop_or_paste=False, optimizer_updates=0, original_scene_parameters_updated=False,
        controls_geometry_support_vs_native_depth_mismatch=True,
        road_nonexistence_claim=False, actual_protected_visibility_after_veto_validated=False,
        caution='one-pass union may reveal deeper road rows; rerender retained vehicles after applying the new asset',
        seconds=time.monotonic()-start, peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(), rows=rows,
        GPU='keep_on')
    loop.write(out / 'receipt.json', receipt)
    print(json.dumps(dict(completed=True, vetoed_world_rows=int(len(removed)), output_road=str(output_road),
                         output_manifest=str(output_manifest), seconds=receipt['seconds'], protected_rerender_required=True)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=C / 'manifests/road_before_handoff.json')
    parser.add_argument('--legacy-frames', type=Path, default=loop.OLD / 'rendered/corrected_delete/frames')
    parser.add_argument('--provenance', type=Path, default=C / 'field/road_provenance.npz')
    parser.add_argument('--output-road', type=Path, default=C / 'field/road_visible_safe.npz')
    parser.add_argument('--output-manifest', type=Path, default=C / 'manifests/road_visible_safe.json')
    parser.add_argument('--output', type=Path, default=C / 'road_visibility_veto')
    execute(parser.parse_args())
