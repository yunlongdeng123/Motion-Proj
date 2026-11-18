"""A1 实际 joint 组贡献与残余73逐像素 alphaT；0优化、0资产改写。"""
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
from road_activation_builder import CAMPAIGN, PREVIOUS, REPO, FRAMES, read_npz, write_json


def run(asset, native_source, output, lease_file, lease_owner):
    sys.path.insert(0, str(PREVIOUS/'code'))
    sys.path.insert(0, str(REPO/'scripts/worldsim_v82'))
    from road_capacity_native import assert_gpu_lease, load_context
    assert_gpu_lease(lease_file, lease_owner)
    import torch
    from drivingeditor_dynamic_edit.native_row_contribution import attribute_prepared_rows
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    torch.set_num_threads(2)
    ctx = load_context(asset_dir=asset, lease_file=lease_file, lease_owner=lease_owner)
    world_ids = ctx.active_world_rows
    new = ctx.provenance['activation_new'][world_ids]
    calls = 0
    contribution = np.zeros(len(world_ids), np.float64)
    rows = []
    for raw in (86, 110):
        saved = read_npz(Path(native_source)/f'{raw:03d}_native.npz')
        remaining = saved['fixed_low_S'] & (saved['road_alpha'] <= .9)
        regions = dict(S_full=saved['S_full'], original_low_S=saved['fixed_low_S'], remaining_thin=remaining)
        camera = ctx.camera(raw)
        packet = ctx.base_packet(raw)
        with torch.no_grad():
            feature = packet['features'].new_zeros((len(world_ids), 24))
            feature[:, :2] = packet['features'][:, :2]
            feature[torch.as_tensor(~new, device=feature.device), 2] = 1
            feature[torch.as_tensor(new, device=feature.device), 3] = 1
            road = ctx.scene._bindings().raster(camera, ctx.scene.pipe, 0, dict(packet, features=feature), ctx.scene._background)
            calls += 1
            road_groups = dict(old_road_alphaT=road['extra'][0].cpu().numpy(), new_road_alphaT=road['extra'][1].cpu().numpy())
            values, owners, camera, _ = ctx.scene._rows(raw, (10,), (), False)
            values = dict(values, mask=values['mask'] & ~ctx.scene._contains(owners, (10,)))
            offset = len(values['means3D']) - len(world_ids)
            if not torch.equal(values['means3D'][offset:], packet['means3D']):
                raise RuntimeError('joint road tail != saved world order')
            feature = values['features'].new_zeros((len(values['means3D']), 24))
            feature[:, :2] = values['features'][:, :2]
            ns = len(ctx.scene.static.get_xyz)
            feature[:ns, 2] = 1
            feature[ns:offset, 3] = 1
            feature[offset+torch.as_tensor(np.flatnonzero(~new),device=feature.device), 4] = 1
            feature[offset+torch.as_tensor(np.flatnonzero(new),device=feature.device), 5] = 1
            joint = ctx.scene._bindings().raster(camera, ctx.scene.pipe, 0, dict(values, features=feature), ctx.scene._background)
            calls += 1
            joint_rgb = joint['rgb']+(1-joint['alpha'])*ctx.scene.environment(camera.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
            joint_groups = {name:joint['extra'][i].cpu().numpy() for i,name in enumerate(
                ('static_alphaT', 'dynamic_alphaT', 'old_road_alphaT', 'new_road_alphaT'))}
            joint_alpha = joint['alpha'][0].cpu().numpy()
            actual_road_alpha = road['alpha'][0].cpu().numpy()
            if not np.array_equal(actual_road_alpha, saved['road_alpha']):
                raise RuntimeError('saved A1 alpha changed during group attribution')
            rgb_error = float(np.abs(joint_rgb.cpu().numpy()-saved['joint_rgb']).max())
            if rgb_error > 1e-6:
                raise RuntimeError('group feature changed native joint RGB')
            conservation = float(np.abs(sum(joint_groups.values())-joint_alpha).max())
            if conservation > 1e-5:
                raise RuntimeError('joint group alphaT conservation failed')
        owners = torch.zeros(len(world_ids), device=ctx.base_rgb.device, dtype=torch.long)
        att = attribute_prepared_rows(ctx.scene, packet, owners, camera, regions)
        calls += 1
        contribution = np.maximum(contribution, att.alpha_mass[:, 2])
        np.savez_compressed(output/f'{raw:03d}_region_contribution.npz', active_world_rows=world_ids,
            region_names=np.asarray(att.region_names), alpha_mass=att.alpha_mass)
        ys, xs = np.nonzero(remaining)
        pixel_indices, local_rows, mass_values = [], [], []
        pixel_rows = []
        for begin in range(0, len(ys), 22):
            batch = {}
            for index in range(begin, min(begin+22, len(ys))):
                mask = np.zeros_like(remaining)
                mask[ys[index], xs[index]] = True
                batch[str(index)] = mask
            per_pixel = attribute_prepared_rows(ctx.scene, packet, owners, camera, batch)
            calls += 1
            rid, col = np.nonzero(per_pixel.alpha_mass > 0)
            local_rows.extend(rid.tolist())
            pixel_indices.extend((col+begin).tolist())
            mass_values.extend(per_pixel.alpha_mass[rid, col].tolist())
            for j,name in enumerate(per_pixel.region_names):
                m = per_pixel.alpha_mass[:, j]
                order = np.flatnonzero(m > 0)
                order = order[np.argsort(m[order])[::-1]]
                index = int(name)
                pixel_rows.append(dict(index=index, y=int(ys[index]), x=int(xs[index]),
                    native_alpha=float(actual_road_alpha[ys[index],xs[index]]), contributors=len(order),
                    old_road_mass=float(m[~new].sum(dtype=np.float64)), new_road_mass=float(m[new].sum(dtype=np.float64)),
                    top_world_rows=world_ids[order[:6]].tolist(), top_alphaT=m[order[:6]].tolist()))
        rid = np.asarray(local_rows, np.int64)
        unique = np.unique(rid)
        with torch.no_grad():
            point = packet['means3D']
            clip = torch.cat([point, torch.ones((len(point),1),device=point.device,dtype=point.dtype)],1) @ camera.full_proj_transform
            xy = (clip[:,:2]/clip[:,3:4]).cpu().numpy()
            xy = (xy+1)*[camera.image_width/2,camera.image_height/2]-.5
        np.savez_compressed(output/f'{raw:03d}_pixel_contribution.npz', pixel_y=ys, pixel_x=xs,
            sparse_pixel_index=np.asarray(pixel_indices,np.int32), sparse_world_row=world_ids[rid],
            sparse_alphaT=np.asarray(mass_values,np.float32), contributing_world_rows=world_ids[unique],
            projected_xy=xy[unique], native_radii=road['radii'].cpu().numpy()[unique],
            native_means3D=packet['means3D'][unique].cpu().numpy(), native_scales=packet['scales'][unique].cpu().numpy(),
            native_rotations=packet['rotations'][unique].cpu().numpy(), native_opacities=packet['opacities'][unique].cpu().numpy(),
            contributor_is_new=new[unique])
        np.savez_compressed(output/f'{raw:03d}_groups.npz', **regions, road_alpha=actual_road_alpha,
            joint_alpha=joint_alpha, **{'road_only_'+k:v for k,v in road_groups.items()},
            **{'joint_'+k:v for k,v in joint_groups.items()})
        summaries = {}
        for name,mask in regions.items():
            summaries[name] = dict(pixels=int(mask.sum()), road_only_alpha_mean=float(actual_road_alpha[mask].mean()),
                joint_alpha_mean=float(joint_alpha[mask].mean()),
                joint_group_mass={key:float(value[mask].sum(dtype=np.float64)) for key,value in joint_groups.items()},
                road_only_group_mass={key:float(value[mask].sum(dtype=np.float64)) for key,value in road_groups.items()})
        rows.append(dict(raw=raw, regions=summaries, pixel_attribution=pixel_rows,
            group_conservation_max=conservation, group_joint_RGB_max_vs_saved=rgb_error))
        print(json.dumps(dict(stage='actual_joint_and_pixel_contribution_complete',raw=raw,calls=calls,
            remaining=len(ys),actual_contributing_rows=len(unique))),flush=True)
    n = len(ctx.road['active'])
    physical = (ctx.provenance['surface_supported'] & ctx.provenance['legal_write'] & ~ctx.provenance['unknown'] &
        (ctx.provenance['source'] > 0) & ~ctx.provenance['guarded_world_rows'] &
        (ctx.road['minimum_model_depth'] <= 0) & ~ctx.provenance['activation_K_guard'] & ~ctx.provenance['activation_near_guard'])
    endpoint = np.zeros(n,bool)
    endpoint[world_ids] = contribution > 0
    endpoint &= physical
    np.savez_compressed(output/'endpoint_candidate_scope.npz',world_eligible=endpoint,
        active_world_rows=world_ids, actual_remaining73_mass=contribution, physical_material_qualified=physical)
    ctx.assert_frozen()
    ctx.close()
    result = dict(schema='road_activation_actual_visibility.v1',GPU_executed=True,optimizer_updates=0,
        source_asset=str(asset),source_native=str(native_source),rows=rows,native_calls=calls,
        campaign_calls_including_first_control=45+calls,seconds=time.monotonic()-start,
        endpoint_actual_contributor_rows=int(endpoint.sum()),stored_scene_or_asset_modified=False,
        per_pixel_alphaT_is_actual_native=True,projected_radii_are_footprint_descriptors_not_attribution=True,
        endpoint_scope_note='actual residual73 positive contributors intersected with physical/material and inherited K/near; maximum-footprint K qualification still pending')
    write_json(output/'results.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asset',type=Path,default=CAMPAIGN/'native_r1/asset_A1_guarded')
    p.add_argument('--native-source',type=Path,default=CAMPAIGN/'native_r1')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--lease-file',type=Path,default=CAMPAIGN/'gpu_lease.json')
    p.add_argument('--lease-owner',required=True)
    a=p.parse_args()
    result=run(a.asset,a.native_source,a.output,a.lease_file,a.lease_owner)
    print(json.dumps(dict(completed=True,native_calls=result['native_calls'],endpoint_rows=result['endpoint_actual_contributor_rows'])),flush=True)
