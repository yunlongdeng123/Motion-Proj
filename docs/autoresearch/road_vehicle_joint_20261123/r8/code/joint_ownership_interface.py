"""CPU-only identity channels and a closed-by-default native update scope.

The renderer must integrate each Gaussian exactly once. Identity masks only
label those existing rows; they neither duplicate fields nor change opacity.
No torch import, raster call, optimizer, asset mutation or implicit permission.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity_channels(branch, native_owner, actor_owner, selected_owners):
    """Return disjoint source channels and semantic identity aliases.

    Branch codes: 0 native static, 1 native dynamic, 2 road, 3 canonical actor.
    actor_owner must explicitly identify every canonical row in a multi-actor
    packet. native_owner is the ACTUAL loaded owner, after source fallback and
    the once-only sidecar; raw semantic zero is not an authoritative owner.
    """
    branch, native_owner, actor_owner = map(np.asarray, (branch, native_owner, actor_owner))
    assert branch.shape == native_owner.shape == actor_owner.shape
    assert np.isin(branch, [0, 1, 2, 3]).all()
    selected_owners = tuple(sorted(set(map(int, selected_owners))))
    assert np.isin(actor_owner[branch == 3], selected_owners).all()
    native = branch < 2
    selected_native = native & np.isin(native_owner, selected_owners)
    channels = {'road': branch == 2,
                'native_unowned': native & (native_owner <= 0),
                'native_other_owner': native & (native_owner > 0) & ~selected_native}
    aliases = {}
    for owner in selected_owners:
        keys = [f'actor_{owner}_{suffix}' for suffix in ('native_static', 'native_dynamic', 'canonical')]
        channels[keys[0]] = (branch == 0) & (native_owner == owner)
        channels[keys[1]] = (branch == 1) & (native_owner == owner)
        channels[keys[2]] = (branch == 3) & (actor_owner == owner)
        aliases[f'actor_{owner}_unified'] = keys
    partition = np.stack(list(channels.values())).sum(0)
    assert (partition == 1).all(), 'Every existing row must be integrated/labeled once'
    return channels, aliases


def native_update_scope(packets, required_frames, selected_owners, old_preserved_ids):
    """Exact-zero measured K guard; a missing contributor record is UNKNOWN.

    Current measurement packets omit K-only rows, so intersection of present
    native IDs is mandatory. No epsilon threshold or missing-to-zero fill.
    Old K-positive preservation is an additional immutable exclusion.
    """
    selected_owners = tuple(map(int, selected_owners))
    by_frame, union = {}, set()
    for raw, packet in packets.items():
        pick = (packet['branch'] < 2) & np.isin(packet['semantic_owner'], selected_owners)
        ids = packet['native_global_id'][pick].astype(np.int64)
        assert len(np.unique(ids)) == len(ids)
        values = packet['ROI_mass'][pick, 2]
        assert (values >= 0).all()
        by_frame[int(raw)] = {int(gid): float(value) for gid, value in zip(ids, values)}
        union.update(by_frame[int(raw)])
    missing = sorted(set(map(int, required_frames)) - set(by_frame))
    present = set.intersection(*(set(by_frame[int(raw)]) for raw in required_frames)) if not missing else set()
    clear = {gid for gid in present if all(by_frame[int(raw)][gid] == 0 for raw in required_frames)}
    eligible = clear - set(map(int, old_preserved_ids))
    return np.asarray(sorted(eligible), np.int64), dict(
        required_frames=list(map(int, required_frames)), measured_frames=sorted(by_frame),
        missing_frames=missing, present_in_all_required_packets=len(present),
        all_required_measured_Kzero_rows=len(clear), old_K_preserved_excluded=len(clear)-len(eligible),
        eligible_native_rows=len(eligible), missing_row_K_is_unknown=True,
        candidate_union_rows=len(union), K_zero_threshold=0.0,
        K_scope_complete=not missing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--measurements', type=Path, required=True)
    parser.add_argument('--required-frames', nargs='+', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = read(args.contract)
    selected = [int(spec['owner_id']) for spec in cfg['actors']]
    plan = read(cfg['context_plan'])
    packets, inputs, checks = {}, [], []
    for raw in args.required_frames:
        path = args.measurements/f'{raw:03d}/actual_contributing_rows.npz'
        if not path.exists():
            continue
        packet, receipt = load(path), read(path.parent/'receipt.json')
        assert sha(path) == receipt['row_packet_SHA']
        actor_owner = packet.get('canonical_owner')
        if actor_owner is None:
            assert len(selected) == 1, 'Multi-actor input requires per-row canonical_owner'
            actor_owner = np.full(len(packet['branch']), selected[0], np.int64)
        channels, aliases = identity_channels(packet['branch'], packet['semantic_owner'], actor_owner, selected)
        masses = {key: packet['ROI_mass'][mask, 0].astype(np.float64).sum().item() for key, mask in channels.items()}
        alias_mass = {key: sum(masses[source] for source in sources) for key, sources in aliases.items()}
        strict = next(x for x in receipt['regions'] if x['ROI'] == 'strictV_iso95_jointbelow95')
        error = abs(sum(masses.values()) - strict['joint_alpha_mass'])
        assert error < .003, ('VJP float accumulation mismatch', raw, error)
        checks.append(dict(raw=raw,source_channel_mass=masses,unified_actor_mass=alias_mass,
                           partition_mass_abs_error=error,semantic_readout_only=True))
        packets[raw] = packet
        inputs.append(dict(raw=raw,path=str(path),sha256=sha(path),receipt_sha256=sha(path.parent/'receipt.json')))
    eligible, scope = native_update_scope(packets,args.required_frames,selected,plan['preserved_native_context_global_row_ids'])
    np.savez_compressed(args.output/'native_update_candidates.npz',eligible_native_global_ids=eligible)
    result = dict(schema='R8.joint_identity_and_update_interface.v1',complete=True,
        CPU_only=True,nativeF=0,nativeB=0,parameter_updates=0,inputs=inputs,
        identity_channel_checks=checks,K_scope=scope,training_enabled=False,
        blocked_by=['R9 source acceptance pending','all required-view native K measurements incomplete',
                    'root semantic/visibility review required'],
        identity_contract=dict(canonical_and_native_same_owner_are_one_instance=True,
             each_Gaussian_rasterized_once=True,static_unowned_not_assimilated=True,
             attribution_alias_changes_RGB=False,do_not_force_canonical_solo_alpha_to_one=True),
        future_parameter_interface=dict(
             native_shared_by_global_id=['RGB delta[gid,3]','activated opacity delta[gid,1]'],
             native_center_scale_rotation_trajectory_fixed=True,
             canonical_actor=['existing local RGB/SH','local opacity/scale only if separately evidenced holes'],
             canonical_centers_rotation_trajectory_active_fixed=True,
             road=['shared accepted-source RGB','local tangent sigma/opacity only if separately evidenced scene holes'],
             road_centers_normal_axis_rotation_active_fixed=True,
             native_opacity_gate_allowed_ids=str(args.output/'native_update_candidates.npz')),
        future_loss_interface=dict(
             RGB='one actual shared raster RGB; actor real photo / road accepted R9 source in distinct certified domains',
             actor_visibility='same shared alphaT sum over canonical + retained native rows of that instance',
             branch_diagnostics='canonical/native-static/native-dynamic alphaT separately retained',
             strictV_and_proxy='separate denominators; proxy never relabeled as strict V',
             protection='corrected-pinhole K plus immutable old K-positive row guard; reject affected row updates',
             ownership_is_not_geometric_truth=True),
        integration_point='prepare identity labels from JointNativeAdapter.context + explicit actor list; write feature channels before the existing single raster; do not change _prepare, row packing, means, near or metric policy')
    (args.output/'joint_ownership_plan.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(complete=True,training_enabled=False,K_scope=scope,checks=checks),ensure_ascii=False))


if __name__ == '__main__':
    main()
