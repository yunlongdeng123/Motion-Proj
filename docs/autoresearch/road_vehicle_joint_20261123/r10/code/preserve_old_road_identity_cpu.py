"""Bind saved context-road contributions to frozen RoadRows world-row IDs.

Identity mapping only: no retirement policy, packet mutation or GPU work.
"""
from pathlib import Path
import argparse
import numpy as np
from local_gaussian_asset import read,write,bank,sha

def pin(p):return dict(path=str(p),sha256=sha(p),bytes=Path(p).stat().st_size)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['parent-contract','measurement','output','adapter-code']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();cfg=read(a.parent_contract);m=read(a.measurement/'receipt.json')
    assert m['complete'] and m['baseline_contract_sha256']==sha(a.parent_contract)
    rp=Path(cfg['road_asset'])/'road.npz';sp=Path(cfg['scene_proof']);proof=read(sp);road=bank(rp)
    active=np.flatnonzero(road['active']).astype(np.int64);offset=int(proof['topology']['static_rows'])+int(proof['topology']['dynamic_rows'])
    assert len(active)>0 and not a.output.exists();a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'road_active_identity.npz',road_active_index=np.arange(len(active),dtype=np.int64),
        world_row_id=active,global_row_id=offset+np.arange(len(active),dtype=np.int64),native_global_row_count=np.asarray(offset))
    rows=[]
    for raw in cfg['frames']:
        source=a.measurement/f'{raw:03d}/all_context_rows.npz';fr=a.measurement/f'{raw:03d}/receipt.json';receipt=read(fr)
        assert receipt['packet']['sha256']==sha(source) and receipt['native_global_row_count']==offset
        z=bank(source);ids=z['global_row_ids'];native=z['native_context'];assert np.array_equal(native,ids<offset)
        pos=np.flatnonzero(~native);ri=ids[pos]-offset
        assert ((ri>=0)&(ri<len(active))).all() and np.array_equal(ri,np.unique(ri))
        world=active[ri];folder=a.output/f'{raw:03d}';folder.mkdir()
        np.savez_compressed(folder/'old_road_row_map.npz',context_prefix_position=pos.astype(np.int64),
            global_row_ids=ids[pos],road_active_index=ri,world_row_id=world,owner_ids=z['owner_ids'][pos],
            ROI_names=z['ROI_names'],ROI_alphaT_mass=z['ROI_alphaT_mass'][pos],
            old_permanent_guard=z['old_permanent_guard'][pos])
        assert np.array_equal(ids[pos],offset+ri)
        record=dict(raw=raw,all_context_prefix_rows=len(ids),saved_old_road_rows=len(ri),
            mapping_rule='road_active_index = global_row_id-native_global_row_count; world_row_id=flatnonzero(road.active)[road_active_index]',
            road_rows_preserved_in_original_measurement=True,original_mass_source=pin(source),original_frame_receipt=pin(fr),
            mapping=pin(folder/'old_road_row_map.npz'),mapping_roundtrip_exact=True)
        write(folder/'receipt.json',record);rows.append(record)
    write(a.output/'receipt.json',dict(schema='R10.old_road_world_row_identity_CPU.v1',complete=True,
        decision='IDENTITY_MAPPING_ONLY_NO_RETIREMENT_QUALIFICATION',parent_contract=pin(a.parent_contract),measurement_receipt=pin(a.measurement/'receipt.json'),
        road_asset=pin(rp),persistent_scene_manifest=pin(Path(cfg['scene_manifest'])),persistent_scene_proof=pin(sp),
        adapter_code=pin(a.adapter_code),native_topology=proof['topology'],native_global_row_count=offset,
        road_total_world_rows=len(road['active']),road_active_rows=len(active),global_identity=pin(a.output/'road_active_identity.npz'),rows=rows,
        all_original_inputs_unchanged=True,retirement_policy_created=False,old_road_opacity_modified=False,
        current_visibility_scope_only=True,expanded_visibility_requires_fresh_measurement=True,
        CPU_only=True,nativeF=0,nativeB=0,updates=0))
    print(dict(complete=True,road_active_rows=len(active),perview=[r['saved_old_road_rows'] for r in rows],GPU_calls=0,retirement_policy=False))

if __name__=='__main__':main()
