"""Bind already accepted geometry/material receipts; never creates approval."""
from pathlib import Path
import argparse
import numpy as np
from local_gaussian_asset import read,write,sha,preflight

def pin(p):
    p=Path(p).resolve();assert p.is_file();return dict(path=p.as_posix(),sha256=sha(p))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['r10','geometry','certificate','coverage','inferred','output']:
        p.add_argument('--'+k,type=Path,required=True)
    for k in ['field','numeric','visual','root']:p.add_argument('--'+k,type=Path)
    p.add_argument('--source-receipt',type=Path,help='Required when the source carries an explicit expanded-canvas embedding')
    p.add_argument('--bind-geometry-only',action='store_true')
    a=p.parse_args();assert not a.output.exists()
    parent_path=a.r10/'baseline_vehicle0_road400.json';parent=read(parent_path)
    cfg=read(a.r10/'local_asset_contract.pending.json');cfg['parent_contract']=pin(parent_path)
    paths=dict(actor_asset=parent['actors'][0]['asset'],road_asset=str(Path(parent['road_asset'])/'road.npz'),
        scene_manifest=parent['scene_manifest'],scene_proof=parent['scene_proof'],context_plan=parent['context_plan'],actor_field_module=parent['actor_field_module'])
    cfg['parent_pins']=[dict(role=role,**pin(path)) for role,path in paths.items()]
    cfg['geometry']=dict(npz=pin(a.geometry),certificate=pin(a.certificate),target_coverage_gate=pin(a.coverage),inferred_gate=pin(a.inferred))
    if not a.bind_geometry_only:
        if not all([a.field,a.numeric,a.visual,a.root]):p.error('Full binding needs --field --numeric --visual --root')
        cfg['source'].update(field=pin(a.field),numeric_gate=pin(a.numeric),visual_gate=pin(a.visual),root_gate=pin(a.root))
        with np.load(a.field,allow_pickle=False) as z:
            if 'canvas_uv_origin_global' in z.files:
                if a.source_receipt is None:p.error('Expanded canvas requires --source-receipt for independent UV-origin binding')
                receipt=read(a.source_receipt);embedding=receipt['same_global_UV_canvas_embedding']
                origin=np.asarray(z['canvas_uv_origin_global'],np.int64)
                assert origin.shape==(2,) and np.array_equal(origin,np.asarray(embedding['origin_global'],np.int64))
                assert np.array_equal(z['canvas_shape'],np.asarray(z['atlas_linear'].shape[:2])) and np.array_equal(z['canvas_shape'],embedding['canvas_shape'])
                assert np.array_equal(z['canvas_parent_shape'],embedding['parent_shape'])
                cfg['source']['uv_origin_global']=origin.tolist();cfg['source']['canvas_receipt']=pin(a.source_receipt)
                if 'qualification_repair_addresses' in receipt:
                    keys=['qualification_repair_addresses','ancestor_real_lock_count','unaffected_inherited_real_lock_count',
                          'fresh_recertified_real_count','branch_real_lock_count','mutable_only_write_pixels','new_material_extent_texels']
                    stats={key:int(receipt[key]) for key in keys}
                    assert int(z['real_locks'].sum())==stats['branch_real_lock_count']
                    assert stats['ancestor_real_lock_count']-stats['unaffected_inherited_real_lock_count']==stats['qualification_repair_addresses']
                    assert stats['unaffected_inherited_real_lock_count']+stats['fresh_recertified_real_count']==stats['branch_real_lock_count']
                    cfg['source']['same_world_requalification']=dict(stats,
                        policy='Changed XYZ requires fresh visibility qualification; unrequalified ancestor addresses are references/inference rather than same-world real locks')
    for role,filename in [('npz','native_gate0.npz'),('proof','native_gate0.proof.json'),('candidates','native_candidates.npz')]:cfg['native_gate'][role]=pin(a.r10/'native_identity_gate_v1'/filename)
    cfg['r8_code']=(a.r10.parent/'road_vehicle_joint_dggt_20261123_r8/code').resolve().as_posix()
    cfg.pop('no_GPU_execution_authorized_by_this_pending_file',None);cfg['ready']=False
    if a.bind_geometry_only:
        cfg['source_acceptance']='PENDING independent and root review; no approval created by this binder'
        cfg['no_GPU_execution_authorized_by_this_pending_file']=True;write(a.output,cfg)
        print(dict(geometry_parent_native_pins_bound=True,source_pending=True,ready=False,GPU_calls=0));return
    write(a.output,cfg);status=preflight(a.output)[0];cfg['ready']=status['ready'];cfg['preflight_result']=status;write(a.output,cfg)
    write(a.output.with_suffix('.preflight.json'),status);print(status)
    if not status['ready']:raise SystemExit(2)

if __name__=='__main__':main()
