"""CPU paired A0/atlas0/formal400 metrics; every original scope is retained.

This writes machine-readable evidence, not a publication/final acceptance.
Unavailable material targets stay unavailable; dark pixels never define scope.
"""
from pathlib import Path
import argparse
import numpy as np
from scipy.ndimage import distance_transform_edt
from joint_rgb_common import read,load,write,sha,srgb_to_linear_numpy,linear_to_srgb_numpy


def pin(path):
    path=Path(path);return dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size) if path.is_file() else dict(path=str(path),missing=True)


def review(path):
    if not path or not Path(path).is_file():return dict(provided=False,decision='PENDING')
    item=read(path);return dict(provided=True,reference=pin(path),complete=item.get('complete'),decision=item.get('decision'),
        accepted=item.get('complete') is True and item.get('decision')=='PASS')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--fresh',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--asset-review',type=Path);p.add_argument('--render-review',type=Path)
    p.add_argument('--replay-review',type=Path);a=p.parse_args();cfg=read(a.run/'fit_contract.json');base=read(a.run/'effective_baseline_contract.json')
    run=read(a.run/'receipt.json');assert run['complete'] and run['logical_steps']==400 and run['actual_optimizer_updates']==400
    a.output.mkdir(parents=True,exist_ok=False);rows=[];inputs=[]
    r7=Path(base['camera_audit_cache']).parent;rim_path=r7/'cpu_review/black_rim_domain_diagnostic_v1/fixed_primary1792_masks.npz'
    for raw in cfg['frames']:
        cache_path=Path(base['camera_audit_cache'])/f'{raw:03d}/view_cache.npz';cache=load(cache_path)
        mask_path=Path(cfg['rectified_masks'])/f'{raw:03d}/rectified_masks.npz';mask=load(mask_path)
        target_path=a.run/f'final_readout/{raw:03d}/source_target.npz';target=load(target_path)
        F=target['canonical_linear'].transpose(2,0,1);support=target['support'];gain=float(target['exposure_gain'])
        zero_target=load(a.run/f'atlas_assignment_0/{raw:03d}/source_target.npz')
        assert all(zero_target[k].dtype==target[k].dtype and zero_target[k].shape==target[k].shape and zero_target[k].tobytes()==target[k].tobytes() for k in target),'Source/denominator changed between atlas0 and final'
        K=mask['protected'];V=mask['actor_visible'] if raw in cfg['train_frames'] else np.zeros_like(K);proxy=mask['actor_visible_proxy']
        photo=cache['photo_pinhole_srgb'].transpose(2,0,1);CS=cache['teacher_edit_requested']
        scopes=dict(all_material_support=support,original_C=cache['requested_C'],original_S=cache['requested_S'],original_CS=CS,
            full_CS_dilation8=distance_transform_edt(~CS)<=8,strict_train_V=V,actor_proxy=proxy,corrected_K=K)
        if raw==110:
            rim=load(rim_path)['rim1792'];assert rim.sum()==1792;scopes['fixed1792_rim']=rim;inputs.append(pin(rim_path))
        stages={};files={}
        for stage,directory in [('A0',Path(cfg['baseline_render'])),('atlas0',a.run/'atlas_assignment_0'),('formal400',a.run/'final_readout')]:
            jpath=directory/f'{raw:03d}/joint_render.npz';cpath=directory/f'{raw:03d}/component_render.npz'
            j,c=load(jpath),load(cpath);stages[stage]=(j,c);files[stage]=[pin(jpath),pin(cpath)]
        reference_joint,reference_component=stages['A0']
        layer_path=a.run/f'supervision_layers/{raw:03d}/layer_integrals.npz';layers=load(layer_path) if layer_path.is_file() else None
        frame=dict(raw=raw,train_view=raw in cfg['train_frames'],gain=gain,scopes={},files=files,
            target=pin(target_path),photo_cache=pin(cache_path),corrected_masks=pin(mask_path),lineage=pin(layer_path))
        for name,domain in scopes.items():
            domain=domain.astype(bool);paired=domain&support;material=paired&(reference_component['road_alpha'].squeeze()>.95)
            jointroad=paired&(reference_joint['road_alphaT']>.95);source=target['source_weights']
            stat=dict(fixed_denominator=int(domain.sum()),paired_material_target_pixels=int(paired.sum()),
                no_certified_shared_target_pixels=int((domain&~support).sum()),material_alpha95_pixels=int(material.sum()),joint_road_alphaT95_pixels=int(jointroad.sum()),
                protected_K_pixels=int((domain&K).sum()),Brown_invalid_pixels=int((domain&~mask['RGB_valid']).sum()),
                cached_geometry_unprojected_pixels=int((domain&~(cache['geometry_valid']&cache['projection_valid'])).sum()),stages={})
            for stage,(j,c) in stages.items():
                assert np.array_equal(j['alpha'],reference_joint['alpha'])
                assert np.array_equal(j['road_alphaT'],reference_joint['road_alphaT'])
                assert np.array_equal(c['road_alpha'],reference_component['road_alpha'])
                rawjoint=j['rgb'];road=c['road_rgb'];alpha=c['road_alpha'].squeeze();straight=road/np.maximum(alpha,1e-6)
                lin=srgb_to_linear_numpy(rawjoint)*gain;roadlin=srgb_to_linear_numpy(straight)*gain
                joint_residual=lin-F;road_residual=roadlin-F
                mae=lambda z,m:float(abs(z)[:,m].mean()) if m.any() else None
                scope_valid=domain&mask['RGB_valid']
                result=dict(joint_RGB_vs_shared_target_linear_MAE_all_supported=mae(joint_residual,paired),
                    joint_road_dominant_target_linear_MAE=mae(joint_residual,jointroad),
                    road_alpha_normalized_material_linear_MAE=mae(road_residual,material),
                    joint_sRGB_mean=rawjoint[:,domain].mean(1).tolist() if domain.any() else None,
                    road_premult_sRGB_mean=road[:,domain].mean(1).tolist() if domain.any() else None,
                    native_joint_dark_max_sRGB_below40_255=int((domain&(rawjoint.max(0)<40/255)).sum()),
                    native_road_premult_dark_max_sRGB_below40_255=int((domain&(road.max(0)<40/255)).sum()),
                    scene_alpha_lt50=int((domain&(j['alpha']<.5)).sum()),scene_alpha_lt95=int((domain&(j['alpha']<.95)).sum()),
                    road_only_alpha_lt95=int((domain&(alpha<.95)).sum()),
                    road_only_lt95_scene_ge95=int((domain&(alpha<.95)&(j['alpha']>=.95)).sum()),
                    joint_sRGB_L1_original_photo_diagnostic=mae(rawjoint-photo,scope_valid),
                    joint_RGB_max_delta_A0=float(abs(rawjoint-reference_joint['rgb'])[:,domain].max()) if domain.any() else None,
                    source_weighted_material_MAE={})
                for index,source_name in enumerate(['same_world','similar_reference','teacher_continuous']):
                    w=source[...,index][material];mass=float(w.sum())
                    result['source_weighted_material_MAE'][source_name]=dict(effective_pixel_weight_mass=mass,
                        MAE=float((abs(road_residual[:,material]).mean(0)*w).sum()/mass) if mass>0 else None)
                if K.any():assert rawjoint[:,K].tobytes()==reference_joint['rgb'][:,K].tobytes()
                stat['stages'][stage]=result
            if layers is not None:
                names=['native_road_alphaT','native_actor_alphaT','native_context_alphaT','trainable_road_alphaT','fixed_old_road_alphaT',
                    'retained_excluded_native_owner_alphaT','native_selected_instance_alphaT','unified_actor_alphaT']
                stat['same_actual_final_depth_order_contributions']={}
                for label in names:
                    value=layers[label];value=value.sum(0) if value.ndim==3 else value
                    stat['same_actual_final_depth_order_contributions'][label]=dict(mass=float(value[domain].astype(np.float64).sum()),
                        mean=float(value[domain].mean()) if domain.any() else None)
            frame['scopes'][name]=stat
        rows.append(frame);write(a.output/f'{raw:03d}_paired_fixed_scope_metrics.json',frame)
    resources=read(a.run/'resources.json');fresh=read(a.fresh/'receipt.json') if (a.fresh/'receipt.json').is_file() else None
    layer_receipt=a.run/'supervision_layers/receipt.json';layer=read(layer_receipt) if layer_receipt.exists() else None
    reviews=dict(persistent_assets=review(a.asset_review),fresh_native_readback=review(a.render_review),smoke_replay=review(a.replay_review))
    all_checks=all(x.get('accepted') is True for x in reviews.values())
    actual=dict(formal=dict(nativeF=resources['nativeF'],nativeB=resources['nativeB'],updates=resources['actual_optimizer_updates']),
        fresh=dict(nativeF=fresh['nativeF'],nativeB=fresh['nativeB'],updates=fresh['updates']) if fresh else None,
        supervision_readout=dict(nativeF=layer['nativeF'],nativeB=layer['nativeB'],updates=layer['updates']) if layer else None)
    result=dict(schema='R8.formal400_fixed_domain_compact_result.v1',complete=True,formal_actual_complete=True,
        source_sha256=cfg['source']['sha256'],formal_run=str(a.run),frames=cfg['frames'],train_frames=cfg['train_frames'],evaluation_only_frames=cfg['evaluation_only_frames'],
        paired_metrics=[pin(a.output/f'{raw:03d}_paired_fixed_scope_metrics.json') for raw in cfg['frames']],
        actual_resources=actual,all_fresh_and_independent_evidence_ready=bool(fresh and fresh['complete'] and layer and layer['complete'] and all_checks),
        independent_reviews=reviews,run_receipt=pin(a.run/'receipt.json'),resources=pin(a.run/'resources.json'),asset_proof=pin(a.run/'asset/proof.json'),
        fresh_receipt=pin(a.fresh/'receipt.json'),supervision_readout=pin(layer_receipt),
        frozen_geometry_and_K_held_by_actual_checks=True,new_diffusion_calls=0,
        scope_policy='Original material, C/S, Euclidean full C/S+8 and fixed1792 denominators retained; target validity/geometry/Brown/K stratified, never zero-filled',
        supervision_labels_are_not_causal_raw_RGB_components=True,human_verdict=None,whole_scene_restoration_accepted=False,problem_solved=None)
    write(a.output/'compact_result.json',result);print(dict(complete=True,compact=str(a.output/'compact_result.json'),sha256=sha(a.output/'compact_result.json'),all_evidence_ready=result['all_fresh_and_independent_evidence_ready']))


if __name__=='__main__':main()
