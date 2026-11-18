"""只CPU消费已完成A1+95原生guard；不声称新三态API运行过native。"""
from pathlib import Path
import argparse
import json
import numpy as np
from module2_activation_state import prepare_request,load_fresh_native_gate,apply_fresh_gate,_sha
from road_activation_builder import CAMPAIGN,read_npz,write_json


def bridge(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    source=CAMPAIGN/'A1_beam95_native_r1'
    parent=CAMPAIGN/'native_r1/asset_A1_guarded'
    candidate=source/'asset_prepared'
    verified=source/'asset_guarded'
    producer=json.loads((source/'results.json').read_text())
    source_review=CAMPAIGN/'supervision_fixed_beam_r1/qualification_receipt.json'
    review=json.loads(source_review.read_text())
    if producer['GPU_executed'] is not True or review['status']!='PASS':raise ValueError('原native或source审查缺失')
    road,p=read_npz(parent/'road.npz'),read_npz(parent/'provenance.npz')
    proposal=read_npz(CAMPAIGN/'footprint_fixed_sigma_cpu_r3/observed_inactive_fixed_sigma_candidates.npz')
    ids=proposal['proposed_world_rows'];n=len(road['active'])
    if len(ids)!=95:raise ValueError('桥接只消费既有95控制')
    requested=np.zeros(n,bool);requested[ids]=True
    material=p['known']&(p['source']==1)&~p['unknown']
    request=prepare_request(parent_render_active=road['active'],geometry_support=p['surface_supported'],
        material_qualified=material,requested=requested,legacy_protected=p['guarded_world_rows']|(road['minimum_model_depth']>0),
        material_qualification_source=str(source_review))
    if not np.array_equal(request.candidate_world_rows,ids):raise ValueError('三态候选与既有95控制不同')
    candidate_road=candidate/'road.npz';candidate_scene=candidate/'persistent/scene.json'
    proof=json.loads((candidate/'persistent/proof.json').read_text())
    if proof['road']['sha256']!=_sha(candidate_road):raise ValueError('既有native候选body proof损坏')
    with np.load(candidate_road) as d:packet=np.flatnonzero(d['active'])
    g=read_npz(source/'new_packet_guards.npz')
    mask_path=output/'bound_existing_native_masks.npz'
    np.savez_compressed(mask_path,world_K_guard=g['world_K_guard'],world_near_guard=g['world_near_guard'],
        candidate_world_rows=ids,packet_active_world_rows=packet)
    K=json.loads((source/'new_packet_K.json').read_text())
    frames=tuple(row['raw'] for row in K['rows']);K_count=sum(row['K_pixels'] for row in K['rows'])
    if frames!=(86,110,134,179,198) or K_count!=77266:raise ValueError('既有native帧/K分母不符')
    receipt_path=output/'bound_existing_native_gate.json'
    gate_record=dict(schema='module2_fresh_native_gate.v1',GPU_executed=False,CPU_bridge_only=True,
        native_producer_GPU_executed=True,native_producer_receipt=str(source/'results.json'),
        native_producer_receipt_sha256=_sha(source/'results.json'),
        candidate_road=str(candidate_road),candidate_road_sha256=_sha(candidate_road),
        mask_sha256=_sha(mask_path),scene_manifest_sha256=_sha(candidate_scene),
        fixed_geometry_sigma_opacity=True,method='actual_native_fixed_sigma_tile_footprint_and_near',
        frames=list(frames),K_pixel_frames=K_count,additional_native_calls=0)
    write_json(receipt_path,gate_record)
    gate=load_fresh_native_gate(mask_path,receipt_path,candidate_road_path=candidate_road,candidate_scene_manifest_path=candidate_scene)
    decision=apply_fresh_gate(request,gate,candidate_road_sha256=_sha(candidate_road),
        required_frames=frames,required_K_pixel_frames=77266)
    actual=read_npz(verified/'road.npz')['active']
    if not np.array_equal(decision.render_active,actual) or int(decision.accepted_new.sum())!=95:
        raise AssertionError('新三态消费者与已验证资产激活集合不同')
    np.savez_compressed(output/'consumed_decision.npz',geometry_support=decision.geometry_support,
        material_qualified=decision.material_qualified,render_active=decision.render_active,
        accepted_new=decision.accepted_new,K_rejected=decision.K_rejected,near_rejected=decision.near_rejected)
    result=dict(schema='module2_existing_native_bridge_CPU.v1',CPU_only=True,GPU_executed=False,
        additional_native_calls=0,optimizer_updates=0,new_native_raster_executed_by_API=False,
        native_producer_control=str(source/'results.json'),native_producer_calls_already_accounted=25,
        bound_candidate_body_sha256=_sha(candidate_road),bound_candidate_world_rows=95,
        actual_native_packet_active_rows=len(packet),frames=list(frames),K_pixel_frames=77266,
        accepted95_exact_existing_asset=True,full_render_active_exact_existing_asset=True,
        consumed_gate=str(receipt_path),default_scene_modified=False,opt_in_only=True)
    write_json(output/'bridge_receipt.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    print(json.dumps(bridge(p.parse_args().output),ensure_ascii=False),flush=True)
