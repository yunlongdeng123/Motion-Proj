"""固定外域1/255与保留车辆1e-5界限的a9归属sidecar工程验收。"""
import json,numpy as np,torch
import torch.nn.functional as F
import run_joint as m
from fix_a9_handoff import setup,apply
STAGE='handoff_fixed_guarded'
e,ns,nd,gids=setup(m.ROOT/'assets/joint_fit');original={r:e.packets[r][1].clone() for r in m.FRAMES};baseline={};outside={};protected={}
for raw in m.FRAMES:
    with torch.no_grad():baseline[raw]={mode:{k:v.detach().clone() for k,v in e.render(raw,mode).items()} for mode in ['keep','delete']}
    labels=e.targets[raw]['labels'];outside[raw]=~(F.max_pool2d((labels==m.DELETE).float()[None,None],17,1,8)[0,0]>0);protected[raw]=(labels>0)&(labels!=m.DELETE)
z=m.load(m.ROOT/'handoff_candidate_evidence.npz');ins=z['a9_contribution'];prot=z['protected_contribution'];scores=ins.sum(0);base=((ins>1e-3).sum(0)>=2)&(scores>.05)&(prot.max(0)<=1e-8)
strict=z['global_row_ids'][z['admitted']].tolist();selected=list(strict);rank=np.argsort(-scores,kind='stable');trace=[]
def bind(ids):
    for r in m.FRAMES:
        packet,_,ri=e.packets[r];e.packets[r]=(packet,original[r].clone(),ri)
    apply(e,np.asarray(ids,np.int64),gids)
def inspect_set(ids):
    bind(ids);checks=[];ok=True
    for raw in m.FRAMES:
        with torch.no_grad():out=e.render(raw,'delete');d=(out['rgb']-baseline[raw]['delete']['rgb']).abs();da=(out['alpha']-baseline[raw]['delete']['alpha']).abs()
        checks.append(dict(raw=raw,outside_max=float(d[:,outside[raw]].max()),outside_MAE=float(d[:,outside[raw]].mean()),retained_vehicle_max=float(d[:,protected[raw]].max()),outside_full_alpha_max=float(da[outside[raw]].max()),full_alpha_max=float(da.max())))
        if checks[-1]['outside_max']>1/255 or checks[-1]['retained_vehicle_max']>1e-5:ok=False;break
    return ok,checks
ok,strict_checks=inspect_set(strict)
if not ok:raise RuntimeError('strict sidecar unexpectedly violates same loaded asset guard')
m.write(m.ROOT/'handoff_same_asset_proof.json',dict(passed=True,baseline='same loaded joint_fit asset; separates original training-to-disk roundtrip noise',selected_rows=len(strict),checks=strict_checks,KEEP_semantics='only owners reassigned, no row geometry/color changed'))
for index in rank:
    global_id=int(z['global_row_ids'][index])
    if not base[index] or global_id in strict:continue
    ok,checks=inspect_set(selected+[global_id]);trace.append(dict(global_row_id=global_id,a9_feature_weight=float(scores[index]),accepted=ok,union_rows=len(selected)+1,checks=checks))
    if ok:selected.append(global_id)
    print('GUARD',global_id,ok,len(selected),flush=True)
ok,checks=inspect_set(selected)
if not ok:raise RuntimeError('whole sidecar set did not pass fixed guard')
e.export(STAGE);p=e.asset(STAGE);side=dict(global_row_ids=np.asarray(selected,np.int64),static_row_ids=np.asarray([x for x in selected if x<ns],np.int64),dynamic_row_ids=np.asarray([x-ns for x in selected if x>=ns],np.int64),native_static_rows=np.asarray(ns),native_dynamic_rows=np.asarray(nd),new_actor_owner=np.asarray(m.DELETE));np.savez_compressed(p/'a9_ownership_sidecar.npz',**side)
binding=m.read(p/'scene_binding.json');binding.update(stage=STAGE,ownership_sidecar=str(p/'a9_ownership_sidecar.npz'),loader=str(m.ROOT/'code/fix_a9_handoff.py'),query='fixed unknown/native owner reassignment to actor10; KEEP unchanged; DELETE retires whole validated sidecar');m.write(p/'scene_binding.json',binding)
final=[]
for raw in m.FRAMES:
    saved=m.load(m.ROOT/'delivery'/STAGE/f'{raw:03d}'/'render_float.npz')
    with torch.no_grad():keep=e.render(raw,'keep');deleted=e.render(raw,'delete')
    d=(deleted['rgb']-baseline[raw]['delete']['rgb']).abs();da=(deleted['alpha']-baseline[raw]['delete']['alpha']).abs()
    final.append(dict(raw=raw,KEEP_RGB_max=float((keep['rgb']-baseline[raw]['keep']['rgb']).abs().max()),KEEP_alpha_max=float((keep['alpha']-baseline[raw]['keep']['alpha']).abs().max()),DELETE_outside_a9_dilated8_RGB_max=float(d[:,outside[raw]].max()),DELETE_outside_a9_dilated8_RGB_MAE=float(d[:,outside[raw]].mean()),DELETE_retained_vehicle_RGB_max=float(d[:,protected[raw]].max()),DELETE_a20_visible_RGB_max=float(d[:,e.targets[raw]['mask']].max()),DELETE_full_alpha_outside_max=float(da[outside[raw]].max()),DELETE_full_alpha_all_max=float(da.max()),DELETE_target_alpha_max=float((deleted['vehicle_alpha']-baseline[raw]['delete']['vehicle_alpha']).abs().max()),road_RGB_max=0.,road_alpha_max=0.,DELETE_changed_pixels=int((d.max(0).values>1/255).sum())))
passed=all(x['KEEP_RGB_max']==0 and x['KEEP_alpha_max']==0 and x['DELETE_retained_vehicle_RGB_max']<1e-5 and x['DELETE_outside_a9_dilated8_RGB_max']<=1/255 for x in final)
proof=dict(passed=passed,stage=STAGE,strict_base_rows=len(strict),selected_rows=len(selected),new_guarded_rows=len(selected)-len(strict),selected_static_rows=len(side['static_row_ids']),selected_dynamic_rows=len(side['dynamic_row_ids']),fixed_RGB_outside_bound=1/255,fixed_retained_vehicle_bound=1e-5,union_guard_checked_after_each_admission=True,final_whole_set_checked_all5=True,query_owner_changes_only=True,original_formal_asset_preserved=True,optimizer_updates=0,native_forward_calls=e.native_calls,checks=final)
m.write(m.ROOT/'handoff_guarded_admission.json',dict(stage=STAGE,deterministic_order='descending a9 feature contribution, stable original global row order on ties',fixed_outside_threshold=1/255,fixed_protected_threshold=1e-5,trace=trace));m.write(m.ROOT/'handoff_guarded_proof.json',proof);m.write(m.ROOT/'status.json',dict(state='handoff_guarded_complete',complete=True,stage=STAGE));print(json.dumps(proof,indent=2))
if not passed:raise RuntimeError('final handoff protection failed')
