"""Read-only context inventory and existing joint-buffer mixture diagnostics."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json,hashlib,numpy as np

R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_vehicle_joint_dggt_20261123_r8');OUT=R/'audit/context_owner_CPU_v1'
def read(p):return json.loads(Path(p).read_text())
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def hist(x):
    k,n=np.unique(x,return_counts=True);return {str(int(a)):int(b) for a,b in zip(k,n)}
def stats(x):
    x=np.asarray(x);return dict(n=int(x.size),mean=float(x.mean()) if x.size else None,p05=float(np.percentile(x,5)) if x.size else None,
                              median=float(np.median(x)) if x.size else None,p95=float(np.percentile(x,95)) if x.size else None)


def main():
    OUT.mkdir(parents=True,exist_ok=False);cfg=read(R/'baseline_contract.json');proof=read(cfg['scene_proof']);ownership=proof['ownership']['path'];a=load(ownership)
    owners=np.r_[a['static_actor_ids'],a['dynamic_actor_ids']].astype(np.int32);ns=len(a['static_actor_ids']);ids=np.arange(len(owners));plan=read(cfg['context_plan'])
    for relabel in cfg['owner_relabels']:
        rows=load(relabel['path'])[relabel.get('key','global_row_ids')];owners[rows]=relabel['owner_id']
    removed=np.isin(ids,plan['remove_native_global_row_ids']);preserved=np.isin(ids,plan['preserved_native_context_global_row_ids'])
    kept=~removed&~np.isin(owners,cfg['exclude_native_owner_ids'])
    inventory={}
    for name,branch in [('static',ids<ns),('dynamic',ids>=ns)]:
        inventory[name]=dict(original_owner_counts=hist(owners[branch]),retained_topology_owner_counts=hist(owners[branch&kept]),
                             removed_owner_counts=hist(owners[branch&removed]),preserved_K_owner_counts=hist(owners[branch&preserved]))
    mixture=[]
    for raw in cfg['frames']:
        root=R/'baseline_A0_five_views'/f'{raw:03d}';j=load(root/'joint_render.npz');c=load(root/'component_render.npz')
        masks=load(R/f'audit/rectified_masks_v1/{raw:03d}/rectified_masks.npz');v=load(R.parent/f'road_shared_appearance_20261122_r7/projection_cache_v1/{raw:03d}/view_cache.npz')
        iso=c['actor_alpha'].squeeze();ja=j['actor_alphaT'];jr=j['road_alphaT'];jc=j['context_alphaT'];photo=v['photo_pinhole_srgb'];rgb=j['rgb'].transpose(1,2,0)
        cohorts={}
        for key in ['actor_visible','actor_visible_proxy']:
            if key not in masks:continue
            scope=masks[key].astype(bool)
            for threshold in [.5,.95]:
                take=scope&(iso>=.95)&(ja<threshold)
                cohorts[f'{key}_iso95_jointbelow{threshold}']=dict(pixels=int(take.sum()),joint_actor=stats(ja[take]),joint_context=stats(jc[take]),joint_road=stats(jr[take]),
                    context_dominates_selected_actor=int((take&(jc>ja)).sum()),road_dominates_selected_actor=int((take&(jr>ja)).sum()),
                    joint_RGB_L1=stats(abs(rgb-photo)[take]),component_alpha_sum_error=stats(abs(ja+jr+jc-j['alpha'])[take]))
        mixture.append(dict(raw=raw,cohorts=cohorts))
    result=dict(schema='R8.native_context_topology_and_existing_mixture_CPU.v1',complete=True,CPU_only=True,GPU_calls=0,optimizer_updates=0,
        ownership_SHA=sha(ownership),context_plan_SHA=sha(cfg['context_plan']),static_rows=ns,dynamic_rows=len(owners)-ns,
        remove_union_rows=int(removed.sum()),preserved_K_rows=int(preserved.sum()),remove_preserve_overlap=int((removed&preserved).sum()),
        inventory=inventory,mixture=mixture,
        inventory_scope='Topology only; per-frame lifetime, native legacy-near, provider retirements and opacity are not visibility contributions. Actual owner measurement is required.',
        context_policy=plan['rule'],later_cleanup_sidecar_applied=False,low_owner_not_automatically_wrong_occlusion=True)
    write(OUT/'context_inventory_and_mixture.json',result)
    print(json.dumps(dict(inventory=inventory,mixture=mixture),ensure_ascii=False))


if __name__=='__main__':main()
