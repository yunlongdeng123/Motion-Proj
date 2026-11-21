"""a9残留归属sidecar：未知native行，经真实贡献与保护域验收后赋owner10。"""
from pathlib import Path
import argparse,json,numpy as np,torch
import torch.nn.functional as F
import run_joint as m

def setup(asset):
    e=m.Experiment(asset);a=m.load(asset/'vehicle.npz');e.initial=a;e.vehicle=m.Vehicle(a).cuda();poses=m.load(asset/'poses.npz');e.poses={int(r):m.ten(p) for r,p in zip(poses['raw'],poses['model_from_actor'])}
    for p in e.vehicle.parameters():p.requires_grad_(False)
    e.road_color.requires_grad_(False)
    ns=len(e.scene.static.get_xyz);nd=len(e.scene.dynamic.get_xyz);global_ids={}
    for raw in m.FRAMES:
        with torch.no_grad():values,owners,cam,_=e.scene._rows(raw,(m.TARGET,),(),False)
        idx=torch.nonzero(values['mask']).reshape(-1);idx=idx[owners[idx]!=m.TARGET]
        if not torch.equal(values['means3D'][idx],e.packets[raw][0]['means3D']):raise RuntimeError('native global row mapping mismatch')
        global_ids[raw]=idx
    return e,ns,nd,global_ids

def apply(e,ids,global_ids):
    selected=m.ten(ids,torch.long)
    for raw in m.FRAMES:
        rows,owners,ri=e.packets[raw];owners=owners.clone();match=torch.isin(global_ids[raw],selected)
        if ((owners[match]>0)&(owners[match]!=m.DELETE)).any():raise RuntimeError('sidecar attempted to retire another actor')
        owners[match]=m.DELETE;e.packets[raw]=(rows,owners,ri)

def probe_rows(e,raw,weights,lookup,global_ids,ns):
    packet,owners,ri=e.packets[raw];gids=global_ids[raw];valid=gids<len(lookup);loc=torch.full_like(gids,-1);loc[valid]=lookup[gids[valid]];idx=torch.nonzero(loc>=0).reshape(-1)
    f=packet['features'].clone();f[:,4:7]=0;f[:,4]=f[:,4].index_copy(0,idx,weights[loc[idx]]);f[idx,5]=(gids[idx]<ns).float();f[idx,6]=(gids[idx]>=ns).float()
    rows=dict(packet,features=f);car=e.vehicle.rows(e.poses[raw],e.cameras[raw]);rows=e.scene._bindings().merge_rows(rows,car);owners=torch.cat([owners,torch.full((len(car['means3D']),),m.TARGET,device='cuda',dtype=torch.long)])
    rows=dict(rows,mask=rows['mask']&(owners!=m.DELETE));out=e.scene._bindings().raster(e.cameras[raw],e.scene.pipe,0,rows,e.scene._background);e.native_calls+=1
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reload-asset',type=Path);args=ap.parse_args()
    asset=args.reload_asset or m.ROOT/'assets/joint_fit';e,ns,nd,gids=setup(asset)
    if args.reload_asset:
        stage=m.read(asset/'scene_binding.json').get('stage','handoff_fixed');side=m.load(asset/'a9_ownership_sidecar.npz');apply(e,side['global_row_ids'],gids);e.export(stage+'_reloaded');rows=[]
        for raw in m.FRAMES:
            a=m.load(m.ROOT/'delivery'/stage/f'{raw:03d}'/'render_float.npz');b=m.load(m.ROOT/'delivery'/(stage+'_reloaded')/f'{raw:03d}'/'render_float.npz');rows.append(dict(raw=raw,max={k:float(np.abs(a[k]-b[k]).max()) for k in a}))
        m.write(m.ROOT/(stage+'_disk_reload_proof.json'),dict(passed=all(v<2e-5 for r in rows for v in r['max'].values()),fresh_process=True,rows=rows,asset=str(asset)));return
    m.status('a9_residual_contribution_diagnosis');eligible=torch.zeros(ns+nd,device='cuda',dtype=torch.bool);geometry_records=[]
    for raw in m.FRAMES:
        packet,owners,ri=e.packets[raw];glob=gids[raw];obs=m.read(m.BANK/'observations'/f'{raw:03d}.json');box=next(x for x in obs['boxes'] if x['track']=='NTjq-OhZ_97OMJ4xwJAxXg');pose=m.ten(e.transform@np.asarray(box['world_from_actor']).reshape(4,4));local=(packet['means3D']-pose[:3,3])@torch.linalg.inv(pose[:3,:3]).T;half=m.ten(box['dimensions'])/2
        plausible=(local[:,:2].abs()<=half[:2]+.75).all(1)&(local[:,2]>-half[2]+.08)&(local[:,2]<half[2]+.4)
        choose=(owners<=0)&(glob<ns+nd)&plausible
        eligible[glob[choose]]=True;geometry_records.append(dict(raw=raw,unknown_background_native_rows_in_measured_a9_volume=int(choose.sum())))
    candidates=torch.nonzero(eligible).reshape(-1);lookup=torch.full((ns+nd,),-1,device='cuda',dtype=torch.long);lookup[candidates]=torch.arange(len(candidates),device='cuda');weights=torch.ones(len(candidates),device='cuda',requires_grad=True)
    inside=np.zeros((len(m.FRAMES),len(candidates)),np.float64);outside=np.zeros_like(inside);protected=np.zeros_like(inside)
    for fi,raw in enumerate(m.FRAMES):
        out=probe_rows(e,raw,weights,lookup,gids,ns);channel=out['extra'][2];labels=e.targets[raw]['labels'];a9=labels==m.DELETE;retained=(labels>0)&(labels!=m.DELETE);expanded=F.max_pool2d(a9.float()[None,None],17,1,8)[0,0]>0
        for i,(domain,target) in enumerate([(a9,inside),(~expanded,outside),(retained,protected)]):target[fi]=m.cpu(torch.autograd.grad(channel[domain].sum(),weights,retain_graph=i<2)[0]).numpy()
        p=m.ROOT/'handoff_diagnostics'/f'{raw:03d}';m.png(p/'native_unknown_candidate_alpha.png',channel.detach());m.png(p/'static_candidate_alpha.png',out['extra'][3].detach());m.png(p/'dynamic_candidate_alpha.png',out['extra'][4].detach());m.png(p/'a9_mask.png',a9)
        print('PROBE',raw,len(candidates),float(inside[fi].sum()),float(outside[fi].sum()),float(protected[fi].sum()),flush=True)
    seen=(inside>1e-3).sum(0);admit=(seen>=2)&(inside.sum(0)>.05)&(outside.max(0)<=1e-8)&(protected.max(0)<=1e-8);selected=m.cpu(candidates).numpy()[admit]
    proof=dict(schema='a9_native_ownership_sidecar.v1',geometric_evidence='known a9 world OBB +0.75m horizontal measurement guard, z above annotated base +0.08m; only original unknown/background owners',actual_contribution_evidence='DELETE unified raster unit-feature VJP, positive a9 mask contribution in >=2/5 fixed frames; all5 protected and outside constraints',outside_domain='outside original a9 mask dilated8px; exact zero accepted feature contribution',protected_domain='all retained accepted vehicle labels including a20 true visible region',candidate_rows=len(candidates),selected_rows=len(selected),selected_static_rows=int((selected<ns).sum()),selected_dynamic_rows=int((selected>=ns).sum()),selected_inmask_unit_weight=float(inside[:,admit].sum()),max_selected_outside_weight=float(outside[:,admit].max()) if admit.any() else 0.,max_selected_protected_weight=float(protected[:,admit].max()) if admit.any() else 0.,per_view_geometry=geometry_records,optimizer_updates=0,original_road_unchanged=True)
    m.write(m.ROOT/'handoff_selection.json',proof);np.savez_compressed(m.ROOT/'handoff_candidate_evidence.npz',global_row_ids=m.cpu(candidates).numpy(),a9_contribution=inside,outside_contribution=outside,protected_contribution=protected,admitted=admit)
    if not len(selected):raise RuntimeError('no unknown a9 residual passed protected contribution selection')
    side=dict(global_row_ids=selected,static_row_ids=selected[selected<ns],dynamic_row_ids=selected[selected>=ns]-ns,native_static_rows=np.asarray(ns),native_dynamic_rows=np.asarray(nd),new_actor_owner=np.asarray(m.DELETE))
    apply(e,selected,gids);e.export('handoff_fixed');p=e.asset('handoff_fixed');np.savez_compressed(p/'a9_ownership_sidecar.npz',**side)
    binding=m.read(p/'scene_binding.json');binding.update(ownership_sidecar=str(p/'a9_ownership_sidecar.npz'),loader=str(m.ROOT/'code/fix_a9_handoff.py'),query='KEEP exactly unchanged; DELETE actor10 includes fixed unknown-row semantic ownership repair');m.write(p/'scene_binding.json',binding)
    checks=[]
    for raw in m.FRAMES:
        a=m.load(m.ROOT/'delivery/joint_fit'/f'{raw:03d}'/'render_float.npz');b=m.load(m.ROOT/'delivery/handoff_fixed'/f'{raw:03d}'/'render_float.npz');labels=m.cpu(e.targets[raw]['labels']).numpy();a9=labels==m.DELETE;outside_mask=~m.cpu(F.max_pool2d(m.ten(a9)[None,None],17,1,8)[0,0]>0).numpy();protected_mask=(labels>0)&(labels!=m.DELETE)
        checks.append(dict(raw=raw,KEEP_RGB_max=float(np.abs(a['joint_keep']-b['joint_keep']).max()),DELETE_outside_a9_dilated8_RGB_max=float(np.abs(a['joint_delete']-b['joint_delete'])[:,outside_mask].max()),DELETE_retained_vehicle_RGB_max=float(np.abs(a['joint_delete']-b['joint_delete'])[:,protected_mask].max()),road_RGB_max=float(np.abs(a['road_rgb']-b['road_rgb']).max()),road_alpha_max=float(np.abs(a['road_alpha']-b['road_alpha']).max()),DELETE_changed_pixels=int((np.abs(a['joint_delete']-b['joint_delete']).max(0)>1e-6).sum())))
    proof.update(checks=checks,passed=all(x['KEEP_RGB_max']==0 and x['DELETE_retained_vehicle_RGB_max']<1e-5 and x['DELETE_outside_a9_dilated8_RGB_max']<1e-5 and x['road_RGB_max']==0 and x['road_alpha_max']==0 for x in checks),native_raster_calls=e.native_calls,feature_vjp_calls=15)
    m.write(m.ROOT/'handoff_proof.json',proof);m.write(m.ROOT/'status.json',dict(state='handoff_fixed_complete',complete=True,stage='handoff_fixed'))
    print(json.dumps(proof,indent=2))
    if not proof['passed']:raise RuntimeError('handoff protection render gate failed')

if __name__=='__main__':main()
