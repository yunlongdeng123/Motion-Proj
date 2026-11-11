"""固定多帧道路表面的静态opacity交接；全帧遍历并独立检查KEEP目标。

沿用v44完整目标候选/保持域，只有静态无车辆归属候选的门可训练。
固定道路几何、颜色与来源，原生SH/运动/环境不改。未知不产生隐藏RGB。
"""
from __future__ import annotations
import argparse,hashlib,json,random,time
from pathlib import Path
import numpy as np
from . import v44_handoff_policy as policy


def _differences(a,b,path=''):
    import torch
    if torch.is_tensor(a):return [] if torch.is_tensor(b) and torch.equal(a.cpu(),b.cpu()) else [path]
    if isinstance(a,np.ndarray):return [] if np.array_equal(a,b) else [path]
    if isinstance(a,dict):
        if not isinstance(b,dict) or a.keys()!=b.keys():return [path]
        return [x for k in a for x in _differences(a[k],b[k],path+'/'+str(k))]
    if isinstance(a,(tuple,list)):
        if type(a)!=type(b) or len(a)!=len(b):return [path]
        return [x for i,(u,v) in enumerate(zip(a,b)) for x in _differences(u,v,path+'/'+str(i))]
    return [] if a==b else [path]


def fit(manifest,views,baseline,output,delete_tracks,keep_baseline,steps=1000,resume=None):
    import torch
    from PIL import Image
    from .v44_factory import from_manifest
    scene,m=from_manifest(manifest);deleted=m.actor_ids(delete_tracks)
    if not deleted or m.static_gates is not None:raise ValueError('Explicit deletion and an ungated scene are required')
    if m.data.get('static_material') is not None or m.data.get('raster_units') is not None:
        raise ValueError('This paired source-road control requires unchanged native raster and no material mixing')
    out=Path(output);out.mkdir(parents=True,exist_ok=False);views=Path(views);baseline=Path(baseline)
    view_manifest=json.loads((views/'manifest.json').read_text());entries=view_manifest['frames']
    view_by_key={r['key']:views/r['view'] for r in entries};base_by_key={r['key']:baseline/r['baseline'] for r in entries}
    if len(view_by_key)!=len(entries) or set(view_by_key)!={f.key for f in m.frames}:
        raise ValueError('Road-view BUILD keys must uniquely match the declared scene frames')
    def checked_ids(value,name):
        if (not isinstance(value,list) or any(type(i) is not int or i<=0 for i in value)
                or len(set(value))!=len(value) or set(value)!=set(deleted)):
            raise ValueError(name+' must match the complete requested delete actor set')
    view_binding=json.loads((views/'binding.json').read_text())
    checked_ids(view_binding.get('delete_actor_ids'),'Road-view binding')
    baseline_provenance=json.loads((baseline/'provenance.json').read_text())
    checked_ids(baseline_provenance.get('resolved_query',{}).get('delete_actor_ids'),'Baseline deletion identity')
    if Path(baseline_provenance['model']).resolve()!=m.model or baseline_provenance['iteration']!=m.iteration:
        raise ValueError('Baseline model/iteration does not match the frozen scene')
    baseline_rows={r['frame_key']:r for r in baseline_provenance['frames']}
    if len(baseline_rows)!=len(baseline_provenance['frames']) or set(baseline_rows)!=set(view_by_key):
        raise ValueError('Baseline BUILD frame keys must match the scene')
    for key,path in base_by_key.items():
        if path.resolve()!=(baseline/baseline_rows[key]['files']['float_npz']).resolve():
            raise ValueError('Baseline frame/file binding differs from declared export')
    def digest(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024**2),b''):h.update(chunk)
        return h.hexdigest()
    labels_by_key={};targets={};bindings=[]
    for frame,data in zip(m.frames,m.data['frames']):
        if 'instance_mask' not in data:raise ValueError('Handoff requires explicit accepted instance masks for every BUILD frame')
        path=Path(data['instance_mask']);path=path if path.is_absolute() else m.base/path
        cam=scene.camera_provider(frame.key)
        with Image.open(path) as image:labels=np.asarray(image.resize((cam.image_width,cam.image_height),Image.Resampling.NEAREST)).copy()
        target=policy.target_mask_from_labels(labels,deleted)
        present=set(np.unique(labels).tolist())-{0}
        if not present<=set(m.actor_to_track):raise ValueError('Instance labels contain unregistered actors')
        labels_by_key[frame.key]=labels;targets[frame.key]=target
        bindings.append(dict(key=frame.key,instance_mask=digest(path),view=digest(view_by_key[frame.key]),baseline=digest(base_by_key[frame.key])))
    source_binding=dict(scene_manifest=digest(manifest),views_manifest=digest(views/'manifest.json'),views_binding=digest(views/'binding.json'),baseline_provenance=digest(baseline/'provenance.json'),
        delete_actor_ids=list(deleted),algorithm_sha256=digest(__file__),policy_sha256=digest(policy.__file__),sources=bindings,
        policy='full_target_candidate_and_preservation_v1')
    keep_baseline=Path(keep_baseline);keep_by_key={f.key:keep_baseline/f'{f.key:03d}'/'keep.npz' for f in m.frames}
    source_binding['keep_native_baselines']=[dict(key=f.key,sha256=digest(keep_by_key[f.key])) for f in m.frames]
    source_binding['sampling']='shuffled_complete_BUILD_passes_seed0'
    torch.manual_seed(0);np.random.seed(0);random.seed(0);sampler=random.Random(0)
    N=len(scene.static.get_xyz);device=scene.static.get_xyz.device
    score=torch.zeros(N,device=device);fit_score=torch.zeros_like(score);guard=torch.zeros_like(score)
    frames=[];selection_frames=[];records=[];domain_records=[];started=time.monotonic();ops=scene._bindings()
    def domains(key,d):
        view={k:d[k] for k in ['fit','keep','inside','other','marking']}
        view['other']=policy.rebind_other_vehicles(view['other'],labels_by_key[key],deleted)
        result=policy.handoff_domains(view,targets[key]);result['other']=view['other']
        return result
    for frame in m.frames:
        key=frame.key
        with np.load(view_by_key[key],allow_pickle=False) as d:dom=domains(key,d)
        support=dom['candidate_support'];fit_mask=(dom['fit']>.05)&~dom['protected'];protect=dom['protected'];target=targets[key]
        if ((dom['fit']>0)&protect).any():
            raise ValueError('Photo references overlap preserved vehicles/markings; prepare matching BUILD views')
        # 无道路photo的帧仍有KEEP/外围保持监督；每一完整遍历覆盖全部BUILD。
        frames.append(key)
        domain_records.append(dict(key=key,target_pixels=int(target.sum()),candidate_pixels=int(support.sum()),
            target_candidate_pixels=int((target&support).sum()),protected_target_conflicts=int(dom['target_protected_conflict'].sum()),
            target_keep_pixels=int((target&(dom['keep']>0)).sum()),target_left_pixels=int((target&dom['left']).sum()),
            unknown_target_pixels=int(dom['unknown_target'].sum()),photo_pixels=int(fit_mask.sum()),
            photo_protected_overlap=int(((dom['fit']>.05)&protect).sum())))
        if not support.any() and not protect.any():continue
        with torch.no_grad():values,owners,cam,_=scene._rows(key,(),(),False)
        features=values['features'].detach().clone();features[:,8:11]=0;features[:N,8:11]=1;features.requires_grad_(True)
        mask=values['mask']&~torch.isin(owners,owners.new_tensor(deleted))
        pack=ops.raster(cam,scene.pipe,0,dict(values,features=features,mask=mask),features.new_zeros(3))
        # 三个独立 feature marker 共用同一排序、一次反向；不修改成像参数。
        objective=(pack['extra'][6][torch.as_tensor(support,device=device)].sum()
            +pack['extra'][7][torch.as_tensor(fit_mask,device=device)].sum()
            +pack['extra'][8][torch.as_tensor(protect,device=device)].sum())
        grad=torch.autograd.grad(objective,features)[0][:N,8:11]
        score=torch.maximum(score,grad[:,0]);fit_score=torch.maximum(fit_score,grad[:,1]);guard=torch.maximum(guard,grad[:,2]);selection_frames.append(key)
        del pack,values,features,grad,objective
    eligible_np=policy.eligible_static_contributors(score.detach().cpu().numpy(),guard.detach().cpu().numpy(),scene._owners_s.detach().cpu().numpy())
    eligible=torch.as_tensor(eligible_np,device=device);indices=torch.nonzero(eligible).flatten()
    old_eligible=(fit_score>1e-4)&~(guard>20*fit_score)&(scene._owners_s<=0)
    np.savez_compressed(out/'contributors.npz',ids=indices.cpu().numpy(),candidate_score=score[eligible].cpu().numpy(),
        fit_score=fit_score[eligible].cpu().numpy(),guard_score=guard[eligible].cpu().numpy())
    coverage=dict(policy=source_binding['policy'],rows=domain_records,selection_frames=selection_frames,optimization_frames=frames,
        candidate_rows=len(indices),reference_supported_candidate_rows=int((eligible&(fit_score>1e-4)).sum()),
        newly_eligible_vs_fit_only=int((eligible&~old_eligible).sum()),fit_only_eligible_rows=int(old_eligible.sum()),
        candidate_without_reference_rows=int((eligible&(fit_score<=1e-4)).sum()),
        initialization=dict(logit=2.,opacity_multiplier=float(torch.tensor(2.).sigmoid()),retention_prior='mean over current candidate rows; expanded denominator recorded'),
        reference_weights_unchanged=True,unknown_target_supervision_added=False)
    (out/'coverage.json').write_text(json.dumps(coverage,indent=2)+'\n')
    if not len(indices) or not frames:
        status='NO_ELIGIBLE_STATIC_OPACITY_ROWS' if not len(indices) else 'NO_REFERENCE_FOR_HANDOFF'
        (out/'receipt.json').write_text(json.dumps(dict(status=status,steps=0,updates=0,candidate_rows=len(indices),parent_unchanged=True)))
        scene.assert_persistent_unchanged();return
    gate=torch.nn.Parameter(torch.full((len(indices),),2.,device=device));opt=torch.optim.Adam([gate],lr=.06,eps=1e-15);startstep=0
    remaining_frames=[];sample_epoch=0
    def state(step):return dict(step=step,gate=gate.detach().clone(),contributors=indices.detach().clone(),
        remaining_frames=list(remaining_frames),sample_epoch=sample_epoch,
        fit_frames=frames,source_binding=source_binding,optimizer=opt.state_dict(),sampler=sampler.getstate(),
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate())
    if resume is not None:
        z=torch.load(resume,map_location=device,weights_only=False)
        if (not torch.equal(z['contributors'],indices) or z['fit_frames']!=frames or z['source_binding']!=source_binding):
            raise RuntimeError('Gate restore changes contributor identity, accepted instance masks, policy or source binding')
        gate.data.copy_(z['gate']);opt.load_state_dict(z['optimizer']);sampler.setstate(z['sampler']);torch.set_rng_state(z['torch_rng'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in z['cuda_rng']]);np.random.set_state(z['numpy_rng']);random.setstate(z['python_rng']);startstep=z['step']
        remaining_frames[:]=z['remaining_frames'];sample_epoch=z['sample_epoch']
        diff=_differences(z,state(startstep))
        if diff:raise RuntimeError('Gate full-state restore mismatch: '+str(diff))
        (out/'restore.json').write_text(json.dumps(dict(exact_full_state=True,differences=diff,step=startstep)))
    # 零更新初始化与真实保存单列，供识别扩展参数导致的初始 .880797 衰减。
    if startstep==0:torch.save(state(0),out/'checkpoint000000.pt')
    def linear(x):
        x=x.clamp(0,1);return torch.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
    for step in range(startstep+1,steps+1):
        if not remaining_frames:
            remaining_frames[:]=frames;sampler.shuffle(remaining_frames);sample_epoch+=1
        key=remaining_frames.pop()
        with np.load(view_by_key[key],allow_pickle=False) as d:
            dom=domains(key,d);reference=d['reference_linear'].copy();marking=d['marking'].copy()
        opt.zero_grad(set_to_none=True);values,owners,cam,_=scene._rows(key,(),(),False)
        opacity=values['opacities'].index_copy(0,indices,values['opacities'][indices]*gate.sigmoid()[:,None])
        mask=values['mask']&~torch.isin(owners,owners.new_tensor(deleted))
        pack=ops.raster(cam,scene.pipe,0,dict(values,opacities=opacity,mask=mask),opacity.new_zeros(3))
        rgb=pack['rgb']+(1-pack['alpha'])*scene.environment(cam.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
        pred=linear(rgb);weight=torch.as_tensor(dom['fit'],device=device);keep=torch.as_tensor(dom['keep'],device=device);target=torch.as_tensor(reference,device=device)
        with np.load(base_by_key[key],allow_pickle=False) as b:base=torch.as_tensor(b['rgb'].copy(),device=device)
        photo=((pred-target).abs()*weight).sum()/(3*weight.sum().clamp_min(1));preserve=((rgb-base).abs()*keep).sum()/(3*keep.sum().clamp_min(1))
        marking=torch.as_tensor(marking,device=device);other=torch.as_tensor(dom['other'],device=device);left=torch.as_tensor(dom['left'],device=device)
        lane=(rgb-base).abs()[:,marking].mean() if marking.any() else photo*0
        vehicle=(rgb-base).abs()[:,other].mean() if other.any() else photo*0
        left_keep=(rgb-base).abs()[:,left].mean() if left.any() else photo*0
        keep_pack=ops.raster(cam,scene.pipe,0,dict(values,opacities=opacity),opacity.new_zeros(3))
        keep_rgb=keep_pack['rgb']+(1-keep_pack['alpha'])*scene.environment(cam.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
        with np.load(keep_by_key[key],allow_pickle=False) as kb:keep_target=torch.as_tensor(kb['rgb'].copy(),device=device)
        target_pixels=torch.as_tensor(targets[key],device=device)
        keep_vehicle=(keep_rgb-keep_target).abs()[:,target_pixels].mean() if target_pixels.any() else photo*0
        loss=photo+2*preserve+4*lane+3*vehicle+2*left_keep+2*keep_vehicle+.0005*(1-gate.sigmoid()).mean()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite gate loss')
        loss.backward()
        if not torch.isfinite(gate.grad).all():raise RuntimeError('Nonfinite gate gradient')
        opt.step();record=dict(step=step,key=key,loss=float(loss),road=float(photo),preserve=float(preserve),vehicle=float(vehicle),keep_target=float(keep_vehicle),seconds=time.monotonic()-started);records.append(record)
        if step%100==0 or step==startstep+1:print(json.dumps(record),flush=True)
        if step in [32,500,1000] or step==steps:torch.save(state(step),out/f'checkpoint{step:06d}.pt')
    scene.assert_persistent_unchanged();(out/'training.json').write_text(json.dumps(records))
    (out/'receipt.json').write_text(json.dumps(dict(status='COMPLETED_FIXED_ROAD_GATE_FIT_PENDING_VISUAL_REVIEW',steps=steps,updates=steps-startstep,selected_static_rows=len(indices),
        fit_frames=frames,selection_frames=selection_frames,policy=source_binding['policy'],parent_unchanged=True,only_gate_trainable=True,
        road_colour_geometry_fixed=True,delete_tracks=delete_tracks,seconds=time.monotonic()-started,
        keep_target_weight=2.,sampling='seed0 shuffled complete passes',native_raster_near_unchanged=True,
        material_mixing=False,metric_depth_issue_known=True),indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['manifest','views','baseline','output']:p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--delete-track-id',action='append',required=True);p.add_argument('--steps',type=int,default=1000);p.add_argument('--resume',type=Path)
    p.add_argument('--keep-baseline',required=True,type=Path)
    a=p.parse_args();fit(a.manifest,a.views,a.baseline,a.output,a.delete_track_id,a.keep_baseline,a.steps,a.resume)

if __name__=='__main__':main()
