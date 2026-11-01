"""五例真实原生GPU贡献核查及固定共享路由。原参数/来源文件只读。"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import argparse, gc, json, sys, time, shutil, filecmp
from copy import deepcopy
from pathlib import Path
import numpy as np
from PIL import Image
import torch

REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
CODE=REPO/'scripts/worldsim_v82/v43_ownership_20261028/v43_decision_r4'
sys.path.insert(0,str(CODE))
import native_shared_route as old
from r4_core import canonical_hash, enumerate_routes, freeze_family, register_tasks
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/r4_fivecase_gpu_20261101_r1')
read=old.read
write=old.write

class FiveScene(old.NativeScene):
    def __init__(self,*args,**kwargs):
        # 仅复用本任务已验证的私有只读文件，不将原始输入硬链接进可写loader。
        from types import SimpleNamespace
        cpu=Path(args[0]);scene_key=read(cpu/'camera_track_context.json')['scene_key']
        caches={'waymo-train-008':ROOT/'gpu/waymo-train-008_a37/audit49/restore/scene_loader/query_scene',
                'waymo-train-212':ROOT/'gpu_r2/waymo-train-212_a0/audit49/restore/scene_loader/query_scene',
                'waymo-train-564':ROOT/'gpu_564_replay/waymo-train-564_a0/audit49/restore/scene_loader/query_scene'}
        cache=caches[scene_key];linked=[0,0]
        def copytree(source,destination,**options):
            source=Path(source)
            def copyfile(src,dst):
                relative=Path(src).relative_to(source);prior=cache/relative
                if Path(src).name!='transform_matrix.txt' and prior.is_file() and filecmp.cmp(src,prior,shallow=False):
                    os.link(prior,dst);linked[0]+=1;linked[1]+=prior.stat().st_size;return dst
                return shutil.copy2(src,dst)
            return shutil.copytree(source,destination,copy_function=copyfile,**options)
        previous_shutil=old.shutil
        old.shutil=SimpleNamespace(copytree=copytree,copy2=shutil.copy2)
        try:super().__init__(*args,**kwargs)
        finally:old.shutil=previous_shutil
        receipt=self.output/'native_restore_receipt.json';details=read(receipt)
        if linked[0]:
            details['source_loader_isolation'].update(all_input_files_independently_copied=False,
                original_source_files_never_hardlinked=True,private_cache=str(cache),
                byte_verified_private_cache_hardlinks=linked[0],private_cache_linked_bytes=linked[1],
                mutable_transform_always_independent_copy=True)
            write(receipt,details)
        native_raster=self.helpers.raster
        def raster(camera,pipe,degree,values,background):
            return native_raster(camera,pipe,degree,{k:v for k,v in values.items() if k!='_original_row'},background)
        self.helpers.raster=raster
        binding=self.cpu/'dynamic_binding.json'
        self.source_versions['target_dynamic_binding']=old.file_hash(binding) if binding.exists() else None

    def frame_rows(self,frame):
        camera,s,d,e=super().frame_rows(frame)
        s['_original_row']=torch.arange(len(s['means3D']),device='cuda')
        d['_original_row']=torch.arange(len(d['means3D']),device='cuda')+len(s['means3D'])
        return camera,s,d,e

    def execute_query(self,frame,static,moving,edit):
        values=super().execute_query(frame,static,moving,edit)
        # 保留原始行顺序，避免拆分动态层引入同深度排序的非研究差异。
        return old.take_rows(values,torch.argsort(values['_original_row']))

    def check_actor_source(self):
        preparation=read(self.scene_root/'dynamic_multi_actor_preparation.json')
        idx=self.context['target_dynamic_actor_index']
        if idx is not None:
            target=[r for r in preparation['actors'] if r['actor_index']==idx and r['raw_track_id']==self.context['target_track']]
            if len(target)!=1:raise ValueError('来源track与目标关联不一致')
        self.target_mode='single_native_actor' if idx is not None and len(preparation['actors'])==1 else 'unbound' if idx is None else 'multi_actor_legal_proxy'
        binding=self.cpu/'dynamic_binding.json'
        self.target_dynamic_ids=read(binding)['target_dynamic_rows'] if binding.exists() else None
        return dict(target_track=self.context['target_track'],target_actor_index=idx,native_actor_count=len(preparation['actors']),
            target_mode=self.target_mode,multi_actor_exact_row_owner_available=False if len(preparation['actors'])>1 else None,
            binding_rule='full49 legal native contribution dominance; provisional association, not physical owner GT' if len(preparation['actors'])>1 else self.target_mode)

    def apply_shared_route(self,route,frame,static,dynamic):
        n=len(dynamic['means3D'])
        if self.target_mode=='single_native_actor':ids=torch.ones(n,dtype=torch.bool,device='cuda')
        elif self.target_mode=='unbound':ids=torch.zeros(n,dtype=torch.bool,device='cuda')
        else:
            if self.target_dynamic_ids is None:raise ValueError('多车层必须先冻结逐行关联，不能整层编辑')
            ids=torch.zeros(n,dtype=torch.bool,device='cuda');ids[self.target_dynamic_ids]=True
        target=old.take_rows(dynamic,ids);others=old.take_rows(dynamic,~ids)
        s,m,c=super().apply_shared_route(route,frame,static,target)
        s=self.helpers.merge_rows(s,others)
        c.update(native_dynamic_total_rows=n,other_dynamic_rows_kept=int((~ids).sum()),target_dynamic_rows=int(ids.sum()),
            preedit_total_rows=len(static['means3D'])+n,kept_context_rows=len(s['means3D']),
            target_binding_mode=self.target_mode)
        return s,m,c

def body_mask(native,frame):
    p=Path(native.track[frame]['mask_ref']);cam=native.cameras[frame]
    a=np.asarray(Image.open(p).convert('L').resize((cam.image_width,cam.image_height),Image.Resampling.NEAREST))>0
    if not a.any():raise ValueError('目标mask空')
    return a,p

def audit(args):
    out=args.output;out.mkdir(exist_ok=False,parents=True)
    shutil.copy2(__file__,out/'executable_snapshot.py')
    n=FiveScene(args.cpu,out/'restore')
    S=n.candidate['static_row_count'];D=len(n.dynamic._xyz)
    sums=np.zeros((2,S+D),np.float64);records=[];start=time.monotonic()
    with torch.no_grad():
        for i,frame in enumerate(n.frames):
            cam,s,d,env=n.frame_rows(frame)
            mask,ref=body_mask(n,frame);body=torch.as_tensor(mask,device='cuda')
            combined=n.helpers.merge_rows(s,d)
            with torch.enable_grad():
                feat=combined['features'].detach().clone().requires_grad_(True)
                result=n.helpers.raster(cam,n.pipe,n.static.active_sh_degree,dict(combined,features=feat),n.base_background)
                lb=(result['alpha'][0].double()*body).sum();lk=(result['alpha'][0].double()*(~body)).sum()
                gb=torch.autograd.grad(lb,feat,retain_graph=True)[0][:,1].detach()
                gk=torch.autograd.grad(lk,feat)[0][:,1].detach()
            bs=gb.cpu().numpy().astype(np.float64);ks=gk.cpu().numpy().astype(np.float64)
            if not np.isfinite(bs).all() or not np.isfinite(ks).all() or bs.min()<-1e-6 or ks.min()<-1e-6:raise ValueError('非有限/负alphaT')
            eb=abs(bs.sum()-float(lb));ek=abs(ks.sum()-float(lk))
            # Native前向/原子伴随均为FP32；跨全图总量用FP64归约，门禁同时记录绝对与相对误差。
            limits=[max(.05,2e-6*abs(float(lb))),max(.05,2e-6*abs(float(lk)))]
            if eb>limits[0] or ek>limits[1]:
                write(out/'conservation_failure.json',dict(frame=frame,body_error=eb,keep_error=ek,limits=limits))
                # 严格门禁仍记FAIL，不调整容差冒充通过；原生前向/恢复实验可以独立继续。
                # 这些伴随值仅用于候选域诊断，不能据此宣称精确贡献或正式P资格。
            sums[0]+=bs;sums[1]+=ks
            arrays,error=n.native_baseline(frame,s,d,env)
            observed=cam.original_image[:3].detach().cpu().numpy().astype(np.float32)
            old.save_rgb(out/(frame+'_observed_RGB.png'),observed)
            # 原始完整none的49帧保存为FP32，独立于候选和编辑结果。
            old.save_arrays(out/'original_none',frame,arrays)
            rec=dict(frame=frame,raw=n.track[frame]['raw'],body_pixels=int(mask.sum()),keep_pixels=int((~mask).sum()),
                body_alphaT=float(bs.sum()),keep_alphaT=float(ks.sum()),body_conservation_abs=eb,keep_conservation_abs=ek,
                conservation_limits=limits,conservation_scalar_reduction='FP64; native raster and adjoints FP32',
                strict_contribution_mass_gate_passed=bool(eb<=limits[0] and ek<=limits[1]),
                native_none_parity_max_abs=error,static_body_alphaT=float(bs[:S].sum()),dynamic_body_alphaT=float(bs[S:].sum()),
                observed_body_rgb_MAE=float(abs(arrays['final'][:,mask]-observed[:,mask]).mean()),
                observed_outside_rgb_MAE=float(abs(arrays['final'][:,~mask]-observed[:,~mask]).mean()),
                mask_ref=str(ref),f_geom_qualification=None,depth_consistency='not certified by total alpha or expected depth')
            if frame in n.anchors:
                np.savez_compressed(out/(frame+'_row_alphaT.npz'),body_alphaT=bs,keep_alphaT=ks,static_row_denominator=np.asarray(S))
            records.append(rec)
            if i%8==0:print(json.dumps(dict(case=n.scene_id,stage='legal49_contribution',frames=i+1)),flush=True)
            del feat,result,gb,gk,combined,s,d,arrays;gc.collect()
    total=sums.sum(0);ratio=np.divide(sums[0],total,out=np.zeros_like(total),where=total>0)
    if n.target_mode=='multi_actor_legal_proxy':
        target=np.flatnonzero((total[S:]>1e-4)&(ratio[S:]>=.95)).tolist()
        if not target:raise ValueError('多车目标关联无可用支持，不制造目标')
    elif n.target_mode=='single_native_actor':target=list(range(D))
    else:target=[]
    binding=dict(target_dynamic_rows=target,total_native_dynamic_rows=D,mode=n.target_mode,
        definition='full49frames original unified native alphaT inside fixed target mask / whole image >=.95 and total>1e-4' if n.target_mode=='multi_actor_legal_proxy' else n.target_mode,
        physical_owner_GT=False,ambiguous_rows_kept_unchanged=True,other_actors_never_whole_layer_edited=True)
    write(args.cpu/'dynamic_binding.json',binding)
    candidate=deepcopy(n.candidate);counts=[];groups=[]
    for g in candidate['groups']:
        prior=g['original_static_rows'];ids=[x for x in prior if total[x]>1e-4 and ratio[x]>=.95]
        counts.append(dict(group=g['group_id'],parent=len(prior),refined=len(ids)))
        if ids:g['original_static_rows']=ids;g['selection_ref']='full49_none_legal_body_alphaT_ratio095';groups.append(g)
    candidate['groups']=groups
    candidate['parent_candidate_hash']=canonical_hash(n.candidate)
    candidate['legal_refinement']=dict(body_ratio=.95,total_alphaT_strict=1e-4,frames=len(n.frames),
        role='development candidate domain; not calibrated admission or identity probability',no_edited_reference=True)
    refined=args.cpu.parent.parent/'cpu_refined';refined.mkdir(exist_ok=True)
    if not (refined/'run_contract.json').exists():write(refined/'run_contract.json',read(args.cpu.parent/'run_contract.json'))
    scene=refined/n.scene_id;scene.mkdir(exist_ok=False)
    evidence=read(args.cpu/'legal_preedit_evidence.json');routes=enumerate_routes(candidate)
    for name in ['camera_track_context.json','checkpoint_cpu_receipt.json','dynamic_binding.json']:
        write(scene/name,read(args.cpu/name))
    for name,value in [('candidate_manifest.json',candidate),('shared_routes.json',routes),('frozen_H0.json',freeze_family(candidate)),
                       ('legal_preedit_evidence.json',evidence),('tasks.json',register_tasks(n.scene_id,candidate,evidence,routes,candidate['checkpoint_static_sha256'],n.frozen))]:write(scene/name,value)
    np.savez_compressed(out/'full49_row_responsibility.npz',body_alphaT=sums[0],keep_alphaT=sums[1],body_ratio=ratio,static_row_denominator=np.asarray(S))
    write(out/'audit.json',dict(case=n.scene_id,records=records,frames=len(records),
        group_counts=counts,parent_candidate_rows=freeze_family(n.candidate)['candidate_row_count'],refined_candidate_rows=freeze_family(candidate)['candidate_row_count'],
        strict_contribution_mass_all_frames_passed=all(x['strict_contribution_mass_gate_passed'] for x in records),
        contribution_qualification='diagnostic FP32 adjoints; strict mass failures retained, not an exact owner/coverage certificate',
        target_dynamic_binding=binding,refined_cpu=str(scene),original_unchanged=n.unchanged(),
        native_query_calls=len(records)*2,contribution_forward_calls=len(records),feature_adjoint_backward_calls=len(records)*2,
        peak_allocated=torch.cuda.max_memory_allocated(),wall_seconds=time.monotonic()-start,formal_eligibility='unknown',training_updates=0))

def render(args):
    args.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    shutil.copy2(__file__,args.output/'executable_snapshot.py')
    n=FiveScene(args.cpu,args.output/'restore')
    ids=list(dict.fromkeys(['h000',n.allrebind_id]))
    if args.phase=='smoke':
        rows,base,parity,wall=old.render_rows(n,args.output,[n.anchors[1]],ids)
        m=old.manifest(n,rows,base,parity,wall,'smoke');write(args.output/'gpu_manifest.json',m)
        write(args.output/'saved_state_receipt.json',n.save_state(args.output/'inference_state.pt',n.allrebind_id,args.output/'gpu_manifest.json'))
    elif args.phase in ['enumerate','sequence']:
        saved=n.restore_state(args.resume);prior=read(saved['smoke_manifest'])
        rows,_,_,_=old.render_rows(n,args.output/'replay',[n.anchors[1]],ids)
        errors=[]
        for row in rows:
            original=next(x for x in prior['rows'] if all(x[k]==row[k] for k in ['action_id','frame','edit']))
            with np.load(original['fp32']) as a,np.load(row['fp32']) as b:
                error=max(float(np.max(abs(a[k]-b[k]))) for k in a.files)
            errors.append(error)
        if max(errors)>2e-6:raise ValueError('独立进程完整恢复重放失败')
        write(args.output/'restore_replay.json',dict(max_abs=max(errors),calls=len(rows),passed=True))
        frames=n.anchors if args.phase=='enumerate' else n.frames
        routes=['h%03d'%i for i in range(len(n.routes))] if args.phase=='enumerate' else ids
        rows,base,parity,wall=old.render_rows(n,args.output,frames,routes)
        m=old.manifest(n,rows,base,parity,wall,args.phase)
        m['target_selection']='registered per-case target; unbound targets edit only rebound rows; multi-actor target uses frozen legal contribution proxy'
        m['evaluated_frames']=frames;m['unmeasured_frames']=len(n.frames)-len(frames)
        write(args.output/'gpu_manifest.json',m)
    write(args.output/'execution.json',dict(status='COMPLETE',phase=args.phase,wall_seconds=time.monotonic()-start,
        cuda_max_allocated=torch.cuda.max_memory_allocated(),original_unchanged=n.unchanged(),training_updates=0))

def peel(args):
    out=args.output;out.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,out/'executable_snapshot.py')
    write(out/'pre_feature_contract.json',dict(feature='pre-registered target-source peel; shared BUILD feature independent of requested q',
        original_parent_U_and_anchor_groups_unchanged=True,source_condition='remove frozen target native dynamic rows only; all other rows and environment fixed',
        legal_mask='same fixed DEVA source masks across49 frames',candidate_rule='union of original-none and peeled-none total>1e-4 and body_ratio>=.95',
        thresholds='same development candidate scope; not formal P or owner probability',not_independent_background_GT=True,
        reason='visible-only candidate domain is empty for212/a0 and564/a0; complete R4 exposed-source control',human_verdict=None))
    n=FiveScene(args.cpu,out/'restore');S=n.candidate['static_row_count'];D=len(n.dynamic._xyz)
    totals=np.zeros((2,S),np.float64);records=[];start=time.monotonic()
    with torch.no_grad():
        for i,frame in enumerate(n.frames):
            cam,s,d,env=n.frame_rows(frame);mask,ref=body_mask(n,frame);body=torch.as_tensor(mask,device='cuda')
            if n.target_mode=='single_native_actor':keep=torch.zeros(D,dtype=torch.bool,device='cuda')
            elif n.target_mode=='unbound':keep=torch.ones(D,dtype=torch.bool,device='cuda')
            else:
                keep=torch.ones(D,dtype=torch.bool,device='cuda');keep[n.target_dynamic_ids]=False
            values=n.helpers.merge_rows(s,old.take_rows(d,keep))
            with torch.enable_grad():
                feat=values['features'].detach().clone().requires_grad_(True)
                pkg=n.helpers.raster(cam,n.pipe,n.static.active_sh_degree,dict(values,features=feat),n.base_background)
                lb=(pkg['alpha'][0].double()*body).sum();lk=(pkg['alpha'][0].double()*(~body)).sum()
                gb=torch.autograd.grad(lb,feat,retain_graph=True)[0][:,1].detach().cpu().numpy().astype(np.float64)
                gk=torch.autograd.grad(lk,feat)[0][:,1].detach().cpu().numpy().astype(np.float64)
            if not np.isfinite(gb).all() or not np.isfinite(gk).all() or gb.min()<-1e-6 or gk.min()<-1e-6:raise ValueError('剥离伴随非有限/负值')
            totals[0]+=gb[:S];totals[1]+=gk[:S]
            eb=abs(gb.sum()-float(lb));ek=abs(gk.sum()-float(lk))
            records.append(dict(frame=frame,target_dynamic_rows_removed=int((~keep).sum()),body_alphaT=float(gb.sum()),
                keep_alphaT=float(gk.sum()),body_mass_difference=eb,keep_mass_difference=ek,
                strict_mass_gate=bool(eb<=max(.05,2e-6*float(lb)) and ek<=max(.05,2e-6*float(lk))),
                source_only_not_background_truth=True))
            if frame in n.anchors:
                arrays=n.raster(frame,values,env);old.save_arrays(out/'peeled_source_diagnostic',frame,arrays)
                np.savez_compressed(out/(frame+'_static_alphaT.npz'),body=gb[:S],keep=gk[:S])
            if i%12==0:print(json.dumps(dict(case=n.scene_id,stage='predeclared_source_peel',frames=i+1)),flush=True)
            del feat,pkg,gb,gk,values,s,d;gc.collect()
    total=totals.sum(0);ratio=np.divide(totals[0],total,out=np.zeros(S),where=total>0)
    visible=read(ROOT/'cpu_refined'/n.scene_id/'candidate_manifest.json')
    selected=np.zeros(S,bool)
    for g in visible['groups']:selected[g['original_static_rows']]=True
    selected|=(total>1e-4)&(ratio>=.95)
    candidate=deepcopy(n.candidate);groups=[]
    for g in candidate['groups']:
        ids=[r for r in g['original_static_rows'] if selected[r]]
        if ids:g['original_static_rows']=ids;g['selection_ref']='original_or_target_source_peel_alphaT_body_ratio095';groups.append(g)
    candidate['groups']=groups;candidate['source_feature_contract']=str(out/'pre_feature_contract.json')
    root=ROOT/'cpu_exposed';root.mkdir(exist_ok=True)
    # 两个并行进程不竞争同一临时合同文件。
    contract=root/'run_contract.json'
    if not contract.exists():
        tmp=root/('run_contract_'+n.scene_id+'.json');write(tmp,read(args.cpu.parent/'run_contract.json'));os.replace(tmp,contract)
    dst=root/n.scene_id;dst.mkdir(exist_ok=False)
    for name in ['camera_track_context.json','checkpoint_cpu_receipt.json','dynamic_binding.json']:shutil.copy2(args.cpu/name,dst/name)
    evidence=read(args.cpu/'legal_preedit_evidence.json');routes=enumerate_routes(candidate)
    for name,value in [('candidate_manifest.json',candidate),('shared_routes.json',routes),('frozen_H0.json',freeze_family(candidate)),
        ('legal_preedit_evidence.json',evidence),('tasks.json',register_tasks(n.scene_id,candidate,evidence,routes,candidate['checkpoint_static_sha256'],n.frozen))]:write(dst/name,value)
    np.savez_compressed(out/'static_source_peel_responsibility.npz',body=totals[0],keep=totals[1],ratio=ratio)
    write(out/'summary.json',dict(case=n.scene_id,frames=49,records=records,
        parent_candidate_rows=freeze_family(n.candidate)['candidate_row_count'],visible_candidate_rows=freeze_family(visible)['candidate_row_count'],
        exposed_candidate_rows=freeze_family(candidate)['candidate_row_count'],routes=len(routes),
        forward_calls=49+len(n.anchors),adjoint_backward_calls=98,original_unchanged=n.unchanged(),
        wall_seconds=time.monotonic()-start,peak_allocated=torch.cuda.max_memory_allocated(),training_updates=0,
        formal_eligibility=None,owner_GT=False,independent_reference_used=False))

def main():
    p=argparse.ArgumentParser();p.add_argument('--cpu',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=['audit','peel','smoke','enumerate','sequence'],required=True);p.add_argument('--resume',type=Path)
    a=p.parse_args();torch.cuda.reset_peak_memory_stats()
    if a.phase=='audit':audit(a)
    elif a.phase=='peel':peel(a)
    else:render(a)
if __name__=='__main__':main()
