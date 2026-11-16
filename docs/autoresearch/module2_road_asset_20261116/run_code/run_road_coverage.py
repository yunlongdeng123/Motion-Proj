"""MODULE2 Phase0/1真实相机覆盖训练，五帧，正式预算固定600。

无静态退役、无材质训练、无新diffusion。smoke/replay单独记账。
"""
from pathlib import Path
from dataclasses import asdict
import argparse,copy,gc,hashlib,json,os,shutil,sys,time
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='2'
import numpy as np
from PIL import Image
import torch
REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
CAMPAIGN=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_asset_20261116_r1')
sys.path.insert(0,str(REPO/'scripts/worldsim_v82'))
from drivingeditor_dynamic_edit.persistent_road_handoff import load_persistent_handoff
from drivingeditor_dynamic_edit.road_coverage_training import (
    CoverageContract,CoverageTrainer,opacity_scope,coverage_domains,alpha_coverage_metrics,_tensor_difference,tangent_scale_scope)
sys.path.insert(0,str(CAMPAIGN/'code'))
from road_asset_pipeline import save_scene

KEYS=[86,110,134,179,198]
VIEW=Path('/root/autodl-tmp/drivingeditor_repro/campaign/waymo_val065_road_completion_20261111_r1/multiview008_r2/world_road/views')


def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.writing')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False));tmp.replace(path)
def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save_rgb(path,value):
    Image.fromarray(np.uint8(np.clip(value.transpose(1,2,0),0,1)*255+.5)).save(path)


class Environment:
    def __init__(self,args):
        self.args=args;torch.set_num_threads(2)
        root_contract=read(CAMPAIGN/'training_contract.json')
        if root_contract['frames']!=KEYS or root_contract['phase1']['formal_steps']!=600 or root_contract['training_seed']!=101:
            raise ValueError('外层用户合同与五帧/600/seed101不一致')
        self.asset=args.asset.resolve();self.road=load(self.asset/'road.npz');self.prov=load(self.asset/'provenance.npz')
        if args.supervision and (args.supervision/'freeze.json').exists():
            frozen=read(args.supervision/'freeze.json')
            if Path(frozen['source_asset']).resolve()!=self.asset:raise ValueError('冻结监督属于另一bootstrap版本')
            for record in frozen['source_files']+frozen['target_files']:
                if digest(record['path'])!=record['sha256']:raise ValueError('冻结监督或bootstrap文件已改变: '+record['path'])
        self.original_guard=np.asarray(self.prov.get('guarded_world_rows',np.zeros(len(self.road['active']),bool)),bool).copy()
        if args.extra_guard:
            extra=load(args.extra_guard)['guard']
            if extra.shape!=self.original_guard.shape or extra.dtype!=np.bool_:raise ValueError('最大状态K保护拓扑不一致')
            self.prov['guarded_world_rows']=self.original_guard|extra
            self.prov['phase1_maximum_state_added_guard']=extra&~self.original_guard
        self.contract=CoverageContract(train_tangent_scales=args.bounded_scale)
        self.scene,self.manifest,self.receipt=load_persistent_handoff(self.asset/'persistent/scene.json',self.asset/'persistent/proof.json')
        if Path(self.manifest.road).resolve()!=self.asset/'road.npz':raise ValueError('proof加载的道路不是声明的bootstrap')
        for model in [self.scene.static,self.scene.dynamic,self.scene.environment]:
            if model is not None and hasattr(model,'parameters'):
                for p in model.parameters():p.requires_grad_(False)
        providers=[p for p in self.scene._providers if p.kind=='road']
        if len(providers)!=1 or len(self.scene._providers)!=1:raise ValueError('此覆盖入口要求单一持久road provider')
        self.provider=providers[0]
        self.eligible,self.scope=opacity_scope(self.road,self.prov)
        self.active_world=np.flatnonzero(self.road['active'])
        self.active_train=np.flatnonzero(self.eligible[self.active_world])
        self.world_train=self.active_world[self.active_train]
        self.scale_eligible=np.zeros_like(self.eligible);self.scale_upper=np.ones(len(self.eligible));self.scale_distance=np.zeros(len(self.eligible));self.scale_scope=None
        if args.bounded_scale:
            self.scale_eligible,self.scale_upper,self.scale_distance,self.scale_scope=tangent_scale_scope(self.road,self.prov,self.eligible)
            self.scope.update(scale_scope=self.scale_scope,trained_parameters=['road_opacity_logits','bounded_nontrusted_road_tangent_scale_factor'])
            self.scope['frozen']=[name for name in self.scope['frozen'] if name!='road_tangent_sigma']+['trusted_guard_or_outside_tangent_sigma']
        self.active_scale=np.flatnonzero(self.scale_eligible[self.active_world]);self.world_scale=self.active_world[self.active_scale]
        self.cameras={};self.rows={};self.domains={};self.numpy_domains={};self.baseline={};self.joint_baseline={};self.inputs=[]
        for key in KEYS:
            camera=self.scene.camera_provider(key);self.cameras[key]=camera
            frame=next(f for f in self.manifest.data['frames'] if f['key']==key)
            labelpath=Path(frame['instance_mask']);labelpath=labelpath if labelpath.is_absolute() else self.manifest.base/labelpath
            with Image.open(labelpath) as im:labels=np.asarray(im.resize((camera.image_width,camera.image_height),Image.Resampling.NEAREST)).copy()
            view=load(args.road_views/f'{key:03d}.npz')
            dom=coverage_domains(view,labels,self.prov,eligible_world=self.eligible)
            pixel_index=np.asarray(view['index']);valid_index=np.ones_like(pixel_index,bool)
            if args.supervision:
                supervision=load(args.supervision/f'{key:03d}.npz')
                if supervision['legal'].shape!=dom['fit'].shape:raise ValueError('监督相机尺寸不一致')
                # 新监督来自冻结世界图集；不会用当前alpha删掉无覆盖像素。
                if 'frozen_evaluation_domain' in supervision and not np.array_equal(dom['evaluation'],supervision['frozen_evaluation_domain']):
                    raise ValueError('监督冻结评价分母与旧固定域不一致')
                if 'world_row_index' in supervision:
                    ids=supervision['world_row_index'];valid_index=(ids>=0)&(ids<len(self.eligible));pixel_index=np.clip(ids,0,len(self.eligible)-1)
                    pixel_eligible=valid_index&self.eligible[pixel_index]
                    legal_color=supervision['legal']&~supervision['unknown']
                    dom['fit']=dom['evaluation']&legal_color&pixel_eligible
                    dom['legal_but_frozen']=dom['evaluation']&legal_color&~pixel_eligible
                else:dom['fit'] &= supervision['legal'].astype(bool)
                dom['geometry_unsupported']=dom['evaluation']&~supervision['geometry_support']
                dom['surface_intersection_failed']=dom['evaluation']&~supervision.get('surface_intersection_valid',supervision['geometry_support'])
                dom['material_unknown']=dom['evaluation']&supervision.get('surface_intersection_valid',supervision['geometry_support'])&supervision['unknown']
                dom['frozen_world_guard']=dom['evaluation']&supervision['retained_world_guard']
                dom['outside']=~dom['fit']&~dom['retained']
            self.numpy_domains[key]=dom
            index=pixel_index;model_view=camera.world_view_transform.detach().cpu().numpy()
            depth=self.road['model_xyz']@model_view[:3,2]+model_view[3,2]
            physical_near=.2*float(self.road['meter_to_model'])
            minimum=np.maximum(np.asarray(self.road.get('minimum_model_depth',np.zeros(len(depth)))),physical_near)
            dom['inactive_center']=dom['evaluation']&valid_index&~self.road['active'][index].astype(bool)
            dom['near_clipped']=dom['evaluation']&valid_index&(depth[index]<=minimum[index])
            self.domains[key]={k:torch.as_tensor(v.copy(),device='cuda') for k,v in dom.items()}
            with torch.no_grad():
                rows=self.provider.rows(key,camera,self.scene.pipe)
                self.rows[key]={k:v.detach() if torch.is_tensor(v) else v for k,v in rows.items()}
                baseline=self.scene._bindings().raster(camera,self.scene.pipe,0,self.rows[key],self.scene._background)
                self.baseline[key]=baseline['alpha'][0].detach()
            self.inputs.append(dict(raw=key,instance_mask_sha256=digest(labelpath),view_sha256=digest(args.road_views/f'{key:03d}.npz'),
                supervision_sha256=digest(args.supervision/f'{key:03d}.npz') if args.supervision else None))
        self.base_opacity=self.rows[KEYS[0]]['opacities'][:,0].detach().clone()
        self.base_scales=self.rows[KEYS[0]]['scales']
        if args.bounded_scale and self.base_scales is None:raise ValueError('此有界scale入口需显式native scales，不接受隐藏covariance修改')
        if any(not torch.equal(rows['opacities'][:,0],self.base_opacity) for rows in self.rows.values()):raise ValueError('道路opacity随帧变化')
        self.binding=dict(source_asset=str(self.asset),source_proof=self.receipt['proof_sha256'],source_road=self.receipt['road_sha256'],
            provenance_sha256=digest(self.asset/'provenance.npz'),frames=KEYS,inputs=self.inputs,opacity_world_ids_sha256=hashlib.sha256(self.world_train.tobytes()).hexdigest(),
            runner_sha256=digest(__file__),algorithm_sha256=digest(REPO/'scripts/worldsim_v82/drivingeditor_dynamic_edit/road_coverage_training.py'),
            parent_training_contract_sha256=digest(CAMPAIGN/'training_contract.json'),
            maximum_state_K_guard_sha256=digest(args.extra_guard) if args.extra_guard else None,
            supervision_freeze_sha256=digest(args.supervision/'freeze.json') if args.supervision and (args.supervision/'freeze.json').exists() else None,
            renderer_units='same metric raster and original cameras as pinned persistent scene',
            geometry_prior='pre-existing all199 BUILD LiDAR-only surface/plane observations; no held-out claim; fit and gradients limited to pilot5',
            world_material='reference-exposure sRGB from immutable atlas; not an illumination-separated reflectance')

    def trainer(self,role='formal'):
        return CoverageTrainer(self.base_opacity,self.active_train,KEYS,self.render_alpha,self.domains,self.baseline,self.binding,self.contract,
            base_scales=self.base_scales,active_scale_ids=self.active_scale,scale_upper=self.scale_upper[self.world_scale],accounting_role=role)

    def render_road(self,key,opacity,scales=None):
        v=dict(self.rows[key],opacities=opacity[:,None])
        if scales is not None:v=dict(v,scales=scales)
        return self.scene._bindings().raster(self.cameras[key],self.scene.pipe,0,v,self.scene._background)

    def render_alpha(self,key,opacity,scales=None):return self.render_road(key,opacity,scales)['alpha'][0]

    def joint(self,key,opacity,scales=None):
        with torch.no_grad():
            v,owners,cam,_=self.scene._rows(key,(10,),(),False)
            prefix=len(self.scene.static.get_xyz)+(len(self.scene.dynamic.get_xyz) if self.scene._include_dynamic else 0)
            if len(v['opacities'])-prefix!=len(opacity):raise ValueError('joint道路拓扑改变')
            opa=torch.cat([v['opacities'][:prefix],opacity[:,None]],0)
            if scales is not None:v=dict(v,scales=torch.cat([v['scales'][:prefix],scales],0))
            out=self.scene._bindings().raster(cam,self.scene.pipe,0,dict(v,opacities=opa,mask=v['mask']&~self.scene._contains(owners,(10,))),self.scene._background)
            env=(1-out['alpha'])*self.scene.environment(cam.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
            return (out['rgb']+env).cpu().numpy()

    def evaluate(self,trainer,dest,maximum=None):
        dest=Path(dest);dest.mkdir(parents=True,exist_ok=False);records=[]
        with torch.no_grad():
            opacity=trainer.opacity()
            scales=trainer.scales()
            for key in KEYS:
                pack=self.render_road(key,opacity,scales);alpha=pack['alpha'][0].cpu().numpy();base=self.baseline[key].cpu().numpy();dom=self.numpy_domains[key]
                current=self.joint(key,opacity,scales)
                if key not in self.joint_baseline:self.joint_baseline[key]=self.joint(key,self.base_opacity)
                old=self.joint_baseline[key];K=dom['retained'];delta=np.abs(current-old).max(0)
                metric=alpha_coverage_metrics(alpha,dom,maximum_alpha=maximum[key] if maximum else None)
                metric.update(raw=key,step=trainer.step,baseline=alpha_coverage_metrics(base,dom),
                    joint_K_rgb_max=float(delta[K].max()) if K.any() else 0.,joint_K_changed_gt1byte_pixels=int((K&(delta>1/255+1e-7)).sum()),
                    joint_K_rgb_mean=float(np.abs(current-old)[:,K].mean()) if K.any() else 0.,
                    road_only_K_alpha_max_delta=float(np.abs(alpha-base)[K].max()) if K.any() else 0.)
                d=dest/f'{key:03d}';d.mkdir();np.savez_compressed(d/'arrays.npz',road_only_rgb=pack['rgb'].cpu().numpy(),road_only_alpha=alpha,
                    baseline_alpha=base,joint_rgb=current,baseline_joint_rgb=old,**dom)
                save_rgb(d/'road_only.png',pack['rgb'].cpu().numpy());save_rgb(d/'joint.png',current)
                records.append(metric)
        self.scene.assert_persistent_unchanged()
        result=dict(rows=records,step=trainer.step,scope='pilot5_only',additional_optimizer_updates=0,
            evaluation_pixels=sum(r['evaluation_pixels'] for r in records),gap_pixels=sum(r['gap_pixels'] for r in records),
            thin_pixels=sum(r['thin_pixels'] for r in records),supported_pixels=sum(r['supported_pixels'] for r in records),
            baseline_gap_pixels=sum(r['baseline']['gap_pixels'] for r in records),baseline_thin_pixels=sum(r['baseline']['thin_pixels'] for r in records),
            joint_K_changed_gt1byte_pixels=sum(r['joint_K_changed_gt1byte_pixels'] for r in records),joint_K_rgb_max=max(r['joint_K_rgb_max'] for r in records),
            unsupported_geometry_pixels=sum(r['geometry_unsupported_pixels'] for r in records),unsupported_geometry_still_evaluated=True)
        write(dest/'evaluation.json',result);return result

    def export(self,trainer,dest):
        dest=Path(dest);dest.mkdir(parents=True,exist_ok=False)
        road={k:v.copy() for k,v in self.road.items()}
        with torch.no_grad():road['road_opacity'][self.world_train]=trainer.opacity()[self.active_train].cpu().numpy()
        if trainer.scale_factor is not None:
            road['gaussian_sigma_m'][self.world_scale,:2]=self.road['gaussian_sigma_m'][self.world_scale,:2]*trainer.scale_factor.detach().cpu().numpy()[:,None]
        for key in road:
            if key not in ['road_opacity','gaussian_sigma_m'] and not np.array_equal(road[key],self.road[key],equal_nan=True):raise RuntimeError('Phase1改变了冻结字段 '+key)
        if (not np.array_equal(road['gaussian_sigma_m'][~self.scale_eligible],self.road['gaussian_sigma_m'][~self.scale_eligible])
                or not np.array_equal(road['gaussian_sigma_m'][:,2],self.road['gaussian_sigma_m'][:,2])):raise RuntimeError('Phase1改变了未授权sigma或法向厚度')
        outside=~self.eligible
        if not np.array_equal(road['road_opacity'][outside],self.road['road_opacity'][outside]):raise RuntimeError('Phase1改变了授权范围外opacity')
        np.savez_compressed(dest/'road.npz',**road)
        prov={k:v.copy() for k,v in self.prov.items()};prov.update(phase1_trainable_world_rows=self.eligible,
            phase1_opacity_before=self.road['road_opacity'],phase1_opacity_after=road['road_opacity'])
        prov.update(phase1_scale_trainable_world_rows=self.scale_eligible,phase1_scale_upper=self.scale_upper,
            phase1_scale_actual_boundary_distance_m=self.scale_distance,effective_sigma_m=road['gaussian_sigma_m'])
        prov['effective_road_opacity']=road['road_opacity']
        np.savez_compressed(dest/'provenance.npz',**prov)
        for filename in ['atlas.npz','markings.npz']:
            if (self.asset/filename).is_file():shutil.copy2(self.asset/filename,dest/filename)
        save_scene(self.asset/'persistent/scene.json',self.asset/'persistent/proof.json',dest/'road.npz',dest/'persistent')
        write(dest/'phase1_export.json',dict(schema='road_coverage_phase1_export.v1',step=trainer.step,source_asset=str(self.asset),
            accounting_role=trainer.accounting_role,
            trainable_world_rows=len(self.world_train),scope=self.scope,contract=asdict(self.contract),source_binding=self.binding,
            opacity_only=trainer.scale_factor is None,next_stage='phase2 local texture delta; phase1 opacity/scales/xyz/normal remain frozen'))
        return dest


def prepare_scale(env,output):
    """固定最大opacity+scale足迹的true K union，后续训练必须载入该额外guard。"""
    from drivingeditor_dynamic_edit.native_row_contribution import attribute_prepared_rows
    if not env.contract.train_tangent_scales:raise ValueError('最大scale准备要求--bounded-scale')
    trainer=env.trainer('phase0');opacity=env.base_opacity.clone();opacity[trainer.ids]=1-1e-5
    scales=trainer.scales(trainer.scale_upper);massmax=np.zeros(len(env.eligible),np.float32);rows=[]
    with torch.no_grad():opacity=opacity.detach();scales=scales.detach()
    for key in KEYS:
        with torch.no_grad():
            values,owners,camera,_=env.scene._rows(key,(),(),False)
            prefix=len(env.scene.static.get_xyz)+(len(env.scene.dynamic.get_xyz) if env.scene._include_dynamic else 0)
            maximum=dict(values,opacities=torch.cat([values['opacities'][:prefix],opacity[:,None]],0),
                scales=torch.cat([values['scales'][:prefix],scales],0),mask=values['mask']&~env.scene._contains(owners,(10,)))
        attribution=attribute_prepared_rows(env.scene,maximum,owners,camera,{'K':env.numpy_domains[key]['retained']})
        mass=attribution.alpha_mass[prefix:,0];massmax[env.active_world]=np.maximum(massmax[env.active_world],mass)
        rows.append(dict(raw=key,road_K_mass=float(mass.sum()),K_world_rows=int((mass>1e-7).sum()),conservation=attribution.conservation_relative_error))
    guard=env.original_guard|(massmax>1e-7);added=guard&~env.original_guard
    np.savez_compressed(output/'maximum_state_K_guard.npz',guard=guard,new_guard=added,maximum_K_mass=massmax,
        provisional_scale_eligible=env.scale_eligible,scale_upper=env.scale_upper,actual_world_boundary_distance_m=env.scale_distance)
    write(output/'maximum_state_K_guard.json',dict(schema='maximum_opacity_scale_K_guard.v1',source_binding=env.binding,
        contract=asdict(env.contract),frames=KEYS,rows=rows,old_guard_rows=int(env.original_guard.sum()),new_guard_rows=int(added.sum()),
        union_guard_rows=int(guard.sum()),formal_updates=0,training_guard_must_be_loaded_before_phase0_and_formal=True,
        max_opacity=float(1-1e-5),maximum_scale_factor=1.15,threshold=1e-7,GPU='keep_on'))
    print('MAXIMUM_K_GUARD',int(added.sum()),'new, union',int(guard.sum()),flush=True)


def phase0(env,output):
    trainer=env.trainer('phase0');maximum={};rows=[];gradient_max=np.zeros(len(env.active_train),np.float32)
    for key in KEYS:
        trainer.optimizer.zero_grad(set_to_none=True);loss,parts=trainer.loss_for(key);loss.backward()
        grad=trainer.logits.grad.detach();gradient_max=np.maximum(gradient_max,grad.abs().cpu().numpy())
        if not torch.isfinite(loss) or not torch.isfinite(grad).all():raise RuntimeError('真实phase0损失或梯度非有限')
        with torch.no_grad():
            upper=env.base_opacity.clone();upper[trainer.ids]=1-1e-5
            upper_scales=trainer.scales(trainer.scale_upper) if trainer.scale_factor is not None else None
            maximum[key]=env.render_alpha(key,upper,upper_scales).cpu().numpy()
            epsilon=.01
            lo=env.render_alpha(key,trainer.opacity(trainer.logits-epsilon));hi=env.render_alpha(key,trainer.opacity(trainer.logits+epsilon))
            control=((hi-lo)/(2*epsilon)).abs().cpu().numpy();dom=env.numpy_domains[key]
            base=env.baseline[key].cpu().numpy();mask=dom['evaluation'];gap=mask&(base<.05)
        row=dict(raw=key,loss=float(loss),**parts,nonzero_gradient_parameters=int((grad.abs()>1e-12).sum()),
            gradient_max=float(grad.abs().max()),evaluation_control_pixels=int((mask&(control>1e-6)).sum()),
            evaluation_control_fraction=float((mask&(control>1e-6)).sum()/max(int(mask.sum()),1)),
            gap_control_pixels=int((gap&(control>1e-6)).sum()),
            metrics=alpha_coverage_metrics(base,dom,maximum_alpha=maximum[key]))
        if trainer.scale_factor is not None:
            sg=trainer.scale_factor.grad
            if sg is None or not bool(torch.isfinite(sg).all()):raise RuntimeError('Phase0面内scale梯度非有限')
            row.update(nonzero_scale_gradient_parameters=int((sg.abs()>1e-12).sum()),scale_gradient_max=float(sg.abs().max()))
        residual=dom['evaluation']&(maximum[key]<.05)
        left=residual.copy();exclusive={}
        for label,mask in [('no_geometry',dom['geometry_unsupported']),
            ('surface_ray_intersection_failed',dom.get('surface_intersection_failed',np.zeros_like(residual))),('inactive_world_center',dom['inactive_center']),
            ('near_clipped',dom['near_clipped']),('material_unknown',dom['material_unknown']),('world_guard',dom['frozen_world_guard'])]:
            exclusive[label]=int((left&mask).sum());left &= ~mask
        exclusive['active_legal_footprint_or_projection_gap']=int(left.sum())
        row['maximum_contract_residual_gap_exclusive']=exclusive
        rows.append(row);print('PHASE0',json.dumps(row),flush=True)
        np.savez_compressed(output/f'phase0_maps_{key:03d}.npz',baseline_alpha=base,maximum_opacity_alpha=maximum[key],
            logit_finite_difference_control=control,**dom)
    torch.save(trainer.state(),output/'checkpoint000000.pt')
    np.savez_compressed(output/'gradient_scope.npz',world_ids=env.world_train,gradient_max=gradient_max)
    np.savez_compressed(output/'scale_scope.npz',world_ids=env.world_scale,scale_upper=env.scale_upper[env.world_scale],
        actual_world_boundary_distance_m=env.scale_distance[env.world_scale],base_sigma_m=env.road['gaussian_sigma_m'][env.world_scale])
    evaluation=env.evaluate(trainer,output/'evaluation',maximum)
    baseline_gap=sum(r['metrics']['gap_pixels'] for r in rows);upper_gap=sum(r['metrics']['opacity_maximum_gap_pixels'] for r in rows)
    support_gain=sum(r['metrics']['opacity_can_reach_supported_pixels'] for r in rows)
    passed=bool((gradient_max>1e-12).any()) and (upper_gap<baseline_gap or support_gain>0)
    result=dict(schema='module2_phase0_coverage.v1',formal_updates=0,real_forward_backward=True,finite=True,
        source_binding=env.binding,scope=env.scope,contract=asdict(env.contract),rows=rows,
        gradient_covered_parameters=int((gradient_max>1e-12).sum()),trainable_parameters=len(gradient_max),
        baseline_gap_pixels=baseline_gap,opacity_upper_bound_gap_pixels=upper_gap,opacity_upper_bound_support_gain_pixels=support_gain,
        stage0_controllability_pass=passed,scale_trainable=trainer.scale_factor is not None,scale_scope=env.scale_scope,evaluation=evaluation,
        gradient_map_method='actual native backward per opacity logit; per-pixel controllability central finite difference epsilon .01; maximum opacity probe independently reported',
        independent_road_rgb_truth_used=False,bootstrap_alpha_is_preservation_only=True,formal_budget_unchanged=600,GPU='keep_on')
    write(output/'phase0.json',result);return trainer,result


def smoke(env,output):
    trainer=env.trainer('smoke');records=[]
    for _ in range(4):records.append(trainer.update())
    checkpoint=trainer.state();torch.save(checkpoint,output/'checkpoint000004.pt')
    evaluation=env.evaluate(trainer,output/'evaluation')
    replay_states=[]
    for name in ['restore','replay']:
        resumed=env.trainer('smoke');restore=resumed.restore(torch.load(output/'checkpoint000004.pt',map_location='cpu',weights_only=False))
        record=resumed.update();state=resumed.state();torch.save(state,output/f'{name}_checkpoint000005.pt');replay_states.append(state)
        write(output/(name+'.json'),dict(**restore,next_update=record))
    differences=_tensor_difference(*replay_states)
    a,b=replay_states
    parameter_error=float((a['logits'].cpu()-b['logits'].cpu()).abs().max())
    if a['scale_factor'] is not None:parameter_error=max(parameter_error,float((a['scale_factor'].cpu()-b['scale_factor'].cpu()).abs().max()))
    moment_error=0.
    for index,state in a['optimizer']['state'].items():
        for name,value in state.items():
            if torch.is_tensor(value):moment_error=max(moment_error,float((value.cpu()-b['optimizer']['state'][index][name].cpu()).abs().max()))
    meta_a={k:v for k,v in a.items() if k not in ['logits','scale_factor','optimizer']}
    meta_b={k:v for k,v in b.items() if k not in ['logits','scale_factor','optimizer']}
    metadata_equal=not _tensor_difference(meta_a,meta_b) and not _tensor_difference(a['optimizer']['param_groups'],b['optimizer']['param_groups'])
    replay_ok=metadata_equal and parameter_error<=1e-6 and moment_error<=1e-10
    diagnostics=dict(bitwise_equal=not differences,parameter_max_abs_difference=parameter_error,optimizer_moment_max_abs_difference=moment_error,
        metadata_and_rng_exact=metadata_equal,tolerance=dict(parameters=1e-6,moments=1e-10),within_tolerance=replay_ok,strict_differences=differences)
    write(output/'next_step_replay.json',diagnostics)
    if not replay_ok:raise RuntimeError('下一步双重重放超出已记录浮点误差界 '+str(diagnostics))
    export=env.export(trainer,output/'export')
    reloaded,manifest,receipt=load_persistent_handoff(export/'persistent/scene.json',export/'persistent/proof.json')
    reloaded.assert_persistent_unchanged();del reloaded;gc.collect();torch.cuda.empty_cache()
    write(output/'smoke.json',dict(completed=True,real_GPU=True,smoke_updates=4,replay_updates=2,formal_updates=0,
        finite_updates=records,full_state_restore_exact=True,next_step_replay_exact=not differences,next_step_replay_within_tolerance=True,replay_diagnostics=diagnostics,evaluation=evaluation,
        next_stage_entry_export=str(export),next_stage_persistent_reload=receipt,formal_budget_unchanged=600,
        expected_formal_duration_under_one_hour=True,GPU='keep_on'))


def formal(env,output,args):
    if env.contract.train_tangent_scales and not args.extra_guard:raise ValueError('有界scale正式训练必须先加载最大状态K union')
    gate=read(args.phase0_receipt)
    if gate['source_binding']!=env.binding or gate['contract']!=asdict(env.contract) or not gate['stage0_controllability_pass']:
        raise ValueError('正式训练要求同一冻结输入与合同的真实Phase0可控性通过')
    trainer=env.trainer()
    if args.resume:trainer.restore(torch.load(args.resume,map_location='cpu',weights_only=False))
    records=[];start=trainer.step;tick=time.monotonic()
    while trainer.step<600:
        row=trainer.update();records.append(row)
        if trainer.step%100==0 or trainer.step==start+1:print('FORMAL',json.dumps(dict(**row,seconds=time.monotonic()-tick)),flush=True)
        if trainer.step in [100,300,600]:torch.save(trainer.state(),output/f'checkpoint{trainer.step:06d}.pt')
    evaluation=env.evaluate(trainer,output/'evaluation');export=env.export(trainer,output/'export')
    write(output/'training.json',records)
    write(output/'completed.json',dict(completed=True,formal_step=600,formal_updates_this_invocation=600-start,
        smoke_updates=0,replay_updates=0,checkpoint=str(output/'checkpoint000600.pt'),evaluation=evaluation,
        export=str(export),source_binding=env.binding,scope=env.scope,contract=asdict(env.contract),
        phase2_entry=str(export/'persistent/scene.json'),phase3_static_not_started=True,GPU='keep_on'))


def main(args):
    args.output.mkdir(parents=True,exist_ok=False);start=time.monotonic();env=Environment(args)
    torch.cuda.reset_peak_memory_stats()
    write(args.output/'contract.json',dict(contract=asdict(env.contract),source_binding=env.binding,scope=env.scope,
        sampler='seed101 deterministic shuffled cycles over five fixed keys',loss_target='road-only alpha=1 only legal fit; K/outside bootstrap alpha preservation regularizers'))
    if args.mode=='prepare-scale':prepare_scale(env,args.output)
    elif args.mode=='phase0':phase0(env,args.output)
    elif args.mode=='smoke':smoke(env,args.output)
    else:formal(env,args.output,args)
    write(args.output/'resources.json',dict(seconds=time.monotonic()-start,
        max_cuda_memory_allocated_bytes=torch.cuda.max_memory_allocated(),max_cuda_memory_reserved_bytes=torch.cuda.max_memory_reserved(),
        device=torch.cuda.get_device_name(),formal_updates=read(args.output/'completed.json')['formal_updates_this_invocation'] if args.mode=='formal' else 0,
        smoke_and_replay_updates=6 if args.mode=='smoke' else 0,GPU='keep_on'))
    env.scene.assert_persistent_unchanged();print('COMPLETE',args.mode,'seconds',time.monotonic()-start,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--asset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['phase0','smoke','formal','prepare-scale'],required=True);p.add_argument('--road-views',type=Path,default=VIEW)
    p.add_argument('--supervision',type=Path);p.add_argument('--phase0-receipt',type=Path);p.add_argument('--resume',type=Path)
    p.add_argument('--bounded-scale',action='store_true');p.add_argument('--extra-guard',type=Path)
    main(p.parse_args())
