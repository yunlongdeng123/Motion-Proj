"""Bounded real-GPU local-road B+C pilot. Source checkpoint and code stay read-only."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
import argparse,copy,datetime,hashlib,json,math,random,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from bc_model import ActorShadow,LocalRoad,srgb_to_linear,sample_render,edited_casters

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1')
BASE=ROOT/'shadow_BC_20261016_v1'
PARENT=ROOT/'rigid_runs/scene0008_089_137/full30k/checkpoint_030000.pt'
sys.path.insert(0,str(ROOT/'infra'))
from priority import require_gpu_slot,inspect_priority
from runtime import load_models
import renderer

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(path,data):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False));os.replace(temporary,path)
def cpu_state(module):return {k:v.detach().cpu().clone() for k,v in module.state_dict().items()}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save_image(path,tensor):
    im=tensor.detach().cpu().numpy();Image.fromarray(np.rint(np.moveaxis(im.clip(0,1),0,-1)*255).astype('uint8')).save(path)

class Experiment:
    def __init__(self,args):
        self.args=args;self.out=Path(args.output);self.out.mkdir(exist_ok=False,parents=True)
        self.status={'schema':'shadow_BC_live_status_v1','state':'initializing','created_utc':now(),'pid':os.getpid(),
          'output':str(self.out),'steps_per_arm':args.steps,'no_subagents':True,'scope':'local evidenced road at raw101; reconstruction pilot',
          'completed_arms':[],'GPU_training_started':False,'parent_checkpoint':str(PARENT)}
        self.update();self.slot,self.admission=require_gpu_slot(str(ROOT.parent/'gpu_slot.lock'))
        torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0);random.seed(0)
        self.parent_identity=(PARENT.stat().st_size,PARENT.stat().st_mtime_ns)
        self.spec=json.loads((ROOT/'rigid_specs/scene0008_089_137.json').read_text())
        self.data_path=BASE/'training_data_v2/training_data.npz'
        with np.load(self.data_path) as f:
            keys=['xyz_model','uv','reference_srgb','reference_linear','observed_linear','clean','shadow','accepted_reference','poses','sizes_model','pose_valid','times','plane_model','gains']
            self.data={k:torch.as_tensor(f[k],device='cuda') for k in keys};self.data['world_scale']=float(f['world_scale'])
            self.frame_ids=f['frame_ids'].tolist();self.pixels=f['pixels']
        with np.load(BASE/'selection_v1/selection.npz') as f:self.selection={k:f[k] for k in f.files}
        if not all(torch.isfinite(x).all() for x in self.data.values() if torch.is_tensor(x) and x.dtype.is_floating_point):raise ValueError('Nonfinite prepared observations')
        self.update(phase='load_actual_checkpoint')
        restored=torch.load(PARENT,map_location='cuda',weights_only=False)
        self.config,self.scene,self.static,self.instances,self.old_optimizer,self.env,self.exclusion=load_models(self.spec,self.out,restore=restored)
        del restored
        for p in self.instances.parameters():p.requires_grad_(False)
        for group in self.static.optimizer.param_groups:
            for p in group['params']:p.requires_grad_(False)
        for p in self.env.parameters():p.requires_grad_(False)
        self.cameras=list(self.scene.getTrainCameras());self.bg=torch.zeros(3,device='cuda')
        assert [Path(c.image_name).stem for c in self.cameras]==self.frame_ids
        assert self.exclusion['removed_points']==0 and len(self.static.get_xyz)==915840
        self.original_prepare=renderer.prepare_static;self.active=None
        self.base_versions={name:getattr(self.static,name)._version for name in ['_xyz','_features_dc','_features_rest','_scaling','_rotation','_opacity','_t','_scaling_t','_velocity']}
        self.instance_versions={name:p._version for name,p in self.instances.named_parameters()}
        self.manifest=json.loads((ROOT/'rigid_post30k/scene0008_089_137/evaluation/scene_evaluation_manifest.json').read_text())
        self.parent_cache={};self.fit_indices=[i for i in range(49) if i not in [12,24,36]]
        self.shadow_indices=[i for i in self.fit_indices if int(self.data['shadow'][i].sum())>=20]
        assert self.shadow_indices
        self.update(phase='validate_baseline')
        with torch.no_grad():
            pkg=self.render(12)
            expected=self.parent_render(12)
            mismatch=float(abs(pkg['render']-expected).max())
            with np.load(ROOT/'rigid_post30k/scene0008_089_137/evaluation'/self.manifest['streams']['observed']['frames'][12]['npz']) as f:
                raw=torch.tensor(f['static_raw_rgb'],device='cuda')
            static=self.render(12,instances=False)
            raw_mismatch=float(abs(static['raw_render']-raw).max())
            assert mismatch<2e-6 and raw_mismatch==0,(mismatch,raw_mismatch)
        self.contract={'baseline_composite_max_abs':mismatch,'baseline_static_raw_max_abs':raw_mismatch,
          'source_checkpoint_unmodified':True,'data_sha256':sha(self.data_path),'selection_sha256':sha(BASE/'selection_v1/selection.npz'),
          'trainable_parent_or_actor_parameters':sum(p.requires_grad for p in self.instances.parameters()),
          'training_loss':'masked normalized pointwise L1 in linear RGB; no SSIM in this local pilot',
          'body_supervision':'Actor geometry and appearance frozen; full joint alpha compositing retained',
          'old_global_and_static_losses':'both replaced by explicit clean/observed routing in this separate trainer',
          'training_data_scope':'geometry/RGB-confidence pseudo labels; not independently annotated cast-shadow GT',
          'heldout_claim':False,'parent_seen_all49':True,'admission':self.admission}
        write(self.out/'runtime_contract.json',self.contract)
    def update(self,**kwargs):
        self.status.update(kwargs);self.status['updated_utc']=now();write(self.out/'status.json',self.status)
    def parent_render(self,i):
        if i not in self.parent_cache:
            path=ROOT/'rigid_post30k/scene0008_089_137/evaluation'/self.manifest['streams']['observed']['frames'][i]['npz']
            with np.load(path) as f:self.parent_cache[i]=torch.tensor(f['composite'],device='cuda')
        return self.parent_cache[i]
    def priority(self):
        check=inspect_priority()
        if not check['allowed']:
            self.update(state='yielded_to_priority_work',priority=check)
            raise RuntimeError('Priority work appeared; checkpoint preserved, no automatic interference')
    def render(self,i,*,shaded=False,instances=True,edits=None,camera=None):
        if self.active:
            self.active.current_frame=i;self.active.shaded=shaded;self.active.edits=edits
        return renderer.render(camera or self.cameras[i],self.static,self.instances if instances else None,self.config,self.bg,
          env_map=self.env,edits=edits if instances else None,max_features=10,render_separate=False,is_training=False)
    def activate(self,arm,shadow=None):
        renderer.prepare_static=self.original_prepare
        self.active=LocalRoad(self.selection,self.static,self.instances,self.data,arm,shadow).cuda()
        self.active.original_prepare=self.original_prepare;renderer.prepare_static=self.active.patch
    def deactivate(self):
        renderer.prepare_static=self.original_prepare;self.active=None
    def warmup(self,kind):
        best=None;directory=self.out/('B_'+kind);directory.mkdir()
        starts=range(8) if kind=='projection' else range(1)
        for seed in starts:
            self.priority();torch.manual_seed(0);rng=random.Random(0)
            model=ActorShadow(self.data['plane_model'].cpu(),self.data['world_scale'],kind,seed*math.pi/4).cuda()
            optimizer=torch.optim.Adam(model.parameters(),lr=.035)
            for step in range(1,self.args.warmup+1):
                i=self.shadow_indices[rng.randrange(len(self.shadow_indices))]
                att=model.attenuation(self.data['xyz_model'],self.data['poses'][i,:,:3,3],self.data['poses'][i,:,:3,:3],self.data['sizes_model'][i],self.data['pose_valid'][i])
                pred=self.data['reference_linear']*att;target=self.data['observed_linear'][i]
                s=self.data['shadow'][i];c=self.data['clean'][i]
                loss=(pred[s]-target[s]).abs().mean()+.35*(pred[c]-target[c]).abs().mean()
                optimizer.zero_grad(set_to_none=True);loss.backward()
                if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('B nonfinite gradient')
                optimizer.step()
                if step%100==0:
                    self.priority();self.update(state='running',phase='B_parameter_warmup',kind=kind,start=seed+1,step=step,total_starts=len(starts),GPU_training_started=True)
            with torch.no_grad():
                scores=[]
                for i in self.shadow_indices:
                    att=model.attenuation(self.data['xyz_model'],self.data['poses'][i,:,:3,3],self.data['poses'][i,:,:3,:3],self.data['sizes_model'][i],self.data['pose_valid'][i])
                    pred=self.data['reference_linear']*att;target=self.data['observed_linear'][i]
                    s=self.data['shadow'][i];c=self.data['clean'][i]
                    scores.append(float((pred[s]-target[s]).abs().mean()+.35*(pred[c]-target[c]).abs().mean()))
                score=float(np.mean(scores))
            row={'start':seed,'score_training_only':score,**model.describe()};write(directory/f'start_{seed}.json',row)
            print(json.dumps({'event':'B_start_complete','kind':kind,**row}),flush=True)
            if best is None or score<best['score']:
                best={'score':score,'state':cpu_state(model),'description':model.describe(),'seed':seed}
        torch.save(best,directory/'best.pt');write(directory/'receipt.json',{k:v for k,v in best.items() if k!='state'})
        return best
    def make_shadow(self,kind,best):
        model=ActorShadow(self.data['plane_model'].cpu(),self.data['world_scale'],kind).cuda();model.load_state_dict(best['state']);return model
    def losses(self,i,arm,step):
        clean_pkg=self.render(i,instances=False);clean=srgb_to_linear(sample_render(clean_pkg['render'],self.data['uv'][i]))
        accepted=self.data['accepted_reference'];ref=self.data['reference_linear'];obs=self.data['observed_linear'][i]
        cm=self.data['clean'][i];sm=self.data['shadow'][i]
        zero=clean.sum()*0
        if arm=='R1':road=(clean[cm]-obs[cm]).abs().mean()
        else:road=(clean[accepted]-ref[accepted]).abs().mean()
        joint=self.render(i,shaded=arm in ('R3','R4','R5'))
        seen=srgb_to_linear(sample_render(joint['render'],self.data['uv'][i]))
        clean_obs=(seen[cm]-obs[cm]).abs().mean() if cm.any() else zero
        shadow_obs=(seen[sm]-obs[sm]).abs().mean() if sm.any() else zero
        anchor=torch.ones((320,480),device='cuda',dtype=torch.bool);anchor[205:305,20:205]=False
        anchor_loss=(joint['render'][:,anchor]-self.parent_render(i)[:,anchor]).abs().mean()
        weight_shadow=.8 if arm in ('R3','R4','R5') and step>250 else 0.
        total=road+weight_shadow*shadow_obs+.20*clean_obs+.25*anchor_loss+self.active.regularization()
        return total,{'clean_road':float(road.detach()),'observed_clean':float(clean_obs.detach()),'observed_shadow':float(shadow_obs.detach()),'anchor':float(anchor_loss.detach()),'loss':float(total.detach())}
    def checkpoint(self,arm,step,optimizer):
        path=self.out/arm/f'addon_{step:06d}.pt'
        if path.exists():return path
        payload={'schema':'drivingeditor_shadow_local_v1','arm':arm,'step':step,
          'parent_checkpoint':str(PARENT),'parent_identity':self.parent_identity,
          'runtime_contract':self.contract,'road_addon':self.active.addon_state(),
          'shadow_state':cpu_state(self.active.shadow) if self.active.shadow else None,
          'shadow_kind':self.active.shadow.kind if self.active.shadow else None,
          'optimizer':optimizer.state_dict(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
          'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
          'sampler_rng':self.sampler.getstate() if hasattr(self,'sampler') else None,
          'run_args':vars(self.args),'original_scene_checkpoint_written':False}
        temporary=path.with_suffix('.pt.partial');torch.save(payload,temporary);os.replace(temporary,path)
        return path
    def gradient_smoke(self):
        self.activate('R4',ActorShadow(self.data['plane_model'].cpu(),self.data['world_scale']).cuda())
        total,_=self.losses(10,'R4',300);total.backward()
        names={'road_color':self.active.color_raw,'road_opacity':self.active.opacity_raw,'artifact_opacity':self.active.artifact_raw,
               'light_azimuth':self.active.shadow.azimuth,'light_elevation':self.active.shadow.elevation_raw,
               'penumbra':self.active.shadow.softness_raw,'direct_fraction':self.active.shadow.direct_raw}
        result={name:{'finite':p.grad is not None and bool(torch.isfinite(p.grad).all()),
                      'max_abs':float(p.grad.abs().max()) if p.grad is not None else None} for name,p in names.items()}
        assert all(r['finite'] for r in result.values()),result
        assert result['road_color']['max_abs']>0 and result['artifact_opacity']['max_abs']>0,result
        # Verify edit state is numerically identical to the body's current world state.
        from rigid_model import quaternion_to_matrix
        edits={0:{'translation':[.07,-.02,0.],'quaternion':[math.cos(.12),0,0,math.sin(.12)]}}
        state=edited_casters(self.instances,self.cameras[12].timestamp,edits,self.data['sizes_model'][12])
        world=self.instances.evaluate(self.cameras[12].timestamp,edits)
        for k in range(self.instances.num_instances):
            row=int(torch.nonzero(world.point_ids==k)[0]);assert torch.allclose(state[0][k],world.object_translation[row])
            assert torch.allclose(state[1][k],quaternion_to_matrix(world.object_rotation[row]),atol=2e-6)
        write(self.out/'GPU_gradient_smoke.json',{'status':'PASS','gradients':result,'caster_body_edit_pose_agreement':True})
        self.deactivate();torch.cuda.empty_cache()
    @torch.no_grad()
    def evaluate(self,arm):
        directory=self.out/arm/'evaluation';directory.mkdir(parents=True,exist_ok=True)
        clean=self.render(12,instances=False);joint=self.render(12,shaded=arm in ('R3','R4','R5'))
        reference=self.data['reference_linear'];accepted=self.data['accepted_reference'];cm=self.data['clean'][12];sm=self.data['shadow'][12]
        cleanpoints=srgb_to_linear(sample_render(clean['render'],self.data['uv'][12]));jointpoints=srgb_to_linear(sample_render(joint['render'],self.data['uv'][12]))
        footprint_file=ROOT/'black_shadow_audit_20261016_v1/focus_cpu/reviewed_artifact_footprint.png'
        fp=np.asarray(Image.open(footprint_file))>0;pix=self.pixels.astype(int);fm=torch.as_tensor(fp[pix[:,1],pix[:,0]],device='cuda')&accepted
        outside=torch.ones((320,480),device='cuda',dtype=torch.bool);outside[205:305,20:205]=False
        metrics={'clean_reference_linear_MAE':float(abs(cleanpoints[accepted]-reference[accepted]).mean()),
            'diagnostic_footprint_clean_linear_MAE':float(abs(cleanpoints[fm]-reference[fm]).mean()),
            'observed_shadow_linear_MAE':float(abs(jointpoints[sm]-self.data['observed_linear'][12,sm]).mean()),
            'observed_clean_linear_MAE':float(abs(jointpoints[cm]-self.data['observed_linear'][12,cm]).mean()),
            'outside_control_sRGB_MAE_vs_parent':float(abs(joint['render'][:,outside]-self.parent_render(12)[:,outside]).mean()),
            'static_alpha_footprint_mean':float(clean['alpha'][0][torch.as_tensor(fp,device='cuda')].mean()),
            'evaluated_raw101_shadow_pseudo_pixels':int(sm.sum()),'not_unseen_test':True,'not_independent_shadow_GT':True}
        def record(name,pkg):
            save_image(directory/(name+'.png'),pkg['render'])
            np.savez_compressed(directory/(name+'.npz'),RGB=pkg['render'].cpu().numpy(),alpha=pkg['alpha'].cpu().numpy())
        record('unshaded_static',clean);record('observed_composite',joint)
        if self.active:
            metrics['road_color_mean']=self.active.color_raw.sigmoid().mean(0).tolist()
            metrics['artifact_retained_opacity_factor_mean']=float(self.active.artifact_raw.sigmoid().mean())
            if self.active.shadow:metrics['shadow_parameters']=self.active.shadow.describe()
        delete_all={k:{'delete':True} for k in range(self.instances.num_instances)}
        deleted=self.render(12,shaded=True,edits=delete_all);record('delete_all_actors',deleted)
        metrics['delete_all_shadow_equals_unshaded_static_max_abs']=float(abs(deleted['render']-clean['render']).max())
        assert metrics['delete_all_shadow_equals_unshaded_static_max_abs']<2e-6
        record('delete_target',self.render(12,shaded=True,edits={0:{'delete':True}}))
        plane=self.data['plane_model'][:3].float();plane=plane/plane.norm();seed=torch.eye(3,device='cuda')[int(plane.abs().argmin())]
        tangent=torch.linalg.cross(plane,seed);tangent=tangent/tangent.norm()
        for metres in [-2.,-1.,1.,2.]:
            translation=(tangent*metres*self.data['world_scale']).tolist()
            record('translate_%+g_m'%metres,self.render(12,shaded=True,edits={0:{'translation':translation}}))
        for degrees in [-30.,-15.,15.,30.]:
            rad=math.radians(degrees);q=[math.cos(rad/2),*(plane*math.sin(rad/2)).tolist()]
            record('road_yaw_%+g_deg'%degrees,self.render(12,shaded=True,edits={0:{'quaternion':q}}))
        temporal=[];fp_t=torch.as_tensor(fp,device='cuda')
        for i in [0,6,12,18,24,36,48]:
            camera=copy.copy(self.cameras[12]);camera.timestamp=self.cameras[i].timestamp
            p=self.render(i,instances=False,camera=camera)
            temporal.append({'frame':self.frame_ids[i],'fixed_camera_footprint_sRGB_MAE_vs_anchor':float(abs(p['render'][:,fp_t]-clean['render'][:,fp_t]).mean())})
            save_image(directory/('fixed_camera_time_'+self.frame_ids[i]+'.png'),p['render'])
        metrics['fixed_camera_temporal']=temporal
        metrics['fixed_camera_mean_nonanchor_time_MAE']=float(np.mean([r['fixed_camera_footprint_sRGB_MAE_vs_anchor'] for r in temporal if r['frame']!='0008101']))
        # An addon restore must reconstruct precisely the same current image.
        if self.active:
            state=self.active.addon_state();self.active.color_raw.add_(.01);self.active.load_addon_state(state)
            restored=self.render(12,shaded=arm in ('R3','R4','R5'))
            metrics['addon_restore_RGB_bitwise']=bool(torch.equal(restored['render'],joint['render']))
            assert metrics['addon_restore_RGB_bitwise']
        write(directory/'metrics.json',metrics);return metrics
    def train_arm(self,arm,best=None):
        directory=self.out/arm;directory.mkdir()
        shadow=self.make_shadow('contact' if arm=='R3' else 'projection',best) if best else None
        self.activate(arm,shadow);groups=self.active.trainable_groups()
        if shadow:groups.append({'params':list(shadow.parameters()),'lr':.003,'name':'shadow'})
        optimizer=torch.optim.Adam(groups);rng=random.Random(0);self.sampler=rng;started=time.monotonic()
        for step in range(1,self.args.steps+1):
            if step==1 or step%100==0:
                if not inspect_priority()['allowed']:
                    self.checkpoint(arm,step-1,optimizer);self.priority()
            i=self.shadow_indices[rng.randrange(len(self.shadow_indices))] if step%2 else self.fit_indices[rng.randrange(len(self.fit_indices))]
            optimizer.zero_grad(set_to_none=True)
            total,values=self.losses(i,arm,step)
            if not torch.isfinite(total):raise ValueError('Nonfinite training loss')
            total.backward()
            for group in groups:
                for p in group['params']:
                    if p.grad is not None and not torch.isfinite(p.grad).all():raise ValueError('Nonfinite training gradient')
            if shadow and step<=250:
                for p in shadow.parameters():p.grad=None
            optimizer.step()
            if step==1 or step%50==0:
                elapsed=time.monotonic()-started
                row={'arm':arm,'step':step,'iteration_seconds':elapsed/step,'elapsed_seconds':elapsed,'frame_id':self.frame_ids[i],**values}
                with (directory/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                self.update(state='running',phase='C_local_road_training',arm=arm,step=step,arm_metrics=row,GPU_training_started=True)
                print(json.dumps(row),flush=True)
            if step%250==0:self.checkpoint(arm,step,optimizer)
        checkpoint=self.checkpoint(arm,self.args.steps,optimizer)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert saved['schema']=='drivingeditor_shadow_local_v1' and saved['parent_identity']==self.parent_identity
        self.active.load_addon_state(saved['road_addon'])
        if shadow:shadow.load_state_dict(saved['shadow_state'])
        def finite(value):
            if torch.is_tensor(value):return bool(torch.isfinite(value).all())
            if isinstance(value,dict):return all(finite(v) for v in value.values())
            if isinstance(value,(list,tuple)):return all(finite(v) for v in value)
            return True
        assert finite(saved['optimizer']) and saved['sampler_rng'] is not None
        optimizer.load_state_dict(saved['optimizer'])
        write(directory/'checkpoint_structure_validation.json',{'schema':saved['schema'],'step':saved['step'],
            'actual_saved_file_loaded':True,'road_and_shadow_restored':True,'Adam_state_finite_and_loaded':True,
            'sampler_state_present':True,'next_step_determinism_not_tested':True})
        del saved
        self.update(phase='self_evaluation',arm=arm,step=self.args.steps)
        metrics=self.evaluate(arm)
        self.status['completed_arms'].append({'arm':arm,'steps':self.args.steps,'metrics':metrics})
        self.update();self.deactivate();del optimizer;torch.cuda.empty_cache()
        return metrics
    def verify_sources(self):
        assert (PARENT.stat().st_size,PARENT.stat().st_mtime_ns)==self.parent_identity
        assert all(getattr(self.static,n)._version==v for n,v in self.base_versions.items())
        assert all(p._version==self.instance_versions[n] for n,p in self.instances.named_parameters())
    def run(self):
        self.gradient_smoke();self.update(phase='baseline_evaluation')
        (self.out/'R0').mkdir();base=self.evaluate('R0');write(self.out/'R0/metrics.json',base)
        self.status['completed_arms'].append({'arm':'R0','steps':0,'metrics':base});self.update()
        projection=self.warmup('projection');contact=self.warmup('contact')
        metrics={}
        for arm,best in [('R2',None),('R3',contact),('R4',projection),('R1',None),('R5',projection)]:
            metrics[arm]=self.train_arm(arm,best)
        self.verify_sources()
        gate={'clean_footprint_reduction_R4_vs_R0':1-metrics['R4']['diagnostic_footprint_clean_linear_MAE']/base['diagnostic_footprint_clean_linear_MAE'],
          'R4_shadow_better_than_C_only':metrics['R4']['observed_shadow_linear_MAE']<metrics['R2']['observed_shadow_linear_MAE'],
          'R4_shadow_better_than_contact':metrics['R4']['observed_shadow_linear_MAE']<metrics['R3']['observed_shadow_linear_MAE'],
          'outside_control_MAE':metrics['R4']['outside_control_sRGB_MAE_vs_parent'],
          'fixed_camera_mean_nonanchor_time_MAE':metrics['R4']['fixed_camera_mean_nonanchor_time_MAE'],
          'extension_not_automatic':'Review actual images and metrics before more steps; no extra training queued by this worker.'}
        write(self.out/'comparison.json',{'baseline':base,'arms':metrics,'gates':gate})
        self.update(state='complete',phase='pilot_comparisons_ready',completed_utc=now(),gates=gate,
                    parent_checkpoint_unchanged=True,source_parameters_unchanged=True,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated())

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=1000);p.add_argument('--warmup',type=int,default=300)
    a=p.parse_args();assert 1<=a.steps<=1000 and 1<=a.warmup<=300
    e=None
    try:e=Experiment(a);e.run()
    except BaseException as error:
        if e:
            e.update(state='failed' if e.status.get('state')!='yielded_to_priority_work' else e.status['state'],error=str(error),traceback=traceback.format_exc())
        else:
            out=Path(a.output)
            if out.exists():write(out/'initialization_failure.json',{'state':'failed','error':str(error),'traceback':traceback.format_exc(),'updated_utc':now()})
        raise
    finally:
        if e:
            renderer.prepare_static=e.original_prepare if hasattr(e,'original_prepare') else renderer.prepare_static
            if hasattr(e,'slot'):e.slot.close()
if __name__=='__main__':main()
