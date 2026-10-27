"""First-scene matched opacity calibration. Frozen structure and DC; no ownership edits.

Each formal arm starts from the SAME frozen Actor10K, for 10K new updates.
Calibration holdout is not unseen reconstruction: the source model saw all49.
"""
import argparse,json,time,random,os,signal,math
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import binary_dilation
from train_actor import Actor,NativeMotion,render,ROOT,OLD,torch,cpu,equal
from segmentation import write
from v43_resource_control import require_start
from diagnose_first_badcases import save

CONTEXT='14705303724557273004_3105_000_3125_000'

class LocalCalibration(torch.nn.Module):
    def __init__(self,actor,confidence):
        super().__init__()
        self.net=torch.nn.Sequential(torch.nn.Linear(8,32),torch.nn.SiLU(),torch.nn.Linear(32,32),torch.nn.SiLU(),torch.nn.Linear(32,1))
        torch.nn.init.zeros_(self.net[-1].weight);torch.nn.init.zeros_(self.net[-1].bias)
        self.register_buffer('confidence',confidence[:,None])
        inp=torch.cat((actor.xyz.detach()/actor.box_half.clamp_min(.1),actor.log_scale.detach(),actor.opacity.detach().sigmoid(),self.confidence),1)
        self.register_buffer('inputs',inp)
    def factors(self):return torch.exp(math.log(4)*self.confidence*torch.tanh(self.net(self.inputs)))

class QueryActor:
    def __init__(self,actor,calibration):self.actor=actor;self.calibration=calibration
    def kwargs(self,*args,**kw):
        d=self.actor.kwargs(*args,**kw)
        if self.calibration is not None:
            o=d['opacities'].clamp(1e-6,1-1e-6);d['opacities']=-torch.expm1(torch.log1p(-o)*self.calibration.factors())
        return d

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['legacy','balanced','field','boundary'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=10000);p.add_argument('--resume',type=Path);p.add_argument('--save-every',type=int,default=1000);p.add_argument('--managed-round',type=Path);a=p.parse_args()
    require_start(CONTEXT);assert 0<a.steps<=10000
    from review_gpu_slot import slot
    a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic();stop=[False];signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    def status(**kw):
        d=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],arm=a.arm,elapsed_seconds=time.monotonic()-started);d.update(kw);write(a.output/'status.json',d)
    status(phase='waiting_GPU',step=0)
    with slot(a.managed_round):
        torch.set_num_threads(1);random.seed(4301);np.random.seed(4301);torch.manual_seed(4301);torch.cuda.manual_seed_all(4301)
        ds=ROOT/'datasets'/CONTEXT;frames=json.loads((ds/'manifest.json').read_text())['frames'];train=[i for i in range(49) if i%3!=1];hold=[i for i in range(49) if i%3==1]
        actor=Actor(np.load(ds/'seeds.npz'));source=torch.load(str(ROOT/'actor_campaign_v1'/CONTEXT/'formal10k/checkpoint010000.pth'),map_location='cuda')['actor'];actor.load_state_dict(source)
        reference={k:v.clone() for k,v in actor.state_dict().items()}
        for param in actor.parameters():param.requires_grad_(False)
        n=NativeMotion(OLD/'native_scene_training_v1'/CONTEXT/'formal30k',30000);ns=len(n.static._xyz);hits=np.zeros(len(actor.xyz),np.int32)
        for i in train:
            raw=frames[i]['raw'];d=np.load(ROOT/'boundary_semantics_v2'/CONTEXT/('actor10k_%03d.npz'%raw));ids=d['ids'];ok=(ids>=ns)&(d['mass']>.01)&(d['target_fraction']>.5)&(d['class_id']>=14)&(d['class_id']<=19)&(d['purity']>.65);hits[ids[ok]-ns]+=1
        confidence=torch.from_numpy(np.sqrt(hits/(hits+3)).astype('float32')).cuda()
        field=LocalCalibration(actor,confidence).cuda() if a.arm=='field' else None
        if field is None:actor.opacity.requires_grad_(True)
        params=list(field.parameters()) if field is not None else [actor.opacity]
        opt=torch.optim.Adam(params,lr=1e-3 if field is not None else 5e-3,eps=1e-8);qactor=QueryActor(actor,field)
        graph=torch.from_numpy(cKDTree(cpu(actor.xyz)).query(cpu(actor.xyz),k=7)[1][:,1:].astype('int64')).cuda()
        obs=[];outside_bands=[];band_radii=[]
        for f in frames:
            d=np.load(ds/('%03d.npz'%f['raw']));obs.append((torch.from_numpy(d['rgb'].transpose(2,0,1).astype('float32')/255).cuda(),torch.from_numpy(d['body']).cuda(),torch.from_numpy(d['depth']).cuda()));radius=int(np.clip(round(.1*np.sqrt(d['body'].sum())),2,12));band_radii.append(radius);outside_bands.append(torch.from_numpy(binary_dilation(d['body'],iterations=radius)&~d['body']).cuda())
        first=0;stack=[]
        def state(step):return dict(step=step,arm=a.arm,actor=actor.state_dict(),field=field.state_dict() if field is not None else None,optimizer=opt.state_dict(),sampler=list(stack),python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        if a.resume:
            s=torch.load(str(a.resume),map_location='cuda');assert s['arm']==a.arm;actor.load_state_dict(s['actor'])
            if field is not None:field.load_state_dict(s['field'])
            opt.load_state_dict(s['optimizer']);stack=s['sampler'];first=s['step'];random.setstate(s['python_rng']);np.random.set_state(s['numpy_rng']);torch.set_rng_state(s['torch_rng'].cpu());torch.cuda.set_rng_state_all([t.cpu() for t in s['cuda_rng']])
            restored=state(first);checks={k:equal(restored[k],s[k]) for k in restored};write(a.output/'restore_verification.json',dict(checks=checks,all_passed=all(checks.values())));assert all(checks.values())
        contract=dict(context=CONTEXT,arm=a.arm,new_updates_budget=a.steps,initial_checkpoint='Actor10K frozen same source for all arms',train_raws=[frames[i]['raw'] for i in train],holdout_raws=[frames[i]['raw'] for i in hold],holdout_scope='calibration only; source reconstruction and seeds already saw all49',frozen=['xyz','scale','quaternion','DC','static','environment'],opacity_loss='.1 full_image_MAE' if a.arm=='legacy' else '.05 mean_body(abs(A-1)) + .05 mean_outside(abs(A))',outside_supervision_preserved=True,RGB_loss='same composite_L1*1 and body_L1*5 as structure arm',sparse_depth_weight=.05,field_regularizers=dict(logq_squared=.001,neighbor_logq_squared=.001) if field is not None else None,field_q_bounds=[.25,4] if field is not None else None,field_no_support_q=1,confidence='sqrt(training_frame_body_and_semantic_support_hits/(hits+3)); no holdout evidence',representation_capacity_changed=field is not None,no_case_or_frame_id_features=True,training_method_not_full_GN11=True,quality_accepted=False,human_verdict=None)
        write(a.output/'contract.json',contract)
        if a.arm=='boundary':
            contract['opacity_loss']='.05 mean_body(abs(A-1)) + .025 mean_near_outside(abs(A)) + .025 mean_far_outside(abs(A))';contract['near_outside_rule']='morphological dilation radius clip(round(.1*sqrt(body_pixels)),2,12), minus body; radii follow object image area';contract['band_radii_px']=band_radii;contract['exploratory_followup']='Added after observing balanced edge leakage; same first-scene calibration holdout reused, not a fresh test';write(a.output/'contract.json',contract)
        @torch.no_grad()
        def evaluate(step):
            dest=a.output/'evaluation'/('%06d'%step);dest.mkdir(parents=True,exist_ok=False);records=[]
            # All observed views plus same old edit requests, not counterfactual GT.
            for i,f in enumerate(frames):
                gt,body,depth=obs[i];composite=render(n,qactor,f,i);iso=render(n,qactor,f,i,static=False);alpha=iso['alpha'][0];weight=composite['extra'][0]
                records.append(dict(raw=f['raw'],split='holdout' if i in hold else 'train',inside_alpha=float(alpha[body].mean()),outside_alpha=float(alpha[~body].mean()),inside_alpha_MAE=float((1-alpha[body]).abs().mean()),outside_alpha_MAE=float(alpha[~body].abs().mean()),visible_fraction=float((weight[body]/alpha[body].clamp_min(1e-6)).clamp(0,1).mean())))
                save(dest/'no_edit/rgb'/('%03d.png'%f['raw']),composite['rgb']);save(dest/'no_edit/actor'/('%03d.png'%f['raw']),iso['rgb'])
            for case in ['turn_left15','translate_lateral_1m']:
                requests=json.loads((OLD/'motion_edit_suite_v1'/CONTEXT/(case+'.json')).read_text())['frames']
                for row in requests:
                    i=row['camera_index'];f=frames[i];edit=dict(rotation=row['target_rotation_delta_world'],center=row['target_requested_center_world']);r=render(n,qactor,f,i,edit=edit);iso=render(n,qactor,f,i,static=False,edit=edit)
                    save(dest/case/'rgb'/('%03d.png'%f['raw']),r['rgb']);save(dest/case/'actor'/('%03d.png'%f['raw']),iso['rgb'])
                    np.savez_compressed(dest/case/('%03d_maps.npz'%f['raw']),alpha=cpu(r['alpha'][0]).astype('float16'),dynamic_weight=cpu(r['extra'][0]).astype('float16'),isolated_alpha=cpu(iso['alpha'][0]).astype('float16'),visibility_ratio=cpu((r['extra'][0]/iso['alpha'][0].clamp_min(1e-6)).clamp(0,1)).astype('float16'))
            effective=qactor.kwargs(n,n.camera(dict(camera_index=0,frame_id=n.records[0]['img_name'],camera_model_time=frames[0]['model_time']),{}),frames[0])['opacities']
            np.savez_compressed(dest/'effective_opacity.npz',opacity=cpu(effective[:,0]),q=cpu(field.factors()[:,0]) if field is not None else np.ones(len(actor.xyz),np.float32))
            write(dest/'summary.json',dict(step=step,records=records,RGB_scores_used=False,point_count=len(actor.xyz),quality_accepted=False,human_verdict=None))
        def save_node(step):
            p=a.output/('checkpoint%06d.pth'%step);tmp=p.with_suffix('.tmp');torch.save(state(step),str(tmp));tmp.replace(p);s=torch.load(str(p),map_location='cpu');assert equal(s['actor'],actor.state_dict());evaluate(step);write(a.output/'latest_complete_node.json',dict(step=step,checkpoint=str(p),evaluation=str(a.output/'evaluation'/('%06d'%step))))
        log=(a.output/'steps.jsonl').open('a');status(phase='training',step=first)
        for step in range(first+1,a.steps+1):
            if not stack:stack=list(train);random.shuffle(stack)
            i=stack.pop();gt,body,depth=obs[i];f=frames[i];opt.zero_grad(set_to_none=True);r=render(n,qactor,f,i);iso=render(n,qactor,f,i,static=False);alpha=iso['alpha'][0]
            L_in=(alpha[body]-1).abs().mean();L_out=alpha[~body].abs().mean();L_alpha=.1*(alpha-body.float()).abs().mean() if a.arm=='legacy' else .05*L_in+.05*L_out
            if a.arm=='boundary':
                near=outside_bands[i];far=(~body)&(~near);L_alpha=.05*L_in+.025*alpha[near].abs().mean()+.025*alpha[far].abs().mean()
            rgb=(r['rgb']-gt).abs().mean();bodyrgb=((iso['rgb']-gt).abs()*body[None]).sum()/(3*body.sum()).clamp_min(1)
            valid=torch.isfinite(depth)&body&(alpha.detach()>.1);z=iso['depth_sum'][0]/alpha.clamp_min(1e-6)/n.scale;ld=(z[valid]-depth[valid]).abs().clamp_max(10).mean() if bool(valid.any()) else z.sum()*0
            regular=alpha.sum()*0
            if field is not None:
                lq=field.factors().log()[:,0];regular=.001*lq.square().mean()+.001*(lq[:,None]-lq[graph]).square().mean()
            loss=rgb+5*bodyrgb+L_alpha+.05*ld+regular;assert torch.isfinite(loss);loss.backward();assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in params);opt.step()
            if field is None:
                with torch.no_grad():actor.opacity.clamp_(-10,8)
            if step%25==0 or step==first+1:
                rec=dict(step=step,camera_index=i,loss=float(loss),alpha_inside_MAE=float(L_in),alpha_outside_MAE=float(L_out),elapsed_seconds=time.monotonic()-started,cuda_current_reserved=torch.cuda.memory_reserved());log.write(json.dumps(rec)+'\n');log.flush();status(phase='training',**rec);print('OPACITY',a.arm,step,round(float(loss),5),flush=True)
            if step%a.save_every==0 or step==a.steps or stop[0] or (ROOT/'USER_STOP').exists():
                save_node(step)
                if stop[0] or (ROOT/'USER_STOP').exists():break
            if step%25==0:require_start(CONTEXT)
        log.close();unchanged={k:torch.equal(v,reference[k]) for k,v in actor.state_dict().items() if k!='opacity' or field is not None};assert all(unchanged.values()) and n.unchanged();write(a.output/'frozen_verification.json',dict(checks=unchanged,source_model_unchanged=True))
        status(phase='complete' if step==a.steps else 'stopped',step=step,cuda_peak_allocated=torch.cuda.max_memory_allocated(),cuda_peak_reserved=torch.cuda.max_memory_reserved())

if __name__=='__main__':main()
