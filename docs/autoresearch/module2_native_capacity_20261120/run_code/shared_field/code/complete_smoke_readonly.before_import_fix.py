"""Unique-checkpoint CPU/native read-only lifecycle completion; zero new updates.

No repetition to manufacture duration: every saved checkpoint is checked once,
then the two final persistent asset authorities are verified in all five views.
"""
from pathlib import Path
import copy,gc,hashlib,json,sys,time,traceback
import numpy as np
from scipy import ndimage as ndi

C=Path('/root/autodl-tmp/drivingeditor_repro/campaign');Q=C/'road_teacher_detail_cpu_20261119_r1';G=Q/'gpu_20261119'
B=G/'native_capacity_20261120_r1';P=B/'smoke_r1';OUT=P/'readonly_completion_r1';OWNER='native_capacity_smoke'
TRAIN=(86,110);ALL=(86,110,134,179,198)
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
    return h.hexdigest()


def independent_seen_terms(predictions,frames,normalizers):
    """Independent FP64 SciPy fullres quadratic terms, only frozen seen masks."""
    report={};total=0.
    for raw,frame in frames.items():
        P=np.asarray(predictions[raw],np.float64);alpha=frame['alpha'].astype(np.float64)
        for name,weight in(('A',1.),('C',.5),('S',1.)):
            context=frame['contexts'][name];score=frame['scores'][name];yy,xx=np.nonzero(context)
            y0,y1,x0,x1=(int(yy.min()),int(yy.max())+1,int(xx.min()),int(xx.max())+1)if len(yy)else(0,1,0,1)
            sl=np.s_[y0:y1,x0:x1];m=context[sl];s=score[sl]&m;a=alpha[sl]
            E=np.where(m[None],(P[:,sl[0],sl[1]]-frame['targets'][name][:,sl[0],sl[1]]*a[None])/np.maximum(a,.95)[None],0.)
            terms={'RGB':float(np.mean(E[:,s]**2))if s.any()else 0.}
            for sigma in(1.,2.,4.):
                radius=int(3*sigma+.5);core=(ndi.minimum_filter(m.astype(np.uint8),size=2*radius+1,mode='constant',cval=0)>0)&s
                mass=ndi.gaussian_filter(m.astype(float),sigma,mode='constant',truncate=3)
                low=np.stack([ndi.gaussian_filter(e,sigma,mode='constant',truncate=3)for e in E])/np.maximum(mass,1e-12)[None]
                terms['HF_'+str(sigma)]=float(np.mean((E-low)[:,core]**2))if core.any()else 0.
            core=(ndi.minimum_filter(m.astype(np.uint8),size=3,mode='constant',cval=0)>0)&s
            gradient=np.stack([np.stack([ndi.sobel(e,axis=axis,mode='constant')/8 for e in E])for axis in(1,0)])
            terms['SOBEL']=float(np.mean(gradient[:,:,core]**2))if core.any()else 0.
            key=str(raw)+':'+name;n=normalizers[key]
            normalized=terms['RGB']/n['RGB']+sum(terms['HF_'+str(v)]/n['HF_'+str(v)]for v in(1.,2.,4.))/3+terms['SOBEL']/n['SOBEL']
            total+=weight*normalized/len(frames);report[key]=dict(raw=terms,normalized=normalized,seen_only=True)
    return dict(total=total,terms=report,implementation='independent FP64 SciPy, exact frozen seen context/strict cores')


def main():
    OUT.mkdir(parents=True,exist_ok=False);old=read(P/'receipt.json');ledger=copy.deepcopy(old['physical_ledger'])
    if ledger['engineering_accepted_updates']>2500 or not old['all5view_K_eligible_mass_zero']:raise RuntimeError('unexpected parent smoke')
    write(OUT/'authorization.json',dict(GPU_authorized=True,formal_allowed=False,updates=0,backwards=0,
        root_authorization='Necessary zero-update completion after fast2500-update cap; unique saved checkpoints/publicfiveviews; aggregate caps unchanged',
        parent=str(P),nativeF_cap=20000,nativeB_cap=8000,engineering_updates_cap=2500,GPU_keep_on=True))
    lease=OUT/'gpu_lease.json';write(lease,dict(gpu_authorized=True,active=True,owner=OWNER,authorization=str(OUT/'authorization.json'),phase='smoke_readonly_completion'))
    started=time.monotonic();ctx=None;results=[]
    try:
        import torch
        torch.set_num_threads(1);torch.cuda.reset_peak_memory_stats()
        sys.path[:0]=[str(B/'code'),str(B/'shared_field/code'),str(B/'gs_oracle/code')]
        from native_capacity_inputs import real_frames,make_context,field_and_reader
        from native_capacity_loss import NativeCapacityLoss
        from shared_training_state import SharedTrainingState
        from gs_color_oracle import BoundedGSColors
        from drivingeditor_dynamic_edit.persistent_road_handoff import load_persistent_handoff
        from drivingeditor_dynamic_edit.v44_manifest import SceneManifest,make_native_camera
        binding={str(Path(__file__)):sha(__file__),str(P/'receipt.json'):sha(P/'receipt.json'),str(P/'loss_snapshot.pt'):sha(P/'loss_snapshot.pt')}
        if any(sha(p)!=h for p,h in old['source_bindings'].items()):raise RuntimeError('original executed source changed')
        frames,_=real_frames('seen');ctx=make_context(lease,OWNER,frames=ALL);model,reader,field_binding=field_and_reader(ctx.base_rgb,'cuda')
        objective=NativeCapacityLoss(frames,device='cuda');objective.restore(torch.load(P/'loss_snapshot.pt',map_location='cpu',weights_only=False))
        fixed={r:torch.as_tensor(frames[r]['alpha'],device='cuda')for r in TRAIN}
        def call(label,raw,fn):
            if ledger['native_forwards']>=20000:raise RuntimeError('aggregate nativeF cap')
            ctx.require_lease();ledger['native_forwards']+=1;ledger['forwards_by_label'][label]=ledger['forwards_by_label'].get(label,0)+1
            return fn()
        class TrainingView:
            frames=TRAIN;active_world_rows=ctx.active_world_rows
            @property
            def native_calls(self):return ctx.native_calls
            def render_road(self,raw,colors):return call('completion_road',raw,lambda:ctx.render_road(raw,colors))
        first=torch.load(P/'B5_checkpoint.pt',map_location='cpu',weights_only=False)
        state=SharedTrainingState(model,reader,TrainingView(),objective,mode='detail',source_identity=first['source_identity'],
            source_metadata=first['source_metadata'],purpose='smoke',stage_counts=(20,2480))
        paths=[P/'B5_checkpoint.pt',P/'C21_checkpoint.pt']+sorted((P/'periodic').glob('shared_*.pt'))+[P/'shared_final_checkpoint.pt']
        # Different physical snapshots with the same logical cursor are retained
        # only once; no replayed identical state is used to pad wall time.
        seen=set();unique=[]
        for p in paths:
            v=torch.load(p,map_location='cpu',weights_only=False);key=v['cursor']
            if key not in seen:unique.append(p);seen.add(key)
        eligibility=torch.as_tensor(reader.eligible,device='cuda')
        folder=OUT/'checkpoints';folder.mkdir()
        for p in unique:
            saved=torch.load(p,map_location='cpu',weights_only=False);state.restore(saved);after=state.snapshot()
            state_exact=all(torch.equal(saved['params'][k],after['params'][k])for k in('low','high'))
            if not state_exact:raise RuntimeError('checkpoint low/high mismatch')
            t=time.monotonic()
            with torch.no_grad():
                ci=reader.colors(model);readonly=torch.equal(ci[~eligibility],ctx.base_rgb[~eligibility])
                outputs={r:call('unique_checkpoint_road',r,lambda r=r:ctx.render_road(r,ci))for r in ALL}
                if not all(torch.equal(outputs[r]['alpha'].squeeze(),fixed[r])for r in TRAIN):raise RuntimeError('checkpoint alpha changed')
                predictions={r:outputs[r]['rgb']for r in TRAIN};value,parts=objective.objective(predictions,'detail')
                cpu={r:p.detach().cpu().numpy()for r,p in predictions.items()}
                actual={key:{k:float(v)for k,v in part['raw_terms'].items()}for key,part in parts.items()}
            independent=independent_seen_terms(cpu,frames,objective.normalizers)
            errors={key:{term:abs(actual[key][term]-independent['terms'][key]['raw'][term])for term in actual[key]}for key in actual}
            relative=abs(independent['total']-float(value))/max(abs(float(value)),1e-8)
            if not readonly or relative>2e-4:raise RuntimeError('independent actual-size objective/readonly mismatch: '+str(relative))
            row=dict(checkpoint=str(p),SHA256=sha(p),cursor=state.cursor,next_stage=state.stage,field_params_exact=state_exact,
                readonly_RGB_exact=readonly,alpha_exact=True,native_views=list(ALL),torch_DETAIL=float(value),
                scipy_DETAIL=independent['total'],objective_relative_difference=relative,raw_term_abs_differences=errors,
                seconds=time.monotonic()-t,original_source_metadata=saved['source_metadata'])
            write(folder/f'{state.cursor:04d}.json',row);results.append(row)
            print('CHECKPOINT_VERIFIED',json.dumps({k:row[k]for k in('cursor','torch_DETAIL','objective_relative_difference','seconds')}),flush=True)
            del saved,after,ci,outputs,predictions,cpu,value,parts,actual,independent;gc.collect();torch.cuda.empty_cache()

        public=[];joint=[];(OUT/'final_native').mkdir()
        authority,metadata=type(model).load(P/'world_field.pt',device='cuda');oracle=BoundedGSColors.load(P/'oracle_colors.pt',device='cuda')
        with torch.no_grad():colors={'shared':reader.colors(authority).detach(),'oracle':oracle().detach()}
        parent_joint={}
        with torch.no_grad():
            for raw in ALL:parent_joint[raw]=call('parent_joint',raw,lambda r=raw:ctx.render_joint(r))
        for name,color in colors.items():
            asset=P/(name+'_asset')
            with np.load(asset/'road.npz',allow_pickle=False)as z:
                if not np.array_equal(color.cpu().numpy(),z['reference_srgb'][ctx.active_world_rows]):raise RuntimeError('authority to saved effective ci differs')
            scene,m,_=load_persistent_handoff(asset/'persistent/scene.json',asset/'persistent/proof.json')
            data=dict(m.data);data['render_wh']=[1920,1280];headers=SceneManifest(data,asset/'persistent').camera_headers()
            provider=next(p for p in scene._providers if p.kind=='road');folder=OUT/'final_native'/name;folder.mkdir()
            with torch.no_grad():
                for raw in ALL:
                    expected=call('final_authority_road',raw,lambda:ctx.render_road(raw,color));cam=make_native_camera(headers[raw])
                    actual=call('final_public_clone',raw,lambda:scene._bindings().raster(cam,scene.pipe,0,provider.rows(raw,cam,scene.pipe),scene._background))
                    row=dict(asset=name,raw=raw,RGB_max=float((actual['rgb']-expected['rgb']).abs().max()),
                        alpha_max=float((actual['alpha']-expected['alpha']).abs().max()),RGB8_max=int((torch.round(actual['rgb']*255)-torch.round(expected['rgb']*255)).abs().max()),resolution=[1920,1280])
                    public.append(row)
                    if row['RGB_max']>1e-6 or row['alpha_max']!=0:raise RuntimeError('fiveview public reload mismatch')
                    J=call('final_candidate_joint',raw,lambda:ctx.render_joint(raw,color));parent=parent_joint[raw];K=torch.as_tensor(ctx.K_masks[raw],device='cuda')
                    jrow=dict(asset=name,raw=raw,K_pixels=int(K.sum()),RGB_max=float((J['rgb']-parent['rgb']).abs().amax(0)[K].max()),alpha_max=float((J['alpha']-parent['alpha']).abs().max()))
                    joint.append(jrow)
                    np.savez_compressed(folder/f'{raw:03d}.npz',road_rgb=expected['rgb'].cpu().numpy(),road_alpha=expected['alpha'].squeeze().cpu().numpy(),
                        joint_rgb=J['rgb'].cpu().numpy(),joint_alpha=J['alpha'].squeeze().cpu().numpy(),K=ctx.K_masks[raw])
            scene.assert_persistent_unchanged();del scene,provider;gc.collect();torch.cuda.empty_cache()
        ctx.assert_frozen();state.assert_frozen(full=True)
        if any(sha(p)!=h for p,h in old['source_bindings'].items()):raise RuntimeError('source changed during completion')
        elapsed=time.monotonic()-started
        receipt=dict(schema='native_capacity_smoke_readonly_completion.v1',complete=True,additional_updates=0,additional_backwards=0,
            unique_checkpoint_count=len(results),checkpoint_results=results,public1920_reload=public,joint_K=joint,
            additional_nativeF=ledger['native_forwards']-old['physical_ledger']['native_forwards'],aggregate_physical_ledger=ledger,
            original_measured_compute_seconds=old['measured_computational_seconds'],additional_readonly_seconds=elapsed,
            measured_active_work_seconds=old['measured_computational_seconds']+elapsed,aggregate_wall_seconds=old['total_seconds']+elapsed,
            original_minimum_duration_failure_preserved=str(P/'failed.json'),duration_acceptance='root judges about10minutes from measured work; no meaningless repetition or duration padding',
            source_binding=binding,original_source_bindings=old['source_bindings'],all_states_discard_for_formal=True,GPU_keep_on=True,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        write(OUT/'receipt.json',receipt);print('COMPLETED',json.dumps({k:v for k,v in receipt.items()if k not in('checkpoint_results','source_binding','original_source_bindings','aggregate_physical_ledger')}),flush=True)
    except Exception:
        write(OUT/'failed.json',dict(error=traceback.format_exc(),aggregate_ledger=ledger,checkpoint_results=results,additional_updates=0,seconds=time.monotonic()-started));raise
    finally:
        if ctx is not None:ctx.close()
        value=read(lease);value.update(active=False,owner=None,ended_at=time.time());write(lease,value)


if __name__=='__main__':main()
