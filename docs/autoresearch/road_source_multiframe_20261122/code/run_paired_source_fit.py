"""固定源几何和400步，只改变参考池；共同曝光标定与均衡随机采样。"""
from pathlib import Path
import argparse,copy,json,random,sys,time
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_erosion,map_coordinates,gaussian_filter

CAMPAIGN=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
ROOT=CAMPAIGN/'road_source_multiframe_20261122_r3'
sys.path.insert(0,str(CAMPAIGN/'road_source_fidelity_20261122_r2/code'))
import probe_source_chain as p
import camera_corrected_source as cam
h=p.h;torch=p.torch
h.ROOT=ROOT;h.native.ROOT=ROOT;h.native.m.ROOT=ROOT;h.native.m.OWNER=ROOT.name
STEPS=400;LR=.01;REG=1e-4;SEED=0;BATCH=2

def linear(x):return np.where(x<=.04045,x/12.92,((np.maximum(x,0)+.055)/1.055)**2.4)
def encoded(x):return np.where(x<=.0031308,12.92*x,1.055*np.maximum(x,0)**(1/2.4)-.055)
def canonical(x,gain,domain):return encoded(linear(x)*gain) if domain=='linear_RGB' else x*gain
def observation(x,gain,domain):return encoded(linear(x)/gain) if domain=='linear_RGB' else x/gain

def random_schedule(pool):
    assert len(pool)%BATCH==0
    generator=np.random.default_rng(SEED);sequence=[]
    while len(sequence)<STEPS*BATCH:sequence.extend(generator.permutation(pool).tolist())
    return np.asarray(sequence[:STEPS*BATCH],np.int64).reshape(STEPS,BATCH)

def metrics(photo,rgb,mask):
    yy,xx=np.nonzero(mask);y0=max(0,int(yy.min())-14);y1=min(len(mask),int(yy.max())+15)
    x0=max(0,int(xx.min())-14);x1=min(mask.shape[1],int(xx.max())+15)
    pp=photo[y0:y1,x0:x1];rr=rgb[y0:y1,x0:x1];mm=mask[y0:y1,x0:x1]
    spectral=[]
    for sigma in [1.2,5/3,3.]:
        aa=(pp-gaussian_filter(pp,(sigma,sigma,0)))[mm].ravel()
        bb=(rr-gaussian_filter(rr,(sigma,sigma,0)))[mm].ravel()
        an,bn=np.linalg.norm(aa),np.linalg.norm(bb)
        spectral.append(dict(sigma_native_px=sigma,HF_cosine=float(aa@bb/(an*bn)) if an*bn>1e-20 else None,
            HF_RMS_ratio=float(bn/an) if an>1e-20 else None))
    return dict(score_n=int(mask.sum()),RGB_MAE=float(abs(photo-rgb)[mask].mean()),
        RGB_MSE=float(((photo-rgb)**2)[mask].mean()),spectral=spectral)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',default=str(ROOT/'pool_audit/probe_plan.json'))
    ap.add_argument('--exposure',default=str(ROOT/'exposure_audit/calibration.json'))
    ap.add_argument('--output',default='paired400_seed0');args=ap.parse_args()
    out=ROOT/args.output
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    plan=h.read(args.plan);cal=h.read(args.exposure)
    pools={'two_frames':[88,115],'eight_frames':[88,90,92,94,96,99,104,115]}
    query=[100,110,117];stress=[86];frames=sorted(set(sum(pools.values(),[])+query+stress))
    assert plan['frozen_after_native_visual_review']
    assert plan['pool_8_raws']==pools['eight_frames'] and plan['two_frame_control_raws']==pools['two_frames']
    assert set(query+stress).isdisjoint(pools['eight_frames'])
    records={int(v['raw']):v for v in plan['records']}
    gain_records={int(v['raw']):v for v in cal['records']}
    gain_domain=cal.get('gain_domain',cal.get('color_space'))
    if gain_domain=='encoded_sRGB':gain_domain='encoded_RGB'
    assert gain_domain in ['linear_RGB','encoded_RGB']
    assert cal['calibration_frozen'] and cal['all_requested_frames_calibrated']
    assert not cal['query_core_used_for_calibration']
    assert set(frames)<=set(records)&set(gain_records)
    schedules={k:random_schedule(v) for k,v in pools.items()}
    for label,schedule in schedules.items():
        assert set(schedule.ravel())==set(pools[label]);assert not set(schedule.ravel())&set(query+stress)
        h.write(out/(label+'_sampling.json'),dict(seed=SEED,policy='Repeated shuffled pool, two distinct frames per step; equal view counts',
            schedule=schedule.tolist(),counts={str(raw):int((schedule==raw).sum()) for raw in pools[label]}))
    contract=dict(task_id='ROAD-SOURCE-MULTIFRAME-20261122',run=ROOT.name,stage=args.output,source='patch_02',
        pools=pools,query_frames=query,low_resolution_stress_frames=stress,frames=frames,
        steps_per_arm=STEPS,batch_frames=BATCH,seed=SEED,optimizer='Adam',RGB_lr=LR,RGB_prior_MSE_weight=REG,
        initialization='Original patch_02 source packet RGB; identical for both arms',
        trainable='900 source core RGB rows only; halo RGB and all XYZ/normal/rotation/scale/opacity frozen',
        supervision='Equal per-view effective native RGB mean MSE on frozen photo-legal eroded physical core and initial alpha>=.95, plus initial RGB prior',
        camera='Brown observations mapped to native pinhole centers; raw-camera evaluation reverses mapping',
        exposure='Same fixed external-core calibration for both arms; no source core or source-fit parameters used to estimate gains',
        exposure_file=args.exposure,gain_domain=gain_domain,only_method_difference='Reference pool size; common sampler and all other settings',
        formal_updates=800,discarded_smoke_restore_updates=6,new_diffusion_calls=0,target_migration=False,
        all_inputs_historical_BUILD=True,query_not_independent_scene_test=True,human_verdict=None,problem_solved=None)
    h.write(out/'contract.json',contract)
    entry=next(v for v in h.read(h.PRIOR/'source_selection/manifest.json')['candidates'] if v['candidate_id']=='patch_02')
    source=h.load(entry['packet']);ids=source['world_row_ids'];core=source['core_mask'];ctx=h.context(frames);torch.set_num_threads(2)
    exposure_source=h.load(cal['source_packet'])
    assert set(source)==set(exposure_source)
    assert all(np.array_equal(v,exposure_source[k]) for k,v in source.items())
    h.write(out/'initialization_equivalence.json',dict(source_packet=entry['packet'],
        exposure_source_packet=cal['source_packet'],all_source_fields_identical=True,
        source_packet_realpath=str(Path(entry['packet']).resolve()),exposure_packet_realpath=str(Path(cal['source_packet']).resolve()),
        core_RGB_rows=int(core.sum()),both_arms_read_same_initial_RGB=True))
    h.write(ROOT/'gpu_lease.json',dict(active=True,owner=h.native.m.OWNER,gpu_authorized=True,optimizer_updates_allowed=806))
    base=ctx.base_rgb.detach().clone();positions=np.searchsorted(ctx.active_world_rows,ids)
    assert np.array_equal(ctx.active_world_rows[positions],ids)
    pos=torch.tensor(positions[core],device='cuda');allpos=torch.tensor(positions,device='cuda')
    keep=torch.tensor(np.isin(ctx.active_world_rows,ids),device='cuda');initial=base[pos].clone()
    assert np.array_equal(initial.cpu().numpy(),source['reference_srgb'][core])
    data={};F=B=0
    def render(raw,param):
        nonlocal F
        packet=ctx.base_packet(raw);packet['mask']=packet['mask']&keep;packet['colors_precomp']=base.index_copy(0,pos,param)
        ctx.require_lease();ctx.native_calls+=1;F+=1
        return ctx.scene._bindings().raster(ctx.camera(raw),ctx.scene.pipe,0,packet,ctx.scene._background)
    def optimizer(param):return torch.optim.Adam([param],lr=LR)
    def step(param,opt,schedule,index):
        nonlocal B
        opt.zero_grad(set_to_none=True);loss=param.sum()*0
        sampled=schedule[index].tolist()
        for raw in sampled:
            d=data[raw];r=render(raw,param);rgb=r['rgb']/d['alpha'].clamp_min(1e-8)
            loss=loss+(rgb[:,d['mask_t']]-d['target_t']).square().mean()/BATCH
        loss=loss+REG*(param-initial).square().mean()
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite loss')
        loss.backward();B+=BATCH
        if not bool(torch.isfinite(param.grad).all()):raise FloatingPointError('Nonfinite source RGB gradient')
        opt.step()
        with torch.no_grad():param.clamp_(0,1)
        return float(loss.detach()),sampled
    def checkpoint(path,param,opt,index,schedule):
        torch.save(dict(RGB=param.detach().cpu(),optimizer=opt.state_dict(),step=index,schedule=schedule,
            python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),contract=contract),path)
    def evaluate(folder,param):
        folder.mkdir(parents=True);rows=[]
        with torch.no_grad():
            for raw in frames:
                d=data[raw];r=render(raw,param);alpha=r['alpha'].squeeze().cpu().numpy()
                assert np.array_equal(alpha,d['alpha_cpu'])
                rgb=r['rgb'].cpu().numpy().transpose(1,2,0)/np.maximum(alpha[:,:,None],1e-8)
                raw_rgb=cam.warp(observation(rgb,d['gain'],gain_domain),d['inverse'])
                x0,y0,x1,y1=d['bbox'];crop=np.s_[y0:y1,x0:x1];f=folder/f'{raw:03d}';f.mkdir()
                for name,img in [('observation_raw',d['photo_raw']),('observation_pinhole_canonical',d['photo']),('render_raw',raw_rgb),('render_pinhole_canonical',rgb),('alpha',alpha)]:h.image(f/(name+'.png'),img[crop])
                Image.fromarray(np.uint8(d['mask'][crop])*255).save(f/'score_mask_pinhole.png')
                Image.fromarray(np.uint8(d['raw_mask'][crop])*255).save(f/'score_mask_raw.png')
                np.savez_compressed(f/'source_float.npz',observation_raw=d['photo_raw'][crop],observation_pinhole_canonical=d['photo'][crop],
                    render_raw=raw_rgb[crop],render_pinhole_canonical=rgb[crop],alpha=alpha[crop],score_mask_pinhole=d['mask'][crop],
                    score_mask_raw=d['raw_mask'][crop],bbox_xyxy=np.asarray(d['bbox']),gain=d['gain'])
                rows.append(dict(raw=raw,role='QUERY' if raw in query else 'LOW_RES_STRESS' if raw in stress else 'POOL_MEMBER',
                    raw_camera_metrics=metrics(d['photo_raw'],raw_rgb,d['raw_mask']),canonical_metrics=metrics(d['photo'],rgb,d['mask']),
                    gain=d['gain'].tolist(),output=str(f),alpha_exact=True,bbox_xyxy=[int(q) for q in d['bbox']]))
        h.write(folder/'receipt.json',dict(complete=True,rows=rows));return rows
    try:
        with torch.no_grad():
            for raw in frames:
                photo,project,path,K,T=p.projection(raw);grid,inverse,roundtrip=cam.brown_grid(raw)
                gain=np.asarray(gain_records[raw]['gain'],np.float32)
                if gain.ndim==0:gain=np.full(3,gain,np.float32)
                assert gain.shape==(3,) and (gain>0).all()
                legal=np.asarray(Image.open(records[raw]['core_photo_valid_mask_full1920_remote_stage']))>0
                uv,dep=project(entry['world_roi_corners']);canvas=Image.new('L',(1920,1280));ImageDraw.Draw(canvas).polygon([tuple(v) for v in uv],fill=255)
                physical=np.asarray(canvas)>0
                result=render(raw,initial);alpha=result['alpha'].squeeze().detach();alpha_cpu=alpha.cpu().numpy()
                mask=binary_erosion(physical,iterations=5)&(alpha_cpu>=.95)&(cam.warp(legal.astype(np.float32),grid,0)>.5)
                if mask.sum()<50:raise ValueError('Insufficient frozen score pixels: '+str(raw))
                raw_mask=(cam.warp(mask.astype(np.float32),inverse,0)>.5)&legal&(cam.warp(alpha_cpu,inverse)>=.95)
                if raw_mask.sum()<50:raise ValueError('Insufficient raw-camera score pixels: '+str(raw))
                target=canonical(cam.warp(photo,grid),gain,gain_domain)
                mask_t=torch.tensor(mask,device='cuda');target_t=torch.tensor(target.transpose(2,0,1),device='cuda')[:,mask_t]
                lo=np.maximum(0,np.floor(uv.min(0)-18).astype(int));hi=np.minimum([1920,1280],np.ceil(uv.max(0)+19).astype(int))
                data[raw]=dict(photo_raw=photo,photo=target,grid=grid,inverse=inverse,gain=gain,
                    alpha=alpha,alpha_cpu=alpha_cpu,mask=mask,mask_t=mask_t,target_t=target_t,raw_mask=raw_mask,
                    bbox=np.r_[lo,hi].tolist(),photo_path=path,camera_inverse_max_error=roundtrip)
        h.write(out/'data_receipt.json',dict(complete=True,frames=[dict(raw=k,score_n=int(v['mask'].sum()),raw_score_n=int(v['raw_mask'].sum()),
            gain=v['gain'].tolist(),photo=v['photo_path'],camera_inverse_max_error=v['camera_inverse_max_error']) for k,v in data.items()]))
        initial_eval=evaluate(out/'initial',initial)
        # 短程覆盖真实随机抽帧、Adam状态、完整RNG和下一更新恢复；所有更新丢弃。
        random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
        smoke=torch.nn.Parameter(initial.clone());so=optimizer(smoke);schedule=schedules['eight_frames']
        for i in range(4):step(smoke,so,schedule,i)
        checkpoint(out/'smoke_checkpoint.pt',smoke,so,4,schedule)
        restored=torch.load(out/'smoke_checkpoint.pt',map_location='cuda');resumed=torch.nn.Parameter(restored['RGB'].clone());ro=optimizer(resumed);ro.load_state_dict(restored['optimizer'])
        random.setstate(restored['python_rng']);np.random.set_state(restored['numpy_rng']);torch.set_rng_state(restored['torch_rng'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in restored['cuda_rng']])
        step(smoke,so,schedule,4);step(resumed,ro,restored['schedule'],4)
        restore_diff=float(abs(smoke-resumed).max().detach())
        if restore_diff>1e-6:raise ValueError('Next-update restore mismatch')
        del smoke,so,resumed,ro
        results=[]
        for label,pool in pools.items():
            random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
            folder=out/label;folder.mkdir();param=torch.nn.Parameter(initial.clone());opt=optimizer(param);history=[];started=time.time()
            checkpoint(folder/'initial_state.pt',param,opt,0,schedules[label])
            for index in range(STEPS):
                value,sampled=step(param,opt,schedules[label],index)
                if index==0 or (index+1)%25==0:
                    history.append(dict(step=index+1,pre_update_batch_loss=value,sampled_frames=sampled))
                    print(json.dumps(dict(arm=label,**history[-1])),flush=True)
            checkpoint(folder/'checkpoint.pt',param,opt,STEPS,schedules[label]);h.write(folder/'history.json',history)
            in_memory=evaluate(folder/'final_in_memory',param)
            repaired={k:v.copy() for k,v in source.items()};repaired['reference_srgb'][core]=param.detach().cpu().numpy()
            np.savez_compressed(folder/'source_patch_02.npz',**repaired);loaded=h.load(folder/'source_patch_02.npz')
            assert all(np.array_equal(v,loaded[k]) for k,v in source.items() if k!='reference_srgb')
            assert np.array_equal(loaded['reference_srgb'][~core],source['reference_srgb'][~core])
            reloaded=torch.tensor(loaded['reference_srgb'][core],device='cuda');final=evaluate(folder/'final_reloaded',reloaded)
            reload_max=0.
            for raw in frames:
                aa=h.load(folder/'final_in_memory'/f'{raw:03d}/source_float.npz');bb=h.load(folder/'final_reloaded'/f'{raw:03d}/source_float.npz')
                reload_max=max(reload_max,float(abs(aa['render_raw']-bb['render_raw']).max()))
                assert np.array_equal(aa['alpha'],bb['alpha'])
            if reload_max!=0:raise ValueError('Persistent source RGB reload differs')
            arm=dict(label=label,pool=pool,steps=STEPS,source_core_RGB_parameters=int(initial.numel()),
                final=final,in_memory=in_memory,reload_render_max_abs=reload_max,non_RGB_fields_exact=True,halo_RGB_exact=True,
                seconds=time.time()-started,checkpoint=str(folder/'checkpoint.pt'),source_packet=str(folder/'source_patch_02.npz'))
            results.append(arm);h.write(out/'progress.json',dict(arms=results,nativeF=F,nativeB=B))
            del param,opt
        h.write(out/'receipt.json',dict(complete=True,contract=contract,initial=initial_eval,arms=results,
            nativeF=F,nativeB=B,formal_updates=800,discarded_smoke_restore_updates=6,
            resume_next_update_RGB_max_abs=restore_diff,initial_RGB_identical=True,source_geometry_unchanged=True,
            target_migration=False,new_diffusion_calls=0,human_verdict=None,problem_solved=None))
    finally:h.native.close()

if __name__=='__main__':main()
