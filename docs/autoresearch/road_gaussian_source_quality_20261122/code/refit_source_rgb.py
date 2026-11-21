"""固定源几何/尺度的真实多帧RGB诊断；与目标接缝拟合分开记账。"""
from pathlib import Path
import json,copy,time,sys
import numpy as np
from scipy.ndimage import binary_erosion,gaussian_filter
from PIL import Image,ImageDraw
import render_source1920 as h

torch=h.torch;ROOT=h.ROOT;OUT=ROOT/'source_rgb_refit_qualified_r2'
TRAIN=(98,137);QUERY=(86,110,173);FRAMES=TRAIN+QUERY
STEPS=400;LR=.01;REG=1e-4

def project_source(raw,entry):
    d=h.read(h.BANK/f'observations/{raw:03d}.json')
    bank={int(v['raw']):v['id'] for v in h.read(h.BANK/'receipt.json')['frames']}
    cal={s.split(':')[0]:np.fromstring(s.split(':')[1],sep=' ') for s in (h.BANK/'calib'/(bank[raw]+'.txt')).read_text().splitlines()}
    K=cal['P0'].reshape(3,4)[:,:3];T=np.eye(4);T[:3]=cal['Tr_velo_to_cam_0'].reshape(3,4)
    T=T@np.linalg.inv(np.asarray(d['ego_world']).reshape(4,4))
    q=np.asarray(entry['world_roi_corners'])@T[:3,:3].T+T[:3,3];uv=q@K.T;uv=uv[:,:2]/uv[:,2:]-.5
    canvas=Image.new('L',(1920,1280));ImageDraw.Draw(canvas).polygon([tuple(v) for v in uv],fill=255)
    core=np.asarray(canvas)>0;lo=np.maximum(0,np.floor(uv.min(0)-12).astype(int));hi=np.minimum([1920,1280],np.ceil(uv.max(0)+13).astype(int))
    photo=np.asarray(Image.open(h.BANK/d['image_path']).convert('RGB'),np.float32)/255
    assert photo.shape==(1280,1920,3)
    return dict(core=core,lo=lo,hi=hi,photo=photo,photo_path=str(h.BANK/d['image_path']))

def main():
    if OUT.exists():raise FileExistsError('Preserve previous diagnostic: '+str(OUT))
    OUT.mkdir()
    contract=dict(task='SOURCE-RGB-REFIT-20261122',source='patch_00',train_frames=TRAIN,query_frames=QUERY,
        steps=STEPS,optimizer='Adam',RGB_lr=LR,source_prior_MSE_weight=REG,render_wh=[1920,1280],
        loss='Equal train-view mean MSE of effective isolated native RGB on fixed eroded source core, initial alpha>=.95 AND qualified original-photo road/dynamic/shadow mask, plus RGB fidelity prior',
        trainable='RGB of source core Gaussians only; halo RGB and all geometry/opacity/sigma frozen',
        source_selection_used_historical_BUILD=True,query_is_not_blind_benchmark=True,new_diffusion_calls=0,
        target_transfer_updates=0,source_full_3D_reconstruction=False,hyperparameter_search=False,human_verdict=None)
    h.write(OUT/'contract.json',contract)
    entry=next(v for v in h.read(h.PRIOR/'source_selection/manifest.json')['candidates'] if v['candidate_id']=='patch_00')
    source=h.load(entry['packet']);ctx=h.context(FRAMES);torch.set_num_threads(2)
    source_ids=source['world_row_ids'];core_ids=source_ids[source['core_mask']]
    core_pos=np.searchsorted(ctx.active_world_rows,core_ids);pos=torch.tensor(core_pos,device='cuda',dtype=torch.long)
    keep=torch.tensor(np.isin(ctx.active_world_rows,source_ids),device='cuda',dtype=torch.bool)
    base=ctx.base_rgb.detach().clone();initial=base[pos].clone();data={};forwards=0;backwards=0
    def render(raw,param):
        nonlocal forwards
        packet=ctx.base_packet(raw);packet['mask']=packet['mask']&keep
        packet['colors_precomp']=base.index_copy(0,pos,param)
        ctx.require_lease();ctx.native_calls+=1;forwards+=1
        return ctx.scene._bindings().raster(ctx.camera(raw),ctx.scene.pipe,0,packet,ctx.scene._background)
    def optimizer(param):return torch.optim.Adam([param],lr=LR)
    def step(param,opt):
        nonlocal backwards
        opt.zero_grad(set_to_none=True);loss=param.sum()*0
        for raw in TRAIN:
            out=render(raw,param);effective=out['rgb']/data[raw]['alpha'].clamp_min(1e-8)
            loss=loss+(effective[:,data[raw]['mask']]-data[raw]['target']).square().mean()/len(TRAIN)
        loss=loss+REG*(param-initial).square().mean()
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite source RGB objective')
        loss.backward();backwards+=len(TRAIN)
        if not bool(torch.isfinite(param.grad).all()):raise FloatingPointError('Nonfinite source RGB gradient')
        opt.step()
        with torch.no_grad():param.clamp_(0,1)
        return float(loss.detach())
    def evaluate(stage,param):
        rows=[]
        with torch.no_grad():
            for raw in FRAMES:
                d=data[raw];out=render(raw,param);alpha=out['alpha'].squeeze().cpu().numpy()
                assert np.array_equal(alpha,d['alpha_cpu']),'Source alpha must stay exact'
                rgb=out['rgb'].cpu().numpy().transpose(1,2,0)/np.maximum(alpha[:,:,None],1e-8)
                x0,y0=d['lo'];x1,y1=d['hi'];crop=np.s_[y0:y1,x0:x1];folder=OUT/stage/f'{raw:03d}';folder.mkdir(parents=True)
                for key,a in [('observation',d['photo']),('isolated',rgb)]:h.image(folder/(key+'.png'),a[crop])
                Image.fromarray(np.uint8(d['mask_cpu'][crop])*255).save(folder/'score_mask.png')
                Image.fromarray(np.uint8(d['core'][crop])*255).save(folder/'core_mask.png')
                np.savez_compressed(folder/'core_float.npz',observation=d['photo'][crop],isolated_rgb=rgb[crop],alpha=alpha[crop],score_mask=d['mask_cpu'][crop],core=d['core'][crop])
                scales=[]
                for sigma in [1.2,5/3,3.]:
                    a=(d['photo']-gaussian_filter(d['photo'],(sigma,sigma,0)))[d['mask_cpu']].ravel()
                    b=(rgb-gaussian_filter(rgb,(sigma,sigma,0)))[d['mask_cpu']].ravel()
                    an=np.linalg.norm(a);bn=np.linalg.norm(b)
                    scales.append(dict(sigma_native_px=sigma,HF_cosine=float(a@b/(an*bn)) if an*bn>1e-20 else None,HF_RMS_ratio=float(bn/an) if an>1e-20 else None))
                rows.append(dict(raw=raw,role='TRAIN' if raw in TRAIN else 'QUERY_not_fit_this_run',score_n=int(d['mask_cpu'].sum()),
                    RGB_MAE=float(abs(rgb-d['photo'])[d['mask_cpu']].mean()),RGB_MSE=float(((rgb-d['photo'])**2)[d['mask_cpu']].mean()),
                    spectral=scales,alpha_exact=True,bbox_xyxy=np.r_[d['lo'],d['hi']].tolist(),output=str(folder)))
        return rows
    photo_plan=h.read(ROOT/'source_view_ranking/source00_probe_plan.json')
    photo_legal={int(v['raw']):np.asarray(Image.open(v['core_photo_valid_mask']))>0 for v in photo_plan['records']}
    mask_audit=[]
    try:
        with torch.no_grad():
            for raw in FRAMES:
                d=project_source(raw,entry);out=render(raw,initial);alpha=out['alpha'].squeeze().detach()
                prior_mask=binary_erosion(d['core'],iterations=5)&(alpha.cpu().numpy()>=.95)
                mask=prior_mask&photo_legal[raw]
                mask_audit.append(dict(raw=raw,pre_photo_mask_pixels=int(prior_mask.sum()),qualified_pixels=int(mask.sum()),removed=int((prior_mask&~photo_legal[raw]).sum())))
                if mask.sum()<100:raise ValueError('Insufficient source-core pixels: '+str(raw))
                mt=torch.tensor(mask,device='cuda',dtype=torch.bool);target=torch.tensor(d['photo'].transpose(2,0,1),device='cuda')[:,mt]
                d.update(alpha=alpha,alpha_cpu=alpha.cpu().numpy(),mask=mt,mask_cpu=mask,target=target);data[raw]=d
        h.write(OUT/'photo_mask_audit.json',dict(passed=True,rows=mask_audit))
        baseline=evaluate('original',initial)
        # 短程覆盖前反向、Adam非空状态与下一步精确恢复；所有短程更新丢弃。
        smoke=torch.nn.Parameter(initial.clone());so=optimizer(smoke)
        smoke_losses=[step(smoke,so) for _ in range(4)]
        torch.save(dict(RGB=smoke.detach().cpu(),optimizer=so.state_dict(),step=4),OUT/'smoke_checkpoint.pt')
        state=torch.load(OUT/'smoke_checkpoint.pt',map_location='cuda');resumed=torch.nn.Parameter(state['RGB'].clone());ro=optimizer(resumed);ro.load_state_dict(state['optimizer'])
        step(smoke,so);step(resumed,ro);resume_diff=float(abs(smoke-resumed).max().detach())
        if resume_diff>1e-7:raise ValueError('Source state restore mismatch')
        del smoke,so,resumed,ro
        param=torch.nn.Parameter(initial.clone());opt=optimizer(param);history=[];started=time.time()
        torch.save(dict(RGB=param.detach().cpu(),optimizer=opt.state_dict(),step=0,contract=contract),OUT/'initial_state.pt')
        for index in range(1,STEPS+1):
            value=step(param,opt)
            if index==1 or index%50==0:
                history.append(dict(step=index,pre_update_loss=value))
                print(json.dumps(history[-1]),flush=True)
        torch.save(dict(RGB=param.detach().cpu(),optimizer=opt.state_dict(),step=STEPS,contract=contract),OUT/'checkpoint.pt')
        final=evaluate('refit',param)
        repaired={k:v.copy() for k,v in source.items()};repaired['reference_srgb'][source['core_mask']]=param.detach().cpu().numpy()
        np.savez_compressed(OUT/'source_patch_00_refit.npz',**repaired)
        assert np.array_equal(source['world_xyz'],repaired['world_xyz']) and np.array_equal(source['gaussian_sigma_m'],repaired['gaussian_sigma_m'])
        h.write(OUT/'history.json',history)
        h.write(OUT/'receipt.json',dict(complete=True,contract=contract,original=baseline,refit=final,
            nativeF=forwards,nativeB=backwards,formal_optimizer_updates=400,discarded_smoke_and_restore_updates=6,
            formal_seconds=time.time()-started,resume_next_step_max_abs=resume_diff,source_core_color_parameters=int(initial.numel()),
            geometry_scales_opacity_unchanged=True,original_asset_modified=False,source_asset_rebuilt=False,
            target_migration_executed=False,new_diffusion_calls=0,human_verdict=None,production_promoted=False))
        print('COMPLETE',flush=True)
    finally:h.native.close()

if __name__=='__main__':main()
