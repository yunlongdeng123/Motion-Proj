"""实测固定高斯颜色算子，再有界求解RGB；不用优化步数掩盖写入链路。"""
from pathlib import Path
import argparse,json,time
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_erosion,map_coordinates
from scipy.sparse import csr_matrix,vstack,eye,save_npz
from scipy.optimize import lsq_linear
import probe_source_chain as p
h=p.h;torch=p.torch;ROOT=p.ROOT

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reference',type=int,default=110)
    parser.add_argument('--patches',nargs='+',default=['patch_00','patch_02'])
    parser.add_argument('--plan',default=str(ROOT/'photo_audit/probe_plan.json'))
    parser.add_argument('--output',default='fixed_color_operator_110')
    args=parser.parse_args();out=ROOT/args.output
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    plan=h.read(args.plan)
    entries={e['candidate_id']:e for e in h.read(h.PRIOR/'source_selection/manifest.json')['candidates']}
    records={(v['candidate_id'],int(v['raw'])):v for v in plan['records']}
    frames=sorted({raw for patch,raw in records if patch in args.patches})
    prior=1e-5
    h.write(out/'contract.json',dict(task='ROAD-SOURCE-FIDELITY-20261122',run=ROOT.name,
        stage=args.output,patches=args.patches,reference=args.reference,
        method='Measure fixed native RGB response by three one-hot rows per render; bounded linear least squares in encoded RGB [0,1]',
        objective='mean pixel RGB square + 1e-5 * mean RGB deviation from direct-photo center samples',
        prior_weight=prior,solver='scipy.optimize.lsq_linear(trf,lsmr), tol=1e-7, lsmr_tol=1e-8, max_iter=100, lsmr_maxiter=2000',
        geometry_scale_opacity_frozen=True,halo_RGB_frozen=True,new_diffusion_calls=0,
        target_migration=False,reference_image_is_BUILD=True,query_is_not_independent_scene_test=True,
        fit_scope='One reference view per source, other listed views never enter this solve',
        optimizer_gradient_updates=0,human_verdict=None,production_promoted=False))
    ctx=h.context(frames);torch.set_num_threads(2)
    base=ctx.base_rgb.detach().clone();counts=0;rows=[];started=time.time()
    def raster(raw,ids,colors):
        nonlocal counts
        packet=ctx.base_packet(raw)
        packet['mask']=packet['mask']&torch.tensor(np.isin(ctx.active_world_rows,ids),device='cuda')
        packet['colors_precomp']=colors
        ctx.require_lease();ctx.native_calls+=1;counts+=1
        r=ctx.scene._bindings().raster(ctx.camera(raw),ctx.scene.pipe,0,packet,ctx.scene._background)
        alpha=r['alpha'].squeeze().cpu().numpy()
        return r['rgb'].cpu().numpy().transpose(1,2,0)/np.maximum(alpha[:,:,None],1e-8),alpha
    try:
      with torch.no_grad():
        for patch in args.patches:
            entry=entries[patch];source=h.load(entry['packet']);ids=source['world_row_ids'];core=source['core_mask']
            positions=np.searchsorted(ctx.active_world_rows,ids)
            assert np.array_equal(ids,ctx.active_world_rows[positions])
            all_pos=torch.tensor(positions,device='cuda');core_pos=torch.tensor(positions[core],device='cuda')
            n=int(core.sum());data={}
            for raw in frames:
                if (patch,raw) not in records:continue
                photo,project,photo_path,K,T=p.projection(raw)
                uv,dep=project(entry['world_roi_corners'])
                if (dep<=0).any():continue
                canvas=Image.new('L',(1920,1280));ImageDraw.Draw(canvas).polygon([tuple(v) for v in uv],fill=255)
                physical=np.asarray(canvas)>0
                original,alpha=raster(raw,ids,base)
                legal=np.asarray(Image.open(records[(patch,raw)]['core_photo_valid_mask_full1920_remote_stage']))>0
                score=binary_erosion(physical,iterations=5)&(alpha>=.95)&legal
                if score.sum()<50:continue
                lo=np.maximum(0,np.floor(uv.min(0)-12).astype(int));hi=np.minimum([1920,1280],np.ceil(uv.max(0)+13).astype(int))
                data[raw]=dict(photo=photo,project=project,photo_path=photo_path,physical=physical,
                    original=original,alpha=alpha,score=score,lo=lo,hi=hi)
            d=data[args.reference];uv,depth=d['project'](source['world_xyz'])
            sampled=np.stack([map_coordinates(d['photo'][:,:,c],[uv[:,1],uv[:,0]],order=1,mode='nearest') for c in range(3)],axis=1)
            x0=sampled[core].copy();m=int(d['score'].sum())
            zero=base.clone();zero[all_pos]=0
            A=np.empty((m,n),dtype=np.float32)
            for i in range(0,n,3):
                colors=zero.clone()
                for ch,j in enumerate(range(i,min(i+3,n))):colors[core_pos[j],ch]=1
                response,alpha=raster(args.reference,ids,colors)
                assert np.array_equal(alpha,d['alpha'])
                for ch,j in enumerate(range(i,min(i+3,n))):A[:,j]=response[:,:,ch][d['score']]
                if i%300==0:print(json.dumps(dict(patch=patch,measured_columns=i,total=n)),flush=True)
            halo=base.clone();halo[core_pos]=0
            fixed,alpha=raster(args.reference,ids,halo)
            direct_colors=base.index_copy(0,core_pos,torch.tensor(x0,device='cuda'))
            direct_native,alpha=raster(args.reference,ids,direct_colors)
            predicted=A@x0+fixed[d['score']]
            linear_error=float(np.abs(predicted-direct_native[d['score']]).max())
            if linear_error>3e-6:raise ValueError('Fixed RGB operator mismatch: '+str(linear_error))
            folder=out/patch;folder.mkdir()
            save_npz(folder/'native_color_operator.npz',csr_matrix(A))
            weight=prior*m/n
            AA=vstack([csr_matrix(A),np.sqrt(weight)*eye(n)],format='csr')
            target=d['photo'][d['score']]-fixed[d['score']]
            solved=[];states=[]
            for ch in range(3):
                result=lsq_linear(AA,np.r_[target[:,ch],np.sqrt(weight)*x0[:,ch]],bounds=(0,1),
                    method='trf',lsq_solver='lsmr',tol=1e-7,lsmr_tol=1e-8,max_iter=100,lsmr_maxiter=2000)
                solved.append(result.x)
                states.append(dict(channel=ch,status=int(result.status),success=bool(result.success),
                    iterations=int(result.nit),cost=float(result.cost),optimality=float(result.optimality),message=result.message))
                print(json.dumps(dict(patch=patch,solver=states[-1])),flush=True)
            fitted=np.stack(solved,axis=1).astype(np.float32)
            np.savez_compressed(folder/'linear_state.npz',fitted_core_RGB=fitted,direct_core_RGB=x0,
                target_minus_fixed=target,fixed_core_image=fixed[d['score']],world_row_ids=ids,
                core_mask=core,score_mask=d['score'])
            repaired={k:v.copy() for k,v in source.items()};repaired['reference_srgb'][core]=fitted
            np.savez_compressed(folder/'source_fixed_RGB.npz',**repaired)
            loaded=h.load(folder/'source_fixed_RGB.npz')
            assert all(np.array_equal(v,loaded[k]) for k,v in source.items() if k!='reference_srgb')
            assert np.array_equal(source['reference_srgb'][~core],loaded['reference_srgb'][~core])
            runtime=base.index_copy(0,all_pos,torch.tensor(loaded['reference_srgb'],device='cuda'))
            evaluations=[]
            for raw,v in data.items():
                direct,alpha=raster(raw,ids,direct_colors)
                fit,fit_alpha=raster(raw,ids,runtime)
                assert np.array_equal(alpha,v['alpha']) and np.array_equal(fit_alpha,v['alpha'])
                xlo,ylo=v['lo'];xhi,yhi=v['hi'];crop=np.s_[ylo:yhi,xlo:xhi]
                framefolder=folder/f'{raw:03d}';framefolder.mkdir()
                for name,a in [('observation',v['photo']),('original',v['original']),('direct_reference',direct),('fixed_color_solve',fit)]:h.image(framefolder/(name+'.png'),a[crop])
                Image.fromarray(np.uint8(v['score'][crop])*255).save(framefolder/'score_mask.png')
                Image.fromarray(np.uint8(v['physical'][crop])*255).save(framefolder/'core_mask.png')
                h.image(framefolder/'alpha.png',alpha[crop])
                np.savez_compressed(framefolder/'source_float.npz',observation=v['photo'][crop],
                    original_rgb=v['original'][crop],direct_rgb=direct[crop],solved_rgb=fit[crop],
                    alpha=alpha[crop],score_mask=v['score'][crop],core=v['physical'][crop],bbox_xyxy=np.r_[v['lo'],v['hi']])
                metrics={}
                for name,a in [('original',v['original']),('direct_reference',direct),('fixed_color_solve',fit)]:
                    metrics[name]=dict(RGB_MAE=float(np.abs(v['photo']-a)[v['score']].mean()),
                        signed_RGB_bias=(a-v['photo'])[v['score']].mean(0).tolist(),spectral=p.spectral(v['photo'],a,v['score']))
                evaluations.append(dict(raw=raw,role='REFERENCE_FIT' if raw==args.reference else 'QUERY_not_fit',
                    score_n=int(v['score'].sum()),metrics=metrics,output=str(framefolder),bbox_xyxy=np.r_[v['lo'],v['hi']].tolist()))
            rows.append(dict(patch=patch,reference=args.reference,core_n=n,score_n=m,
                source_packet=str(folder/'source_fixed_RGB.npz'),operator_linear_check_max_abs=linear_error,
                source_RGB_noncolor_exact=True,halo_RGB_exact=True,alpha_exact=True,
                solver=states,evaluations=evaluations))
            h.write(out/'progress.json',dict(rows=rows,nativeF=counts,nativeB=0))
        h.write(out/'receipt.json',dict(complete=True,rows=rows,nativeF=counts,nativeB=0,
            RGB_linear_solves=len(rows)*3,gradient_optimizer_updates=0,new_diffusion_calls=0,
            original_asset_modified=False,target_migration=False,seconds=time.time()-started))
    finally:h.native.close()

if __name__=='__main__':main()
